/**
 * @file HybridExecutor.cpp
 * @brief 异构并行执行器的实现 —— 编译执行计划 + 分阶段并行执行
 *
 * ====== 实现说明 ======
 *
 * 1. compile() 的阶段划分算法
 *    基于 Kahn 拓扑排序的"层次分解"变体：
 *    - 所有入度为 0 的节点归入 Stage 0
 *    - "删除"Stage 0 的所有节点（将其后继入度减 1）
 *    - 新的入度为 0 的节点归入 Stage 1
 *    - 以此类推
 *    这种方法保证：同阶段节点无依赖（可并行），阶段间串行。
 *
 * 2. execute() 的并行模型
 *    ┌─────────────────────────────────────────────────────┐
 *    │  Stage 0: [conv1(GPU), data_norm(CPU)]               │ ← 并行启动
 *    │    ↓ 等待所有 future 完成（同步点）                      │
 *    │  Stage 1: [relu1(CPU), branch_op(CPU)]               │ ← 并行启动
 *    │    ↓ 等待所有 future 完成                               │
 *    │  Stage 2: [conv2(GPU)]                                 │
 *    │    ↓                                                    │
 *    │  Stage 3: [pool(CPU)]                                  │
 *    └─────────────────────────────────────────────────────┘
 *
 *    同阶段内，CPU 算子和 GPU 算子通过 std::async 并行启动。
 *    但要注意：真正的 CPU↔GPU 并行性取决于硬件（CPU 核数、GPU 命令队列）。
 *    如果只有一个 CPU 线程，多个 CPU 算子实际上是伪并行。
 *
 * 3. std::async 的使用注意
 *    - std::async(std::launch::async, ...) 强制创建新线程
 *    - 返回的 std::future 必须被 get() 或 wait()，否则析构时可能阻塞
 *    - 不能在 future 析构前启动依赖它的下一阶段任务
 */

#include "scheduler/HybridExecutor.h"
#include "scheduler/HeuristicDispatch.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

namespace edgeai {

// ============================================================================
// ExecutionPlan 实现
// ============================================================================

void ExecutionPlan::print() const {
    std::cout << "========== 执行计划 ==========\n\n";

    // ── 打印调度决策 ──
    std::cout << "调度决策:\n";
    for (const auto& d : decisions) {
        std::cout << "  [Op#" << d.opId << "] -> "
                  << deviceToString(d.device)
                  << " (" << d.reason << ")\n";
    }
    std::cout << "\n";

    // ── 打印执行阶段 ──
    std::cout << "执行阶段:\n";
    for (size_t i = 0; i < stages.size(); ++i) {
        std::cout << "  Stage " << i << ": [";
        for (size_t j = 0; j < stages[i].size(); ++j) {
            if (j > 0) std::cout << ", ";
            // 找到对应的决策，显示设备
            int opId = stages[i][j];
            const char* dev = "CPU";
            if (opId >= 0 && opId < static_cast<int>(decisions.size())) {
                dev = deviceToString(decisions[opId].device);
            }
            std::cout << "Op#" << opId << "(" << dev << ")";
        }
        std::cout << "]\n";
    }
    std::cout << "\n==============================\n";
}

// ============================================================================
// HybridExecutor 构造/析构
// ============================================================================

HybridExecutor::HybridExecutor()
    : strategy_(std::make_shared<HeuristicDispatch>())  // 默认使用启发式策略
{
}

HybridExecutor::~HybridExecutor() {
    waitForStage();  // 确保所有异步任务完成
}

HybridExecutor::HybridExecutor(HybridExecutor&& other) noexcept
    : strategy_(std::move(other.strategy_))
    , pendingFutures_(std::move(other.pendingFutures_))
{
}

HybridExecutor& HybridExecutor::operator=(HybridExecutor&& other) noexcept {
    if (this != &other) {
        waitForStage();
        strategy_ = std::move(other.strategy_);
        pendingFutures_ = std::move(other.pendingFutures_);
    }
    return *this;
}

// ============================================================================
// 策略配置
// ============================================================================

void HybridExecutor::setStrategy(std::shared_ptr<DispatchStrategy> strategy) {
    if (strategy) {
        strategy_ = std::move(strategy);
    }
}

// ============================================================================
// 编译执行计划
// ============================================================================

ExecutionPlan HybridExecutor::compile(const OpGraph& graph) {
    ExecutionPlan plan;
    int n = graph.numOps();

    if (n == 0) {
        return plan;  // 空图
    }

    // ── 步骤 1：对每个算子进行调度决策 ──
    plan.decisions.resize(n);
    for (int i = 0; i < n; ++i) {
        const OpNode& op = graph.getOp(i);
        plan.decisions[i] = strategy_->dispatch(op, graph);
    }

    // ── 步骤 2：层次分解——将算子划分为执行阶段 ──
    // 使用类似 Kahn 算法的方式，但按"层"而非单个节点出队
    //
    // 核心思想：
    // - 入度为 0 的节点组成当前层（可并行执行）
    // - 当前层全部完成后，后继的入度减到 0 的节点组成下一层
    // - 每一层就是一个 stage

    // 计算初始入度
    std::vector<int> inDegree(n, 0);
    for (int i = 0; i < n; ++i) {
        inDegree[i] = static_cast<int>(graph.getOp(i).inputs.size());
    }

    // 找到所有初始零入度节点（Stage 0）
    std::vector<int> currentLayer;
    for (int i = 0; i < n; ++i) {
        if (inDegree[i] == 0) {
            currentLayer.push_back(i);
        }
    }

    int processedCount = 0;
    while (!currentLayer.empty()) {
        // 当前层作为一个 stage
        plan.stages.push_back(currentLayer);
        processedCount += static_cast<int>(currentLayer.size());

        // 将当前层所有节点的后继入度减 1
        // 收集下一层零入度节点
        std::vector<int> nextLayer;
        for (int u : currentLayer) {
            const OpNode& op = graph.getOp(u);
            for (int v : op.outputs) {
                --inDegree[v];
                if (inDegree[v] == 0) {
                    nextLayer.push_back(v);
                }
            }
        }

        currentLayer = std::move(nextLayer);
    }

    // 检查是否有环（processedCount < n 说明有环）
    if (processedCount < n) {
        std::cerr << "[HybridExecutor::compile] 警告: 图中存在环！"
                  << " 已处理 " << processedCount << "/" << n << " 个节点\n";
    }

    return plan;
}

// ============================================================================
// 等待当前阶段完成
// ============================================================================

void HybridExecutor::waitForStage() {
    for (auto& f : pendingFutures_) {
        if (f.valid()) {
            f.get();  // 阻塞等待任务完成
        }
    }
    pendingFutures_.clear();
}

// ============================================================================
// 执行（不带 profiling）
// ============================================================================

void HybridExecutor::execute(const ExecutionPlan& plan,
                               std::function<void(int, Device)> opRunner) {
    if (!opRunner) {
        std::cerr << "[HybridExecutor::execute] opRunner 为空，无法执行\n";
        return;
    }

    // ── 按阶段串行执行，阶段内并行 ──
    for (size_t stageIdx = 0; stageIdx < plan.stages.size(); ++stageIdx) {
        const auto& stage = plan.stages[stageIdx];

        // 对同阶段内的所有算子，根据调度决策并行启动
        for (int opId : stage) {
            // 查找该算子的调度决策
            Device device = Device::CPU;  // 默认 CPU
            if (opId >= 0 && opId < static_cast<int>(plan.decisions.size())) {
                device = plan.decisions[opId].device;
            }

            // 异步执行算子
            // std::launch::async 强制创建新线程（而非延迟执行）
            // 这确保 CPU 和 GPU 算子能真正并行
            pendingFutures_.push_back(
                std::async(std::launch::async, [opId, device, &opRunner]() {
                    opRunner(opId, device);
                })
            );
        }

        // ── 阶段同步点：等待当前阶段所有算子完成 ──
        // 这是异构执行的关键——必须保证数据依赖满足后才能执行下一阶段
        // 如果不等，后续算子可能读到未完成的前驱输出
        waitForStage();
    }
}

// ============================================================================
// 带 Profiling 的执行
// ============================================================================

void HybridExecutor::executeWithProfile(const ExecutionPlan& plan,
                                          std::function<void(int, Device)> opRunner,
                                          OpProfiler& profiler) {
    if (!opRunner) {
        std::cerr << "[HybridExecutor::executeWithProfile] opRunner 为空，无法执行\n";
        return;
    }

    // 与 execute() 类似，但在每个算子执行前后插入 profiling 钩子
    // 注意：profiling 会引入额外开销（计时器、锁），不适合生产环境
    // 但在调优阶段非常有价值

    // 用于保护 profiler 的互斥锁
    // 因为多个异步任务可能同时访问 profiler
    std::mutex profilerMutex;

    for (size_t stageIdx = 0; stageIdx < plan.stages.size(); ++stageIdx) {
        const auto& stage = plan.stages[stageIdx];

        for (int opId : stage) {
            Device device = Device::CPU;
            std::string opName = "op_" + std::to_string(opId);
            std::string opType = "unknown";

            if (opId >= 0 && opId < static_cast<int>(plan.decisions.size())) {
                device = plan.decisions[opId].device;
            }

            // 异步执行，包含 profiling 包装
            pendingFutures_.push_back(
                std::async(std::launch::async,
                    [&opRunner, &profiler, &profilerMutex, opId, device]() {
                        // 在执行前记录开始时间
                        // 注意：beginOp 必须在执行线程中调用，
                        // 否则计时器会包含任务调度延迟

                        // 由于 OpProfiler::beginOp/endOp 不是线程安全的，
                        // 我们需要使用互斥锁保护
                        // 但更好的做法是每个线程有自己的 profiler，
                        // 最后合并结果。这里为了简化使用互斥锁。

                        std::string name = "op_" + std::to_string(opId);
                        std::string type = "unknown";
                        std::string devStr = deviceToString(device);

                        {
                            std::lock_guard<std::mutex> lock(profilerMutex);
                            profiler.beginOp(name, type, devStr);
                        }

                        // 执行实际算子
                        opRunner(opId, device);

                        // 结束计时
                        {
                            std::lock_guard<std::mutex> lock(profilerMutex);
                            profiler.endOp(0, 0);  // FLOPs 和 memBytes 由调用者补充
                        }
                    }
                )
            );
        }

        // 等待当前阶段完成
        waitForStage();
    }
}

} // namespace edgeai
