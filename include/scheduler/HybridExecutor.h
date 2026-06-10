/**
 * @file HybridExecutor.h
 * @brief 异构并行执行器 —— 核心调度引擎，将算子图编译为分阶段执行方案并并行执行
 *
 * ====== 学习要点 ======
 *
 * 1. 异构计算（Heterogeneous Computing）
 *    现代系统有 CPU + GPU + ... 多种计算设备，它们各有所长：
 *    - CPU：擅长控制流、小批量、逐元素操作
 *    - GPU：擅长大规模并行、矩阵运算、卷积
 *    "异构执行" = 把不同类型的任务分给最合适的设备。
 *    这比"全放 CPU"或"全放 GPU"都要快——各取所长。
 *
 * 2. 分阶段执行（Staged Execution）
 *    算子图可能有并行分支，但分支汇合处需要同步。
 *    我们将算子按依赖关系分成若干"阶段"（stage）：
 *    - 同一阶段内的算子互不依赖，可以并行
 *    - 阶段之间有严格的串行依赖
 *    例如：
 *      Stage 0: [conv1]
 *      Stage 1: [relu1]        ← 依赖 conv1
 *      Stage 2: [conv2a, conv2b] ← 依赖 relu1，彼此独立可并行
 *      Stage 3: [add1]         ← 依赖 conv2a 和 conv2b
 *
 * 3. 执行计划编译（Compile Phase）
 *    "编译"就是提前把"谁在哪个设备上跑"全部决定好。
 *    好处：
 *    - 运行时不需要重复决策，减少开销
 *    - 可以全局优化（如合并连续的 GPU 算子减少内存拷贝）
 *    - 方便可视化调试
 *    类比：解释执行 vs JIT 编译——前者灵活但慢，后者有编译开销但运行快。
 *
 * 4. std::future 与异步执行
 *    std::async() 启动一个异步任务，返回 std::future<T>。
 *    调用 future.get() 会阻塞直到任务完成。
 *    在 execute() 中：
 *    - 同一阶段的 CPU 和 GPU 算子通过 std::async 并行启动
 *    - 阶段之间通过 future.get() 等待同步
 *    这样实现了"阶段内并行、阶段间串行"的执行模式。
 *
 * 5. opRunner 回调设计
 *    HybridExecutor 不直接执行算子——它不知道怎么执行。
 *    调用者提供一个回调函数：void opRunner(int opId, Device device)
 *    这解耦了"调度逻辑"和"执行逻辑"：
 *    - 执行器负责"谁在什么时候跑"
 *    - 回调负责"怎么跑"（调用 ncnn、调用 Vulkan compute shader 等）
 *    这使得执行器可以用于任何推理引擎。
 */

#pragma once

#include "scheduler/DispatchStrategy.h"
#include "scheduler/OpGraph.h"
#include "profiler/OpProfiler.h"
#include <functional>
#include <future>
#include <memory>
#include <vector>

namespace edgeai {

// ============================================================================
// 执行计划
// ============================================================================

/**
 * @struct ExecutionPlan
 * @brief 编译好的执行方案，描述了每个算子的调度决策和执行阶段
 *
 * 示例：一个简单的 Conv → ReLU → Conv → Pooling 图
 * @code
 *   decisions = [
 *     {opId:0, device:GPU, reason:"large conv"},
 *     {opId:1, device:CPU, reason:"element-wise"},
 *     {opId:2, device:GPU, reason:"large conv"},
 *     {opId:3, device:CPU, reason:"small pooling"}
 *   ]
 *   stages = [
 *     [0],        ← Stage 0: 只有 conv1
 *     [1],        ← Stage 1: relu1 依赖 conv1
 *     [2],        ← Stage 2: conv2 依赖 relu1
 *     [3]         ← Stage 3: pool 依赖 conv2
 *   ]
 * @endcode
 *
 * 更复杂的例子（有并行分支）：
 * @code
 *   stages = [
 *     [0],           ← conv1
 *     [1, 2],        ← relu1 和 branch_op 可并行
 *     [3]            ← add1 依赖 1 和 2
 *   ]
 * @endcode
 */
struct ExecutionPlan {
    /// 每个算子的调度决策（按 opId 索引）
    std::vector<DispatchDecision> decisions;

    /// 执行阶段列表，stages[i] 是第 i 阶段可并行执行的算子 id 集合
    /// 阶段之间串行，阶段内部并行
    std::vector<std::vector<int>> stages;

    /// 打印执行计划到 stdout
    void print() const;
};

// ============================================================================
// 异构并行执行器
// ============================================================================

/**
 * @class HybridExecutor
 * @brief 异构并行执行器 —— 将算子图编译为执行计划并按阶段并行执行
 *
 * 核心工作流：
 * @code
 *   // 1. 创建执行器，设置调度策略
 *   HybridExecutor executor;
 *   executor.setStrategy(std::make_shared<HeuristicDispatch>());
 *
 *   // 2. 编译执行计划
 *   ExecutionPlan plan = executor.compile(graph);
 *   plan.print();  // 查看调度结果
 *
 *   // 3. 执行（不带 profiling）
 *   executor.execute(plan, [](int opId, Device dev) {
 *       // 调用实际的推理引擎执行算子
 *       runOp(opId, dev);
 *   });
 *
 *   // 4. 执行（带 profiling）
 *   OpProfiler profiler;
 *   executor.executeWithProfile(plan, runner, profiler);
 *   profiler.printReport();
 * @endcode
 */
class HybridExecutor {
public:
    HybridExecutor();
    ~HybridExecutor();

    // 禁止拷贝（内部有 std::future，不可拷贝）
    HybridExecutor(const HybridExecutor&) = delete;
    HybridExecutor& operator=(const HybridExecutor&) = delete;

    // 允许移动
    HybridExecutor(HybridExecutor&&) noexcept;
    HybridExecutor& operator=(HybridExecutor&&) noexcept;

    // ──────────────── 策略配置 ────────────────

    /**
     * @brief 设置调度策略
     * @param strategy 调度策略的共享指针
     *
     * 必须在 compile() 之前调用。如果不设置，默认使用 HeuristicDispatch。
     */
    void setStrategy(std::shared_ptr<DispatchStrategy> strategy);

    // ──────────────── 编译 ────────────────

    /**
     * @brief 编译执行计划：根据算子图和调度策略生成分阶段执行方案
     * @param graph 算子依赖图
     * @return 编译好的 ExecutionPlan
     *
     * 编译过程：
     * 1. 对图进行拓扑排序，得到合法执行顺序
     * 2. 对每个算子调用 strategy_->dispatch() 获取调度决策
     * 3. 根据依赖关系将算子划分为执行阶段（同一阶段的算子无依赖）
     * 4. 组装 ExecutionPlan
     *
     * 划分阶段的算法（基于拓扑排序的层次分解）：
     * - 入度为 0 的节点归入 Stage 0
     * - 删除 Stage 0 的节点后，新的入度为 0 的节点归入 Stage 1
     * - 以此类推，直到所有节点归入某个阶段
     */
    ExecutionPlan compile(const OpGraph& graph);

    // ──────────────── 执行 ────────────────

    /**
     * @brief 按阶段执行算子图，同阶段算子并行到不同设备
     * @param plan    编译好的执行计划
     * @param opRunner 算子执行回调：opRunner(opId, device)
     *
     * 执行逻辑：
     * for each stage in plan.stages:
     *     对同阶段内所有算子，按设备并行启动：
     *       - CPU 算子 → std::async 在 CPU 线程池执行
     *       - GPU 算子 → std::async 提交到 GPU 队列
     *     等待所有 future 完成（阶段同步点）
     *
     * 注意：真正的 GPU 并行执行取决于 GPU 驱动的命令队列机制。
     * 这里 std::async 只是让 CPU 不阻塞在 GPU 提交上。
     */
    void execute(const ExecutionPlan& plan,
                 std::function<void(int, Device)> opRunner);

    /**
     * @brief 带 Profiling 的执行 —— 记录每个算子的执行时间
     * @param plan     编译好的执行计划
     * @param opRunner 算子执行回调
     * @param profiler 用于记录性能数据的 OpProfiler 引用
     *
     * 对每个算子，在执行前后调用 profiler.beginOp() / endOp()。
     * 执行完成后，profiler 中包含每个算子的完整性能记录。
     */
    void executeWithProfile(const ExecutionPlan& plan,
                            std::function<void(int, Device)> opRunner,
                            OpProfiler& profiler);

private:
    std::shared_ptr<DispatchStrategy> strategy_;     ///< 调度策略
    std::vector<std::future<void>> pendingFutures_;  ///< 当前阶段的异步任务

    /// 等待当前阶段的所有异步任务完成
    void waitForStage();
};

} // namespace edgeai
