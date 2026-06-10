/**
 * @file step5_hybrid_execution.cpp
 * @brief Lab2 Step5: CPU+GPU 混合并行执行 —— 异构调度的最终形态
 *
 * ====== 本实验学习目标 ======
 * 1. 理解分阶段执行（Staged Execution）的原理
 * 2. 学会将算子图编译为执行计划
 * 3. 模拟 CPU+GPU 并行执行，观察时间线加速效果
 * 4. 对比三种模式：纯CPU、纯GPU、混合并行的性能差异
 *
 * ====== 核心概念 ======
 *
 * 为什么混合并行能赢？
 * ┌─────────────────────────────────────────────────────────────────┐
 * │ 纯 CPU 模式：所有算子串行在 CPU 上执行                          │
 * │   ──[conv1]──[relu]──[conv2]──[relu]──[pool]──[add]──...──    │
 * │   总时间 = Σ 所有算子的 CPU 时间                                │
 * │                                                                 │
 * │ 纯 GPU 模式：所有算子串行在 GPU 上执行                          │
 * │   ──[conv1]──[relu]──[conv2]──[relu]──[pool]──[add]──...──    │
 * │   总时间 = Σ 所有算子的 GPU 时间 + Σ launch 开销               │
 * │   注意：小算子在 GPU 上反而更慢（launch 开销占比大）             │
 * │                                                                 │
 * │ 混合并行模式：按阶段执行，同阶段内 CPU 和 GPU 可并行             │
 * │   Stage 0: CPU──[shortcut]──  同时  GPU──[conv1]──             │
 * │   Stage 1: CPU──[relu1]──     同时  GPU 闲置                    │
 * │   Stage 2: CPU──[relu2]──     同时  GPU──[conv2]──             │
 * │   Stage 3: CPU──[add]──       同时  GPU 闲置                    │
 * │   ...                                                           │
 * │   总时间 ≈ max(CPU时间线, GPU时间线) → 比任一纯模式都快！       │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * 混合并行取胜的条件：
 * 1. 图中有可并行的算子对（无依赖）
 * 2. 并行算子可以分别放在 CPU 和 GPU 上
 * 3. CPU 和 GPU 的计算时间有一定重叠
 *
 * 编译运行:
 *   g++ -std=c++17 -O2 -o step5_hybrid_execution step5_hybrid_execution.cpp
 *   ./step5_hybrid_execution
 */

#include <algorithm>
#include <chrono>
#include <functional>
#include <future>
#include <iostream>
#include <iomanip>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ============================================================================
// 基础数据结构
// ============================================================================

enum class Device { CPU, GPU };
inline const char* deviceToString(Device d) {
    return d == Device::CPU ? "CPU" : "GPU";
}

struct OpNode {
    int id = -1;
    std::string name;
    std::string type;
    std::vector<int> inputs;
    std::vector<int> outputs;
    int64_t estimatedFlops = 0;
};

class OpGraph {
public:
    int addOp(const std::string& name, const std::string& type,
              const std::vector<int>& inputs = {}, int64_t flops = 0) {
        int id = static_cast<int>(ops_.size());
        ops_.push_back({id, name, type, inputs, {}, flops});
        for (int dep : inputs) ops_[dep].outputs.push_back(id);
        return id;
    }
    const OpNode& getOp(int id) const { return ops_.at(id); }
    const std::vector<OpNode>& getAllOps() const { return ops_; }
    int numOps() const { return static_cast<int>(ops_.size()); }

    std::vector<int> topologicalSort() const {
        int n = numOps();
        std::vector<int> inDeg(n, 0);
        for (const auto& op : ops_) inDeg[op.id] = static_cast<int>(op.inputs.size());
        std::vector<int> q;
        for (int i = 0; i < n; ++i) if (inDeg[i] == 0) q.push_back(i);
        std::vector<int> order;
        for (size_t i = 0; i < q.size(); ++i) {
            int u = q[i];
            order.push_back(u);
            for (int v : ops_[u].outputs) {
                if (--inDeg[v] == 0) q.push_back(v);
            }
        }
        return order;
    }

    /// 计算分层执行阶段：同层算子互不依赖，可并行
    std::vector<std::vector<int>> computeStages() const {
        int n = numOps();
        auto order = topologicalSort();
        std::vector<int> level(n, 0);
        for (int u : order) {
            for (int v : ops_[u].outputs) {
                level[v] = std::max(level[v], level[u] + 1);
            }
        }
        int maxL = 0;
        for (int l : level) maxL = std::max(maxL, l);
        std::vector<std::vector<int>> stages(maxL + 1);
        for (int i = 0; i < n; ++i) stages[level[i]].push_back(i);
        return stages;
    }

private:
    std::vector<OpNode> ops_;
};

/// 调度决策
struct DispatchDecision {
    int opId = -1;
    Device device = Device::CPU;
    std::string reason;
};

// ============================================================================
// 执行计划（与框架 HybridExecutor.h 中的 ExecutionPlan 对齐）
// ============================================================================

/**
 * @struct ExecutionPlan
 * @brief 编译好的执行方案：每个算子的调度决策 + 分阶段执行列表
 *
 * 示例（ResNet 风格的图）：
 *   decisions = [
 *     {opId:0, device:GPU, reason:"大卷积"},        // conv1
 *     {opId:1, device:CPU, reason:"逐元素"},         // relu1
 *     {opId:2, device:CPU, reason:"逐元素"},         // shortcut (identity)
 *     {opId:3, device:GPU, reason:"大卷积"},         // conv2
 *     {opId:4, device:CPU, reason:"逐元素"},         // relu2
 *     {opId:5, device:CPU, reason:"逐元素"},         // add
 *   ]
 *   stages = [
 *     [0, 2],   ← Stage 0: conv1(GPU) 和 shortcut(CPU) 可并行！
 *     [1],      ← Stage 1: relu1 依赖 conv1
 *     [3],      ← Stage 2: conv2 依赖 relu1
 *     [4],      ← Stage 3: relu2 依赖 conv2
 *     [5],      ← Stage 4: add 依赖 relu2 和 shortcut
 *   ]
 */
struct ExecutionPlan {
    std::vector<DispatchDecision> decisions;  ///< 每个算子的调度决策
    std::vector<std::vector<int>> stages;     ///< 分阶段执行列表

    /// 打印执行计划
    void print(const OpGraph& graph) const {
        std::cout << "=== 执行计划 ===\n\n";

        // 打印调度决策
        std::cout << "调度决策:\n";
        std::cout << std::string(70, '-') << "\n";
        for (const auto& dec : decisions) {
            const auto& op = graph.getOp(dec.opId);
            std::cout << "  " << std::left << std::setw(12) << op.name
                      << std::setw(16) << op.type
                      << "→ " << std::setw(4) << deviceToString(dec.device)
                      << " (" << dec.reason << ")\n";
        }
        std::cout << "\n";

        // 打印分阶段执行
        std::cout << "分阶段执行方案:\n";
        std::cout << std::string(60, '-') << "\n";
        for (size_t s = 0; s < stages.size(); ++s) {
            std::cout << "  Stage " << s << ": ";
            for (size_t i = 0; i < stages[s].size(); ++i) {
                int opId = stages[s][i];
                const auto& op = graph.getOp(opId);
                // 找到该算子的调度决策
                Device dev = Device::CPU;
                for (const auto& d : decisions) {
                    if (d.opId == opId) { dev = d.device; break; }
                }
                std::cout << op.name << "[" << deviceToString(dev) << "]";
                if (i + 1 < stages[s].size()) std::cout << " || ";
            }
            if (stages[s].size() > 1) {
                std::cout << "  ← 可并行!";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
};

// ============================================================================
// 启发式调度策略（简化版，与 step3 对齐）
// ============================================================================

class HeuristicDispatch {
public:
    DispatchDecision dispatch(const OpNode& op) {
        DispatchDecision d;
        d.opId = op.id;
        if (isElementWiseOp(op.type)) {
            d.device = Device::CPU;
            d.reason = "逐元素操作";
        } else if (op.estimatedFlops < 10'000'000) {
            d.device = Device::CPU;
            d.reason = "小算子(<10M FLOPs)";
        } else if (op.estimatedFlops > 100'000'000) {
            d.device = Device::GPU;
            d.reason = "大算子(>100M FLOPs)";
        } else {
            d.device = Device::CPU;
            d.reason = "中等算子,保守CPU";
        }
        return d;
    }
private:
    static bool isElementWiseOp(const std::string& t) {
        return t == "ReLU" || t == "Sigmoid" || t == "Add"
            || t == "Mul" || t == "TanH" || t == "Identity";
    }
};

// ============================================================================
// 异构并行执行器（与框架 HybridExecutor.h 对齐）
// ============================================================================

class HybridExecutor {
public:
    /**
     * @brief 编译执行计划：将算子图 + 调度策略 → 分阶段执行方案
     *
     * 编译过程（对应框架 API: HybridExecutor::compile(graph)）：
     * 1. 对图进行拓扑排序
     * 2. 对每个算子应用调度策略，获取调度决策
     * 3. 根据依赖关系将算子划分为执行阶段
     *    - 同阶段内的算子互不依赖，可并行
     *    - 阶段之间串行（必须等前一阶段完成）
     */
    ExecutionPlan compile(const OpGraph& graph) {
        ExecutionPlan plan;

        // 步骤1: 对每个算子应用调度策略
        HeuristicDispatch heuristic;
        for (const auto& op : graph.getAllOps()) {
            plan.decisions.push_back(heuristic.dispatch(op));
        }

        // 步骤2: 计算执行阶段
        plan.stages = graph.computeStages();

        return plan;
    }

    /**
     * @brief 时间线条目
     */
    struct TimelineEntry {
        std::string opName;
        Device device;
        double startMs;
        double durationMs;
        int stage;
    };

    /**
     * @brief 模拟并行执行 —— 生成时间线
     *
     * 真正的执行逻辑（框架中）：
     *   for each stage in plan.stages:
     *       对同阶段内所有算子，按设备并行启动:
     *         CPU 算子 → std::async 在 CPU 线程池执行
     *         GPU 算子 → std::async 提交到 GPU 队列
     *       等待所有 future 完成（阶段同步点）
     *
     * 这里用模拟替代真实执行，以便观察时间线。
     *
     * @param graph       算子图
     * @param plan        执行计划
     * @param forceDevice 纯模式：强制所有算子到同一设备
     * @param forceMode   是否启用纯模式
     */
    std::vector<TimelineEntry> simulateExecution(
            const OpGraph& graph,
            const ExecutionPlan& plan,
            Device forceDevice = Device::CPU,
            bool forceMode = false) const {

        std::vector<TimelineEntry> timeline;
        double cpuBusyUntil = 0.0;  // CPU 当前忙到什么时候
        double gpuBusyUntil = 0.0;  // GPU 当前忙到什么时候
        double stageStartMs = 0.0;  // 当前阶段的最早开始时间

        for (size_t s = 0; s < plan.stages.size(); ++s) {
            const auto& stage = plan.stages[s];

            double stageEndTime = stageStartMs;

            for (int opId : stage) {
                const auto& op = graph.getOp(opId);

                // 确定执行设备
                Device dev = Device::CPU;
                if (forceMode) {
                    dev = forceDevice;  // 纯模式：强制到同一设备
                } else {
                    // 混合模式：使用调度决策
                    for (const auto& d : plan.decisions) {
                        if (d.opId == opId) { dev = d.device; break; }
                    }
                }

                // 模拟执行时间
                double timeMs = estimateTime(op, dev);

                // 计算开始时间
                // 关键逻辑：同设备需串行，不同设备可并行
                double startMs;
                if (dev == Device::CPU) {
                    startMs = cpuBusyUntil;  // CPU 必须等上一个 CPU 任务完成
                    cpuBusyUntil = startMs + timeMs;
                } else {
                    startMs = gpuBusyUntil;  // GPU 必须等上一个 GPU 任务完成
                    gpuBusyUntil = startMs + timeMs;
                }

                timeline.push_back({op.name, dev, startMs, timeMs, static_cast<int>(s)});
                stageEndTime = std::max(stageEndTime, startMs + timeMs);
            }

            // 阶段同步：下一阶段必须等当前阶段所有算子完成
            stageStartMs = stageEndTime;
            cpuBusyUntil = std::max(cpuBusyUntil, stageStartMs);
            gpuBusyUntil = std::max(gpuBusyUntil, stageStartMs);
        }

        return timeline;
    }

private:
    /**
     * 模拟执行时间估算
     *
     * 模型参数：
     * - CPU: 逐元素约 8 GFLOPS，计算密集约 40 GFLOPS
     * - GPU: 约 400 GFLOPS + 20us launch 开销
     */
    double estimateTime(const OpNode& op, Device device) const {
        double flops = static_cast<double>(op.estimatedFlops);

        if (device == Device::CPU) {
            bool isElem = (op.type == "ReLU" || op.type == "Add"
                        || op.type == "Identity" || op.type == "Sigmoid"
                        || op.type == "Mul" || op.type == "TanH");
            double gflops = isElem ? 8.0 : 40.0;
            double computeMs = flops / (gflops * 1e6);
            return computeMs + 0.005;  // 加微小固定开销
        } else {
            double computeMs = flops / (400.0 * 1e6);
            double launchMs = 0.02;    // 20us kernel launch 开销
            return computeMs + launchMs;
        }
    }
};

// ============================================================================
// 构建实验图（ResNet 风格：带并行分支）
// ============================================================================

OpGraph buildResNetStyleGraph() {
    OpGraph graph;

    // 主分支：大卷积链
    // conv1: 64→128 通道, 3x3, 约 150M FLOPs → GPU 候选
    int conv1 = graph.addOp("conv1", "Convolution", {}, 150'000'000);
    int bn1   = graph.addOp("bn1", "Add", {conv1}, 65'000);  // BatchNorm ≈ 逐元素
    int relu1 = graph.addOp("relu1", "ReLU", {bn1}, 65'000);

    // conv2: 128→256 通道, 3x3, 约 300M FLOPs → GPU 候选
    int conv2 = graph.addOp("conv2", "Convolution", {relu1}, 300'000'000);
    int bn2   = graph.addOp("bn2", "Add", {conv2}, 130'000);
    int relu2 = graph.addOp("relu2", "ReLU", {bn2}, 130'000);

    // conv3: 256→512 通道, 1x1, 约 200M FLOPs → GPU 候选
    int conv3 = graph.addOp("conv3", "Convolution", {relu2}, 200'000'000);
    int bn3   = graph.addOp("bn3", "Add", {conv3}, 260'000);
    int relu3 = graph.addOp("relu3", "ReLU", {bn3}, 260'000);

    // 残差捷径分支（和主分支并行！）
    int shortcut = graph.addOp("shortcut", "Identity", {}, 0);

    // 汇合
    int add   = graph.addOp("add", "Add", {relu3, shortcut}, 260'000);

    // 分类头
    int pool  = graph.addOp("pool", "Pooling", {add}, 130'000);
    int fc    = graph.addOp("fc", "FullyConnected", {pool}, 82'000);

    return graph;
}

// ============================================================================
// 辅助函数
// ============================================================================

static std::string formatFlops(int64_t f) {
    if (f >= 1'000'000'000) return std::to_string(f / 1'000'000'000) + "G";
    if (f >= 1'000'000) return std::to_string(f / 1'000'000) + "M";
    if (f >= 1'000) return std::to_string(f / 1'000) + "K";
    return std::to_string(f);
}

/// 绘制时间线条形图
static std::string makeBar(double start, double end, double scale, int width, char fill) {
    std::string bar(width, ' ');
    int s = std::max(0, std::min(static_cast<int>(start / scale), width - 1));
    int e = std::max(0, std::min(static_cast<int>(end / scale), width));
    for (int i = s; i < e && i < width; ++i) bar[i] = fill;
    return bar;
}

/// 打印时间线
void printTimeline(const std::string& title,
                   const std::vector<HybridExecutor::TimelineEntry>& timeline,
                   double scale) {
    std::cout << "  " << title << ":\n";
    std::cout << "  " << std::string(75, '-') << "\n";
    std::cout << "  " << std::left
              << std::setw(12) << "算子"
              << std::setw(6) << "设备"
              << std::fixed << std::setprecision(3)
              << std::setw(8) << "开始"
              << std::setw(8) << "耗时"
              << "CPU时间线              GPU时间线\n";

    double maxEnd = 0.0;
    for (const auto& t : timeline) {
        maxEnd = std::max(maxEnd, t.startMs + t.durationMs);
    }
    double s = maxEnd / 35.0;  // 时间线宽度约35字符

    for (const auto& t : timeline) {
        double end = t.startMs + t.durationMs;
        std::string cpuBar = t.device == Device::CPU
            ? makeBar(t.startMs, end, s, 20, '#') : std::string(20, ' ');
        std::string gpuBar = t.device == Device::GPU
            ? makeBar(t.startMs, end, s, 20, '=') : std::string(20, ' ');

        std::cout << "  " << std::left << std::setw(12) << t.opName
                  << std::setw(6) << deviceToString(t.device)
                  << std::fixed << std::setprecision(3)
                  << std::setw(8) << t.startMs
                  << std::setw(8) << t.durationMs
                  << "[" << cpuBar << "] "
                  << "[" << gpuBar << "]\n";
    }

    std::cout << "  总耗时: " << std::fixed << std::setprecision(3) << maxEnd << " ms\n\n";
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "==============================================================\n";
    std::cout << "  Lab2 Step5: CPU+GPU 混合并行执行 (Hybrid Execution)\n";
    std::cout << "==============================================================\n\n";

    // ── 构建算子图 ──
    OpGraph graph = buildResNetStyleGraph();

    std::cout << "实验图结构 (ResNet 风格，带并行分支):\n\n";
    std::cout << "  conv1 → bn1 → relu1 → conv2 → bn2 → relu2 → conv3 → bn3 → relu3 ──┐\n";
    std::cout << "                                                                     ├→ add → pool → fc\n";
    std::cout << "  shortcut ─────────────────────────────────────────────────────────┘\n\n";

    // ── 编译执行计划 ──
    // 对应框架 API: HybridExecutor::compile(graph)
    HybridExecutor executor;
    ExecutionPlan plan = executor.compile(graph);

    plan.print(graph);

    // ── 模式1: 纯 CPU 执行 ──
    std::cout << "=== 模式1: 纯 CPU 执行 ===\n\n";
    std::cout << "所有算子都在 CPU 上串行执行，无并行。\n";
    std::cout << "大卷积（conv1/conv2/conv3）在 CPU 上较慢，成为瓶颈。\n\n";

    auto cpuTimeline = executor.simulateExecution(graph, plan, Device::CPU, true);
    printTimeline("纯CPU时间线", cpuTimeline, 1.0);

    // ── 模式2: 纯 GPU 执行 ──
    std::cout << "=== 模式2: 纯 GPU 执行 ===\n\n";
    std::cout << "所有算子都在 GPU 上串行执行，无并行。\n";
    std::cout << "大卷积在 GPU 上很快，但小算子（ReLU/BN）的 launch 开销占比大，反而慢。\n\n";

    auto gpuTimeline = executor.simulateExecution(graph, plan, Device::GPU, true);
    printTimeline("纯GPU时间线", gpuTimeline, 1.0);

    // ── 模式3: 混合并行执行 ──
    std::cout << "=== 模式3: CPU+GPU 混合并行执行 ===\n\n";
    std::cout << "关键：同阶段内，CPU 算子和 GPU 算子可以并行执行！\n";
    std::cout << "例如：conv1(GPU) 和 shortcut(CPU) 同时开始，\n";
    std::cout << "      bn1/relu1(CPU) 在 conv1 完成后紧接着在 CPU 上执行。\n\n";

    auto hybridTimeline = executor.simulateExecution(graph, plan, Device::CPU, false);
    printTimeline("混合并行时间线", hybridTimeline, 1.0);

    // ── 性能对比表 ──
    std::cout << "==============================================================\n";
    std::cout << "  性能对比总表\n";
    std::cout << "==============================================================\n\n";

    double cpuTotalMs = 0, gpuTotalMs = 0, hybridTotalMs = 0;
    for (const auto& t : cpuTimeline) cpuTotalMs = std::max(cpuTotalMs, t.startMs + t.durationMs);
    for (const auto& t : gpuTimeline) gpuTotalMs = std::max(gpuTotalMs, t.startMs + t.durationMs);
    for (const auto& t : hybridTimeline) hybridTotalMs = std::max(hybridTotalMs, t.startMs + t.durationMs);

    std::cout << "  +------------------+-------------+-----------+------------------------+\n";
    std::cout << "  | 执行模式         | 总耗时(ms)  | 相对速度  | 说明                   |\n";
    std::cout << "  +------------------+-------------+-----------+------------------------+\n";

    std::cout << "  | 纯 CPU           | " << std::fixed << std::setprecision(3)
              << std::setw(11) << cpuTotalMs
              << " | " << std::setprecision(2) << std::setw(9) << (cpuTotalMs / cpuTotalMs) << "x"
              << " | 基准                  |\n";

    std::cout << "  | 纯 GPU           | " << std::setw(11) << gpuTotalMs
              << " | " << std::setw(9) << (cpuTotalMs / gpuTotalMs) << "x"
              << " | 大卷积快,小算子慢     |\n";

    std::cout << "  | CPU+GPU 混合并行 | " << std::setw(11) << hybridTotalMs
              << " | " << std::setw(9) << (cpuTotalMs / hybridTotalMs) << "x"
              << " | 各取所长+并行加速     |\n";

    std::cout << "  +------------------+-------------+-----------+------------------------+\n\n";

    // ── 详细分析混合并行的优势 ──
    std::cout << "=== 混合并行优势分析 ===\n\n";

    // 统计各模式的 CPU/GPU 利用率
    double cpuBusyTime = 0, gpuBusyTime = 0;
    for (const auto& t : hybridTimeline) {
        if (t.device == Device::CPU) cpuBusyTime += t.durationMs;
        else gpuBusyTime += t.durationMs;
    }

    std::cout << "混合模式下设备利用率:\n";
    std::cout << "  CPU 总工作时间: " << std::fixed << std::setprecision(3)
              << cpuBusyTime << " ms\n";
    std::cout << "  GPU 总工作时间: " << gpuBusyTime << " ms\n";
    std::cout << "  CPU 利用率: " << std::setprecision(1)
              << (cpuBusyTime / hybridTotalMs * 100.0) << "%\n";
    std::cout << "  GPU 利用率: " << std::setprecision(1)
              << (gpuBusyTime / hybridTotalMs * 100.0) << "%\n\n";

    std::cout << "为什么混合并行能赢？\n";
    std::cout << "  1. 大卷积(conv1/conv2/conv3)放GPU: 充分利用GPU并行算力\n";
    std::cout << "  2. 小算子(ReLU/BN/Add)放CPU: 避免GPU launch开销\n";
    std::cout << "  3. 并行分支(shortcut vs conv链): CPU和GPU各跑一个分支\n";
    std::cout << "  4. 阶段内并行: 同一时刻CPU和GPU都在工作，硬件利用率高\n\n";

    if (hybridTotalMs < cpuTotalMs && hybridTotalMs < gpuTotalMs) {
        double speedupVsCpu = cpuTotalMs / hybridTotalMs;
        double speedupVsGpu = gpuTotalMs / hybridTotalMs;
        std::cout << "结论: 混合并行比纯CPU快 " << std::fixed << std::setprecision(2)
                  << speedupVsCpu << "x，比纯GPU快 " << speedupVsGpu << "x！\n";
        std::cout << "异构计算的核心优势：让每个设备做它最擅长的事，同时工作。\n";
    }

    // ── 混合并行的条件与局限 ──
    std::cout << "\n=== 混合并行的适用条件 ===\n\n";
    std::cout << "混合并行能加速的前提条件:\n";
    std::cout << "  1. 图中有可并行分支（无依赖的算子对）\n";
    std::cout << "     → ResNet/Inception 等有并行分支的模型效果最好\n";
    std::cout << "     → 纯串行链式模型(VGG)没有并行机会\n";
    std::cout << "  2. CPU 和 GPU 都有足够的工作量\n";
    std::cout << "     → 如果 GPU 占了 95% 的工作量，CPU 的贡献微乎其微\n";
    std::cout << "  3. CPU-GPU 数据搬运开销可接受\n";
    std::cout << "     → 频繁的 CPU↔GPU 内存拷贝会抵消并行收益\n";
    std::cout << "     → 优化的关键：减少跨设备数据拷贝次数\n\n";

    std::cout << "局限与改进方向:\n";
    std::cout << "  - 当前实现是粗粒度调度（整个算子分到一个设备）\n";
    std::cout << "  - 改进1: 细粒度流水线 - 将 conv 拆分为 tile，CPU/GPU 处理不同 tile\n";
    std::cout << "  - 改进2: 数据预取 - GPU 执行当前算子时，CPU 提前准备下一个算子的数据\n";
    std::cout << "  - 改进3: 动态调度 - 运行时根据设备负载动态调整分配\n";

    // ── Lab2 总结 ──
    std::cout << "\n==============================================================\n";
    std::cout << "  Lab2 总结: 异构调度全流程\n";
    std::cout << "==============================================================\n\n";
    std::cout << "Step1: 设备探测  ── 了解硬件能力 (CPU核心/缓存/SIMD, GPU显存/Vulkan)\n";
    std::cout << "Step2: 算子图    ── 构建DAG, 拓扑排序, 分析并行性与关键路径\n";
    std::cout << "Step3: 启发式    ── 基于经验规则的调度 (小算子CPU, 大算子GPU)\n";
    std::cout << "Step4: PGO      ── 用真实性能数据指导调度 (比启发式更精确)\n";
    std::cout << "Step5: 混合并行  ── CPU+GPU协同执行, 各取所长, 并行加速\n\n";
    std::cout << "核心思想: 让每个设备做它最擅长的事，并让它们同时工作！\n";

    return 0;
}
