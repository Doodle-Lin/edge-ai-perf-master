/**
 * @file step3_heuristic_dispatch.cpp
 * @brief Lab2 Step3: 启发式规则调度 —— 根据算子特征和经验规则决定 CPU/GPU 分配
 *
 * ====== 本实验学习目标 ======
 * 1. 理解启发式调度的核心思想：用经验规则替代全局搜索
 * 2. 掌握 CPU vs GPU 的本质区别（kernel launch 开销、计算密度）
 * 3. 学会定义和应用调度规则
 * 4. 模拟执行，观察 CPU/GPU 时间线
 *
 * ====== 核心规则 ======
 * ┌──────────────────────────────────────────────────────────────┐
 * │ 规则1: FLOPs < 10M → CPU                                     │
 * │   原因: GPU kernel launch 开销约 5~50us，小算子的计算时间    │
 * │   可能比 launch 开销还短，放 GPU 反而更慢                      │
 * │                                                                │
 * │ 规则2: FLOPs > 100M → GPU                                     │
 * │   原因: 计算量足够大，GPU 的大量核心可以充分发挥并行优势       │
 * │   launch 开销相对于计算时间可忽略                               │
 * │                                                                │
 * │ 规则3: 逐元素操作 (ReLU/Add/Mul/Sigmoid) → CPU               │
 * │   原因: 逐元素操作是 memory-bound（计算密度 < 1），             │
 * │   GPU 带宽优势不足以弥补 launch 开销和数据搬运                  │
 * │                                                                │
 * │ 规则4: Conv2D 大 batch → GPU                                  │
 * │   原因: 大 batch 意味着更多并行数据，GPU 核心利用率更高        │
 * │   batch=1 时 GPU 利用率可能只有 10%，batch=32 可达 80%        │
 * └──────────────────────────────────────────────────────────────┘
 *
 * 编译运行:
 *   g++ -std=c++17 -O2 -o step3_heuristic_dispatch step3_heuristic_dispatch.cpp
 *   ./step3_heuristic_dispatch
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ============================================================================
// 前向声明：复用 step2 的 OpNode / OpGraph（简化版本）
// ============================================================================

struct OpNode {
    int id = -1;
    std::string name;
    std::string type;
    std::vector<int> inputs;
    std::vector<int> outputs;
    int64_t estimatedFlops = 0;
    int64_t estimatedMemBytes = 0;
    int batchSize = 1;  // 批量大小，用于 Conv2D 的规则判断
};

class OpGraph {
public:
    int addOp(const std::string& name, const std::string& type,
              const std::vector<int>& inputs = {},
              int64_t flops = 0, int64_t memBytes = 0, int batch = 1) {
        int id = static_cast<int>(ops_.size());
        ops_.push_back({id, name, type, inputs, {}, flops, memBytes, batch});
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
        // 用 vector 模拟 queue（小图无需优化）
        for (size_t i = 0; i < q.size(); ++i) {
            int u = q[i];
            order.push_back(u);
            for (int v : ops_[u].outputs) {
                if (--inDeg[v] == 0) q.push_back(v);
            }
        }
        return order;
    }

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

// ============================================================================
// 设备枚举与调度决策（与框架 DispatchStrategy.h 对齐）
// ============================================================================

enum class Device { CPU, GPU };

inline const char* deviceToString(Device d) {
    return d == Device::CPU ? "CPU" : "GPU";
}

/// 调度决策：不仅记录"分到哪"，还记录"为什么"
struct DispatchDecision {
    int opId = -1;
    Device device = Device::CPU;
    std::string reason;  // 决策原因——生产环境中极其重要的可追溯性
};

// ============================================================================
// 启发式调度策略（与框架 HeuristicDispatch.h 对齐）
// ============================================================================

class HeuristicDispatch {
public:
    HeuristicDispatch() = default;

    // ──────────────── 核心调度逻辑 ────────────────

    /**
     * @brief 为单个算子决定执行设备
     *
     * 决策流程（按优先级从高到低）：
     * 1. 逐元素操作 → CPU（计算密度极低，GPU 无优势）
     * 2. 小算子 (FLOPs < smallFlopsThreshold) → CPU（launch 开销占比大）
     * 3. 大算子 (FLOPs > largeFlopsThreshold) → GPU（计算量足以摊薄 launch 开销）
     * 4. Conv2D 大 batch → GPU（GPU 核心利用率高）
     * 5. 默认 → CPU（保守策略，避免小算子浪费 GPU 资源）
     */
    DispatchDecision dispatch(const OpNode& op) {
        DispatchDecision decision;
        decision.opId = op.id;

        // ── 规则1: 逐元素操作 → CPU ──
        // 逐元素操作包括：ReLU、Sigmoid、Add、Mul、TanH 等
        // 特点：每个元素只做 1 次比较/运算，FLOPs ≈ 元素数
        // 计算密度 FLOPs/Byte ≈ 0.25（每个 float: 4B 读取 + 1 FLOP + 4B 写入 = 9B/1FLOP）
        // 这种 memory-bound 操作在 GPU 上没有优势
        if (isElementWiseOp(op.type)) {
            decision.device = Device::CPU;
            decision.reason = "逐元素操作(计算密度极低, memory-bound) → CPU";
            return decision;
        }

        // ── 规则2: 小算子 (FLOPs < 10M) → CPU ──
        // GPU kernel launch 开销约 5~50us
        // 如果算子本身只执行 2us，launch 开销就是瓶颈
        // 10M FLOPs 在 CPU 上约 0.5~2ms，在 GPU 上可能 0.1ms + 0.05ms launch = 0.15ms
        // 但加上 CPU→GPU 数据拷贝（~0.1ms/MB），总时间可能比 CPU 还长
        if (op.estimatedFlops < smallFlopsThreshold_) {
            decision.device = Device::CPU;
            decision.reason = "小算子(FLOPs=" + formatFlops(op.estimatedFlops)
                            + " < " + formatFlops(smallFlopsThreshold_)
                            + "), GPU launch 开销占比大 → CPU";
            return decision;
        }

        // ── 规则3: 大算子 (FLOPs > 100M) → GPU ──
        // 100M FLOPs 在 CPU 上约 5~20ms，在 GPU 上约 0.5~2ms
        // launch 开销 0.05ms 相对于 1ms 的计算时间可忽略
        // GPU 的数据并行优势充分体现
        if (op.estimatedFlops > largeFlopsThreshold_) {
            decision.device = Device::GPU;
            decision.reason = "大算子(FLOPs=" + formatFlops(op.estimatedFlops)
                            + " > " + formatFlops(largeFlopsThreshold_)
                            + "), GPU 数据并行优势明显 → GPU";
            return decision;
        }

        // ── 规则4: Conv2D 大 batch → GPU ──
        // batch=1 时 GPU 利用率可能只有 10%
        // batch=8 可达 40%，batch=32 可达 80%
        // 批量越大，GPU 的并行度越高，计算效率越好
        if (op.type == "Convolution" && op.batchSize >= 8) {
            decision.device = Device::GPU;
            decision.reason = "Conv2D 大批量(batch=" + std::to_string(op.batchSize)
                            + "), GPU 核心利用率高 → GPU";
            return decision;
        }

        // ── 规则5: 默认 → CPU ──
        // 保守策略：不确定时放 CPU
        // 因为 CPU 的 launch 开销为 0，最坏情况不会比 GPU 慢太多
        // 而 GPU 的最坏情况（小算子）可能比 CPU 慢数倍
        decision.device = Device::CPU;
        decision.reason = "默认(中等规模算子, 保守策略避免 GPU 开销) → CPU";
        return decision;
    }

    // ──────────────── 配置接口 ────────────────

    void setSmallFlopsThreshold(int64_t t) { smallFlopsThreshold_ = t; }
    void setLargeFlopsThreshold(int64_t t) { largeFlopsThreshold_ = t; }

private:
    int64_t smallFlopsThreshold_ = 10'000'000;   ///< 10M FLOPs
    int64_t largeFlopsThreshold_ = 100'000'000;  ///< 100M FLOPs

    /// 判断是否为逐元素操作
    bool isElementWiseOp(const std::string& type) const {
        return type == "ReLU" || type == "Sigmoid" || type == "TanH"
            || type == "Add"  || type == "Mul"     || type == "Identity";
    }

    static std::string formatFlops(int64_t f) {
        if (f >= 1'000'000'000) return std::to_string(f / 1'000'000'000) + "G";
        if (f >= 1'000'000) return std::to_string(f / 1'000'000) + "M";
        if (f >= 1'000) return std::to_string(f / 1'000) + "K";
        return std::to_string(f);
    }
};

// ============================================================================
// 模拟执行器：根据调度决策模拟 CPU/GPU 执行时间
// ============================================================================

struct SimResult {
    std::string opName;
    std::string opType;
    Device device;
    double simTimeMs;   // 模拟执行时间
    std::string reason;
};

/**
 * 模拟执行时间估算
 *
 * 模型简化假设：
 * - CPU 算力: 约 50 GFLOPS (AVX2) 或 10 GFLOPS (朴素)
 * - GPU 算力: 约 500 GFLOPS (但有小算子惩罚)
 * - GPU kernel launch 开销: 约 0.02ms (20us)
 * - CPU↔GPU 数据搬运: 约 0.005ms/MB
 *
 * 注意：这只是教学用的简化模型，实际性能取决于具体硬件和实现。
 */
class SimulatedExecutor {
public:
    SimulatedExecutor() : rng_(42) {}

    /// 模拟单个算子的执行时间
    double estimateTime(const OpNode& op, Device device) const {
        double flops = static_cast<double>(op.estimatedFlops);
        double memMB = static_cast<double>(op.estimatedMemBytes) / (1024.0 * 1024.0);

        if (device == Device::CPU) {
            // CPU 模型: 时间 = FLOPs / GFLOPS + 内存访问延迟
            // 逐元素操作用低 GFLOPS（内存瓶颈），其他用高 GFLOPS
            double gflops = isElementWise(op.type) ? 8.0 : 40.0;
            double computeMs = flops / (gflops * 1e6);  // GFLOPS → ms
            double memMs = memMB * 0.01;  // 约 10us/MB 内存延迟
            return computeMs + memMs + 0.001;  // 加微小固定开销
        } else {
            // GPU 模型: 时间 = FLOPs / GFLOPS + launch开销 + 数据搬运
            double gflops = 400.0;  // GPU 计算吞吐
            double computeMs = flops / (gflops * 1e6);
            double launchMs = 0.02;  // 20us kernel launch 开销
            double transferMs = memMB * 0.005;  // 约 5us/MB 搬运
            return computeMs + launchMs + transferMs;
        }
    }

    /// 模拟执行整个图，按阶段输出时间线
    std::vector<SimResult> executeWithTimeline(
            const OpGraph& graph,
            const std::vector<DispatchDecision>& decisions) {

        std::vector<SimResult> results;

        // 建立快速查找：opId → decision
        std::map<int, DispatchDecision> decisionMap;
        for (const auto& d : decisions) decisionMap[d.opId] = d;

        auto stages = graph.computeStages();

        for (size_t s = 0; s < stages.size(); ++s) {
            for (int opId : stages[s]) {
                const auto& op = graph.getOp(opId);
                const auto& dec = decisionMap[opId];
                double timeMs = estimateTime(op, dec.device);

                // 添加少量随机抖动，使模拟更真实
                std::uniform_real_distribution<double> jitter(-0.005, 0.005);
                timeMs += jitter(rng_);
                if (timeMs < 0.001) timeMs = 0.001;

                results.push_back({
                    op.name, op.type, dec.device, timeMs, dec.reason
                });
            }
        }

        return results;
    }

private:
    mutable std::mt19937 rng_;

    static bool isElementWise(const std::string& type) {
        return type == "ReLU" || type == "Sigmoid" || type == "Add"
            || type == "Mul" || type == "TanH" || type == "Identity";
    }
};

// ============================================================================
// 构建实验用的算子图（典型 CNN + 残差结构）
// ============================================================================

OpGraph buildExperimentGraph() {
    OpGraph graph;
    // 主分支
    int conv1 = graph.addOp("conv1", "Convolution", {}, 150'000'000, 80'000'000, 1);
    int relu1 = graph.addOp("relu1", "ReLU", {conv1}, 65'000, 260'000);
    int conv2 = graph.addOp("conv2", "Convolution", {relu1}, 9'400'000, 19'000'000, 1);
    int relu2 = graph.addOp("relu2", "ReLU", {conv2}, 16'000, 65'000);
    int pool1 = graph.addOp("pool1", "Pooling", {relu2}, 16'000, 65'000);
    // 残差捷径
    int shortcut = graph.addOp("shortcut", "Identity", {}, 0, 32'000);
    // 汇合
    int add = graph.addOp("add", "Add", {pool1, shortcut}, 16'000, 65'000);
    // 后续层
    int conv3 = graph.addOp("conv3", "Convolution", {add}, 200'000'000, 100'000'000, 1);
    int relu3 = graph.addOp("relu3", "ReLU", {conv3}, 130'000, 520'000);
    int fc    = graph.addOp("fc", "FullyConnected", {relu3}, 82'000, 164'000);
    return graph;
}

// ============================================================================
// 格式化辅助
// ============================================================================

static std::string formatFlops(int64_t f) {
    if (f >= 1'000'000'000) return std::to_string(f / 1'000'000'000) + "G";
    if (f >= 1'000'000) return std::to_string(f / 1'000'000) + "M";
    if (f >= 1'000) return std::to_string(f / 1'000) + "K";
    return std::to_string(f);
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "==============================================================\n";
    std::cout << "  Lab2 Step3: 启发式规则调度 (Heuristic Dispatch)\n";
    std::cout << "==============================================================\n\n";

    // ── 构建算子图 ──
    OpGraph graph = buildExperimentGraph();

    std::cout << "实验图结构 (CNN + 残差连接):\n";
    std::cout << "  conv1(150M) → relu1(65K) → conv2(9.4M) → relu2(16K) → pool1(16K) ─┐\n";
    std::cout << "                                                                      ├→ add → conv3(200M) → relu3(130K) → fc(82K)\n";
    std::cout << "  shortcut(0) ───────────────────────────────────────────────────────┘\n\n";

    // ── 创建调度器并应用规则 ──
    HeuristicDispatch heuristic;

    std::cout << "=== 调度规则 ===\n";
    std::cout << "  规则1: 逐元素操作 (ReLU/Add/Mul/Sigmoid/Identity) → CPU\n";
    std::cout << "         原因: 计算密度极低 (memory-bound), GPU launch开销 > 计算收益\n";
    std::cout << "  规则2: FLOPs < 10M → CPU\n";
    std::cout << "         原因: 小算子的计算时间可能比 GPU kernel launch 开销还短\n";
    std::cout << "  规则3: FLOPs > 100M → GPU\n";
    std::cout << "         原因: 计算量足够大, GPU 并行优势充分, launch 开销可忽略\n";
    std::cout << "  规则4: Conv2D 大 batch (≥8) → GPU\n";
    std::cout << "         原因: 批量大时 GPU 核心利用率高\n";
    std::cout << "  规则5: 默认 → CPU (保守策略)\n\n";

    // ── 对每个算子应用启发式规则 ──
    std::vector<DispatchDecision> decisions;
    auto topoOrder = graph.topologicalSort();

    std::cout << "=== 调度决策 ===\n";
    std::cout << std::string(100, '-') << "\n";
    std::cout << std::left
              << std::setw(6)  << "ID"
              << std::setw(12) << "名称"
              << std::setw(16) << "类型"
              << std::setw(12) << "FLOPs"
              << std::setw(6)  << "设备"
              << "原因\n";
    std::cout << std::string(100, '-') << "\n";

    for (int id : topoOrder) {
        const auto& op = graph.getOp(id);
        auto decision = heuristic.dispatch(op);
        decisions.push_back(decision);

        std::cout << std::left
                  << std::setw(6)  << op.id
                  << std::setw(12) << op.name
                  << std::setw(16) << op.type
                  << std::setw(12) << formatFlops(op.estimatedFlops)
                  << std::setw(6)  << deviceToString(decision.device)
                  << decision.reason << "\n";
    }
    std::cout << std::string(100, '-') << "\n\n";

    // ── 统计调度结果 ──
    int cpuCount = 0, gpuCount = 0;
    int64_t cpuFlops = 0, gpuFlops = 0;
    for (size_t i = 0; i < decisions.size(); ++i) {
        if (decisions[i].device == Device::CPU) {
            cpuCount++;
            cpuFlops += graph.getOp(decisions[i].opId).estimatedFlops;
        } else {
            gpuCount++;
            gpuFlops += graph.getOp(decisions[i].opId).estimatedFlops;
        }
    }

    std::cout << "调度统计:\n";
    std::cout << "  CPU 算子: " << cpuCount << " 个, 总 FLOPs = "
              << formatFlops(cpuFlops) << "\n";
    std::cout << "  GPU 算子: " << gpuCount << " 个, 总 FLOPs = "
              << formatFlops(gpuFlops) << "\n";
    std::cout << "  GPU 计算占比: "
              << std::fixed << std::setprecision(1)
              << (100.0 * gpuFlops / (cpuFlops + gpuFlops)) << "%\n";
    std::cout << "  → 大部分计算量分配给 GPU (合理!)，小算子留在 CPU (也合理!)\n\n";

    // ── 模拟执行并展示时间线 ──
    std::cout << "=== 模拟执行时间线 ===\n\n";
    SimulatedExecutor executor;
    auto results = executor.executeWithTimeline(graph, decisions);

    // 计算按阶段的时间线
    auto stages = graph.computeStages();

    double cpuBusyUntil = 0.0;  // CPU 忙到什么时候
    double gpuBusyUntil = 0.0;  // GPU 忙到什么时候
    double totalEnd = 0.0;

    std::cout << "  阶段  算子          设备   执行(ms)   CPU时间线              GPU时间线\n";
    std::cout << std::string(90, '-') << "\n";

    size_t resIdx = 0;
    for (size_t s = 0; s < stages.size(); ++s) {
        // 同一阶段内的算子可以并行
        double stageEndTime = 0.0;

        for (int opId : stages[s]) {
            const auto& r = results[resIdx++];
            double start, end;

            if (r.device == Device::CPU) {
                start = cpuBusyUntil;
                end = start + r.simTimeMs;
                cpuBusyUntil = end;
            } else {
                start = gpuBusyUntil;
                end = start + r.simTimeMs;
                gpuBusyUntil = end;
            }
            stageEndTime = std::max(stageEndTime, end);
            totalEnd = std::max(totalEnd, end);

            // 生成时间线条形图
            auto makeBar = [](double start, double dur, int width, char c) -> std::string {
                std::string bar(width, ' ');
                int s = std::min(static_cast<int>(start / 0.5), width - 1);
                int e = std::min(static_cast<int>((start + dur) / 0.5), width);
                for (int i = s; i < e && i < width; ++i) bar[i] = c;
                return bar;
            };

            std::string cpuBar = r.device == Device::CPU
                ? makeBar(start, r.simTimeMs, 20, '#') : std::string(20, ' ');
            std::string gpuBar = r.device == Device::GPU
                ? makeBar(start, r.simTimeMs, 20, '=') : std::string(20, ' ');

            std::cout << "  S" << s
                      << std::left << std::setw(4) << ""
                      << std::setw(14) << r.opName
                      << std::setw(6) << deviceToString(r.device)
                      << std::fixed << std::setprecision(3)
                      << std::setw(10) << r.simTimeMs
                      << "[" << cpuBar << "] "
                      << "[" << gpuBar << "]\n";
        }
    }

    double totalSimMs = 0;
    for (const auto& r : results) totalSimMs += r.simTimeMs;

    std::cout << "\n";
    std::cout << "  图例: CPU=# (并行可能重叠)  GPU== (并行可能重叠)\n";
    std::cout << "  说明: 同一阶段内的 CPU 和 GPU 算子可以并行执行！\n";
    std::cout << "  串行总耗时(所有算子加起来): " << std::fixed
              << std::setprecision(3) << totalSimMs << " ms\n";
    std::cout << "  并行模拟耗时(考虑阶段内并行): " << std::fixed
              << std::setprecision(3) << totalEnd << " ms\n\n";

    // ── 对比：如果所有算子都放 CPU 或都放 GPU ──
    std::cout << "=== 调度策略对比 ===\n\n";

    // 全 CPU
    double allCpuMs = 0.0;
    for (int id : topoOrder) {
        allCpuMs += executor.estimateTime(graph.getOp(id), Device::CPU);
    }

    // 全 GPU
    double allGpuMs = 0.0;
    for (int id : topoOrder) {
        allGpuMs += executor.estimateTime(graph.getOp(id), Device::GPU);
    }

    // 启发式混合
    double heuristicMs = 0.0;
    for (const auto& dec : decisions) {
        heuristicMs += executor.estimateTime(graph.getOp(dec.opId), dec.device);
    }

    std::cout << "  策略            总耗时(ms)   相对速度\n";
    std::cout << "  ────────────────────────────────────────\n";
    std::cout << "  全 CPU          " << std::fixed << std::setprecision(3)
              << std::setw(12) << allCpuMs << "  1.00x (基准)\n";
    std::cout << "  全 GPU          " << std::setw(12) << allGpuMs
              << "  " << std::setprecision(2) << (allCpuMs / allGpuMs) << "x\n";
    std::cout << "  启发式混合      " << std::setw(12) << heuristicMs
              << "  " << std::setprecision(2) << (allCpuMs / heuristicMs) << "x\n";
    std::cout << "\n";
    std::cout << "  分析:\n";
    std::cout << "  - 全 CPU: 大卷积(conv1/conv3)被 CPU 慢速拖累\n";
    std::cout << "  - 全 GPU: 小算子(ReLU/Add/Pool)的 launch 开销占比大，反而慢\n";
    std::cout << "  - 启发式混合: 大算子放 GPU，小算子放 CPU，各取所长\n";

    if (heuristicMs < allCpuMs && heuristicMs < allGpuMs) {
        std::cout << "  → 启发式混合调度胜出！异构计算的优势体现出来了。\n";
    }

    // ── 总结 ──
    std::cout << "\n==============================================================\n";
    std::cout << "  关键收获\n";
    std::cout << "==============================================================\n\n";
    std::cout << "1. 启发式调度 = 经验规则，不需要 profiling 数据，冷启动即可工作\n";
    std::cout << "2. 核心权衡: GPU launch 开销 vs 计算收益\n";
    std::cout << "   - FLOPs < 10M: launch 开销占比大 → CPU\n";
    std::cout << "   - FLOPs > 100M: launch 开销可忽略 → GPU\n";
    std::cout << "3. 逐元素操作总是放 CPU: 计算密度极低 (memory-bound)\n";
    std::cout << "4. 启发式的局限: 规则是静态的，不知道特定硬件的真实性能\n";
    std::cout << "   → 下一步用 profile-guided 调度解决\n";
    std::cout << "\n下一步: step4_profile_guided_dispatch.cpp\n";

    return 0;
}
