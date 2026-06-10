/**
 * @file step4_profile_guided_dispatch.cpp
 * @brief Lab2 Step4: Profile-Guided 调度 —— 用真实性能数据指导调度决策
 *
 * ====== 本实验学习目标 ======
 * 1. 理解 Profile-Guided Optimization (PGO) 的核心思想
 * 2. 学会构建和利用 profiling 查找表
 * 3. 对比启发式 vs profile-guided 的决策差异
 * 4. 理解为什么 PGO 总是 >= 启发式
 *
 * ====== 核心概念 ======
 *
 * 启发式调度 vs Profile-Guided 调度：
 * ┌──────────────────┬───────────────────────┬─────────────────────────┐
 * │                  │ 启发式 (Heuristic)     │ Profile-Guided (PGO)   │
 * ├──────────────────┼───────────────────────┼─────────────────────────┤
 * │ 数据来源          │ 经验规则（硬编码）      │ 真实测量数据             │
 * │ 冷启动            │ 可用（无需预热）        │ 不可用（需回退到启发式）  │
 * │ 精确度            │ 中等（规则不可能覆盖    │ 高（用数据说话）         │
 * │                  │ 所有硬件组合）           │                         │
 * │ 自适应            │ 否（规则是静态的）       │ 是（随硬件/负载变化）    │
 * │ 维护成本          │ 低（无需 profiling）    │ 中（需定期更新数据）     │
 * │ 典型提升          │ 基准                   │ 比启发式快 10~30%       │
 * └──────────────────┴───────────────────────┴─────────────────────────┘
 *
 * PGO 的工作流程：
 * 1. 第一轮推理：用启发式调度，同时用 OpProfiler 记录每个算子的 CPU/GPU 时间
 * 2. 构建 profile 表: opName → {cpuTime, gpuTime, bestDevice}
 * 3. 后续推理：查 profile 表，选择更快的设备
 * 4. 如果 profile 表中没有某算子 → 回退到启发式
 *
 * 为什么 PGO 总是 >= 启发式？
 * - 启发式的规则是通用的，不能适应特定硬件
 * - PGO 用真实测量数据替代猜测
 * - 类比：启发式像"看菜单点菜"，PGO 像"先试吃再点菜"
 *
 * 编译运行:
 *   g++ -std=c++17 -O2 -o step4_profile_guided_dispatch step4_profile_guided_dispatch.cpp
 *   ./step4_profile_guided_dispatch
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ============================================================================
// 基础数据结构（与框架对齐）
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
// 单个算子的 Profile 数据（与框架 ProfileGuidedDispatch.h 对齐）
// ============================================================================

struct OpProfileEntry {
    double cpuTimeMs = 0.0;   ///< CPU 执行耗时（毫秒）
    double gpuTimeMs = 0.0;   ///< GPU 执行耗时（毫秒）
    Device bestDevice = Device::CPU;  ///< 实测更快的设备
    int sampleCount  = 0;     ///< 采样次数

    /// 加速比 = cpuTimeMs / gpuTimeMs
    /// > 1.0 表示 GPU 更快，< 1.0 表示 CPU 更快
    double speedup() const {
        if (gpuTimeMs <= 0.0) return 0.0;
        return cpuTimeMs / gpuTimeMs;
    }
};

// ============================================================================
// 启发式调度（简化版，与 step3 对齐）
// ============================================================================

class HeuristicDispatch {
public:
    DispatchDecision dispatch(const OpNode& op) {
        DispatchDecision d;
        d.opId = op.id;

        if (isElementWiseOp(op.type)) {
            d.device = Device::CPU;
            d.reason = "逐元素操作 → CPU";
        } else if (op.estimatedFlops < 10'000'000) {
            d.device = Device::CPU;
            d.reason = "小算子(< 10M FLOPs) → CPU";
        } else if (op.estimatedFlops > 100'000'000) {
            d.device = Device::GPU;
            d.reason = "大算子(> 100M FLOPs) → GPU";
        } else {
            d.device = Device::CPU;
            d.reason = "默认(中等规模) → CPU";
        }
        return d;
    }

private:
    static bool isElementWiseOp(const std::string& type) {
        return type == "ReLU" || type == "Sigmoid" || type == "Add"
            || type == "Mul" || type == "TanH" || type == "Identity";
    }
};

// ============================================================================
// Profile-Guided 调度策略（与框架 ProfileGuidedDispatch.h 对齐）
// ============================================================================

class ProfileGuidedDispatch {
public:
    ProfileGuidedDispatch() : gpuSpeedupThreshold_(1.2) {}

    /**
     * @brief 根据历史 Profile 数据决定算子执行设备
     *
     * 决策流程：
     * 1. 在 profile 表中查找 op.name
     * 2. 如果找到且有 CPU 和 GPU 数据，选择更快的设备
     *    但要满足加速比阈值（GPU 须快 1.2x 以上，因为还有数据拷贝开销）
     * 3. 如果只有单设备数据，返回该设备
     * 4. 如果未找到，回退到启发式策略
     */
    DispatchDecision dispatch(const OpNode& op) {
        DispatchDecision d;
        d.opId = op.id;

        auto it = profileTable_.find(op.name);
        if (it != profileTable_.end() && it->second.sampleCount > 0) {
            const auto& entry = it->second;

            if (entry.cpuTimeMs > 0.0 && entry.gpuTimeMs > 0.0) {
                // 有双设备数据，做精确比较
                // 为什么需要加速比阈值？
                // 因为 GPU 执行需要 CPU→GPU 数据拷贝，
                // 仅计算核心快不够，必须快到能抵消拷贝开销才值得
                if (entry.gpuTimeMs * gpuSpeedupThreshold_ < entry.cpuTimeMs) {
                    d.device = Device::GPU;
                    d.reason = "Profile: GPU=" + formatTime(entry.gpuTimeMs)
                             + "ms < CPU=" + formatTime(entry.cpuTimeMs)
                             + "ms (加速" + std::to_string(static_cast<int>(entry.speedup() * 100))
                             + "%)";
                } else {
                    d.device = Device::CPU;
                    d.reason = "Profile: CPU=" + formatTime(entry.cpuTimeMs)
                             + "ms <= GPU=" + formatTime(entry.gpuTimeMs)
                             + "ms (加速比不足" + std::to_string(static_cast<int>(gpuSpeedupThreshold_ * 100))
                             + "%)";
                }
            } else if (entry.cpuTimeMs > 0.0) {
                d.device = Device::CPU;
                d.reason = "Profile: 只有CPU数据=" + formatTime(entry.cpuTimeMs) + "ms";
            } else if (entry.gpuTimeMs > 0.0) {
                d.device = Device::GPU;
                d.reason = "Profile: 只有GPU数据=" + formatTime(entry.gpuTimeMs) + "ms";
            } else {
                // 数据无效，回退
                auto fb = heuristic_.dispatch(op);
                d.device = fb.device;
                d.reason = "Profile数据无效, 回退启发式: " + fb.reason;
            }
        } else {
            // 没有该算子的 profile 数据，回退到启发式
            auto fb = heuristic_.dispatch(op);
            d.device = fb.device;
            d.reason = "无Profile数据, 回退启发式: " + fb.reason;
        }

        return d;
    }

    /// 手动添加一条 profile 记录
    void addProfileEntry(const std::string& opName, Device device, double timeMs) {
        auto& entry = profileTable_[opName];
        if (device == Device::CPU) {
            entry.cpuTimeMs = (entry.cpuTimeMs * entry.sampleCount + timeMs) / (entry.sampleCount + 1);
        } else {
            entry.gpuTimeMs = (entry.gpuTimeMs * entry.sampleCount + timeMs) / (entry.sampleCount + 1);
        }
        entry.sampleCount++;
        // 更新 bestDevice
        if (entry.cpuTimeMs > 0 && entry.gpuTimeMs > 0) {
            entry.bestDevice = (entry.gpuTimeMs * gpuSpeedupThreshold_ < entry.cpuTimeMs)
                               ? Device::GPU : Device::CPU;
        } else if (entry.cpuTimeMs > 0) {
            entry.bestDevice = Device::CPU;
        } else {
            entry.bestDevice = Device::GPU;
        }
    }

    /// 获取某个算子的 profile 数据
    const OpProfileEntry* getProfile(const std::string& opName) const {
        auto it = profileTable_.find(opName);
        return it != profileTable_.end() ? &it->second : nullptr;
    }

    /// 获取所有 profile 数据
    const std::map<std::string, OpProfileEntry>& allProfiles() const {
        return profileTable_;
    }

    /// 设置 GPU 加速比阈值
    void setGpuSpeedupThreshold(double t) { gpuSpeedupThreshold_ = t; }

    /// 将 profile 数据序列化为 JSON 字符串
    /// 与框架 ProfileGuidedDispatch::profileToJson() 对齐
    std::string profileToJson() const {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"version\": 1,\n";
        oss << "  \"op_profiles\": {\n";
        bool first = true;
        for (const auto& [name, entry] : profileTable_) {
            if (!first) oss << ",\n";
            first = false;
            oss << "    \"" << name << "\": {";
            oss << " \"cpu_ms\": " << std::fixed << std::setprecision(3) << entry.cpuTimeMs;
            oss << ", \"gpu_ms\": " << entry.gpuTimeMs;
            oss << ", \"best\": \"" << deviceToString(entry.bestDevice) << "\"";
            oss << ", \"samples\": " << entry.sampleCount;
            oss << " }";
        }
        oss << "\n  }\n";
        oss << "}\n";
        return oss.str();
    }

private:
    std::map<std::string, OpProfileEntry> profileTable_;
    HeuristicDispatch heuristic_;  ///< 回退策略
    double gpuSpeedupThreshold_;   ///< GPU 加速比阈值

    static std::string formatTime(double ms) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3) << ms;
        return oss.str();
    }
};

// ============================================================================
// 模拟 Profiler：生成模拟的 CPU/GPU 执行时间
// ============================================================================

class MockProfiler {
public:
    MockProfiler() : rng_(42) {}

    /**
     * 为每个算子生成模拟的 CPU 和 GPU 执行时间。
     *
     * 模拟模型（简化但合理）：
     * - CPU 时间 = FLOPs / GFLOPS + 内存延迟
     * - GPU 时间 = FLOPs / GFLOPS + launch 开销 + 数据搬运
     *
     * 关键：在某些硬件组合下，"中等算子"的 GPU 可能比启发式预期更快或更慢
     * 这就是 PGO 比启发式更准的原因——它能发现启发式规则覆盖不到的情况。
     */
    void profileAll(const OpGraph& graph) {
        for (const auto& op : graph.getAllOps()) {
            double flops = static_cast<double>(op.estimatedFlops);

            // CPU 时间：计算密集型算子约 40 GFLOPS，逐元素约 8 GFLOPS
            bool isElemWise = (op.type == "ReLU" || op.type == "Add"
                            || op.type == "Identity" || op.type == "Sigmoid");
            double cpuGflops = isElemWise ? 8.0 : 40.0;
            double cpuTime = flops / (cpuGflops * 1e6) + 0.005;  // 加固定开销

            // GPU 时间：计算吞吐约 400 GFLOPS，但有 20us launch 开销
            double gpuTime = flops / (400.0 * 1e6) + 0.02;  // 20us launch

            // 添加少量随机抖动（模拟真实测量波动）
            std::uniform_real_distribution<double> jitter(-0.002, 0.002);
            cpuTime += jitter(rng_);
            gpuTime += jitter(rng_);
            if (cpuTime < 0.001) cpuTime = 0.001;
            if (gpuTime < 0.001) gpuTime = 0.001;

            // 记录 profile 数据（同时测量 CPU 和 GPU 的时间）
            profileData_[op.name] = {cpuTime, gpuTime,
                (gpuTime < cpuTime) ? Device::GPU : Device::CPU, 1};
        }
    }

    /// 获取某算子的 profile 数据
    const OpProfileEntry* getProfile(const std::string& name) const {
        auto it = profileData_.find(name);
        return it != profileData_.end() ? &it->second : nullptr;
    }

    /// 导出为可被 ProfileGuidedDispatch 加载的数据
    const std::map<std::string, OpProfileEntry>& data() const { return profileData_; }

private:
    std::map<std::string, OpProfileEntry> profileData_;
    std::mt19937 rng_;
};

// ============================================================================
// 构建实验图
// ============================================================================

OpGraph buildExperimentGraph() {
    OpGraph graph;
    int conv1 = graph.addOp("conv1", "Convolution", {}, 150'000'000);
    int relu1 = graph.addOp("relu1", "ReLU", {conv1}, 65'000);
    int conv2 = graph.addOp("conv2", "Convolution", {relu1}, 9'400'000);
    int relu2 = graph.addOp("relu2", "ReLU", {conv2}, 16'000);
    int pool1 = graph.addOp("pool1", "Pooling", {relu2}, 16'000);
    int shortcut = graph.addOp("shortcut", "Identity", {}, 0);
    int add = graph.addOp("add", "Add", {pool1, shortcut}, 16'000);
    int conv3 = graph.addOp("conv3", "Convolution", {add}, 200'000'000);
    int relu3 = graph.addOp("relu3", "ReLU", {conv3}, 130'000);
    int fc = graph.addOp("fc", "FullyConnected", {relu3}, 82'000);
    return graph;
}

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
    std::cout << "  Lab2 Step4: Profile-Guided 调度 (Profile-Guided Dispatch)\n";
    std::cout << "==============================================================\n\n";

    OpGraph graph = buildExperimentGraph();

    // ── Part 1: 模拟 Profiling ──
    std::cout << "=== Part 1: 收集 Profiling 数据 ===\n\n";
    std::cout << "模拟第一轮推理：用启发式调度执行，同时用 OpProfiler 记录\n";
    std::cout << "每个算子在 CPU 和 GPU 上的执行时间。\n\n";

    MockProfiler profiler;
    profiler.profileAll(graph);

    std::cout << "Profiling 结果:\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << std::left
              << std::setw(12) << "算子"
              << std::setw(14) << "类型"
              << std::setw(10) << "FLOPs"
              << std::setw(12) << "CPU(ms)"
              << std::setw(12) << "GPU(ms)"
              << std::setw(8) << "加速比"
              << "更快设备\n";
    std::cout << std::string(80, '-') << "\n";

    for (const auto& op : graph.getAllOps()) {
        const auto* p = profiler.getProfile(op.name);
        if (p) {
            std::cout << std::left
                      << std::setw(12) << op.name
                      << std::setw(14) << op.type
                      << std::setw(10) << formatFlops(op.estimatedFlops)
                      << std::fixed << std::setprecision(3)
                      << std::setw(12) << p->cpuTimeMs
                      << std::setw(12) << p->gpuTimeMs
                      << std::setw(8) << std::setprecision(2) << p->speedup()
                      << deviceToString(p->bestDevice) << "\n";
        }
    }
    std::cout << "\n";

    // ── Part 2: 构建 Profile-Guided 调度策略 ──
    std::cout << "=== Part 2: 用 Profile 数据构建 PGO 调度策略 ===\n\n";

    ProfileGuidedDispatch pgd;

    // 将 profiler 的数据导入 PGO 调度器
    // 对应框架 API: pgd.updateFromProfiler(profiler)
    for (const auto& [name, entry] : profiler.data()) {
        pgd.addProfileEntry(name, Device::CPU, entry.cpuTimeMs);
        pgd.addProfileEntry(name, Device::GPU, entry.gpuTimeMs);
    }

    // ── Part 3: 对比启发式 vs PGO 的调度决策 ──
    std::cout << "=== Part 3: 启发式 vs Profile-Guided 决策对比 ===\n\n";

    HeuristicDispatch heuristic;

    std::cout << std::string(110, '-') << "\n";
    std::cout << std::left
              << std::setw(12) << "算子"
              << std::setw(14) << "类型"
              << std::setw(10) << "FLOPs"
              << std::setw(6)  << "启发式"
              << std::setw(6)  << "PGO"
              << std::setw(6)  << "差异"
              << "PGO 原因\n";
    std::cout << std::string(110, '-') << "\n";

    int agreeCount = 0, disagreeCount = 0;
    double heuristicTotalMs = 0.0;
    double pgdTotalMs = 0.0;

    for (const auto& op : graph.getAllOps()) {
        auto hDec = heuristic.dispatch(op);
        auto pDec = pgd.dispatch(op);
        const auto* prof = profiler.getProfile(op.name);

        bool same = (hDec.device == pDec.device);
        if (same) agreeCount++; else disagreeCount++;

        // 计算各自的总耗时
        double hTime = (hDec.device == Device::CPU) ? prof->cpuTimeMs : prof->gpuTimeMs;
        double pTime = (pDec.device == Device::CPU) ? prof->cpuTimeMs : prof->gpuTimeMs;
        heuristicTotalMs += hTime;
        pgdTotalMs += pTime;

        std::string diff = same ? "" : "<<<";
        std::cout << std::left
                  << std::setw(12) << op.name
                  << std::setw(14) << op.type
                  << std::setw(10) << formatFlops(op.estimatedFlops)
                  << std::setw(6)  << deviceToString(hDec.device)
                  << std::setw(6)  << deviceToString(pDec.device)
                  << std::setw(6)  << diff
                  << pDec.reason << "\n";
    }

    std::cout << std::string(110, '-') << "\n\n";

    std::cout << "统计: 一致=" << agreeCount << " 不一致=" << disagreeCount << "\n\n";

    if (disagreeCount > 0) {
        std::cout << "不一致的决策分析:\n";
        for (const auto& op : graph.getAllOps()) {
            auto hDec = heuristic.dispatch(op);
            auto pDec = pgd.dispatch(op);
            if (hDec.device != pDec.device) {
                const auto* prof = profiler.getProfile(op.name);
                std::cout << "  " << op.name << " (" << op.type
                          << ", FLOPs=" << formatFlops(op.estimatedFlops) << "):\n";
                std::cout << "    启发式: " << deviceToString(hDec.device)
                          << " (" << hDec.reason << ")\n";
                std::cout << "    PGO:    " << deviceToString(pDec.device)
                          << " (" << pDec.reason << ")\n";
                double hTime = (hDec.device == Device::CPU) ? prof->cpuTimeMs : prof->gpuTimeMs;
                double pTime = (pDec.device == Device::CPU) ? prof->cpuTimeMs : prof->gpuTimeMs;
                std::cout << "    实测: 启发式=" << std::fixed << std::setprecision(3)
                          << hTime << "ms, PGO=" << pTime << "ms";
                if (pTime < hTime) {
                    std::cout << " → PGO 更快 " << std::setprecision(1)
                              << (hTime / pTime) << "x\n";
                } else {
                    std::cout << " → 启发式更快 (PGO 判断可能受阈值影响)\n";
                }
                std::cout << "\n";
            }
        }
    }

    // ── Part 4: 性能对比 ──
    std::cout << "=== Part 4: 性能对比 ===\n\n";

    // 启发式调度下每个算子的耗时
    double heuristicCpuMs = 0.0, heuristicGpuMs = 0.0;
    for (const auto& op : graph.getAllOps()) {
        auto dec = heuristic.dispatch(op);
        const auto* prof = profiler.getProfile(op.name);
        double t = (dec.device == Device::CPU) ? prof->cpuTimeMs : prof->gpuTimeMs;
        if (dec.device == Device::CPU) heuristicCpuMs += t;
        else heuristicGpuMs += t;
    }

    // PGO 调度下每个算子的耗时
    double pgdCpuMs = 0.0, pgdGpuMs = 0.0;
    for (const auto& op : graph.getAllOps()) {
        auto dec = pgd.dispatch(op);
        const auto* prof = profiler.getProfile(op.name);
        double t = (dec.device == Device::CPU) ? prof->cpuTimeMs : prof->gpuTimeMs;
        if (dec.device == Device::CPU) pgdCpuMs += t;
        else pgdGpuMs += t;
    }

    std::cout << "  策略         CPU耗时(ms)  GPU耗时(ms)  总耗时(ms)  相对速度\n";
    std::cout << "  ─────────────────────────────────────────────────────\n";

    double baselineMs = heuristicTotalMs;
    auto printRow = [&](const std::string& name, double cpuMs, double gpuMs, double totalMs) {
        std::cout << "  " << std::left << std::setw(14) << name
                  << std::fixed << std::setprecision(3)
                  << std::setw(13) << cpuMs
                  << std::setw(13) << gpuMs
                  << std::setw(12) << totalMs
                  << std::setprecision(2) << (baselineMs / totalMs) << "x\n";
    };

    printRow("启发式", heuristicCpuMs, heuristicGpuMs, heuristicTotalMs);
    printRow("Profile-Guided", pgdCpuMs, pgdGpuMs, pgdTotalMs);

    std::cout << "\n";
    if (pgdTotalMs <= heuristicTotalMs) {
        std::cout << "  → Profile-Guided 调度" << std::fixed << std::setprecision(1)
                  << ((heuristicTotalMs - pgdTotalMs) / heuristicTotalMs * 100.0)
                  << "% 更快！\n";
        std::cout << "  原因: PGO 用真实测量数据替代猜测，能发现启发式规则覆盖不到的情况。\n";
    } else {
        std::cout << "  → 在本例中两种策略性能接近（启发式规则已较合理）。\n";
        std::cout << "  但在更复杂的硬件/模型组合下，PGO 通常更优。\n";
    }

    // ── Part 5: 导出 Profile JSON ──
    std::cout << "\n=== Part 5: Profile 数据导出 (JSON) ===\n\n";
    std::cout << "Profile 数据可以序列化为 JSON，供下次推理直接加载，\n";
    std::cout << "无需重新 profiling（对应框架 API: pgd.saveProfile() / loadProfile()）\n\n";
    std::cout << "JSON 格式:\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << pgd.profileToJson();
    std::cout << std::string(60, '-') << "\n";

    // ── 总结 ──
    std::cout << "\n==============================================================\n";
    std::cout << "  关键收获\n";
    std::cout << "==============================================================\n\n";
    std::cout << "1. PGO = Profile-Guided Optimization: 先测量，再决策\n";
    std::cout << "2. PGO 的核心数据: opName → {cpuTime, gpuTime, bestDevice}\n";
    std::cout << "3. PGO 需要 GPU 加速比阈值(默认1.2x): 抵消数据拷贝开销\n";
    std::cout << "4. 冷启动问题: 无 profile 数据时回退到启发式\n";
    std::cout << "5. PGO 总是 >= 启发式: 用真实数据替代猜测\n";
    std::cout << "6. Profile 数据需要定期更新: 硬件状态可能变化(温度墙等)\n";
    std::cout << "\n下一步: step5_hybrid_execution.cpp —— CPU+GPU 混合并行执行\n";

    return 0;
}
