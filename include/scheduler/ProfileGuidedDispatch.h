/**
 * @file ProfileGuidedDispatch.h
 * @brief 基于 Profiling 数据的调度策略 —— 用历史性能数据指导未来的调度决策
 *
 * ====== 学习要点 ======
 *
 * 1. 什么是 Profile-Guided 优化？
 *    传统编译器/调度器靠启发式规则猜测最优方案。
 *    Profile-Guided 则是"先跑一遍，看看实际表现，再用数据说话"。
 *    类比：启发式像"看菜单点菜"，PGO 像"先试吃再点菜"。
 *
 *    典型流程：
 *    a) 第一轮：用 HeuristicDispatch 跑推理，同时用 OpProfiler 记录每个算子的
 *       CPU 和 GPU 执行时间
 *    b) 分析：对比同一算子在 CPU 和 GPU 上的耗时，选择更快的设备
 *    c) 保存：将 "opName → bestDevice" 的映射表序列化为 JSON
 *    d) 后续：加载映射表，直接查表分配，无需再猜
 *
 * 2. 为什么 PGO 比启发式更准？
 *    - 启发式不知道特定硬件的实际性能（如某个 GPU 跑小卷积反而比 CPU 快）
 *    - 启发式不知道数据布局对性能的影响（如 NCHW vs NHWC）
 *    - 启发式不知道缓存预热效果（同一个算子连续执行比冷启动快得多）
 *    PGO 用真实测量数据替代猜测，通常能提升 10~30% 的端到端性能。
 *
 * 3. 冷启动问题
 *    第一次运行时没有 profiling 数据，怎么办？
 *    答案：回退到 HeuristicDispatch（Fallback 策略）。
 *    这是"渐进优化"的思路——先保证能跑，再越跑越快。
 *
 * 4. Profile 数据的保鲜期
 *    硬件状态可能变化：温度墙（thermal throttling）、功耗策略切换、
 *    后台进程抢占资源。因此 profile 数据应该定期更新。
 *    本实现中，用户可以手动调用 updateProfile() 刷新数据。
 *
 * 5. JSON 格式设计
 *    @code
 *    {
 *      "version": 1,
 *      "device_info": { "cpu": "Intel i7-10700", "gpu": "RTX 3060" },
 *      "op_profiles": {
 *        "conv1": { "cpu_ms": 2.3, "gpu_ms": 0.8, "best": "GPU" },
 *        "relu1": { "cpu_ms": 0.1, "gpu_ms": 0.5, "best": "CPU" }
 *      }
 *    }
 *    @endcode
 *    包含版本号方便未来格式升级，包含设备信息防止误用在错误硬件上。
 */

#pragma once

#include "scheduler/DispatchStrategy.h"
#include "profiler/OpProfiler.h"
#include <string>
#include <unordered_map>
#include <memory>

namespace edgeai {

// ============================================================================
// 单个算子的 Profile 数据
// ============================================================================

/// 算子在某个设备上的性能记录
struct OpProfileEntry {
    double cpuTimeMs = 0.0;   ///< CPU 执行耗时（毫秒），0 表示未测量
    double gpuTimeMs = 0.0;   ///< GPU 执行耗时（毫秒），0 表示未测量
    Device bestDevice = Device::CPU;  ///< 实测更快的设备
    int sampleCount  = 0;     ///< 采样次数（次数越多，数据越可靠）

    /// 计算加速比 = cpuTimeMs / gpuTimeMs
    /// > 1.0 表示 GPU 更快，< 1.0 表示 CPU 更快
    double speedup() const {
        if (gpuTimeMs <= 0.0) return 0.0;
        return cpuTimeMs / gpuTimeMs;
    }
};

// ============================================================================
// Profile-Guided 调度策略
// ============================================================================

/**
 * @class ProfileGuidedDispatch
 * @brief 基于 Profiling 数据的调度策略
 *
 * 工作流程：
 * @code
 *   // 1. 创建策略，设置回退策略
 *   ProfileGuidedDispatch pgd;
 *   pgd.setFallback(std::make_shared<HeuristicDispatch>());
 *
 *   // 2. 从 JSON 文件加载历史 profile 数据（可选）
 *   pgd.loadProfile("op_profile.json");
 *
 *   // 3. 用于调度
 *   auto decision = pgd.dispatch(op, graph);
 *
 *   // 4. 执行后，用 OpProfiler 更新 profile 数据
 *   pgd.updateFromProfiler(profiler);
 *
 *   // 5. 保存 profile 数据供下次使用
 *   pgd.saveProfile("op_profile.json");
 * @endcode
 */
class ProfileGuidedDispatch : public DispatchStrategy {
public:
    ProfileGuidedDispatch();

    // ──────────────── 核心接口 ────────────────

    /**
     * @brief 根据历史 Profile 数据决定算子执行设备
     * @param op    待调度的算子
     * @param graph 算子图
     * @return 调度决策
     *
     * 决策流程：
     * 1. 在 profile 表中查找 op.name
     * 2. 如果找到且有 CPU 和 GPU 数据，选择更快的设备
     * 3. 如果只有单设备数据，返回该设备（总比没有强）
     * 4. 如果未找到，回退到 fallback 策略
     */
    DispatchDecision dispatch(const OpNode& op, const OpGraph& graph) override;

    /// 返回策略名称
    std::string name() const override { return "ProfileGuidedDispatch"; }

    // ──────────────── 配置接口 ────────────────

    /**
     * @brief 设置回退策略（用于未见过的算子）
     * @param fallback 回退策略的共享指针
     *
     * 推荐使用 HeuristicDispatch 作为回退，
     * 因为它不需要额外数据，冷启动即可工作。
     */
    void setFallback(std::shared_ptr<DispatchStrategy> fallback);

    /**
     * @brief 设置加速比阈值
     * @param threshold 只有 GPU 比 CPU 快超过此倍数时才选 GPU（默认 1.2）
     *
     * 为什么不直接比较时间？因为 GPU 执行需要 CPU→GPU 数据拷贝，
     * 仅计算核心快不够，必须快到能抵消拷贝开销才值得。
     * 1.2 倍的"安全裕量"是工程经验值。
     */
    void setGpuSpeedupThreshold(double threshold);

    // ──────────────── 数据管理 ────────────────

    /**
     * @brief 从 OpProfiler 的记录更新 profile 表
     * @param profiler 包含最近一次推理的性能数据
     *
     * 遍历 profiler 中的所有 OpRecord，按算子名聚合 CPU/GPU 耗时，
     * 取多次采样的平均值，并更新 bestDevice。
     */
    void updateFromProfiler(const OpProfiler& profiler);

    /**
     * @brief 手动添加一条 profile 记录
     * @param opName  算子名称
     * @param device  执行设备
     * @param timeMs  执行耗时
     *
     * 用于测试或从外部数据源导入。
     */
    void addProfileEntry(const std::string& opName, Device device, double timeMs);

    /// 获取某个算子的 profile 数据（可能返回 nullptr）
    const OpProfileEntry* getProfile(const std::string& opName) const;

    /// 获取所有 profile 数据（只读）
    const std::unordered_map<std::string, OpProfileEntry>& allProfiles() const;

    /// 清空所有 profile 数据
    void clearProfile();

    // ──────────────── 持久化 ────────────────

    /**
     * @brief 从 JSON 文件加载 profile 数据
     * @param path JSON 文件路径
     * @return true 加载成功
     *
     * JSON 格式见文件头部注释。如果文件不存在或格式错误，返回 false。
     */
    bool loadProfile(const std::string& path);

    /**
     * @brief 保存 profile 数据到 JSON 文件
     * @param path 输出文件路径
     * @return true 保存成功
     */
    bool saveProfile(const std::string& path) const;

    /**
     * @brief 将 profile 数据序列化为 JSON 字符串
     * @return JSON 格式的 profile 数据
     */
    std::string profileToJson() const;

private:
    std::unordered_map<std::string, OpProfileEntry> profileTable_; ///< 算子名 → profile 数据
    std::shared_ptr<DispatchStrategy> fallback_;                   ///< 回退策略
    double gpuSpeedupThreshold_ = 1.2;  ///< GPU 加速比阈值
};

} // namespace edgeai
