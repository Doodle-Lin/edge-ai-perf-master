/**
 * @file PowerMonitor.h
 * @brief 功耗监控器 —— 采样 CPU/GPU 功耗，计算总能耗与能效比
 *
 * ====== 学习要点 ======
 * 1. 功耗 (Power) vs 能耗 (Energy)：
 *    - 功耗 = 瞬时功率（瓦特 W），如 CPU 当前消耗 15W
 *    - 能耗 = 功率 × 时间的积分（焦耳 J），如推理总能耗 0.5J
 *    - 1 J = 1 W × 1 s；1 度电 = 1 kWh = 3.6 MJ
 *
 * 2. 能效比 (Energy Efficiency)：
 *    - 推理能效 = 推理数/秒 / 功耗(W) = inferences/(s·W)
 *    - 边缘设备追求 "每瓦性能"，而非纯粹峰值算力
 *
 * 3. 平台差异：
 *    - Windows: 可用 PDH (Performance Data Helper) API 读取处理器计数器
 *    - Linux:   可读 /sys/class/powercap/intel-rapl/ 获取 RAPL 功耗数据
 *    - GPU:     需要 NVIDIA NVML 库（nvidia-smi 的底层库）
 *    - 本实现采用 Pimpl 惯用法隔离平台差异
 *
 * 4. Pimpl 惯用法 (Pointer to Implementation)：
 *    - 头文件只声明 std::unique_ptr<Impl>，不暴露平台相关头文件
 *    - 实现文件中定义 struct Impl，包含 PDH 句柄或文件路径等
 *    - 优点：减少头文件依赖、加快编译、稳定 ABI
 *
 * 5. 采样线程：
 *    - 以 ~100ms 间隔在后台采样功耗，避免阻塞主线程
 *    - 使用 std::atomic<bool> 作为停止标志，保证线程安全
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace edgeai {

/// 单个时间点的功耗采样
struct PowerSample {
    double timestamp  = 0.0;  ///< 从 start() 起的秒数
    double cpuPowerW  = 0.0;  ///< CPU 瞬时功耗（瓦特）
    double gpuPowerW  = 0.0;  ///< GPU 瞬时功耗（瓦特），0 表示未获取
};

/**
 * @class PowerMonitor
 * @brief 后台功耗采样监控器
 *
 * 使用方法：
 * @code
 *   PowerMonitor monitor;
 *   monitor.start();            // 启动后台采样线程
 *
 *   // ... 执行推理 ...
 *
 *   monitor.stop();             // 停止采样
 *   monitor.printReport();      // 打印功耗报告
 *
 *   double energyJ = monitor.totalEnergyJ();  // 总能耗（焦耳）
 *   double avgW    = monitor.avgCpuPower();    // 平均 CPU 功耗
 * @endcode
 *
 * 实现说明：
 * - CPU 功耗：Windows 下读取 PDH 计数器或估算；Linux 下读取 RAPL
 * - GPU 功耗：需要 NVML，本实现中为 stub（返回 0 或从 /sys 读取）
 * - 采样间隔约 100ms，受系统调度精度影响
 */
class PowerMonitor {
public:
    PowerMonitor();
    ~PowerMonitor();

    // 禁止拷贝和移动 —— 内含线程和句柄，不可转移
    PowerMonitor(const PowerMonitor&) = delete;
    PowerMonitor& operator=(const PowerMonitor&) = delete;
    PowerMonitor(PowerMonitor&&) = delete;
    PowerMonitor& operator=(PowerMonitor&&) = delete;

    /// 启动后台功耗采样线程
    /// @return true 启动成功
    bool start();

    /// 停止采样线程并汇总结果
    /// @return true 停止成功
    bool stop();

    /// 采样线程是否正在运行
    bool isRunning() const;

    // ──────────────── 统计结果 ────────────────

    /// CPU 平均功耗（瓦特）
    double avgCpuPower() const;

    /// CPU 峰值功耗（瓦特）
    double peakCpuPower() const;

    /// GPU 平均功耗（瓦特），0 表示未监控
    double avgGpuPower() const;

    /// 总能耗（焦耳）= Σ(sample.cpuPowerW + sample.gpuPowerW) × dt
    /// 其中 dt ≈ 0.1s 为采样间隔
    double totalEnergyJ() const;

    /// 获取所有采样点（只读）
    const std::vector<PowerSample>& samples() const;

    /// 打印功耗报告到 stdout
    void printReport() const;

    /// 序列化为 JSON 字符串
    std::string toJson() const;

private:
    /// Pimpl：将平台相关细节隐藏在实现文件中
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace edgeai
