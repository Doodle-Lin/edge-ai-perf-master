/**
 * @file PowerMonitor.cpp
 * @brief 功耗监控器的实现 —— 使用 PDH (Windows) 或 RAPL (Linux) 采样 CPU 功耗
 *
 * ====== 实现要点 ======
 * 1. Pimpl 惯用法的实现结构：
 *    - 头文件只声明 struct Impl 的前向声明和 unique_ptr
 *    - 本文件定义 struct PowerMonitor::Impl 的完整内容
 *    - 构造/析构在 .cpp 中完成，避免头文件依赖平台 API
 *
 * 2. Windows PDH API：
 *    - PdhOpenQuery()  创建查询句柄
 *    - PdhAddCounter() 添加计数器路径，如 "\\Processor Information(_Total)\\Processor Frequency"
 *    - PdhCollectQueryData() 采样数据
 *    - 注意：PDH 没有直接提供 CPU 功耗计数器，实际 CPU 功耗通常需要通过
 *      Intel RAPL 接口或硬件寄存器读取。本实现使用估算方法。
 *
 * 3. Linux RAPL (Running Average Power Limit)：
 *    - 路径: /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
 *    - 读取两次 energy_uj 的差值 / 时间差 = 平均功耗
 *    - 需要 root 权限或用户在 powercap 组中
 *
 * 4. GPU 功耗：
 *    - 真实 GPU 功耗需要 NVIDIA NVML 库（nvidia-smi 的底层库）
 *    - NVML API: nvmlInit(), nvmlDeviceGetPowerUsage(), nvmlShutdown()
 *    - 为学习目的，本实现中 GPU 功耗为 stub（返回 0）
 *    - 可通过读取 /sys/class/drm/card0/device/hwmon/hwmon*/power1_input
 *      获取部分 GPU 的功耗（AMD/Intel 核显）
 *
 * 5. 采样线程实现：
 *    - 使用 std::thread 创建后台线程
 *    - std::atomic<bool> 作为停止标志
 *    - 采样间隔 ~100ms，使用 std::this_thread::sleep_for
 *    - 析构时确保线程被 join，避免悬空线程
 */

#include "profiler/PowerMonitor.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <vector>

// ──────────────── 平台头文件 ────────────────
#ifdef PLATFORM_WINDOWS
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <pdh.h>
    #include <pdhmsg.h>
    #pragma comment(lib, "pdh.lib")
#elif defined(PLATFORM_LINUX)
    #include <cstdio>
    #include <dirent.h>
    #include <cstring>
#endif

namespace edgeai {

// ═══════════════════════════════════════════════════════════
// PowerMonitor::Impl —— 平台相关实现细节
// ═══════════════════════════════════════════════════════════

struct PowerMonitor::Impl {
    std::vector<PowerSample> samples;   ///< 采样数据缓冲区
    std::atomic<bool> running{false};   ///< 采样线程运行标志
    std::thread samplingThread;         ///< 后台采样线程

    // 采样起始时间点（用于计算 timestamp）
    std::chrono::steady_clock::time_point startTime;

    // ──── 平台相关数据 ────
#ifdef PLATFORM_WINDOWS
    // PDH 查询句柄，用于读取 Windows 性能计数器
    // 注意：PDH 没有直接的 CPU 功耗计数器，这里用于读取 CPU 利用率作为估算基础
    PDH_HQUERY cpuQuery        = nullptr;
    PDH_HCOUNTER cpuCounter    = nullptr;  // CPU 利用率计数器
    bool pdhAvailable          = false;     // PDH 是否初始化成功
    double tdpEstimateW        = 65.0;     // CPU TDP 估算值（瓦），用于从利用率估算功耗
    // TDP (Thermal Design Power) = 热设计功耗，是 CPU 的最大持续功耗
    // 常见值：笔记本 15-45W，台式机 65-125W，服务器 150-250W
#elif defined(PLATFORM_LINUX)
    // RAPL 能量文件路径
    std::string raplPath;        // 如 "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
    bool raplAvailable = false;  // RAPL 是否可用
    uint64_t lastEnergyUj = 0;   // 上一次读取的能量值（微焦耳）
    std::chrono::steady_clock::time_point lastSampleTime;  // 上一次采样时间
    double tdpEstimateW = 65.0;  // CPU TDP 估算值（瓦），RAPL 不可用时用作回退
#endif

    /// 读取当前 CPU 功耗（瓦），返回 0 表示无法获取
    double readCpuPower();

    /// 读取当前 GPU 功耗（瓦），返回 0 表示无法获取
    /// 注意：真实 GPU 功耗需要 NVIDIA NVML 库
    double readGpuPower();

    /// 采样线程的主循环
    void samplingLoop();

    /// 初始化平台相关的功耗读取接口
    bool initPlatform();

    /// 清理平台资源
    void cleanupPlatform();
};

// ═══════════════════════════════════════════════════════════
// 平台初始化 / 清理
// ═══════════════════════════════════════════════════════════

bool PowerMonitor::Impl::initPlatform()
{
#ifdef PLATFORM_WINDOWS
    // ──── Windows: 使用 PDH API 读取 CPU 利用率 ────
    //
    // 学习要点：PDH (Performance Data Helper) 是 Windows 性能监控的核心 API
    // 计数器路径格式：\\ObjectName(Instance)\\CounterName
    // 常用计数器：
    //   \\Processor(_Total)\\% Processor Time    → CPU 利用率
    //   \\Memory\\Available MBytes               → 可用内存
    //   \\PhysicalDisk(_Total)\\Disk Read Bytes/sec  → 磁盘读取速率
    //
    // 注意：PDH 没有 CPU 功耗计数器！真实 CPU 功耗需通过：
    //   1. Intel RAPL MSR 寄存器（需要内核驱动，如 WinRing0）
    //   2. IPMI（服务器平台）
    //   3. OEM 提供的 WMI 接口
    // 这里我们使用 CPU 利用率 × TDP 来估算功耗

    PDH_STATUS status = PdhOpenQuery(nullptr, 0, &cpuQuery);
    if (status != ERROR_SUCCESS) {
        std::fprintf(stderr, "[PowerMonitor] PdhOpenQuery failed: 0x%08X\n", status);
        pdhAvailable = false;
        return false;
    }

    // 添加 CPU 利用率计数器
    // \\Processor Information(_Total)\\% Processor Time 比 \\Processor 更精确
    // 它能区分物理核心与逻辑核心
    status = PdhAddCounter(cpuQuery,
        L"\\Processor Information(_Total)\\% Processor Time",
        0, &cpuCounter);

    if (status != ERROR_SUCCESS) {
        std::fprintf(stderr, "[PowerMonitor] PdhAddCounter failed: 0x%08X\n", status);
        PdhCloseQuery(cpuQuery);
        cpuQuery = nullptr;
        pdhAvailable = false;
        return false;
    }

    // 首次采集数据（PDH 需要至少两次采集才能计算增量值）
    PdhCollectQueryData();

    pdhAvailable = true;
    std::printf("[PowerMonitor] Windows PDH initialized (CPU utilization → power estimation)\n");
    return true;

#elif defined(PLATFORM_LINUX)
    // ──── Linux: 尝试读取 RAPL 接口 ────
    //
    // RAPL (Running Average Power Limit) 是 Intel 从 Sandy Bridge 起引入的功耗监控接口
    // 它通过 MSR 寄存器和 sysfs 接口暴露功耗数据
    //
    // sysfs 路径结构：
    //   /sys/class/powercap/intel-rapl/
    //   ├── intel-rapl:0/           ← Package 0 (整个 CPU 封装)
    //   │   ├── energy_uj           ← 累计能量消耗（微焦耳）
    //   │   ├── max_energy_range_uj ← 能量计数器最大值
    //   │   ├── constraint_0_name    ← 功耗限制名称 (long_term/short_term)
    //   │   ├── constraint_0_power_limit_uw  ← 功耗限制（微瓦）
    //   │   └── intel-rapl:0:0/     ← Core 域（CPU 核心）
    //   │       └── intel-rapl:0:1/ ← Uncore 域（L3缓存、内存控制器等）
    //   └── intel-rapl:1/           ← Package 1（多路服务器）
    //
    // 计算功耗的方法：
    //   1. 读取 energy_uj 得到 E1
    //   2. 等待 dt 秒
    //   3. 读取 energy_uj 得到 E2
    //   4. 功耗 = (E2 - E1) / dt / 1e6 瓦
    //
    // 注意事项：
    //   - energy_uj 会溢出回绕（32位计数器，最大约 4.3 TJ）
    //   - 需要处理回绕：if (E2 < E1) E2 += max_energy_range
    //   - 需要 root 权限或将用户加入 powercap 组

    const char* basePath = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj";

    // 尝试打开 RAPL 能量文件
    std::ifstream ifs(basePath);
    if (ifs.is_open()) {
        raplPath = basePath;
        raplAvailable = true;
        ifs.close();
        std::printf("[PowerMonitor] Linux RAPL initialized: %s\n", basePath);
        return true;
    }

    // 回退：尝试搜索其他 RAPL 路径
    // 某些系统上路径可能不同（如多路服务器有 rapl:1, rapl:2 等）
    const char* raplDir = "/sys/class/powercap/";
    // 简化实现：直接使用默认路径
    raplAvailable = false;
    std::fprintf(stderr, "[PowerMonitor] RAPL not available at %s, using TDP estimation\n", basePath);
    return false;

#else
    // 未知平台：使用 TDP 估算
    std::printf("[PowerMonitor] Unknown platform, using TDP-based power estimation\n");
    return false;
#endif
}

void PowerMonitor::Impl::cleanupPlatform()
{
#ifdef PLATFORM_WINDOWS
    if (cpuQuery) {
        PdhCloseQuery(cpuQuery);
        cpuQuery = nullptr;
    }
    pdhAvailable = false;
#endif
    // Linux: 无需清理文件句柄（每次采样都重新打开读取）
}

// ═══════════════════════════════════════════════════════════
// 功耗读取
// ═══════════════════════════════════════════════════════════

double PowerMonitor::Impl::readCpuPower()
{
#ifdef PLATFORM_WINDOWS
    // ──── Windows: 从 CPU 利用率估算功耗 ────
    //
    // 估算公式：P = P_idle + (P_tdp - P_idle) × utilization
    //   - P_idle: CPU 空闲功耗（通常约 5-15W）
    //   - P_tdp:  CPU 满载功耗（TDP 值）
    //   - utilization: CPU 利用率 [0, 1]
    //
    // 这是一个粗略估算！实际功耗还受频率调节（DVFS）、AVX 指令功耗等影响
    // AVX2/AVX-512 指令会使 CPU 功耗比同利用率下高 10-30%

    if (!pdhAvailable || !cpuQuery) {
        return 0.0;
    }

    // 采集当前数据
    PDH_STATUS status = PdhCollectQueryData();
    if (status != ERROR_SUCCESS) {
        return 0.0;
    }

    // 读取 CPU 利用率（PDH 返回的是 0-100 的百分比值）
    PDH_FMT_COUNTERVALUE fmtValue;
    status = PdhGetFormattedCounterValue(cpuCounter, PDH_FMT_DOUBLE, nullptr, &fmtValue);
    if (status != ERROR_SUCCESS) {
        return 0.0;
    }

    double utilization = fmtValue.doubleValue / 100.0;  // 转为 [0, 1]
    if (utilization < 0.0) utilization = 0.0;
    if (utilization > 1.0) utilization = 1.0;

    // 简化估算：空闲功耗 ≈ TDP × 15%，满载功耗 ≈ TDP
    double idlePower = tdpEstimateW * 0.15;
    double cpuPower = idlePower + (tdpEstimateW - idlePower) * utilization;

    return cpuPower;

#elif defined(PLATFORM_LINUX)
    // ──── Linux: 读取 RAPL 能量计数器 ────
    if (raplAvailable && !raplPath.empty()) {
        std::ifstream ifs(raplPath);
        if (ifs.is_open()) {
            uint64_t energyUj = 0;
            ifs >> energyUj;
            ifs.close();

            auto now = std::chrono::steady_clock::now();

            if (lastEnergyUj > 0) {
                // 计算两次采样间的能量差和时间差
                auto dtMs = std::chrono::duration<double, std::milli>(now - lastSampleTime).count();
                double dtSec = dtMs / 1000.0;

                if (dtSec > 0.0) {
                    uint64_t deltaEnergy = 0;
                    // 处理计数器溢出回绕
                    if (energyUj >= lastEnergyUj) {
                        deltaEnergy = energyUj - lastEnergyUj;
                    } else {
                        // 计数器溢出回绕——最大值约 4.3 TJ (2^32 微焦耳)
                        // 简化处理：假设只溢出一次
                        deltaEnergy = (UINT32_MAX - lastEnergyUj) + energyUj;
                    }

                    // 功率 = 能量差 / 时间差（微焦耳 → 瓦）
                    // 1 W = 1 J/s = 1e6 uJ/s
                    double powerW = static_cast<double>(deltaEnergy) / dtSec / 1e6;
                    lastEnergyUj = energyUj;
                    lastSampleTime = now;
                    return powerW;
                }
            }

            // 首次采样：只记录值，不计算功耗
            lastEnergyUj = energyUj;
            lastSampleTime = now;
            return 0.0;
        }
    }

    // RAPL 不可用：回退到 TDP 估算
    // 使用 /proc/stat 读取 CPU 利用率（简化实现，返回 TDP 的一半作为估算）
    // 生产环境应读取 /proc/stat 计算 CPU 利用率
    return tdpEstimateW * 0.5;

#else
    return tdpEstimateW * 0.5;
#endif
}

double PowerMonitor::Impl::readGpuPower()
{
    // ──── GPU 功耗读取 ────
    //
    // 真实 GPU 功耗需要 NVIDIA NVML 库，步骤如下：
    //
    //   #include <nvml.h>
    //   nvmlInit();
    //   nvmlDevice_t device;
    //   nvmlDeviceGetHandleByIndex(0, &device);
    //   unsigned int power;  // 毫瓦
    //   nvmlDeviceGetPowerUsage(device, &power);
    //   double gpuPowerW = power / 1000.0;
    //   nvmlShutdown();
    //
    // NVML 的优点：
    //   - 不需要 root 权限
    //   - 支持所有 NVIDIA GPU（GeForce, Quadro, Tesla）
    //   - 还能读取温度、频率、显存使用量等
    //
    // 本实现中，GPU 功耗为 stub，返回 0。
    // 作为替代方案，可以尝试读取 AMD/Intel GPU 的 sysfs 接口：
    //   /sys/class/drm/card0/device/hwmon/hwmon*/power1_input

    // 尝试 Linux AMD/Intel GPU sysfs 接口
#ifdef PLATFORM_LINUX
    // AMD GPU: /sys/class/drm/card0/device/hwmon/hwmon*/power1_input (微瓦)
    // Intel GPU: /sys/class/drm/card0/device/hwmon/hwmon*/power1_input (微瓦)
    // 简化实现：不搜索路径，直接返回 0
    // 完整实现需要遍历 /sys/class/drm/ 下的设备
#endif

    return 0.0;  // GPU 功耗未实现，返回 0
}

// ═══════════════════════════════════════════════════════════
// 采样线程
// ═══════════════════════════════════════════════════════════

void PowerMonitor::Impl::samplingLoop()
{
    // 采样线程主循环
    // 以 ~100ms 间隔采样，直到 running 被设为 false
    //
    // 学习要点：
    // 1. std::atomic<bool> 的 load() 是无锁操作，开销极小
    // 2. sleep_for 不是精确的 100ms——受系统调度器影响，实际间隔可能为 100-115ms
    // 3. 采样精度对功耗估算的影响：间隔越短越精确，但 CPU 开销越大
    //    100ms 间隔是常见的折中选择

    while (running.load()) {
        auto sampleStart = std::chrono::steady_clock::now();

        PowerSample sample;
        sample.timestamp = std::chrono::duration<double>(
            sampleStart - startTime).count();
        sample.cpuPowerW = readCpuPower();
        sample.gpuPowerW = readGpuPower();

        samples.push_back(sample);

        // 休眠 100ms
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// ═══════════════════════════════════════════════════════════
// PowerMonitor 公共接口
// ═══════════════════════════════════════════════════════════

PowerMonitor::PowerMonitor()
    : impl_(std::make_unique<Impl>())
{
}

PowerMonitor::~PowerMonitor()
{
    // 析构时确保采样线程已停止
    // 如果用户忘记调用 stop()，这里自动停止
    if (impl_ && impl_->running.load()) {
        stop();
    }
    if (impl_) {
        impl_->cleanupPlatform();
    }
}

bool PowerMonitor::start()
{
    if (impl_->running.load()) {
        std::fprintf(stderr, "[PowerMonitor] Already running\n");
        return false;
    }

    // 清空上次的采样数据
    impl_->samples.clear();

    // 初始化平台接口
    impl_->initPlatform();

    // 记录起始时间
    impl_->startTime = std::chrono::steady_clock::now();

    // Linux RAPL: 初始化首次采样时间
#ifdef PLATFORM_LINUX
    impl_->lastSampleTime = impl_->startTime;
#endif

    // 启动采样线程
    impl_->running.store(true);
    impl_->samplingThread = std::thread(&Impl::samplingLoop, impl_.get());

    std::printf("[PowerMonitor] Sampling started (~100ms interval)\n");
    return true;
}

bool PowerMonitor::stop()
{
    if (!impl_->running.load()) {
        return false;
    }

    // 设置停止标志
    impl_->running.store(false);

    // 等待采样线程结束
    // 必须调用 join()，否则线程会在析构时导致 std::terminate
    if (impl_->samplingThread.joinable()) {
        impl_->samplingThread.join();
    }

    std::printf("[PowerMonitor] Sampling stopped. Collected %zu samples.\n",
                impl_->samples.size());
    return true;
}

bool PowerMonitor::isRunning() const
{
    return impl_->running.load();
}

// ═══════════════════════════════════════════════════════════
// 统计结果
// ═══════════════════════════════════════════════════════════

double PowerMonitor::avgCpuPower() const
{
    if (impl_->samples.empty()) return 0.0;

    double sum = 0.0;
    for (const auto& s : impl_->samples) {
        sum += s.cpuPowerW;
    }
    return sum / static_cast<double>(impl_->samples.size());
}

double PowerMonitor::peakCpuPower() const
{
    if (impl_->samples.empty()) return 0.0;

    double peak = 0.0;
    for (const auto& s : impl_->samples) {
        if (s.cpuPowerW > peak) peak = s.cpuPowerW;
    }
    return peak;
}

double PowerMonitor::avgGpuPower() const
{
    if (impl_->samples.empty()) return 0.0;

    double sum = 0.0;
    int validCount = 0;
    for (const auto& s : impl_->samples) {
        if (s.gpuPowerW > 0.0) {
            sum += s.gpuPowerW;
            validCount++;
        }
    }
    return (validCount > 0) ? (sum / validCount) : 0.0;
}

double PowerMonitor::totalEnergyJ() const
{
    // 能耗 = 功率 × 时间的积分
    // 离散近似：E = Σ P_i × dt_i
    // 其中 dt_i = 相邻采样点的时间差
    //
    // 第一个采样点没有前一个点，跳过
    // 最后一个采样点的功耗延续到 stop() 时刻，这里简化处理

    if (impl_->samples.size() < 2) return 0.0;

    double energyJ = 0.0;
    for (size_t i = 1; i < impl_->samples.size(); ++i) {
        double dt = impl_->samples[i].timestamp - impl_->samples[i - 1].timestamp;
        double avgPower = (impl_->samples[i].cpuPowerW + impl_->samples[i - 1].cpuPowerW +
                           impl_->samples[i].gpuPowerW + impl_->samples[i - 1].gpuPowerW) / 2.0;
        energyJ += avgPower * dt;  // J = W × s
    }
    return energyJ;
}

const std::vector<PowerSample>& PowerMonitor::samples() const
{
    return impl_->samples;
}

// ═══════════════════════════════════════════════════════════
// 报告与序列化
// ═══════════════════════════════════════════════════════════

void PowerMonitor::printReport() const
{
    std::printf("\n");
    std::printf("╔══════════════════════════════════════════╗\n");
    std::printf("║       Power Monitor Report               ║\n");
    std::printf("╠══════════════════════════════════════════╣\n");

    std::printf("║ Samples Collected: %zu\n", impl_->samples.size());
    std::printf("║ Avg CPU Power    : %.2f W\n", avgCpuPower());
    std::printf("║ Peak CPU Power   : %.2f W\n", peakCpuPower());
    std::printf("║ Avg GPU Power    : %.2f W\n", avgGpuPower());
    std::printf("║ Total Energy     : %.3f J\n", totalEnergyJ());

    if (!impl_->samples.empty()) {
        double duration = impl_->samples.back().timestamp;
        std::printf("║ Duration         : %.2f s\n", duration);
        if (duration > 0.0) {
            double avgTotalPower = totalEnergyJ() / duration;
            std::printf("║ Avg Total Power  : %.2f W\n", avgTotalPower);
        }
    }

    std::printf("╚══════════════════════════════════════════╝\n");

    // 能效提示
    double energy = totalEnergyJ();
    if (energy > 0.0 && energy < 1.0) {
        std::printf("[PowerMonitor] Inference energy < 1J → Good for edge deployment!\n");
    } else if (energy >= 1.0 && energy < 10.0) {
        std::printf("[PowerMonitor] Inference energy 1-10J → Acceptable for edge, "
                    "consider model quantization.\n");
    } else if (energy >= 10.0) {
        std::printf("[PowerMonitor] Inference energy > 10J → High energy, "
                    "consider model pruning, quantization, or distillation.\n");
    }

    // GPU 功耗未获取提示
    if (avgGpuPower() <= 0.0) {
        std::printf("[PowerMonitor] Note: GPU power not available. "
                    "For NVIDIA GPUs, link against NVML library.\n");
    }

    std::printf("\n");
}

std::string PowerMonitor::toJson() const
{
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"avgCpuPowerW\": "    << std::fixed << std::setprecision(2) << avgCpuPower()  << ",\n";
    oss << "  \"peakCpuPowerW\": "   << std::fixed << std::setprecision(2) << peakCpuPower() << ",\n";
    oss << "  \"avgGpuPowerW\": "    << std::fixed << std::setprecision(2) << avgGpuPower()  << ",\n";
    oss << "  \"totalEnergyJ\": "     << std::fixed << std::setprecision(4) << totalEnergyJ() << ",\n";
    oss << "  \"sampleCount\": "      << impl_->samples.size() << ",\n";

    oss << "  \"samples\": [\n";
    for (size_t i = 0; i < impl_->samples.size(); ++i) {
        const auto& s = impl_->samples[i];
        oss << "    {\"t\":"     << std::fixed << std::setprecision(3) << s.timestamp
            << ", \"cpuW\":"   << std::fixed << std::setprecision(2) << s.cpuPowerW
            << ", \"gpuW\":"   << std::fixed << std::setprecision(2) << s.gpuPowerW
            << "}";
        if (i + 1 < impl_->samples.size()) oss << ",";
        oss << "\n";
    }
    oss << "  ]\n";
    oss << "}\n";
    return oss.str();
}

} // namespace edgeai
