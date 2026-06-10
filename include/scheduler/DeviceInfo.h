/**
 * @file DeviceInfo.h
 * @brief 设备能力查询 —— 运行时探测 CPU / GPU 硬件特性，为调度决策提供依据
 *
 * ====== 学习要点 ======
 *
 * 1. 为什么需要运行时探测？
 *    边缘设备的硬件差异极大：笔记本可能只有 2 核 CPU + 集显，
 *    工控机可能有 8 核 + 独立 GPU。调度器必须知道"手里有什么牌"，
 *    才能做出合理的算子分配。
 *
 * 2. CPUID 指令（x86 特有）
 *    CPUID 是 x86 处理器提供的一条"自我介绍"指令：
 *    - EAX=1 → 返回家族型号、特性标志（EDX/ECX 各位代表不同 SIMD 扩展）
 *    - EAX=7, ECX=0 → 返回 AVX2、AVX-512 等扩展特性
 *    - EAX=0x80000000/1/6 → 返回缓存拓扑、L1/L2/L3 大小
 *    编译器提供了 __cpuid() / __cpuidex() 内建函数来替代手写汇编。
 *
 * 3. 缓存层次对性能的影响
 *    L1 (~32KB) → L2 (~256KB) → L3 (~数MB) → 主存 (~数GB)
 *    每升一级，延迟增加约 3-10 倍。调度器应尽量让数据留在缓存里：
 *    - 小张量 → 放 CPU（L1/L2 命中率高）
 *    - 大张量 → 放 GPU（带宽优势明显）
 *
 * 4. Vulkan 作为 GPU 查询接口
 *    Vulkan 是跨平台图形/计算 API，通过 vkEnumeratePhysicalDevices
 *    可以拿到 GPU 名称、显存大小、计算单元数等信息。
 *    相比 CUDA/OpenCL，Vulkan 在移动端和嵌入式平台的覆盖更广。
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace edgeai {

// ============================================================================
// CPU 信息结构体
// ============================================================================

/// CPU 硬件能力描述
struct CpuInfo {
    int numCores    = 0;   ///< 物理核心数（不含超线程）
    int numThreads  = 0;   ///< 逻辑线程数（含超线程），如 8C/16T → 16
    bool hasAvx2    = false; ///< 是否支持 AVX2（256-bit SIMD，float×8 / int8×32）
    bool hasFma     = false; ///< 是否支持 FMA3（融合乘加，一条指令完成 a*b+c）
    bool hasAvx512  = false; ///< 是否支持 AVX-512（512-bit SIMD，float×16）
    size_t l1CacheSize = 0; ///< L1 数据缓存大小（字节），通常每核 32KB
    size_t l2CacheSize = 0; ///< L2 缓存大小（字节），通常每核 256KB~1MB
    size_t l3CacheSize = 0; ///< L3 共享缓存大小（字节），通常 4~32MB
    std::string modelName;   ///< CPU 型号名称，如 "Intel(R) Core(TM) i7-10700"
};

// ============================================================================
// GPU 信息结构体
// ============================================================================

/// GPU 硬件能力描述
struct GpuInfo {
    std::string deviceName;       ///< 设备名称，如 "NVIDIA GeForce RTX 3060"
    uint32_t vendorId       = 0;  ///< 厂商 ID：0x10DE=NVIDIA, 0x8086=Intel, 0x1002=AMD
    size_t globalMemoryBytes = 0;  ///< 全局显存大小（字节），决定了能加载多大的模型
    uint32_t computeUnits   = 0;  ///< 计算单元数量（CUDA 叫 SM，Vulkan 叫 CU）
    uint32_t maxWorkGroupSize = 0; ///< 单个工作组最大线程数（通常 256~1024）
    bool supportsVulkan     = false; ///< 是否支持 Vulkan compute（1.1+）
};

// ============================================================================
// DeviceInfo 查询类
// ============================================================================

/**
 * @class DeviceInfo
 * @brief 静态工具类，提供 CPU/GPU 硬件信息查询
 *
 * 典型用法：
 * @code
 *   auto cpu = DeviceInfo::queryCpu();
 *   auto gpu = DeviceInfo::queryGpu();
 *   DeviceInfo::printSummary();
 * @endcode
 *
 * 设计为纯静态方法，无需实例化 —— 因为硬件信息在程序运行期间不变。
 */
class DeviceInfo {
public:
    /// 查询 CPU 信息（使用 CPUID 指令）
    /// @return 填充好的 CpuInfo 结构体
    /// @note 在非 x86 平台上会返回合理的默认值
    static CpuInfo queryCpu();

    /// 查询 GPU 信息（通过 Vulkan API）
    /// @return 填充好的 GpuInfo 结构体
    /// @note 如果 Vulkan 不可用，返回空的 GpuInfo（supportsVulkan = false）
    static GpuInfo queryGpu();

    /// 打印设备信息摘要到 stdout
    /// 格式化输出 CPU 和 GPU 的关键参数，方便调试
    static void printSummary();
};

} // namespace edgeai
