/**
 * @file DeviceInfo.cpp
 * @brief 设备能力查询的实现 —— 使用 CPUID 查询 CPU，Vulkan 查询 GPU
 *
 * ====== 实现说明 ======
 *
 * CPU 查询（x86 平台）：
 * - 使用编译器内建函数 __cpuid() / __cpuidex() 读取处理器信息
 * - MSVC 和 GCC/Clang 的内建函数签名不同，需要 #ifdef 区分
 * - 缓存大小通过 CPUID leaf 0x80000000/0x80000006 获取（AMD 方式）
 *   或 leaf 0x04 获取（Intel 方式）
 *
 * GPU 查询（Vulkan）：
 * - 通过 vkEnumeratePhysicalDevices 获取物理设备列表
 * - 通过 vkGetPhysicalDeviceProperties 获取设备名称和厂商 ID
 * - 通过 vkGetPhysicalDeviceMemoryProperties 获取显存大小
 * - 所有 Vulkan 调用包裹在 #ifdef ENABLE_VULKAN 中
 * - 如果 Vulkan 不可用，返回空的 GpuInfo
 */

#include "scheduler/DeviceInfo.h"

#include <iostream>
#include <sstream>
#include <cstring>
#include <thread>

#ifdef ENABLE_VULKAN
    #include <vulkan/vulkan.h>
#endif

// ============================================================================
// x86 CPUID 查询
// ============================================================================

#if defined(_MSC_VER)
    // MSVC 提供的 CPUID 内建函数，位于 <intrin.h>
    #include <intrin.h>
    #define CPUID(regs, leaf)        __cpuid(regs, leaf)
    #define CPUIDEX(regs, leaf, sub) __cpuidex(regs, leaf, sub)
#elif defined(__GNUC__) || defined(__clang__)
    // GCC/Clang 使用内联汇编封装
    #include <cpuid.h>
    static inline void CPUID(int regs[4], int leaf) {
        __cpuid(leaf, regs[0], regs[1], regs[2], regs[3]);
    }
    static inline void CPUIDEX(int regs[4], int leaf, int sub) {
        __cpuid_count(leaf, sub, regs[0], regs[1], regs[2], regs[3]);
    }
#else
    // 非 x86 平台或未知编译器，使用空实现
    #define CPUID(regs, leaf)   do { (void)(leaf); regs[0]=regs[1]=regs[2]=regs[3]=0; } while(0)
    #define CPUIDEX(regs, leaf, sub) do { (void)(leaf); (void)(sub); regs[0]=regs[1]=regs[2]=regs[3]=0; } while(0)
#endif

namespace edgeai {

// ============================================================================
// CPU 查询实现
// ============================================================================

CpuInfo DeviceInfo::queryCpu() {
    CpuInfo info;

    // ── 1. 查询逻辑线程数 ──
    // std::thread::hardware_concurrency() 是跨平台的标准方法
    // 返回值等于操作系统的逻辑 CPU 数（含超线程）
    info.numThreads = static_cast<int>(std::thread::hardware_concurrency());

    // ── 2. CPUID Leaf 0: 获取最大支持的 leaf 值和厂商字符串 ──
    int regs[4] = {};
    CPUID(regs, 0);
    int maxLeaf = regs[0];  // EAX = 支持的最大标准 leaf 值

    // ── 3. CPUID Leaf 1: 特性标志 ──
    // ECX 和 EDX 的各个位表示不同的 CPU 特性
    if (maxLeaf >= 1) {
        CPUID(regs, 1);
        // ECX bit 28 = AVX 支持
        // 但 AVX 不等于 AVX2，AVX2 还需要查 Leaf 7
        // EDX bit 0~3 包含家族型号信息
    }

    // ── 4. CPUID Leaf 7, Sub-leaf 0: 扩展特性 ──
    // 这里才有 AVX2 和 AVX-512 的标志位
    if (maxLeaf >= 7) {
        CPUIDEX(regs, 7, 0);
        // EBX bit 5 = AVX2 支持
        info.hasAvx2 = (regs[1] & (1 << 5)) != 0;
        // EBX bit 3 = FMA3 支持（融合乘加，a*b+c 一条指令完成）
        info.hasFma = (regs[1] & (1 << 3)) != 0;
        // EBX bit 16 = AVX-512 F 支持（AVX-512 基础标志之一）
        info.hasAvx512 = (regs[1] & (1 << 16)) != 0;
    }

    // ── 5. CPUID Leaf 0x80000000: 扩展 leaf 范围 ──
    // AMD/Intel 扩展信息需要先查询最大扩展 leaf 值
    CPUID(regs, 0x80000000);
    int maxExtLeaf = regs[0];

    // ── 6. CPUID Leaf 0x80000002~04: 处理器品牌字符串 ──
    // 这三个 leaf 各返回 16 字节的 ASCII 字符串，拼接即为完整的 CPU 名称
    // 例如 "Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz"
    if (maxExtLeaf >= 0x80000004) {
        char brand[49] = {};  // 3 leaves * 16 bytes + null terminator
        CPUID(regs, 0x80000002);
        memcpy(brand,      regs, 16);
        CPUID(regs, 0x80000003);
        memcpy(brand + 16, regs, 16);
        CPUID(regs, 0x80000004);
        memcpy(brand + 32, regs, 16);
        brand[48] = '\0';

        // 去除首尾空格（品牌字符串可能前导空格）
        std::string name(brand);
        size_t start = name.find_first_not_of(' ');
        size_t end   = name.find_last_not_of(' ');
        if (start != std::string::npos && end != std::string::npos) {
            info.modelName = name.substr(start, end - start + 1);
        }
    }

    // ── 7. 缓存大小查询 ──
    // 方式 A：扩展 leaf 0x80000005/06（AMD 风格，Intel 也支持）
    if (maxExtLeaf >= 0x80000006) {
        CPUID(regs, 0x80000006);
        // ECX[31:18] = L3 缓存大小（单位：512KB）
        // 例如 ECX = 0x00406040 → L3 = 0x40 * 512KB = 32MB
        int l3SizeIn512K = (regs[2] >> 18) & 0x3FFF;
        info.l3CacheSize = static_cast<size_t>(l3SizeIn512K) * 512 * 1024;
    }

    // L1 和 L2 缓存大小
    if (maxExtLeaf >= 0x80000005) {
        CPUID(regs, 0x80000005);
        // ECX[31:24] = L1 数据缓存大小（单位：KB）
        int l1SizeK = (regs[2] >> 24) & 0xFF;
        info.l1CacheSize = static_cast<size_t>(l1SizeK) * 1024;
    }
    if (maxExtLeaf >= 0x80000006) {
        CPUID(regs, 0x80000006);
        // ECX[23:16] = L2 缓存大小（单位：KB）
        int l2SizeK = (regs[2] >> 16) & 0xFF;
        info.l2CacheSize = static_cast<size_t>(l2SizeK) * 1024;
    }

    // ── 8. 物理核心数估算 ──
    // 精确查询需要 CPUID Leaf 0x0B（Intel）或 Leaf 0x8000001E（AMD）
    // 这里使用简单估算：逻辑线程数 / 2（假设开启了超线程）
    if (info.numThreads > 0) {
        // 如果支持超线程，物理核心数通常是逻辑线程数的一半
        // 更准确的做法是查询 CPUID Leaf 11 (Intel) 或 Leaf 0x8000001E (AMD)
        info.numCores = info.numThreads / 2;
        if (info.numCores < 1) info.numCores = 1;
    }

    return info;
}

// ============================================================================
// GPU 查询实现（Vulkan）
// ============================================================================

GpuInfo DeviceInfo::queryGpu() {
    GpuInfo info;

#ifdef ENABLE_VULKAN
    // ── Vulkan GPU 查询 ──
    // 流程：创建实例 → 枚举物理设备 → 查询属性和内存信息

    // 第一步：创建 Vulkan Instance
    // Instance 是与 Vulkan 驱动的连接入口
    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.enabledExtensionCount = 0;
    createInfo.ppEnabledExtensionNames = nullptr;

    VkInstance instance = VK_NULL_HANDLE;
    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

    if (result != VK_SUCCESS) {
        // Vulkan Instance 创建失败，返回空的 GpuInfo
        info.supportsVulkan = false;
        return info;
    }

    // 第二步：枚举物理设备
    // vkEnumeratePhysicalDevices 先查询数量，再获取设备列表
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
        // 没有找到任何 Vulkan 兼容的 GPU
        vkDestroyInstance(instance, nullptr);
        info.supportsVulkan = false;
        return info;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    // 第三步：选择第一个离散 GPU（如果有），否则选第一个设备
    // 理想情况下应该评估每个设备的算力，选择最优的
    // 这里简化处理：优先选 discrete GPU，否则选第一个
    VkPhysicalDevice selectedDevice = devices[0];
    for (auto dev : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            selectedDevice = dev;
            break;
        }
    }

    // 第四步：查询设备属性
    VkPhysicalDeviceProperties deviceProps;
    vkGetPhysicalDeviceProperties(selectedDevice, &deviceProps);

    info.deviceName = deviceProps.deviceName;  // UTF-8 字符串，如 "GeForce RTX 3060"
    info.vendorId   = deviceProps.vendorID;     // 厂商 ID：0x10DE=NVIDIA 等
    info.computeUnits = 0;  // Vulkan 不直接暴露 CU/SM 数量，需要通过 subgroup 或扩展查询

    // maxWorkGroupSize = maxComputeWorkGroupInvocations
    // 这是单个 workgroup 的最大线程数
    info.maxWorkGroupSize = deviceProps.limits.maxComputeWorkGroupInvocations;

    // 检查是否支持 compute shader（Vulkan 1.1+ 或 VK_KHR_shader_compute）
    // 简化判断：Vulkan 1.1 及以上默认支持 compute
    info.supportsVulkan = (deviceProps.apiVersion >= VK_MAKE_VERSION(1, 1, 0));

    // 第五步：查询显存大小
    // Vulkan 的显存信息通过 memory properties 获取
    // 我们找最大的 heap 作为全局显存
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(selectedDevice, &memProps);

    size_t maxHeapSize = 0;
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
        // VK_MEMORY_HEAP_DEVICE_LOCAL_BIT 表示设备本地内存（即显存）
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            if (memProps.memoryHeaps[i].size > maxHeapSize) {
                maxHeapSize = memProps.memoryHeaps[i].size;
            }
        }
    }
    info.globalMemoryBytes = maxHeapSize;

    // 清理 Vulkan Instance
    vkDestroyInstance(instance, nullptr);
#else
    // Vulkan 未启用，返回默认的空 GpuInfo
    info.supportsVulkan = false;
#endif // ENABLE_VULKAN

    return info;
}

// ============================================================================
// 打印设备信息摘要
// ============================================================================

void DeviceInfo::printSummary() {
    std::cout << "========================================\n";
    std::cout << "     设备能力查询结果 (Device Info)\n";
    std::cout << "========================================\n\n";

    // ── CPU 信息 ──
    CpuInfo cpu = queryCpu();
    std::cout << "[CPU]\n";
    std::cout << "  型号名称       : " << cpu.modelName << "\n";
    std::cout << "  物理核心数     : " << cpu.numCores << "\n";
    std::cout << "  逻辑线程数     : " << cpu.numThreads << "\n";
    std::cout << "  AVX2 支持      : " << (cpu.hasAvx2  ? "是" : "否") << "\n";
    std::cout << "  FMA3 支持      : " << (cpu.hasFma   ? "是" : "否") << "\n";
    std::cout << "  AVX-512 支持   : " << (cpu.hasAvx512 ? "是" : "否") << "\n";
    std::cout << "  L1 缓存大小    : " << cpu.l1CacheSize / 1024 << " KB\n";
    std::cout << "  L2 缓存大小    : " << cpu.l2CacheSize / 1024 << " KB\n";
    std::cout << "  L3 缓存大小    : " << cpu.l3CacheSize / (1024 * 1024) << " MB\n";
    std::cout << "\n";

    // ── GPU 信息 ──
    GpuInfo gpu = queryGpu();
    std::cout << "[GPU]\n";
    if (gpu.supportsVulkan && !gpu.deviceName.empty()) {
        std::cout << "  设备名称       : " << gpu.deviceName << "\n";
        // 将厂商 ID 转换为可读名称
        const char* vendorName = "未知";
        if (gpu.vendorId == 0x10DE)      vendorName = "NVIDIA";
        else if (gpu.vendorId == 0x8086) vendorName = "Intel";
        else if (gpu.vendorId == 0x1002) vendorName = "AMD";
        else if (gpu.vendorId == 0x13B5) vendorName = "ARM (Mali)";
        else if (gpu.vendorId == 0x5143) vendorName = "Qualcomm (Adreno)";
        std::cout << "  厂商           : " << vendorName
                  << " (0x" << std::hex << gpu.vendorId << std::dec << ")\n";
        std::cout << "  显存大小       : " << gpu.globalMemoryBytes / (1024 * 1024) << " MB\n";
        std::cout << "  计算单元数     : " << gpu.computeUnits << "\n";
        std::cout << "  最大工作组大小 : " << gpu.maxWorkGroupSize << "\n";
        std::cout << "  Vulkan Compute : " << (gpu.supportsVulkan ? "支持" : "不支持") << "\n";
    } else {
        std::cout << "  （未检测到 Vulkan 兼容的 GPU）\n";
    }

    std::cout << "\n========================================\n";
}

} // namespace edgeai
