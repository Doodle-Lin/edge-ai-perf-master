/**
 * @file step1_device_query.cpp
 * @brief Lab2 Step1: 设备能力探测 —— 了解你的硬件，才能做出合理的调度决策
 *
 * ====== 本实验学习目标 ======
 *
 * 1. 理解为什么调度器需要先探测硬件：不同的 CPU/GPU 组合，最优策略完全不同
 * 2. 掌握 CPUID 指令查询 CPU 特性的原理（核心数、缓存、SIMD 扩展）
 * 3. 理解 GPU 通过 Vulkan API 查询的流程（设备名、显存、计算单元）
 * 4. 学会用探测结果指导调度决策
 *
 * ====== 关键概念 ======
 *
 * 为什么设备信息对调度如此重要？
 * ┌──────────────────────────────────────────────────────────────────┐
 * │ 1. AVX2 支持决定了能否使用 SIMD 优化的 CPU 算子                │
 * │    - 无 AVX2: Conv2D 只能用朴素实现，约 2~5 GFLOPS              │
 * │    - 有 AVX2: Conv2D 可用向量化实现，约 20~40 GFLOPS            │
 * │    → 如果 CPU 有 AVX2，部分小算子在 CPU 上可能比 GPU 更快       │
 * │    （因为省去了 GPU kernel launch 开销和内存拷贝）               │
 * │                                                                  │
 * │ 2. GPU 显存大小限制了能加载的最大模型                             │
 * │    - 2GB 显存: 最多放 ~500MB 模型（剩余给中间激活值）            │
 * │    - 8GB 显存: 可以放 2GB 级别的模型                              │
 * │    → 超出显存的模型必须用 CPU，或做模型分片（model partitioning） │
 * │                                                                  │
 * │ 3. 缓存大小决定了算子的最优分块（tiling）策略                    │
 * │    - L1 32KB: 分块 8x8 的 GEMM 可以完全在 L1 中完成             │
 * │    - L2 256KB: 分块 32x32 可以利用 L2                            │
 * │    - L3 8MB: 大分块可以在 L3 中复用权重                           │
 * │    → 缓存越大，能复用的数据越多，内存带宽瓶颈越不明显            │
 * └──────────────────────────────────────────────────────────────────┘
 *
 * 编译运行:
 *   g++ -std=c++17 -O2 -o step1_device_query step1_device_query.cpp
 *   ./step1_device_query
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

// ============================================================================
// 模拟框架中的 CpuInfo / GpuInfo 结构体（与 include/scheduler/DeviceInfo.h 对齐）
// 实际框架中这些定义在 edgeai::CpuInfo 和 edgeai::GpuInfo
// 这里我们自行实现查询逻辑，方便独立编译学习
// ============================================================================

/// CPU 硬件能力描述
struct CpuInfo {
    int numCores      = 0;     ///< 物理核心数（不含超线程）
    int numThreads    = 0;     ///< 逻辑线程数（含超线程），如 8C/16T → 16
    bool hasAvx2      = false; ///< 是否支持 AVX2（256-bit SIMD，float×8 / int8×32）
    bool hasFma       = false; ///< 是否支持 FMA3（融合乘加，一条指令完成 a*b+c）
    bool hasAvx512    = false; ///< 是否支持 AVX-512（512-bit SIMD，float×16）
    size_t l1CacheSize = 0;    ///< L1 数据缓存大小（字节），通常每核 32KB
    size_t l2CacheSize = 0;    ///< L2 缓存大小（字节），通常每核 256KB~1MB
    size_t l3CacheSize = 0;    ///< L3 共享缓存大小（字节），通常 4~32MB
    std::string modelName;     ///< CPU 型号名称
};

/// GPU 硬件能力描述
struct GpuInfo {
    std::string deviceName;       ///< 设备名称
    uint32_t vendorId       = 0; ///< 厂商 ID：0x10DE=NVIDIA, 0x8086=Intel, 0x1002=AMD
    size_t globalMemoryBytes = 0; ///< 全局显存大小（字节）
    uint32_t computeUnits   = 0; ///< 计算单元数量
    uint32_t maxWorkGroupSize = 0; ///< 单个工作组最大线程数
    bool supportsVulkan     = false; ///< 是否支持 Vulkan compute
};

// ============================================================================
// CPUID 查询实现（x86 平台专用）
// ============================================================================

/// 执行 CPUID 指令的跨平台封装
/// @param leaf    EAX 输入值（查询的功能号）
/// @param subLeaf ECX 输入值（子功能号，通常为 0）
/// @param regs    输出 4 个 32 位寄存器值 [EAX, EBX, ECX, EDX]
static void cpuidImpl(uint32_t leaf, uint32_t subLeaf, uint32_t regs[4]) {
#if defined(_MSC_VER)
    // MSVC 提供的 __cpuidex 内建函数，直接编译为 CPUID 指令
    __cpuidex(reinterpret_cast<int*>(regs), static_cast<int>(leaf),
              static_cast<int>(subLeaf));
#elif defined(__GNUC__) || defined(__clang__)
    // GCC/Clang 使用内联汇编调用 CPUID
    // "=a"/"=b"/"=c"/"=d" 分别对应 EAX/EBX/ECX/EDX 输出
    // "a"/"c" 分别对应 EAX/ECX 输入
    __asm__ __volatile__(
        "cpuid"
        : "=a"(regs[0]), "=b"(regs[1]), "=c"(regs[2]), "=d"(regs[3])
        : "a"(leaf), "c"(subLeaf)
    );
#else
    // 不支持的平台，清零返回
    regs[0] = regs[1] = regs[2] = regs[3] = 0;
#endif
}

// ============================================================================
// 设备信息查询 —— 模拟框架中 DeviceInfo::queryCpu() / queryGpu() 的行为
// ============================================================================

/**
 * @brief 查询 CPU 信息
 *
 * 底层原理：
 * - CPUID EAX=0: 返回厂商字符串（"GenuineIntel" 或 "AuthenticAMD"）
 * - CPUID EAX=1: 返回家族型号 + 特性标志（EDX bit 26=SSE2, ECX bit 28=AVX）
 * - CPUID EAX=7, ECX=0: 返回扩展特性（EBX bit 5=AVX2, bit 16=AVX-512 F）
 * - CPUID EAX=0x80000000: 返回扩展功能最大值
 * - CPUID EAX=0x80000002~4: 返回处理器品牌字符串
 * - CPUID EAX=4: 返回缓存拓扑信息（L1/L2/L3 大小和关联度）
 *
 * @return 填充好的 CpuInfo 结构体
 */
CpuInfo queryCpu() {
    CpuInfo info;

    // ── 第一步：获取 CPU 厂商和最大支持的功能号 ──
    uint32_t regs[4] = {};
    cpuidImpl(0, 0, regs);
    // regs[1]=EBX, regs[3]=EDX, regs[2]=ECX 包含 12 字节厂商字符串
    char vendor[13] = {};
    memcpy(vendor + 0, &regs[1], 4);  // EBX
    memcpy(vendor + 4, &regs[3], 4);  // EDX
    memcpy(vendor + 8, &regs[2], 4);  // ECX
    // vendor 此时是 "GenuineIntel" 或 "AuthenticAMD"

    uint32_t maxLeaf = regs[0];  // EAX = 最大支持的 EAX 输入值

    // ── 第二步：获取 CPU 型号名称（品牌字符串） ──
    // 需要 3 次 CPUID 调用 (EAX=0x80000002/3/4)，每次返回 16 字节
    // 这个功能是 AMD 率先引入的，Intel 后来也支持（从 P4 开始）
    uint32_t extMaxLeaf = 0;
    cpuidImpl(0x80000000, 0, regs);
    extMaxLeaf = regs[0];

    if (extMaxLeaf >= 0x80000004) {
        // 扩展品牌信息可用，读取 48 字节的处理器名称
        char brand[49] = {};
        for (uint32_t i = 0; i < 3; ++i) {
            cpuidImpl(0x80000002 + i, 0, regs);
            memcpy(brand + i * 16, regs, 16);
        }
        brand[48] = '\0';

        // 去除前导空格（Intel 品牌字符串通常前面有很多空格）
        const char* p = brand;
        while (*p == ' ') ++p;
        info.modelName = p;
    } else {
        info.modelName = "Unknown (ext brand not supported)";
    }

    // ── 第三步：检测 SIMD 扩展支持 ──
    if (maxLeaf >= 1) {
        cpuidImpl(1, 0, regs);
        // ECX bit 28 = AVX 支持
        bool hasAvx = (regs[2] >> 28) & 1;
        // 不检测 AVX 本身，只检测 AVX2（因为 AVX2 才是我们优化的最低要求）
        // AVX2 在 EAX=7 的 EBX 中
    }

    if (maxLeaf >= 7) {
        cpuidImpl(7, 0, regs);
        // EBX bit 5 = AVX2 支持
        // AVX2 是 256-bit SIMD，可以一次处理 8 个 float 或 32 个 int8
        // 我们的 Conv2dAvx2 实现就依赖这个特性
        info.hasAvx2 = (regs[1] >> 5) & 1;
        // EBX bit 3 = BMI1（位操作指令集）
        // EBX bit 8 = BMI2
        // EBX bit 16 = AVX-512 F（Foundation）
        info.hasAvx512 = (regs[1] >> 16) & 1;
        // EBX bit 12 = AVX-512 VL（Vector Length Extensions）
    }

    // FMA3 在 CPUID EAX=1, ECX bit 12
    if (maxLeaf >= 1) {
        cpuidImpl(1, 0, regs);
        info.hasFma = (regs[2] >> 12) & 1;
    }

    // ── 第四步：查询缓存拓扑 ──
    // CPUID EAX=4 是 Intel 专用的缓存枚举方法
    // AMD 使用 EAX=0x8000001D，这里为简化只实现 Intel 的
    if (maxLeaf >= 4) {
        for (uint32_t cacheId = 0; ; ++cacheId) {
            cpuidImpl(4, cacheId, regs);
            uint32_t cacheType = regs[0] & 0x1F;
            if (cacheType == 0) break;  // 没有更多缓存了

            // EAX[31:22] = (Ways of associativity) - 1
            // EAX[21:12] = (Number of sets) - 1
            // EBX[11:0]  = (Line size) - 1
            // EBX[21:12] = (Physical line partitions) - 1
            // EBX[31:22] = (Ways of associativity) - 1
            // 缓存大小 = (Ways + 1) * (Partitions + 1) * (LineSize + 1) * (Sets + 1)
            uint32_t ways      = ((regs[1] >> 22) & 0x3FF) + 1;
            uint32_t partitions = ((regs[1] >> 12) & 0x3FF) + 1;
            uint32_t lineSize  = (regs[1] & 0xFFF) + 1;
            uint32_t sets      = ((regs[0] >> 22) & 0x3FF) + 1;
            size_t cacheSize   = static_cast<size_t>(ways) * partitions * lineSize * sets;

            // cacheType: 1=Data, 2=Instruction, 3=Unified
            if (cacheType == 1 || cacheType == 3) {
                // 根据 cacheId 判断是 L1/L2/L3
                // 通常 cacheId=0 是 L1D, 1 是 L1I, 2 是 L2, 3 是 L3
                if (cacheId <= 1 && info.l1CacheSize == 0) {
                    info.l1CacheSize = cacheSize;
                } else if (cacheId == 2 && info.l2CacheSize == 0) {
                    info.l2CacheSize = cacheSize;
                } else if (cacheId >= 3 && info.l3CacheSize == 0) {
                    info.l3CacheSize = cacheSize;
                }
            }
        }
    }

    // ── 第五步：获取核心数 ──
    // 方法一：CPUID EAX=0xB (Intel Extended Topology)
    // 方法二：OS 提供的接口 std::thread::hardware_concurrency()
    // 这里优先用 CPUID，失败则回退到 OS 接口

#if defined(_MSC_VER)
    // Windows: 使用 GetLogicalProcessorInformationEx
    // 简化实现：使用 std::thread::hardware_concurrency()
    info.numThreads = static_cast<int>(std::thread::hardware_concurrency());
    // 物理核心数通常 = 逻辑线程数 / 2（假设开启超线程）
    // 更准确的做法是查 CPUID EAX=0xB 或 SMBIOS，但这里简化处理
    info.numCores = info.numThreads / 2;
    if (info.numCores == 0) info.numCores = info.numThreads;  // 无超线程的情况
#else
    // Linux: 读取 /proc/cpuinfo 或用 sysconf
    info.numThreads = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
    info.numCores = info.numThreads / 2;
    if (info.numCores == 0) info.numCores = info.numThreads;
#endif

    return info;
}

/**
 * @brief 查询 GPU 信息（通过 Vulkan API）
 *
 * 底层原理：
 * 1. 创建 Vulkan Instance（应用与驱动的连接）
 * 2. vkEnumeratePhysicalDevices() 枚举所有 GPU
 * 3. vkGetPhysicalDeviceProperties() 获取设备属性
 * 4. vkGetPhysicalDeviceMemoryProperties() 获取显存信息
 *
 * 这里我们简化实现——如果 Vulkan 不可用就返回空信息。
 * 实际框架中会完整调用 Vulkan API。
 *
 * @return 填充好的 GpuInfo 结构体
 */
GpuInfo queryGpu() {
    GpuInfo info;

    // ── 尝试加载 Vulkan ──
    // 在实际框架中，这里会调用:
    //   VkInstance instance;
    //   vkCreateInstance(&createInfo, nullptr, &instance);
    //   uint32_t deviceCount = 0;
    //   vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    //   ... 获取每个设备的属性 ...

    // 简化实现：我们模拟一个典型的笔记本 GPU
    // 实际生产代码应该真正调用 Vulkan API
    // 这里只是演示"查询 GPU 信息"这个概念

    // 检查是否有 Vulkan 运行时（通过尝试加载 vulkan-1.dll / libvulkan.so）
    bool vulkanAvailable = false;

#if defined(_WIN32)
    // Windows: 尝试加载 vulkan-1.dll
    HMODULE vulkanLib = LoadLibraryA("vulkan-1.dll");
    if (vulkanLib) {
        vulkanAvailable = true;
        FreeLibrary(vulkanLib);
    }
#else
    // Linux: 尝试加载 libvulkan.so
    void* vulkanLib = dlopen("libvulkan.so.1", RTLD_LAZY);
    if (vulkanLib) {
        vulkanAvailable = true;
        dlclose(vulkanLib);
    }
#endif

    if (vulkanAvailable) {
        // 有 Vulkan 运行时，但在本实验中不深入调用 API
        // 实际框架的 DeviceInfo::queryGpu() 会完整查询
        info.supportsVulkan = true;
        info.deviceName = "Vulkan-capable GPU detected (details require API call)";
        info.globalMemoryBytes = 0;  // 需要实际 API 调用才能获取
        info.computeUnits = 0;
    } else {
        // 没有 Vulkan 运行时
        info.supportsVulkan = false;
        info.deviceName = "No Vulkan runtime found";
    }

    return info;
}

// ============================================================================
// 辅助函数：格式化字节数为人类可读的字符串
// ============================================================================

/// 将字节数转为 "xxx KB" / "xxx MB" / "xxx GB" 的可读格式
static std::string formatBytes(size_t bytes) {
    const double KB = 1024.0;
    const double MB = 1024.0 * 1024.0;
    const double GB = 1024.0 * 1024.0 * 1024.0;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    if (bytes >= GB) {
        oss << (bytes / GB) << " GB";
    } else if (bytes >= MB) {
        oss << (bytes / MB) << " MB";
    } else if (bytes >= KB) {
        oss << (bytes / KB) << " KB";
    } else {
        oss << bytes << " B";
    }
    return oss.str();
}

// ============================================================================
// 打印格式化的设备信息摘要
// ============================================================================

void printDeviceSummary(const CpuInfo& cpu, const GpuInfo& gpu) {
    // ── 分隔线宽度 ──
    const int W = 70;

    auto printLine = [&](const std::string& label, const std::string& value) {
        std::cout << "│ " << std::left << std::setw(30) << label
                  << "│ " << std::left << std::setw(36) << value << "│\n";
    };

    auto printSep = [&]() {
        std::cout << "├" << std::string(31, '─')
                  << "┼" << std::string(37, '─') << "┤\n";
    };

    // ── CPU 信息表 ──
    std::cout << "\n";
    std::cout << "╔" << std::string(W, '═') << "╗\n";
    std::cout << "║  CPU 设备信息 (via CPUID)                                      ║\n";
    std::cout << "╠" << std::string(W, '═') << "╣\n";

    printLine("CPU 型号", cpu.modelName);
    printSep();
    printLine("物理核心数", std::to_string(cpu.numCores) + " 核");
    printLine("逻辑线程数", std::to_string(cpu.numThreads) + " 线程"
               + (cpu.numThreads > cpu.numCores ? " (超线程已开启)" : ""));
    printSep();
    printLine("AVX2 支持", cpu.hasAvx2 ? "YES - 可用 SIMD 优化算子" : "NO - 只能用朴素实现");
    printLine("FMA3 支持", cpu.hasFma ? "YES - 融合乘加指令" : "NO");
    printLine("AVX-512 支持", cpu.hasAvx512 ? "YES - 512-bit SIMD" : "NO");
    printSep();
    printLine("L1 缓存", cpu.l1CacheSize > 0 ? formatBytes(cpu.l1CacheSize) : "未探测到");
    printLine("L2 缓存", cpu.l2CacheSize > 0 ? formatBytes(cpu.l2CacheSize) : "未探测到");
    printLine("L3 缓存", cpu.l3CacheSize > 0 ? formatBytes(cpu.l3CacheSize) : "未探测到");

    std::cout << "╚" << std::string(W, '═') << "╝\n";

    // ── GPU 信息表 ──
    std::cout << "\n";
    std::cout << "╔" << std::string(W, '═') << "╗\n";
    std::cout << "║  GPU 设备信息 (via Vulkan API)                                 ║\n";
    std::cout << "╠" << std::string(W, '═') << "╣\n";

    printLine("GPU 设备名", gpu.deviceName);
    printSep();
    printLine("Vulkan 支持", gpu.supportsVulkan ? "YES - 可使用 GPU Compute" : "NO - 仅 CPU 可用");
    printLine("显存大小", gpu.globalMemoryBytes > 0
               ? formatBytes(gpu.globalMemoryBytes) : "N/A (需 Vulkan API 查询)");
    printLine("计算单元数", gpu.computeUnits > 0
               ? std::to_string(gpu.computeUnits) : "N/A");
    printLine("最大工作组大小", gpu.maxWorkGroupSize > 0
               ? std::to_string(gpu.maxWorkGroupSize) : "N/A");

    std::cout << "╚" << std::string(W, '═') << "╝\n";
}

// ============================================================================
// 根据设备信息生成调度建议
// ============================================================================

void printSchedulingAdvice(const CpuInfo& cpu, const GpuInfo& gpu) {
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  调度建议（基于探测到的硬件能力）\n";
    std::cout << "================================================================\n\n";

    // ── 建议 1: CPU SIMD 能力 ──
    if (cpu.hasAvx2) {
        std::cout << "[CPU] 检测到 AVX2 支持 → 可以使用 Conv2dAvx2 优化实现\n";
        std::cout << "      - 256-bit SIMD 可一次处理 8 个 float (朴素实现只能 1 个)\n";
        std::cout << "      - 对于中小卷积（FLOPs < 10M），CPU + AVX2 可能比 GPU 更快\n";
        std::cout << "      - 原因：省去了 GPU kernel launch 开销（~5-50us）和内存拷贝\n";
    } else {
        std::cout << "[CPU] 未检测到 AVX2 → 只能使用 Conv2dNaive 朴素实现\n";
        std::cout << "      - 朴素实现性能约为 AVX2 版本的 1/4 ~ 1/8\n";
        std::cout << "      - 建议：尽量将计算密集型算子分给 GPU\n";
    }
    std::cout << "\n";

    // ── 建议 2: GPU 可用性 ──
    if (gpu.supportsVulkan) {
        std::cout << "[GPU] Vulkan 可用 → 可以使用 Vulkan Compute Shader 加速\n";
        std::cout << "      - 大卷积（FLOPs > 100M）建议放 GPU\n";
        std::cout << "      - 大批量矩阵乘法建议放 GPU\n";
        if (gpu.globalMemoryBytes > 0) {
            double gpuMemGB = static_cast<double>(gpu.globalMemoryBytes) / (1024.0 * 1024.0 * 1024.0);
            if (gpuMemGB < 4.0) {
                std::cout << "      - 注意：显存较小 (" << gpuMemGB
                          << " GB)，大模型可能无法完整加载到 GPU\n";
                std::cout << "      - 需要考虑 CPU-GPU 混合调度或模型分片\n";
            } else {
                std::cout << "      - 显存充足 (" << gpuMemGB
                          << " GB)，大多数边缘模型可完整加载\n";
            }
        }
    } else {
        std::cout << "[GPU] Vulkan 不可用 → 纯 CPU 执行模式\n";
        std::cout << "      - 无法使用 GPU 加速，所有算子只能在 CPU 上运行\n";
        std::cout << "      - 优化重点：CPU 缓存优化、SIMD 向量化、多线程并行\n";
    }
    std::cout << "\n";

    // ── 建议 3: 缓存感知调度 ──
    if (cpu.l2CacheSize > 0) {
        // 根据 L2 缓存大小估算最优分块
        // 经验法则：分块大小应使输入+权重+输出都能放进 L2
        // 例如 L2=256KB，float=4B，最多 64K 个 float
        // 分块 16x16 的 GEMM：输入 16xK + Kx16 + 输出 16x16 ≈ 32K float
        // 当 K=16 时，总数据 32*16*4B = 2KB，远小于 256KB
        // 当 K=1024 时，总数据 32*1024*4B = 128KB，也在 L2 范围内
        std::cout << "[缓存] L2 大小 = " << formatBytes(cpu.l2CacheSize) << "\n";
        size_t l2Floats = cpu.l2CacheSize / sizeof(float);
        std::cout << "       L2 可容纳 ~" << (l2Floats / 1024) << "K 个 float\n";
        std::cout << "       - GEMM 分块建议：使 A_tile + B_tile + C_tile < L2 的 80%\n";
        std::cout << "       - Conv2D im2col 分块：建议 tile_H * tile_W * C_in < L2\n";
    }
    std::cout << "\n";

    // ── 建议 4: 综合调度策略 ──
    std::cout << "[综合] 推荐的调度策略：\n";
    if (gpu.supportsVulkan && cpu.hasAvx2) {
        std::cout << "        → 混合调度（CPU + GPU 协同）\n";
        std::cout << "        - 大卷积 → GPU（数据并行优势）\n";
        std::cout << "        - 小算子/逐元素操作 → CPU（AVX2 足够快，避免 GPU 开销）\n";
        std::cout << "        - 并行分支 → CPU 和 GPU 各跑一个分支\n";
    } else if (gpu.supportsVulkan) {
        std::cout << "        → GPU 优先（CPU 无 AVX2，性能差距大）\n";
        std::cout << "        - 尽可能多的算子放 GPU\n";
        std::cout << "        - 仅 GPU 不支持的算子放 CPU\n";
    } else {
        std::cout << "        → 纯 CPU 调度（无 GPU 可用）\n";
        std::cout << "        - 所有算子在 CPU 执行\n";
        if (cpu.hasAvx2) {
            std::cout << "        - 利用 AVX2 优化提升 CPU 性能\n";
        }
        std::cout << "        - 利用多核并行加速独立算子\n";
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "==============================================================\n";
    std::cout << "  Lab2 Step1: 设备能力探测 (Device Capability Query)\n";
    std::cout << "==============================================================\n";
    std::cout << "\n";
    std::cout << "本实验演示如何使用 CPUID 和 Vulkan API 探测硬件能力，\n";
    std::cout << "这是异构调度器的第一步——了解你手中的硬件牌面。\n";
    std::cout << "\n";

    // ── 步骤 1: 查询 CPU 信息 ──
    // 对应框架 API: edgeai::DeviceInfo::queryCpu()
    std::cout << "[1/2] 正在查询 CPU 信息 (CPUID)...\n";
    CpuInfo cpu = queryCpu();
    std::cout << "      完成！\n";

    // ── 步骤 2: 查询 GPU 信息 ──
    // 对应框架 API: edgeai::DeviceInfo::queryGpu()
    std::cout << "[2/2] 正在查询 GPU 信息 (Vulkan)...\n";
    GpuInfo gpu = queryGpu();
    std::cout << "      完成！\n";

    // ── 步骤 3: 打印格式化的设备摘要 ──
    // 对应框架 API: edgeai::DeviceInfo::printSummary()
    printDeviceSummary(cpu, gpu);

    // ── 步骤 4: 基于探测结果给出调度建议 ──
    printSchedulingAdvice(cpu, gpu);

    // ── 总结 ──
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  关键收获\n";
    std::cout << "================================================================\n";
    std::cout << "\n";
    std::cout << "1. 设备探测是调度的前提：不知道硬件能力就无法做出合理分配\n";
    std::cout << "2. AVX2 支持 → 决定 CPU 能否用 SIMD 优化（性能差距 4-8 倍）\n";
    std::cout << "3. GPU 显存 → 限制可加载的模型大小，超出则必须 CPU 参与\n";
    std::cout << "4. 缓存大小 → 决定最优分块策略，影响 CPU 算子性能\n";
    std::cout << "5. Vulkan 可用性 → 决定是否能使用 GPU Compute Shader\n";
    std::cout << "\n";
    std::cout << "下一步: step2_op_graph_build.cpp —— 学习如何构建算子依赖图\n";

    return 0;
}
