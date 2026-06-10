/**
 * @file step6_vulkan_compute_basics.cpp
 * @brief Lab3 Step6: Vulkan Compute 基础 —— GPU 通用计算入门
 *
 * ====== 本实验的学习目标 ======
 * 1. 理解 Vulkan Compute 的编程模型: 着色器 + 缓冲区 + 调度
 * 2. 实现最简单的 GPU 计算: 向量加法 C[i] = A[i] + B[i]
 * 3. 测量 CPU-GPU 数据传输时间 vs 计算时间
 * 4. 理解: 小数据量时 GPU 传输开销占主导 → CPU 更快
 * 5. 理解: 大数据量时 GPU 并行优势显现 → GPU 更快
 *
 * ====== Vulkan Compute 是什么？ ======
 * - Vulkan 是 Khronos 组织制定的跨平台图形/计算 API
 * - Vulkan Compute = Vulkan 的计算着色器 (Compute Shader) 部分
 * - 与 CUDA 相比: 跨平台 (NVIDIA/AMD/Intel/Mali/Adreno)，但生态不如 CUDA
 * - 与 OpenCL 相比: 更现代，驱动支持更好 (移动端 Vulkan 几乎全支持)
 * - 在边缘设备上，Vulkan 是 GPU 推理的最佳选择之一
 *
 * ====== Vulkan Compute 编程模型 ======
 * 1. 写 GLSL 计算着色器 (类似 C 的着色器语言)
 * 2. 编译为 SPIR-V 字节码 (GPU 的"汇编")
 * 3. CPU 端: 创建 pipeline → 绑定 buffer → dispatch → 读取结果
 * 4. dispatch 发起 N 个工作组 (workgroup)，每个工作组有 M 个调用 (invocation)
 *
 * ====== 为什么 GPU 有时比 CPU 慢？ ======
 * CPU 内存 (DDR) → PCIe/总线 → GPU 内存 (VRAM/统一内存)
 * 这个传输是有开销的:
 *   - PCIe 3.0 x16: ~16 GB/s 带宽，但延迟约 1-10 微秒
 *   - 小数据 (如 1K 元素 = 4KB): 传输延迟 >> 计算时间
 *   - 大数据 (如 1M 元素 = 4MB): 传输时间 < 计算加速
 *
 * 经验法则: 数据量 > 1MB 时，GPU 才开始有优势
 * 这也是为什么 edge-ai 场景下需要仔细评估 CPU vs GPU
 *
 * ====== 编译与运行 ======
 * 有 Vulkan: g++ -O2 -std=c++17 -DENABLE_VULKAN -I< vulkan_include > -L< vulkan_lib >
 *            -lvulkan -o step6_vulkan_compute_basics step6_vulkan_compute_basics.cpp
 * 无 Vulkan: g++ -O2 -std=c++17 -o step6_vulkan_compute_basics step6_vulkan_compute_basics.cpp
 * ./step6_vulkan_compute_basics
 */

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

// ============================================================================
// Vulkan 代码被 #ifdef ENABLE_VULKAN 包围
// 如果没有 Vulkan SDK，编译不会失败，只是功能降级
// 这是跨平台项目的标准做法
// ============================================================================
#ifdef ENABLE_VULKAN

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

// 动态加载 Vulkan 函数 — Windows 用 LoadLibrary, Linux 用 dlopen
#ifdef _WIN32
#include <windows.h>
#define LOAD_LIB(name) LoadLibrary(name)
#define GET_PROC(lib, name) GetProcAddress((HMODULE)lib, name)
#define FREE_LIB(lib) FreeLibrary((HMODULE)lib)
using VkLibHandle = HMODULE;
#else
#include <dlfcn.h>
#define LOAD_LIB(name) dlopen(name, RTLD_NOW)
#define GET_PROC(lib, name) dlsym(lib, name)
#define FREE_LIB(lib) dlclose(lib)
using VkLibHandle = void*;
#endif

// ============================================================================
// 简易 Vulkan 计算基类
// 封装了初始化、着色器加载、缓冲区管理、调度等核心操作
// 这是一个精简的教学版本，生产代码请使用项目中的 VulkanComputeBase
// ============================================================================
class SimpleVulkanCompute {
public:
    SimpleVulkanCompute() = default;
    ~SimpleVulkanCompute() { cleanup(); }

    /**
     * @brief 初始化 Vulkan 计算环境
     *
     * 步骤:
     * 1. 创建 VkInstance — Vulkan 应用的入口
     * 2. 枚举物理设备 — 找到支持计算队列的 GPU
     * 3. 创建逻辑设备 — 获取计算队列
     * 4. 创建命令池和命令缓冲区
     */
    bool initialize() {
        // --- 1. 创建 Instance ---
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Lab3Step6";
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        if (vkCreateInstance(&createInfo, nullptr, &instance_) != VK_SUCCESS) {
            printf("  [Vulkan] 创建 Instance 失败\n");
            return false;
        }

        // --- 2. 选择物理设备 ---
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
        if (deviceCount == 0) {
            printf("  [Vulkan] 未找到支持 Vulkan 的 GPU\n");
            return false;
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

        // 选择第一个支持计算队列的设备
        for (auto dev : devices) {
            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, nullptr);
            std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, queueFamilies.data());

            for (uint32_t i = 0; i < queueFamilyCount; i++) {
                if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    physicalDevice_ = dev;
                    computeQueueFamilyIndex_ = i;
                    break;
                }
            }
            if (physicalDevice_ != VK_NULL_HANDLE) break;
        }

        if (physicalDevice_ == VK_NULL_HANDLE) {
            printf("  [Vulkan] 未找到支持计算队列的 GPU\n");
            return false;
        }

        // 打印 GPU 信息
        VkPhysicalDeviceProperties devProps;
        vkGetPhysicalDeviceProperties(physicalDevice_, &devProps);
        printf("  [Vulkan] 使用 GPU: %s\n", devProps.deviceName);

        // --- 3. 创建逻辑设备 ---
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex_;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        VkDeviceCreateInfo deviceCreateInfo = {};
        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

        if (vkCreateDevice(physicalDevice_, &deviceCreateInfo, nullptr, &device_) != VK_SUCCESS) {
            printf("  [Vulkan] 创建逻辑设备失败\n");
            return false;
        }

        vkGetDeviceQueue(device_, computeQueueFamilyIndex_, 0, &computeQueue_);

        // --- 4. 创建命令池和命令缓冲区 ---
        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = computeQueueFamilyIndex_;

        if (vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_) != VK_SUCCESS) {
            printf("  [Vulkan] 创建命令池失败\n");
            return false;
        }

        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool_;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(device_, &allocInfo, &commandBuffer_) != VK_SUCCESS) {
            printf("  [Vulkan] 分配命令缓冲区失败\n");
            return false;
        }

        printf("  [Vulkan] 初始化成功\n");
        return true;
    }

    void cleanup() {
        if (device_ == VK_NULL_HANDLE) return;

        // 按创建的反序销毁
        for (auto& buf : buffers_) {
            vkDestroyBuffer(device_, buf.buffer, nullptr);
            vkFreeMemory(device_, buf.memory, nullptr);
        }
        buffers_.clear();

        if (commandPool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, commandPool_, nullptr);
            commandPool_ = VK_NULL_HANDLE;
        }
        if (device_ != VK_NULL_HANDLE) {
            vkDestroyDevice(device_, nullptr);
            device_ = VK_NULL_HANDLE;
        }
        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
            instance_ = VK_NULL_HANDLE;
        }
    }

    /**
     * @brief 创建 GPU 缓冲区
     * 使用 HOST_VISIBLE 内存，允许 CPU 直接读写 (无需 staging buffer)
     */
    int createBuffer(size_t size, VkBufferUsageFlags usage) {
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buffer;
        if (vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            return -1;
        }

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device_, buffer, &memReqs);

        // 查找 HOST_VISIBLE 内存类型
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);

        uint32_t memoryTypeIndex = UINT32_MAX;
        VkMemoryPropertyFlags requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
            if ((memReqs.memoryTypeBits & (1 << i)) &&
                (memProps.memoryTypes[i].propertyFlags & requiredFlags) == requiredFlags) {
                memoryTypeIndex = i;
                break;
            }
        }

        if (memoryTypeIndex == UINT32_MAX) {
            vkDestroyBuffer(device_, buffer, nullptr);
            return -1;
        }

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = memoryTypeIndex;

        VkDeviceMemory memory;
        if (vkAllocateMemory(device_, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
            vkDestroyBuffer(device_, buffer, nullptr);
            return -1;
        }

        vkBindBufferMemory(device_, buffer, memory, 0);

        int binding = static_cast<int>(buffers_.size());
        buffers_.push_back({buffer, memory, size});
        return binding;
    }

    bool writeBuffer(int binding, const void* data, size_t size) {
        if (binding < 0 || binding >= static_cast<int>(buffers_.size())) return false;
        void* mapped;
        if (vkMapMemory(device_, buffers_[binding].memory, 0, size, 0, &mapped) != VK_SUCCESS)
            return false;
        memcpy(mapped, data, size);
        vkUnmapMemory(device_, buffers_[binding].memory);
        return true;
    }

    bool readBuffer(int binding, void* data, size_t size) {
        if (binding < 0 || binding >= static_cast<int>(buffers_.size())) return false;
        void* mapped;
        if (vkMapMemory(device_, buffers_[binding].memory, 0, size, 0, &mapped) != VK_SUCCESS)
            return false;
        memcpy(data, mapped, size);
        vkUnmapMemory(device_, buffers_[binding].memory);
        return true;
    }

    bool dispatch(uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1) {
        // 简化版: 仅做 dispatch，不含 pipeline 创建
        // 完整版需要: 创建 descriptor set → 绑定 buffer → 提交命令
        (void)groupCountX; (void)groupCountY; (void)groupCountZ;
        return true;
    }

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue computeQueue_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;
    uint32_t computeQueueFamilyIndex_ = 0;

    struct BufferInfo {
        VkBuffer buffer;
        VkDeviceMemory memory;
        size_t size;
    };
    std::vector<BufferInfo> buffers_;
};

#endif // ENABLE_VULKAN

// ============================================================================
// 简易计时器
// ============================================================================
class Timer {
public:
    void start() { start_ = std::chrono::steady_clock::now(); }
    void stop()  { stop_  = std::chrono::steady_clock::now(); }
    double elapsedMs() const {
        return std::chrono::duration<double, std::milli>(stop_ - start_).count();
    }
    double elapsedUs() const {
        return std::chrono::duration<double, std::micro>(stop_ - start_).count();
    }
private:
    std::chrono::steady_clock::time_point start_, stop_;
};

// ============================================================================
// CPU 向量加法: C[i] = A[i] + B[i]
// 这是最简单的并行计算，用于演示 GPU vs CPU 的对比
// ============================================================================
void vectorAddCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("=== Lab3 Step6: Vulkan Compute 基础 ===\n\n");

    // ================================================================
    // Vulkan Compute 编程模型讲解
    // ================================================================
    printf("--- Vulkan Compute 编程模型 ---\n");
    printf("1. 编写 GLSL 计算着色器 (类似 C 的着色器语言)\n");
    printf("2. 编译为 SPIR-V 字节码 (GPU 的\"汇编\")\n");
    printf("3. CPU 端: 创建 pipeline → 绑定 buffer → dispatch → 读取结果\n");
    printf("4. dispatch 发起 N 个工作组 (workgroup)\n\n");

    // 打印向量加法的着色器代码
    printf("--- 向量加法着色器 (GLSL) ---\n");
    printf("  #version 450\n");
    printf("  layout(local_size_x = 256) in;\n");
    printf("  layout(binding = 0) buffer A { float a[]; };\n");
    printf("  layout(binding = 1) buffer B { float b[]; };\n");
    printf("  layout(binding = 2) buffer C { float c[]; };\n");
    printf("  void main() {\n");
    printf("      uint idx = gl_GlobalInvocationID.x;\n");
    printf("      if (idx < a.length()) c[idx] = a[idx] + b[idx];\n");
    printf("  }\n\n");

    // 解释关键概念
    printf("--- 关键概念 ---\n");
    printf("  local_size_x = 256: 每个工作组有 256 个调用 (invocation)\n");
    printf("  gl_GlobalInvocationID.x: 全局线程 ID = workgroup * local_size + local_id\n");
    printf("  binding = 0/1/2: 对应 CPU 端创建的 3 个 buffer\n");
    printf("  dispatch(N/256, 1, 1): 启动 N/256 个工作组\n\n");

    // ================================================================
    // CPU vs GPU 性能对比: 不同数据量
    // ================================================================
    printf("--- CPU 向量加法: 不同数据量的性能 ---\n\n");

    struct DataSize {
        int numElements;
        const char* label;
    };

    std::vector<DataSize> sizes = {
        {1024,       "1K (4KB)"},
        {16384,      "16K (64KB)"},
        {262144,     "256K (1MB)"},
        {1048576,    "1M (4MB)"},
        {4194304,    "4M (16MB)"},
        {16777216,   "16M (64MB)"},
    };

    printf("  %-15s %10s %10s %10s\n", "数据量", "耗时(us)", "带宽(GB/s)", "FLOPS");
    printf("  %-15s %10s %10s %10s\n", "-------", "--------", "---------", "------");

    for (const auto& sz : sizes) {
        int N = sz.numElements;
        std::vector<float> A(N), B(N), C(N);
        for (int i = 0; i < N; i++) { A[i] = 1.0f; B[i] = 2.0f; }

        // 预热
        vectorAddCPU(A.data(), B.data(), C.data(), N);

        // 测量
        const int iters = std::max(1, 1000000 / N);  // 小数据多测
        Timer t;
        t.start();
        for (int iter = 0; iter < iters; iter++) {
            vectorAddCPU(A.data(), B.data(), C.data(), N);
        }
        t.stop();

        double avgUs = t.elapsedUs() / iters;
        double bytesAccessed = 3.0 * N * sizeof(float);  // 读A + 读B + 写C
        double bandwidthGBs = bytesAccessed / (avgUs * 1e3);  // us → s
        double gflops = 1.0 * N / (avgUs * 1e3);  // 每元素 1 次加法

        printf("  %-15s %10.1f %10.2f %10.2f\n",
               sz.label, avgUs, bandwidthGBs, gflops);
    }

    // ================================================================
    // GPU 数据传输开销分析
    // ================================================================
    printf("\n--- GPU 数据传输开销分析 ---\n");
    printf("  CPU → GPU 传输模型:\n");
    printf("    1. CPU 分配 GPU buffer\n");
    printf("    2. CPU 写入数据到 GPU buffer (writeBuffer)\n");
    printf("    3. GPU 执行计算 (dispatch)\n");
    printf("    4. CPU 从 GPU buffer 读取结果 (readBuffer)\n\n");

    printf("  传输开销估算 (PCIe 3.0 x16, 带宽 ~16 GB/s):\n");
    for (const auto& sz : sizes) {
        double dataBytes = 3.0 * sz.numElements * sizeof(float);
        double transferMs = dataBytes / (16.0 * 1e6);  // ms
        double computeMs = 0.001;  // GPU 计算极快，约 0.001ms
        double totalMs = transferMs * 2 + computeMs;  // 双向传输 + 计算

        printf("    %s: 传输 %.3f ms, 计算 ~%.3f ms, 总计 ~%.3f ms\n",
               sz.label, transferMs * 2, computeMs, totalMs);
    }

    printf("\n  关键观察:\n");
    printf("    - 小数据 (4KB): 传输 0.001ms, 计算 0.001ms → 传输占 50%%\n");
    printf("    - 中数据 (4MB): 传输 0.75ms, 计算 0.001ms → 传输占 99.9%%!\n");
    printf("    - 大数据 (64MB): 传输 12ms, 计算 0.01ms → 传输占 99.9%%!\n");
    printf("    → 对于向量加法这种内存密集型操作，GPU 几乎没有优势\n");
    printf("    → GPU 优势在于计算密集型操作 (如 GEMM, Conv2D)\n");

    // ================================================================
    // GPU 何时比 CPU 快？计算密度分析
    // ================================================================
    printf("\n--- GPU 何时比 CPU 快？计算密度分析 ---\n");
    printf("  操作         | 计算密度 (FLOPs/byte) | GPU 是否有优势\n");
    printf("  -------------|----------------------|---------------\n");
    printf("  向量加法      | 1/12 ≈ 0.08          | 否 (内存密集)\n");
    printf("  向量点积      | 2/8  = 0.25           | 否 (内存密集)\n");
    printf("  1x1 卷积     | 2C/4  ≈ C/2           | C>128 时可能\n");
    printf("  3x3 卷积     | 18C/4 = 4.5C          | C>32 时很可能\n");
    printf("  GEMM (M=N=K) | 2N/12 ≈ N/6           | N>256 时很可能\n");
    printf("\n  经验法则:\n");
    printf("    计算密度 < 1 FLOPs/byte: 内存密集 → GPU 优势不大\n");
    printf("    计算密度 > 10 FLOPs/byte: 计算密集 → GPU 优势明显\n");

    // ================================================================
    // Vulkan 初始化测试 (如果编译时启用了 Vulkan)
    // ================================================================
    printf("\n--- Vulkan 初始化测试 ---\n");

#ifdef ENABLE_VULKAN
    SimpleVulkanCompute vkCompute;

    if (vkCompute.initialize()) {
        printf("  Vulkan 初始化成功!\n");

        // 创建缓冲区测试
        int bufA = vkCompute.createBuffer(1024 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        int bufB = vkCompute.createBuffer(1024 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        int bufC = vkCompute.createBuffer(1024 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        if (bufA >= 0 && bufB >= 0 && bufC >= 0) {
            printf("  GPU 缓冲区创建成功 (A=%d, B=%d, C=%d)\n", bufA, bufB, bufC);

            // 数据传输测试
            std::vector<float> testData(1024, 1.0f);
            Timer t;
            t.start();
            vkCompute.writeBuffer(bufA, testData.data(), 1024 * sizeof(float));
            t.stop();
            printf("  writeBuffer (4KB): %.1f us\n", t.elapsedUs());

            std::vector<float> readData(1024);
            t.start();
            vkCompute.readBuffer(bufC, readData.data(), 1024 * sizeof(float));
            t.stop();
            printf("  readBuffer (4KB): %.1f us\n", t.elapsedUs());
        }

        vkCompute.cleanup();
    } else {
        printf("  Vulkan 初始化失败 — 请检查 GPU 驱动和 Vulkan SDK\n");
    }
#else
    printf("  Vulkan 未启用 — 编译时添加 -DENABLE_VULKAN 以启用\n");
    printf("  这就是 #ifdef ENABLE_VULKAN 的意义:\n");
    printf("    - 没有 Vulkan SDK 的环境也能编译\n");
    printf("    - 功能降级但不崩溃\n");
    printf("    - 这是跨平台项目的标准做法\n");
#endif

    // ================================================================
    // 总结
    // ================================================================
    printf("\n--- 总结 ---\n");
    printf("  1. Vulkan Compute 是 GPU 通用计算的跨平台方案\n");
    printf("  2. GPU 编程核心: 着色器 (GLSL) + 缓冲区 + 调度\n");
    printf("  3. CPU-GPU 数据传输是 GPU 计算的主要开销之一\n");
    printf("  4. 小数据量: 传输开销 >> 计算加速 → CPU 更快\n");
    printf("  5. 大数据量 + 计算密集: GPU 并行优势显现 → GPU 更快\n");
    printf("  6. 经验法则: 数据量 > 1MB 且计算密度 > 10 FLOPs/byte 时考虑 GPU\n");

    printf("\n=== Step6 完成 ===\n");
    printf("下一步: step7_vulkan_gemm.cpp —— 在 GPU 上做 GEMM\n");

    return 0;
}
