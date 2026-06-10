/**
 * @file VulkanComputeBase.h
 * @brief Vulkan 计算基类 —— GPU 通用计算的底层封装
 *
 * ====== 学习要点 ======
 * 1. 为什么用 GPU 做推理？
 *    - CPU 核心数少（4-16），但单核性能强，适合串行/分支密集型任务
 *    - GPU 核心数多（数百到数千），但单核简单，适合大规模并行任务
 *    - 矩阵乘法：M*N 个输出元素互相独立 → 天然并行 → GPU 优势
 *    - 批量推理：多个样本同时推理 → GPU 优势更明显
 *
 * 2. Vulkan Compute 是什么？
 *    - Vulkan 是跨平台图形/计算 API（Khronos 组织制定）
 *    - Vulkan Compute = Vulkan 的计算着色器 (Compute Shader) 部分
 *    - 与 CUDA 相比：跨平台（NVIDIA/AMD/Intel/Mali/Adreno），但生态不如 CUDA
 *    - 与 OpenCL 相比：更现代，驱动支持更好（移动端 Vulkan 几乎全支持）
 *    - 在边缘设备上，Vulkan 是 GPU 推理的最佳选择之一
 *
 * 3. Vulkan Compute 的编程模型：
 *    - 写 GLSL 计算着色器（类似 C 的着色器语言）
 *    - 编译为 SPIR-V 字节码（GPU 的"汇编"）
 *    - CPU 端：创建 pipeline → 绑定 buffer → dispatch → 读取结果
 *    - dispatch 发起 N 个工作组 (workgroup)，每个工作组有 M 个调用 (invocation)
 *    - 典型配置：workgroup = (16, 16, 1)，invocation = (1, 1, 1) per workgroup
 *
 * 4. 本类的使用流程：
 *    a. initialize()     — 创建 Vulkan 实例、选择物理设备、创建逻辑设备
 *    b. loadShader()     — 加载 SPIR-V 着色器文件
 *    c. createPipeline() — 创建计算管线（着色器 + 布局）
 *    d. createBuffer()   — 创建 GPU 缓冲区（用于输入/输出/参数）
 *    e. writeBuffer()    — 将 CPU 数据写入 GPU 缓冲区
 *    f. dispatch()       — 发起计算
 *    g. readBuffer()     — 从 GPU 缓冲区读回结果
 *    h. cleanup()        — 释放所有 Vulkan 资源
 *
 * 5. CPU-GPU 数据传输的开销：
 *    - writeBuffer / readBuffer 涉及 PCIe 传输（桌面）或总线传输（移动端）
 *    - 传输延迟约 1-10 微秒（小数据）到 1-10 毫秒（大数据）
 *    - 对于小矩阵，传输时间可能超过计算时间 → GPU 不划算
 *    - 经验法则：数据量 > 1MB 时，GPU 才开始有优势
 *    - 这也是为什么 edge-ai 场景下需要仔细评估 CPU vs GPU
 *
 * 6. #ifdef ENABLE_VULKAN 的作用：
 *    - Vulkan SDK 可能未安装，此时编译不应该失败
 *    - CMake 检测 Vulkan 是否可用，设置 ENABLE_VULKAN 宏
 *    - 整个 Vulkan 相关代码被条件编译包围
 *    - 这是跨平台项目的标准做法：特性可用时启用，不可用时降级
 */

#pragma once

#include "common/Tensor.h"
#include <cstdint>
#include <string>
#include <vector>

#ifdef ENABLE_VULKAN

#define VK_NO_PROTOTYPES  // 不引入 Vulkan 函数原型，由动态加载
#include <vulkan/vulkan.h>

namespace edgeai {

class VulkanComputeBase {
public:
    // ──────────────── 生命周期 ────────────────

    VulkanComputeBase() = default;
    ~VulkanComputeBase() { cleanup(); }

    // 禁止拷贝 —— Vulkan 对象是独占资源
    VulkanComputeBase(const VulkanComputeBase&) = delete;
    VulkanComputeBase& operator=(const VulkanComputeBase&) = delete;

    // ──────────────── 初始化与清理 ────────────────

    /**
     * @brief 初始化 Vulkan 计算环境
     *
     * 执行步骤：
     * 1. 创建 VkInstance — Vulkan 应用的入口，加载验证层和扩展
     * 2. 枚举物理设备 — 找到支持计算队列的 GPU
     * 3. 创建逻辑设备 — 获取计算队列，指定需要的特性
     * 4. 创建命令池和命令缓冲区 — 用于录制和提交 GPU 命令
     *
     * @return true 初始化成功
     */
    bool initialize();

    /**
     * @brief 释放所有 Vulkan 资源
     *
     * 按创建的反序销毁：buffer → pipeline → descriptor → command → device → instance
     * Vulkan 的资源有依赖关系，必须按正确顺序销毁，否则驱动可能崩溃
     */
    void cleanup();

    // ──────────────── 着色器与管线 ────────────────

    /**
     * @brief 加载 SPIR-V 着色器文件
     * @param spvPath    SPIR-V 文件路径
     * @param entryPoint 着色器入口函数名（默认 "main"）
     * @return true 加载成功
     */
    bool loadShader(const std::string& spvPath,
                    const std::string& entryPoint = "main");

    /**
     * @brief 创建计算管线
     *
     * 管线 (Pipeline) 是 Vulkan 执行计算的核心对象：
     * - 绑定着色器代码
     * - 定义描述符布局（binding 点的个数和类型）
     * - 定义推送常量范围
     *
     * @return true 创建成功
     */
    bool createPipeline();

    // ──────────────── 缓冲区管理 ────────────────

    /**
     * @brief 创建 GPU 缓冲区
     * @param size  缓冲区大小（字节）
     * @param usage 缓冲区用途（VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 等）
     * @return true 创建成功
     *
     * 内存类型选择策略：
     * - 优先选择 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT（CPU 可访问）
     *   这样可以用 memcpy 而不是 vkCmdCopyBuffer 来传输数据
     * - 同时要求 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
     *   这样不需要手动调用 vkInvalidateMapping / vkFlushMapping
     */
    bool createBuffer(size_t size, VkBufferUsageFlags usage);

    /**
     * @brief 将 CPU 数据写入 GPU 缓冲区
     * @param binding binding 编号（对应着色器中的 layout(binding=N)）
     * @param data    源数据指针
     * @param size    数据大小（字节）
     * @return true 写入成功
     *
     * 实现方式：map GPU 内存 → memcpy → unmap
     * 这是零拷贝方式（如果内存是 HOST_VISIBLE 的）
     */
    bool writeBuffer(int binding, const void* data, size_t size);

    /**
     * @brief 从 GPU 缓冲区读取数据到 CPU
     * @param binding binding 编号
     * @param data    目标缓冲区
     * @param size    数据大小（字节）
     * @return true 读取成功
     */
    bool readBuffer(int binding, void* data, size_t size);

    // ──────────────── 执行 ────────────────

    /**
     * @brief 发起计算调度
     * @param groupCountX X 方向的工作组数量
     * @param groupCountY Y 方向的工作组数量（默认 1）
     * @param groupCountZ Z 方向的工作组数量（默认 1）
     * @return true 调度成功
     *
     * dispatch 的含义：
     * - GPU 上启动 groupCountX * groupCountY * groupCountZ 个工作组
     * - 每个工作组包含着色器中 layout(local_size_x, y, z) 定义的调用数
     * - 例如 GEMM：输出 M x N 矩阵，workgroup 16x16
     *   → dispatch((M+15)/16, (N+15)/16, 1)
     */
    bool dispatch(uint32_t groupCountX, uint32_t groupCountY = 1,
                  uint32_t groupCountZ = 1);

private:
    // ──────────────── Vulkan 核心对象 ────────────────
    VkInstance       instance_         = VK_NULL_HANDLE;  ///< Vulkan 实例
    VkPhysicalDevice physicalDevice_   = VK_NULL_HANDLE;  ///< 物理设备（GPU）
    VkDevice         device_           = VK_NULL_HANDLE;  ///< 逻辑设备
    VkQueue          computeQueue_     = VK_NULL_HANDLE;  ///< 计算队列
    VkCommandPool    commandPool_      = VK_NULL_HANDLE;  ///< 命令池
    VkCommandBuffer  commandBuffer_    = VK_NULL_HANDLE;  ///< 命令缓冲区
    VkPipeline       pipeline_         = VK_NULL_HANDLE;  ///< 计算管线
    VkPipelineLayout pipelineLayout_   = VK_NULL_HANDLE;  ///< 管线布局
    VkDescriptorSet  descriptorSet_    = VK_NULL_HANDLE;  ///< 描述符集
    VkDescriptorPool descriptorPool_   = VK_NULL_HANDLE;  ///< 描述符池
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE; ///< 描述符布局

    /// GPU 缓冲区信息：VkBuffer + 对应的 VkDeviceMemory + 大小
    struct BufferInfo {
        VkBuffer     buffer;   ///< 缓冲区对象
        VkDeviceMemory memory;  ///< 绑定的设备内存
        size_t       size;     ///< 缓冲区大小（字节）
    };
    std::vector<BufferInfo> buffers_;  ///< 所有已创建的缓冲区

    uint32_t computeQueueFamilyIndex_ = 0;  ///< 计算队列族索引

    /// 着色器模块（临时，创建 pipeline 后可销毁）
    VkShaderModule shaderModule_ = VK_NULL_HANDLE;

    /// 着色器字节码（loadShader 时暂存）
    std::vector<uint32_t> spirvCode_;

    /// 已创建的缓冲区数量（用作 binding 编号）
    int bufferCount_ = 0;
};

} // namespace edgeai

#endif // ENABLE_VULKAN
