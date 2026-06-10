/**
 * @file VulkanConv2d.h
 * @brief Vulkan GPU 加速的 2D 卷积算子（im2col + GEMM 方案）
 *
 * ====== 学习要点 ======
 * 1. GPU 上怎么做卷积？
 *    - 和 CPU 的 im2col + GEMM 思路完全一样，只是计算从 CPU 搬到 GPU
 *    - 步骤：CPU 上做 im2col → 传到 GPU → GPU 做 GEMM → 传回 CPU → reshape
 *    - 但更好的方案是把 im2col 也放在 GPU 上做！
 *    - 本实现使用 compute shader 同时完成 im2col 和 GEMM
 *
 * 2. 为什么要在 GPU 上做 im2col？
 *    - im2col 本身就是逐元素复制，非常适合并行化
 *    - 如果在 CPU 上做 im2col，结果还要传到 GPU → 额外传输开销
 *    - GPU im2col = 在 GPU 上展开输入，无需 CPU-GPU 传输
 *    - 而且 GPU 的内存带宽远高于 CPU → im2col 在 GPU 上更快
 *
 * 3. GPU 卷积的两种策略：
 *    a. 分步法：先 GPU im2col，再 GPU GEMM
 *       - 需要两个 compute shader 或一个分两步的 shader
 *       - im2col 产生临时矩阵（内存消耗大）
 *       - 实现简单，复用已有的 GEMM shader
 *
 *    b. 融合法：一个 shader 完成整个卷积
 *       - 不产生中间矩阵，节省显存
 *       - 每个线程直接读取输入像素并累加到输出
 *       - 实现复杂，但性能可能更好（减少全局内存访问）
 *       - 本实现使用融合法（参见 conv2d.comp.glsl）
 *
 * 4. push constants 的作用：
 *    - 卷积参数（维度、stride、padding 等）需要传给 GPU shader
 *    - 有三种方式：uniform buffer、storage buffer、push constant
 *    - push constant 最快：数据直接存在命令缓冲区中，不需要额外的 buffer
 *    - 限制：最大 128 字节（通常够用）
 *    - 适合传递少量参数（如卷积的维度信息）
 *
 * 5. 与 NCNN Vulkan 后端的关系：
 *    - NCNN 的 vulkan 计算也是类似的架构
 *    - 但 NCNN 做了更多优化：权重预变换、管线缓存、内存池等
 *    - 本实现是简化版，重点在于理解 GPU 卷积的基本流程
 */

#pragma once

#include "IOperator.h"
#include <vector>
#include <string>

#ifdef ENABLE_VULKAN

#include "VulkanComputeBase.h"

namespace edgeai {

class VulkanConv2d : public IOperator {
public:
    /**
     * @brief 构造 Vulkan 卷积算子
     * @param inC      输入通道数
     * @param outC     输出通道数
     * @param kH       卷积核高度
     * @param kW       卷积核宽度
     * @param stride   步长
     * @param pad      填充
     * @param groups   分组数
     * @param spvPath  卷积着色器的 SPIR-V 文件路径
     */
    VulkanConv2d(int inC, int outC, int kH, int kW,
                  int stride = 1, int pad = 0, int groups = 1,
                  const std::string& spvPath = "shaders/conv2d.comp.spv");

    void forward(const Tensor& input, Tensor& output) override;
    const char* name() const override { return "VulkanConv2d"; }
    int64_t flops(const std::vector<int>& inputShape,
                  const std::vector<int>& outputShape) const override;

private:
    int inC_, outC_, kH_, kW_, stride_, pad_, groups_;
    std::string spvPath_;
    VulkanComputeBase vkBase_;
    Tensor weight_;    ///< 卷积核权重 [outC, inC/groups, kH, kW]
    Tensor bias_;      ///< 偏置 [outC]
    bool initialized_ = false;
};

} // namespace edgeai

#endif // ENABLE_VULKAN
