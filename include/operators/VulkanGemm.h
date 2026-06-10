/**
 * @file VulkanGemm.h
 * @brief Vulkan GPU 加速的矩阵乘法算子
 *
 * ====== 学习要点 ======
 * 1. 为什么用 GPU 做 GEMM？
 *    - CPU 8 核 @ 4GHz：峰值 ~256 GFLOPS (AVX2)
 *    - GPU 1024 核 @ 1.5GHz：峰值 ~3000 GFLOPS (FP32)
 *    - GEMM 是高度并行的：M*N 个输出元素互相独立
 *    - 理论上 GPU 可以同时计算 M*N 个元素（如果核数足够）
 *    - 这就是为什么训练大模型必须用 GPU
 *
 * 2. GPU GEMM 的线程映射：
 *    - 输出矩阵 C[M,N] 的每个元素由一个 GPU 线程计算
 *    - 线程组织：workgroup (16, 16, 1) → 每个工作组计算 16x16 的输出块
 *    - 总调度：dispatch((M+15)/16, (N+15)/16, 1)
 *    - 例如 M=1024, N=1024：dispatch(64, 64, 1) = 4096 个工作组
 *    - GPU 1024 核，每个核执行 ~4 个工作组 → 并行度极高
 *
 * 3. CPU vs GPU 的权衡：
 *    - CPU 优势：低延迟（小矩阵）、缓存大、分支预测
 *    - GPU 优势：高吞吐（大矩阵）、内存带宽大（HBM vs DDR）
 *    - 交叉点：通常在矩阵维度 256-512 之间
 *    - 小于交叉点：CPU 更快（GPU 的调度和数据传输开销占主导）
 *    - 大于交叉点：GPU 更快（计算量大到可以摊薄固定开销）
 *
 * 4. 本实现的简化之处：
 *    - 使用简单 GEMM 着色器（无共享内存分块）→ 适合学习理解
 *    - 实际生产级 GEMM 着色器会用 shared memory 做 tiling
 *    - 每个线程从全局内存读 K 个 A 元素和 K 个 B 元素 → 内存带宽是瓶颈
 *    - 优化版：用 shared memory 让一个 workgroup 共享 A/B 数据 → 减少 K 倍全局读取
 *
 * 5. Vulkan GEMM 的数据流：
 *    CPU 内存 → writeBuffer → GPU 内存 → dispatch → GPU 内存 → readBuffer → CPU 内存
 *    整个流程中的瓶颈通常是：
 *    - 小矩阵：CPU↔GPU 传输
 *    - 大矩阵：GPU 全局内存带宽
 *    - 超大矩阵：GPU 计算能力
 */

#pragma once

#include "IOperator.h"
#include <vector>
#include <string>

#ifdef ENABLE_VULKAN

#include "VulkanComputeBase.h"

namespace edgeai {

class VulkanGemm : public IOperator {
public:
    /**
     * @brief 构造 Vulkan GEMM 算子
     * @param M        左矩阵行数
     * @param N        右矩阵列数
     * @param K        左矩阵列数 = 右矩阵行数
     * @param spvPath  GEMM 计算着色器的 SPIR-V 文件路径
     */
    VulkanGemm(int M, int N, int K, const std::string& spvPath = "shaders/gemm.comp.spv");

    void forward(const Tensor& input, Tensor& output) override;
    const char* name() const override { return "VulkanGemm"; }
    int64_t flops(const std::vector<int>& inputShape,
                  const std::vector<int>& outputShape) const override;

private:
    int M_, N_, K_;
    std::string spvPath_;             ///< 着色器文件路径
    VulkanComputeBase vkBase_;        ///< Vulkan 计算基类
    Tensor weightB_;                  ///< B 矩阵 [K, N]
    bool initialized_ = false;        ///< 是否已初始化 Vulkan
};

} // namespace edgeai

#endif // ENABLE_VULKAN
