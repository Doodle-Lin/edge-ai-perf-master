/**
 * @file Conv2dAvx2.h
 * @brief AVX2 SIMD 优化的卷积实现 —— im2col + GEMM 方案
 *
 * ====== 学习要点 ======
 * 1. 为什么要把卷积变成矩阵乘法 (GEMM)？
 *    - 卷积的 7 层嵌套循环中，最内层 3 层 (ic, kH, kW) 是乘加归约
 *    - 如果把输入 "展开" 成矩阵，卷积就变成了矩阵乘法
 *    - GEMM 是线性代数中最成熟的操作，有大量优化技术可以直接应用
 *    - 这是 cuDNN、MKL-DNN 等推理框架的核心思路
 *
 * 2. im2col (image to column) 是什么？
 *    - 原始输入：[N, C, H, W]，卷积核在 H/W 维度上滑动
 *    - im2col 将每个"感受野"（kernel 对应的输入块）展成一行
 *    - 变换后：输入矩阵 [N*outH*outW, C*kH*kW]
 *    - 权重矩阵：[outC, C*kH*kW]（直接 reshape，无需变换）
 *    - 矩阵乘法：[N*outH*outW, C*kH*kW] x [C*kH*kW, outC] → [N*outH*outW, outC]
 *    - 再 reshape 回 [N, outC, outH, outW] 即为输出
 *
 *    举例：3x3 卷积，输入 5x5，stride=1，pad=0
 *    - 每个 3x3 输入块被展成 1 行 9 列
 *    - 输出 3x3，共 9 个位置 → im2col 矩阵 9 行 x 9 列
 *    - 权重矩阵：outC 行 x 9 列
 *    - GEMM 得到 9 行 x outC 列 → reshape 为 outC x 3 x 3
 *
 * 3. im2col 的缺点：
 *    - 内存膨胀：输入矩阵大小从 N*C*H*W 膨胀到 N*outH*outW * C*kH*kW
 *    - 对于大卷积核或大 feature map，im2col 矩阵可能非常大
 *    - 这就是为什么 3x3 卷积更适合用 Winograd（无内存膨胀）
 *
 * 4. AVX2 如何加速 GEMM？
 *    - AVX2 寄存器宽 256 bit = 32 byte = 8 个 float
 *    - _mm256_loadu_ps：一次加载 8 个连续 float（不需要对齐）
 *    - _mm256_fmadd_ps：一条指令同时完成乘法和加法 (a*b+c)，即 FMA
 *    - 相比朴素循环每次处理 1 个 float，AVX2 每次处理 8 个 → 理论加速 8 倍
 *    - 实际加速通常 4-6 倍，因为还有内存带宽瓶颈
 *
 * 5. #ifdef USE_AVX2 的作用：
 *    - 编译器通过 CMake 检测 CPU 是否支持 AVX2
 *    - 如果不支持，整个文件不会被编译（只有 #ifdef 内的代码）
 *    - 这样同一份代码可以在不同 CPU 上编译运行
 *    - 这就是"编译期特性检测"，比运行时检测更安全（不会因非法指令崩溃）
 */

#pragma once

#include "IOperator.h"
#include <vector>

namespace edgeai {

#ifdef USE_AVX2

class Conv2dAvx2 : public IOperator {
public:
    /**
     * @brief 构造 AVX2 优化卷积算子
     * @param inC    输入通道数
     * @param outC   输出通道数
     * @param kH     卷积核高度
     * @param kW     卷积核宽度
     * @param stride 步长（默认 1）
     * @param pad    填充（默认 0）
     * @param groups 分组数（默认 1）
     */
    Conv2dAvx2(int inC, int outC, int kH, int kW,
               int stride = 1, int pad = 0, int groups = 1);

    void forward(const Tensor& input, Tensor& output) override;
    const char* name() const override { return "Conv2dAvx2"; }
    int64_t flops(const std::vector<int>& inputShape,
                  const std::vector<int>& outputShape) const override;

private:
    int inC_, outC_, kH_, kW_, stride_, pad_, groups_;
    Tensor weight_;   ///< 卷积核权重 [outC, inC/groups, kH, kW]
    Tensor bias_;     ///< 偏置 [outC]

    /**
     * @brief im2col 变换 —— 将输入图像展开为矩阵
     *
     * 将输入张量 [N, C, H, W] 变换为矩阵 [N*outH*outW, C*kH*kW]
     * 每一行对应输出中的一个空间位置，包含该位置的感受野内所有像素
     *
     * @param input   输入张量 [1, C, H, W]（batch 维在外层循环处理）
     * @param col     输出矩阵 [outH*outW, C*kH*kW]
     * @param outH    输出高度
     * @param outW    输出宽度
     */
    void im2col(const Tensor& input, Tensor& col, int outH, int outW);

    /**
     * @brief AVX2 优化的 GEMM 微内核
     *
     * 计算 C[M,N] += A[M,K] * B[K,N]
     * 内层循环用 _mm256_fmadd_ps 同时处理 8 个元素
     *
     * @param A     左矩阵，M x K
     * @param B     右矩阵，K x N
     * @param C     结果矩阵，M x N
     * @param M     左矩阵行数
     * @param K     左矩阵列数 = 右矩阵行数
     * @param N     右矩阵列数
     */
    void sgemmAvx2(const float* A, const float* B, float* C,
                   int M, int K, int N);
};

#endif // USE_AVX2

} // namespace edgeai
