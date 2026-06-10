/**
 * @file Conv2dNaive.h
 * @brief 朴素卷积实现 —— 6 层嵌套循环的 O(N^6) 基线版本
 *
 * ====== 学习要点 ======
 * 1. 卷积的数学定义：
 *    输出 out[n][oc][oh][ow] = sum over ic,kh,kw of
 *        input[n][ic][oh*s+kh][ow*s+kw] * weight[oc][ic][kh][kw] + bias[oc]
 *
 *    其中：
 *    - N = batch size, ic = 输入通道, oc = 输出通道
 *    - kH/kW = 卷积核大小, s = stride, p = padding
 *    - oh = (H + 2*p - kH) / s + 1,  ow 同理
 *
 * 2. 为什么这是 O(N^6)？
 *    循环层次：N * outC * outH * outW * inC * kH * kW = 7 层嵌套
 *    但通常把 kH*kW 算作一个维度（卷积核元素数），所以常说 O(N^6)
 *    对于 3x3 卷积、224x224 输入、256 通道，计算量巨大
 *
 * 3. 为什么朴素实现慢？
 *    - 缓存不友好：内层循环在 input 上跳跃式访问，空间局部性差
 *    - 无法利用 SIMD：每次只算 1 个乘加，AVX2 可以同时算 8 个
 *    - 循环开销大：7 层循环本身的分支预测和迭代变量更新也是开销
 *
 * 4. 这个实现的价值：
 *    - 作为性能基线（baseline），验证优化版本的正确性
 *    - 理解卷积的数学定义，后续优化都是在此基础上的等价变换
 *    - 对比朴素 vs 优化的 GFLOPS，直观感受优化的效果
 *
 * 5. groups 参数的含义：
 *    - groups=1：标准卷积，每个输出通道看所有输入通道
 *    - groups=inC：深度可分离卷积 (Depthwise)，每个输出通道只看 1 个输入通道
 *    - 1 < groups < inC：分组卷积，将通道分组，组内独立卷积
 *    - groups 改变了通道维度的映射：ic_per_group = inC/groups, oc_per_group = outC/groups
 */

#pragma once

#include "IOperator.h"
#include <vector>

namespace edgeai {

class Conv2dNaive : public IOperator {
public:
    /**
     * @brief 构造朴素卷积算子
     * @param inC    输入通道数
     * @param outC   输出通道数
     * @param kH     卷积核高度
     * @param kW     卷积核宽度
     * @param stride 步长（默认 1）
     * @param pad    填充（默认 0）
     * @param groups 分组数（默认 1，即标准卷积）
     */
    Conv2dNaive(int inC, int outC, int kH, int kW,
                int stride = 1, int pad = 0, int groups = 1);

    void forward(const Tensor& input, Tensor& output) override;
    const char* name() const override { return "Conv2dNaive"; }
    int64_t flops(const std::vector<int>& inputShape,
                  const std::vector<int>& outputShape) const override;

private:
    int inC_, outC_, kH_, kW_, stride_, pad_, groups_;
    Tensor weight_;   ///< 卷积核权重 [outC, inC/groups, kH, kW]
    Tensor bias_;     ///< 偏置 [outC]
};

} // namespace edgeai
