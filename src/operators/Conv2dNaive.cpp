/**
 * @file Conv2dNaive.cpp
 * @brief 朴素卷积实现 —— 6 层嵌套循环的 O(N^6) 基线版本
 *
 * ====== 实现说明 ======
 * 1. 本实现严格遵循卷积的数学定义，没有任何优化技巧
 * 2. 通过与优化版本对比，可以验证正确性（结果应完全一致）
 * 3. 通过与优化版本对比 GFLOPS，可以直观感受优化的效果
 * 4. 权重使用随机初始化，实际应用中应从模型文件加载
 */

#include "operators/Conv2dNaive.h"
#include <cassert>
#include <cmath>

namespace edgeai {

Conv2dNaive::Conv2dNaive(int inC, int outC, int kH, int kW,
                         int stride, int pad, int groups)
    : inC_(inC), outC_(outC), kH_(kH), kW_(kW),
      stride_(stride), pad_(pad), groups_(groups)
{
    /*
     * ====== 参数校验 ======
     * 1. 通道数必须能被 groups 整除
     *    - inC/groups = 每组的输入通道数
     *    - outC/groups = 每组的输出通道数
     *    - 如果不能整除，分组卷积就无法正确映射通道
     *
     * 2. 常见分组卷积场景：
     *    - groups=1：标准卷积，如 ResNet 的 3x3 conv
     *    - groups=inC=outC：深度可分离卷积 (Depthwise)，如 MobileNet
     *    - groups=32, inC=outC=128：分组卷积，如 ShuffleNet
     */
    assert(inC % groups == 0 && "inC must be divisible by groups");
    assert(outC % groups == 0 && "outC must be divisible by groups");

    /*
     * ====== 权重和偏置初始化 ======
     * 权重形状：[outC, inC/groups, kH, kW]
     * - 每个输出通道有 inC/groups 个卷积核（每个 kH x kW）
     * - groups>1 时，每个组只看 inC/groups 个输入通道
     * - 所以权重的第二维是 inC/groups 而非 inC
     *
     * 偏置形状：[outC]
     * - 每个输出通道一个偏置值
     *
     * 初始化方式：Xavier 均匀初始化
     * - 范围：[-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))]
     * - fan_in = inC/groups * kH * kW（每个输出神经元接收的输入数）
     * - fan_out = outC/groups * kH * kW（每个输入神经元连接的输出数）
     * - Xavier 初始化使前向传播时信号方差不变，有助于训练收敛
     * - 本项目只做推理，初始化方式不影响正确性，但影响数值范围
     */
    int icPerGroup = inC / groups;
    weight_ = Tensor({outC, icPerGroup, kH, kW});
    bias_ = Tensor({outC});

    // Xavier 均匀初始化
    float fanIn = static_cast<float>(icPerGroup * kH * kW);
    float fanOut = static_cast<float>((outC / groups) * kH * kW);
    float limit = std::sqrt(6.0f / (fanIn + fanOut));
    weight_.randomize(-limit, limit);
    bias_.zero();
}

void Conv2dNaive::forward(const Tensor& input, Tensor& output)
{
    /*
     * ====== 输入形状解析 ======
     * 输入：[N, inC, H, W]
     * - N = batch size（一次推理多少张图）
     * - inC = 输入通道数（RGB 图为 3，feature map 为 256 等）
     * - H, W = 输入特征图的高度和宽度
     */
    assert(input.dims() == 4);
    int N = input.size(0);
    int H = input.size(2);
    int W = input.size(3);

    /*
     * ====== 输出尺寸计算 ======
     * outH = (H + 2*pad - kH) / stride + 1
     * outW = (W + 2*pad - kW) / stride + 1
     *
     * 推导：
     * - 输入加上 padding 后的有效高度 = H + 2*pad
     * - 卷积核从第一个位置滑到最后一个位置，共滑动 (H+2*pad-kH) 个像素
     * - 每次滑动 stride 个像素，所以输出尺寸 = 滑动次数 + 1
     * - 即 (H+2*pad-kH)/stride + 1
     *
     * 举例：输入 32x32，3x3 卷积，stride=1，pad=1
     * outH = (32 + 2*1 - 3) / 1 + 1 = 32（尺寸不变，常见配置）
     */
    int outH = (H + 2 * pad_ - kH_) / stride_ + 1;
    int outW = (W + 2 * pad_ - kW_) / stride_ + 1;

    // 分组卷积参数
    int icPerGroup = inC_ / groups_;
    int ocPerGroup = outC_ / groups_;

    // 确保输出张量形状正确
    output = Tensor({N, outC_, outH, outW});

    /*
     * ====== 7 层嵌套循环 —— 朴素卷积的核心 ======
     *
     * 循环顺序（从外到内）：
     * 1. n    : batch 维度 — 不同图片之间独立
     * 2. g    : group 维度 — 不同组之间独立
     * 3. oc   : 输出通道维度 — 不同输出通道之间独立
     * 4. oh/ow: 输出空间维度 — 不同输出位置之间独立
     * 5. ic   : 输入通道维度 — 需要累加（不独立！）
     * 6. kh/kw: 卷积核空间维度 — 需要累加（不独立！）
     *
     * 为什么这个顺序？
     * - 外层循环（n, g, oc, oh, ow）各次迭代互相独立，理论上可以并行
     * - 内层循环（ic, kh, kw）需要累加到同一个输出元素，不能并行
     * - 这就是 OpenMP 并行化的切入点：#pragma omp parallel for 外层循环
     *
     * 为什么慢？
     * - 内层循环访问 input[n][ic][oh*s+kh][ow*s+kw]
     * - 当 ic 变化时，input 上的访问跳跃 C*H*W 个 float（整个通道）
     * - 当 kh/kw 变化时，跳跃 H*W 或 W 个 float
     * - 这些跳跃导致缓存行无法复用，每次都可能 cache miss
     */
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < groups_; ++g) {
            for (int oc = 0; oc < ocPerGroup; ++oc) {
                // 全局输出通道索引 = g * ocPerGroup + oc
                int ocGlobal = g * ocPerGroup + oc;
                for (int oh = 0; oh < outH; ++oh) {
                    for (int ow = 0; ow < outW; ++ow) {
                        // 累加变量 —— 初始化为偏置值
                        float sum = bias_[ocGlobal];

                        // 内层循环：在输入通道和卷积核空间维度上累加
                        for (int ic = 0; ic < icPerGroup; ++ic) {
                            // 全局输入通道索引
                            int icGlobal = g * icPerGroup + ic;
                            for (int kh = 0; kh < kH_; ++kh) {
                                for (int kw = 0; kw < kW_; ++kw) {
                                    /*
                                     * 输入坐标计算：
                                     * ih = oh * stride + kh - pad
                                     * iw = ow * stride + kw - pad
                                     *
                                     * padding 的处理：
                                     * - 如果 ih < 0 或 ih >= H，说明在 padding 区域，值为 0
                                     * - 如果 iw < 0 或 iw >= W，同上
                                     * - 只有在有效范围内的输入才参与计算
                                     *
                                     * 这就是 "valid" vs "same" padding 的区别：
                                     * - pad=0 (valid)：输出尺寸小于输入
                                     * - pad=kH/2 (same)：输出尺寸等于输入（kH 为奇数时）
                                     */
                                    int ih = oh * stride_ + kh - pad_;
                                    int iw = ow * stride_ + kw - pad_;

                                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                        float inputValue = input.at(n, icGlobal, ih, iw);
                                        float weightValue = weight_.at(ocGlobal, ic, kh, kw);
                                        sum += inputValue * weightValue;
                                    }
                                    // 否则输入值为 0（padding 区域），跳过
                                }
                            }
                        }

                        output.at(n, ocGlobal, oh, ow) = sum;
                    }
                }
            }
        }
    }
}

int64_t Conv2dNaive::flops(const std::vector<int>& inputShape,
                            const std::vector<int>& outputShape) const
{
    /*
     * ====== FLOPs 计算公式 ======
     *
     * 对于每个输出元素：
     *   - 需要 kH * kW * (inC/groups) 次乘法
     *   - 需要 kH * kW * (inC/groups) - 1 次加法
     *   - 加上 1 次偏置加法
     *   - 总计约 2 * kH * kW * (inC/groups) FLOPs（乘加对算 2 FLOPs）
     *
     * 总 FLOPs = N * outC * outH * outW * 2 * kH * kW * (inC/groups)
     *          = outputElements * 2 * kernelElements * channelsPerGroup
     *
     * 为什么 1 次 MAC = 2 FLOPs？
     * - MAC = Multiply-ACcumulate = 1 次乘法 + 1 次加法
     * - FLOPs 计数按浮点运算次数算，乘法和加法各算 1 次
     * - 所以 1 MAC = 2 FLOPs
     * - 这是性能分析中的标准约定
     *
     * 举例：3x3 卷积，256 输入通道，112x112 输出
     * - 每个 output 元素：3*3*256 = 2304 MAC = 4608 FLOPs
     * - 总 output 元素：256*112*112 = 3,211,264
     * - 总 FLOPs ≈ 14.8 GFLOPs
     */
    if (inputShape.size() < 4 || outputShape.size() < 4) return 0;

    int64_t N    = inputShape[0];
    int64_t outH = outputShape[2];
    int64_t outW = outputShape[3];
    int64_t icPerGroup = inC_ / groups_;

    // 每个输出元素的 MAC 次数 = kH * kW * icPerGroup
    int64_t macsPerElement = static_cast<int64_t>(kH_) * kW_ * icPerGroup;

    // 总输出元素数
    int64_t outputElements = N * outC_ * outH * outW;

    // 1 MAC = 2 FLOPs（1 次乘 + 1 次加）
    return outputElements * macsPerElement * 2;
}

} // namespace edgeai
