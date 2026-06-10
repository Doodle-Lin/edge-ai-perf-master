/**
 * @file Conv2dWinograd.cpp
 * @brief Winograd F(2,3) 卷积实现 —— 代数变换减少乘法次数
 *
 * ====== 实现说明 ======
 * 1. F(2,3) Winograd 将 3x3 卷积的乘法次数减少 2.25 倍
 * 2. 变换矩阵硬编码，避免运行时计算
 * 3. 权重在初始化时预变换，减少推理时的计算量
 * 4. 仅支持 stride=1, kernel=3x3 的标准卷积
 */

#include "operators/Conv2dWinograd.h"
#include <cassert>
#include <cmath>

namespace edgeai {

/*
 * ====== Winograd F(2,3) 变换矩阵 ======
 *
 * 这些矩阵是 Winograd 算法的核心，它们将卷积计算
 * 从"空间域"变换到"Winograd 域"，在 Winograd 域中
 * 乘法次数更少。
 *
 * 变换矩阵来自 Winograd 的最小滤波器算法：
 * Y = A^T * [ (G g) ⊙ (B^T d) ] * A
 *
 * 其中：
 * - g 是 3 个卷积核参数 (r=3)
 * - d 是 4 个输入参数 (m+r-1=4, m=2)
 * - Y 是 2 个输出参数 (m=2)
 * - ⊙ 是逐元素乘法
 * - G, B^T, A^T 是变换矩阵
 *
 * 这些矩阵是唯一确定的（给定多项式插值点），不需要推导
 * 详细的数学推导参见：Fast Algorithms for Convolutional Neural Networks (Lavin & Gray, 2016)
 */

/**
 * G 矩阵 (4x3)：卷积核变换矩阵
 * 将 3 个卷积核参数变换到 Winograd 域（4 个参数）
 *
 * G = | 1    0    0   |
 *     | 1/2  1/2  1/2 |
 *     | 1/2 -1/2  1/2 |
 *     | 0    0    1   |
 *
 * 注意：1/2 可以用乘法实现（x * 0.5f），比除法快
 * 在 ARM 上甚至可以用右移 1 位（VFP 的一半精度）
 */
static constexpr float G[4][3] = {
    {1.0f,     0.0f,     0.0f    },
    {0.5f,     0.5f,     0.5f    },
    {0.5f,    -0.5f,     0.5f    },
    {0.0f,     0.0f,     1.0f    }
};

/**
 * B^T 矩阵 (4x4)：输入变换矩阵
 * 将 4 个输入参数变换到 Winograd 域
 *
 * B^T = | 1  0 -1  0 |
 *       | 0  1  1  0 |
 *       | 0 -1  1  0 |
 *       | 1  0  1  0 |
 *
 * 注意：B^T 中只有 0, 1, -1，所以变换只需要加减法
 * 这就是为什么输入变换"几乎免费"
 */
static constexpr float BT[4][4] = {
    {1.0f,  0.0f, -1.0f,  0.0f},
    {0.0f,  1.0f,  1.0f,  0.0f},
    {0.0f, -1.0f,  1.0f,  0.0f},
    {1.0f,  0.0f,  1.0f,  0.0f}
};

/**
 * A^T 矩阵 (2x4)：输出逆变换矩阵
 * 将 Winograd 域的 4 个参数逆变换回 2 个输出
 *
 * A^T = | 1  1  1  0 |
 *       | 0  1 -1  1 |
 *
 * 注意：A^T 中也只有 0, 1, -1，逆变换也只需要加减法
 * 而且只有 2 行 x 4 列，计算量更少
 */
static constexpr float AT[2][4] = {
    {1.0f,  1.0f,  1.0f,  0.0f},
    {0.0f,  1.0f, -1.0f,  1.0f}
};

Conv2dWinograd::Conv2dWinograd(int inC, int outC, int stride, int pad, int groups)
    : inC_(inC), outC_(outC), stride_(stride), pad_(pad), groups_(groups)
{
    /*
     * ====== Winograd 的限制 ======
     * 1. 只支持 stride=1
     *    - stride>1 时输入采样不连续，Winograd 变换不适用
     *    - 如果需要 stride=2 的 3x3 卷积，可以：
     *      a. 先 stride=1 Winograd，再下采样（计算量太大）
     *      b. 用 im2col + GEMM（推荐）
     *
     * 2. 只支持 3x3 卷积核
     *    - F(2,3) 的变换矩阵是为 3 阶多项式设计的
     *    - 其他核大小需要不同的变换矩阵（如 F(2,5)）
     *    - 1x1 卷积本身就是逐元素乘法，Winograd 无意义
     */
    assert(stride == 1 && "Winograd F(2,3) only supports stride=1");
    assert(inC % groups == 0 && "inC must be divisible by groups");
    assert(outC % groups == 0 && "outC must be divisible by groups");

    int icPerGroup = inC / groups;

    // 原始权重 [outC, inC/groups, 3, 3]
    Tensor weightOrig({outC, icPerGroup, 3, 3});
    float fanIn = static_cast<float>(icPerGroup * 9);
    float fanOut = static_cast<float>((outC / groups) * 9);
    float limit = std::sqrt(6.0f / (fanIn + fanOut));
    weightOrig.randomize(-limit, limit);

    bias_ = Tensor({outC});
    bias_.zero();

    /*
     * ====== 权重预变换 ======
     * U = G * g * G^T
     *
     * 这个变换只需要做一次！因为权重在推理过程中不变
     * 预变换将 3x3 权重变为 4x4 Winograd 域权重
     *
     * 为什么预变换很重要？
     * - 推理可能执行数百万次，每次都变换权重太浪费
     * - 预变换只在模型加载时做一次，之后永远用变换后的权重
     * - 这是"离线计算"的典型例子：将不变的运算提前完成
     */
    weight_ = Tensor({outC, icPerGroup, 4, 4});
    transformWeight(weightOrig, weight_);
}

void Conv2dWinograd::transformWeight(const Tensor& weightOrig, Tensor& weightTrans)
{
    /*
     * ====== 权重变换：U = G * g * G^T ======
     *
     * 步骤：
     * 1. 对 3x3 权重 g 做 G 变换：temp = G * g（4x3 矩阵乘 3x3 矩阵 = 4x3 矩阵）
     * 2. 对 temp 做 G^T 变换：U = temp * G^T（4x3 矩阵乘 3x4 矩阵 = 4x4 矩阵）
     *
     * 等价于：U = G * g * G^T
     *
     * 变换结果：
     * - 原始权重是 3x3（9 个参数）
     * - 变换后权重是 4x4（16 个参数）
     * - 参数数增加了，但 2D Winograd 的乘法次数从 36 减少到 16
     */
    int icPerGroup = inC_ / groups_;
    int ocPerGroup = outC_ / groups_;

    for (int oc = 0; oc < outC_; ++oc) {
        for (int ic = 0; ic < icPerGroup; ++ic) {
            // 步骤 1：temp = G * g (4x3 * 3x3 = 4x3)
            float temp[4][3] = {};
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 3; ++j) {
                    for (int k = 0; k < 3; ++k) {
                        temp[i][j] += G[i][k] * weightOrig.at(oc, ic, k, j);
                    }
                }
            }

            // 步骤 2：U = temp * G^T (4x3 * 3x4 = 4x4)
            // 注意：G^T 就是 G 的转置
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < 3; ++k) {
                        sum += temp[i][k] * G[j][k];  // G^T[k][j] = G[j][k]
                    }
                    weightTrans.at(oc, ic, i, j) = sum;
                }
            }
        }
    }
}

void Conv2dWinograd::transformInput(const float* input, float* output)
{
    /*
     * ====== 输入变换：V = B^T * d * B ======
     *
     * 输入 d 是 4x4 的输入块（4 个连续行 x 4 个连续列）
     * 步骤：
     * 1. temp = d * B（4x4 矩阵乘 4x4 矩阵，但 B 的元素只有 0,1,-1）
     * 2. V = B^T * temp（4x4 矩阵乘 4x4 矩阵，B^T 的元素也只有 0,1,-1）
     *
     * 关键观察：B 和 B^T 中只有 0, 1, -1
     * 所以整个变换只需要加减法，没有乘法！
     * 这就是为什么 Winograd 的变换开销很小
     *
     * 具体到 B^T：
     * B^T[0] = [1, 0, -1, 0] → temp[0] = d[0] - d[2]
     * B^T[1] = [0, 1,  1, 0] → temp[1] = d[1] + d[2]
     * B^T[2] = [0,-1,  1, 0] → temp[2] = -d[1] + d[2]
     * B^T[3] = [1, 0,  1, 0] → temp[3] = d[0] + d[2]
     */
    float temp[4][4] = {};

    // temp = d * B^T（对输入的每一行做 B^T 变换）
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                temp[i][j] += input[i * 4 + k] * BT[j][k];
            }
        }
    }

    // V = B^T * temp（对 temp 的每一列做 B^T 变换）
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += BT[i][k] * temp[k][j];
            }
            output[i * 4 + j] = sum;
        }
    }
}

void Conv2dWinograd::transformOutput(const float* input, float* output)
{
    /*
     * ====== 输出逆变换：Y = A^T * M * A ======
     *
     * 输入 M 是 4x4 的 Winograd 域结果（元素级乘法后的结果）
     * 输出 Y 是 2x2 的最终卷积输出
     *
     * 步骤：
     * 1. temp = M * A（4x4 矩阵乘 4x2 矩阵 = 4x2 矩阵）
     * 2. Y = A^T * temp（2x4 矩阵乘 4x2 矩阵 = 2x2 矩阵）
     *
     * A^T 中也只有 0, 1, -1，所以逆变换也只需要加减法
     *
     * 具体：
     * A^T[0] = [1, 1, 1, 0] → Y[0] = M[0] + M[1] + M[2]
     * A^T[1] = [0, 1,-1, 1] → Y[1] = M[1] - M[2] + M[3]
     */
    float temp[4][2] = {};

    // temp = M * A（4x4 乘 4x2）
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 4; ++k) {
                temp[i][j] += input[i * 4 + k] * AT[j][k];
            }
        }
    }

    // Y = A^T * temp（2x4 乘 4x2）
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += AT[i][k] * temp[k][j];
            }
            output[i * 2 + j] = sum;
        }
    }
}

void Conv2dWinograd::forward(const Tensor& input, Tensor& output)
{
    assert(input.dims() == 4);
    int N = input.size(0);
    int H = input.size(2);
    int W = input.size(3);

    /*
     * ====== 输出尺寸计算 ======
     * F(2,3) 每次产出 2x2 的输出块
     * 所以 outH 和 outW 必须是偶数（或 padding 到偶数）
     *
     * 如果 outH/outW 不是偶数，需要 padding 一行/列
     * 这里简化处理：假设 outH 和 outW 是偶数
     * 实际工程中需要处理奇数情况（填充 0）
     */
    int outH = (H + 2 * pad_ - 3) / stride_ + 1;  // kH=kW=3 硬编码
    int outW = (W + 2 * pad_ - 3) / stride_ + 1;

    int icPerGroup = inC_ / groups_;
    int ocPerGroup = outC_ / groups_;

    output = Tensor({N, outC_, outH, outW});
    output.zero();

    /*
     * ====== Winograd 卷积主循环 ======
     *
     * 整体流程（对于每个 2x2 输出块）：
     * 1. 提取 4x4 输入块
     * 2. 对输入块做 Winograd 变换：V = B^T * d * B
     * 3. 元素级乘法：M = U ⊙ V（U 是预变换的权重）
     * 4. 逆变换：Y = A^T * M * A
     *
     * 其中步骤 3 是唯一有乘法的步骤
     * 朴素 3x3 卷积：2x2 输出需要 2*2*3*3 = 36 次乘法
     * Winograd F(2,3)：2x2 输出需要 4*4 = 16 次乘法
     * 减少 36/16 = 2.25 倍！
     */
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < groups_; ++g) {
            int icGlobalStart = g * icPerGroup;
            int ocGlobalStart = g * ocPerGroup;

            // 遍历 2x2 输出块
            // 步长为 2，因为每个块产生 2x2 的输出
            for (int ohStart = 0; ohStart < outH; ohStart += 2) {
                for (int owStart = 0; owStart < outW; owStart += 2) {

                    // 对每个输出通道，累加所有输入通道的贡献
                    for (int oc = 0; oc < ocPerGroup; ++oc) {
                        int ocGlobal = ocGlobalStart + oc;

                        // 用于累加的 4x4 Winograd 域结果
                        float M[4][4] = {};

                        for (int ic = 0; ic < icPerGroup; ++ic) {
                            int icGlobal = icGlobalStart + ic;

                            /*
                             * 步骤 1：提取 4x4 输入块
                             *
                             * F(2,3) 需要 4 个输入来产生 2 个输出
                             * 所以 2D 情况需要 4x4 输入块
                             *
                             * 输入块起始位置 = 输出块起始位置 - padding
                             * 因为 stride=1，输入块的行列与输出块对齐
                             */
                            float inputBlock[4][4] = {};
                            for (int i = 0; i < 4; ++i) {
                                for (int j = 0; j < 4; ++j) {
                                    int ih = ohStart + i - pad_;
                                    int iw = owStart + j - pad_;
                                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                        inputBlock[i][j] = input.at(n, icGlobal, ih, iw);
                                    }
                                    // padding 区域为 0
                                }
                            }

                            // 步骤 2：输入变换 V = B^T * d * B
                            float V[16] = {};
                            transformInput(&inputBlock[0][0], V);

                            // 步骤 3：元素级乘法 M += U ⊙ V
                            // U 是预变换的权重，V 是变换后的输入
                            // 注意：这是逐元素乘法（Hadamard 积），不是矩阵乘法
                            for (int i = 0; i < 4; ++i) {
                                for (int j = 0; j < 4; ++j) {
                                    float u = weight_.at(ocGlobal, ic, i, j);
                                    M[i][j] += u * V[i * 4 + j];
                                }
                            }
                        }

                        // 步骤 4：逆变换 Y = A^T * M * A
                        float Y[4] = {};
                        transformOutput(&M[0][0], Y);

                        // 写入输出 + 偏置
                        float b = bias_[ocGlobal];
                        for (int i = 0; i < 2; ++i) {
                            for (int j = 0; j < 2; ++j) {
                                int oh = ohStart + i;
                                int ow = owStart + j;
                                if (oh < outH && ow < outW) {
                                    output.at(n, ocGlobal, oh, ow) = Y[i * 2 + j] + b;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

int64_t Conv2dWinograd::flops(const std::vector<int>& inputShape,
                                const std::vector<int>& outputShape) const
{
    /*
     * ====== Winograd 的 FLOPs 计算 ======
     *
     * 朴素 3x3 卷积：每个输出元素需要 3*3*icPerGroup 次 MAC = 2*9*icPerGroup FLOPs
     *
     * Winograd F(2,3)：每个 2x2 输出块需要：
     * - 输入变换：~32 次加法（B^T 只有 0,1,-1）
     * - 元素级乘法：4*4 = 16 次乘法，每个输入通道
     * - 输出逆变换：~12 次加法
     * - 合计：16 次乘法 + ~44 次加法 per input channel per 2x2 block
     *
     * 简化计算（只计乘法，加法相对可以忽略）：
     * - 每个输出元素：16 / 4 = 4 次乘法（每个 2x2 块有 4 个输出元素）
     * - vs 朴素：9 次乘法
     * - 乘法减少比：9/4 = 2.25x
     *
     * 总 FLOPs ≈ N * outC * outH * outW * 2 * 4 * (inC/groups)
     *           = outputElements * 8 * icPerGroup
     *           （只算乘法的 FLOPs，加法开销约占 10-15%）
     *
     * 实际上变换的加法开销大约增加 10-15% 的总 FLOPs
     * 但加法比乘法快 3-5 倍，所以实际时间减少更接近 2x
     */
    if (inputShape.size() < 4 || outputShape.size() < 4) return 0;

    int64_t N    = inputShape[0];
    int64_t outH = outputShape[2];
    int64_t outW = outputShape[3];
    int64_t icPerGroup = inC_ / groups_;

    // Naive FLOPs for 3x3 convolution
    int64_t naiveFlops = N * outC_ * outH * outW * 2 * 9 * icPerGroup;

    // Winograd F(2,3) reduces multiplications to 16/36 of naive
    // 16/36 = 4/9
    return naiveFlops * 4 / 9;
}

} // namespace edgeai
