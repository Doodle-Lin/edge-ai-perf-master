/**
 * @file Conv2dWinograd.h
 * @brief Winograd F(2,3) 卷积 —— 用代数变换减少乘法次数
 *
 * ====== 学习要点 ======
 * 1. Winograd 卷积的核心思想：
 *    - 朴素卷积对每个输出元素需要 kH*kW*inC 次乘法
 *    - Winograd 通过代数变换，将计算转移到"Winograd 域"
 *    - 在 Winograd 域中，乘法次数大幅减少
 *    - 变换本身只有加减法（几乎免费），乘法才是瓶颈
 *    - 用少量加法换大量乘法 → 总运算量下降
 *
 * 2. F(2,3) 的含义：
 *    - F(m, r) 表示：一次计算 m 个输出，使用 r 大小的卷积核
 *    - F(2,3)：一次计算 2 个输出值，使用 3x3 卷积核
 *    - 朴素卷积：2 个输出需要 2 * 3 = 6 次乘法（1D 情况）
 *    - Winograd F(2,3)：2 个输出只需要 4 次乘法
 *    - 加法从 4 次增加到 8 次，但加法远比乘法快
 *    - 2D 情况：6x6 = 36 次乘法 → 4x4 = 16 次乘法，减少 2.25 倍！
 *
 * 3. Winograd F(2,3) 的变换矩阵（1D 情况）：
 *    输入变换：B^T * d（d 是 4 个输入值）
 *    Y = A^T * (G * g ⊙ B^T * d) * A
 *    其中 g 是 3 个卷积核值，d 是 4 个输入值，Y 是 2 个输出值
 *
 *    G (kernel transform):    B^T (input transform):   A^T (output transform):
 *    | 1    0    0   |        | 1  0 -1  0 |            | 1  1 |
 *    | 1/2  1/2  1/2 |        | 0  1  1  0 |            | 0  1 |
 *    | 1/2 -1/2  1/2 |        | 0 -1  1  0 |
 *    | 0    0    1   |        | 1  0  1  0 |
 *
 *    注意：G 中的 1/2 可以用移位实现，避免浮点除法
 *
 * 4. 2D Winograd = 1D Winograd 的外积：
 *    - 先对行做 1D Winograd，再对列做 1D Winograd
 *    - 输入块大小：4x4（因为 F(2,3) 需要 r+m-1 = 4 个输入）
 *    - 输出块大小：2x2
 *    - 元素级乘法次数：4x4 = 16 次（vs 朴素的 2x2*3x3 = 36 次）
 *
 * 5. Winograd 的适用场景和限制：
 *    - 只支持 stride=1（因为 Winograd 变换假设输入连续）
 *    - 只支持 3x3 卷积（F(2,3) 的变换矩阵是为 3 阶多项式设计的）
 *    - 对小卷积核效果最好（3x3 是最常见的小卷积核）
 *    - 1x1 卷积用 Winograd 没有意义（本身就是 1 次乘法，无法减少）
 *    - 大卷积核（5x5, 7x7）可以用嵌套 Winograd，但收益递减
 *
 * 6. 实际应用：
 *    - cuDNN 的 Winograd 实现是 NVIDIA GPU 上 3x3 卷积的最快方案
 *    - NCNN 的 Winograd 实现在 ARM CPU 上也有显著加速
 *    - 但 Winograd 的数值精度略低于朴素卷积（变换引入浮点误差）
 *    - 对于推理通常可以接受，训练时需要注意
 */

#pragma once

#include "IOperator.h"
#include <vector>

namespace edgeai {

class Conv2dWinograd : public IOperator {
public:
    /**
     * @brief 构造 Winograd F(2,3) 卷积算子
     * @param inC    输入通道数
     * @param outC   输出通道数
     * @param stride 步长（必须为 1，Winograd 只支持 stride=1）
     * @param pad    填充（默认 0）
     * @param groups 分组数（默认 1）
     *
     * 注意：不提供 kH/kW 参数，因为 F(2,3) 只支持 3x3 卷积核
     */
    Conv2dWinograd(int inC, int outC, int stride = 1, int pad = 0, int groups = 1);

    void forward(const Tensor& input, Tensor& output) override;
    const char* name() const override { return "Conv2dWinograd"; }
    int64_t flops(const std::vector<int>& inputShape,
                  const std::vector<int>& outputShape) const override;

private:
    int inC_, outC_, stride_, pad_, groups_;
    Tensor weight_;   ///< 卷积核权重 [outC, inC/groups, 3, 3]
    Tensor bias_;     ///< 偏置 [outC]

    /**
     * @brief 对权重做 Winograd 变换
     *
     * 将 3x3 卷积核变换到 Winograd 域：U = G g G^T
     * 变换后的权重是 4x4 矩阵（而非 3x3）
     * 这个变换只需要做一次（权重不变），可以预计算
     *
     * @param weightOrig 原始 3x3 权重 [outC, inC/groups, 3, 3]
     * @param weightTrans 变换后的权重 [outC, inC/groups, 4, 4]
     */
    void transformWeight(const Tensor& weightOrig, Tensor& weightTrans);

    /**
     * @brief 对输入块做 Winograd 变换
     *
     * 将 4x4 输入块变换到 Winograd 域：V = B^T d B
     *
     * @param input 输入 4x4 块的 16 个值
     * @param output 变换后的 4x4 块的 16 个值
     */
    void transformInput(const float* input, float* output);

    /**
     * @brief 对结果做 Winograd 逆变换
     *
     * 将 4x4 Winograd 域结果变换回 2x2 输出：Y = A^T M A
     *
     * @param input Winograd 域的 4x4 块
     * @param output 输出的 2x2 块
     */
    void transformOutput(const float* input, float* output);
};

} // namespace edgeai
