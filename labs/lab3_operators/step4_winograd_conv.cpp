/**
 * @file step4_winograd_conv.cpp
 * @brief Lab3 Step4: Winograd F(2,3) 卷积 —— 用代数变换减少乘法次数
 *
 * ====== 本实验的学习目标 ======
 * 1. 理解 Winograd 卷积的数学原理: 用加减法换乘法
 * 2. 掌握 F(2,3) 变换矩阵及其实际计算
 * 3. 理解 2D Winograd = 1D Winograd 的外积
 * 4. 了解 Winograd 的适用场景和限制
 *
 * ====== Winograd 卷积的核心思想 ======
 *
 * 朴素卷积对每个输出元素需要 kH*kW*inC 次乘法
 * Winograd 通过代数变换，将计算转移到"Winograd 域"
 * 在 Winograd 域中，乘法次数大幅减少
 * 变换本身只有加减法（几乎免费），乘法才是瓶颈
 * → 用少量加法换大量乘法 → 总运算量下降
 *
 * ====== F(m,r) 的含义 ======
 * F(m,r) 表示: 一次计算 m 个输出，使用 r 大小的卷积核
 *
 * 1D 情况 (F(2,3)):
 *   - 朴素卷积: 2 个输出需要 2 * 3 = 6 次乘法
 *   - Winograd: 2 个输出只需要 4 次乘法
 *   - 加法从 4 次增加到 8 次，但加法远比乘法快
 *   - 加速比: 6/4 = 1.5x
 *
 * 2D 情况 (F(2,3) x F(2,3)):
 *   - 朴素卷积: 2x2 个输出需要 2x2 * 3x3 = 36 次乘法
 *   - Winograd: 2x2 个输出只需要 4x4 = 16 次乘法
 *   - 加速比: 36/16 = 2.25x!
 *   - 这就是 Winograd 对 3x3 卷积的理论加速上限
 *
 * ====== F(2,3) 的变换矩阵 ======
 *
 * 标准算法 (1D):
 *   Y = A^T * (G*g ⊙ B^T*d) * A
 *
 * 其中 g 是 3 个卷积核值, d 是 4 个输入值, Y 是 2 个输出值
 *
 * G (kernel transform):       B^T (input transform):     A^T (output transform):
 * | 1    0    0    |          | 1  0 -1  0 |             | 1  1 |
 * | 1/2  1/2  1/2 |          | 0  1  1  0 |             | 0  1 |
 * | 1/2 -1/2  1/2 |          | 0 -1  1  0 |
 * | 0    0    1    |          | 1  0  1  0 |
 *
 * 注意: G 中的 1/2 可以用浮点乘法实现（只做一次，权重变换是预计算的）
 *
 * 变换步骤:
 * 1. 权重变换: U = G * g * G^T   (只做一次，预计算)
 * 2. 输入变换: V = B^T * d * B   (每块输入做一次)
 * 3. 元素级乘: M = U ⊙ V        (16 次乘法，核心计算)
 * 4. 输出逆变换: Y = A^T * M * A (每块输出做一次)
 *
 * 2D 情况: 输入块 4x4, 输出块 2x2
 * - 输入变换: V = B^T * d * B (对行和列各做一次 1D 变换)
 * - 逆变换:   Y = A^T * M * A
 *
 * ====== Winograd 的适用场景和限制 ======
 * - 只支持 stride=1 (Winograd 变换假设输入连续)
 * - 只支持 3x3 卷积 (F(2,3) 的变换矩阵是为 3 阶多项式设计的)
 * - 对小卷积核效果最好 (3x3 是最常见的小卷积核)
 * - 1x1 卷积用 Winograd 没有意义 (本身就是 1 次乘法，无法减少)
 * - Winograd 的数值精度略低于朴素卷积 (变换引入浮点误差)
 * - 大通道数时优势明显 (乘法减少的收益更大)
 *
 * ====== 编译与运行 ======
 * g++ -O2 -std=c++17 -o step4_winograd_conv step4_winograd_conv.cpp
 * ./step4_winograd_conv
 */

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>

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
private:
    std::chrono::steady_clock::time_point start_, stop_;
};

// ============================================================================
// 张量类
// ============================================================================
struct Tensor {
    std::vector<int> shape;
    std::vector<float> data;
    Tensor() = default;
    explicit Tensor(const std::vector<int>& s) : shape(s) {
        int total = 1; for (int d : shape) total *= d;
        data.resize(total, 0.0f);
    }
    int total() const { int t = 1; for (int d : shape) t *= d; return t; }
    float& at(int n, int c, int h, int w) {
        int C = shape[1], H = shape[2], W = shape[3];
        return data[n * C * H * W + c * H * W + h * W + w];
    }
    const float& at(int n, int c, int h, int w) const {
        int C = shape[1], H = shape[2], W = shape[3];
        return data[n * C * H * W + c * H * W + h * W + w];
    }
    void randomize() {
        for (auto& v : data)
            v = -1.0f + 2.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    void zero() { std::fill(data.begin(), data.end(), 0.0f); }
};

// ============================================================================
// FLOPs 计算
// ============================================================================
int64_t flopCountConv2d(int inC, int outC, int kH, int kW,
                        int outH, int outW) {
    int64_t kernelsMulAdd = static_cast<int64_t>(inC) * kH * kW;
    int64_t outputElements = static_cast<int64_t>(outC) * outH * outW;
    return 2 * outputElements * kernelsMulAdd;
}

/**
 * 计算 Winograd F(2,3) 卷积的 FLOPs
 *
 * 对于每个 2x2 输出块 (4 个输出元素):
 * - 输入变换: 4x4 块, 大约 40 次加法 (B^T * d * B)
 * - 元素级乘: 4x4 = 16 次乘法 (U ⊙ V), 跨 inC 通道累加
 * - 输出逆变换: 大约 20 次加法 (A^T * M * A)
 *
 * 考虑通道累加:
 * - 对每对 (oc, ic): 16 次乘法 (4x4 元素级乘)
 * - 乘法总数: outC * inC * 16
 * - 输出块数: (outH/2) * (outW/2)
 * - 总乘法: (outH/2) * (outW/2) * outC * inC * 16
 *
 * 对比朴素: outH * outW * outC * inC * 9
 * 加速比: (9) / (16/4) = 9/4 = 2.25x (只算乘法)
 * 实际 FLOPs 节省: 乘法减少 2.25 倍，但增加了变换的开销
 */
int64_t flopCountWinograd(int inC, int outC, int outH, int outW) {
    // 输出块数 (2x2 分块)
    int64_t nTilesH = (outH + 1) / 2;
    int64_t nTilesW = (outW + 1) / 2;
    int64_t nTiles = nTilesH * nTilesW;

    // 每个块的乘法次数: 4x4 = 16 次 (元素级乘)
    // 每个块需要 inC 个通道累加
    int64_t mulsPerTile = 16LL * inC;

    // 每个输出通道的乘法次数
    int64_t totalMuls = nTiles * outC * mulsPerTile;

    // 变换开销 (加法): 大约 60 次/块/通道 (输入变换 40 + 输出变换 20)
    int64_t addPerTile = 60LL;
    int64_t totalAdds = nTiles * outC * addPerTile;

    // 总 FLOPs (1 乘 = 1 FLOP, 1 加 = 1 FLOP)
    return totalMuls + totalAdds;
}

// ============================================================================
// 朴素卷积 (用于对比和正确性验证)
// ============================================================================
void naiveConv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
                 Tensor& output, int stride, int pad) {
    int N = input.shape[0], inC = input.shape[1];
    int inH = input.shape[2], inW = input.shape[3];
    int outC = weight.shape[0];
    int kH = weight.shape[2], kW = weight.shape[3];
    int outH = (inH + 2 * pad - kH) / stride + 1;
    int outW = (inW + 2 * pad - kW) / stride + 1;

    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < outC; oc++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    float sum = bias.data[oc];
                    for (int ic = 0; ic < inC; ic++) {
                        for (int kh = 0; kh < kH; kh++) {
                            for (int kw = 0; kw < kW; kw++) {
                                int ih = oh * stride + kh - pad;
                                int iw = ow * stride + kw - pad;
                                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                    sum += input.at(n, ic, ih, iw) * weight.at(oc, ic, kh, kw);
                                }
                            }
                        }
                    }
                    output.at(n, oc, oh, ow) = sum;
                }
            }
        }
    }
}

// ============================================================================
// Winograd F(2,3) 变换矩阵
//
// 这些矩阵是根据多项式中国剩余定理 (CRT) 推导出来的
// 不需要自己推导，只需要知道怎么用
//
// 核心思想: 将 3x3 卷积在有限域上的计算，转化为更少的乘法
// 这是代数层面的等价变换，结果在数值上可能有微小差异
// ============================================================================

// G: 卷积核变换矩阵 (4x3)
// 将 3 个卷积核值变换到 Winograd 域的 4 个值
static const float G[4][3] = {
    { 1.0f,     0.0f,     0.0f     },
    { 1.0f/2,   1.0f/2,   1.0f/2   },
    { 1.0f/2,  -1.0f/2,   1.0f/2   },
    { 0.0f,     0.0f,     1.0f     }
};

// BT: 输入变换矩阵 (4x4)
// 将 4 个输入值变换到 Winograd 域的 4 个值
static const float BT[4][4] = {
    { 1.0f,  0.0f, -1.0f,  0.0f },
    { 0.0f,  1.0f,  1.0f,  0.0f },
    { 0.0f, -1.0f,  1.0f,  0.0f },
    { 1.0f,  0.0f,  1.0f,  0.0f }
};

// AT: 输出逆变换矩阵 (2x4)
// 将 Winograd 域的 4 个值变换回 2 个输出值
static const float AT[2][4] = {
    { 1.0f, 1.0f,  1.0f, 0.0f },
    { 0.0f, 1.0f, -1.0f, 1.0f }
};

// ============================================================================
// 1D 矩阵乘法 (小矩阵)
// C[m,n] = A[m,k] * B[k,n]
// 用于 Winograd 变换中的小矩阵运算
// ============================================================================
void matmul1d(const float* A, int ra, int ca,
              const float* B, int rb, int cb,
              float* C) {
    for (int i = 0; i < ra; i++) {
        for (int j = 0; j < cb; j++) {
            float sum = 0.0f;
            for (int p = 0; p < ca; p++) {
                sum += A[i * ca + p] * B[p * cb + j];
            }
            C[i * cb + j] = sum;
        }
    }
}

// ============================================================================
// Winograd F(2,3) 2D 卷积实现
//
// 步骤:
// 1. 预计算权重变换: U[oc][ic] = G * g[oc][ic] * G^T (4x4 矩阵)
// 2. 对每个 4x4 输入块:
//    a. 输入变换: V[ic] = B^T * d[ic] * B (4x4 矩阵)
//    b. 元素级乘: M[oc][ic] = U[oc][ic] ⊙ V[ic]
//    c. 通道累加: S[oc] = sum_ic M[oc][ic]
//    d. 输出逆变换: Y[oc] = A^T * S[oc] * A (2x2 矩阵)
// 3. 加上偏置
// ============================================================================
void winogradConv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
                    Tensor& output, int /* stride */, int pad) {
    // Winograd F(2,3) 只支持 stride=1, kH=kW=3
    int N = input.shape[0], inC = input.shape[1];
    int inH = input.shape[2], inW = input.shape[3];
    int outC = weight.shape[0];
    int outH = output.shape[2];
    int outW = output.shape[3];

    // 输出块数: 2x2 分块
    int nTilesH = (outH + 1) / 2;
    int nTilesW = (outW + 1) / 2;

    // ================================================================
    // Step 1: 预计算权重变换 U[oc][ic] = G * g[oc][ic] * G^T
    //
    // 对每个 (oc, ic) 对, 将 3x3 卷积核变换为 4x4 Winograd 域矩阵
    // 这个变换只需要做一次 (权重不变)，可以预计算
    //
    // 数学: U = G * g * G^T
    //   G: 4x3, g: 3x3, G^T: 3x4
    //   U: 4x4
    //
    // 中间结果: temp = G * g (4x3 * 3x3 = 4x3)
    //           U = temp * G^T (4x3 * 3x4 = 4x4)
    // ================================================================
    std::vector<float> U(outC * inC * 16);  // 4x4 = 16 个元素

    // G^T (3x4)
    float GT[3][4];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            GT[i][j] = G[j][i];

    for (int oc = 0; oc < outC; oc++) {
        for (int ic = 0; ic < inC; ic++) {
            // 提取 3x3 卷积核
            float g[9];
            for (int kh = 0; kh < 3; kh++)
                for (int kw = 0; kw < 3; kw++)
                    g[kh * 3 + kw] = weight.at(oc, ic, kh, kw);

            // temp = G * g (4x3 * 3x3 = 4x3)
            float temp[12];
            matmul1d(&G[0][0], 4, 3, g, 3, 3, temp);

            // U = temp * G^T (4x3 * 3x4 = 4x4)
            float* u = &U[(oc * inC + ic) * 16];
            matmul1d(temp, 4, 3, &GT[0][0], 3, 4, u);
        }
    }

    // ================================================================
    // Step 2-4: 对每个 2x2 输出块计算
    // ================================================================

    // B (4x4) — BT 的转置
    float B[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            B[i][j] = BT[j][i];

    // A (4x2) — AT 的转置
    float A[4][2];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 2; j++)
            A[i][j] = AT[j][i];

    for (int n = 0; n < N; n++) {
        for (int tH = 0; tH < nTilesH; tH++) {
            for (int tW = 0; tW < nTilesW; tW++) {
                // 输出块的起始位置 (2x2)
                int ohStart = tH * 2;
                int owStart = tW * 2;

                // 对应输入块的起始位置 (4x4, 因为 stride=1)
                int ihStart = ohStart - pad;
                int iwStart = owStart - pad;

                // ====================================================
                // Step 2: 输入变换 V[ic] = B^T * d[ic] * B
                //
                // d[ic] 是 4x4 输入块
                // 先做 B^T * d: 对行做 1D 变换
                // 再做结果 * B: 对列做 1D 变换
                // ====================================================
                std::vector<float> V(inC * 16);

                for (int ic = 0; ic < inC; ic++) {
                    // 提取 4x4 输入块 (带边界处理)
                    float d[16];
                    for (int dh = 0; dh < 4; dh++) {
                        for (int dw = 0; dw < 4; dw++) {
                            int ih = ihStart + dh;
                            int iw = iwStart + dw;
                            if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                d[dh * 4 + dw] = input.at(n, ic, ih, iw);
                            } else {
                                d[dh * 4 + dw] = 0.0f;  // zero-padding
                            }
                        }
                    }

                    // V = B^T * d * B
                    // 先: temp = B^T * d (4x4 * 4x4 = 4x4)
                    float temp[16];
                    matmul1d(&BT[0][0], 4, 4, d, 4, 4, temp);
                    // 再: V = temp * B (4x4 * 4x4 = 4x4)
                    matmul1d(temp, 4, 4, &B[0][0], 4, 4, &V[ic * 16]);
                }

                // ====================================================
                // Step 3: 元素级乘 + 通道累加
                //
                // M[oc] = sum_ic (U[oc][ic] ⊙ V[ic])
                //
                // 这是最核心的计算!
                // - 朴素的 3x3 卷积: 每对 (oc,ic) 需要 9 次乘法
                // - Winograd: 每对 (oc,ic) 只需要 16 次乘法 (4x4 元素级乘)
                // - 但 Winograd 一次算 4 个输出 (2x2 块)
                // - 所以每个输出: 16/4 = 4 次乘法 vs 朴素 9 次乘法
                // - 加速比: 9/4 = 2.25x!
                // ====================================================
                std::vector<float> M(outC * 16, 0.0f);

                for (int oc = 0; oc < outC; oc++) {
                    for (int ic = 0; ic < inC; ic++) {
                        const float* u = &U[(oc * inC + ic) * 16];
                        const float* v = &V[ic * 16];
                        float* m = &M[oc * 16];

                        // 元素级乘累加
                        for (int idx = 0; idx < 16; idx++) {
                            m[idx] += u[idx] * v[idx];
                        }
                    }
                }

                // ====================================================
                // Step 4: 输出逆变换 Y[oc] = A^T * M[oc] * A
                //
                // 将 4x4 的 Winograd 域结果变换回 2x2 的输出
                // ====================================================
                for (int oc = 0; oc < outC; oc++) {
                    // Y = A^T * M * A
                    float temp[8];  // 2x4
                    matmul1d(&AT[0][0], 2, 4, &M[oc * 16], 4, 4, temp);
                    float Y[4];  // 2x2
                    matmul1d(temp, 2, 4, &A[0][0], 4, 2, Y);

                    // 写入输出 + 偏置
                    for (int yh = 0; yh < 2; yh++) {
                        for (int yw = 0; yw < 2; yw++) {
                            int oh = ohStart + yh;
                            int ow = owStart + yw;
                            if (oh < outH && ow < outW) {
                                output.at(n, oc, oh, ow) = Y[yh * 2 + yw] + bias.data[oc];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    const int N = 1, inC = 3, outC = 64;
    const int H = 224, W = 224;
    const int kH = 3, kW = 3;
    const int stride = 1, pad = 1;
    const int outH = (H + 2 * pad - kH) / stride + 1;
    const int outW = (W + 2 * pad - kW) / stride + 1;

    printf("=== Lab3 Step4: Winograd F(2,3) 卷积 ===\n");
    printf("问题规模: %dx%dx%d → %dx%dx%d, kernel=%dx%d\n",
           inC, H, W, outC, outH, outW, kH, kW);

    // FLOPs 对比
    int64_t flops_naive = flopCountConv2d(inC, outC, kH, kW, outH, outW);
    int64_t flops_winograd = flopCountWinograd(inC, outC, outH, outW);
    printf("朴素卷积 FLOPs: %lld (%.2fG)\n", (long long)flops_naive, flops_naive / 1e9);
    printf("Winograd FLOPs: %lld (%.2fG)\n", (long long)flops_winograd, flops_winograd / 1e9);
    printf("理论加速比 (乘法减少): %.2fx\n",
           static_cast<double>(flops_naive) / flops_winograd);

    // 变换矩阵打印
    printf("\n--- Winograd F(2,3) 变换矩阵 ---\n");
    printf("G (kernel transform, 4x3):\n");
    for (int i = 0; i < 4; i++) {
        printf("  |");
        for (int j = 0; j < 3; j++) printf(" %7.3f", G[i][j]);
        printf(" |\n");
    }
    printf("B^T (input transform, 4x4):\n");
    for (int i = 0; i < 4; i++) {
        printf("  |");
        for (int j = 0; j < 4; j++) printf(" %5.1f", BT[i][j]);
        printf(" |\n");
    }
    printf("A^T (output transform, 2x4):\n");
    for (int i = 0; i < 2; i++) {
        printf("  |");
        for (int j = 0; j < 4; j++) printf(" %5.1f", AT[i][j]);
        printf(" |\n");
    }

    // 创建张量
    Tensor input({1, inC, H, W});
    Tensor weight({outC, inC, kH, kW});
    Tensor bias({outC});
    Tensor output_naive({1, outC, outH, outW});
    Tensor output_winograd({1, outC, outH, outW});

    srand(42);
    input.randomize();
    weight.randomize();
    for (int i = 0; i < outC; i++) bias.data[i] = 0.01f * (rand() % 100) / 100.0f;

    // ================================================================
    // 性能对比
    // ================================================================
    const int NUM_ITERS = 5;  // Winograd 比较慢 (教学版本)，减少迭代

    // 朴素卷积
    printf("\n--- 朴素卷积 (Baseline) ---\n");
    output_naive.zero();
    naiveConv2d(input, weight, bias, output_naive, stride, pad);

    double totalNaiveMs = 0.0;
    for (int iter = 0; iter < NUM_ITERS; iter++) {
        output_naive.zero();
        Timer t;
        t.start();
        naiveConv2d(input, weight, bias, output_naive, stride, pad);
        t.stop();
        totalNaiveMs += t.elapsedMs();
    }
    double avgNaiveMs = totalNaiveMs / NUM_ITERS;
    double naiveGflops = static_cast<double>(flops_naive) / (avgNaiveMs * 1e6);
    printf("  平均耗时: %.2f ms, GFLOPS: %.2f\n", avgNaiveMs, naiveGflops);

    // Winograd 卷积
    printf("\n--- Winograd F(2,3) 卷积 ---\n");
    output_winograd.zero();
    winogradConv2d(input, weight, bias, output_winograd, stride, pad);

    double totalWinogradMs = 0.0;
    for (int iter = 0; iter < NUM_ITERS; iter++) {
        output_winograd.zero();
        Timer t;
        t.start();
        winogradConv2d(input, weight, bias, output_winograd, stride, pad);
        t.stop();
        totalWinogradMs += t.elapsedMs();
    }
    double avgWinogradMs = totalWinogradMs / NUM_ITERS;
    double winogradGflops = static_cast<double>(flops_naive) / (avgWinogradMs * 1e6);
    printf("  平均耗时: %.2f ms, GFLOPS: %.2f\n", avgWinogradMs, winogradGflops);

    // 加速比
    double speedup = avgNaiveMs / avgWinogradMs;
    printf("\n--- 性能对比 ---\n");
    printf("  加速比: %.2fx\n", speedup);
    printf("  理论加速 (乘法减少): 2.25x\n");
    printf("  实际加速可能低于理论值:\n");
    printf("    1. 本实现的教学版本变换不是最优的\n");
    printf("    2. 实际优化版会用内联汇编/手写变换\n");
    printf("    3. 小通道数时变换开销占比大\n");

    // ================================================================
    // 正确性验证
    // ================================================================
    printf("\n--- 正确性验证 ---\n");
    double maxDiff = 0.0;
    double avgDiff = 0.0;
    int numElements = output_naive.total();
    int numChecked = 0;
    for (int i = 0; i < numElements; i++) {
        double diff = std::fabs(output_naive.data[i] - output_winograd.data[i]);
        maxDiff = std::max(maxDiff, diff);
        avgDiff += diff;
        numChecked++;
    }
    avgDiff /= numChecked;
    printf("  Winograd vs 朴素:\n");
    printf("    最大误差: %.2e\n", maxDiff);
    printf("    平均误差: %.2e\n", avgDiff);
    if (maxDiff < 0.1) {
        printf("    结果验证: PASS (误差在可接受范围内)\n");
    } else {
        printf("    结果验证: WARNING (误差较大)\n");
    }
    printf("\n  注: Winograd 的误差比 SIMD 更大，因为变换引入了额外浮点运算\n");
    printf("      误差量级通常在 1e-4 ~ 1e-3，对于推理通常可接受\n");

    // ================================================================
    // Winograd 何时有优势？分析不同通道数的影响
    // ================================================================
    printf("\n--- Winograd 的适用场景分析 ---\n");
    printf("  Winograd 的开销: 输入变换 (B^T*d*B) + 输出变换 (A^T*M*A)\n");
    printf("  这些开销与通道数无关，是每个块的固定成本\n");
    printf("  乘法减少的收益 = inC * outC * 9/4 次/块\n");
    printf("  → 通道数越多，乘法减少的收益越大\n");
    printf("  → 通道数少时，变换开销占比大，可能反而更慢\n\n");

    printf("  适用场景:\n");
    printf("    3x3 卷积 + stride=1 + 大通道数 → Winograd 最优\n");
    printf("    3x3 卷积 + stride=1 + 小通道数 → SIMD/im2col 可能更好\n");
    printf("    1x1 卷积 → Winograd 无意义 (乘法无法减少)\n");
    printf("    stride>1 → Winograd 不支持\n");
    printf("    5x5/7x7 卷积 → 嵌套 Winograd 可行，但收益递减\n");

    // ================================================================
    // ASCII 对比图
    // ================================================================
    printf("\n--- GFLOPS 对比图 ---\n");
    double maxGf = std::max(naiveGflops, winogradGflops);
    int barW = 50;
    int nBar = static_cast<int>(naiveGflops / maxGf * barW);
    int wBar = static_cast<int>(winogradGflops / maxGf * barW);

    printf("  Naive    |");
    for (int i = 0; i < nBar; i++) printf("#");
    printf(" %.1f\n", naiveGflops);

    printf("  Winograd |");
    for (int i = 0; i < wBar; i++) printf("#");
    printf(" %.1f\n", winogradGflops);

    printf("\n=== Step4 完成 ===\n");
    printf("关键结论: Winograd F(2,3) 理论上减少 2.25x 乘法，对大通道 3x3 卷积最有效\n");
    printf("下一步: step5_gemm_blocked.cpp —— 分块 GEMM，BLAS 底层优化技术\n");

    return 0;
}
