/**
 * @file step2_simd_conv.cpp
 * @brief Lab3 Step2: AVX2 SIMD 卷积 —— im2col + GEMM 方案
 *
 * ====== 本实验的学习目标 ======
 * 1. 理解 im2col 变换：将卷积转换为矩阵乘法 (GEMM)
 * 2. 理解 AVX2 SIMD：一条指令同时处理 8 个 float
 * 3. 理解 FMA (Fused Multiply-Add)：一条指令完成 a*b+c，2 FLOPs/cycle
 * 4. 对比朴素卷积，体验 4-8x 的性能提升
 *
 * ====== im2col (Image to Column) 变换 ======
 *
 * 核心思想：把卷积的"滑动窗口"操作变成矩阵乘法
 *
 * 原始输入 [1, C, H, W]，3x3 卷积，stride=1，pad=0：
 *
 *   输入 5x5 示例 (单通道)：
 *   +---+---+---+---+---+
 *   | a | b | c | d | e |     3x3 卷积滑动窗口：
 *   +---+---+---+---+---+     位置(0,0): [a,b,c, f,g,h, k,l,m]
 *   | f | g | h | i | j |     位置(0,1): [b,c,d, g,h,i, l,m,n]
 *   +---+---+---+---+---+     位置(1,0): [f,g,h, k,l,m, p,q,r]
 *   | k | l | m | n | o |     ...
 *   +---+---+---+---+---+
 *   | p | q | r | s | t |
 *   +---+---+---+---+---+
 *   | u | v | w | x | y |
 *   +---+---+---+---+---+
 *
 *   im2col 后的矩阵 (每行是一个感受野的展开):
 *   +---+---+---+---+---+---+---+---+---+
 *   | a | b | c | f | g | h | k | l | m |  ← 位置(0,0)
 *   +---+---+---+---+---+---+---+---+---+
 *   | b | c | d | g | h | i | l | m | n |  ← 位置(0,1)
 *   +---+---+---+---+---+---+---+---+---+
 *   | ...                               |
 *   +---+---+---+---+---+---+---+---+---+
 *
 *   矩阵维度: [outH*outW, C*kH*kW]
 *   例如: 3x3 输出 → 9 行, 3*3*3=27 列 (3 通道)
 *
 * 权重矩阵 (直接 reshape，无需变换):
 *   [outC, C*kH*kW]
 *   例如: [64, 27]
 *
 * GEMM: [outH*outW, C*kH*kW] × [C*kH*kW, outC] → [outH*outW, outC]
 * 输出再 reshape 为 [1, outC, outH, outW]
 *
 * ====== 为什么 im2col + GEMM 更快？ ======
 * 1. GEMM 是二维循环，访存模式规律，编译器容易优化
 * 2. GEMM 有大量成熟优化：SIMD、分块、多线程...
 * 3. 矩阵乘法是 BLAS 库的核心，经过几十年的极致优化
 * 4. cuDNN、MKL-DNN 等推理框架的核心思路就是 im2col + GEMM
 *
 * ====== im2col 的缺点 ======
 * 1. 内存膨胀: 输入从 N*C*H*W 膨胀到 N*outH*outW*C*kH*kW
 *    - 3x3 卷积: 膨胀 9 倍!
 *    - 7x7 卷积: 膨胀 49 倍!!
 * 2. 额外内存拷贝: im2col 本身需要将数据重新排列
 * 3. 这就是为什么 3x3 卷积更适合用 Winograd (Step4)
 *
 * ====== AVX2 寄存器说明 ======
 * - 256-bit 宽 = 32 字节 = 8 个 float
 * - __m256 类型: 一个 AVX2 寄存器，包含 8 个 float
 * - _mm256_loadu_ps: 加载 8 个非对齐 float (u = unaligned)
 * - _mm256_load_ps:  加载 8 个对齐 float (要求 32 字节对齐)
 * - _mm256_fmadd_ps: FMA, a*b+c 在一条指令中完成
 *   - 为什么 FMA 快？因为乘法和加法在硬件中是同一个流水线
 *   - 而且只有一次舍入，精度也更高
 *
 * ====== 编译与运行 ======
 * g++ -O2 -std=c++17 -mavx2 -mfma -o step2_simd_conv step2_simd_conv.cpp
 * ./step2_simd_conv
 */

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>

// AVX2 头文件
#ifdef USE_AVX2
#include <immintrin.h>
#else
// 如果编译器不支持 AVX2，我们手动检测并在运行时决定
#include <immintrin.h>
#endif

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
// 张量类 (与 Step1 相同)
// ============================================================================
struct Tensor {
    std::vector<int> shape;
    std::vector<float> data;

    Tensor() = default;
    explicit Tensor(const std::vector<int>& s) : shape(s) {
        int total = 1;
        for (int d : shape) total *= d;
        data.resize(total, 0.0f);
    }
    Tensor(const std::vector<int>& s, float val) : shape(s) {
        int total = 1;
        for (int d : shape) total *= d;
        data.resize(total, val);
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
// FLOPs 和内存访问量计算 (与 Step1 相同)
// ============================================================================
int64_t flopCountConv2d(int inC, int outC, int kH, int kW,
                        int outH, int outW, int groups = 1) {
    int64_t kernelsMulAdd = static_cast<int64_t>(inC / groups) * kH * kW;
    int64_t outputElements = static_cast<int64_t>(outC) * outH * outW;
    return 2 * outputElements * kernelsMulAdd;
}

int64_t memSizeConv2d(int inC, int outC, int kH, int kW,
                      int inH, int inW, int groups = 1) {
    int64_t inputBytes  = static_cast<int64_t>(inC) * inH * inW * sizeof(float);
    int64_t weightBytes = static_cast<int64_t>(outC) * (inC / groups) * kH * kW * sizeof(float);
    int64_t outputBytes = static_cast<int64_t>(outC) * inH * inW * sizeof(float);
    return inputBytes + weightBytes + outputBytes;
}

// ============================================================================
// 朴素卷积 (与 Step1 相同，用于性能对比)
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
// im2col 变换 —— 将输入图像展开为矩阵
//
// 输入: input [1, C, H, W]
// 输出: col [outH*outW, C*kH*kW]
//
// 变换规则:
//   col[oh*outW+ow][ic*kH*kW + kh*kW + kw] = input[0][ic][oh*s+kh-p][ow*s+kw-p]
//
// 直观理解:
//   - 输出矩阵的每一行对应输出特征图中的一个空间位置 (oh, ow)
//   - 每行包含该位置感受野内所有通道的所有像素值
//   - 共 C*kH*kW 个值 (C 个通道，每个通道 kH*kW 个像素)
//
// 内存布局:
//   col 按行优先存储，行 = 空间位置，列 = 通道×核大小
//   这种布局使得后续 GEMM 的访存模式最优
// ============================================================================
void im2col(const Tensor& input, float* col,
            int inC, int inH, int inW,
            int kH, int kW, int stride, int pad,
            int outH, int outW) {
    // col 矩阵大小: (outH*outW) x (inC*kH*kW)
    int colCols = inC * kH * kW;  // 每行的列数

    for (int oh = 0; oh < outH; oh++) {
        for (int ow = 0; ow < outW; ow++) {
            // 当前行在 col 矩阵中的偏移
            int rowIdx = oh * outW + ow;
            float* colRow = col + rowIdx * colCols;

            for (int ic = 0; ic < inC; ic++) {
                for (int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++) {
                        int ih = oh * stride + kh - pad;
                        int iw = ow * stride + kw - pad;

                        // col[oh*outW+ow][ic*kH*kW + kh*kW + kw]
                        int colIdx = ic * kH * kW + kh * kW + kw;

                        if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                            colRow[colIdx] = input.at(0, ic, ih, iw);
                        } else {
                            colRow[colIdx] = 0.0f;  // zero-padding 区域填 0
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// 朴素 GEMM —— 作为对照，不含 SIMD
//
// C[M,N] += A[M,K] * B[K,N]
// 三重循环: i (M) → j (N) → p (K)
// 朴素实现中 B 按列访问 (B[p][j] 步长为 N)，缓存不友好
// ============================================================================
void naiveGemm(const float* A, const float* B, float* C,
               int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = C[i * N + j];  // 累加初始值 (偏置)
            for (int p = 0; p < K; p++) {
                sum += A[i * K + p] * B[p * N + j];
                //              ↑ B 按列访问! 每次跳 N 个 float
                //              如果 N 很大，B[p*N+j] 每次都可能 cache miss
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// AVX2 GEMM 微内核 —— 利用 SIMD 加速矩阵乘法
//
// C[M,N] += A[M,K] * B[K,N]
//
// 优化思路:
// 1. 外层循环遍历 i (M 的行)
// 2. 内层循环遍历 p (K 的归约维度)
// 3. 对每个 p，同时处理 8 个 j 值 (AVX2 = 8 个 float)
//    - 一次加载 8 个 B 元素: B[p*N+j..j+7]
//    - 一次加载 8 个 C 元素: C[i*N+j..j+7]
//    - FMA: C[i*N+j..j+7] += A[i*K+p] * B[p*N+j..j+7]
//    - 广播 A[i*K+p] 到 8 个位置，与 B 的 8 个元素相乘
//
// 4. 尾部处理: N 不一定是 8 的倍数，剩余部分用标量处理
//
// 为什么这个顺序比朴素 GEMM 快？
// - 朴素: j 在内层 → B 按列访问 → 缓存不友好
// - AVX2: p 在内层 → B 按行访问 (连续 8 个) → 缓存友好
// - 同时处理 8 个 j → 8 倍数据并行
// ============================================================================
void sgemmAvx2(const float* A, const float* B, float* C,
               int M, int K, int N) {
    // 遍历 A 的每一行 (对应 C 的每一行)
    for (int i = 0; i < M; i++) {
        // C 行的起始指针
        float* cRow = C + i * N;
        // A 行的起始指针
        const float* aRow = A + i * K;

        // 遍历 K 维度 (归约维度)
        for (int p = 0; p < K; p++) {
            // 广播 A[i][p] 到 AVX2 寄存器的 8 个位置
            // _mm256_set1_ps(v): 将 v 复制 8 份填满 256-bit 寄存器
            __m256 aVal = _mm256_set1_ps(aRow[p]);

            // AVX2 主循环: 每次处理 8 个 j
            int j = 0;
            for (; j + 7 < N; j += 8) {
                // 加载 B[p][j..j+7]: B 的连续 8 个元素
                // _mm256_loadu_ps: 加载非对齐的 8 个 float
                // "u" = unaligned, 不要求地址 32 字节对齐
                __m256 bVal = _mm256_loadu_ps(&B[p * N + j]);

                // 加载 C[i][j..j+7]: C 的连续 8 个元素
                __m256 cVal = _mm256_loadu_ps(&cRow[j]);

                // FMA: cVal = aVal * bVal + cVal
                // _mm256_fmadd_ps(a, b, c) = a * b + c
                // 这条指令在硬件中只占 1 个时钟周期 (FMA 单元)
                // 但完成了 8 次乘法 + 8 次加法 = 16 FLOPs!
                __m256 result = _mm256_fmadd_ps(aVal, bVal, cVal);

                // 存回 C[i][j..j+7]
                _mm256_storeu_ps(&cRow[j], result);
            }

            // 尾部处理: N 不是 8 的倍数，剩余 1-7 个元素用标量处理
            for (; j < N; j++) {
                cRow[j] += aRow[p] * B[p * N + j];
            }
        }
    }
}

// ============================================================================
// im2col + AVX2 GEMM 卷积
//
// 步骤:
// 1. 对输入做 im2col 变换: [1,C,H,W] → [outH*outW, C*kH*kW]
// 2. 权重 reshape 为矩阵: [outC, C*kH*kW] (无需变换，内存布局天然匹配)
// 3. GEMM: [outH*outW, C*kH*kW] × [C*kH*kW, outC] → [outH*outW, outC]
// 4. 加上偏置
// 5. reshape 结果: [outH*outW, outC] → [1, outC, outH, outW]
// ============================================================================
void simdConv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
                Tensor& output, int stride, int pad) {
    int inC = input.shape[1], inH = input.shape[2], inW = input.shape[3];
    int outC = weight.shape[0];
    int kH = weight.shape[2], kW = weight.shape[3];
    int outH = (inH + 2 * pad - kH) / stride + 1;
    int outW = (inW + 2 * pad - kW) / stride + 1;

    // ---------------------------------------------------------------
    // Step 1: im2col 变换
    // 输入: [1, 3, 224, 224]
    // 输出: [224*224, 3*3*3] = [50176, 27]
    // 内存: 50176 * 27 * 4 = 5.4 MB
    // 这就是 im2col 的内存膨胀！原始输入只有 3*224*224*4 = 0.6 MB
    // ---------------------------------------------------------------
    int M = outH * outW;            // GEMM 的 M 维度
    int K = inC * kH * kW;          // GEMM 的 K 维度
    int N = outC;                   // GEMM 的 N 维度

    std::vector<float> colData(M * K);  // im2col 矩阵
    im2col(input, colData.data(), inC, inH, inW, kH, kW, stride, pad, outH, outW);

    // ---------------------------------------------------------------
    // Step 2: 权重已经是 [outC, C*kH*kW] 布局，可以直接用作 B 矩阵
    // 因为 Tensor::data 的存储顺序是 [outC][inC][kH][kW]
    // 展平后就是 [outC, inC*kH*kW]，正好是 GEMM 需要的 B^T
    // 但注意: GEMM 定义是 A[M,K] * B[K,N]
    // 这里: A = col [M,K], B = weight 的转置 [K,N]
    // 我们的权重是 [outC, inC*kH*kW] = [N, K]
    // 所以实际做的是 col[M,K] * weight^T[K,N]
    // ============================================================
    // 为了简化，我们重新排列权重为 B[K,N] = [C*kH*kW, outC]
    // ---------------------------------------------------------------
    std::vector<float> bData(K * N);
    for (int oc = 0; oc < outC; oc++) {
        for (int ik = 0; ik < K; ik++) {
            // B[ik][oc] = weight[oc][ik 的三维展开]
            bData[ik * N + oc] = weight.data[oc * K + ik];
        }
    }

    // ---------------------------------------------------------------
    // Step 3: GEMM: C[M,N] = A[M,K] * B[K,N]
    // 输出: [50176, 64]
    // ---------------------------------------------------------------

    // 先将偏置填入 C 矩阵的每一行
    std::vector<float> cData(M * N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            cData[i * N + j] = bias.data[j];  // 每行的初始值 = 偏置
        }
    }

    // 执行 AVX2 GEMM
    sgemmAvx2(colData.data(), bData.data(), cData.data(), M, K, N);

    // ---------------------------------------------------------------
    // Step 4: 将 GEMM 结果 reshape 为 [1, outC, outH, outW]
    // GEMM 输出 cData[i][j] 中:
    //   i = oh*outW + ow (空间位置)
    //   j = oc (输出通道)
    // 需要转换为 output[0][oc][oh][ow]
    // ---------------------------------------------------------------
    for (int oh = 0; oh < outH; oh++) {
        for (int ow = 0; ow < outW; ow++) {
            int i = oh * outW + ow;
            for (int oc = 0; oc < outC; oc++) {
                output.at(0, oc, oh, ow) = cData[i * N + oc];
            }
        }
    }
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    // 问题规模 (与 Step1 相同)
    const int N = 1, inC = 3, outC = 64;
    const int H = 224, W = 224;
    const int kH = 3, kW = 3;
    const int stride = 1, pad = 1;
    const int outH = (H + 2 * pad - kH) / stride + 1;
    const int outW = (W + 2 * pad - kW) / stride + 1;

    printf("=== Lab3 Step2: AVX2 SIMD 卷积 (im2col + GEMM) ===\n");
    printf("问题规模: %dx%dx%d → %dx%dx%d, kernel=%dx%d\n",
           inC, H, W, outC, outH, outW, kH, kW);

    int64_t flops = flopCountConv2d(inC, outC, kH, kW, outH, outW);
    printf("FLOPs: %lld (%.2fG)\n", (long long)flops, flops / 1e9);

    // 创建张量
    Tensor input({1, inC, H, W});
    Tensor weight({outC, inC, kH, kW});
    Tensor bias({outC});
    Tensor output_naive({1, outC, outH, outW});
    Tensor output_simd({1, outC, outH, outW});

    srand(42);
    input.randomize();
    weight.randomize();
    for (int i = 0; i < outC; i++) bias.data[i] = 0.01f * (rand() % 100) / 100.0f;

    // ================================================================
    // 性能对比: 朴素卷积 vs AVX2 GEMM 卷积
    // ================================================================
    const int NUM_ITERS = 10;

    // --- 朴素卷积 ---
    printf("\n--- 朴素卷积 (Step1 Baseline) ---\n");
    output_naive.zero();
    naiveConv2d(input, weight, bias, output_naive, stride, pad);  // 预热

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
    double naiveGflops = static_cast<double>(flops) / (avgNaiveMs * 1e6);
    printf("  平均耗时: %.2f ms\n", avgNaiveMs);
    printf("  GFLOPS: %.2f\n", naiveGflops);

    // --- AVX2 GEMM 卷积 ---
    printf("\n--- AVX2 GEMM 卷积 ---\n");

    // 先单独测量 im2col 的耗时
    int M = outH * outW;
    int K = inC * kH * kW;
    std::vector<float> colData(M * K);
    double totalIm2colMs = 0.0;
    for (int iter = 0; iter < NUM_ITERS; iter++) {
        Timer t;
        t.start();
        im2col(input, colData.data(), inC, H, W, kH, kW, stride, pad, outH, outW);
        t.stop();
        totalIm2colMs += t.elapsedMs();
    }
    double avgIm2colMs = totalIm2colMs / NUM_ITERS;
    printf("  im2col 耗时: %.2f ms (内存膨胀: %.1f MB → %.1f MB)\n",
           avgIm2colMs,
           static_cast<double>(inC * H * W * sizeof(float)) / (1024 * 1024),
           static_cast<double>(M * K * sizeof(float)) / (1024 * 1024));

    // 测量完整的 SIMD 卷积 (im2col + GEMM)
    output_simd.zero();
    simdConv2d(input, weight, bias, output_simd, stride, pad);  // 预热

    double totalSimdMs = 0.0;
    for (int iter = 0; iter < NUM_ITERS; iter++) {
        output_simd.zero();
        Timer t;
        t.start();
        simdConv2d(input, weight, bias, output_simd, stride, pad);
        t.stop();
        totalSimdMs += t.elapsedMs();
    }
    double avgSimdMs = totalSimdMs / NUM_ITERS;
    double simdGflops = static_cast<double>(flops) / (avgSimdMs * 1e6);
    printf("  总耗时: %.2f ms\n", avgSimdMs);
    printf("  GFLOPS: %.2f\n", simdGflops);

    // ================================================================
    // 加速比分析
    // ================================================================
    double speedup = avgNaiveMs / avgSimdMs;
    printf("\n--- 性能对比 ---\n");
    printf("  加速比: %.2fx\n", speedup);
    printf("  AVX2 理论加速: 8x (8 个 float 并行)\n");
    printf("  实际加速低于理论值的原因:\n");
    printf("    1. im2col 内存膨胀带来额外开销\n");
    printf("    2. GEMM 内层循环仍有内存带宽瓶颈\n");
    printf("    3. 尾部标量处理的开销\n");
    printf("    4. 权重重排 (transpose) 的开销\n");

    // ================================================================
    // ASCII 柱状图对比
    // ================================================================
    printf("\n--- GFLOPS 对比图 ---\n");
    double maxGflops = std::max(naiveGflops, simdGflops);
    int barWidth = 50;
    int naiveBar = static_cast<int>(naiveGflops / maxGflops * barWidth);
    int simdBar  = static_cast<int>(simdGflops / maxGflops * barWidth);

    printf("  Naive  |");
    for (int i = 0; i < naiveBar; i++) printf("#");
    printf(" %.1f GFLOPS\n", naiveGflops);

    printf("  AVX2   |");
    for (int i = 0; i < simdBar; i++) printf("#");
    printf(" %.1f GFLOPS\n", simdGflops);

    // ================================================================
    // 正确性验证 —— 对比 AVX2 卷积与朴素卷积的输出
    // 由于浮点运算顺序不同，结果可能有微小差异 (1e-4 级别)
    // 这是正常的！不同计算顺序的浮点结果本就不同
    // ================================================================
    printf("\n--- 正确性验证 ---\n");
    double maxDiff = 0.0;
    double avgDiff = 0.0;
    int numElements = output_naive.total();
    for (int i = 0; i < numElements; i++) {
        double diff = std::fabs(output_naive.data[i] - output_simd.data[i]);
        maxDiff = std::max(maxDiff, diff);
        avgDiff += diff;
    }
    avgDiff /= numElements;
    printf("  与朴素卷积对比:\n");
    printf("    最大误差: %.2e\n", maxDiff);
    printf("    平均误差: %.2e\n", avgDiff);
    if (maxDiff < 1e-2) {
        printf("    结果验证: PASS (误差在可接受范围内)\n");
    } else {
        printf("    结果验证: WARNING (误差较大，请检查实现)\n");
    }
    printf("\n  注: 浮点误差来源: SIMD 改变了累加顺序，导致舍入方向不同\n");
    printf("      这不是 bug，是浮点运算的固有特性\n");

    printf("\n=== Step2 完成 ===\n");
    printf("下一步: step3_cache_tiled_conv.cpp —— 优化缓存利用率\n");

    return 0;
}
