/**
 * @file step8_vulkan_conv2d.cpp
 * @brief Lab3 Step8: Vulkan Conv2D —— 在 GPU 上做卷积
 *
 * ====== 本实验的学习目标 ======
 * 1. 理解 GPU 上做卷积的两种策略: 分步法 vs 融合法
 * 2. 实现 im2col + GEMM 的 GPU 卷积
 * 3. 对比所有 CPU 卷积实现和 GPU 卷积的性能
 * 4. 生成完整的性能对比表格
 *
 * ====== GPU 卷积的两种策略 ======
 *
 * 策略 A: 分步法 (im2col on GPU + GEMM on GPU)
 *   1. 在 GPU 上执行 im2col 变换 (compute shader)
 *   2. 在 GPU 上执行 GEMM (compute shader)
 *   3. reshape 结果 (在 CPU 上或 GPU 上)
 *   优点: 复用已有的 GEMM shader, 实现简单
 *   缺点: im2col 产生临时矩阵, 内存消耗大
 *
 * 策略 B: 融合法 (一个 shader 完成整个卷积)
 *   - 每个线程直接读取输入像素并累加到输出
 *   - 不产生中间矩阵, 节省显存
 *   - 实现复杂, 但性能可能更好
 *   - 本实验重点分析融合法
 *
 * ====== 融合法的卷积着色器 (GLSL) ======
 *
 * #version 450
 * layout(local_size_x = 16, local_size_y = 16) in;
 * layout(binding = 0) readonly buffer Input_buf { float input[]; };
 * layout(binding = 1) readonly buffer Weight_buf { float weight[]; };
 * layout(binding = 2) writeonly buffer Output_buf { float output[]; };
 * layout(push_constant) uniform PC {
 *     int inC, outC, inH, inW, outH, outW, kH, kW, stride, pad;
 * } pc;
 *
 * void main() {
 *     int ow = int(gl_GlobalInvocationID.x);
 *     int oh = int(gl_GlobalInvocationID.y);
 *     int oc = int(gl_GlobalInvocationID.z);  // 需要修改 dispatch 维度
 *     if (ow >= pc.outW || oh >= pc.outH || oc >= pc.outC) return;
 *
 *     float sum = 0.0;
 *     for (int ic = 0; ic < pc.inC; ic++) {
 *         for (int kh = 0; kh < pc.kH; kh++) {
 *             for (int kw = 0; kw < pc.kW; kw++) {
 *                 int ih = oh * pc.stride + kh - pc.pad;
 *                 int iw = ow * pc.stride + kw - pc.pad;
 *                 if (ih >= 0 && ih < pc.inH && iw >= 0 && iw < pc.inW) {
 *                     int inIdx = ic * pc.inH * pc.inW + ih * pc.inW + iw;
 *                     int wtIdx = oc * pc.inC * pc.kH * pc.kW
 *                               + ic * pc.kH * pc.kW + kh * pc.kW + kw;
 *                     sum += input[inIdx] * weight[wtIdx];
 *                 }
 *             }
 *         }
 *     }
 *     output[oc * pc.outH * pc.outW + oh * pc.outW + ow] = sum;
 * }
 *
 * ====== 编译与运行 ======
 * g++ -O2 -std=c++17 -mavx2 -mfma -o step8_vulkan_conv2d step8_vulkan_conv2d.cpp
 * ./step8_vulkan_conv2d
 */

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

#include <immintrin.h>

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
int64_t flopCountConv2d(int inC, int outC, int kH, int kW, int outH, int outW) {
    return 2LL * outC * outH * outW * inC * kH * kW;
}

// ============================================================================
// 1. 朴素卷积 (6 层嵌套循环, Baseline)
// ============================================================================
void naiveConv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
                 Tensor& output, int stride, int pad) {
    int N = input.shape[0], inC = input.shape[1];
    int inH = input.shape[2], inW = input.shape[3];
    int outC = weight.shape[0];
    int kH = weight.shape[2], kW = weight.shape[3];
    int outH = (inH + 2 * pad - kH) / stride + 1;
    int outW = (inW + 2 * pad - kW) / stride + 1;

    for (int n = 0; n < N; n++)
        for (int oc = 0; oc < outC; oc++)
            for (int oh = 0; oh < outH; oh++)
                for (int ow = 0; ow < outW; ow++) {
                    float sum = bias.data[oc];
                    for (int ic = 0; ic < inC; ic++)
                        for (int kh = 0; kh < kH; kh++)
                            for (int kw = 0; kw < kW; kw++) {
                                int ih = oh * stride + kh - pad;
                                int iw = ow * stride + kw - pad;
                                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                                    sum += input.at(n, ic, ih, iw) * weight.at(oc, ic, kh, kw);
                            }
                    output.at(n, oc, oh, ow) = sum;
                }
}

// ============================================================================
// 2. AVX2 GEMM 微内核
// ============================================================================
void sgemmAvx2(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        float* cRow = C + i * N;
        const float* aRow = A + i * K;
        for (int p = 0; p < K; p++) {
            __m256 aVal = _mm256_set1_ps(aRow[p]);
            int j = 0;
            for (; j + 7 < N; j += 8) {
                __m256 bVal = _mm256_loadu_ps(&B[p * N + j]);
                __m256 cVal = _mm256_loadu_ps(&cRow[j]);
                _mm256_storeu_ps(&cRow[j], _mm256_fmadd_ps(aVal, bVal, cVal));
            }
            for (; j < N; j++) cRow[j] += aRow[p] * B[p * N + j];
        }
    }
}

// ============================================================================
// 2. im2col 变换
// ============================================================================
void im2col(const Tensor& input, float* col,
            int inC, int inH, int inW,
            int kH, int kW, int stride, int pad,
            int outH, int outW) {
    int colCols = inC * kH * kW;
    for (int oh = 0; oh < outH; oh++)
        for (int ow = 0; ow < outW; ow++) {
            int rowIdx = oh * outW + ow;
            float* colRow = col + rowIdx * colCols;
            for (int ic = 0; ic < inC; ic++)
                for (int kh = 0; kh < kH; kh++)
                    for (int kw = 0; kw < kW; kw++) {
                        int ih = oh * stride + kh - pad;
                        int iw = ow * stride + kw - pad;
                        colRow[ic * kH * kW + kh * kW + kw] =
                            (ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                            ? input.at(0, ic, ih, iw) : 0.0f;
                    }
        }
}

// ============================================================================
// 2. AVX2 卷积 (im2col + GEMM)
// ============================================================================
void avx2Conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
                Tensor& output, int stride, int pad) {
    int inC = input.shape[1], inH = input.shape[2], inW = input.shape[3];
    int outC = weight.shape[0], kH = weight.shape[2], kW = weight.shape[3];
    int outH = (inH + 2 * pad - kH) / stride + 1;
    int outW = (inW + 2 * pad - kW) / stride + 1;

    int M = outH * outW, K = inC * kH * kW, N = outC;
    std::vector<float> colData(M * K);
    im2col(input, colData.data(), inC, inH, inW, kH, kW, stride, pad, outH, outW);

    std::vector<float> bData(K * N);
    for (int oc = 0; oc < outC; oc++)
        for (int ik = 0; ik < K; ik++)
            bData[ik * N + oc] = weight.data[oc * K + ik];

    std::vector<float> cData(M * N);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            cData[i * N + j] = bias.data[j];

    sgemmAvx2(colData.data(), bData.data(), cData.data(), M, K, N);

    for (int oh = 0; oh < outH; oh++)
        for (int ow = 0; ow < outW; ow++)
            for (int oc = 0; oc < outC; oc++)
                output.at(0, oc, oh, ow) = cData[(oh * outW + ow) * N + oc];
}

// ============================================================================
// 3. 缓存分块卷积
// ============================================================================
void tiledConv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
                 Tensor& output, int stride, int pad,
                 int outHTile = 16, int outWTile = 16, int ocTile = 32) {
    int N = input.shape[0], inC = input.shape[1];
    int inH = input.shape[2], inW = input.shape[3];
    int outC = weight.shape[0];
    int kH = weight.shape[2], kW = weight.shape[3];
    int outH = (inH + 2 * pad - kH) / stride + 1;
    int outW = (inW + 2 * pad - kW) / stride + 1;

    // 填充偏置
    for (int n = 0; n < N; n++)
        for (int oc = 0; oc < outC; oc++)
            for (int oh = 0; oh < outH; oh++)
                for (int ow = 0; ow < outW; ow++)
                    output.at(n, oc, oh, ow) = bias.data[oc];

    for (int n = 0; n < N; n++)
        for (int ohS = 0; ohS < outH; ohS += outHTile)
            for (int owS = 0; owS < outW; owS += outWTile)
                for (int ocS = 0; ocS < outC; ocS += ocTile) {
                    int ohE = std::min(ohS + outHTile, outH);
                    int owE = std::min(owS + outWTile, outW);
                    int ocE = std::min(ocS + ocTile, outC);
                    for (int ic = 0; ic < inC; ic++)
                        for (int kh = 0; kh < kH; kh++)
                            for (int kw = 0; kw < kW; kw++)
                                for (int oh = ohS; oh < ohE; oh++) {
                                    int ih = oh * stride + kh - pad;
                                    if (ih < 0 || ih >= inH) continue;
                                    for (int ow = owS; ow < owE; ow++) {
                                        int iw = ow * stride + kw - pad;
                                        if (iw < 0 || iw >= inW) continue;
                                        float iv = input.at(n, ic, ih, iw);
                                        for (int oc = ocS; oc < ocE; oc++)
                                            output.at(n, oc, oh, ow) += iv * weight.at(oc, ic, kh, kw);
                                    }
                                }
                }
}

// ============================================================================
// 4. Winograd F(2,3) 卷积 (简化版, 仅用于性能对比)
// 这里用一个简化的 Winograd 实现来做对比
// 完整实现见 step4_winograd_conv.cpp
// ============================================================================
void winogradConv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
                    Tensor& output, int /*stride*/, int pad) {
    // Winograd 只支持 stride=1, kernel=3x3
    int inC = input.shape[1], inH = input.shape[2], inW = input.shape[3];
    int outC = weight.shape[0];
    int outH = output.shape[2], outW = output.shape[3];

    // 对 Winograd 的简化实现: 直接用朴素卷积代替
    // 在 step4 中有完整的 Winograd 实现
    // 这里仅作占位, 实际运行时退化到朴素实现
    int kH = 3, kW = 3, stride = 1;
    for (int n = 0; n < 1; n++)
        for (int oc = 0; oc < outC; oc++)
            for (int oh = 0; oh < outH; oh++)
                for (int ow = 0; ow < outW; ow++) {
                    float sum = bias.data[oc];
                    for (int ic = 0; ic < inC; ic++)
                        for (int kh = 0; kh < kH; kh++)
                            for (int kw = 0; kw < kW; kw++) {
                                int ih = oh * stride + kh - pad;
                                int iw = ow * stride + kw - pad;
                                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                                    sum += input.at(n, ic, ih, iw) * weight.at(oc, ic, kh, kw);
                            }
                    output.at(n, oc, oh, ow) = sum;
                }
}

// ============================================================================
// 主函数: 全面对比 CPU 和 GPU 的卷积性能
// ============================================================================
int main() {
    const int N = 1, inC = 3, outC = 64;
    const int H = 224, W = 224;
    const int kH = 3, kW = 3;
    const int stride = 1, pad = 1;
    const int outH = (H + 2 * pad - kH) / stride + 1;
    const int outW = (W + 2 * pad - kW) / stride + 1;

    printf("=== Lab3 Step8: Vulkan Conv2D —— 全面对比 ===\n\n");
    printf("问题规模: %dx%dx%d → %dx%dx%d, kernel=%dx%d, stride=%d, pad=%d\n",
           inC, H, W, outC, outH, outW, kH, kW, stride, pad);

    int64_t flops = flopCountConv2d(inC, outC, kH, kW, outH, outW);
    printf("FLOPs: %lld (%.2fG)\n\n", (long long)flops, flops / 1e9);

    // 创建张量
    Tensor input({1, inC, H, W});
    Tensor weight({outC, inC, kH, kW});
    Tensor bias({outC});
    Tensor output({1, outC, outH, outW});

    srand(42);
    input.randomize();
    weight.randomize();
    for (int i = 0; i < outC; i++) bias.data[i] = 0.01f * (rand() % 100) / 100.0f;

    // ================================================================
    // 运行所有 CPU 卷积变体并测量性能
    // ================================================================
    const int NUM_ITERS = 5;

    struct ConvResult {
        std::string name;
        double avgMs;
        double gflops;
    };
    std::vector<ConvResult> results;

    // --- 1. Naive ---
    printf("--- 运行各卷积变体 ---\n");
    output.zero();
    naiveConv2d(input, weight, bias, output, stride, pad);

    double totalMs = 0.0;
    for (int iter = 0; iter < NUM_ITERS; iter++) {
        output.zero();
        Timer t; t.start();
        naiveConv2d(input, weight, bias, output, stride, pad);
        t.stop();
        totalMs += t.elapsedMs();
    }
    results.push_back({"Naive", totalMs / NUM_ITERS,
                        static_cast<double>(flops) / (totalMs / NUM_ITERS * 1e6)});
    printf("  Naive:          %.2f ms, %.1f GFLOPS\n",
           results.back().avgMs, results.back().gflops);

    // --- 2. AVX2 ---
    output.zero();
    avx2Conv2d(input, weight, bias, output, stride, pad);

    totalMs = 0.0;
    for (int iter = 0; iter < NUM_ITERS; iter++) {
        output.zero();
        Timer t; t.start();
        avx2Conv2d(input, weight, bias, output, stride, pad);
        t.stop();
        totalMs += t.elapsedMs();
    }
    results.push_back({"AVX2", totalMs / NUM_ITERS,
                        static_cast<double>(flops) / (totalMs / NUM_ITERS * 1e6)});
    printf("  AVX2:           %.2f ms, %.1f GFLOPS\n",
           results.back().avgMs, results.back().gflops);

    // --- 3. Tiled ---
    output.zero();
    tiledConv2d(input, weight, bias, output, stride, pad);

    totalMs = 0.0;
    for (int iter = 0; iter < NUM_ITERS; iter++) {
        output.zero();
        Timer t; t.start();
        tiledConv2d(input, weight, bias, output, stride, pad);
        t.stop();
        totalMs += t.elapsedMs();
    }
    results.push_back({"Tiled", totalMs / NUM_ITERS,
                        static_cast<double>(flops) / (totalMs / NUM_ITERS * 1e6)});
    printf("  Tiled:          %.2f ms, %.1f GFLOPS\n",
           results.back().avgMs, results.back().gflops);

    // --- 4. Winograd ---
    output.zero();
    winogradConv2d(input, weight, bias, output, stride, pad);

    totalMs = 0.0;
    for (int iter = 0; iter < NUM_ITERS; iter++) {
        output.zero();
        Timer t; t.start();
        winogradConv2d(input, weight, bias, output, stride, pad);
        t.stop();
        totalMs += t.elapsedMs();
    }
    results.push_back({"Winograd*", totalMs / NUM_ITERS,
                        static_cast<double>(flops) / (totalMs / NUM_ITERS * 1e6)});
    printf("  Winograd*:      %.2f ms, %.1f GFLOPS (*教学简化版)\n",
           results.back().avgMs, results.back().gflops);

    // --- 5. Vulkan (模拟) ---
    // GPU 性能估算
    double gpuPeakGflops = 2000.0;  // 典型集成显卡
    double gpuUtil = 0.3;
    double gpuComputeMs = static_cast<double>(flops) / (gpuPeakGflops * gpuUtil * 1e6) * 1e3;
    double dataBytes = static_cast<double>(inC * H * W + outC * inC * kH * kW + outC * outH * outW) * 4;
    double transferMs = dataBytes / (16.0 * 1e6) * 1e3;  // PCIe ~16 GB/s
    double gpuTotalMs = gpuComputeMs + transferMs;
    double gpuGflops = static_cast<double>(flops) / (gpuTotalMs * 1e6);

    results.push_back({"Vulkan(est)", gpuTotalMs, gpuGflops});
    printf("  Vulkan(est):    %.2f ms, %.1f GFLOPS (估算, 传输%.2f+计算%.2f)\n",
           gpuTotalMs, gpuGflops, transferMs, gpuComputeMs);

    // ================================================================
    // 完整对比表格
    // ================================================================
    printf("\n====================================\n");
    printf("  完整性能对比表格\n");
    printf("====================================\n\n");
    printf("  %-12s %10s %10s %10s\n", "方法", "耗时(ms)", "GFLOPS", "加速比");
    printf("  %-12s %10s %10s %10s\n", "----", "--------", "------", "------");

    double baselineMs = results[0].avgMs;
    for (const auto& r : results) {
        printf("  %-12s %10.2f %10.1f %9.2fx\n",
               r.name.c_str(), r.avgMs, r.gflops, baselineMs / r.avgMs);
    }

    // ================================================================
    // ASCII 柱状图
    // ================================================================
    printf("\n--- GFLOPS 对比图 ---\n\n");
    double maxGf = 0;
    for (const auto& r : results) maxGf = std::max(maxGf, r.gflops);
    int barW = 50;

    for (const auto& r : results) {
        int bar = static_cast<int>(r.gflops / maxGf * barW);
        printf("  %-12s|", r.name.c_str());
        for (int i = 0; i < bar; i++) printf("#");
        printf(" %.1f\n", r.gflops);
    }

    // ================================================================
    // GPU 卷积的着色器讲解
    // ================================================================
    printf("\n--- GPU 卷积着色器 (GLSL, 融合法) ---\n");
    printf("  // 每个线程计算输出 C[oc][oh][ow] 的一个元素\n");
    printf("  // 需要读取 inC*kH*kW 个输入和权重\n");
    printf("  void main() {\n");
    printf("      int ow = int(gl_GlobalInvocationID.x);\n");
    printf("      int oh = int(gl_GlobalInvocationID.y);\n");
    printf("      // oc 通过 push_constant 或第3维传入\n");
    printf("      float sum = 0.0;\n");
    printf("      for (int ic = 0; ic < inC; ic++)\n");
    printf("          for (int kh = 0; kh < kH; kh++)\n");
    printf("              for (int kw = 0; kw < kW; kw++)\n");
    printf("                  sum += input[ic*H*W+ih*W+iw]\n");
    printf("                       * weight[oc*inC*kH*kW+ic*kH*kH+kh*kW+kw];\n");
    printf("      output[oc*outH*outW+oh*outW+ow] = sum;\n");
    printf("  }\n\n");

    // ================================================================
    // GPU vs CPU 策略选择指南
    // ================================================================
    printf("--- GPU vs CPU 策略选择指南 ---\n\n");
    printf("  场景                     | 推荐方法           | 原因\n");
    printf("  -------------------------|-------------------|------\n");
    printf("  小模型 (MobileNet)        | CPU (AVX2/Tiled)  | 数据量小, GPU 传输开销大\n");
    printf("  中模型 (ResNet-50)        | CPU 或 GPU         | 需要逐层评估\n");
    printf("  大模型 (ResNet-101+)      | GPU (Vulkan)       | 计算量大, GPU 优势明显\n");
    printf("  批量推理 (>4 samples)     | GPU (Vulkan)       | 批量摊薄传输开销\n");
    printf("  实时推理 (latency <10ms)  | CPU               | 延迟敏感, 避免传输\n");
    printf("  吞吐优先 (throughput)     | GPU               | 高吞吐, 不在乎单次延迟\n\n");

    printf("  实际部署的考虑:\n");
    printf("    1. 功耗: GPU 通常比 CPU 功耗高 2-3x\n");
    printf("    2. 内存: im2col 膨胀 + GPU 缓冲区 = 更多内存\n");
    printf("    3. 延迟: GPU 首次推理有初始化开销\n");
    printf("    4. 精度: 不同实现可能有微小数值差异\n");
    printf("    5. 这就是为什么 edge-ai-perf 项目需要调度器!\n");

    printf("\n=== Step8 完成 ===\n");
    printf("下一步: step9_custom_ncnn_op.cpp —— 在 NCNN 中部署自定义算子\n");

    return 0;
}
