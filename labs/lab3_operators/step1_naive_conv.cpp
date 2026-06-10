/**
 * @file step1_naive_conv.cpp
 * @brief Lab3 Step1: 朴素卷积 (Baseline) —— 6 层嵌套循环的 O(N^6) 基线版本
 *
 * ====== 本实验的学习目标 ======
 * 1. 理解卷积的数学定义，用最直白的 6 层嵌套循环实现
 * 2. 学会计算 FLOPs（浮点运算次数）和内存访问量
 * 3. 建立性能基线（baseline），后续所有优化版本都以此为参照
 * 4. 理解朴素卷积为什么慢——缓存不友好、无 SIMD、循环开销大
 *
 * ====== 卷积的数学定义 ======
 * 输出 out[n][oc][oh][ow] = sum_{ic,kh,kw} input[n][ic][oh*s+kh-p][ow*s+kw-p] * weight[oc][ic][kh][kw] + bias[oc]
 *
 * 其中：
 *   - N = batch size, ic = 输入通道, oc = 输出通道
 *   - kH/kW = 卷积核大小, s = stride, p = padding
 *   - oh = (H + 2*p - kH) / s + 1,  ow 同理
 *
 * ====== 为什么朴素实现慢？ ======
 * 1. 缓存不友好：内层循环在 input 上跳跃式访问，空间局部性差
 *    - 访问 input[n][ic][oh*s+kh][ow*s+kw] 时，kh/kw 变化导致跳跃
 *    - 同一个输入像素被多个输出位置重复读取，但可能已被逐出缓存
 * 2. 无法利用 SIMD：每次只算 1 个乘加，AVX2 可以同时算 8 个
 * 3. 循环开销大：6 层循环本身的分支预测和迭代变量更新也是开销
 * 4. 编译器难以自动向量化：嵌套循环的访存模式太复杂，编译器无法优化
 *
 * ====== 编译与运行 ======
 * g++ -O2 -std=c++17 -o step1_naive_conv step1_naive_conv.cpp
 * ./step1_naive_conv
 */

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>

// ============================================================================
// 简易计时器 —— 使用 C++11 高精度时钟
// 为什么不用 clock()？因为 clock() 统计的是 CPU 时间，不包含等待内存的时间
// steady_clock 是单调递增的物理时钟，适合测量实际耗时
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
// 张量类 —— 与项目 Tensor.h 对齐的简易版本
// 采用行优先 (row-major) 连续存储: [N, C, H, W]
// 元素 (n,c,h,w) 的偏移量 = n*C*H*W + c*H*W + h*W + w
// ============================================================================
struct Tensor {
    std::vector<int> shape;     // 各维度大小，如 {1, 3, 224, 224}
    std::vector<float> data;    // 连续存储的浮点数据

    Tensor() = default;

    // 按形状构造，元素初始化为 0
    explicit Tensor(const std::vector<int>& s) : shape(s) {
        int total = 1;
        for (int d : shape) total *= d;
        data.resize(total, 0.0f);
    }

    // 按形状 + 填充值构造
    Tensor(const std::vector<int>& s, float val) : shape(s) {
        int total = 1;
        for (int d : shape) total *= d;
        data.resize(total, val);
    }

    // 总元素数
    int total() const {
        int t = 1;
        for (int d : shape) t *= d;
        return t;
    }

    // 按 4D 索引访问: (n, c, h, w)
    float& at(int n, int c, int h, int w) {
        int C = shape[1], H = shape[2], W = shape[3];
        return data[n * C * H * W + c * H * W + h * W + w];
    }
    const float& at(int n, int c, int h, int w) const {
        int C = shape[1], H = shape[2], W = shape[3];
        return data[n * C * H * W + c * H * W + h * W + w];
    }

    // 用随机值填充 (均匀分布 [-1, 1])
    void randomize() {
        for (auto& v : data) {
            v = -1.0f + 2.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }

    // 将所有元素设为 0
    void zero() {
        std::fill(data.begin(), data.end(), 0.0f);
    }
};

// ============================================================================
// 计算标准 Conv2D 的 FLOPs (乘加算 2 次运算)
// 公式: 2 * batchSize * outChannels * outH * outW * (inChannels/groups * kH * kW)
//
// 为什么乘加算 2 次运算？
//   一次 MAC (Multiply-Accumulate) = 一次乘法 + 一次加法 = 2 FLOPs
//   这是因为硬件通常有独立的乘法器和加法器，FMA 指令同时完成两者
// ============================================================================
int64_t flopCountConv2d(int inC, int outC, int kH, int kW,
                        int outH, int outW, int groups = 1) {
    int64_t kernelsMulAdd = static_cast<int64_t>(inC / groups) * kH * kW;
    int64_t outputElements = static_cast<int64_t>(outC) * outH * outW;
    return 2 * outputElements * kernelsMulAdd;  // 每个输出元素: kernelsMulAdd 次 MAC
}

// ============================================================================
// 计算标准 Conv2D 的内存访问量 (bytes)
// = 输入 + 权重 + 输出
// 这是一个理论下界——实际内存访问量远大于此（因为缓存未命中时重复加载）
// ============================================================================
int64_t memSizeConv2d(int inC, int outC, int kH, int kW,
                      int inH, int inW, int groups = 1) {
    int64_t inputBytes  = static_cast<int64_t>(inC) * inH * inW * sizeof(float);
    int64_t weightBytes = static_cast<int64_t>(outC) * (inC / groups) * kH * kW * sizeof(float);
    int64_t outputBytes = static_cast<int64_t>(outC) * inH * inW * sizeof(float);  // 近似: outH ≈ inH
    return inputBytes + weightBytes + outputBytes;
}

// ============================================================================
// 朴素卷积 —— 6 层嵌套循环
//
// 循环顺序：batch → outC → outH → outW → inC → kH → kW
// 这个顺序是最"自然"的写法，但不是最高效的
// 后续步骤会展示如何通过 im2col、SIMD、分块等手段大幅提升性能
//
// 关于 padding 的处理：
// - 当 pad > 0 时，输入的有效范围扩展到 [-pad, H+pad) x [-pad, W+pad)
// - 超出原始输入范围的像素视为 0（zero-padding）
// - 在实现中，我们检查 ih 和 iw 是否在 [0, H) x [0, W) 范围内
// ============================================================================
void naiveConv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
                 Tensor& output,
                 int stride, int pad) {
    // 解析维度 —— 方便阅读，避免到处用 shape[i]
    int N    = input.shape[0];   // batch size
    int inC  = input.shape[1];   // 输入通道数
    int inH  = input.shape[2];   // 输入高度
    int inW  = input.shape[3];   // 输入宽度
    int outC = weight.shape[0];  // 输出通道数 = 权重的第 0 维
    int kH   = weight.shape[2];  // 卷积核高度
    int kW   = weight.shape[3];  // 卷积核宽度

    // 计算输出尺寸
    // 公式: outH = (inH + 2*pad - kH) / stride + 1
    int outH = (inH + 2 * pad - kH) / stride + 1;
    int outW = (inW + 2 * pad - kW) / stride + 1;

    // ================================================================
    // 6 层嵌套循环 —— 这是整个实验的性能基线
    //
    // 循环层次分析：
    //   n:   batch 维度，通常为 1（边缘推理场景）
    //   oc:  输出通道，对应不同的卷积核（权重不同）
    //   oh:  输出高度，每个位置对应一个感受野
    //   ow:  输出宽度，每个位置对应一个感受野
    //   ic:  输入通道，需要累加所有通道的贡献
    //   kh:  卷积核高度，在感受野内滑动
    //   kw:  卷积核宽度，在感受野内滑动
    //
    // 总计算量 = N * outC * outH * outW * inC * kH * kW
    // 对于 3x224x224 → 64x224x224, kernel=3x3:
    //   = 1 * 64 * 224 * 224 * 3 * 3 * 3
    //   = 862,836,672 次 MAC = 1,725,673,472 FLOPs ≈ 1.73G FLOPs
    // ================================================================
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < outC; oc++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    // 累加器 —— 先加偏置，再累加卷积结果
                    float sum = bias.data[oc];

                    for (int ic = 0; ic < inC; ic++) {
                        for (int kh = 0; kh < kH; kh++) {
                            for (int kw = 0; kw < kW; kw++) {
                                // 计算输入对应的坐标
                                // oh*stride + kh - pad: 从输出坐标反推输入坐标
                                int ih = oh * stride + kh - pad;
                                int iw = ow * stride + kw - pad;

                                // 边界检查: zero-padding
                                // 如果坐标超出输入范围，该像素值为 0
                                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                    sum += input.at(n, ic, ih, iw) * weight.at(oc, ic, kh, kw);
                                }
                                // 如果 ih < 0 || ih >= inH || iw < 0 || iw >= inW:
                                // 相当于乘以 0，不需要加到 sum 中
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
// 主函数
// ============================================================================
int main() {
    // ================================================================
    // 问题规模定义
    // 模拟典型 CNN 第一层卷积: 3 通道 RGB 图像 → 64 通道特征图
    // 这是 ResNet、MobileNet 等网络中非常常见的配置
    // ================================================================
    const int N     = 1;     // batch size (边缘推理通常为 1)
    const int inC   = 3;     // 输入通道数 (RGB)
    const int outC  = 64;    // 输出通道数
    const int H     = 224;   // 输入高度 (ImageNet 标准尺寸)
    const int W     = 224;   // 输入宽度
    const int kH    = 3;     // 卷积核高度
    const int kW    = 3;     // 卷积核宽度
    const int stride = 1;    // 步长
    const int pad    = 1;    // 填充 (same padding: 输出尺寸 = 输入尺寸)

    // 计算输出尺寸
    const int outH = (H + 2 * pad - kH) / stride + 1;  // = 224
    const int outW = (W + 2 * pad - kW) / stride + 1;  // = 224

    // ================================================================
    // 打印实验信息
    // ================================================================
    printf("=== Lab3 Step1: 朴素卷积 (Baseline) ===\n");
    printf("问题规模: %dx%dx%d → %dx%dx%d, kernel=%dx%d\n",
           inC, H, W, outC, outH, outW, kH, kW);

    // 计算 FLOPs 和内存访问量
    int64_t flops = flopCountConv2d(inC, outC, kH, kW, outH, outW);
    int64_t memBytes = memSizeConv2d(inC, outC, kH, kW, H, W);

    // 格式化输出 FLOPs (带千分位分隔)
    // C++ 的 printf 没有千分位格式，手动处理
    printf("FLOPs: %lld (%.2fG)\n", (long long)flops, flops / 1e9);
    printf("内存访问: %.1f MB\n", memBytes / (1024.0 * 1024.0));

    // 计算 Arithmetic Intensity (计算密度)
    // = FLOPs / 内存访问字节数
    // 这个值决定了算子是计算密集型还是内存密集型
    // - AI > 10: 计算密集型 → 优化重点是 SIMD/并行
    // - AI < 5:  内存密集型 → 优化重点是缓存/带宽
    double arithIntensity = static_cast<double>(flops) / memBytes;
    printf("计算密度 (Arithmetic Intensity): %.2f FLOPs/byte\n", arithIntensity);
    if (arithIntensity > 10.0) {
        printf("  → 计算密集型: 优化重点 = SIMD/并行\n");
    } else if (arithIntensity < 5.0) {
        printf("  → 内存密集型: 优化重点 = 缓存/带宽\n");
    } else {
        printf("  → 混合型: 需要同时优化计算和访存\n");
    }

    // ================================================================
    // 创建输入、权重、偏置、输出张量
    // ================================================================
    Tensor input({N, inC, H, W});       // 输入: [1, 3, 224, 224]
    Tensor weight({outC, inC, kH, kW}); // 权重: [64, 3, 3, 3]
    Tensor bias({outC});                 // 偏置: [64]
    Tensor output({N, outC, outH, outW}); // 输出: [1, 64, 224, 224]

    // 用随机值初始化输入和权重
    // 注意: 推理框架中，权重是预训练好的，这里用随机值模拟
    srand(42);  // 固定种子，保证可重复性
    input.randomize();
    weight.randomize();
    for (int i = 0; i < outC; i++) bias.data[i] = 0.01f * (rand() % 100) / 100.0f;

    // ================================================================
    // 性能测量 —— 运行 10 次迭代，报告平均时间
    //
    // 为什么要多次运行取平均？
    // 1. 消除系统噪声: OS 调度、中断、缓存冷启动等因素导致单次测量不稳定
    // 2. 预热效应: 第一次运行时数据不在缓存中（冷启动），后续运行更快
    // 3. 统计可靠性: 多次测量的平均值比单次测量更接近真实性能
    //
    // 为什么要排除第一次？
    // - 第一次运行包含: 页表分配 (page fault)、指令缓存缺失、数据缓存冷启动
    // - 这些开销在实际推理中只发生一次，不应计入稳态性能
    // ================================================================
    const int NUM_ITERS = 10;
    double totalTimeMs = 0.0;

    // 预热: 先运行一次，让数据进入缓存
    output.zero();
    naiveConv2d(input, weight, bias, output, stride, pad);

    printf("\n运行 %d 次迭代:\n", NUM_ITERS);
    for (int iter = 0; iter < NUM_ITERS; iter++) {
        output.zero();  // 每次清零输出，避免累加

        Timer timer;
        timer.start();
        naiveConv2d(input, weight, bias, output, stride, pad);
        timer.stop();

        double ms = timer.elapsedMs();
        totalTimeMs += ms;
        printf("  第 %2d 次: %.2f ms\n", iter + 1, ms);
    }

    // ================================================================
    // 计算性能指标
    // ================================================================
    double avgTimeMs = totalTimeMs / NUM_ITERS;

    // GFLOPS = FLOPs / (时间(s) * 10^9)
    // = FLOPs / (时间(ms) * 10^6)
    double gflops = static_cast<double>(flops) / (avgTimeMs * 1e6);

    // 内存带宽 = 内存访问量 / 时间
    double bandwidthGBs = static_cast<double>(memBytes) / (avgTimeMs * 1e6);

    printf("\n--- 性能汇总 ---\n");
    printf("  平均耗时: %.2f ms\n", avgTimeMs);
    printf("  GFLOPS: %.2f\n", gflops);
    printf("  内存带宽: %.2f GB/s\n", bandwidthGBs);

    // ================================================================
    // 与硬件峰值对比
    //
    // 假设 CPU 为 Intel Core i7 (Skylake):
    // - 频率: ~3.5 GHz
    // - AVX2: 256-bit = 8 float/FMA
    // - 2 个 FMA 单元
    // - 峰值 FP32: 3.5 * 8 * 2 * 2 = 112 GFLOPS (单核)
    //
    // 朴素实现的 GFLOPS 通常只有峰值的 1-5%
    // 这就是为什么优化如此重要！
    // ================================================================
    double peakGflops = 112.0;  // 典型 Skylake 单核 FP32 峰值
    double utilization = gflops / peakGflops * 100.0;
    printf("\n--- 与硬件峰值对比 ---\n");
    printf("  假设单核峰值: %.0f GFLOPS (Skylake @3.5GHz, AVX2)\n", peakGflops);
    printf("  硬件利用率: %.2f%%\n", utilization);
    printf("  → 朴素实现只用了硬件能力的 %.1f%%，还有 %.1f 倍的优化空间!\n",
           utilization, (100.0 / utilization - 1.0) * 100.0 / 100.0);

    // ================================================================
    // 正确性检查 —— 随机验证几个输出元素
    // 选取几个 (n, oc, oh, ow) 位置，手动计算并与输出对比
    // 这是验证优化版本正确性的标准方法
    // ================================================================
    printf("\n--- 正确性验证 (抽样) ---\n");
    int numChecks = 5;
    for (int chk = 0; chk < numChecks; chk++) {
        int oc = rand() % outC;
        int oh = rand() % outH;
        int ow = rand() % outW;

        // 手动重新计算
        float expected = bias.data[oc];
        for (int ic = 0; ic < inC; ic++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    int ih = oh * stride + kh - pad;
                    int iw = ow * stride + kw - pad;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        expected += input.at(0, ic, ih, iw) * weight.at(oc, ic, kh, kw);
                    }
                }
            }
        }

        float actual = output.at(0, oc, oh, ow);
        float diff = std::fabs(actual - expected);
        printf("  out[0][%d][%d][%d]: expected=%.6f, actual=%.6f, diff=%.2e %s\n",
               oc, oh, ow, expected, actual, diff,
               diff < 1e-4f ? "OK" : "MISMATCH!");
    }

    printf("\n=== Step1 完成 ===\n");
    printf("下一步: step2_simd_conv.cpp —— 用 im2col + AVX2 将 GFLOPS 提升 4-8 倍\n");

    return 0;
}
