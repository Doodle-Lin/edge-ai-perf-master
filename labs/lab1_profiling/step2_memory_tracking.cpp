/**
 * @file step2_memory_tracking.cpp
 * @brief Lab1 Step2 —— 内存追踪实验：分配追踪、带宽测量与缓存效应
 *
 * ====== 本实验学习要点 ======
 * 1. 内存分配追踪:
 *    - 使用 MemoryTracker 单例记录每次分配/释放
 *    - 峰值内存决定了模型能否部署到目标设备
 *    - 典型推理引擎的内存构成: 权重(静态) + 激活(动态) + 工作空间(临时)
 *
 * 2. 内存带宽测量:
 *    - 顺序读取大数组，测量实际内存带宽
 *    - 带宽 = 数据量 / 时间，反映内存子系统吞吐能力
 *    - 很多算子不是计算瓶颈，而是"数据搬不动"——内存墙(Memory Wall)
 *
 * 3. 缓存效应:
 *    - 顺序访问: 数据预取器(prefetcher)能提前加载数据到缓存，吞吐高
 *    - 随机访问: 每次访问都可能缓存未命中，需从主存读取，吞吐低
 *    - 差异可达 10x 以上，这就是为什么优化缓存命中率如此重要!
 *
 * 4. 实际意义:
 *    - 深度学习推理中，很多算子是内存密集型(memory-bound)
 *    - 对内存密集型算子，优化计算(SIMD)效果有限，优化数据布局和访问模式更有效
 *    - Roofline 模型: 计算密度 < 带宽拐点的算子落在"斜坡"上，受带宽限制
 *
 * 编译: g++ -std=c++17 -O2 -I../../include step2_memory_tracking.cpp \
 *            ../../src/profiler/MemoryTracker.cpp -o step2_memory_tracking
 */

#include <algorithm>    // std::shuffle, std::sort
#include <chrono>       // std::chrono::steady_clock
#include <cmath>        // std::sqrt
#include <cstdio>       // printf
#include <cstdlib>      // rand, srand, malloc, free
#include <numeric>      // std::iota
#include <random>       // std::mt19937
#include <vector>

// 使用项目中的 MemoryTracker 追踪内存分配
#include "profiler/MemoryTracker.h"
// 使用 MathUtils 中的带宽计算函数
#include "common/MathUtils.h"

// ============================================================================
// 辅助函数: 格式化字节数为人类可读格式
// ============================================================================

/**
 * 将字节数转换为人类可读的字符串
 * 例如: 1536 -> "1.50 KB",  1048576 -> "1.00 MB"
 *
 * @param bytes 字节数
 * @return 格式化字符串（静态缓冲区，非线程安全，仅用于打印）
 */
const char* formatBytes(size_t bytes) {
    static char buf[64];
    if (bytes < 1024) {
        snprintf(buf, sizeof(buf), "%zu B", bytes);
    } else if (bytes < 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%.2f KB", static_cast<double>(bytes) / 1024.0);
    } else if (bytes < 1024ULL * 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%.2f MB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    } else {
        snprintf(buf, sizeof(buf), "%.2f GB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    }
    return buf;
}

// ============================================================================
// 辅助函数: 计算统计量（与 step1 相同，此处自包含）
// ============================================================================

double computeMean(const std::vector<double>& samples) {
    if (samples.empty()) return 0.0;
    double sum = 0.0;
    for (double s : samples) sum += s;
    return sum / static_cast<double>(samples.size());
}

double computeMedian(std::vector<double> samples) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    size_t n = samples.size();
    if (n % 2 == 0) {
        return (samples[n / 2 - 1] + samples[n / 2]) / 2.0;
    } else {
        return samples[n / 2];
    }
}

// ============================================================================
// 实验: 内存带宽测试
// ============================================================================

/**
 * 测量顺序内存读取带宽
 *
 * 原理: 分配一个大数组，以步长 1 顺序读取所有元素。
 * 由于 CPU 硬件预取器(prefetcher)能识别顺序访问模式，
 * 它会提前将下一批数据从内存加载到缓存中，
 * 使得顺序访问几乎总能命中 L1/L2 缓存，从而接近内存的理论带宽。
 *
 * @param sizeBytes 数组大小（字节）
 * @param iterations 重复读取次数（增加测量时间，提高精度）
 * @return 带宽 (GB/s)
 */
double measureSequentialBandwidth(size_t sizeBytes, int iterations = 10) {
    size_t numFloats = sizeBytes / sizeof(float);

    // 分配并初始化数组
    // 使用 float 而非 double，因为深度学习推理主要使用 float32
    float* data = static_cast<float*>(std::malloc(sizeBytes));
    if (!data) {
        printf("   [错误] 无法分配 %s 内存!\n", formatBytes(sizeBytes));
        return 0.0;
    }

    // 初始化数据，确保物理页已分配（避免缺页中断影响测量）
    for (size_t i = 0; i < numFloats; ++i) {
        data[i] = static_cast<float>(i);
    }

    // Warmup: 预热缓存和 TLB
    volatile float sink = 0.0f;
    for (size_t i = 0; i < numFloats; ++i) {
        sink += data[i];
    }

    // 正式测量: 多次顺序遍历
    auto start = std::chrono::steady_clock::now();
    float acc = 0.0f;
    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < numFloats; ++i) {
            // 累加防止编译器优化掉循环
            acc += data[i];
        }
    }
    auto end = std::chrono::steady_clock::now();

    // 防止编译器优化掉 acc
    if (acc > 1e30f) {
        printf("   ");  // 不会发生，仅防止优化
    }

    double timeMs = std::chrono::duration<double, std::milli>(end - start).count();

    // 总数据量 = 数组大小 * 迭代次数
    int64_t totalBytes = static_cast<int64_t>(sizeBytes) * iterations;
    double bandwidth = edgeai::memoryBandwidth(totalBytes, timeMs);

    std::free(data);
    return bandwidth;
}

// ============================================================================
// 实验: 缓存效应对比
// ============================================================================

/**
 * 对比顺序访问与随机访问的延迟差异
 *
 * 缓存工作原理（简化版）:
 *   L1 Cache:  ~32KB,  延迟 ~1ns,   每核心独享
 *   L2 Cache:  ~256KB, 延迟 ~4ns,   每核心独享
 *   L3 Cache:  ~数MB,  延迟 ~10ns,  所有核心共享
 *   主存 DDR4: ~数GB,  延迟 ~60-100ns
 *
 * 顺序访问时，CPU 的硬件预取器检测到线性访问模式，
 * 会提前把下一个缓存行(cacheline, 64字节)加载到 L1/L2，
 * 因此大部分访问都能命中缓存，延迟接近 L1/L2。
 *
 * 随机访问时，预取器无法预测下一个访问地址，
 * 每次访问大概率缓存未命中，必须从主存读取，延迟高 10~50 倍。
 *
 * @param arraySize 数组元素个数
 * @param stride    随机访问的步长（元素数）
 * @param iterations 重复次数
 */
void compareCacheEffects(size_t arraySize, int stride, int iterations) {
    size_t sizeBytes = arraySize * sizeof(float);
    float* data = static_cast<float*>(std::malloc(sizeBytes));
    if (!data) {
        printf("   [错误] 内存分配失败!\n");
        return;
    }

    // 初始化
    for (size_t i = 0; i < arraySize; ++i) {
        data[i] = static_cast<float>(i) * 0.001f;
    }

    // --- 顺序访问测试 ---
    // 预热
    volatile float sink = 0.0f;
    for (size_t i = 0; i < arraySize; ++i) {
        sink += data[i];
    }

    auto seqStart = std::chrono::steady_clock::now();
    float acc = 0.0f;
    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < arraySize; ++i) {
            acc += data[i];
        }
    }
    auto seqEnd = std::chrono::steady_clock::now();
    double seqTimeMs = std::chrono::duration<double, std::milli>(seqEnd - seqStart).count();

    if (acc > 1e30f) printf(" ");

    // --- 随机访问测试 ---
    // 生成随机索引数组，模拟随机访问模式
    std::vector<size_t> indices(arraySize);
    std::iota(indices.begin(), indices.end(), 0);  // 填充 0, 1, 2, ...

    // 使用步长打乱: 不是完全随机，而是以 stride 为步长跳跃
    // 这模拟了推理中的常见模式: 卷积在输入特征图上的滑动窗口访问
    // 步长越大，相邻两次访问的地址间隔越大，缓存命中率越低
    std::vector<size_t> stridedIndices;
    for (size_t base = 0; base < static_cast<size_t>(stride); ++base) {
        for (size_t i = base; i < arraySize; i += stride) {
            stridedIndices.push_back(i);
        }
    }

    // 预热随机访问
    for (size_t idx = 0; idx < stridedIndices.size(); ++idx) {
        sink += data[stridedIndices[idx]];
    }

    auto randStart = std::chrono::steady_clock::now();
    acc = 0.0f;
    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t idx = 0; idx < stridedIndices.size(); ++idx) {
            acc += data[stridedIndices[idx]];
        }
    }
    auto randEnd = std::chrono::steady_clock::now();
    double randTimeMs = std::chrono::duration<double, std::milli>(randEnd - randStart).count();

    if (acc > 1e30f) printf(" ");

    // 打印结果
    double ratio = randTimeMs / seqTimeMs;
    printf("   顺序访问: %8.3f ms\n", seqTimeMs / iterations);
    printf("   随机访问(步长%d): %8.3f ms\n", stride, randTimeMs / iterations);
    printf("   差异: %.1fx <-- 这就是缓存的重要性!\n", ratio);

    std::free(data);
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    srand(42);
    printf("=== 内存追踪实验 ===\n\n");

    // ========================================================================
    // 实验 1: 分配追踪 —— 使用 MemoryTracker 模拟推理过程的内存分配
    // ========================================================================
    printf("1. 分配追踪:\n");
    printf("   原理: 推理引擎在运行时需要分配多种内存:\n");
    printf("   - 输入张量: 存放输入数据，如一张 224x224 的 RGB 图像\n");
    printf("   - 权重: 卷积核参数，加载模型时分配，推理期间不变\n");
    printf("   - 激活(特征图): 每层计算的中间结果，逐层分配和释放\n");
    printf("   - 工作空间: 算子内部的临时缓冲区\n");
    printf("   峰值内存 = 所有同时驻留的内存之和 → 决定模型能否在设备上运行\n\n");

    // 获取 MemoryTracker 单例
    auto& tracker = edgeai::MemoryTracker::instance();
    tracker.reset();  // 清空历史记录

    // --- 模拟 MobileNet 风格的推理内存分配 ---

    // 输入张量: 1x3x224x224 (一张 224x224 的 RGB 图像)
    // 内存 = 1 * 3 * 224 * 224 * 4 bytes = 602,112 bytes ≈ 588 KB
    size_t inputSize = 1 * 3 * 224 * 224 * sizeof(float);
    tracker.onAlloc("input_tensor", inputSize);
    printf("   [alloc] input_tensor: %zu bytes (%s)\n", inputSize, formatBytes(inputSize));

    // Conv1 权重: 32 个 3x3 卷积核，输入 3 通道
    // 内存 = 32 * 3 * 3 * 3 * 4 = 1728 bytes
    size_t conv1WeightSize = 32 * 3 * 3 * 3 * sizeof(float);
    tracker.onAlloc("weight_conv1", conv1WeightSize);
    printf("   [alloc] weight_conv1: %zu bytes (%s)\n", conv1WeightSize, formatBytes(conv1WeightSize));

    // Conv1 输出特征图: 1x32x112x112 (stride=2, 输出减半)
    // 内存 = 1 * 32 * 112 * 112 * 4 = 1,605,632 bytes ≈ 1.53 MB
    size_t conv1OutputSize = 1 * 32 * 112 * 112 * sizeof(float);
    tracker.onAlloc("feat_conv1", conv1OutputSize);
    printf("   [alloc] feat_conv1:   %zu bytes (%s)\n", conv1OutputSize, formatBytes(conv1OutputSize));

    // Conv2 权重: 64 个 3x3 卷积核，输入 32 通道 (depthwise: groups=32)
    // Depthwise 卷积: 每组 1 个 3x3 核，共 32 组 = 32*1*3*3 = 288 个参数
    size_t conv2DwWeightSize = 32 * 1 * 3 * 3 * sizeof(float);
    tracker.onAlloc("weight_conv2_dw", conv2DwWeightSize);
    printf("   [alloc] weight_conv2_dw: %zu bytes\n", conv2DwWeightSize);

    // Pointwise 卷积: 64 个 1x1 卷积核，输入 32 通道
    // 内存 = 64 * 32 * 1 * 1 * 4 = 8,192 bytes
    size_t conv2PwWeightSize = 64 * 32 * 1 * 1 * sizeof(float);
    tracker.onAlloc("weight_conv2_pw", conv2PwWeightSize);
    printf("   [alloc] weight_conv2_pw: %zu bytes (%s)\n", conv2PwWeightSize, formatBytes(conv2PwWeightSize));

    // Conv2 输出特征图: 1x64x112x112
    size_t conv2OutputSize = 1 * 64 * 112 * 112 * sizeof(float);
    tracker.onAlloc("feat_conv2", conv2OutputSize);
    printf("   [alloc] feat_conv2:   %zu bytes (%s)\n", conv2OutputSize, formatBytes(conv2OutputSize));

    // Conv3 权重: 128 个 1x1 卷积核，输入 64 通道
    size_t conv3WeightSize = 128 * 64 * 1 * 1 * sizeof(float);
    tracker.onAlloc("weight_conv3", conv3WeightSize);
    printf("   [alloc] weight_conv3: %zu bytes (%s)\n", conv3WeightSize, formatBytes(conv3WeightSize));

    // Conv3 输出特征图: 1x128x56x56 (stride=2)
    size_t conv3OutputSize = 1 * 128 * 56 * 56 * sizeof(float);
    tracker.onAlloc("feat_conv3", conv3OutputSize);
    printf("   [alloc] feat_conv3:   %zu bytes (%s)\n", conv3OutputSize, formatBytes(conv3OutputSize));

    // --- 模拟激活内存释放（Conv1 的输出不再需要，Conv3 已计算完毕）---
    tracker.onFree("feat_conv1", conv1OutputSize);
    printf("   [free]  feat_conv1:   %zu bytes (%s)\n", conv1OutputSize, formatBytes(conv1OutputSize));

    tracker.onFree("feat_conv2", conv2OutputSize);
    printf("   [free]  feat_conv2:   %zu bytes (%s)\n", conv2OutputSize, formatBytes(conv2OutputSize));

    // 打印追踪报告
    printf("\n   ");
    tracker.printReport();
    printf("\n   峰值内存: %s\n\n", formatBytes(tracker.peakUsage()));

    // ========================================================================
    // 实验 2: 内存带宽测试
    // ========================================================================
    printf("2. 内存带宽测试:\n");
    printf("   原理: 顺序读取大数组，测量内存子系统实际能提供的数据吞吐量。\n");
    printf("   如果算子所需带宽超过此值，则该算子是内存密集型(memory-bound)。\n\n");

    // 测试不同大小的数组，观察缓存效果
    struct BandwidthTest {
        size_t sizeBytes;
        const char* label;
    };

    std::vector<BandwidthTest> bwTests = {
        {4 * 1024,           "4 KB   (L1 缓存内)"},
        {256 * 1024,         "256 KB (L2 缓存内)"},
        {8 * 1024 * 1024,    "8 MB   (L3 缓存内)"},
        {64 * 1024 * 1024,  "64 MB  (超出缓存)"},
        {256 * 1024 * 1024, "256 MB (纯内存访问)"},
    };

    for (const auto& test : bwTests) {
        double bw = measureSequentialBandwidth(test.sizeBytes, /*iterations=*/5);
        printf("   %s: 带宽 %7.2f GB/s\n", test.label, bw);
    }

    // 重点展示 256MB 的结果
    double bw256 = measureSequentialBandwidth(256 * 1024 * 1024, /*iterations=*/5);
    auto bw256Start = std::chrono::steady_clock::now();
    // 重新测量并打印详细时间
    size_t bw256Size = 256 * 1024 * 1024;
    size_t numFloats256 = bw256Size / sizeof(float);
    float* bwData = static_cast<float*>(std::malloc(bw256Size));
    for (size_t i = 0; i < numFloats256; ++i) bwData[i] = static_cast<float>(i);

    // warmup
    volatile float vsink = 0.0f;
    for (size_t i = 0; i < numFloats256; ++i) vsink += bwData[i];

    auto t0 = std::chrono::steady_clock::now();
    float acc = 0.0f;
    for (int it = 0; it < 5; ++it) {
        for (size_t i = 0; i < numFloats256; ++i) {
            acc += bwData[i];
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double readTimeMs = std::chrono::duration<double, std::milli>(t1 - t0).count() / 5.0;
    if (acc > 1e30f) printf(" ");
    std::free(bwData);

    printf("\n   顺序读取 256MB:\n");
    printf("     耗时: %.1f ms\n", readTimeMs);
    printf("     带宽: %.1f GB/s\n\n", bw256);

    // ========================================================================
    // 实验 3: Cache 效应 —— 顺序访问 vs 随机访问
    // ========================================================================
    printf("3. Cache 效应:\n");
    printf("   原理: CPU 缓存利用局部性原理加速访问:\n");
    printf("   - 时间局部性: 最近访问的数据很可能会再次访问\n");
    printf("   - 空间局部性: 地址相近的数据很可能会被访问\n");
    printf("   顺序访问完美满足这两种局部性，随机访问则都不满足。\n\n");

    // 使用 4MB 数组（大于 L2，包含 L3 和内存访问）
    size_t cacheArraySize = 1024 * 1024;  // 1M floats = 4MB
    int cacheIter = 20;

    printf("   [数组大小: 4MB, 超出 L2 缓存]\n");
    compareCacheEffects(cacheArraySize, /*stride=*/64, cacheIter);

    printf("\n   [数组大小: 16MB, 超出 L3 缓存]\n");
    compareCacheEffects(4 * 1024 * 1024, /*stride=*/64, 10);

    // ========================================================================
    // 实验 4: 缓存友好的数据布局 —— 结构体数组 vs 数组结构体
    // ========================================================================
    printf("\n4. 数据布局对缓存的影响 (AoS vs SoA):\n");
    printf("   AoS (Array of Structures): struct {float x,y,z;} points[N];\n");
    printf("   SoA (Structure of Arrays): float x[N], y[N], z[N];\n");
    printf("   如果只处理 x 分量，SoA 的内存连续性更好，缓存命中率更高。\n\n");

    const int POINT_COUNT = 4 * 1024 * 1024;  // 4M 个点
    const int AOS_ITER = 20;

    // --- AoS 布局 ---
    struct Point3D {
        float x, y, z;
    };

    std::vector<Point3D> aosPoints(POINT_COUNT);
    for (int i = 0; i < POINT_COUNT; ++i) {
        aosPoints[i].x = static_cast<float>(i);
        aosPoints[i].y = static_cast<float>(i) * 0.5f;
        aosPoints[i].z = static_cast<float>(i) * 0.3f;
    }

    // 只访问 x 分量: 每次访问跨 12 字节(3 floats)，2/3 的缓存行数据无用
    auto aosStart = std::chrono::steady_clock::now();
    float aosAcc = 0.0f;
    for (int iter = 0; iter < AOS_ITER; ++iter) {
        for (int i = 0; i < POINT_COUNT; ++i) {
            aosAcc += aosPoints[i].x;  // 只用 x，但 y 和 z 也被加载到缓存行
        }
    }
    auto aosEnd = std::chrono::steady_clock::now();
    double aosTimeMs = std::chrono::duration<double, std::milli>(aosEnd - aosStart).count();

    // --- SoA 布局 ---
    std::vector<float> soaX(POINT_COUNT), soaY(POINT_COUNT), soaZ(POINT_COUNT);
    for (int i = 0; i < POINT_COUNT; ++i) {
        soaX[i] = static_cast<float>(i);
        soaY[i] = static_cast<float>(i) * 0.5f;
        soaZ[i] = static_cast<float>(i) * 0.3f;
    }

    // 只访问 x 分量: 内存完全连续，每个缓存行的数据都有用
    auto soaStart = std::chrono::steady_clock::now();
    float soaAcc = 0.0f;
    for (int iter = 0; iter < AOS_ITER; ++iter) {
        for (int i = 0; i < POINT_COUNT; ++i) {
            soaAcc += soaX[i];  // 连续访问，完美缓存利用
        }
    }
    auto soaEnd = std::chrono::steady_clock::now();
    double soaTimeMs = std::chrono::duration<double, std::milli>(soaEnd - soaStart).count();

    if (aosAcc > 1e30f || soaAcc > 1e30f) printf(" ");

    printf("   AoS (只读 x, y/z 白占缓存): %8.3f ms\n", aosTimeMs / AOS_ITER);
    printf("   SoA (只读 x, 完美缓存利用):  %8.3f ms\n", soaTimeMs / AOS_ITER);
    printf("   SoA 比 AoS 快: %.2fx\n\n", aosTimeMs / soaTimeMs);

    printf("   启示: 深度学习中的 NCHW 布局(通道在外，宽高在内)\n");
    printf("   就是一种 SoA 思想 —— 同一通道的数据在内存中连续存储。\n\n");

    // ========================================================================
    // 总结
    // ========================================================================
    printf("=== 实验完成 ===\n");
    printf("关键收获:\n");
    printf("1. 使用 MemoryTracker 追踪内存分配，峰值内存是部署的关键指标\n");
    printf("2. 顺序访问带宽远高于随机访问 → 优化数据布局和访问模式\n");
    printf("3. AoS vs SoA: 当只使用部分字段时，SoA 布局缓存更友好\n");
    printf("4. 内存墙(Memory Wall): 很多算子受带宽限制而非计算限制\n");

    return 0;
}
