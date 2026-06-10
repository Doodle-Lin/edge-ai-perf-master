/**
 * @file step1_timer_basics.cpp
 * @brief Lab1 Step1 —— 计时基础实验：高精度计时的原理、选择与方法论
 *
 * ====== 本实验学习要点 ======
 * 1. 三种时钟 API 的区别与选择:
 *    - clock()            : C 标准库，精度低（约 10~15ms），受 CPU 时间片影响
 *    - high_resolution_clock: 精度最高但可能不稳定（某些平台是 steady_clock 的别名，某些不是）
 *    - steady_clock       : 单调递增，不受 NTP 校时影响，是性能测量的最佳选择
 *
 * 2. Warmup（预热）效应:
 *    - 首几次执行较慢，因为：指令缓存未命中、分支预测未建立、
 *      操作系统按需分配物理页、CPU 频率尚未提升（turbo boost）
 *    - 正确做法：丢弃前 5~10 次的测量结果
 *
 * 3. 计时统计方法论:
 *    - 单次测量不可靠（受系统调度干扰），应多次测量取统计值
 *    - 均值：反映整体水平，但受极端值影响
 *    - 中位数：更稳健，不受偶发高延迟影响
 *    - 最小值：反映"最好情况"，接近硬件真实能力
 *    - 最大值：反映"最坏情况"，可能包含操作系统调度延迟
 *
 * 4. 测量开销:
 *    - 计时器本身也有开销（系统调用、缓存失效等）
 *    - 空循环计时可以估算计时器的基础开销
 *    - 当测量对象耗时接近计时器开销时，结果不可信
 *
 * 编译: g++ -std=c++17 -O2 -I../../include step1_timer_basics.cpp -o step1_timer_basics
 */

#include <algorithm>    // std::sort, std::min_element, std::max_element
#include <chrono>       // std::chrono::steady_clock, high_resolution_clock
#include <cmath>        // std::sqrt
#include <cstdio>       // printf
#include <cstdlib>      // rand, srand
#include <ctime>        // clock()
#include <numeric>      // std::accumulate
#include <vector>

// 使用项目中的 Tensor 类进行矩阵乘法实验
#include "common/Tensor.h"
// 使用 MathUtils 中的 GFLOPS 计算函数
#include "common/MathUtils.h"

// ============================================================================
// 辅助函数：计算统计量
// ============================================================================

/// 计算均值
/// @param samples 样本集合（毫秒）
/// @return 均值（毫秒）
double computeMean(const std::vector<double>& samples) {
    if (samples.empty()) return 0.0;
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    return sum / static_cast<double>(samples.size());
}

/// 计算中位数
/// @param samples 样本集合（毫秒）
/// @return 中位数（毫秒）
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

/// 计算标准差
/// @param samples 样本集合（毫秒）
/// @return 标准差（毫秒）
double computeStdDev(const std::vector<double>& samples) {
    if (samples.size() < 2) return 0.0;
    double mean = computeMean(samples);
    double sqSum = 0.0;
    for (double s : samples) {
        double diff = s - mean;
        sqSum += diff * diff;
    }
    return std::sqrt(sqSum / static_cast<double>(samples.size() - 1));
}

// ============================================================================
// 实验用计算内核：朴素矩阵乘法
// ============================================================================

/**
 * 朴素三重循环矩阵乘法 C = A * B
 *
 * 矩阵乘法是性能测试的经典工作负载:
 * - 计算量可精确计算: 2 * N^3 FLOPs (N^3 次乘法 + N^3 次加法)
 * - 内存访问模式清晰，方便分析缓存行为
 * - 是深度学习卷积算子的基础操作
 *
 * @param A 输入矩阵 [N x K]
 * @param B 输入矩阵 [K x M]
 * @param C 输出矩阵 [N x M]
 * @param N A 的行数
 * @param K A 的列数 / B 的行数
 * @param M B 的列数
 */
void matmulNaive(const float* A, const float* B, float* C, int N, int K, int M) {
    // 初始化输出矩阵为 0（重要！否则会累加垃圾值）
    for (int i = 0; i < N * M; ++i) {
        C[i] = 0.0f;
    }

    // 三重循环: 最外层遍历输出行，中间层遍历输出列，最内层做点积
    // 这是最朴素的实现，缓存不友好:
    //   - B 的访问是列优先的（跳跃式），每次访问跨越 M*sizeof(float) 字节
    //   - 当 M 很大时，B[k][j] 和 B[k+1][j] 不在同一缓存行
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                // A[i][k] 是行优先连续访问，缓存友好
                // B[k][j] 是列方向跳跃访问，缓存不友好
                sum += A[i * K + k] * B[k * M + j];
            }
            C[i * M + j] = sum;
        }
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    // 设置随机种子，保证每次运行结果可复现
    srand(42);

    printf("=== 计时基础实验 ===\n\n");

    // ========================================================================
    // 实验 1: 三种时钟对比
    // ========================================================================
    printf("1. 三种时钟对比:\n");
    printf("   原理: clock() 统计 CPU 时间, high_resolution_clock 精度最高但不保证单调,\n");
    printf("         steady_clock 单调递增，不受系统时间调整(NTP)影响，是性能测量首选。\n\n");

    // 准备一个适度的工作负载: 128x128 矩阵乘法
    // 使用项目中的 Tensor 类
    const int MAT_SIZE = 128;
    edgeai::Tensor matA({1, 1, MAT_SIZE, MAT_SIZE});  // 用 4D 张量表示矩阵
    edgeai::Tensor matB({1, 1, MAT_SIZE, MAT_SIZE});
    edgeai::Tensor matC({1, 1, MAT_SIZE, MAT_SIZE});
    matA.randomize();
    matB.randomize();

    // --- 方法 1: clock() ---
    // clock() 返回的是进程使用的 CPU 时钟滴答数，除以 CLOCKS_PER_SEC 得到秒数
    // 缺点: 精度低（Windows 上约 15ms），且多线程下可能不准确
    clock_t start_clock = clock();
    matmulNaive(matA.data(), matB.data(), matC.data(), MAT_SIZE, MAT_SIZE, MAT_SIZE);
    clock_t end_clock = clock();
    double time_clock_ms = static_cast<double>(end_clock - start_clock) / CLOCKS_PER_SEC * 1000.0;

    // --- 方法 2: chrono::high_resolution_clock ---
    // 精度最高的时钟，但在某些平台上可能等价于 system_clock（受 NTP 调整影响）
    auto start_hrc = std::chrono::high_resolution_clock::now();
    matmulNaive(matA.data(), matB.data(), matC.data(), MAT_SIZE, MAT_SIZE, MAT_SIZE);
    auto end_hrc = std::chrono::high_resolution_clock::now();
    double time_hrc_ms = std::chrono::duration<double, std::milli>(end_hrc - start_hrc).count();

    // --- 方法 3: chrono::steady_clock（推荐！）---
    // 保证单调递增: 后调用 now() 返回的时间一定 >= 前调用
    // 不受 NTP 校时、休眠唤醒等系统事件影响
    // 这是性能测量中应该使用的时钟
    auto start_steady = std::chrono::steady_clock::now();
    matmulNaive(matA.data(), matB.data(), matC.data(), MAT_SIZE, MAT_SIZE, MAT_SIZE);
    auto end_steady = std::chrono::steady_clock::now();
    double time_steady_ms = std::chrono::duration<double, std::milli>(end_steady - start_steady).count();

    printf("   clock():               %8.3f ms\n", time_clock_ms);
    printf("   high_resolution_clock:  %8.3f ms\n", time_hrc_ms);
    printf("   steady_clock:          %8.3f ms  <-- 推荐\n", time_steady_ms);
    printf("\n   结论: steady_clock 单调递增、精度足够，是性能测量的最佳选择。\n\n");

    // ========================================================================
    // 实验 2: Warmup（预热）效应
    // ========================================================================
    printf("2. Warmup 效应:\n");
    printf("   原理: 首几次执行较慢，原因包括:\n");
    printf("   - 指令缓存(I-Cache)冷启动: CPU 首次执行代码需从内存加载指令\n");
    printf("   - 数据缓存(D-Cache)冷启动: 输入数据需从内存加载到缓存\n");
    printf("   - CPU 频率提升: 现代 CPU 有 turbo boost，需要几毫秒才能提升到最高频率\n");
    printf("   - 操作系统按需分配: 内存页首次访问触发缺页中断\n\n");

    // 运行 100 次，打印前 15 次，观察耗时递减趋势
    const int WARMUP_ITER = 100;
    std::vector<double> warmupTimes(WARMUP_ITER);

    for (int i = 0; i < WARMUP_ITER; ++i) {
        // 每次循环都完整执行一次矩阵乘法
        auto t0 = std::chrono::steady_clock::now();
        matmulNaive(matA.data(), matB.data(), matC.data(), MAT_SIZE, MAT_SIZE, MAT_SIZE);
        auto t1 = std::chrono::steady_clock::now();
        warmupTimes[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 打印前 15 次和第 50、100 次
        if (i < 15 || i == 49 || i == 99) {
            const char* tag = "";
            if (i == 0) tag = " <-- 最慢(冷启动)";
            else if (i == 4) tag = " <-- 即将预热完成";
            else if (i == 9) tag = " (已预热)";
            else if (i == 49) tag = " (已预热)";
            else if (i == 99) tag = " (已预热)";
            printf("   第%3d次: %8.3f ms%s\n", i + 1, warmupTimes[i], tag);
        }
    }

    printf("\n   观察: 前几次明显比后面的慢，这就是 warmup 效应!\n");
    printf("   实践: 丢弃前 5~10 次的测量结果，只统计后面的。\n\n");

    // ========================================================================
    // 实验 3: 计时统计（去掉 warmup 后）
    // ========================================================================
    printf("3. 计时统计 (去掉前5次warmup):\n");
    printf("   原理: 性能测量需要统计方法，而非只看单次结果:\n");
    printf("   - 均值: 反映平均性能，但受极端值影响\n");
    printf("   - 中位数: 更稳健，不受偶发高延迟影响\n");
    printf("   - 最小值: 反映最好情况，接近硬件真实能力\n");
    printf("   - 最大值: 反映最坏情况，可能包含操作系统调度延迟\n\n");

    // 去掉前 5 次 warmup
    const int SKIP = 5;
    std::vector<double> validTimes(warmupTimes.begin() + SKIP, warmupTimes.end());

    double meanVal   = computeMean(validTimes);
    double medianVal = computeMedian(validTimes);
    double minVal    = *std::min_element(validTimes.begin(), validTimes.end());
    double maxVal    = *std::max_element(validTimes.begin(), validTimes.end());
    double stddevVal = computeStdDev(validTimes);

    printf("   样本数: %d (去掉前 %d 次 warmup)\n", (int)validTimes.size(), SKIP);
    printf("   均值:   %8.3f ms\n", meanVal);
    printf("   中位数: %8.3f ms\n", medianVal);
    printf("   最小值: %8.3f ms  <-- 硬件能力上界\n", minVal);
    printf("   最大值: %8.3f ms  <-- 受系统调度影响\n", maxVal);
    printf("   标准差: %8.3f ms  <-- 离散程度\n\n", stddevVal);

    // ========================================================================
    // 实验 4: 测量开销 —— 空循环计时
    // ========================================================================
    printf("4. 计时器开销:\n");
    printf("   原理: 计时器本身也需要时间（系统调用、缓存失效等）。\n");
    printf("   用空循环测量计时器的基础开销，判断测量是否可信。\n\n");

    // 测量空循环的时间
    const int EMPTY_ITER = 10000;
    auto emptyStart = std::chrono::steady_clock::now();
    for (int i = 0; i < EMPTY_ITER; ++i) {
        // 只有循环变量递增和条件判断，几乎没有实际计算
        // volatile 防止编译器优化掉这个循环
        volatile int dummy = i;
        (void)dummy;
    }
    auto emptyEnd = std::chrono::steady_clock::now();
    double emptyTimeMs = std::chrono::duration<double, std::milli>(emptyEnd - emptyStart).count();
    double perIterOverhead = emptyTimeMs / EMPTY_ITER;

    printf("   空循环 %d 次总耗时: %8.3f ms\n", EMPTY_ITER, emptyTimeMs);
    printf("   每次循环开销: %8.6f ms (%.3f us)\n", perIterOverhead, perIterOverhead * 1000.0);
    printf("   对比矩阵乘法耗时: %8.3f ms\n\n", meanVal);

    if (perIterOverhead < meanVal * 0.01) {
        printf("   结论: 计时器开销远小于测量对象，结果可信。\n\n");
    } else {
        printf("   警告: 计时器开销接近测量对象的 1%%，需注意误差!\n\n");
    }

    // ========================================================================
    // 实验 5: 矩阵乘法正式测量 (128x128)
    // ========================================================================
    printf("5. 矩阵乘法正式测量 (128x128):\n");
    printf("   使用正确的测量方法论: warmup + 多次测量 + 统计分析\n\n");

    // 先做 warmup
    for (int i = 0; i < 10; ++i) {
        matmulNaive(matA.data(), matB.data(), matC.data(), MAT_SIZE, MAT_SIZE, MAT_SIZE);
    }

    // 正式测量 50 次
    const int FORMAL_ITER = 50;
    std::vector<double> formalTimes(FORMAL_ITER);
    for (int i = 0; i < FORMAL_ITER; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        matmulNaive(matA.data(), matB.data(), matC.data(), MAT_SIZE, MAT_SIZE, MAT_SIZE);
        auto t1 = std::chrono::steady_clock::now();
        formalTimes[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // 统计分析
    double formalMean   = computeMean(formalTimes);
    double formalMedian = computeMedian(formalTimes);
    double formalMin    = *std::min_element(formalTimes.begin(), formalTimes.end());
    double formalMax    = *std::max_element(formalTimes.begin(), formalTimes.end());

    // 计算理论 FLOPs:
    //   矩阵乘法 C[NxM] = A[NxK] * B[KxM]
    //   每个输出元素需要 K 次乘法 + K 次加法 = 2K 次浮点运算
    //   总 FLOPs = 2 * N * M * K
    //   对于方阵: N = M = K = 128, 总 FLOPs = 2 * 128^3 = 4,194,304
    int64_t totalFlops = 2LL * MAT_SIZE * MAT_SIZE * MAT_SIZE;

    // 使用项目 MathUtils 中的函数计算 GFLOPS
    double gflopsMean = edgeai::gflops(totalFlops, formalMean);
    double gflopsMedian = edgeai::gflops(totalFlops, formalMedian);
    double gflopsMin    = edgeai::gflops(totalFlops, formalMin);
    double gflopsMax    = edgeai::gflops(totalFlops, formalMax);

    printf("   矩阵规模: %d x %d (方阵)\n", MAT_SIZE, MAT_SIZE);
    printf("   理论 FLOPs: %lld (%.3f MFLOPs)\n",
           (long long)totalFlops, static_cast<double>(totalFlops) / 1e6);
    printf("   耗时(均值):   %8.3f ms -> %8.3f GFLOPS\n", formalMean, gflopsMean);
    printf("   耗时(中位数): %8.3f ms -> %8.3f GFLOPS\n", formalMedian, gflopsMedian);
    printf("   耗时(最小值): %8.3f ms -> %8.3f GFLOPS\n", formalMin, gflopsMin);
    printf("   耗时(最大值): %8.3f ms -> %8.3f GFLOPS\n\n", formalMax, gflopsMax);

    // 性能解读
    printf("   性能解读:\n");
    printf("   - 典型 CPU 单核浮点峰值: ~20-80 GFLOPS (取决于频率和 SIMD 宽度)\n");
    printf("   - 朴素实现通常只利用 1-5%% 的峰值算力\n");
    printf("   - 优化方向: 分块(tiling)提升缓存命中率、SIMD(AVX2)并行化、\n");
    printf("     多线程(OpenMP)、算法变换(Strassen/Winograd)\n\n");

    printf("=== 实验完成 ===\n");
    printf("关键收获:\n");
    printf("1. 使用 steady_clock 做性能测量（单调递增，不受 NTP 影响）\n");
    printf("2. 测量前先 warmup（丢弃前 5~10 次结果）\n");
    printf("3. 多次测量取统计值（均值/中位数/最小值），不要只看单次\n");
    printf("4. 用 GFLOPS 衡量计算效率 = FLOPs / 耗时 / 1e9\n");

    return 0;
}
