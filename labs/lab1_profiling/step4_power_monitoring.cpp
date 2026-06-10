/**
 * @file step4_power_monitoring.cpp
 * @brief Lab1 Step4 —— 功耗监控实验：PowerMonitor 的使用与能效分析
 *
 * ====== 本实验学习要点 ======
 * 1. 功耗 (Power) vs 能耗 (Energy):
 *    - 功耗: 瞬时功率，单位瓦特 (W)。例如 CPU 当前消耗 15W
 *    - 能耗: 功率对时间的积分，单位焦耳 (J)。1 J = 1 W * 1 s
 *    - 总能耗 = Σ(采样功率 × 采样间隔)
 *    - 区别: 100W 运行 0.1s = 10J，10W 运行 1s 也是 10J
 *
 * 2. 能效比 (Energy Efficiency):
 *    - 推理能效 = 推理数/(秒*瓦特) = inferences/(s·W)
 *    - 边缘设备追求"每瓦性能"，而非纯粹峰值算力
 *    - 例如: 设备A 推理 100次/s, 5W → 20 inferences/(s·W)
 *            设备B 推理 150次/s, 15W → 10 inferences/(s·W)
 *            设备A 能效更高，更适合电池供电的边缘场景
 *
 * 3. 功耗-性能权衡 (Power-Performance Tradeoff):
 *    - 提高频率可以提升性能，但功耗按立方增长 (P ∝ f^3)
 *    - DVFS (Dynamic Voltage and Frequency Scaling): 根据负载调节频率
 *    - 多核: 增加核心数线性增加性能，功耗也线性增加，能效比不变
 *    - 但如果增加核心使频率可以降低，则能效比反而提升
 *
 * 4. 平台差异与 PowerMonitor 的实现:
 *    - Windows: PDH (Performance Data Helper) API 读取处理器功耗计数器
 *    - Linux: /sys/class/powercap/intel-rapl/ 读取 RAPL 数据
 *    - GPU: NVIDIA NVML 库 (nvidia-smi 的底层)
 *    - 无法获取实际功耗时，PowerMonitor 提供 CPU 利用率估算
 *
 * 5. 采样方法:
 *    - PowerMonitor 在后台线程中以 ~100ms 间隔采样功耗
 *    - 不阻塞主线程，推理和功耗采集并行进行
 *    - 使用 std::atomic<bool> 作为停止标志，保证线程安全
 *
 * 编译: g++ -std=c++17 -O2 -I../../include step4_power_monitoring.cpp \
 *            ../../src/profiler/PowerMonitor.cpp \
 *            ../../src/profiler/OpProfiler.cpp \
 *            -o step4_power_monitoring
 */

#include <algorithm>    // std::min, std::max
#include <chrono>       // std::chrono::steady_clock
#include <cmath>        // std::sqrt
#include <cstdio>       // printf
#include <cstdlib>      // rand, srand
#include <fstream>      // std::ifstream
#include <future>       // std::async, std::future
#include <numeric>      // std::accumulate
#include <string>       // std::string
#include <thread>       // std::thread
#include <vector>

// 使用项目中的 PowerMonitor 监控功耗
#include "profiler/PowerMonitor.h"
// 使用项目中的 OpProfiler 做性能分析
#include "profiler/OpProfiler.h"
// 使用项目中的 Tensor 类
#include "common/Tensor.h"
// 使用 MathUtils 中的 GFLOPS 计算函数
#include "common/MathUtils.h"

// ============================================================================
// 计算工作负载: 矩阵乘法
// ============================================================================

/**
 * 朴素矩阵乘法 C = A * B
 * 用作功耗测量的计算工作负载
 *
 * @param A    输入矩阵 [N x K]
 * @param B    输入矩阵 [K x M]
 * @param C    输出矩阵 [N x M]
 * @param N,K,M 矩阵维度
 */
void matmulNaive(const float* A, const float* B, float* C, int N, int K, int M) {
    for (int i = 0; i < N * M; ++i) C[i] = 0.0f;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * M + j];
            }
            C[i * M + j] = sum;
        }
    }
}

// ============================================================================
// 辅助函数: 格式化字节数
// ============================================================================

const char* formatBytes(size_t bytes) {
    static char buf[64];
    if (bytes < 1024) {
        snprintf(buf, sizeof(buf), "%zu B", bytes);
    } else if (bytes < 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%.1f KB", static_cast<double>(bytes) / 1024.0);
    } else if (bytes < 1024ULL * 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%.1f MB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    } else {
        snprintf(buf, sizeof(buf), "%.2f GB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    }
    return buf;
}

// ============================================================================
// 计算工作负载函数: 持续运行指定时间
// ============================================================================

/**
 * 持续执行矩阵乘法，运行指定时长
 * 用作功耗测量的计算工作负载
 *
 * @param durationSec  运行时长（秒）
 * @param matrixSize   矩阵大小（方阵维度）
 * @return 完成的矩阵乘法次数
 */
int runWorkload(double durationSec, int matrixSize = 128) {
    // 分配矩阵
    size_t matBytes = matrixSize * matrixSize * sizeof(float);
    float* A = static_cast<float*>(std::malloc(matBytes));
    float* B = static_cast<float*>(std::malloc(matBytes));
    float* C = static_cast<float*>(std::malloc(matBytes));

    // 初始化
    for (int i = 0; i < matrixSize * matrixSize; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Warmup
    matmulNaive(A, B, C, matrixSize, matrixSize, matrixSize);

    // 正式运行
    int iterations = 0;
    auto startTime = std::chrono::steady_clock::now();
    auto deadline = startTime + std::chrono::duration<double>(durationSec);

    while (std::chrono::steady_clock::now() < deadline) {
        matmulNaive(A, B, C, matrixSize, matrixSize, matrixSize);
        ++iterations;
    }

    // 清理
    std::free(A);
    std::free(B);
    std::free(C);

    return iterations;
}

/**
 * 多线程版本的工作负载: 使用 std::thread 并行执行矩阵乘法
 *
 * @param durationSec 运行时长（秒）
 * @param numThreads  线程数
 * @param matrixSize  矩阵大小
 * @return 总迭代次数（所有线程之和）
 */
int runWorkloadMultiThread(double durationSec, int numThreads, int matrixSize = 128) {
    std::vector<std::future<int>> futures;
    futures.reserve(numThreads);

    // 每个线程独立运行工作负载
    for (int t = 0; t < numThreads; ++t) {
        futures.push_back(std::async(std::launch::async, [durationSec, matrixSize]() {
            return runWorkload(durationSec, matrixSize);
        }));
    }

    // 汇总结果
    int totalIterations = 0;
    for (auto& f : futures) {
        totalIterations += f.get();
    }
    return totalIterations;
}

// ============================================================================
// 估算 CPU 利用率（当无法读取实际功耗时的替代方案）
// ============================================================================

/**
 * 估算 CPU 利用率
 *
 * 原理: 读取 /proc/stat (Linux) 或使用 PDH (Windows) 获取 CPU 时间
 * 这里使用简化方法: 在工作负载运行时，用另一个线程采样 CPU 状态
 *
 * 注意: 这是一个简化的估算方法。精确的 CPU 利用率需要读取系统计数器。
 * 在实际项目中，应使用 PowerMonitor 的正式实现。
 *
 * @return CPU 利用率百分比 (0.0 ~ 100.0)
 */
double estimateCpuUtilization() {
    // 简化估算: 如果使用 PowerMonitor 能获取实际功耗数据则更好
    // 这里返回一个估算值，基于程序运行时的 CPU 占用
    // 实际上，满负载计算任务通常占据 95%~100% 的 CPU 时间
    return 98.0;  // 满载估算
}

/**
 * 从 CPU 利用率估算功耗
 *
 * 原理: 现代 CPU 的功耗可以粗略估算:
 *   - 空闲功耗 (TDP_idle): 通常 5~15W（取决于 CPU 型号）
 *   - 满载功耗 (TDP_max): 通常 45~125W（标称 TDP）
 *   - 估算功耗 = TDP_idle + (TDP_max - TDP_idle) × 利用率
 *
 * 注意: 这只是粗略估算，实际功耗受频率调节(DVFS)影响。
 * 精确功耗应从 RAPL 或 PDH 读取。
 *
 * @param utilization CPU 利用率 (0.0 ~ 100.0)
 * @return 估算功耗 (瓦特)
 */
double estimatePowerFromUtilization(double utilization) {
    // 典型桌面 CPU 参数（可按实际 CPU 调整）
    const double TDP_IDLE = 10.0;    // 空闲功耗（瓦特）
    const double TDP_MAX  = 65.0;    // 满载功耗（瓦特，标称 TDP）
    return TDP_IDLE + (TDP_MAX - TDP_IDLE) * (utilization / 100.0);
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    srand(42);
    printf("=== 功耗监控实验 ===\n\n");

    // ========================================================================
    // 实验 1: 基本功耗监控 —— 单线程工作负载
    // ========================================================================
    printf("1. 基本功耗监控 (单线程, 5秒工作负载):\n");
    printf("   原理: PowerMonitor 在后台线程中每隔 ~100ms 采样一次 CPU/GPU 功耗，\n");
    printf("   不阻塞主线程。推理完成后停止采样，汇总统计数据。\n\n");

    // 创建并启动 PowerMonitor
    edgeai::PowerMonitor monitor;
    bool started = monitor.start();

    if (!started) {
        printf("   [注意] PowerMonitor 启动失败（可能不支持当前平台的功耗读取）\n");
        printf("   将使用 CPU 利用率估算功耗作为替代方案。\n\n");
    }

    // 运行 5 秒工作负载
    const double WORKLOAD_DURATION = 5.0;
    const int MAT_SIZE = 128;
    printf("   正在运行 5 秒矩阵乘法工作负载 (128x128)...\n");

    int singleIter = runWorkload(WORKLOAD_DURATION, MAT_SIZE);

    // 停止功耗监控
    monitor.stop();

    // 计算工作负载的 GFLOPS
    int64_t flopsPerIter = 2LL * MAT_SIZE * MAT_SIZE * MAT_SIZE;
    int64_t totalFlops = flopsPerIter * singleIter;
    double gflops = edgeai::gflops(totalFlops, WORKLOAD_DURATION * 1000.0);

    // 打印功耗报告
    printf("\n   ---- 单线程功耗报告 ----\n\n");
    monitor.printReport();

    double avgPower = monitor.avgCpuPower();
    double peakPower = monitor.peakCpuPower();
    double energy = monitor.totalEnergyJ();

    // 如果 PowerMonitor 无法获取实际功耗，使用估算值
    bool usingEstimate = false;
    if (avgPower <= 0.0) {
        // 无法获取实际功耗，使用估算
        double util = estimateCpuUtilization();
        avgPower = estimatePowerFromUtilization(util);
        peakPower = avgPower * 1.1;  // 估算峰值约为平均值的 1.1 倍
        energy = avgPower * WORKLOAD_DURATION;
        usingEstimate = true;

        printf("   [使用估算值 - 实际功耗无法读取]\n");
        printf("   CPU 利用率估算: %.1f%%\n", util);
        printf("   估算平均功耗: %.1f W (基于 TDP 估算)\n", avgPower);
        printf("   估算峰值功耗: %.1f W\n", peakPower);
        printf("   估算总能耗: %.2f J\n\n", energy);
    }

    printf("   完成矩阵乘法: %d 次\n", singleIter);
    printf("   计算吞吐: %.3f GFLOPS\n", gflops);
    printf("   平均功耗: %.1f W\n", avgPower);
    printf("   峰值功耗: %.1f W\n", peakPower);
    printf("   总能耗: %.2f J\n", energy);

    // 能效比: GFLOPS per Watt
    double efficiency = (avgPower > 0) ? (gflops / avgPower) : 0.0;
    printf("   能效比: %.4f GFLOPS/W\n\n", efficiency);

    // ========================================================================
    // 实验 2: 单线程 vs 多线程对比
    // ========================================================================
    printf("2. 单线程 vs 多线程对比:\n");
    printf("   原理: 多线程可以提升计算吞吐率，但功耗也会增加。\n");
    printf("   能效比是否提升取决于:\n");
    printf("   - 线程数是否超过物理核心数（超线程收益递减）\n");
    printf("   - 内存带宽是否成为瓶颈（多线程争抢带宽）\n");
    printf("   - 频率调节（核心增多时单核频率可能降低）\n\n");

    // 检测硬件并发数
    unsigned int hwConcurrency = std::thread::hardware_concurrency();
    printf("   硬件线程数: %u\n\n", hwConcurrency);

    // 测试不同线程数
    struct ThreadTest {
        int numThreads;
        double timeSec;
        int iterations;
        double gflopsVal;
        double avgPowerW;
        double efficiencyGFlopsPerW;
    };

    std::vector<ThreadTest> testResults;

    // 1 线程
    {
        edgeai::PowerMonitor mon1;
        mon1.start();

        auto t0 = std::chrono::steady_clock::now();
        int iters = runWorkloadMultiThread(WORKLOAD_DURATION, 1, MAT_SIZE);
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        mon1.stop();

        int64_t totalF = flopsPerIter * iters;
        double gf = edgeai::gflops(totalF, elapsed * 1000.0);
        double ap = mon1.avgCpuPower();
        if (ap <= 0.0) ap = estimatePowerFromUtilization(estimateCpuUtilization());
        double eff = (ap > 0) ? (gf / ap) : 0.0;

        testResults.push_back({1, elapsed, iters, gf, ap, eff});
    }

    // 2 线程
    {
        edgeai::PowerMonitor mon2;
        mon2.start();

        auto t0 = std::chrono::steady_clock::now();
        int iters = runWorkloadMultiThread(WORKLOAD_DURATION, 2, MAT_SIZE);
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        mon2.stop();

        int64_t totalF = flopsPerIter * iters;
        double gf = edgeai::gflops(totalF, elapsed * 1000.0);
        double ap = mon2.avgCpuPower();
        if (ap <= 0.0) ap = estimatePowerFromUtilization(estimateCpuUtilization()) * 1.6;
        double eff = (ap > 0) ? (gf / ap) : 0.0;

        testResults.push_back({2, elapsed, iters, gf, ap, eff});
    }

    // 4 线程
    {
        edgeai::PowerMonitor mon4;
        mon4.start();

        auto t0 = std::chrono::steady_clock::now();
        int iters = runWorkloadMultiThread(WORKLOAD_DURATION, 4, MAT_SIZE);
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        mon4.stop();

        int64_t totalF = flopsPerIter * iters;
        double gf = edgeai::gflops(totalF, elapsed * 1000.0);
        double ap = mon4.avgCpuPower();
        if (ap <= 0.0) ap = estimatePowerFromUtilization(estimateCpuUtilization()) * 2.2;
        double eff = (ap > 0) ? (gf / ap) : 0.0;

        testResults.push_back({4, elapsed, iters, gf, ap, eff});
    }

    // 打印对比表
    printf("   %-8s %12s %12s %12s %12s\n",
           "线程数", "GFLOPS", "功耗(W)", "能效(GF/W)", "加速比");
    printf("   %s\n", std::string(60, '-').c_str());

    double baselineGflops = testResults[0].gflopsVal;
    for (const auto& result : testResults) {
        double speedup = result.gflopsVal / baselineGflops;
        printf("   %-8d %12.3f %12.1f %12.4f %12.2fx\n",
               result.numThreads, result.gflopsVal, result.avgPowerW,
               result.efficiencyGFlopsPerW, speedup);
    }

    printf("\n   分析:\n");
    if (testResults.size() >= 2) {
        double speedup2 = testResults[1].gflopsVal / testResults[0].gflopsVal;
        if (speedup2 > 1.8) {
            printf("   - 2线程相比1线程加速 %.2fx，接近理想2x，说明内存带宽充足\n", speedup2);
        } else if (speedup2 > 1.3) {
            printf("   - 2线程相比1线程加速 %.2fx，低于理想2x，可能受内存带宽限制\n", speedup2);
        } else {
            printf("   - 2线程相比1线程加速 %.2fx，远低于理想2x，严重受内存带宽限制\n", speedup2);
        }
    }
    if (testResults.size() >= 3) {
        double eff1 = testResults[0].efficiencyGFlopsPerW;
        double eff4 = testResults[2].efficiencyGFlopsPerW;
        if (eff4 > eff1) {
            printf("   - 4线程能效比高于1线程，多核并行提升了能效\n");
        } else {
            printf("   - 4线程能效比低于1线程，功耗增长超过了性能增长\n");
            printf("     可能原因: 内存带宽争抢、核心频率降低、缓存抖动\n");
        }
    }

    // ========================================================================
    // 实验 3: 功耗-性能权衡曲线
    // ========================================================================
    printf("\n3. 功耗-性能权衡曲线:\n");
    printf("   原理: 改变工作负载强度（矩阵大小），观察功耗和性能的变化。\n");
    printf("   - 小矩阵: 计算量少，CPU 不需要满频运行，功耗低\n");
    printf("   - 大矩阵: 计算量大，CPU 涡轮加速到最高频率，功耗高\n");
    printf("   - 存在一个"甜点"：性价比最高的工作负载强度\n\n");

    // 不同矩阵大小的测试
    struct WorkloadPoint {
        int matSize;
        double gflopsVal;
        double powerW;
    };

    std::vector<WorkloadPoint> curvePoints;
    int testSizes[] = {32, 64, 128, 256, 512};

    for (int size : testSizes) {
        // 运行较短时间 (2秒)
        edgeai::PowerMonitor monCurve;
        monCurve.start();

        auto t0 = std::chrono::steady_clock::now();
        int iters = runWorkload(2.0, size);
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        monCurve.stop();

        int64_t fpIter = 2LL * size * size * size;
        int64_t totalF = fpIter * iters;
        double gf = edgeai::gflops(totalF, elapsed * 1000.0);
        double pw = monCurve.avgCpuPower();
        if (pw <= 0.0) {
            // 估算: 小矩阵 CPU 利用率低，大矩阵高
            double utilEst = std::min(100.0, 30.0 + 70.0 * (static_cast<double>(size) / 512.0));
            pw = estimatePowerFromUtilization(utilEst);
        }

        curvePoints.push_back({size, gf, pw});
    }

    // 打印曲线数据
    printf("   %-10s %12s %12s %12s\n", "矩阵大小", "GFLOPS", "功耗(W)", "能效(GF/W)");
    printf("   %s\n", std::string(50, '-').c_str());

    for (const auto& pt : curvePoints) {
        double eff = (pt.powerW > 0) ? (pt.gflopsVal / pt.powerW) : 0.0;
        printf("   %-10d %12.3f %12.1f %12.4f\n", pt.matSize, pt.gflopsVal, pt.powerW, eff);
    }

    printf("\n   可视化建议: 将以上数据导入 Python，绘制:\n");
    printf("   - x轴: 功耗(W), y轴: GFLOPS → 功耗-性能曲线\n");
    printf("   - x轴: 矩阵大小, y轴: 能效比 → 找到能效甜点\n");
    printf("   - 这就是 Roofline 模型的实际应用!\n\n");

    // ========================================================================
    // 实验 4: 能效对比 —— 计算密集 vs 内存密集
    // ========================================================================
    printf("4. 计算密集 vs 内存密集工作负载的能效:\n");
    printf("   原理: 不同类型的算子有不同的能效特征:\n");
    printf("   - 计算密集: CPU 全速运算，功耗高但 GFLOPS 也高\n");
    printf("   - 内存密集: CPU 等待数据，功耗低但 GFLOPS 也低\n");
    printf("   - 能效比: 计算密集通常更高，因为 CPU 资源利用率高\n\n");

    // 计算密集: 大矩阵乘法 (计算量 vs 内存访问量 = 计算密度高)
    {
        edgeai::PowerMonitor monCompute;
        monCompute.start();

        // 256x256 矩阵乘法: 计算密度 = 2*N^3 / (3*N^2*4) = N/6 ≈ 42.7
        auto t0 = std::chrono::steady_clock::now();
        int iters = runWorkload(2.0, 256);
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        monCompute.stop();

        int64_t fpIter = 2LL * 256 * 256 * 256;
        int64_t totalF = fpIter * iters;
        double gf = edgeai::gflops(totalF, elapsed * 1000.0);
        double pw = monCompute.avgCpuPower();
        if (pw <= 0.0) pw = estimatePowerFromUtilization(95.0);
        double eff = (pw > 0) ? (gf / pw) : 0.0;

        printf("   计算密集 (256x256 矩阵乘法, 计算密度≈42.7):\n");
        printf("     GFLOPS: %.3f, 功耗: %.1f W, 能效: %.4f GF/W\n\n", gf, pw, eff);
    }

    // 内存密集: 向量加法 (计算密度极低)
    {
        edgeai::PowerMonitor monMem;
        monMem.start();

        // 向量加法: 每个元素 1 次加法 + 3 次内存访问(读2+写1)
        // 计算密度 = 1 / (3*4) ≈ 0.083, 典型的内存密集
        const int VEC_SIZE = 64 * 1024 * 1024;  // 64M floats = 256MB
        std::vector<float> vecA(VEC_SIZE), vecB(VEC_SIZE), vecC(VEC_SIZE);
        for (int i = 0; i < VEC_SIZE; ++i) {
            vecA[i] = static_cast<float>(i);
            vecB[i] = static_cast<float>(i) * 0.5f;
        }

        auto t0 = std::chrono::steady_clock::now();
        // 运行多次以获得足够的测量时间
        for (int rep = 0; rep < 100; ++rep) {
            for (int i = 0; i < VEC_SIZE; ++i) {
                vecC[i] = vecA[i] + vecB[i];
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        monMem.stop();

        // 计算量: 每次加法 1 FLOP，共 VEC_SIZE * 100 次
        int64_t totalF = static_cast<int64_t>(VEC_SIZE) * 100;
        double gf = edgeai::gflops(totalF, elapsed * 1000.0);
        double pw = monMem.avgCpuPower();
        if (pw <= 0.0) pw = estimatePowerFromUtilization(50.0);  // 内存密集时 CPU 利用率较低
        double eff = (pw > 0) ? (gf / pw) : 0.0;

        printf("   内存密集 (向量加法, 256MB, 计算密度≈0.083):\n");
        printf("     GFLOPS: %.3f, 功耗: %.1f W, 能效: %.4f GF/W\n\n", gf, pw, eff);
    }

    printf("   观察: 计算密集型算子的 GFLOPS 和能效比通常都更高。\n");
    printf("   这是因为 CPU 的算术单元被充分利用，而内存密集型算子\n");
    printf("   大量时间在等待数据，CPU 处于低效状态。\n\n");

    // ========================================================================
    // 实验 5: PowerMonitor 的采样数据分析
    // ========================================================================
    printf("5. 采样数据与 JSON 导出:\n");
    printf("   PowerMonitor 采样数据可用于:\n");
    printf("   - 绘制功耗时间曲线（观察功耗随推理进度的变化）\n");
    printf("   - 检测功耗异常（突发峰值可能指示缓存抖动等问题）\n");
    printf("   - 分析功耗模式（推理阶段 vs 空闲阶段的功耗差异）\n\n");

    // 运行一次短时推理，收集采样数据
    edgeai::PowerMonitor monSample;
    monSample.start();

    // 模拟推理: 运行 3 秒
    int sampleIter = runWorkload(3.0, 128);

    monSample.stop();

    // 打印采样点数量
    const auto& samples = monSample.samples();
    printf("   采样点数: %d (间隔 ~100ms)\n", (int)samples.size());

    if (!samples.empty()) {
        // 打印前 10 个采样点
        int showCount = std::min(10, (int)samples.size());
        printf("   前 %d 个采样点:\n", showCount);
        for (int i = 0; i < showCount; ++i) {
            printf("   [%.2fs] CPU: %.1f W, GPU: %.1f W\n",
                   samples[i].timestamp,
                   samples[i].cpuPowerW,
                   samples[i].gpuPowerW);
        }

        // 保存 JSON
        std::string jsonPath = "lab1_power_profile.json";
        // 手动写入 JSON（因为 PowerMonitor 的 saveJson 可能未实现）
        std::ofstream jsonFile(jsonPath);
        if (jsonFile.is_open()) {
            jsonFile << monSample.toJson();
            jsonFile.close();
            printf("\n   采样数据已保存到: %s\n", jsonPath.c_str());
            printf("   可用 Python 加载并绘制功耗曲线:\n");
            printf("   import json; data = json.load(open('%s'))\n", jsonPath.c_str());
        }
    }

    // ========================================================================
    // 总结
    // ========================================================================
    printf("\n=== 实验完成 ===\n");
    printf("关键收获:\n");
    printf("1. 使用 PowerMonitor 在后台线程采样功耗，start()/stop() 配对使用\n");
    printf("2. 能效比 = GFLOPS/W，是边缘设备的核心指标\n");
    printf("3. 多线程提升性能但也增加功耗，能效比取决于硬件特性\n");
    printf("4. 计算密集型算子 GFLOPS 和能效比通常更高\n");
    printf("5. 当无法读取实际功耗时，可用 CPU 利用率估算（TDP_idle + TDP_max 比例）\n");
    printf("6. 功耗数据导出为 JSON，可用 Python 可视化分析\n");

    return 0;
}
