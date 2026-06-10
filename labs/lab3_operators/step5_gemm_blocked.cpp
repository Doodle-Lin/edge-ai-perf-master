/**
 * @file step5_gemm_blocked.cpp
 * @brief Lab3 Step5: 分块 GEMM —— 经典的缓存优化矩阵乘法
 *
 * ====== 本实验的学习目标 ======
 * 1. 理解朴素 GEMM 为什么有 O(N^3) 缓存缺失
 * 2. 掌握分块 GEMM 的三层分块策略: L1 微内核、L2 块、L3 面板
 * 3. 理解 Packing (打包) 的作用: 将不连续的数据拷贝到连续缓冲区
 * 4. 扫描不同分块大小，观察性能曲线
 * 5. 理解: 这就是 BLAS 库 (OpenBLAS, MKL) 内部做的事情
 *
 * ====== 朴素 GEMM 的问题 ======
 *
 * C[M,N] = A[M,K] * B[K,N]
 *
 * 朴素三重循环:
 *   for i in 0..M:
 *     for j in 0..N:
 *       for p in 0..K:
 *         C[i][j] += A[i][p] * B[p][j]
 *
 * 问题: B 按列访问 (B[p][j] 步长为 N)
 *   - 每次 B[p][j] → B[p+1][j] 要跳过 N 个 float = 4N 字节
 *   - 如果 N > 16 (cache line / sizeof(float) = 64/4 = 16)
 *     则 B[p][j] 和 B[p+1][j] 不在同一 cache line → cache miss
 *   - 对每个输出 C[i][j], K 次 B 访问中几乎全部 cache miss
 *   - 对 N=1024: B 的列元素间距 4KB，远超 cache line (64B)
 *
 * 缓存缺失次数:
 *   - A 的访问: 按行访问，M*K 次，几乎全部命中 → O(M*K) hits
 *   - B 的访问: 按列访问，M*N*K 次，几乎全部缺失 → O(M*N*K) misses!
 *   - C 的访问: 写入 M*N 次，可放在寄存器 → O(M*N) writes
 *   - 总缺失: O(M*N*K) → 和计算量 O(M*N*K) 同阶!
 *   - 这意味着: 几乎每次计算都要等内存 → 极低效
 *
 * ====== 分块 (Blocking / Tiling) 原理 ======
 *
 * 核心思想: 将大矩阵划分成小块，使小块能放入缓存
 *
 * 三层分块:
 *   for ii in 0..M step MC:          // 外层: 遍历行块 (L2 大小)
 *     for jj in 0..N step NC:        // 外层: 遍历列块 (L2 大小)
 *       for pp in 0..K step KC:      // 外层: 遍历 K 块 (L2 大小)
 *         // 内层: 计算 MC x NC 子矩阵，只访问 KC 列的 A 和 KC 行的 B
 *         for i in ii..ii+MC:
 *           for j in jj..jj+NC:
//             for p in pp..pp+KC:
 *               C[i][j] += A[i][p] * B[p][j]
 *
 * 关键: 当 MC*NC + MC*KC + KC*NC < L2_cache_size 时
 *        A 的块、B 的块、C 的块都在 L2 中 → 无 cache miss
 *
 * ====== 分块大小的选择 (以 Intel Skylake 为例) ======
 *
 *   L1: 32KB, L2: 256KB (per core), L3: shared
 *
 *   方案 1 (保守):
 *     MC=128, NC=256, KC=64:
 *       A 块: 128*64*4 = 32KB (可放入 L1!)
 *       B 块: 64*256*4 = 64KB
 *       C 块: 128*256*4 = 128KB
 *       A 块放 L1，B 和 C 放 L2 → 层次化缓存利用
 *
 *   方案 2 (激进):
 *     MC=256, NC=256, KC=128:
 *       A 块: 256*128*4 = 128KB
 *       B 块: 128*256*4 = 128KB
 *       C 块: 256*256*4 = 256KB
 *       总计 > 256KB L2 → 需要调整
 *
 * ====== Packing (打包) 优化 ======
 *
 * 即使分块后，A 和 B 的子矩阵在原矩阵中仍然不连续
 *   A[i][p] 的步长是 K (整行长度), 而非 KC (块宽度)
 *   B[p][j] 的步长是 N (整行长度), 而非 NC (块宽度)
 *
 * Packing: 在计算前将 A 和 B 的块拷贝到连续缓冲区
 *   - 拷贝的开销远小于避免 cache miss 的收益
 *   - 这就是"空间换时间": 用额外的内存换取连续的访问模式
 *
 * ====== 微内核 (Micro-kernel) ======
 *
 * 最内层的计算通常进一步优化为"微内核"
 *   - 处理 MR x NR 的小块 (如 4x8 或 6x8)
 *   - 利用寄存器暂存 MR 个 A 元素和 NR 个 B 元素
 *   - 寄存器速度 ≈ 0 周期延迟，是所有存储中最快的
 *   - 本实现中的微内核结合 AVX2 处理 8 个输出元素
 *
 * ====== 与 BLAS 库的关系 ======
 *
 * OpenBLAS、MKL、BLIS 都使用了相同的分块策略
 *   不同之处在于分块参数和微内核的精细程度
 *   本实现是教学版本，参数可调，方便理解各层分块的意义
 *
 * ====== 编译与运行 ======
 * g++ -O2 -std=c++17 -mavx2 -mfma -o step5_gemm_blocked step5_gemm_blocked.cpp
 * ./step5_gemm_blocked
 */

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

#ifdef USE_AVX2
#include <immintrin.h>
#endif
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
// GEMM FLOPs 计算: 2*M*N*K (1 MAC = 1 乘 + 1 加 = 2 FLOPs)
// ============================================================================
int64_t flopCountGemm(int M, int N, int K) {
    return 2LL * M * N * K;
}

// ============================================================================
// 朴素 GEMM: C[M,N] += A[M,K] * B[K,N]
// 三重循环: i → j → p
// 问题: B 按列访问, 缓存不友好
// ============================================================================
void naiveGemm(const float* A, int lda,
               const float* B, int ldb,
               float* C, int ldc,
               int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = C[i * ldc + j];
            for (int p = 0; p < K; p++) {
                sum += A[i * lda + p] * B[p * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

// ============================================================================
// 重排循环顺序的 GEMM: C[M,N] += A[M,K] * B[K,N]
// 循环顺序: i → p → j
// 优势: A[i][p] 在 p 循环外层不变，可以缓存在寄存器中
//        B[p][j] 按行访问，缓存友好
// ============================================================================
void reorderGemm(const float* A, int lda,
                 const float* B, int ldb,
                 float* C, int ldc,
                 int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int p = 0; p < K; p++) {
            float aVal = A[i * lda + p];  // A[i][p] 在寄存器中复用 N 次!
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] += aVal * B[p * ldb + j];
                //                  ↑ B 按行访问! 连续内存! 缓存友好!
            }
        }
    }
}

// ============================================================================
// AVX2 优化的 GEMM 微内核
// C[M,N] += A[M,K] * B[K,N]
// 内层循环用 _mm256_fmadd_ps 同时处理 8 个元素
// ============================================================================
void avx2Gemm(const float* A, int lda,
              const float* B, int ldb,
              float* C, int ldc,
              int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        float* cRow = C + i * ldc;
        const float* aRow = A + i * lda;
        for (int p = 0; p < K; p++) {
            __m256 aVal = _mm256_set1_ps(aRow[p]);
            int j = 0;
            for (; j + 7 < N; j += 8) {
                __m256 bVal = _mm256_loadu_ps(&B[p * ldb + j]);
                __m256 cVal = _mm256_loadu_ps(&cRow[j]);
                __m256 result = _mm256_fmadd_ps(aVal, bVal, cVal);
                _mm256_storeu_ps(&cRow[j], result);
            }
            for (; j < N; j++) {
                cRow[j] += aRow[p] * B[p * ldb + j];
            }
        }
    }
}

// ============================================================================
// 打包 A 矩阵的一个面板 (panel)
//
// 将 A 中 [iStart, iStart+mc) x [pStart, pStart+kc) 的元素
// 拷贝到连续缓冲区 packedA 中
//
// 打包前: A[i][p] 的步长是 lda (整行长度)，不连续
// 打包后: packedA[0..mc*kc-1] 连续存储，按行优先
// 好处: 计算微内核时 A 的访问完全连续，无 cache miss
// ============================================================================
void packA(const float* A, int lda, float* packedA,
           int iStart, int pStart, int mc, int kc) {
    for (int i = 0; i < mc; i++) {
        const float* aRow = A + (iStart + i) * lda + pStart;
        float* pRow = packedA + i * kc;
        // 拷贝一行中的 kc 个元素到 packedA
        // 注意: A 中的元素在内存中是连续的 (同一行内)
        // 但跨行时跳跃 lda 个元素
        // 打包后，所有行紧密排列，没有间隔
        for (int p = 0; p < kc; p++) {
            pRow[p] = aRow[p];
        }
    }
}

// ============================================================================
// 打包 B 矩阵的一个面板
//
// 将 B 中 [pStart, pStart+kc) x [jStart, jStart+nc) 的元素
// 拷贝到连续缓冲区 packedB 中
//
// 打包前: B[p][j] 的步长是 ldb (整行长度)
// 打包后: packedB[0..kc*nc-1] 连续存储，按行优先
// ============================================================================
void packB(const float* B, int ldb, float* packedB,
           int pStart, int jStart, int kc, int nc) {
    for (int p = 0; p < kc; p++) {
        const float* bRow = B + (pStart + p) * ldb + jStart;
        float* pRow = packedB + p * nc;
        for (int j = 0; j < nc; j++) {
            pRow[j] = bRow[j];
        }
    }
}

// ============================================================================
// 分块 GEMM 主函数
//
// C[M,N] += A[M,K] * B[K,N]
// 三层分块循环 + 内层 AVX2 微内核
//
// 分块层次:
//   L3 (面板): 遍历 K 方向的分块 (pp)
//   L2 (块):   遍历 M 和 N 方向的分块 (ii, jj)
//   L1 (微内核): 在 packed 数据上做 AVX2 乘法
// ============================================================================
void blockedGemm(const float* A, int lda,
                 const float* B, int ldb,
                 float* C, int ldc,
                 int M, int N, int K,
                 int MC, int NC, int KC) {
    // 分配打包缓冲区
    // packedA 大小: MC * KC (行优先)
    // packedB 大小: KC * NC (行优先)
    std::vector<float> packedA(MC * KC);
    std::vector<float> packedB(KC * NC);

    // ================================================================
    // 三层分块循环
    //
    // 最外层: 遍历 K 方向的分块 (pp)
    //   - 每次只取 A 的 KC 列和 B 的 KC 行
    //   - 这样 A 的面板和 B 的面板可以放入 L2
    //
    // 中层: 遍历 M 方向的分块 (ii)
    //   - A 的 MC 行可以放入 L2
    //
    // 内层: 遍历 N 方向的分块 (jj)
    //   - B 的 NC 列和 C 的 MC*NC 子矩阵可以放入 L2
    // ================================================================

    // 遍历 K 方向分块 (最外层)
    for (int pp = 0; pp < K; pp += KC) {
        int kc = std::min(KC, K - pp);  // 实际块大小 (边界处理)

        // 遍历 N 方向分块
        for (int jj = 0; jj < N; jj += NC) {
            int nc = std::min(NC, N - jj);

            // 打包 B 的面板: B[pp:pp+kc, jj:jj+nc]
            // 打包后: packedB[0..kc*nc-1] 连续存储
            packB(B, ldb, packedB.data(), pp, jj, kc, nc);

            // 遍历 M 方向分块
            for (int ii = 0; ii < M; ii += MC) {
                int mc = std::min(MC, M - ii);

                // 打包 A 的面板: A[ii:ii+mc, pp:pp+kc]
                // 打包后: packedA[0..mc*kc-1] 连续存储
                packA(A, lda, packedA.data(), ii, pp, mc, kc);

                // ====================================================
                // 内层: 在打包数据上执行微内核
                //
                // 此时:
                // - packedA 是连续的 mc x kc 矩阵
                // - packedB 是连续的 kc x nc 矩阵
                // - C 的子矩阵: C[ii:ii+mc, jj:jj+nc]
                //
                // 因为 packedA 和 packedB 是连续的，
                // 微内核中的访问都是顺序的，缓存命中率极高
                // ====================================================
                avx2Gemm(packedA.data(), kc,           // A: mc x kc, lda = kc (连续!)
                         packedB.data(), nc,           // B: kc x nc, ldb = nc (连续!)
                         C + ii * ldc + jj, ldc,       // C: mc x nc
                         mc, nc, kc);
            }
        }
    }
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    // ================================================================
    // GEMM 问题规模
    // 使用与 im2col 后卷积对应的矩阵大小:
    //   im2col 后: A = [50176, 27], B = [27, 64]
    //   这里用更通用的方阵测试
    // ================================================================
    const int M = 512;
    const int N = 512;
    const int K = 512;

    printf("=== Lab3 Step5: 分块 GEMM (Blocked GEMM) ===\n");
    printf("问题规模: C[%d,%d] = A[%d,%d] * B[%d,%d]\n", M, N, M, K, K, N);

    int64_t flops = flopCountGemm(M, N, K);
    printf("FLOPs: %lld (%.2fG)\n", (long long)flops, flops / 1e9);

    // 分配矩阵 (行优先存储)
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_naive(M * N);
    std::vector<float> C_reorder(M * N);
    std::vector<float> C_avx2(M * N);
    std::vector<float> C_blocked(M * N);

    // 用随机值初始化
    srand(42);
    for (auto& v : A) v = -1.0f + 2.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    for (auto& v : B) v = -1.0f + 2.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    const int NUM_ITERS = 5;

    // ================================================================
    // 1. 朴素 GEMM (i-j-p 循环顺序)
    // ================================================================
    printf("\n--- 1. 朴素 GEMM (i-j-p, B按列访问) ---\n");
    for (int i = 0; i < M * N; i++) C_naive[i] = 0.0f;
    naiveGemm(A.data(), K, B.data(), N, C_naive.data(), N, M, N, K);

    double totalNaiveMs = 0.0;
    for (int iter = 0; iter < NUM_ITERS; iter++) {
        for (int i = 0; i < M * N; i++) C_naive[i] = 0.0f;
        Timer t;
        t.start();
        naiveGemm(A.data(), K, B.data(), N, C_naive.data(), N, M, N, K);
        t.stop();
        totalNaiveMs += t.elapsedMs();
    }
    double avgNaiveMs = totalNaiveMs / NUM_ITERS;
    double naiveGf = static_cast<double>(flops) / (avgNaiveMs * 1e6);
    printf("  平均耗时: %.2f ms, GFLOPS: %.2f\n", avgNaiveMs, naiveGf);

    // ================================================================
    // 2. 重排循环顺序的 GEMM (i-p-j)
    // ================================================================
    printf("\n--- 2. 重排 GEMM (i-p-j, B按行访问) ---\n");
    for (int i = 0; i < M * N; i++) C_reorder[i] = 0.0f;
    reorderGemm(A.data(), K, B.data(), N, C_reorder.data(), N, M, N, K);

    double totalReorderMs = 0.0;
    for (int iter = 0; iter < NUM_ITERS; iter++) {
        for (int i = 0; i < M * N; i++) C_reorder[i] = 0.0f;
        Timer t;
        t.start();
        reorderGemm(A.data(), K, B.data(), N, C_reorder.data(), N, M, N, K);
        t.stop();
        totalReorderMs += t.elapsedMs();
    }
    double avgReorderMs = totalReorderMs / NUM_ITERS;
    double reorderGf = static_cast<double>(flops) / (avgReorderMs * 1e6);
    printf("  平均耗时: %.2f ms, GFLOPS: %.2f (%.2fx vs naive)\n",
           avgReorderMs, reorderGf, avgNaiveMs / avgReorderMs);

    // ================================================================
    // 3. AVX2 GEMM (i-p-j + SIMD)
    // ================================================================
    printf("\n--- 3. AVX2 GEMM (i-p-j + SIMD) ---\n");
    for (int i = 0; i < M * N; i++) C_avx2[i] = 0.0f;
    avx2Gemm(A.data(), K, B.data(), N, C_avx2.data(), N, M, N, K);

    double totalAvx2Ms = 0.0;
    for (int iter = 0; iter < NUM_ITERS; iter++) {
        for (int i = 0; i < M * N; i++) C_avx2[i] = 0.0f;
        Timer t;
        t.start();
        avx2Gemm(A.data(), K, B.data(), N, C_avx2.data(), N, M, N, K);
        t.stop();
        totalAvx2Ms += t.elapsedMs();
    }
    double avgAvx2Ms = totalAvx2Ms / NUM_ITERS;
    double avx2Gf = static_cast<double>(flops) / (avgAvx2Ms * 1e6);
    printf("  平均耗时: %.2f ms, GFLOPS: %.2f (%.2fx vs naive)\n",
           avgAvx2Ms, avx2Gf, avgNaiveMs / avgAvx2Ms);

    // ================================================================
    // 4. 分块 GEMM (不同分块大小)
    // ================================================================
    printf("\n--- 4. 分块 GEMM (不同分块大小) ---\n");

    struct BlockConfig {
        int MC, NC, KC;
        std::string label;
    };

    std::vector<BlockConfig> configs = {
        {64,  64,  64,  "64x64x64"},
        {128, 128, 64,  "128x128x64"},
        {128, 256, 64,  "128x256x64"},
        {256, 256, 64,  "256x256x64"},
        {128, 256, 128, "128x256x128"},
        {256, 256, 128, "256x256x128"},
    };

    printf("  %-18s %10s %8s %8s\n", "分块 (MCxNCxKC)", "耗时(ms)", "GFLOPS", "加速比");
    printf("  %-18s %10s %8s %8s\n", "----------------", "--------", "------", "------");

    struct GemmResult { std::string label; double ms; double gf; double speedup; };
    std::vector<GemmResult> gemmResults;

    for (const auto& cfg : configs) {
        // 检查分块大小是否合理 (不超过矩阵维度)
        if (cfg.MC > M || cfg.NC > N || cfg.KC > K) continue;

        for (int i = 0; i < M * N; i++) C_blocked[i] = 0.0f;
        blockedGemm(A.data(), K, B.data(), N, C_blocked.data(), N,
                    M, N, K, cfg.MC, cfg.NC, cfg.KC);

        double totalMs = 0.0;
        for (int iter = 0; iter < NUM_ITERS; iter++) {
            for (int i = 0; i < M * N; i++) C_blocked[i] = 0.0f;
            Timer t;
            t.start();
            blockedGemm(A.data(), K, B.data(), N, C_blocked.data(), N,
                        M, N, K, cfg.MC, cfg.NC, cfg.KC);
            t.stop();
            totalMs += t.elapsedMs();
        }

        double avgMs = totalMs / NUM_ITERS;
        double gf = static_cast<double>(flops) / (avgMs * 1e6);
        double speedup = avgNaiveMs / avgMs;

        gemmResults.push_back({cfg.label, avgMs, gf, speedup});

        printf("  %-18s %10.2f %8.2f %7.2fx\n",
               cfg.label.c_str(), avgMs, gf, speedup);
    }

    // ================================================================
    // 缓存占用的详细分析
    // ================================================================
    printf("\n--- 分块大小 vs 缓存占用分析 ---\n");
    printf("  L1: ~32 KB, L2: ~256 KB\n\n");
    for (const auto& cfg : configs) {
        double aBlock = cfg.MC * cfg.KC * 4.0 / 1024;  // KB
        double bBlock = cfg.KC * cfg.NC * 4.0 / 1024;
        double cBlock = cfg.MC * cfg.NC * 4.0 / 1024;
        double total = aBlock + bBlock + cBlock;
        printf("  %s: A=%.0fKB + B=%.0fKB + C=%.0fKB = %.0fKB %s\n",
               cfg.label.c_str(), aBlock, bBlock, cBlock, total,
               total < 256 ? "[L2]" : ">L2");
    }

    // ================================================================
    // ASCII 性能对比图
    // ================================================================
    printf("\n--- GFLOPS 对比图 ---\n");
    double maxGf = naiveGf;
    for (const auto& r : gemmResults) maxGf = std::max(maxGf, r.gf);
    int barW = 50;

    auto printBar = [&](const char* name, double gf) {
        int bar = static_cast<int>(gf / maxGf * barW);
        printf("  %-16s|", name);
        for (int i = 0; i < bar; i++) printf("#");
        printf(" %.1f\n", gf);
    };

    printBar("Naive (i-j-p)", naiveGf);
    printBar("Reorder (i-p-j)", reorderGf);
    printBar("AVX2", avx2Gf);
    for (const auto& r : gemmResults) {
        printBar(r.label.c_str(), r.gf);
    }

    // ================================================================
    // 正确性验证
    // ================================================================
    printf("\n--- 正确性验证 ---\n");
    // 用重排版本作为参考 (其正确性已通过与朴素版本对比确认)
    double maxDiffReorder = 0.0;
    for (int i = 0; i < M * N; i++) {
        double diff = std::fabs(C_naive[i] - C_reorder[i]);
        maxDiffReorder = std::max(maxDiffReorder, diff);
    }
    printf("  Reorder vs Naive: 最大误差 = %.2e\n", maxDiffReorder);

    // 用最优分块结果对比
    if (!gemmResults.empty()) {
        // 重新运行最优分块
        const auto& best = gemmResults[0];
        for (int i = 0; i < M * N; i++) C_blocked[i] = 0.0f;
        // 找到对应的配置
        for (const auto& cfg : configs) {
            if (cfg.label == best.label) {
                blockedGemm(A.data(), K, B.data(), N, C_blocked.data(), N,
                            M, N, K, cfg.MC, cfg.NC, cfg.KC);
                break;
            }
        }
        double maxDiffBlocked = 0.0;
        for (int i = 0; i < M * N; i++) {
            double diff = std::fabs(C_naive[i] - C_blocked[i]);
            maxDiffBlocked = std::max(maxDiffBlocked, diff);
        }
        printf("  Blocked vs Naive: 最大误差 = %.2e\n", maxDiffBlocked);
    }

    // ================================================================
    // 关键结论
    // ================================================================
    printf("\n--- 关键结论 ---\n");
    printf("  1. 朴素 GEMM (i-j-p): B 按列访问 → 大量 cache miss → 最慢\n");
    printf("  2. 重排 GEMM (i-p-j): B 按行访问 → 缓存友好 → 显著加速\n");
    printf("  3. AVX2 GEMM: SIMD 8x 并行 + 缓存友好 → 进一步加速\n");
    printf("  4. 分块 GEMM: packing + 分块 → 最优缓存利用 → 最高性能\n");
    printf("  5. 分块大小需要匹配 CPU 缓存: MC*KC 放 L1, 总计放 L2\n");
    printf("  6. 这就是 OpenBLAS/MKL 内部做的事情!\n");

    printf("\n=== Step5 完成 ===\n");
    printf("下一步: step6_vulkan_compute_basics.cpp —— GPU 计算入门\n");

    return 0;
}
