/**
 * @file GemmBlocked.h
 * @brief 分块 GEMM —— 经典的缓存优化矩阵乘法
 *
 * ====== 学习要点 ======
 * 1. 为什么 GEMM 需要分块？
 *    - 矩阵乘法 C[M,N] = A[M,K] * B[K,N]，朴素三重循环：
 *      for i in 0..M:
 *        for j in 0..N:
 *          for p in 0..K:
 *            C[i][j] += A[i][p] * B[p][j]
 *    - 问题：A 按行访问（缓存友好），但 B 按列访问（缓存不友好！）
 *    - B 的列访问跨越 N 个元素，步长为 N*sizeof(float)
 *    - 如果 N > cache_line_size / sizeof(float)，每访问一个 B 元素就是一次 cache miss
 *    - 对于 N=1024，B 的列元素间距 4KB，远超 cache line (64B)
 *
 * 2. 分块 (Blocking / Tiling) 原理：
 *    - 将大矩阵划分成小块，使小块能放入缓存
 *    - 三层分块：外层遍历块，内层遍历块内元素
 *      for ii in 0..M step MC:          // 外层：遍历行块
 *        for jj in 0..N step NC:        // 外层：遍历列块
 *          for pp in 0..K step KC:      // 外层：遍历 K 块
 *            // 内层：计算 MC x NC 子矩阵，只访问 KC 列的 A 和 KC 行的 B
 *            for i in ii..ii+MC:
 *              for j in jj..jj+NC:
 *                for p in pp..pp+KC:
 *                  C[i][j] += A[i][p] * B[p][j]
 *    - 关键：当 MC*NC + MC*KC + KC*NC < L2_cache_size 时
 *      A 的块、B 的块、C 的块都在 L2 中 → 无 cache miss
 *
 * 3. 分块大小的选择（以 Intel Skylake 为例）：
 *    - L1: 32KB, L2: 256KB (per core), L3: shared
 *    - MC=256, NC=256, KC=128:
 *      A 块: 256*128*4 = 128KB
 *      B 块: 128*256*4 = 128KB
 *      C 块: 256*256*4 = 256KB
 *      总计 > 256KB L2 → 需要调整
 *    - 更合理的选择：MC=128, NC=256, KC=64:
 *      A 块: 128*64*4 = 32KB (可放入 L1!)
 *      B 块: 64*256*4 = 64KB
 *      C 块: 128*256*4 = 128KB
 *      A 块放 L1，B 和 C 放 L2 → 层次化缓存利用
 *
 * 4. Packing（打包）优化：
 *    - 即使分块后，A 和 B 的子矩阵在原矩阵中仍然不连续
 *    - A[i][p] 的步长是 K（整行长度），而非 KC（块宽度）
 *    - Packing：在计算前将 A 和 B 的块拷贝到连续缓冲区
 *    - 拷贝的开销远小于避免 cache miss 的收益
 *    - 这就是"空间换时间"：用额外的内存换取连续的访问模式
 *
 * 5. 微内核 (Micro-kernel)：
 *    - 最内层的计算通常进一步优化为"微内核"
 *    - 微内核处理 MR x NR 的小块（如 4x8 或 6x8）
 *    - 利用寄存器暂存 MR 个 A 元素和 NR 个 B 元素
 *    - 寄存器速度 ≈ 0 周期延迟，是所有存储中最快的
 *    - 本实现中的微内核结合 AVX2 处理 8 个输出元素
 *
 * 6. 本实现与 BLAS 库的关系：
 *    - OpenBLAS、MKL、BLIS 都使用了相同的分块策略
 *    - 不同之处在于分块参数和微内核的精细程度
 *    - 本实现是教学版本，参数可调，方便理解各层分块的意义
 */

#pragma once

#include "IOperator.h"
#include <vector>

namespace edgeai {

class GemmBlocked : public IOperator {
public:
    /**
     * @brief 构造分块 GEMM 算子
     * @param M      左矩阵行数
     * @param N      右矩阵列数
     * @param K      左矩阵列数 = 右矩阵行数
     * @param MC     L2 分块行大小（默认 128）
     * @param NC     L2 分块列大小（默认 256）
     * @param KC     L2 分块 K 维大小（默认 64）
     *
     * 注意：GEMM 的 forward() 中 input 被解释为 A (M x K)，
     * output 被解释为 C (M x N)。B 矩阵作为权重存储。
     */
    GemmBlocked(int M, int N, int K,
                int MC = 128, int NC = 256, int KC = 64);

    void forward(const Tensor& input, Tensor& output) override;
    const char* name() const override { return "GemmBlocked"; }
    int64_t flops(const std::vector<int>& inputShape,
                  const std::vector<int>& outputShape) const override;

private:
    int M_, N_, K_;
    int MC_, NC_, KC_;  ///< 分块参数
    Tensor weightB_;    ///< B 矩阵 [K, N]，作为权重存储

    /**
     * @brief 打包 A 矩阵的一个面板 (panel)
     *
     * 将 A 中 [iStart, iStart+MC) x [pStart, pStart+KC) 的元素
     * 拷贝到连续缓冲区 packedA 中
     *
     * 打包后：packedA[0..MC*KC-1] 连续存储，按行优先
     * 好处：计算微内核时 A 的访问完全连续，无 cache miss
     *
     * @param A       原始 A 矩阵，行优先存储
     * @param lda     A 的行宽（leading dimension）
     * @param packedA 打包缓冲区，大小 MC*KC
     * @param iStart  行起始
     * @param pStart  列起始
     * @param mc      实际行数（可能小于 MC，边界情况）
     * @param kc      实际列数
     */
    void packA(const float* A, int lda, float* packedA,
               int iStart, int pStart, int mc, int kc);

    /**
     * @brief 打包 B 矩阵的一个面板
     *
     * 将 B 中 [pStart, pStart+KC) x [jStart, jStart+NC) 的元素
     * 拷贝到连续缓冲区 packedB 中
     *
     * @param B       原始 B 矩阵，行优先存储
     * @param ldb     B 的行宽
     * @param packedB 打包缓冲区，大小 KC*NC
     * @param pStart  行起始
     * @param jStart  列起始
     * @param kc      实际行数
     * @param nc      实际列数
     */
    void packB(const float* B, int ldb, float* packedB,
               int pStart, int jStart, int kc, int nc);

    /**
     * @brief 分块 GEMM 主函数
     *
     * C[M,N] += A[M,K] * B[K,N]
     * 三层分块循环 + 内层微内核
     */
    void blockedGemm(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc);
};

} // namespace edgeai
