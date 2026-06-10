/**
 * @file GemmBlocked.cpp
 * @brief 分块 GEMM 实现 —— 经典的缓存优化矩阵乘法
 *
 * ====== 实现说明 ======
 * 1. 三层分块：外层遍历 MC x NC x KC 的块，内层计算块内矩阵乘法
 * 2. 打包 (Packing)：将 A 和 B 的子矩阵拷贝到连续缓冲区，消除缓存行浪费
 * 3. 微内核：最内层利用循环重排序，使 A 的访问在寄存器中复用
 * 4. 分块大小 MC/NC/KC 根据缓存层次选择，可调参数
 */

#include "operators/GemmBlocked.h"
#include <cassert>
#include <cmath>
#include <cstring>

namespace edgeai {

GemmBlocked::GemmBlocked(int M, int N, int K,
                         int MC, int NC, int KC)
    : M_(M), N_(N), K_(K), MC_(MC), NC_(NC), KC_(KC)
{
    // Allocate B matrix [K, N] as weight, fill with random
    weightB_ = Tensor({K, N});
    float fanIn = static_cast<float>(K);
    float fanOut = static_cast<float>(N);
    float limit = std::sqrt(6.0f / (fanIn + fanOut));
    weightB_.randomize(-limit, limit);
}

void GemmBlocked::packA(const float* A, int lda, float* packedA,
                         int iStart, int pStart, int mc, int kc)
{
    /*
     * Pack A[iStart:iStart+mc, pStart:pStart+kc] into contiguous buffer.
     *
     * Original A has stride lda between rows.
     * Packed A stores elements row by row with stride kc.
     *
     * Layout before packing (A is M x K, row-major):
     *   A[i][p] is at A[i * lda + p]
     *   Stride between consecutive elements in a row: 1 (contiguous)
     *   Stride between rows: lda (may be > kc, wasting cache lines)
     *
     * Layout after packing:
     *   packedA[i * kc + p] = A[(iStart+i) * lda + (pStart+p)]
     *   Stride between rows: kc (exactly what we need, no waste)
     */
    for (int i = 0; i < mc; ++i) {
        const float* srcRow = A + (iStart + i) * lda + pStart;
        float* dstRow = packedA + i * kc;
        std::memcpy(dstRow, srcRow, static_cast<size_t>(kc) * sizeof(float));
    }
}

void GemmBlocked::packB(const float* B, int ldb, float* packedB,
                         int pStart, int jStart, int kc, int nc)
{
    /*
     * Pack B[pStart:pStart+kc, jStart:jStart+nc] into contiguous buffer.
     *
     * Original B has stride ldb between rows.
     * Packed B stores elements row by row with stride nc.
     *
     * Layout before packing (B is K x N, row-major):
     *   B[p][j] is at B[p * ldb + j]
     *   Stride between consecutive elements in a row: 1
     *   Stride between rows: ldb (may be > nc)
     *
     * Layout after packing:
     *   packedB[p * nc + j] = B[(pStart+p) * ldb + (jStart+j)]
     */
    for (int p = 0; p < kc; ++p) {
        const float* srcRow = B + (pStart + p) * ldb + jStart;
        float* dstRow = packedB + p * nc;
        std::memcpy(dstRow, srcRow, static_cast<size_t>(nc) * sizeof(float));
    }
}

void GemmBlocked::blockedGemm(const float* A, const float* B, float* C,
                                int M, int N, int K, int lda, int ldb, int ldc)
{
    /*
     * ====== Three-Level Blocked GEMM ======
     *
     * C[M,N] += A[M,K] * B[K,N]
     *
     * Level 1: Iterate over row blocks of C (dimension M, step MC)
     * Level 2: Iterate over column blocks of C (dimension N, step NC)
     * Level 3: Iterate over K dimension (step KC)
     *
     * For each (ii, jj, pp) block triple:
     *   Pack A[ii:ii+MC, pp:pp+KC] → packedA (contiguous, fits in L1)
     *   Pack B[pp:pp+KC, jj:jj+NC] → packedB (contiguous, fits in L2)
     *   Compute micro-kernel: packedA[MC,KC] * packedB[KC,NC] → C[ii:ii+MC, jj:jj+NC]
     *
     * Micro-kernel loop order (i, j, p):
     *   - Outer: i (iterate over MC rows of packedA)
     *   - Middle: j (iterate over NC columns of packedB)
     *   - Inner: p (iterate over KC inner dimension)
     *   - This makes A[i][p] reused for all j, keeping A[p] values in registers
     */

    // Allocate packing buffers on the heap (they may be too large for stack)
    std::vector<float> packedABuf(static_cast<size_t>(MC_) * KC_);
    std::vector<float> packedBBuf(static_cast<size_t>(KC_) * NC_);

    // Iterate over row blocks
    for (int ii = 0; ii < M; ii += MC_) {
        int mc = std::min(MC_, M - ii);

        // Iterate over column blocks
        for (int jj = 0; jj < N; jj += NC_) {
            int nc = std::min(NC_, N - jj);

            // Iterate over K blocks
            for (int pp = 0; pp < K; pp += KC_) {
                int kc = std::min(KC_, K - pp);

                // Pack A panel: A[ii:ii+mc, pp:pp+kc]
                packA(A, lda, packedABuf.data(), ii, pp, mc, kc);

                // Pack B panel: B[pp:pp+kc, jj:jj+nc]
                packB(B, ldb, packedBBuf.data(), pp, jj, kc, nc);

                // Micro-kernel: packedA[mc, kc] * packedB[kc, nc] → C[ii:ii+mc, jj:jj+nc]
                for (int i = 0; i < mc; ++i) {
                    for (int j = 0; j < nc; ++j) {
                        float sum = 0.0f;
                        for (int p = 0; p < kc; ++p) {
                            sum += packedABuf[i * kc + p] * packedBBuf[p * nc + j];
                        }
                        C[(ii + i) * ldc + (jj + j)] += sum;
                    }
                }
            }
        }
    }
}

void GemmBlocked::forward(const Tensor& input, Tensor& output)
{
    /*
     * ====== Forward pass ======
     *
     * Input tensor is interpreted as A matrix [M, K].
     * WeightB_ is the B matrix [K, N].
     * Output tensor is C matrix [M, N].
     *
     * The input tensor may have shape [M, K] or [1, 1, M, K].
     * We reshape to [M, K] for the GEMM operation.
     */

    // Reshape input to [M, K] if needed
    const float* A = input.data();
    int lda = K_;

    // Reshape output to [M, N]
    output = Tensor({M_, N_});
    output.zero();

    // C[M, N] = A[M, K] * B[K, N]
    const float* B = weightB_.data();
    int ldb = N_;
    float* C = output.data();
    int ldc = N_;

    blockedGemm(A, B, C, M_, N_, K_, lda, ldb, ldc);
}

int64_t GemmBlocked::flops(const std::vector<int>& inputShape,
                             const std::vector<int>& outputShape) const
{
    /*
     * ====== GEMM FLOPs ======
     *
     * C[M,N] = A[M,K] * B[K,N]
     *
     * Each output element C[i][j] requires:
     *   - K multiplications
     *   - K-1 additions
     *   - Total ≈ 2*K FLOPs
     *
     * Total FLOPs = M * N * 2 * K
     *
     * This is the same regardless of blocking.
     * Blocking only improves cache efficiency, not total operations.
     */
    (void)inputShape;
    (void)outputShape;

    return 2 * static_cast<int64_t>(M_) * N_ * K_;
}

} // namespace edgeai
