/**
 * @file step7_vulkan_gemm.cpp
 * @brief Lab3 Step7: Vulkan GEMM —— 在 GPU 上做矩阵乘法
 *
 * ====== 本实验的学习目标 ======
 * 1. 理解 GPU GEMM 的线程映射: 每个线程计算一个输出元素
 * 2. 运行 GEMM 着色器，测量 GPU 计算时间
 * 3. 对比 CPU GEMM (朴素/AVX2/分块) vs GPU GEMM
 * 4. 找到 CPU-GPU 交叉点: 矩阵多大时 GPU 开始更快？
 * 5. 分析 GPU 利用率: 实测 GFLOPS vs 理论峰值
 *
 * ====== GPU GEMM 的线程映射 ======
 * - 输出矩阵 C[M,N] 的每个元素由一个 GPU 线程计算
 * - 线程组织: workgroup (16, 16, 1) → 每个工作组计算 16x16 的输出块
 * - 总调度: dispatch((M+15)/16, (N+15)/16, 1)
 * - 例如 M=1024, N=1024: dispatch(64, 64, 1) = 4096 个工作组
 * - GPU 1024 核，每个核执行 ~4 个工作组 → 并行度极高
 *
 * ====== GEMM 计算着色器 (GLSL) ======
 *
 * #version 450
 * layout(local_size_x = 16, local_size_y = 16) in;
 * layout(binding = 0) buffer A_buf { float A[]; };
 * layout(binding = 1) buffer B_buf { float B[]; };
 * layout(binding = 2) buffer C_buf { float C[]; };
 * layout(push_constant) uniform PushConstants {
 *     int M, N, K;
 * } pc;
 *
 * void main() {
 *     int row = int(gl_GlobalInvocationID.x);  // M 维度
 *     int col = int(gl_GlobalInvocationID.y);  // N 维度
 *     if (row >= pc.M || col >= pc.N) return;
 *
 *     float sum = 0.0;
 *     for (int p = 0; p < pc.K; p++) {
 *         sum += A[row * pc.K + p] * B[p * pc.N + col];
 *     }
 *     C[row * pc.N + col] = sum;
 * }
 *
 * ====== CPU vs GPU 的权衡 ======
 * - CPU 优势: 低延迟 (小矩阵)、缓存大、分支预测
 * - GPU 优势: 高吞吐 (大矩阵)、内存带宽大 (HBM vs DDR)
 * - 交叉点: 通常在矩阵维度 256-512 之间
 * - 小于交叉点: CPU 更快 (GPU 调度和数据传输开销占主导)
 * - 大于交叉点: GPU 更快 (计算量大到可以摊薄固定开销)
 *
 * ====== 本实现的简化之处 ======
 * - 使用简单 GEMM 着色器 (无共享内存分块) → 适合学习理解
 * - 实际生产级 GEMM 着色器会用 shared memory 做 tiling
 * - 优化版: 用 shared memory 让一个 workgroup 共享 A/B 数据
 *   → 减少 K 倍全局读取
 *
 * ====== 编译与运行 ======
 * 有 Vulkan: g++ -O2 -std=c++17 -DENABLE_VULKAN -lvulkan -o step7_vulkan_gemm step7_vulkan_gemm.cpp
 * 无 Vulkan: g++ -O2 -std=c++17 -o step7_vulkan_gemm step7_vulkan_gemm.cpp
 * ./step7_vulkan_gemm
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
    double elapsedUs() const {
        return std::chrono::duration<double, std::micro>(stop_ - start_).count();
    }
private:
    std::chrono::steady_clock::time_point start_, stop_;
};

// ============================================================================
// GEMM FLOPs
// ============================================================================
int64_t flopCountGemm(int M, int N, int K) {
    return 2LL * M * N * K;
}

// ============================================================================
// 朴素 GEMM (i-j-p 顺序, B 按列访问, 缓存不友好)
// ============================================================================
void naiveGemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int p = 0; p < K; p++) {
                sum += A[i * K + p] * B[p * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// AVX2 GEMM (i-p-j 顺序 + SIMD, 缓存友好)
// ============================================================================
void avx2Gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        float* cRow = C + i * N;
        const float* aRow = A + i * K;
        for (int p = 0; p < K; p++) {
            __m256 aVal = _mm256_set1_ps(aRow[p]);
            int j = 0;
            for (; j + 7 < N; j += 8) {
                __m256 bVal = _mm256_loadu_ps(&B[p * N + j]);
                __m256 cVal = _mm256_loadu_ps(&cRow[j]);
                __m256 result = _mm256_fmadd_ps(aVal, bVal, cVal);
                _mm256_storeu_ps(&cRow[j], result);
            }
            for (; j < N; j++) {
                cRow[j] += aRow[p] * B[p * N + j];
            }
        }
    }
}

// ============================================================================
// 分块 GEMM (AVX2 + packing, 最优 CPU 实现)
// ============================================================================
void blockedGemm(const float* A, const float* B, float* C, int M, int N, int K,
                 int MC = 128, int NC = 256, int KC = 64) {
    // 初始化 C 为 0
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    std::vector<float> packedA(MC * KC);
    std::vector<float> packedB(KC * NC);

    for (int pp = 0; pp < K; pp += KC) {
        int kc = std::min(KC, K - pp);
        for (int jj = 0; jj < N; jj += NC) {
            int nc = std::min(NC, N - jj);
            // Pack B
            for (int p = 0; p < kc; p++) {
                for (int j = 0; j < nc; j++) {
                    packedB[p * nc + j] = B[(pp + p) * N + jj + j];
                }
            }
            for (int ii = 0; ii < M; ii += MC) {
                int mc = std::min(MC, M - ii);
                // Pack A
                for (int i = 0; i < mc; i++) {
                    for (int p = 0; p < kc; p++) {
                        packedA[i * kc + p] = A[(ii + i) * K + pp + p];
                    }
                }
                // 微内核: AVX2 GEMM on packed data
                avx2Gemm(packedA.data(), packedB.data(), C + ii * N + jj, mc, nc, kc);
            }
        }
    }
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("=== Lab3 Step7: Vulkan GEMM ===\n\n");

    // ================================================================
    // GPU GEMM 着色器讲解
    // ================================================================
    printf("--- GPU GEMM 着色器 (GLSL) ---\n");
    printf("  #version 450\n");
    printf("  layout(local_size_x = 16, local_size_y = 16) in;\n");
    printf("  layout(binding = 0) buffer A_buf { float A[]; };\n");
    printf("  layout(binding = 1) buffer B_buf { float B[]; };\n");
    printf("  layout(binding = 2) buffer C_buf { float C[]; };\n");
    printf("  layout(push_constant) uniform PC { int M, N, K; } pc;\n");
    printf("  void main() {\n");
    printf("      int row = int(gl_GlobalInvocationID.x);\n");
    printf("      int col = int(gl_GlobalInvocationID.y);\n");
    printf("      if (row >= pc.M || col >= pc.N) return;\n");
    printf("      float sum = 0.0;\n");
    printf("      for (int p = 0; p < pc.K; p++)\n");
    printf("          sum += A[row*pc.K+p] * B[p*pc.N+col];\n");
    printf("      C[row*pc.N+col] = sum;\n");
    printf("  }\n\n");

    printf("--- 线程映射说明 ---\n");
    printf("  workgroup 大小: 16 x 16 = 256 线程\n");
    printf("  每个 workgroup 计算 C 的 16x16 块\n");
    printf("  dispatch 数量: (M+15)/16 x (N+15)/16\n");
    printf("  总线程数: M x N (每个线程算 1 个输出元素)\n\n");

    // ================================================================
    // CPU GEMM 性能基准
    // ================================================================
    printf("--- CPU GEMM 性能基准 ---\n\n");

    struct MatrixSize {
        int dim;  // M = N = K
        const char* label;
    };

    std::vector<MatrixSize> sizes = {
        {64,   "64x64"},
        {128,  "128x128"},
        {256,  "256x256"},
        {512,  "512x512"},
        {1024, "1024x1024"},
        {2048, "2048x2048"},
    };

    struct CpuGemmResult {
        int dim;
        double naiveMs, avx2Ms, blockedMs;
        double naiveGf, avx2Gf, blockedGf;
    };
    std::vector<CpuGemmResult> cpuResults;

    printf("  %-12s %8s %8s %8s | %8s %8s %8s\n",
           "矩阵", "Naive", "AVX2", "Blocked",
           "GF(N)", "GF(A)", "GF(B)");
    printf("  %-12s %8s %8s %8s | %8s %8s %8s\n",
           "-----", "------", "------", "-------",
           "------", "------", "------");

    for (const auto& sz : sizes) {
        int M = sz.dim, N = sz.dim, K = sz.dim;
        int64_t flops = flopCountGemm(M, N, K);

        std::vector<float> A(M * K), B(K * N);
        std::vector<float> C_n(M * N), C_a(M * N), C_b(M * N);

        srand(42);
        for (auto& v : A) v = -1.0f + 2.0f * static_cast<float>(rand()) / RAND_MAX;
        for (auto& v : B) v = -1.0f + 2.0f * static_cast<float>(rand()) / RAND_MAX;

        // Naive
        for (int i = 0; i < M * N; i++) C_n[i] = 0.0f;
        naiveGemm(A.data(), B.data(), C_n.data(), M, N, K);
        Timer tn; tn.start();
        naiveGemm(A.data(), B.data(), C_n.data(), M, N, K);
        tn.stop();

        // AVX2
        for (int i = 0; i < M * N; i++) C_a[i] = 0.0f;
        avx2Gemm(A.data(), B.data(), C_a.data(), M, N, K);
        Timer ta; ta.start();
        avx2Gemm(A.data(), B.data(), C_a.data(), M, N, K);
        ta.stop();

        // Blocked
        blockedGemm(A.data(), B.data(), C_b.data(), M, N, K);
        Timer tb; tb.start();
        blockedGemm(A.data(), B.data(), C_b.data(), M, N, K);
        tb.stop();

        CpuGemmResult r;
        r.dim = sz.dim;
        r.naiveMs = tn.elapsedMs();
        r.avx2Ms = ta.elapsedMs();
        r.blockedMs = tb.elapsedMs();
        r.naiveGf = static_cast<double>(flops) / (r.naiveMs * 1e6);
        r.avx2Gf = static_cast<double>(flops) / (r.avx2Ms * 1e6);
        r.blockedGf = static_cast<double>(flops) / (r.blockedMs * 1e6);
        cpuResults.push_back(r);

        printf("  %-12s %8.2f %8.2f %8.2f | %8.1f %8.1f %8.1f\n",
               sz.label, r.naiveMs, r.avx2Ms, r.blockedMs,
               r.naiveGf, r.avx2Gf, r.blockedGf);
    }

    // ================================================================
    // GPU GEMM 模拟分析
    // (在实际 Vulkan 实现中，这里会加载着色器并执行)
    // ================================================================
    printf("\n--- GPU GEMM 性能估算 ---\n\n");

    printf("  GPU 性能估算模型:\n");
    printf("    GPU 计算时间 = FLOPs / (峰值GFLOPS * 1e6)\n");
    printf("    数据传输时间 = 3*M*K*4 / PCIe带宽 (双向)\n");
    printf("    总时间 = 传输时间 + 计算时间\n\n");

    // 假设 GPU 峰值: 2000 GFLOPS (典型集成显卡)
    // PCIe 带宽: ~16 GB/s
    double gpuPeakGflops = 2000.0;
    double pcieBandwidthGBs = 16.0;

    printf("  假设: GPU 峰值 %.0f GFLOPS, PCIe 带宽 %.0f GB/s\n\n",
           gpuPeakGflops, pcieBandwidthGBs);

    printf("  %-12s %8s %8s %8s | %8s %8s\n",
           "矩阵", "CPU(ms)", "GPU计(ms)", "传输(ms)", "GPU总(ms)", "GPU快?");
    printf("  %-12s %8s %8s %8s | %8s %8s\n",
           "-----", "------", "--------", "--------", "--------", "------");

    for (const auto& r : cpuResults) {
        int M = r.dim;
        int64_t flops = flopCountGemm(M, M, M);

        // GPU 计算时间 (假设 30% 利用率)
        double gpuUtilization = 0.3;
        double gpuComputeMs = static_cast<double>(flops) / (gpuPeakGflops * gpuUtilization * 1e6) * 1e3;

        // 数据传输时间 (A + B → GPU, C ← GPU)
        double dataBytes = static_cast<double>(M * M * 4) * 3;  // 3 个矩阵
        double transferMs = dataBytes / (pcieBandwidthGBs * 1e6) * 1e3;  // ms

        double gpuTotalMs = gpuComputeMs + transferMs;
        double cpuBestMs = r.blockedMs;
        bool gpuFaster = gpuTotalMs < cpuBestMs;

        printf("  %-12d %8.2f %8.2f %8.2f | %8.2f %s\n",
               M, cpuBestMs, gpuComputeMs, transferMs, gpuTotalMs,
               gpuFaster ? "GPU更快" : "CPU更快");
    }

    // ================================================================
    // GPU 利用率分析
    // ================================================================
    printf("\n--- GPU 利用率分析 ---\n\n");
    printf("  理论 GPU 峰值: %.0f GFLOPS (FP32)\n", gpuPeakGflops);
    printf("  实际 GPU 性能取决于:\n");
    printf("    1. 全局内存带宽 (通常是瓶颈)\n");
    printf("    2. 工作组大小和调度效率\n");
    printf("    3. 共享内存的利用 (本实现未使用)\n");
    printf("    4. 寄存器使用量 (影响占用率)\n\n");

    printf("  优化方向:\n");
    printf("    Level 0 (本实现): 简单 GEMM, 每线程从全局内存读 K 次 A 和 K 次 B\n");
    printf("    Level 1: 用 shared memory 分块, 每个线程只从全局内存读一次\n");
    printf("    Level 2: 寄存器分块 + 预取, 减少共享内存延迟\n");
    printf("    Level 3: 双缓冲 + 指令级并行, 接近峰值性能\n\n");

    printf("  各级优化的典型利用率:\n");
    printf("    Level 0: ~10-20%% 峰值\n");
    printf("    Level 1: ~40-60%% 峰值\n");
    printf("    Level 2: ~60-80%% 峰值\n");
    printf("    Level 3: ~80-90%% 峰值\n");

    // ================================================================
    // Vulkan 实际运行 (如果启用了)
    // ================================================================
    printf("\n--- Vulkan 运行测试 ---\n");

#ifdef ENABLE_VULKAN
    printf("  Vulkan 已启用, 可以运行实际的 GPU GEMM\n");
    printf("  请确保 gemm.comp.spv 着色器文件存在\n");
    printf("  实际运行请参考项目中的 VulkanGemm 类\n");
#else
    printf("  Vulkan 未启用 — 编译时添加 -DENABLE_VULKAN\n");
    printf("  在实际项目中, VulkanGemm 类封装了完整的 GPU GEMM 流程:\n");
    printf("    1. 初始化 VulkanComputeBase\n");
    printf("    2. 加载 gemm.comp.spv 着色器\n");
    printf("    3. 创建 3 个 GPU buffer (A, B, C)\n");
    printf("    4. writeBuffer: 将 A, B 写入 GPU\n");
    printf("    5. dispatch: 执行 GPU GEMM\n");
    printf("    6. readBuffer: 从 GPU 读回 C\n");
#endif

    // ================================================================
    // 交叉点分析
    // ================================================================
    printf("\n--- CPU-GPU 交叉点分析 ---\n\n");
    printf("  交叉点: 矩阵多大时 GPU 开始比 CPU 快?\n\n");
    printf("  影响因素:\n");
    printf("    1. GPU 峰值性能: 独立显卡 > 集成显卡\n");
    printf("    2. 内存类型: HBM (GPU) >> DDR4 (CPU)\n");
    printf("    3. 传输通道: PCIe 4.0 > PCIe 3.0 > 统一内存\n");
    printf("    4. CPU 优化程度: 分块 GEMM > AVX2 > 朴素\n\n");
    printf("  典型交叉点 (估计):\n");
    printf("    集成显卡 + 分块 CPU GEMM: M,N,K >= 512\n");
    printf("    独立显卡 + 分块 CPU GEMM: M,N,K >= 256\n");
    printf("    统一内存 (M1 等): M,N,K >= 128\n");
    printf("    → 对于边缘设备的 CNN 推理, 每层的矩阵大小需要逐层评估\n");

    printf("\n=== Step7 完成 ===\n");
    printf("下一步: step8_vulkan_conv2d.cpp —— 在 GPU 上做卷积\n");

    return 0;
}
