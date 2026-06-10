/**
 * @file step3_cache_tiled_conv.cpp
 * @brief Lab3 Step3: 缓存分块卷积 —— 针对缓存层次结构的优化
 *
 * ====== 本实验的学习目标 ======
 * 1. 理解 CPU 缓存层次结构及其对性能的决定性影响
 * 2. 理解朴素卷积为什么缓存不友好——每个输入元素被重复加载
 * 3. 实现缓存分块卷积：将输出划分为小块，使数据留在缓存中
 * 4. 对比不同分块大小的性能，找到最优分块
 *
 * ====== CPU 缓存层次结构 ======
 *
 *   ┌─────────────────────────────────────────────────────┐
 *   │  寄存器 (Register)                                  │
 *   │  大小: ~几百字节    延迟: 0 周期    带宽: 无限        │
 *   └──────────────────────┬──────────────────────────────┘
 *                          │ ~1 周期
 *   ┌──────────────────────▼──────────────────────────────┐
 *   │  L1 缓存 (每核独占)                                  │
 *   │  大小: ~32 KB      延迟: ~4 周期   带宽: ~1 TB/s    │
 *   └──────────────────────┬──────────────────────────────┘
 *                          │ ~10 周期
 *   ┌──────────────────────▼──────────────────────────────┐
 *   │  L2 缓存 (每核独占)                                  │
 *   │  大小: ~256 KB     延迟: ~12 周期   带宽: ~500 GB/s  │
 *   └──────────────────────┬──────────────────────────────┘
 *                          │ ~40 周期
 *   ┌──────────────────────▼──────────────────────────────┐
 *   │  L3 缓存 (所有核共享)                                │
 *   │  大小: ~8 MB       延迟: ~40 周期   带宽: ~200 GB/s  │
 *   └──────────────────────┬──────────────────────────────┘
 *                          │ ~200 周期
 *   ┌──────────────────────▼──────────────────────────────┐
 *   │  主存 (DRAM)                                        │
 *   │  大小: ~8 GB       延迟: ~80 ns      带宽: ~50 GB/s   │
 *   └─────────────────────────────────────────────────────┘
 *
 *   关键观察: 访问主存比访问 L1 慢 ~100 倍!
 *   → 如果数据在缓存中，计算飞快；否则 CPU 空等内存
 *   → "缓存友好"的代码可以比"缓存不友好"的代码快几十倍
 *
 * ====== 朴素卷积为什么缓存不友好？ ======
 *
 * 以 3x3 卷积为例，计算输出位置 (oh, ow) 需要:
 *   - 读取 9 个输入像素 (3x3 感受野)
 *   - 读取 9 个权重
 *   - 1 次累加写入
 *
 * 问题1: 输入重复读取
 *   - 相邻输出 (oh,ow) 和 (oh,ow+1) 共享 6 个输入像素 (2/3 重叠)
 *   - 但计算 (oh,ow+1) 时，(oh,ow) 用过的输入可能已被逐出缓存
 *   - 因为中间计算了很多 outC 个通道，缓存已被其他数据占满
 *
 * 问题2: 权重反复重载
 *   - 同一组权重被所有空间位置 (oh,ow) 使用
 *   - 但 outC*outH*outW 次迭代中，权重可能多次被逐出再加载
 *
 * 问题3: 输出缓存抖动
 *   - 输出按 outC → outH → outW 顺序写入
 *   - 对同一输出像素 (oc,oh,ow) 的累加跨越了整个 inC 维度
 *   - 中间有 inC*kH*kW 次其他访问，输出可能已被逐出
 *
 * ====== 分块 (Tiling) 的核心思想 ======
 *
 * 将输出划分为小块 (tile)，使得计算一个 tile 所需的所有数据
 * 能同时放入 L2 缓存。
 *
 * 具体分析:
 *   - 输出 tile: outHTile × outWTile × ocTile 个 float
 *   - 输入 patch: (outHTile*s+kH-1) × (outWTile*s+kW-1) × inC 个 float
 *   - 权重: kH × kW × inC × ocTile 个 float
 *
 *   需满足: 输入 patch + 权重 + 输出 tile < L2 cache size (256KB)
 *
 *   例如: inC=3, outC=64, kH=kW=3, stride=1
 *   - outHTile=16, outWTile=16, ocTile=32:
 *     输入: (16+2)*(16+2)*3*4 = 18*18*3*4 = 3,888 bytes
 *     权重: 3*3*3*32*4 = 3,456 bytes
 *     输出: 16*16*32*4 = 32,768 bytes
 *     总计: ~40 KB << 256 KB → 可以放入 L2，且还有余量
 *
 *   - outHTile=32, outWTile=32, ocTile=64:
 *     输入: (32+2)*(32+2)*3*4 = 34*34*3*4 = 13,872 bytes
 *     权重: 3*3*3*64*4 = 6,912 bytes
 *     输出: 32*32*64*4 = 262,144 bytes
 *     总计: ~283 KB > 256 KB → 超出 L2，会产生 cache miss
 *
 * ====== 编译与运行 ======
 * g++ -O2 -std=c++17 -o step3_cache_tiled_conv step3_cache_tiled_conv.cpp
 * ./step3_cache_tiled_conv
 */

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

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
// 张量类 (与前面步骤相同)
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
// FLOPs 计算 (与前面步骤相同)
// ============================================================================
int64_t flopCountConv2d(int inC, int outC, int kH, int kW,
                        int outH, int outW) {
    int64_t kernelsMulAdd = static_cast<int64_t>(inC) * kH * kW;
    int64_t outputElements = static_cast<int64_t>(outC) * outH * outW;
    return 2 * outputElements * kernelsMulAdd;
}

// ============================================================================
// 朴素卷积 (与 Step1 相同，用于对比)
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
// 缓存分块卷积
//
// 核心思想: 将输出划分为 outHTile × outWTile 的空间块，
//           将输出通道划分为 ocTile 大小的通道块
//           确保每个块所需的数据能放入 L2 缓存
//
// 循环顺序（关键优化！）:
//   外层: 遍历空间块 (oh_tile, ow_tile)
//   中层: 遍历通道块 (oc_tile)
//   内层: 遍历块内元素 (ic, kh, kw, oh_in_tile, ow_in_tile, oc_in_tile)
//
// 为什么这个循环顺序更好？
// 1. 空间块内，相邻输出位置共享大部分输入像素
// 2. 通道块内，同一组权重被复用 ocTile 次
// 3. 输入数据在块计算期间一直留在缓存中
// ============================================================================
void tiledConv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
                 Tensor& output, int stride, int pad,
                 int outHTile, int outWTile, int ocTile) {
    int N = input.shape[0], inC = input.shape[1];
    int inH = input.shape[2], inW = input.shape[3];
    int outC = weight.shape[0];
    int kH = weight.shape[2], kW = weight.shape[3];
    int outH = (inH + 2 * pad - kH) / stride + 1;
    int outW = (inW + 2 * pad - kW) / stride + 1;

    // 先将偏置填入输出
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < outC; oc++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    output.at(n, oc, oh, ow) = bias.data[oc];
                }
            }
        }
    }

    // ================================================================
    // 分块卷积主循环
    //
    // 循环层次（从外到内）:
    // 1. n:                batch 维度
    // 2. oh_tile_start:    空间块起始行
    // 3. ow_tile_start:    空间块起始列
    // 4. oc_tile_start:    通道块起始
    // 5. ic:               输入通道
    // 6. kh, kw:           卷积核位置
    // 7. oh_in_tile:       块内输出行
    // 8. ow_in_tile:       块内输出列
    // 9. oc_in_tile:       块内输出通道
    //
    // 注意最内三层循环的顺序!
    //   oc_in_tile 在最内层 → 权重连续访问
    //   这是因为权重布局是 [outC][inC][kH][kW]
    //   相邻的 oc 对应的权重是连续存储的
    // ================================================================
    for (int n = 0; n < N; n++) {
        // 遍历空间块 (高度方向)
        for (int ohTileStart = 0; ohTileStart < outH; ohTileStart += outHTile) {
            int ohTileEnd = std::min(ohTileStart + outHTile, outH);

            // 遍历空间块 (宽度方向)
            for (int owTileStart = 0; owTileStart < outW; owTileStart += outWTile) {
                int owTileEnd = std::min(owTileStart + outWTile, outW);

                // 遍历通道块
                for (int ocTileStart = 0; ocTileStart < outC; ocTileStart += ocTile) {
                    int ocTileEnd = std::min(ocTileStart + ocTile, outC);

                    // ====================================================
                    // 内层计算: 在当前 tile 内累加卷积结果
                    //
                    // 此时的数据访问模式:
                    // - 输入: 只有 (outHTile+kH-1) × (outWTile+kW-1) × inC 的局部 patch
                    // - 权重: 只有 kH × kW × inC × ocTile 的子集
                    // - 输出: 只有 outHTile × outWTile × ocTile 的块
                    //
                    // 这些数据总量应该 < L2 cache size (256KB)
                    // ====================================================
                    for (int ic = 0; ic < inC; ic++) {
                        for (int kh = 0; kh < kH; kh++) {
                            for (int kw = 0; kw < kW; kw++) {
                                // 遍历块内的输出位置
                                for (int oh = ohTileStart; oh < ohTileEnd; oh++) {
                                    int ih = oh * stride + kh - pad;

                                    // 边界检查: 如果输入坐标越界，跳过整行
                                    if (ih < 0 || ih >= inH) continue;

                                    for (int ow = owTileStart; ow < owTileEnd; ow++) {
                                        int iw = ow * stride + kw - pad;

                                        // 边界检查
                                        if (iw < 0 || iw >= inW) continue;

                                        // 读取输入像素 —— 在整个块计算期间，这个值会被缓存
                                        float inputVal = input.at(n, ic, ih, iw);

                                        // ====================================================
                                        // 最内层循环: 遍历块内的输出通道
                                        //
                                        // 关键优化: oc_in_tile 在最内层
                                        // - 权重 weight[oc][ic][kh][kw] 的存储:
                                        //   oc 相邻 → 内存相邻 (步长 = inC*kH*kW)
                                        // - 虽然不是完全连续，但 ocTile 个权重
                                        //   可以预取到缓存行中
                                        // - 输入值 inputVal 在寄存器中复用 ocTile 次!
                                        //   → 减少了 ocTile-1 次输入读取
                                        // ====================================================
                                        for (int oc = ocTileStart; oc < ocTileEnd; oc++) {
                                            output.at(n, oc, oh, ow) +=
                                                inputVal * weight.at(oc, ic, kh, kw);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    const int N = 1, inC = 3, outC = 64;
    const int H = 224, W = 224;
    const int kH = 3, kW = 3;
    const int stride = 1, pad = 1;
    const int outH = (H + 2 * pad - kH) / stride + 1;
    const int outW = (W + 2 * pad - kW) / stride + 1;

    printf("=== Lab3 Step3: 缓存分块卷积 ===\n");
    printf("问题规模: %dx%dx%d → %dx%dx%d, kernel=%dx%d\n",
           inC, H, W, outC, outH, outW, kH, kW);

    int64_t flops = flopCountConv2d(inC, outC, kH, kW, outH, outW);
    printf("FLOPs: %lld (%.2fG)\n\n", (long long)flops, flops / 1e9);

    // ================================================================
    // 缓存层次信息
    // ================================================================
    printf("--- CPU 缓存层次 (典型值) ---\n");
    printf("  L1: ~32 KB,  延迟 ~4 周期,  带宽 ~1 TB/s\n");
    printf("  L2: ~256 KB, 延迟 ~12 周期, 带宽 ~500 GB/s\n");
    printf("  L3: ~8 MB,   延迟 ~40 周期, 带宽 ~200 GB/s\n");
    printf("  DRAM: ~8 GB, 延迟 ~80 ns,   带宽 ~50 GB/s\n\n");

    // 创建张量
    Tensor input({1, inC, H, W});
    Tensor weight({outC, inC, kH, kW});
    Tensor bias({outC});
    Tensor output_naive({1, outC, outH, outW});
    Tensor output_tiled({1, outC, outH, outW});

    srand(42);
    input.randomize();
    weight.randomize();
    for (int i = 0; i < outC; i++) bias.data[i] = 0.01f * (rand() % 100) / 100.0f;

    // ================================================================
    // Baseline: 朴素卷积
    // ================================================================
    const int NUM_ITERS = 10;

    output_naive.zero();
    naiveConv2d(input, weight, bias, output_naive, stride, pad);

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
    printf("--- 朴素卷积 (Baseline) ---\n");
    printf("  平均耗时: %.2f ms, GFLOPS: %.2f\n\n", avgNaiveMs, naiveGflops);

    // ================================================================
    // 对比不同分块大小的性能
    //
    // 测试三种配置:
    //   1. 16x16 空间块: 较小块 → 缓存命中率高，但块数多（循环开销大）
    //   2. 32x32 空间块: 中等块 → 平衡缓存利用和循环开销
    //   3. 64x64 空间块: 较大块 → 块数少，但可能超出 L2 缓存
    //
    // 预期结果:
    //   - 16x16 和 32x32 应该比朴素快 (数据留在缓存中)
    //   - 64x64 可能不如 32x32 (超出 L2，缓存开始抖动)
    //   - 最优分块取决于具体的 CPU 缓存大小
    // ================================================================
    struct TileConfig {
        int ohTile, owTile, ocTile;
        std::string label;
        int estimatedBytes;  // 估计的缓存占用
    };

    std::vector<TileConfig> configs = {
        {16, 16, 32, "16x16x32",
         /* 输入 */ (16+2)*(16+2)*3*4 + /* 权重 */ 3*3*3*32*4 + /* 输出 */ 16*16*32*4},
        {32, 32, 32, "32x32x32",
         (32+2)*(32+2)*3*4 + 3*3*3*32*4 + 32*32*32*4},
        {64, 64, 16, "64x64x16",
         (64+2)*(64+2)*3*4 + 3*3*3*16*4 + 64*64*16*4},
    };

    printf("--- 缓存分块卷积 (不同分块大小对比) ---\n");
    printf("  %-12s %8s %10s %8s %8s\n",
           "分块", "缓存占用", "耗时(ms)", "GFLOPS", "加速比");
    printf("  %-12s %8s %10s %8s %8s\n",
           "----", "--------", "--------", "------", "------");

    struct Result { std::string label; double ms; double gf; double speedup; };
    std::vector<Result> results;

    for (const auto& cfg : configs) {
        output_tiled.zero();
        tiledConv2d(input, weight, bias, output_tiled, stride, pad,
                    cfg.ohTile, cfg.owTile, cfg.ocTile);  // 预热

        double totalMs = 0.0;
        for (int iter = 0; iter < NUM_ITERS; iter++) {
            output_tiled.zero();
            Timer t;
            t.start();
            tiledConv2d(input, weight, bias, output_tiled, stride, pad,
                        cfg.ohTile, cfg.owTile, cfg.ocTile);
            t.stop();
            totalMs += t.elapsedMs();
        }

        double avgMs = totalMs / NUM_ITERS;
        double gf = static_cast<double>(flops) / (avgMs * 1e6);
        double speedup = avgNaiveMs / avgMs;

        results.push_back({cfg.label, avgMs, gf, speedup});

        // 格式化缓存占用
        char cacheStr[32];
        if (cfg.estimatedBytes < 1024) {
            snprintf(cacheStr, sizeof(cacheStr), "%d B", cfg.estimatedBytes);
        } else if (cfg.estimatedBytes < 1024 * 1024) {
            snprintf(cacheStr, sizeof(cacheStr), "%.0f KB", cfg.estimatedBytes / 1024.0);
        } else {
            snprintf(cacheStr, sizeof(cacheStr), "%.1f MB", cfg.estimatedBytes / (1024.0 * 1024));
        }

        // 判断缓存是否在 L2 中
        std::string fitStatus = (cfg.estimatedBytes < 256 * 1024) ? "L2" : ">L2";

        printf("  %-12s %6s %-3s %10.2f %8.2f %7.2fx\n",
               cfg.label.c_str(), cacheStr, fitStatus.c_str(),
               avgMs, gf, speedup);
    }

    // ================================================================
    // 缓存占用的详细分析
    // ================================================================
    printf("\n--- 分块大小 vs 缓存占用分析 ---\n");
    printf("  L2 缓存大小: ~256 KB\n\n");

    for (size_t i = 0; i < configs.size(); i++) {
        const auto& cfg = configs[i];
        printf("  分块 %s:\n", cfg.label.c_str());
        double inputPatch = (cfg.ohTile + kH - 1) * (cfg.owTile + kW - 1) * inC * sizeof(float);
        double weightPatch = kH * kW * inC * cfg.ocTile * sizeof(float);
        double outputPatch = cfg.ohTile * cfg.owTile * cfg.ocTile * sizeof(float);
        printf("    输入 patch: %.1f KB\n", inputPatch / 1024);
        printf("    权重 patch: %.1f KB\n", weightPatch / 1024);
        printf("    输出 patch: %.1f KB\n", outputPatch / 1024);
        printf("    总计: %.1f KB %s\n",
               (inputPatch + weightPatch + outputPatch) / 1024,
               (cfg.estimatedBytes < 256 * 1024) ? "[在 L2 中]" : "[超出 L2!]");
    }

    // ================================================================
    // ASCII 柱状图对比
    // ================================================================
    printf("\n--- 性能对比图 ---\n");
    double maxGflops = naiveGflops;
    for (const auto& r : results) maxGflops = std::max(maxGflops, r.gf);
    int barWidth = 50;

    printf("  Naive  |");
    int naiveBar = static_cast<int>(naiveGflops / maxGflops * barWidth);
    for (int i = 0; i < naiveBar; i++) printf("#");
    printf(" %.1f\n", naiveGflops);

    for (const auto& r : results) {
        printf("  %-8s|", r.label.c_str());
        int bar = static_cast<int>(r.gf / maxGflops * barWidth);
        for (int i = 0; i < bar; i++) printf("#");
        printf(" %.1f (%.1fx)\n", r.gf, r.speedup);
    }

    // ================================================================
    // 正确性验证
    // ================================================================
    printf("\n--- 正确性验证 ---\n");
    // 用最优分块的结果与朴素卷积对比
    const auto& bestResult = results[0];
    output_tiled.zero();
    tiledConv2d(input, weight, bias, output_tiled, stride, pad,
                configs[0].ohTile, configs[0].owTile, configs[0].ocTile);

    double maxDiff = 0.0;
    int numElements = output_naive.total();
    for (int i = 0; i < numElements; i++) {
        double diff = std::fabs(output_naive.data[i] - output_tiled.data[i]);
        maxDiff = std::max(maxDiff, diff);
    }
    printf("  分块卷积 vs 朴素卷积: 最大误差 = %.2e %s\n",
           maxDiff, maxDiff < 1e-3 ? "PASS" : "MISMATCH");

    printf("\n=== Step3 完成 ===\n");
    printf("关键结论: 选择与 L2 缓存大小匹配的分块，可获得最佳性能\n");
    printf("下一步: step4_winograd_conv.cpp —— 用代数变换减少乘法次数\n");

    return 0;
}
