/**
 * @file Conv2dTiled.cpp
 * @brief 缓存分块卷积实现 —— 针对缓存层次结构的优化
 *
 * ====== 实现说明 ======
 * 1. 将输出划分为 outHTile x outWTile 的小块
 * 2. 计算每个小块时，所需的输入数据和权重都尽量在 L2 缓存中
 * 3. 同时对输出通道做分块 (ocTile)，使权重也能被缓存
 * 4. 循环重排序：最内层循环是 outC，因为权重在 outC 维度连续
 */

#include "operators/Conv2dTiled.h"
#include <cassert>
#include <cmath>
#include <algorithm>

namespace edgeai {

Conv2dTiled::Conv2dTiled(int inC, int outC, int kH, int kW,
                         int stride, int pad, int groups,
                         int outHTile, int outWTile, int ocTile)
    : inC_(inC), outC_(outC), kH_(kH), kW_(kW),
      stride_(stride), pad_(pad), groups_(groups),
      outHTile_(outHTile), outWTile_(outWTile), ocTile_(ocTile)
{
    assert(inC % groups == 0 && "inC must be divisible by groups");
    assert(outC % groups == 0 && "outC must be divisible by groups");

    int icPerGroup = inC / groups;
    weight_ = Tensor({outC, icPerGroup, kH, kW});
    bias_ = Tensor({outC});

    float fanIn = static_cast<float>(icPerGroup * kH * kW);
    float fanOut = static_cast<float>((outC / groups) * kH * kW);
    float limit = std::sqrt(6.0f / (fanIn + fanOut));
    weight_.randomize(-limit, limit);
    bias_.zero();
}

void Conv2dTiled::forward(const Tensor& input, Tensor& output)
{
    assert(input.dims() == 4);
    int N = input.size(0);
    int H = input.size(2);
    int W = input.size(3);

    int outH = (H + 2 * pad_ - kH_) / stride_ + 1;
    int outW = (W + 2 * pad_ - kW_) / stride_ + 1;

    int icPerGroup = inC_ / groups_;
    int ocPerGroup = outC_ / groups_;

    output = Tensor({N, outC_, outH, outW});
    output.zero();

    /*
     * ====== 分块卷积的三层循环结构 ======
     *
     * 外层循环：遍历分块
     *   - oh_tile：输出高度方向的分块
     *   - ow_tile：输出宽度方向的分块
     *   - oc_tile：输出通道方向的分块
     *
     * 内层循环：计算分块内的输出元素
     *   - ic：输入通道（累加维度）
     *   - kh, kw：卷积核空间维度（累加维度）
     *
     * ====== 缓存分析 ======
     *
     * 计算一个 outHTile x outWTile x ocTile 的输出块时，需要访问：
     *
     * 1. 输入数据：
     *    - 空间范围：从 (oh_tile*s - pad) 到 (oh_tile*s + (outHTile-1)*s + kH - pad)
     *    - 大约 (outHTile*s + kH) * (outWTile*s + kW) * icPerGroup 个 float
     *    - 对于 outHTile=outWTile=16, s=1, kH=kW=3, icPerGroup=64:
     *      19 * 19 * 64 * 4 = 92.5 KB
     *
     * 2. 权重数据：
     *    - ocTile * icPerGroup * kH * kW 个 float
     *    - 对于 ocTile=32, icPerGroup=64, kH=kW=3:
     *      32 * 64 * 9 * 4 = 73.7 KB
     *
     * 3. 输出数据：
     *    - outHTile * outWTile * ocTile 个 float
     *    - 对于 16*16*32:
     *      16 * 16 * 32 * 4 = 32.8 KB
     *
     * 总计：约 199 KB → 适合放入 256 KB 的 L2 缓存！
     * 这就是分块大小选择的理论依据
     *
     * ====== 与朴素卷积的对比 ======
     * 朴素卷积计算 oh=0 时需要整个 ic 维度的输入
     * 当计算 oh=1 时，这些输入数据已经被后续的 oh=0 计算逐出了缓存
     * → 每个 oh 都要从主存重新加载输入 → 大量 cache miss
     *
     * 分块卷积：
     * 计算一个 tile 内的所有 oh 和 ow 时，输入数据始终在 L2 中
     * → 同一 tile 内的 oh/ow 共享输入数据 → 缓存命中率极高
     */

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < groups_; ++g) {
            int icGlobalStart = g * icPerGroup;
            int ocGlobalStart = g * ocPerGroup;

            // ──────────────── 分块遍历输出空间 ────────────────
            for (int ohStart = 0; ohStart < outH; ohStart += outHTile_) {
                int ohEnd = std::min(ohStart + outHTile_, outH);
                for (int owStart = 0; owStart < outW; owStart += outWTile_) {
                    int owEnd = std::min(owStart + outWTile_, outW);

                    // ──────────────── 分块遍历输出通道 ────────────────
                    for (int ocStart = ocGlobalStart; ocStart < ocGlobalStart + ocPerGroup; ocStart += ocTile_) {
                        int ocEnd = std::min(ocStart + ocTile_, ocGlobalStart + ocPerGroup);

                        /*
                         * 在这个分块内，所有需要的数据都在 L2 缓存中：
                         * - 输入：ohStart*s ~ ohEnd*s+kH 之间的行
                         * - 权重：ocStart ~ ocEnd 之间的通道
                         * - 输出：ohStart ~ ohEnd, owStart ~ owEnd
                         *
                         * 内层循环的顺序是关键优化：
                         * - 最外层是 ic（累加维度），这样同一 ic 的权重可以
                         *   被该 tile 内的所有输出位置共享
                         * - 最内层是 oc_in_block，因为权重在 oc 维度连续存储
                         *   这样内层循环访问权重是顺序的，预取器可以提前加载
                         */
                        for (int ic = 0; ic < icPerGroup; ++ic) {
                            int icGlobal = icGlobalStart + ic;
                            for (int kh = 0; kh < kH_; ++kh) {
                                for (int kw = 0; kw < kW_; ++kw) {
                                    // 遍历分块内的输出位置
                                    for (int oh = ohStart; oh < ohEnd; ++oh) {
                                        int ih = oh * stride_ + kh - pad_;
                                        if (ih < 0 || ih >= H) continue;

                                        for (int ow = owStart; ow < owEnd; ++ow) {
                                            int iw = ow * stride_ + kw - pad_;
                                            if (iw < 0 || iw >= W) continue;

                                            float inputValue = input.at(n, icGlobal, ih, iw);

                                            /*
                                             * 最内层循环：遍历输出通道
                                             * 这是缓存优化的关键！
                                             *
                                             * 为什么把 oc 放在最内层？
                                             * - 权重布局：weight[oc][ic][kh][kw]
                                             *   在内存中，相邻的 oc 对应的权重是连续的
                                             * - 当 oh, ow, kh, kw, ic 固定时，
                                             *   weight[oc][ic][kh][kw] 对不同 oc 是连续内存访问
                                             * - CPU 预取器能检测到这种顺序访问模式，
                                             *   提前将下一批权重加载到缓存
                                             * - 这就是"流式访问"(streaming access) → 缓存友好
                                             */
                                            for (int oc = ocStart; oc < ocEnd; ++oc) {
                                                float w = weight_.at(oc, ic, kh, kw);
                                                output.at(n, oc, oh, ow) += inputValue * w;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // 加上偏置
            for (int oc = ocGlobalStart; oc < ocGlobalStart + ocPerGroup; ++oc) {
                float b = bias_[oc];
                for (int oh = 0; oh < outH; ++oh) {
                    for (int ow = 0; ow < outW; ++ow) {
                        output.at(n, oc, oh, ow) += b;
                    }
                }
            }
        }
    }
}

int64_t Conv2dTiled::flops(const std::vector<int>& inputShape,
                            const std::vector<int>& outputShape) const
{
    // FLOPs 与朴素卷积相同 —— 分块是缓存优化，不改变计算量
    if (inputShape.size() < 4 || outputShape.size() < 4) return 0;

    int64_t N    = inputShape[0];
    int64_t outH = outputShape[2];
    int64_t outW = outputShape[3];
    int64_t icPerGroup = inC_ / groups_;

    int64_t macsPerElement = static_cast<int64_t>(kH_) * kW_ * icPerGroup;
    int64_t outputElements = N * outC_ * outH * outW;

    return outputElements * macsPerElement * 2;
}

} // namespace edgeai
