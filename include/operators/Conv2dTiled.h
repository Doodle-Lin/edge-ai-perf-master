/**
 * @file Conv2dTiled.h
 * @brief 缓存分块卷积 —— 针对缓存层次结构的优化
 *
 * ====== 学习要点 ======
 * 1. 为什么缓存对性能至关重要？
 *    - 现代 CPU 内存层次：寄存器 < L1 (~32KB, 1ns) < L2 (~256KB, 4ns) < L3 (~8MB, 10ns) < 主存 (GB, 100ns)
 *    - 访问主存比访问 L1 缓存慢约 100 倍
 *    - 如果数据在缓存中，计算速度飞快；否则要等内存，CPU 空转
 *    - 这就是为什么"缓存友好"的代码可以比"缓存不友好"的代码快几倍甚至几十倍
 *
 * 2. 朴素卷积为什么缓存不友好？
 *    - 内层循环：遍历 ic 维度，对 input 的访问跨度为 H*W 个 float
 *    - 这意味着每次访问 input[n][ic+1][...] 都要跳过 H*W 个元素
 *    - 如果 H*W * sizeof(float) > L2 cache size，前一个 ic 的数据就被逐出了
 *    - 下次需要同一行 input 的不同 ic 时，又要从主存重新加载 → 缓存抖动 (thrashing)
 *
 * 3. 分块 (Tiling) 的核心思想：
 *    - 将输出划分为小块 (tile)，使得计算一个 tile 所需的所有数据能同时放入缓存
 *    - 具体：计算 outH_tile x outW_tile 大小的输出时，需要：
 *      - 输入数据：(outH_tile*s + kH - 1) * (outW_tile*s + kW - 1) * ic 个 float
 *      - 权重数据：kH * kW * ic * oc_tile 个 float
 *    - 选择 tile 大小，使得 输入 + 权重 < L2 cache size
 *    - 这样计算一个 tile 时，所有数据都在 L2 中，无需访问主存
 *
 * 4. 循环重排序 (Loop Reordering)：
 *    - 朴素顺序：N → outC → outH → outW → inC → kH → kW
 *    - 分块顺序：N → outC_block → outH_tile → outW_tile → inC → kH → kW → outC_in_block
 *    - 关键区别：最内层循环变成连续的 outC 维度
 *    - 因为权重在 outC 维度连续，这样权重访问是顺序的，缓存命中率最高
 *
 * 5. 分块大小的选择：
 *    - 理论最佳：使得 input_patch + kernel_weights 恰好填满 L2
 *    - 实际选择：留一些余量（L2 的 50-75%），因为缓存是组相联的，不能完全利用
 *    - 默认值：outH_tile = 16, outW_tile = 16, oc_tile = 32
 *    - 这些值需要根据具体 CPU 微架构调整，这就是"调参"的一部分
 *
 * 6. 分块 vs im2col + GEMM：
 *    - im2col + GEMM 内存膨胀大，但 GEMM 优化技术成熟
 *    - 分块卷积无内存膨胀，但循环更复杂
 *    - 实际中两者常常结合：先 im2col 再分块 GEMM
 *    - 本文件演示的是直接分块卷积（不经过 im2col）
 */

#pragma once

#include "IOperator.h"
#include <vector>

namespace edgeai {

class Conv2dTiled : public IOperator {
public:
    /**
     * @brief 构造缓存分块卷积算子
     * @param inC        输入通道数
     * @param outC       输出通道数
     * @param kH         卷积核高度
     * @param kW         卷积核宽度
     * @param stride     步长（默认 1）
     * @param pad        填充（默认 0）
     * @param groups     分组数（默认 1）
     * @param outHTile   输出高度方向的分块大小（默认 16）
     * @param outWTile   输出宽度方向的分块大小（默认 16）
     * @param ocTile     输出通道方向的分块大小（默认 32）
     */
    Conv2dTiled(int inC, int outC, int kH, int kW,
                int stride = 1, int pad = 0, int groups = 1,
                int outHTile = 16, int outWTile = 16, int ocTile = 32);

    void forward(const Tensor& input, Tensor& output) override;
    const char* name() const override { return "Conv2dTiled"; }
    int64_t flops(const std::vector<int>& inputShape,
                  const std::vector<int>& outputShape) const override;

private:
    int inC_, outC_, kH_, kW_, stride_, pad_, groups_;
    int outHTile_, outWTile_, ocTile_;  ///< 分块大小
    Tensor weight_;   ///< 卷积核权重 [outC, inC/groups, kH, kW]
    Tensor bias_;     ///< 偏置 [outC]
};

} // namespace edgeai
