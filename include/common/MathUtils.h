/**
 * @file MathUtils.h
 * @brief 通用数学工具 —— 算子开发和性能分析的基础函数
 *
 * ====== 学习要点 ======
 * 1. FLOPs 计算是性能分析的核心:
 *    - 知道算子做了多少次浮点运算，才能算出 GFLOPS
 *    - GFLOPS = FLOPs / 时间，反映硬件利用率
 *    - 不同算子有不同的 FLOPs 公式，理解公式才能判断是否接近理论上限
 *
 * 2. 内存带宽分析:
 *    - 很多算子不是计算瓶颈，而是内存瓶颈（数据搬不动）
 *    - arithmetic intensity = FLOPs / 内存访问量，区分计算密集 vs 内存密集
 *    - Roofline Model: 用 arithmetic intensity 判断算子落在哪个区域
 *
 * 3. 对齐:
 *    - SIMD 指令 (AVX2) 要求数据 32 字节对齐
 *    - GPU buffer 通常要求 256 字节对齐
 *    - 内存对齐减少跨 cache line 访问，提升吞吐
 */

#pragma once

#include <cstdint>
#include <cstdlib>

namespace edgeai {

/// 内存对齐: 将 size 向上取整到 alignment 的整数倍
/// 例: alignUp(17, 16) = 32
inline size_t alignUp(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

/// 返回大于等于 n 的最小 2 的幂
/// 例: nextPowerOf2(5) = 8, nextPowerOf2(8) = 8
inline int nextPowerOf2(int n) {
    if (n <= 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

/// 计算标准 Conv2D 的 FLOPs (乘加算 2 次运算)
/// 公式: 2 * batchSize * outChannels * outH * outW * (inChannels/groups * kH * kW)
/// @param inC  输入通道数
/// @param outC 输出通道数
/// @param kH   卷积核高度
/// @param kW   卷积核宽度
/// @param outH 输出特征图高度
/// @param outW 输出特征图宽度
/// @param groups 分组数 (depthwise 时 groups = inC = outC)
/// @return 浮点运算次数
inline int64_t flopCountConv2d(int inC, int outC, int kH, int kW,
                                int outH, int outW, int groups = 1) {
    int64_t kernelsMulAdd = static_cast<int64_t>(inC / groups) * kH * kW;
    int64_t outputElements = static_cast<int64_t>(outC) * outH * outW;
    // 每个输出元素: kernelsMulAdd 次乘法 + kernelsMulAdd 次加法 = 2 * kernelsMulAdd
    return 2 * outputElements * kernelsMulAdd;
}

/// 计算标准 Conv2D 的内存访问量 (bytes)
/// = 输入 + 权重 + 输出
/// @param inC  输入通道数
/// @param outC 输出通道数
/// @param kH   卷积核高度
/// @param kW   卷积核宽度
/// @param inH  输入特征图高度
/// @param inW  输入特征图宽度
/// @param groups 分组数
/// @return 内存字节数 (float = 4 bytes)
inline int64_t memSizeConv2d(int inC, int outC, int kH, int kW,
                              int inH, int inW, int groups = 1) {
    int64_t inputBytes  = static_cast<int64_t>(inC) * inH * inW * sizeof(float);
    int64_t weightBytes = static_cast<int64_t>(outC) * (inC / groups) * kH * kW * sizeof(float);
    int64_t outputBytes = static_cast<int64_t>(outC) * inH * inW * sizeof(float); // 近似: outH ≈ inH
    return inputBytes + weightBytes + outputBytes;
}

/// 从 FLOPs 和时间计算 GFLOPS
/// @param flops  浮点运算次数
/// @param timeMs 耗时(毫秒)
/// @return GFLOPS (10^9 次浮点运算/秒)
inline double gflops(int64_t flops, double timeMs) {
    if (timeMs <= 0.0) return 0.0;
    return static_cast<double>(flops) / (timeMs * 1e6);
}

/// 从内存访问量和时间计算带宽 (GB/s)
/// @param bytes  内存访问字节数
/// @param timeMs 耗时(毫秒)
/// @return GB/s (10^9 字节/秒)
inline double memoryBandwidth(int64_t bytes, double timeMs) {
    if (timeMs <= 0.0) return 0.0;
    return static_cast<double>(bytes) / (timeMs * 1e6);
}

} // namespace edgeai
