/**
 * @file HeuristicDispatch.h
 * @brief 基于启发式规则的调度策略 —— 根据算子特征和经验规则决定设备分配
 *
 * ====== 学习要点 ======
 *
 * 1. 什么是启发式（Heuristic）？
 *    启发式 = "凭经验的规则"，不一定最优但通常够好。
 *    例如："大的卷积放 GPU"——这不是数学证明，而是工程实践中的经验。
 *    优点：不需要 profiling 数据，冷启动就能工作。
 *    缺点：可能不够精确，无法适应所有硬件组合。
 *
 * 2. CPU vs GPU 的本质区别
 *    ┌─────────────────┬──────────────────────┬──────────────────────┐
 *    │                 │ CPU                  │ GPU                   │
 *    ├─────────────────┼──────────────────────┼──────────────────────┤
 *    │ 核心数          │ 少（4~64）           │ 多（数百~数千）       │
 *    │ 单核性能        │ 强（高主频、大缓存）  │ 弱（低主频、小缓存）  │
 *    │ 内存带宽        │ 低（~50 GB/s）       │ 高（~500 GB/s）      │
 *    │ 核启动开销      │ 无                   │ 有（~5~50μs）        │
 *    │ 适合任务        │ 小算子、控制密集型   │ 大算子、数据并行     │
 *    └─────────────────┴──────────────────────┴──────────────────────┘
 *
 * 3. Kernel Launch Overhead
 *    GPU 执行任何操作前，CPU 需要通过驱动把命令提交到 GPU 队列，
 *    这个提交过程本身就需要 ~5-50 微秒。如果一个算子本身只执行 2μs，
 *    那 launch overhead 就是瓶颈，反而比 CPU 慢。
 *    所以"小算子放 CPU"是核心启发式规则之一。
 *
 * 4. Memory-bound vs Compute-bound
 *    - Compute-bound：计算密度高（FLOPs/Byte > 10），如大卷积 → GPU 优势大
 *    - Memory-bound：计算密度低（FLOPs/Byte < 5），如 ReLU、ElementWise Add
 *      GPU 虽然带宽高，但也要搬运数据，对小张量不一定划算 → CPU 更优
 *
 * 5. 规则的可配置性
 *    setRule() 允许用户覆盖默认阈值，适配特定硬件。
 *    例如：嵌入式 GPU 的 launch overhead 可能只有 2μs，阈值可以调低。
 */

#pragma once

#include "scheduler/DispatchStrategy.h"
#include <functional>
#include <string>
#include <vector>

namespace edgeai {

// ============================================================================
// 调度规则
// ============================================================================

/// 单条启发式规则：匹配条件 + 目标设备
struct DispatchRule {
    std::string condition;  ///< 规则条件描述，如 "flops < 1000000"
    Device targetDevice;    ///< 满足条件时分配的设备
    int priority = 0;       ///< 优先级，数值越大越先匹配
};

// ============================================================================
// 启发式调度策略
// ============================================================================

/**
 * @class HeuristicDispatch
 * @brief 基于启发式规则的调度策略
 *
 * 内置规则（按优先级从高到低）：
 * 1. 用户指定的 preferredDevice（OpNode::preferredDevice != "AUTO"时）
 * 2. 小算子（FLOPs < flopsThreshold）→ CPU
 * 3. 逐元素操作（ReLU、Add、Mul、Sigmoid 等）→ CPU
 * 4. 大卷积（Conv2D 且输出通道 > largeConvChannels）→ GPU
 * 5. 大矩阵乘法（Gemm）→ GPU
 * 6. 默认 → CPU（保守策略，避免小算子浪费 GPU 资源）
 *
 * 用法：
 * @code
 *   HeuristicDispatch heuristic;
 *   heuristic.setFlopsThreshold(5'000'000);  // 5M FLOPs 以下放 CPU
 *   heuristic.setLargeConvChannels(128);      // 输出通道 > 128 的卷积放 GPU
 *
 *   // 添加自定义规则
 *   heuristic.setRule("custom_pool", [](const OpNode& op) {
 *       return op.type == "Pooling" && op.estimatedFlops > 1000000;
 *   }, Device::GPU, 5);
 * @endcode
 */
class HeuristicDispatch : public DispatchStrategy {
public:
    HeuristicDispatch();

    // ──────────────── 核心接口 ────────────────

    /**
     * @brief 根据启发式规则决定算子的执行设备
     * @param op    待调度的算子
     * @param graph 算子图（提供上下文）
     * @return 调度决策
     *
     * 决策流程：
     * 1. 检查 preferredDevice 是否显式指定
     * 2. 遍历自定义规则（按优先级排序）
     * 3. 应用内置规则
     * 4. 兜底返回 CPU
     */
    DispatchDecision dispatch(const OpNode& op, const OpGraph& graph) override;

    /// 返回策略名称
    std::string name() const override { return "HeuristicDispatch"; }

    // ──────────────── 配置接口 ────────────────

    /// 设置"小算子"FLOPs 阈值，低于此值分到 CPU（默认 1M）
    void setFlopsThreshold(int64_t threshold) { flopsThreshold_ = threshold; }

    /// 设置"大卷积"输出通道阈值，高于此值分到 GPU（默认 64）
    void setLargeConvChannels(int channels) { largeConvChannels_ = channels; }

    /// 设置"大算子"FLOPs 阈值，高于此值分到 GPU（默认 10M）
    void setLargeFlopsThreshold(int64_t threshold) { largeFlopsThreshold_ = threshold; }

    /**
     * @brief 添加自定义调度规则
     * @param ruleName   规则名称（用于日志）
     * @param matcher    匹配函数，返回 true 表示规则命中
     * @param target     命中时分配的设备
     * @param priority   优先级（默认 0，值越大越优先匹配）
     *
     * 自定义规则会先于内置规则按优先级从高到低匹配。
     * 同一优先级内，后添加的规则先匹配（栈式）。
     */
    void setRule(const std::string& ruleName,
                 std::function<bool(const OpNode&)> matcher,
                 Device target,
                 int priority = 0);

    /// 清除所有自定义规则
    void clearCustomRules();

    /// 获取当前的 FLOPs 阈值
    int64_t flopsThreshold() const { return flopsThreshold_; }

private:
    // ──────────────── 内置规则判断 ────────────────

    /// 判断是否为逐元素操作类型（ReLU、Add、Mul、Sigmoid 等）
    /// 这些操作的特点：FLOPs ≈ 元素数，计算密度极低，GPU 无优势
    bool isElementWiseOp(const std::string& type) const;

    /// 判断是否为计算密集型卷积（输出通道数 > largeConvChannels_）
    bool isLargeConv(const OpNode& op) const;

    // ──────────────── 配置参数 ────────────────

    int64_t flopsThreshold_     = 1'000'000;  ///< 小算子 FLOPs 阈值
    int64_t largeFlopsThreshold_ = 10'000'000; ///< 大算子 FLOPs 阈值
    int largeConvChannels_      = 64;           ///< 大卷积输出通道阈值

    // ──────────────── 自定义规则存储 ────────────────

    /// 自定义规则条目
    struct CustomRule {
        std::string name;                       ///< 规则名
        std::function<bool(const OpNode&)> matcher; ///< 匹配函数
        Device target;                          ///< 目标设备
        int priority;                           ///< 优先级
    };

    std::vector<CustomRule> customRules_;  ///< 自定义规则列表
};

} // namespace edgeai
