/**
 * @file IOperator.h
 * @brief 算子基类接口 —— 所有优化实现的统一抽象
 *
 * ====== 学习要点 ======
 * 1. 为什么要定义统一接口？
 *    - 不同的卷积实现（朴素、AVX2、分块、Winograd、Vulkan）在内部算法上完全不同
 *    - 但它们的外部行为完全一致：输入一张图，输出一张图
 *    - 通过 IOperator 接口，上层代码只需调用 forward()，不关心底层用哪种算法
 *    - 这就是"策略模式"(Strategy Pattern)：运行时切换算法而不改变调用方
 *
 * 2. 为什么 forward() 用 (input, output) 而不是 return output？
 *    - 避免每次调用都分配新内存（Tensor 构造涉及堆分配，很慢）
 *    - 预分配 output 张量，反复复用同一块内存 → 减少内存分配开销
 *    - 这是推理引擎中的常见做法：内存池预分配 + 复用
 *
 * 3. flops() 方法的作用：
 *    - FLOPs (Floating Point Operations) 是衡量计算量的绝对指标
 *    - 不同算法的 FLOPs 可能不同（如 Winograd 比朴素卷积少 2.25 倍乘法）
 *    - FLOPs / 执行时间 = GFLOPS（吞吐率），可以衡量硬件利用率
 *    - 如果实测 GFLOPS 接近硬件峰值 → 说明优化到位；远低于峰值 → 还有优化空间
 *
 * 4. name() 返回 const char* 而非 std::string：
 *    - 避免每次调用都构造 std::string 对象（涉及堆分配）
 *    - 字符串字面量存储在只读段，返回指针零开销
 */

#pragma once

#include "common/Tensor.h"
#include <cstdint>
#include <vector>

namespace edgeai {

class IOperator {
public:
    /// 虚析构函数 —— 确保通过基类指针删除时调用正确的派生类析构函数
    /// 如果不写 virtual，delete 基类指针只会调用基类析构，派生类资源泄漏
    virtual ~IOperator() = default;

    /// 前向推理：input → 算子计算 → output
    /// @param input  输入张量，形状为 [N, inC, H, W]
    /// @param output 输出张量，由调用方预分配好空间
    virtual void forward(const Tensor& input, Tensor& output) = 0;

    /// 获取算子名称，用于日志和性能分析
    /// 返回 const char* 而非 std::string，避免堆分配
    virtual const char* name() const = 0;

    /// 计算该算子的浮点运算次数 (FLOPs)
    /// @param inputShape  输入形状 [N, C, H, W]
    /// @param outputShape 输出形状 [N, C, H, W]
    /// @return 浮点运算次数。1 次 MAC（乘加）= 2 FLOPs
    ///
    /// 为什么需要 inputShape 和 outputShape 作为参数，而不是用成员变量？
    /// - 这样 flops() 可以在不需要实例化算子的情况下计算 FLOPs
    /// - 方便在模型编译阶段（而非推理阶段）进行性能预估
    virtual int64_t flops(const std::vector<int>& inputShape,
                          const std::vector<int>& outputShape) const = 0;
};

} // namespace edgeai
