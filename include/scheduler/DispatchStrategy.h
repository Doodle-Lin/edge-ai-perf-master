/**
 * @file DispatchStrategy.h
 * @brief 调度策略接口 —— 定义算子到设备的分配契约
 *
 * ====== 学习要点 ======
 *
 * 1. 策略模式（Strategy Pattern）
 *    这是经典设计模式之一：将算法封装为可互换的对象。
 *    HybridExecutor 只依赖 DispatchStrategy 接口，不关心具体策略。
 *    运行时可以切换策略而不修改执行器代码 —— 符合开闭原则（OCP）。
 *
 * 2. 为什么用虚基类而不是 std::function？
 *    - 策略有状态（如 ProfileGuidedDispatch 维护历史数据表）
 *    - 策略需要自描述（name() 方法用于日志和调试）
 *    - 未来可能需要序列化策略配置
 *    虚基类比 std::function 更适合这种"有身份、有状态"的场景。
 *
 * 3. DispatchDecision 的设计
 *    不仅记录"分到哪个设备"，还记录"为什么这样分"（reason 字段）。
 *    这在生产中极其重要：
 *    - 调试时可以追溯调度逻辑
 *    - A/B 测试时可以比较不同策略的决策差异
 *    - 审计时可以证明决策是确定性的
 *
 * 4. Device 枚举
 *    目前只有 CPU / GPU 两种。未来可扩展：
 *    - NPU（神经网络处理器，如华为昇腾）
 *    - DSP（数字信号处理器，如高通 Hexagon）
 *    - FPGA
 *    枚举类比字符串更安全——编译期就能发现拼写错误。
 */

#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include "scheduler/OpGraph.h"

namespace edgeai {

// ============================================================================
// 设备枚举
// ============================================================================

/// 目标计算设备类型
enum class Device {
    CPU,    ///< 中央处理器 —— 适合小算子、逐元素操作、控制密集型任务
    GPU     ///< 图形处理器 —— 适合大算子、计算密集型、数据并行任务
    // 未来扩展: NPU, DSP, FPGA ...
};

/// 将 Device 枚举转为字符串（用于日志输出）
inline const char* deviceToString(Device d) {
    return d == Device::CPU ? "CPU" : "GPU";
}

// ============================================================================
// 调度决策
// ============================================================================

/**
 * @struct DispatchDecision
 * @brief 单个算子的调度决策结果
 *
 * 包含三要素：
 * 1. opId   —— 哪个算子
 * 2. device —— 分配到哪个设备
 * 3. reason —— 为什么这样分配（可追溯的决策依据）
 *
 * 示例：
 * @code
 *   DispatchDecision d;
 *   d.opId   = 3;
 *   d.device = Device::GPU;
 *   d.reason = "Conv2D with 512 channels, FLOPs=118M > threshold 10M";
 * @endcode
 */
struct DispatchDecision {
    int opId          = -1;     ///< 算子 id
    Device device     = Device::CPU;  ///< 分配的设备
    std::string reason;        ///< 决策原因描述（人类可读）
};

// ============================================================================
// 调度策略接口
// ============================================================================

/**
 * @class DispatchStrategy
 * @brief 调度策略的抽象基类
 *
 * 所有调度策略必须实现两个方法：
 * 1. dispatch() —— 核心方法，根据算子特征和图结构做出设备分配
 * 2. name()     —— 返回策略名称，用于日志和配置
 *
 * 使用多态而非模板的原因：
 * - 运行时可切换策略（如先 heuristic，后 profile-guided）
 * - 策略可能需要跨多次调用累积状态（profile-guided 需要历史数据）
 * - 方便在配置文件中指定策略类名
 */
class DispatchStrategy {
public:
    virtual ~DispatchStrategy() = default;

    /**
     * @brief 为单个算子决定执行设备
     * @param op    待调度的算子节点
     * @param graph 完整的算子图（提供上下文信息，如邻居算子）
     * @return 调度决策（包含设备选择和原因）
     *
     * 为什么需要传入整个 graph？
     * - 数据局部性：如果前驱算子在 GPU 上，当前算子也分到 GPU
     *   可以避免 CPU↔GPU 之间不必要的内存拷贝
     * - 负载均衡：可以查看同阶段有多少 GPU 任务，避免 GPU 过载
     * - 全局优化：某些策略需要知道整图结构才能做出最优决策
     */
    virtual DispatchDecision dispatch(const OpNode& op, const OpGraph& graph) = 0;

    /**
     * @brief 返回策略名称
     * @return 策略标识符，如 "HeuristicDispatch"、"ProfileGuidedDispatch"
     *
     * 用于日志输出和配置匹配。
     */
    virtual std::string name() const = 0;
};

// ============================================================================
// 工具函数
// ============================================================================

/// 将 DispatchDecision 格式化为可读字符串
/// 示例: "[Op#3 conv1] -> GPU (large conv with 512 channels)"
inline std::string formatDecision(const DispatchDecision& d) {
    std::ostringstream oss;
    oss << "[Op#" << d.opId << "] -> "
        << deviceToString(d.device)
        << " (" << d.reason << ")";
    return oss.str();
}

/// 打印调度决策到 stdout
inline void printDecision(const DispatchDecision& d) {
    std::cout << formatDecision(d) << "\n";
}

} // namespace edgeai
