/**
 * @file HeuristicDispatch.cpp
 * @brief 启发式调度策略的实现 —— 基于算子特征的规则匹配引擎
 *
 * ====== 实现说明 ======
 *
 * dispatch() 的决策流程：
 *
 *   ┌─────────────────────────────────────┐
 *   │ 1. 检查 preferredDevice 是否指定     │ ← 最高优先级：用户显式指定
 *   │    如果 != "AUTO"，直接返回           │
 *   └──────────────┬──────────────────────┘
 *                  ↓
 *   ┌─────────────────────────────────────┐
 *   │ 2. 遍历自定义规则（按优先级排序）     │ ← 用户可扩展的规则层
 *   │    匹配则返回                         │
 *   └──────────────┬──────────────────────┘
 *                  ↓
 *   ┌─────────────────────────────────────┐
 *   │ 3. 内置规则匹配                      │ ← 核心启发式逻辑
 *   │    a. 小算子 (FLOPs < threshold)→CPU │
 *   │    b. 逐元素操作 → CPU                │
 *   │    c. 大卷积 → GPU                   │
 *   │    d. 大算子 (FLOPs > threshold)→GPU │
 *   └──────────────┬──────────────────────┘
 *                  ↓
 *   ┌─────────────────────────────────────┐
 *   │ 4. 默认 → CPU                        │ ← 保守策略
 *   └─────────────────────────────────────┘
 *
 * 每条规则匹配后都会记录 reason 字符串，解释为什么选择该设备。
 * 这对调试和性能分析非常重要。
 */

#include "scheduler/HeuristicDispatch.h"

#include <algorithm>
#include <sstream>

namespace edgeai {

// ============================================================================
// 构造函数
// ============================================================================

HeuristicDispatch::HeuristicDispatch() = default;

// ============================================================================
// 核心调度逻辑
// ============================================================================

DispatchDecision HeuristicDispatch::dispatch(const OpNode& op, const OpGraph& /*graph*/) {
    DispatchDecision decision;
    decision.opId = op.id;

    // ── 规则 0：用户显式指定设备 ──
    // 如果 OpNode 的 preferredDevice 不是 "AUTO"，直接使用用户指定
    if (!op.preferredDevice.empty() && op.preferredDevice != "AUTO") {
        if (op.preferredDevice == "GPU") {
            decision.device = Device::GPU;
            decision.reason = "user-specified GPU preference for '" + op.name + "'";
            return decision;
        } else if (op.preferredDevice == "CPU") {
            decision.device = Device::CPU;
            decision.reason = "user-specified CPU preference for '" + op.name + "'";
            return decision;
        }
    }

    // ── 规则 1：自定义规则匹配（按优先级从高到低）──
    // 先按优先级排序（stable_sort 保证同优先级内维持插入顺序）
    // 注意：每次 dispatch 都排序可能影响性能，但自定义规则通常很少，
    // 且编译阶段才调用，不影响推理性能
    if (!customRules_.empty()) {
        // 按优先级从高到低排序的临时索引
        std::vector<size_t> indices(customRules_.size());
        for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
        std::stable_sort(indices.begin(), indices.end(),
            [this](size_t a, size_t b) {
                return customRules_[a].priority > customRules_[b].priority;
            });

        for (size_t idx : indices) {
            const auto& rule = customRules_[idx];
            if (rule.matcher(op)) {
                decision.device = rule.target;
                decision.reason = "custom rule '" + rule.name + "' matched for '" + op.name + "'";
                return decision;
            }
        }
    }

    // ── 规则 2a：小算子 → CPU ──
    // "小"的定义：estimatedFlops < flopsThreshold_
    // 小算子在 GPU 上会遭遇 kernel launch overhead 瓶颈
    // 典型场景：1x1 卷积 with 16 channels，FLOPs ≈ 30K
    if (op.estimatedFlops > 0 && op.estimatedFlops < flopsThreshold_) {
        decision.device = Device::CPU;
        std::ostringstream reason;
        reason << "small op (FLOPs=" << op.estimatedFlops
               << " < threshold=" << flopsThreshold_
               << "), kernel launch overhead dominates on GPU";
        decision.reason = reason.str();
        return decision;
    }

    // ── 规则 2b：逐元素操作 → CPU ──
    // 逐元素操作（ReLU、Add、Sigmoid 等）的计算密度极低
    // FLOPs ≈ 元素数，Bytes ≈ 2 * 元素数，计算密度 ≈ 0.5
    // GPU 虽然带宽高，但数据需要从 CPU 拷贝到 GPU，
    // 对小张量来说拷贝开销就抵消了带宽优势
    if (isElementWiseOp(op.type)) {
        decision.device = Device::CPU;
        decision.reason = "element-wise op '" + op.type +
                          "', memory-bound with no compute advantage on GPU";
        return decision;
    }

    // ── 规则 3：大卷积 → GPU ──
    // 卷积是典型的计算密集型操作：
    // FLOPs = 2 * K * K * Cin * Cout * H * W
    // 当输出通道数大时，FLOPs 非常高，GPU 的并行计算优势明显
    // 这里用 estimatedFlops 粗略判断是否为"大卷积"
    if (isLargeConv(op)) {
        decision.device = Device::GPU;
        std::ostringstream reason;
        reason << "large conv (type=" << op.type
               << ", FLOPs=" << op.estimatedFlops
               << "), compute-intensive → GPU favored";
        decision.reason = reason.str();
        return decision;
    }

    // ── 规则 4：大算子 → GPU ──
    // 不只是卷积，大的 Gemm/MatMul 也适合 GPU
    if (op.estimatedFlops >= largeFlopsThreshold_) {
        decision.device = Device::GPU;
        std::ostringstream reason;
        reason << "large op (FLOPs=" << op.estimatedFlops
               << " >= threshold=" << largeFlopsThreshold_
               << "), compute-intensive → GPU favored";
        decision.reason = reason.str();
        return decision;
    }

    // ── 规则 5：池化和形状操作 → CPU ──
    // Pooling 和 Reshape 等操作的 FLOPs 很低，
    // 主要是数据搬运，属于内存密集型，CPU 更合适
    std::string typeLower = op.type;
    std::transform(typeLower.begin(), typeLower.end(), typeLower.begin(), ::tolower);
    if (typeLower.find("pool") != std::string::npos ||
        typeLower.find("reshape") != std::string::npos ||
        typeLower.find("flatten") != std::string::npos ||
        typeLower.find("concat") != std::string::npos ||
        typeLower.find("split") != std::string::npos ||
        typeLower.find("slice") != std::string::npos ||
        typeLower.find("permute") != std::string::npos ||
        typeLower.find("transpose") != std::string::npos) {
        decision.device = Device::CPU;
        decision.reason = "memory/layout op '" + op.type + "', no compute benefit on GPU";
        return decision;
    }

    // ── 默认规则：CPU ──
    // 保守策略：如果算子特征不明显，优先分配 CPU
    // 原因：CPU 不需要数据搬运，延迟确定
    // 而 GPU 需要额外的 CPU↔GPU 拷贝，对小算子可能更慢
    decision.device = Device::CPU;
    decision.reason = "default fallback to CPU (no strong GPU indicator)";
    return decision;
}

// ============================================================================
// 内置规则判断
// ============================================================================

bool HeuristicDispatch::isElementWiseOp(const std::string& type) const {
    // 逐元素操作的特征：
    // - 输出形状 = 输入形状
    // - 每个输出元素只依赖输入的对应元素
    // - FLOPs ≈ 元素数（1~2 FLOP/element）
    // - 计算密度极低（< 1 FLOP/Byte）

    std::string typeLower = type;
    std::transform(typeLower.begin(), typeLower.end(), typeLower.begin(), ::tolower);

    // 常见逐元素操作列表
    // ReLU: y = max(0, x)，1 FLOP/element
    // LeakyReLU: y = x > 0 ? x : alpha * x，2 FLOP/element
    // Sigmoid: y = 1 / (1 + exp(-x))，~4 FLOP/element
    // Tanh: y = tanh(x)，~6 FLOP/element
    // Add: y = a + b，1 FLOP/element
    // Mul: y = a * b，1 FLOP/element
    // Clip/Clamp: y = min(max(x, lo), hi)，2 FLOP/element
    // Scale: y = a * x + b，2 FLOP/element
    // BN (inference): y = gamma * (x - mean) / sqrt(var + eps) + beta，~5 FLOP/element
    static const std::vector<std::string> elementWiseTypes = {
        "relu", "relun", "leakyrelu", "prelu",
        "sigmoid", "swish", "hardsigmoid", "hardswish",
        "tanh", "hardtanh",
        "add", "sum", "eltwise",
        "mul", "product",
        "clip", "clamp", "threshold",
        "scale", "bias",
        "batchnorm", "bn", "batchnorm1d", "batchnorm2d", "batchnorm3d",
        "instancenorm", "groupnorm", "layernorm",
        "elu", "selu", "celu", "gelu", "mish",
        "abs", "neg", "pow", "sqrt", "exp", "log",
        "dropout"
    };

    for (const auto& ewType : elementWiseTypes) {
        if (typeLower.find(ewType) != std::string::npos) {
            return true;
        }
    }

    return false;
}

bool HeuristicDispatch::isLargeConv(const OpNode& op) const {
    // 判断是否为"大卷积"—— 适合 GPU 的计算密集型卷积
    std::string typeLower = op.type;
    std::transform(typeLower.begin(), typeLower.end(), typeLower.begin(), ::tolower);

    // 检查是否为卷积类型
    bool isConvType = (typeLower.find("conv") != std::string::npos ||
                       typeLower.find("deconv") != std::string::npos ||
                       typeLower.find("convolution") != std::string::npos ||
                       typeLower.find("depthwise") != std::string::npos);

    if (!isConvType) return false;

    // 判断卷积规模：
    // 方式 1：直接看 estimatedFlops（最可靠）
    // 方式 2：看名称中的通道数信息（不可靠但作为辅助）
    if (op.estimatedFlops >= largeFlopsThreshold_) {
        return true;
    }

    // 如果 estimatedFlops 为 0（未设置），保守地认为不是大卷积
    return false;
}

// ============================================================================
// 自定义规则管理
// ============================================================================

void HeuristicDispatch::setRule(const std::string& ruleName,
                                 std::function<bool(const OpNode&)> matcher,
                                 Device target,
                                 int priority) {
    CustomRule rule;
    rule.name     = ruleName;
    rule.matcher  = std::move(matcher);
    rule.target   = target;
    rule.priority = priority;
    customRules_.push_back(std::move(rule));
}

void HeuristicDispatch::clearCustomRules() {
    customRules_.clear();
}

} // namespace edgeai
