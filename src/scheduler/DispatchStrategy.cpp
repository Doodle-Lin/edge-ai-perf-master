/**
 * @file DispatchStrategy.cpp
 * @brief 调度策略接口的实现 —— 由于工具函数已内联到头文件，此文件仅包含翻译单元占位
 *
 * ====== 实现说明 ======
 *
 * DispatchStrategy 的核心功能（deviceToString, formatDecision, printDecision）
 * 已作为 inline 函数定义在头文件中，无需在此 .cpp 中重复定义。
 *
 * 此文件仍然保留，原因：
 * 1. 确保 scheduler/DispatchStrategy.h 的 include 语法正确
 * 2. 未来如果有非 inline 的工具函数（如从 JSON 加载策略配置），
 *    可以在此扩展
 * 3. 保持文件结构的一致性（每个 .h 都有对应的 .cpp）
 */

#include "scheduler/DispatchStrategy.h"

namespace edgeai {

// 当前无额外的非 inline 实现。
// 如果未来需要添加以下功能，可以在此实现：
// - 从 JSON 配置文件加载策略
// - 策略注册表（按名称创建策略实例）
// - 策略序列化/反序列化

} // namespace edgeai
