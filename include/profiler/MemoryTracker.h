/**
 * @file MemoryTracker.h
 * @brief 内存分配追踪器 —— 监控推理过程中的内存分配/释放，识别峰值和泄漏
 *
 * ====== 学习要点 ======
 * 1. 推理引擎的内存模型：
 *    - 权重内存（静态）：模型参数，加载后不变
 *    - 激活内存（动态）：中间特征图，逐层分配/释放
 *    - 工作空间（临时）：算子内部临时缓冲区
 *
 * 2. 峰值内存 = 最大同时驻留内存 → 决定模型能否在设备上运行
 *    - 优化手段：内存复用（in-place 算子）、共享内存池、子图融合
 *
 * 3. 单例模式 (Singleton)：
 *    - 全局唯一的追踪器实例，避免多个追踪器冲突
 *    - 使用 Meyer's Singleton（静态局部变量），线程安全（C++11 起保证）
 *
 * 4. 内存带宽墙 (Memory Wall)：
 *    - 即使算力足够，内存带宽也可能成为瓶颈
 *    - 关键指标：峰值带宽利用率 = 实际带宽 / 理论带宽
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace edgeai {

/// 单次内存分配/释放的记录
struct MemRecord {
    std::string tag;            ///< 分配标签，如 "conv1_weight", "feature_map_3"
    size_t sizeBytes    = 0;    ///< 本次分配/释放的大小（字节）
    size_t totalAllocated = 0;  ///< 操作后的累计分配总量
    bool isAllocation   = true; ///< true=分配事件, false=释放事件
};

/**
 * @class MemoryTracker
 * @brief 全局内存分配追踪器（单例）
 *
 * 使用方法：
 * @code
 *   auto& tracker = MemoryTracker::instance();
 *
 *   // 在自定义分配器中嵌入追踪
 *   void* ptr = malloc(1024);
 *   tracker.onAlloc("my_tensor", 1024);
 *
 *   free(ptr);
 *   tracker.onFree("my_tensor", 1024);
 *
 *   tracker.printReport();   // 打印内存使用报告
 *   printf("Peak: %zu bytes\n", tracker.peakUsage());
 * @endcode
 *
 * 设计说明：
 * - onAlloc/onFree 是轻量级操作，开销仅限 vector push_back
 * - history_ 记录完整分配历史，可用于后处理生成内存时间线
 * - 在生产环境中可编译期宏控制是否启用追踪，避免性能退化
 */
class MemoryTracker {
public:
    /// 获取全局唯一实例（Meyer's Singleton，线程安全）
    static MemoryTracker& instance();

    /// 记录一次内存分配
    /// @param tag   分配标签，便于识别是哪个张量/算子的内存
    /// @param bytes 分配的字节数
    void onAlloc(const std::string& tag, size_t bytes);

    /// 记录一次内存释放
    /// @param tag   释放标签，应与 onAlloc 的 tag 对应
    /// @param bytes 释放的字节数
    void onFree(const std::string& tag, size_t bytes);

    /// 当前内存使用量（字节）= 累计分配 - 累计释放
    size_t currentUsage() const { return current_; }

    /// 峰值内存使用量（字节）→ 最关键指标，决定是否能部署到目标设备
    size_t peakUsage() const { return peak_; }

    /// 累计分配总量（字节）
    size_t totalAllocated() const { return totalAllocated_; }

    /// 累计释放总量（字节）
    size_t totalFreed() const { return totalFreed_; }

    /// 打印内存使用报告到 stdout
    /// 包含：当前用量、峰值、累计分配/释放、碎片率
    void printReport() const;

    /// 序列化为 JSON 字符串
    /// 格式：{ "current": ..., "peak": ..., "history": [ ... ] }
    std::string toJson() const;

    /// 重置所有追踪状态（用于多次推理之间清理）
    void reset();

    /// 将字节数以人类可读格式附加输出（如 " (12.34 MB)"）
    void printSizeHuman(size_t bytes) const;

private:
    // 单例模式：私有构造，禁止拷贝和移动
    MemoryTracker() = default;
    MemoryTracker(const MemoryTracker&) = delete;
    MemoryTracker& operator=(const MemoryTracker&) = delete;

    size_t current_        = 0;   ///< 当前在用内存（字节）
    size_t peak_           = 0;   ///< 历史峰值内存（字节）
    size_t totalAllocated_ = 0;   ///< 累计分配总量（字节）
    size_t totalFreed_     = 0;   ///< 累计释放总量（字节）

    /// 完整的分配/释放历史记录
    /// 注意：推理过程中可能产生大量记录，在大模型场景下需考虑内存开销
    std::vector<MemRecord> history_;
};

} // namespace edgeai
