/**
 * @file MemoryTracker.cpp
 * @brief 内存分配追踪器的实现
 *
 * ====== 实现要点 ======
 * 1. Meyer's Singleton（静态局部变量）：
 *    - C++11 起保证静态局部变量的初始化是线程安全的
 *    - 编译器自动生成内嵌锁，无需手动加锁
 *    - 析构顺序：与编译单元的静态析构顺序一致，可能存在跨编译单元依赖问题
 *    - 在本场景中，MemoryTracker 不依赖其他单例，所以安全
 *
 * 2. 峰值内存追踪：
 *    - 每次分配后检查 current_ > peak_，更新峰值
 *    - 释放不会降低峰值——峰值代表"曾经最高使用量"
 *    - 峰值决定了模型能否在内存受限的边缘设备上运行
 *
 * 3. 碎片率计算：
 *    - 碎片率 = 1 - (peakUsage / totalAllocated)
 *    - 如果碎片率高，说明频繁分配/释放导致内存池效率低
 *    - 优化手段：内存池（Memory Pool）、区域分配器（Arena Allocator）
 *
 * 4. JSON 序列化中的转义：
 *    - 简单实现中未对 tag 做转义处理
 *    - 生产环境应使用 JSON 库，避免 tag 中含 " 或 \ 导致格式错误
 */

#include "profiler/MemoryTracker.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace edgeai {

// ═══════════════════════════════════════════════════════════
// 单例获取
// ═══════════════════════════════════════════════════════════

MemoryTracker& MemoryTracker::instance()
{
    // Meyer's Singleton：第一次调用时构造，之后返回同一实例
    // C++11 标准保证：如果多个线程同时进入，只有一个会执行构造，其余阻塞等待
    static MemoryTracker instance;
    return instance;
}

// ═══════════════════════════════════════════════════════════
// 分配/释放追踪
// ═══════════════════════════════════════════════════════════

void MemoryTracker::onAlloc(const std::string& tag, size_t bytes)
{
    // 更新累计统计
    current_        += bytes;
    totalAllocated_ += bytes;

    // 检测峰值：这是最关键的一步——
    // 峰值内存 = 推理过程中任何时刻的最大内存占用量
    // 它直接决定模型能否部署到目标设备
    if (current_ > peak_) {
        peak_ = current_;
    }

    // 记录到历史（用于后续分析内存时间线）
    history_.push_back(MemRecord{
        tag,
        bytes,
        totalAllocated_,
        true  // isAllocation
    });
}

void MemoryTracker::onFree(const std::string& tag, size_t bytes)
{
    // 安全检查：防止释放量超过当前使用量（说明有 bug）
    if (bytes > current_) {
        std::fprintf(stderr,
                     "[MemoryTracker] WARNING: Free %zu bytes exceeds current usage %zu bytes (tag: %s)\n",
                     bytes, current_, tag.c_str());
        current_ = 0;  // 防止变为负数（size_t 下溢出）
    } else {
        current_ -= bytes;
    }

    totalFreed_ += bytes;

    history_.push_back(MemRecord{
        tag,
        bytes,
        totalAllocated_,
        false  // isAllocation → 释放事件
    });
}

// ═══════════════════════════════════════════════════════════
// 报告与序列化
// ═══════════════════════════════════════════════════════════

void MemoryTracker::printReport() const
{
    std::printf("\n");
    std::printf("╔══════════════════════════════════════════╗\n");
    std::printf("║       Memory Tracker Report              ║\n");
    std::printf("╠══════════════════════════════════════════╣\n");

    // 当前内存使用
    std::printf("║ Current Usage    : %10zu bytes", current_);
    printSizeHuman(current_);
    std::printf("\n");

    // 峰值内存 —— 最关键指标
    std::printf("║ Peak Usage       : %10zu bytes", peak_);
    printSizeHuman(peak_);
    std::printf("\n");

    // 累计分配总量
    std::printf("║ Total Allocated  : %10zu bytes", totalAllocated_);
    printSizeHuman(totalAllocated_);
    std::printf("\n");

    // 累计释放总量
    std::printf("║ Total Freed      : %10zu bytes", totalFreed_);
    printSizeHuman(totalFreed_);
    std::printf("\n");

    // 内存碎片率 = 1 - (峰值 / 累计分配)
    // 高碎片率意味着频繁的小块分配/释放，内存池效率低
    double fragmentation = 0.0;
    if (totalAllocated_ > 0) {
        fragmentation = 1.0 - static_cast<double>(peak_) / static_cast<double>(totalAllocated_);
    }

    std::printf("║ Fragmentation    : %.1f%%\n", fragmentation * 100.0);

    // 活跃分配事件数
    size_t allocCount = 0;
    size_t freeCount  = 0;
    for (const auto& h : history_) {
        if (h.isAllocation) allocCount++;
        else                freeCount++;
    }

    std::printf("║ Alloc Events     : %zu\n", allocCount);
    std::printf("║ Free  Events     : %zu\n", freeCount);

    std::printf("╚══════════════════════════════════════════╝\n");

    // 碎片率警告
    if (fragmentation > 0.5) {
        std::printf("[MemoryTracker] WARNING: High fragmentation (%.1f%%). "
                    "Consider using a memory pool or arena allocator.\n",
                    fragmentation * 100.0);
    }

    // 泄漏检测：如果还有未释放的内存
    if (current_ > 0) {
        std::printf("[MemoryTracker] WARNING: %zu bytes still in use (possible leak or intentional).\n",
                    current_);
    }

    std::printf("\n");
}

void MemoryTracker::printSizeHuman(size_t bytes) const
{
    // 辅助函数：在字节数后面追加人类可读的单位
    if (bytes >= 1024ULL * 1024 * 1024) {
        std::printf(" (%.2f GB)", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    } else if (bytes >= 1024ULL * 1024) {
        std::printf(" (%.2f MB)", static_cast<double>(bytes) / (1024.0 * 1024.0));
    } else if (bytes >= 1024ULL) {
        std::printf(" (%.2f KB)", static_cast<double>(bytes) / 1024.0);
    }
}

std::string MemoryTracker::toJson() const
{
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"currentBytes\": "   << current_        << ",\n";
    oss << "  \"peakBytes\": "      << peak_           << ",\n";
    oss << "  \"totalAllocated\": "  << totalAllocated_ << ",\n";
    oss << "  \"totalFreed\": "      << totalFreed_     << ",\n";

    double fragmentation = 0.0;
    if (totalAllocated_ > 0) {
        fragmentation = 1.0 - static_cast<double>(peak_) / static_cast<double>(totalAllocated_);
    }
    oss << "  \"fragmentation\": " << std::fixed << std::setprecision(4) << fragmentation << ",\n";

    // 写入历史记录（可能很长，生产环境可考虑分页或截断）
    oss << "  \"history\": [\n";
    for (size_t i = 0; i < history_.size(); ++i) {
        const auto& h = history_[i];
        oss << "    {\"tag\":\""       << h.tag
            << "\", \"bytes\":"      << h.sizeBytes
            << ", \"total\":"        << h.totalAllocated
            << ", \"isAlloc\":"     << (h.isAllocation ? "true" : "false")
            << "}";
        if (i + 1 < history_.size()) oss << ",";
        oss << "\n";
    }
    oss << "  ]\n";
    oss << "}\n";
    return oss.str();
}

void MemoryTracker::reset()
{
    current_        = 0;
    peak_           = 0;
    totalAllocated_ = 0;
    totalFreed_     = 0;
    history_.clear();
}

} // namespace edgeai
