/**
 * @file Logger.h
 * @brief 简易日志工具 —— 项目统一日志输出
 *
 * ====== 学习要点 ======
 * 1. 日志级别: DEBUG < INFO < WARN < ERROR
 *    - 开发阶段用 DEBUG 追踪程序流程
 *    - 正式运行时设为 INFO 或 WARN，减少输出开销
 *
 * 2. 性能敏感代码中的日志:
 *    - 日志 I/O 是系统调用，高频调用会拖慢计时
 *    - Profiling 时应将日志级别设为 WARN 以上
 *    - 或者用条件编译完全去掉日志: -DDISABLE_LOGGING
 *
 * 3. 格式化: [时间戳][级别][文件:行号] 消息
 *    - 文件:行号 帮助快速定位问题
 *    - 时间戳帮助关联日志顺序
 */

#pragma once

#include <chrono>
#include <cstdio>
#include <string>

namespace edgeai {

enum class LogLevel {
    DEBUG = 0,
    INFO  = 1,
    WARN  = 2,
    ERROR = 3
};

class Logger {
public:
    /// 获取单例
    static Logger& instance();

    /// 设置全局日志级别
    void setLevel(LogLevel level);

    /// 获取当前日志级别
    LogLevel level() const;

    /// 输出日志
    void log(LogLevel level, const char* file, int line, const char* fmt, ...);

private:
    Logger() = default;
    LogLevel level_ = LogLevel::INFO;

    const char* levelToString(LogLevel level) const;
};

} // namespace edgeai

// ====== 便捷宏 ======
// 使用 __builtin_expect 提示分支预测，热路径中日志判断几乎零开销
#define EDGEAI_LOG(level, fmt, ...)                                            \
    do {                                                                        \
        if (::edgeai::Logger::instance().level() <= ::edgeai::LogLevel::level) \
            ::edgeai::Logger::instance().log(                                   \
                ::edgeai::LogLevel::level, __FILE__, __LINE__, fmt, ##__VA_ARGS__); \
    } while (0)

#define LOG_DEBUG(fmt, ...) EDGEAI_LOG(DEBUG, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  EDGEAI_LOG(INFO,  fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  EDGEAI_LOG(WARN,  fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) EDGEAI_LOG(ERROR, fmt, ##__VA_ARGS__)
