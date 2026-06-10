/**
 * @file Logger.cpp
 * @brief Logger 实现
 */

#include "common/Logger.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace edgeai {

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

void Logger::setLevel(LogLevel level) {
    level_ = level;
}

LogLevel Logger::level() const {
    return level_;
}

const char* Logger::levelToString(LogLevel level) const {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO ";
        case LogLevel::WARN:  return "WARN ";
        case LogLevel::ERROR: return "ERROR";
        default:              return "?????";
    }
}

void Logger::log(LogLevel level, const char* file, int line, const char* fmt, ...) {
    if (level < level_) return;

    // 提取文件名（去掉路径前缀）
    const char* basename = strrchr(file, '/');
    const char* basename2 = strrchr(file, '\\');
    if (basename2 && (!basename || basename2 > basename)) basename = basename2;
    if (basename) basename++; else basename = file;

    // 时间戳
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    struct tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_buf);
#endif

    // 打印头部
    fprintf(stderr, "[%02d:%02d:%02d.%03d][%s][%s:%d] ",
            tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec, static_cast<int>(ms.count()),
            levelToString(level), basename, line);

    // 打印消息
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fprintf(stderr, "\n");
    fflush(stderr);
}

} // namespace edgeai
