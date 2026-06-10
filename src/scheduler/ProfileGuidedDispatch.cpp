/**
 * @file ProfileGuidedDispatch.cpp
 * @brief 基于 Profiling 数据的调度策略实现 —— 用历史测量数据指导调度决策
 *
 * ====== 实现说明 ======
 *
 * 1. profile 表的构建
 *    updateFromProfiler() 遍历 OpProfiler 中的所有 OpRecord，
 *    按算子名分组，分别累计 CPU 和 GPU 的执行时间，取平均值。
 *    然后比较平均时间，选择更快的设备作为 bestDevice。
 *
 * 2. JSON 序列化/反序列化
 *    手写简易 JSON 解析器（不依赖第三方库）。
 *    只支持本模块定义的 JSON 格式，不做通用 JSON 解析。
 *    这样做的好处：零依赖、编译快、不需要链接 nlohmann/json 等。
 *
 * 3. 加速比阈值（gpuSpeedupThreshold_）
 *    即使 GPU 上更快，也要快到一定程度（默认 1.2x）才值得切换。
 *    原因：GPU 执行需要 CPU→GPU 的数据拷贝，这部分开销在 profiler
 *    中可能没有完全体现。1.2x 的裕量是工程实践中的经验值。
 */

#include "scheduler/ProfileGuidedDispatch.h"
#include "scheduler/HeuristicDispatch.h"

#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

namespace edgeai {

// ============================================================================
// 构造函数
// ============================================================================

ProfileGuidedDispatch::ProfileGuidedDispatch()
    : fallback_(std::make_shared<HeuristicDispatch>())  // 默认使用启发式策略作为回退
{
}

// ============================================================================
// 核心调度逻辑
// ============================================================================

DispatchDecision ProfileGuidedDispatch::dispatch(const OpNode& op, const OpGraph& graph) {
    DispatchDecision decision;
    decision.opId = op.id;

    // ── 在 profile 表中查找算子 ──
    auto it = profileTable_.find(op.name);
    if (it != profileTable_.end()) {
        const OpProfileEntry& entry = it->second;

        // 情况 1：同时有 CPU 和 GPU 数据，可以比较
        if (entry.cpuTimeMs > 0.0 && entry.gpuTimeMs > 0.0) {
            double speedup = entry.speedup();  // cpuTimeMs / gpuTimeMs

            if (speedup > gpuSpeedupThreshold_) {
                // GPU 显著更快（超过阈值），选择 GPU
                decision.device = Device::GPU;
                std::ostringstream reason;
                reason << std::fixed;
                reason.precision(2);
                reason << "profile-guided: GPU faster by "
                       << speedup << "x (cpu="
                       << entry.cpuTimeMs << "ms, gpu="
                       << entry.gpuTimeMs << "ms, samples="
                       << entry.sampleCount << ")";
                decision.reason = reason.str();
                return decision;
            } else {
                // CPU 更快或 GPU 加速不够显著，选择 CPU
                decision.device = Device::CPU;
                std::ostringstream reason;
                reason << std::fixed;
                reason.precision(2);
                reason << "profile-guided: CPU preferred (gpu speedup="
                       << speedup << "x < threshold="
                       << gpuSpeedupThreshold_ << "x, cpu="
                       << entry.cpuTimeMs << "ms, gpu="
                       << entry.gpuTimeMs << "ms, samples="
                       << entry.sampleCount << ")";
                decision.reason = reason.str();
                return decision;
            }
        }

        // 情况 2：只有 CPU 数据
        if (entry.cpuTimeMs > 0.0 && entry.gpuTimeMs <= 0.0) {
            decision.device = Device::CPU;
            std::ostringstream reason;
            reason << "profile-guided: only CPU data available ("
                   << entry.cpuTimeMs << "ms), using CPU";
            decision.reason = reason.str();
            return decision;
        }

        // 情况 3：只有 GPU 数据
        if (entry.gpuTimeMs > 0.0 && entry.cpuTimeMs <= 0.0) {
            decision.device = Device::GPU;
            std::ostringstream reason;
            reason << "profile-guided: only GPU data available ("
                   << entry.gpuTimeMs << "ms), using GPU";
            decision.reason = reason.str();
            return decision;
        }
    }

    // ── 未找到 profile 数据，回退到 fallback 策略 ──
    if (fallback_) {
        auto fallbackDecision = fallback_->dispatch(op, graph);
        // 在 reason 中标注这是回退决策
        fallbackDecision.reason = "[fallback] " + fallbackDecision.reason;
        return fallbackDecision;
    }

    // 兜底：没有任何策略可用，选择 CPU
    decision.device = Device::CPU;
    decision.reason = "no profile data and no fallback strategy, defaulting to CPU";
    return decision;
}

// ============================================================================
// 配置接口
// ============================================================================

void ProfileGuidedDispatch::setFallback(std::shared_ptr<DispatchStrategy> fallback) {
    fallback_ = std::move(fallback);
}

void ProfileGuidedDispatch::setGpuSpeedupThreshold(double threshold) {
    gpuSpeedupThreshold_ = threshold;
}

// ============================================================================
// 从 OpProfiler 更新 profile 数据
// ============================================================================

void ProfileGuidedDispatch::updateFromProfiler(const OpProfiler& profiler) {
    // 遍历所有记录，按算子名和设备分组累计
    // 使用中间映射：opName → {cpuTimeSum, gpuTimeSum, cpuCount, gpuCount}
    struct AccTime {
        double cpuTimeSum = 0.0;
        double gpuTimeSum = 0.0;
        int cpuCount = 0;
        int gpuCount = 0;
    };

    std::unordered_map<std::string, AccTime> acc;

    for (const auto& record : profiler.records()) {
        auto& a = acc[record.name];
        if (record.device == "CPU") {
            a.cpuTimeSum += record.timeMs;
            a.cpuCount++;
        } else if (record.device == "GPU") {
            a.gpuTimeSum += record.timeMs;
            a.gpuCount++;
        }
    }

    // 计算平均时间，更新 profile 表
    for (auto& [name, a] : acc) {
        OpProfileEntry entry;

        // 取平均值（多次运行取平均更稳定）
        entry.cpuTimeMs = (a.cpuCount > 0) ? (a.cpuTimeSum / a.cpuCount) : 0.0;
        entry.gpuTimeMs = (a.gpuCount > 0) ? (a.gpuTimeSum / a.gpuCount) : 0.0;
        entry.sampleCount = std::max(a.cpuCount, a.gpuCount);

        // 决定 bestDevice
        if (entry.cpuTimeMs > 0.0 && entry.gpuTimeMs > 0.0) {
            entry.bestDevice = (entry.speedup() > gpuSpeedupThreshold_)
                               ? Device::GPU : Device::CPU;
        } else if (entry.gpuTimeMs > 0.0) {
            entry.bestDevice = Device::GPU;
        } else {
            entry.bestDevice = Device::CPU;
        }

        profileTable_[name] = entry;
    }
}

// ============================================================================
// 手动添加 profile 条目
// ============================================================================

void ProfileGuidedDispatch::addProfileEntry(const std::string& opName,
                                              Device device,
                                              double timeMs) {
    auto it = profileTable_.find(opName);
    if (it == profileTable_.end()) {
        // 新条目
        OpProfileEntry entry;
        entry.sampleCount = 1;
        if (device == Device::CPU) {
            entry.cpuTimeMs = timeMs;
        } else {
            entry.gpuTimeMs = timeMs;
        }
        entry.bestDevice = device;
        profileTable_[opName] = entry;
    } else {
        // 已有条目，更新对应设备的时间
        OpProfileEntry& entry = it->second;
        if (device == Device::CPU) {
            // 指数移动平均（EMA）更新，新数据权重 0.5
            // EMA 比简单平均更能反映最近的性能变化
            entry.cpuTimeMs = (entry.cpuTimeMs > 0.0)
                              ? (entry.cpuTimeMs * 0.5 + timeMs * 0.5)
                              : timeMs;
        } else {
            entry.gpuTimeMs = (entry.gpuTimeMs > 0.0)
                              ? (entry.gpuTimeMs * 0.5 + timeMs * 0.5)
                              : timeMs;
        }
        entry.sampleCount++;

        // 重新评估 bestDevice
        if (entry.cpuTimeMs > 0.0 && entry.gpuTimeMs > 0.0) {
            entry.bestDevice = (entry.speedup() > gpuSpeedupThreshold_)
                               ? Device::GPU : Device::CPU;
        }
    }
}

// ============================================================================
// 数据访问
// ============================================================================

const OpProfileEntry* ProfileGuidedDispatch::getProfile(const std::string& opName) const {
    auto it = profileTable_.find(opName);
    return (it != profileTable_.end()) ? &(it->second) : nullptr;
}

const std::unordered_map<std::string, OpProfileEntry>&
ProfileGuidedDispatch::allProfiles() const {
    return profileTable_;
}

void ProfileGuidedDispatch::clearProfile() {
    profileTable_.clear();
}

// ============================================================================
// JSON 序列化
// ============================================================================

std::string ProfileGuidedDispatch::profileToJson() const {
    std::ostringstream oss;
    oss << std::fixed;
    oss.precision(3);

    oss << "{\n";
    oss << "  \"version\": 1,\n";

    // ── 算子 profile 数据 ──
    oss << "  \"op_profiles\": {\n";
    bool first = true;
    for (const auto& [name, entry] : profileTable_) {
        if (!first) oss << ",\n";
        first = false;

        oss << "    \"" << name << "\": {";
        oss << " \"cpu_ms\": " << entry.cpuTimeMs;
        oss << ", \"gpu_ms\": " << entry.gpuTimeMs;
        oss << ", \"best\": \"" << deviceToString(entry.bestDevice) << "\"";
        oss << ", \"samples\": " << entry.sampleCount;
        oss << " }";
    }
    oss << "\n  }\n";

    oss << "}\n";
    return oss.str();
}

bool ProfileGuidedDispatch::saveProfile(const std::string& path) const {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::cerr << "[ProfileGuidedDispatch] 无法打开文件写入: " << path << "\n";
        return false;
    }

    ofs << profileToJson();
    ofs.close();
    return true;
}

bool ProfileGuidedDispatch::loadProfile(const std::string& path) {
    // 读取整个文件到字符串
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "[ProfileGuidedDispatch] 无法打开文件读取: " << path << "\n";
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());
    ifs.close();

    if (content.empty()) {
        std::cerr << "[ProfileGuidedDispatch] 文件为空: " << path << "\n";
        return false;
    }

    // ── 简易 JSON 解析 ──
    // 只解析我们定义的格式，不做通用 JSON 解析
    // 格式示例：
    // { "version": 1, "op_profiles": { "conv1": { "cpu_ms": 2.3, "gpu_ms": 0.8, "best": "GPU", "samples": 5 } } }

    // 查找 "op_profiles" 键
    size_t pos = content.find("\"op_profiles\"");
    if (pos == std::string::npos) {
        std::cerr << "[ProfileGuidedDispatch] JSON 中未找到 'op_profiles' 键\n";
        return false;
    }

    // 找到 op_profiles 值的开始位置（跳过冒号和空白）
    pos = content.find('{', pos + 13);
    if (pos == std::string::npos) {
        std::cerr << "[ProfileGuidedDispatch] op_profiles 值格式错误\n";
        return false;
    }

    // 逐个解析 op 条目
    // 寻找 "name": { ... } 模式
    while (true) {
        // 查找下一个字符串键（算子名）
        size_t nameStart = content.find('"', pos + 1);
        if (nameStart == std::string::npos) break;

        size_t nameEnd = content.find('"', nameStart + 1);
        if (nameEnd == std::string::npos) break;
        std::string opName = content.substr(nameStart + 1, nameEnd - nameStart - 1);

        // 查找值对象 {
        size_t objStart = content.find('{', nameEnd);
        if (objStart == std::string::npos) break;

        // 找到匹配的 }
        size_t objEnd = content.find('}', objStart);
        if (objEnd == std::string::npos) break;
        std::string objContent = content.substr(objStart, objEnd - objStart + 1);

        // 解析值对象中的字段
        OpProfileEntry entry;

        // 解析 cpu_ms
        size_t cpuPos = objContent.find("\"cpu_ms\"");
        if (cpuPos != std::string::npos) {
            size_t colonPos = objContent.find(':', cpuPos);
            if (colonPos != std::string::npos) {
                // 使用 strtod/strtol 替代 std::stod/stoi，
                // 后者在某些平台上可能抛出异常且需要 <cstdlib>
                const char* start = objContent.c_str() + colonPos + 1;
                char* end = nullptr;
                entry.cpuTimeMs = std::strtod(start, &end);
            }
        }

        // 解析 gpu_ms
        size_t gpuPos = objContent.find("\"gpu_ms\"");
        if (gpuPos != std::string::npos) {
            size_t colonPos = objContent.find(':', gpuPos);
            if (colonPos != std::string::npos) {
                const char* start2 = objContent.c_str() + colonPos + 1;
                char* end2 = nullptr;
                entry.gpuTimeMs = std::strtod(start2, &end2);
            }
        }

        // 解析 best
        size_t bestPos = objContent.find("\"best\"");
        if (bestPos != std::string::npos) {
            size_t valStart = objContent.find('"', bestPos + 6);
            size_t valEnd   = objContent.find('"', valStart + 1);
            if (valStart != std::string::npos && valEnd != std::string::npos) {
                std::string bestStr = objContent.substr(valStart + 1, valEnd - valStart - 1);
                entry.bestDevice = (bestStr == "GPU") ? Device::GPU : Device::CPU;
            }
        }

        // 解析 samples
        size_t samplesPos = objContent.find("\"samples\"");
        if (samplesPos != std::string::npos) {
            size_t colonPos = objContent.find(':', samplesPos);
            if (colonPos != std::string::npos) {
                const char* start3 = objContent.c_str() + colonPos + 1;
                char* end3 = nullptr;
                entry.sampleCount = static_cast<int>(std::strtol(start3, &end3, 10));
            }
        }

        profileTable_[opName] = entry;

        // 继续找下一个条目
        pos = objEnd;
    }

    return true;
}

} // namespace edgeai
