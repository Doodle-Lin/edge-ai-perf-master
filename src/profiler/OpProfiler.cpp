/**
 * @file OpProfiler.cpp
 * @brief 逐算子性能分析器的实现
 *
 * ====== 实现要点 ======
 * 1. 使用 std::chrono::steady_clock 作为高精度计时器
 *    - steady_clock 保证单调递增，不受系统时间调整影响
 *    - 替代方案：clock() 受 CPU 调度影响，gettimeofday() 受 NTP 调整
 *
 * 2. GFLOPS 计算公式的推导：
 *    GFLOPS = FLOPs / time(s) / 1e9
 *           = FLOPs / (timeMs * 1e-3) / 1e9
 *           = FLOPs / timeMs / 1e6
 *
 * 3. FLOPs 估算示例（Conv2D）：
 *    FLOPs = 2 × K × K × Cin × Cout × Hout × Wout
 *    其中系数 2 是因为 1 次 MAC（乘累加）= 1 次乘法 + 1 次加法 = 2 FLOPs
 *
 * 4. 内存访问量估算（Conv2D）：
 *    读输入:  Cin × Hin × Win × sizeof(float)
 *    读权重:  K × K × Cin × Cout × sizeof(float)
 *    写输出:  Cout × Hout × Wout × sizeof(float)
 */

#include "profiler/OpProfiler.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <numeric>

namespace edgeai {

// ═══════════════════════════════════════════════════════════
// OpProfiler
// ═══════════════════════════════════════════════════════════

void OpProfiler::beginOp(const std::string& name,
                         const std::string& type,
                         const std::string& device)
{
    // 如果上一个算子未调用 endOp，先自动结束（防止遗漏）
    if (inOp_) {
        endOp(0, 0);
    }

    currentOp_.name   = name;
    currentOp_.type   = type;
    currentOp_.device = device;
    currentOp_.timeMs = 0.0;
    currentOp_.flops  = 0;
    currentOp_.memBytes = 0;

    // 记录起始时间点
    // steady_clock::now() 在 x86 上通常使用 RDTSC 指令，开销约 20-40ns
    opStart_ = std::chrono::steady_clock::now();
    inOp_ = true;
}

void OpProfiler::endOp(int64_t flops, int64_t memBytes)
{
    if (!inOp_) {
        // 没有 beginOp 就调用 endOp，忽略
        return;
    }

    // 计算耗时：duration_cast 转换为毫秒精度
    auto opEnd = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(opEnd - opStart_);
    currentOp_.timeMs   = duration.count();
    currentOp_.flops    = flops;
    currentOp_.memBytes = memBytes;

    // 将完成的记录追加到列表
    records_.push_back(currentOp_);
    inOp_ = false;
}

void OpProfiler::clear()
{
    records_.clear();
    inOp_ = false;
}

// ═══════════════════════════════════════════════════════════
// 分析报告
// ═══════════════════════════════════════════════════════════

void OpProfiler::printReport() const
{
    if (records_.empty()) {
        std::printf("[OpProfiler] No operator records to report.\n");
        return;
    }

    double totalMs = totalTimeMs();

    // 表头：使用固定宽度对齐，方便阅读
    std::printf("\n");
    std::printf("╔══════════════════════════════════════════════════════════════════════════════════════╗\n");
    std::printf("║                        Operator Profiling Report                                   ║\n");
    std::printf("╠═══════════╦═════════════╦════════╦═══════╦══════════╦═══════════╦═══════════╣\n");
    std::printf("║ %-9s ║ %-11s ║ %6s ║ %5s%% ║ %8s ║ %9s ║ %9s ║\n",
               "Name", "Type", "ms", "Ratio", "GFLOPS", "GB/s", "AI");
    std::printf("╠═══════════╬═════════════╬════════╬═══════╬══════════╬═══════════╬═══════════╣\n");

    for (const auto& r : records_) {
        double ratio = (totalMs > 0.0) ? (r.timeMs / totalMs * 100.0) : 0.0;
        double gf    = r.gflops();
        double bw    = r.bandwidth();
        double ai    = r.arithmeticIntensity();

        std::printf("║ %-9s ║ %-11s ║ %6.2f ║ %5.1f%% ║ %8.2f ║ %9.2f ║ %9.2f ║\n",
                     r.name.c_str(),
                     r.type.c_str(),
                     r.timeMs,
                     ratio,
                     gf,
                     bw,
                     ai);
    }

    std::printf("╠═══════════╬═════════════╬════════╬═══════╬══════════╬═══════════╬═══════════╣\n");
    std::printf("║ %-9s ║ %-11s ║ %6.2f ║ %5.1f%% ║ %8s ║ %9s ║ %9s ║\n",
                 "TOTAL", "", totalMs, 100.0, "-", "-", "-");
    std::printf("╚═══════════╩═════════════╩════════╩═══════╩══════════╩═══════════╩═══════════╝\n");

    // 打印瓶颈分析
    OpRecord bottleneck = findBottleneck();
    double boundRatio   = computeBoundRatio();

    std::printf("\n[Bottleneck] %s (%s) : %.2f ms (%.1f%% of total)\n",
                bottleneck.name.c_str(),
                bottleneck.type.c_str(),
                bottleneck.timeMs,
                boundRatio * 100.0);

    // Roofline 判断
    double ai = bottleneck.arithmeticIntensity();
    if (ai > 10.0) {
        std::printf("[Roofline]  Arithmetic Intensity = %.1f → Compute-bound (计算瓶颈)\n", ai);
        std::printf("            → 优化方向：增加并行度、使用 TensorCore/NPU、降低精度(INT8/FP16)\n");
    } else if (ai > 0.0) {
        std::printf("[Roofline]  Arithmetic Intensity = %.1f → Memory-bound (内存瓶颈)\n", ai);
        std::printf("            → 优化方向：内存复用、缓存分块(tiling)、算子融合(fusion)\n");
    }

    std::printf("\n");
}

std::string OpProfiler::toJson() const
{
    // 手动拼接 JSON，避免引入第三方 JSON 库
    // 生产环境建议使用 nlohmann/json 或 rapidjson
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"totalTimeMs\": " << std::fixed << std::setprecision(3) << totalTimeMs() << ",\n";
    oss << "  \"records\": [\n";

    for (size_t i = 0; i < records_.size(); ++i) {
        const auto& r = records_[i];
        oss << "    {\n";
        oss << "      \"name\": \""       << r.name   << "\",\n";
        oss << "      \"type\": \""       << r.type   << "\",\n";
        oss << "      \"device\": \""     << r.device << "\",\n";
        oss << "      \"timeMs\": "       << std::fixed << std::setprecision(3) << r.timeMs  << ",\n";
        oss << "      \"flops\": "        << r.flops    << ",\n";
        oss << "      \"memBytes\": "     << r.memBytes << ",\n";
        oss << "      \"gflops\": "       << std::fixed << std::setprecision(2) << r.gflops() << ",\n";
        oss << "      \"bandwidthGBs\": " << std::fixed << std::setprecision(2) << r.bandwidth() << ",\n";
        oss << "      \"arithmeticIntensity\": " << std::fixed << std::setprecision(2) << r.arithmeticIntensity() << "\n";
        oss << "    }";
        if (i + 1 < records_.size()) oss << ",";
        oss << "\n";
    }

    oss << "  ]\n";
    oss << "}\n";
    return oss.str();
}

void OpProfiler::saveJson(const std::string& path) const
{
    // 写入文件（覆盖模式）
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::fprintf(stderr, "[OpProfiler] ERROR: Cannot open file for writing: %s\n", path.c_str());
        return;
    }
    ofs << toJson();
    ofs.close();
    std::printf("[OpProfiler] Report saved to: %s\n", path.c_str());
}

// ═══════════════════════════════════════════════════════════
// 汇总统计
// ═══════════════════════════════════════════════════════════

double OpProfiler::totalTimeMs() const
{
    double sum = 0.0;
    for (const auto& r : records_) {
        sum += r.timeMs;
    }
    return sum;
}

OpRecord OpProfiler::findBottleneck() const
{
    if (records_.empty()) {
        return OpRecord{};
    }

    // 使用 std::max_element 按耗时查找瓶颈算子
    auto it = std::max_element(records_.begin(), records_.end(),
        [](const OpRecord& a, const OpRecord& b) {
            return a.timeMs < b.timeMs;
        });
    return *it;
}

double OpProfiler::computeBoundRatio() const
{
    double total = totalTimeMs();
    if (total <= 0.0 || records_.empty()) {
        return 0.0;
    }
    return findBottleneck().timeMs / total;
}

} // namespace edgeai
