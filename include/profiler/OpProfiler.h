/**
 * @file OpProfiler.h
 * @brief 逐算子性能分析器 —— 测量每个算子的执行时间、计算量与内存带宽
 *
 * ====== 学习要点 ======
 * 1. FLOPs vs GFLOPS：FLOPs 是"浮点运算次数"（绝对量），GFLOPS 是"每秒十亿次浮点运算"（速率）
 *    GFLOPS = FLOPs / (timeMs * 1e-3) / 1e9 = FLOPs / timeMs / 1e6
 *
 * 2. 计算密度 (Arithmetic Intensity) = FLOPs / Bytes
 *    - 值越大 → 计算密集型 (compute-bound)，应优化计算并行度
 *    - 值越小 → 内存密集型 (memory-bound)，应优化数据搬运与缓存
 *
 * 3. Roofline 模型：将算子落在 "屋顶线" 图上
 *    - 横轴 = 计算密度，纵轴 = GFLOPS
 *    - 屋顶左侧是带宽瓶颈区（斜线），右侧是算力瓶颈区（水平线）
 *
 * 4. RAII 式计时：beginOp/endOp 配对使用，也可以用 OpScopeGuard 在作用域自动结束
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace edgeai {

/// 单个算子的性能记录
struct OpRecord {
    std::string name;        ///< 算子名称，如 "conv1", "relu_2"
    std::string type;        ///< 算子类型，如 "Convolution", "Pooling", "ReLU"
    double timeMs   = 0.0;   ///< 执行耗时（毫秒）
    int64_t flops   = 0;     ///< 计算量（FLOPs），1 次 MAC = 2 FLOPs
    int64_t memBytes = 0;    ///< 内存访问量（字节），含输入+输出+权重
    std::string device;      ///< 执行设备: "CPU" / "GPU"

    /// 计算吞吐率 GFLOPS = FLOPs / 耗时(s) / 1e9
    ///   例如：1 GFLOP 在 10ms 内完成 → 100 GFLOPS
    double gflops() const {
        if (timeMs <= 0.0) return 0.0;
        return static_cast<double>(flops) / (timeMs * 1e-3) / 1e9;
    }

    /// 有效内存带宽 GB/s = Bytes / 耗时(s) / 1e9
    ///   例如：读 100MB 数据花 1ms → 带宽 ≈ 100 GB/s
    double bandwidth() const {
        if (timeMs <= 0.0) return 0.0;
        return static_cast<double>(memBytes) / (timeMs * 1e-3) / 1e9;
    }

    /// 计算密度 (Arithmetic Intensity) = FLOPs / Bytes
    ///   > 10 → compute-bound;  < 5 → memory-bound;  中间为过渡区
    double arithmeticIntensity() const {
        if (memBytes == 0) return 0.0;
        return static_cast<double>(flops) / static_cast<double>(memBytes);
    }
};

/**
 * @class OpProfiler
 * @brief 逐算子性能分析器
 *
 * 使用方法：
 * @code
 *   OpProfiler profiler;
 *   profiler.beginOp("conv1", "Convolution", "CPU");
 *   // ... 执行 conv1 算子 ...
 *   profiler.endOp(flops, memBytes);  // 自动计算耗时
 *
 *   profiler.printReport();   // 打印逐层性能表
 *   profiler.saveJson("op_profile.json");
 * @endcode
 */
class OpProfiler {
public:
    /// 开始计时一个新算子
    /// @param name   算子实例名，如 "conv1"
    /// @param type   算子类型，如 "Convolution"
    /// @param device 执行设备，"CPU" 或 "GPU"
    void beginOp(const std::string& name,
                 const std::string& type,
                 const std::string& device);

    /// 结束当前算子计时，并记录 FLOPs 和内存访问量
    /// @param flops    本次算子的浮点运算次数，0 表示未统计
    /// @param memBytes 本次算子的内存访问字节数，0 表示未统计
    void endOp(int64_t flops = 0, int64_t memBytes = 0);

    /// 获取所有已记录的算子记录（只读）
    const std::vector<OpRecord>& records() const { return records_; }

    /// 清空所有记录
    void clear();

    // ──────────────── 分析报告 ────────────────

    /// 打印逐层性能表到 stdout
    /// 包含：名称、类型、耗时、占比、GFLOPS、带宽、计算密度
    void printReport() const;

    /// 将所有记录序列化为 JSON 字符串
    /// 格式：{ "records": [ { "name":..., ... }, ... ], "totalMs": ... }
    std::string toJson() const;

    /// 将 JSON 报告写入文件
    /// @param path 输出文件路径
    void saveJson(const std::string& path) const;

    // ──────────────── 汇总统计 ────────────────

    /// 所有算子总耗时（毫秒）
    double totalTimeMs() const;

    /// 找到最耗时的算子（瓶颈算子）
    /// @return 耗时最长的 OpRecord
    OpRecord findBottleneck() const;

    /// 计算瓶颈占比 = 瓶颈算子耗时 / 总耗时
    ///   如果 > 0.5，说明模型严重被单个算子拖慢，应优先优化它
    double computeBoundRatio() const;

private:
    std::vector<OpRecord> records_;                          ///< 已完成的算子记录
    std::chrono::steady_clock::time_point opStart_;          ///< 当前算子起始时间点
    OpRecord currentOp_;                                     ///< 正在计时的算子信息
    bool inOp_ = false;                                      ///< 是否正在计时
};

/**
 * @class OpScopeGuard
 * @brief RAII 风格的算子计时守卫，在作用域退出时自动调用 endOp
 *
 * 使用方法：
 * @code
 *   {
 *       OpScopeGuard guard(profiler, "conv1", "Convolution", "CPU", flops, memBytes);
 *       // ... 执行算子 ...
 *   }   // ← 此处自动 endOp
 * @endcode
 */
class OpScopeGuard {
public:
    OpScopeGuard(OpProfiler& profiler,
                 const std::string& name,
                 const std::string& type,
                 const std::string& device,
                 int64_t flops = 0,
                 int64_t memBytes = 0)
        : profiler_(profiler), flops_(flops), memBytes_(memBytes)
    {
        profiler_.beginOp(name, type, device);
    }

    ~OpScopeGuard() {
        profiler_.endOp(flops_, memBytes_);
    }

    // 禁止拷贝和移动 —— 守卫语义不应被转移
    OpScopeGuard(const OpScopeGuard&) = delete;
    OpScopeGuard& operator=(const OpScopeGuard&) = delete;

private:
    OpProfiler& profiler_;
    int64_t flops_;
    int64_t memBytes_;
};

} // namespace edgeai
