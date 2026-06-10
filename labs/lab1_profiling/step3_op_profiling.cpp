/**
 * @file step3_op_profiling.cpp
 * @brief Lab1 Step3 —— 逐算子性能分析实验：OpProfiler 的使用与瓶颈定位
 *
 * ====== 本实验学习要点 ======
 * 1. 逐算子分析(Op-level Profiling):
 *    - 推理引擎由多个算子串联组成（Conv → BN → ReLU → Pooling → ...）
 *    - 不同算子的耗时差异巨大，找出最慢的算子（瓶颈）是优化的第一步
 *    - OpProfiler 通过 beginOp/endOp 记录每个算子的耗时、FLOPs 和内存访问量
 *
 * 2. 关键性能指标:
 *    - GFLOPS = FLOPs / 时间: 反映计算吞吐率
 *    - 带宽 (GB/s) = 内存访问量 / 时间: 反映数据搬运效率
 *    - 计算密度 = FLOPs / 内存字节数: 区分计算密集 vs 内存密集
 *      - > 10: 计算密集 (compute-bound)，优化方向: SIMD、多线程
 *      - < 5:  内存密集 (memory-bound)，优化方向: 缓存优化、数据布局
 *      - 5~10: 过渡区
 *
 * 3. Roofline 模型:
 *    - 横轴 = 计算密度，纵轴 = GFLOPS
 *    - 屋顶线分为两个区域:
 *      - 左侧斜坡区: 受内存带宽限制，GFLOPS 随计算密度线性增长
 *      - 右侧平台区: 受计算峰值限制，GFLOPS 达到上限
 *    - 每个算子是一个点，落在哪里决定优化方向
 *
 * 4. 瓶颈占比 (Compute Bound Ratio):
 *    - = 最慢算子耗时 / 总耗时
 *    - > 0.5: 严重瓶颈，优先优化该算子
 *    - < 0.2: 各算子负载均匀，优化任一个收益有限
 *
 * 5. NCNN 模型分析:
 *    - 如果有 NCNN 模型，使用 NcnnModelRunner::forwardWithProfile() 逐层计时
 *    - 如果没有模型，用模拟数据演示完整的工作流程
 *
 * 编译: g++ -std=c++17 -O2 -I../../include step3_op_profiling.cpp \
 *            ../../src/profiler/OpProfiler.cpp -o step3_op_profiling
 */

#include <algorithm>    // std::sort
#include <chrono>       // std::chrono::steady_clock
#include <cmath>        // std::sqrt
#include <cstdio>       // printf
#include <cstdlib>      // rand, srand
#include <fstream>      // std::ofstream
#include <string>       // std::string
#include <vector>

// 使用项目中的 OpProfiler 做逐算子分析
#include "profiler/OpProfiler.h"
// 使用项目中的 Tensor 类
#include "common/Tensor.h"
// 使用 MathUtils 中的 FLOPs/内存计算函数
#include "common/MathUtils.h"

// ============================================================================
// 模拟算子: 用于在没有 NCNN 模型时演示 profiling 工作流
// ============================================================================

/**
 * 模拟卷积算子: 用朴素三重循环实现
 * 这模拟了推理引擎中最常见的算子类型
 *
 * @param input  输入张量 [1, inC, H, W]
 * @param weight 权重张量 [outC, inC, kH, kW]
 * @param output 输出张量 [1, outC, outH, outW]
 * @param stride 步长
 * @param pad    填充
 */
void simulateConv2d(const edgeai::Tensor& input,
                   const edgeai::Tensor& weight,
                   edgeai::Tensor& output,
                   int stride = 1, int pad = 0) {
    // 从张量形状推断维度
    int inC  = input.size(1);
    int H    = input.size(2);
    int W    = input.size(3);
    int outC = weight.size(0);
    int kH   = weight.size(2);
    int kW   = weight.size(3);

    // 计算输出尺寸
    int outH = (H + 2 * pad - kH) / stride + 1;
    int outW = (W + 2 * pad - kW) / stride + 1;

    // 朴素卷积: 6 层循环（省略 batch 维度，假设 batch=1）
    // 输出 out[0][oc][oh][ow] = sum over ic,kh,kw:
    //     input[0][ic][oh*s-kh+pad][ow*s-kw+pad] * weight[oc][ic][kh][kw]
    for (int oc = 0; oc < outC; ++oc) {
        for (int oh = 0; oh < outH; ++oh) {
            for (int ow = 0; ow < outW; ++ow) {
                float sum = 0.0f;
                for (int ic = 0; ic < inC; ++ic) {
                    for (int kh = 0; kh < kH; ++kh) {
                        for (int kw = 0; kw < kW; ++kw) {
                            int ih = oh * stride - pad + kh;
                            int iw = ow * stride - pad + kw;
                            // 边界检查: padding 区域视为 0
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                sum += input.at(0, ic, ih, iw) * weight.at(oc, ic, kh, kw);
                            }
                        }
                    }
                }
                output.at(0, oc, oh, ow) = sum;
            }
        }
    }
}

/**
 * 模拟 ReLU 算子: 逐元素 max(0, x)
 * 这是一个典型的内存密集型算子: 几乎不计算，但需要遍历所有数据
 *
 * @param input  输入张量
 * @param output 输出张量
 */
void simulateReLU(const edgeai::Tensor& input, edgeai::Tensor& output) {
    for (int i = 0; i < input.total(); ++i) {
        float val = input[i];
        output[i] = val > 0.0f ? val : 0.0f;
    }
}

/**
 * 模拟全局平均池化 (Global Average Pooling):
 * 对每个通道在整个空间维度上求平均
 * 典型的内存密集型算子: 计算量很小，但要读完整张特征图
 *
 * @param input  输入张量 [1, C, H, W]
 * @param output 输出张量 [1, C, 1, 1]
 */
void simulateGlobalAvgPool(const edgeai::Tensor& input, edgeai::Tensor& output) {
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    for (int c = 0; c < C; ++c) {
        float sum = 0.0f;
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                sum += input.at(0, c, h, w);
            }
        }
        output.at(0, c, 0, 0) = sum / static_cast<float>(H * W);
    }
}

/**
 * 模拟全连接层 (Fully Connected / InnerProduct):
 * output = weight * input + bias
 * 权重大、计算量大的算子，通常在模型末尾
 *
 * @param input  输入向量 [1, 1, 1, inDim]
 * @param weight 权重矩阵 [1, 1, outDim, inDim]
 * @param output 输出向量 [1, 1, 1, outDim]
 */
void simulateFC(const edgeai::Tensor& input,
                const edgeai::Tensor& weight,
                edgeai::Tensor& output) {
    int inDim  = input.size(3);
    int outDim = weight.size(2);  // weight 形状 [1, 1, outDim, inDim]

    for (int o = 0; o < outDim; ++o) {
        float sum = 0.0f;
        for (int i = 0; i < inDim; ++i) {
            sum += input[0 * inDim + 0 * 0 + 0 * 0 + i] * weight.at(0, 0, o, i);
        }
        output[0 * outDim + 0 * 0 + 0 * 0 + o] = sum;
    }
}

// ============================================================================
// 辅助函数: 格式化字节数
// ============================================================================

const char* formatBytes(int64_t bytes) {
    static char buf[64];
    if (bytes < 1024) {
        snprintf(buf, sizeof(buf), "%lld B", (long long)bytes);
    } else if (bytes < 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%.1f KB", static_cast<double>(bytes) / 1024.0);
    } else if (bytes < 1024LL * 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%.1f MB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    } else {
        snprintf(buf, sizeof(buf), "%.2f GB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    }
    return buf;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    srand(42);
    printf("=== 逐算子性能分析实验 ===\n\n");

    // ========================================================================
    // 实验 1: 使用 OpProfiler 分析模拟推理流水线
    // ========================================================================
    printf("1. 模拟 MobileNet 风格推理流水线:\n");
    printf("   原理: 推理引擎按层顺序执行，OpProfiler 记录每层的耗时和计算量。\n");
    printf("   通过分析各层的 GFLOPS、带宽和计算密度，可以:\n");
    printf("   - 找出瓶颈算子（最慢的层）\n");
    printf("   - 判断算子是计算密集还是内存密集\n");
    printf("   - 决定优化方向（SIMD/多线程 vs 缓存优化/数据布局）\n\n");

    // 创建 OpProfiler 实例
    edgeai::OpProfiler profiler;

    // ---- 定义模拟 MobileNet 的各层参数 ----
    // 为了在合理时间内完成，使用较小的通道数

    // 层 1: Conv1 - 3→16, 3x3, stride=2, 输入 224x224 → 输出 112x112
    int conv1_inC = 3, conv1_outC = 16, conv1_kH = 3, conv1_kW = 3;
    int conv1_outH = 112, conv1_outW = 112;
    int64_t conv1_flops = edgeai::flopCountConv2d(conv1_inC, conv1_outC, conv1_kH, conv1_kW,
                                                   conv1_outH, conv1_outW);
    int64_t conv1_mem = edgeai::memSizeConv2d(conv1_inC, conv1_outC, conv1_kH, conv1_kW,
                                               224, 224);

    // 层 2: ReLU1
    int64_t relu1_flops = conv1_outC * conv1_outH * conv1_outW;  // 逐元素，1 FLOP/元素
    int64_t relu1_mem = conv1_outC * conv1_outH * conv1_outW * sizeof(float) * 2;  // 读+写

    // 层 3: Conv2 - 16→32, 3x3, stride=1, 输入 112x112 → 输出 112x112
    int conv2_inC = 16, conv2_outC = 32, conv2_kH = 3, conv2_kW = 3;
    int conv2_outH = 112, conv2_outW = 112;
    int64_t conv2_flops = edgeai::flopCountConv2d(conv2_inC, conv2_outC, conv2_kH, conv2_kW,
                                                   conv2_outH, conv2_outW);
    int64_t conv2_mem = edgeai::memSizeConv2d(conv2_inC, conv2_outC, conv2_kH, conv2_kW,
                                               112, 112);

    // 层 4: ReLU2
    int64_t relu2_flops = conv2_outC * conv2_outH * conv2_outW;
    int64_t relu2_mem = conv2_outC * conv2_outH * conv2_outW * sizeof(float) * 2;

    // 层 5: Conv3 - 32→64, 3x3, stride=2, 输入 112x112 → 输出 56x56
    int conv3_inC = 32, conv3_outC = 64, conv3_kH = 3, conv3_kW = 3;
    int conv3_outH = 56, conv3_outW = 56;
    int64_t conv3_flops = edgeai::flopCountConv2d(conv3_inC, conv3_outC, conv3_kH, conv3_kW,
                                                   conv3_outH, conv3_outW);
    int64_t conv3_mem = edgeai::memSizeConv2d(conv3_inC, conv3_outC, conv3_kH, conv3_kW,
                                               112, 112);

    // 层 6: ReLU3
    int64_t relu3_flops = conv3_outC * conv3_outH * conv3_outW;
    int64_t relu3_mem = conv3_outC * conv3_outH * conv3_outW * sizeof(float) * 2;

    // 层 7: Global Average Pooling
    int gap_C = conv3_outC;
    int64_t gap_flops = gap_C * 56 * 56;  // 每个通道求和
    int64_t gap_mem = gap_C * 56 * 56 * sizeof(float) + gap_C * sizeof(float);

    // 层 8: FC - 64→10 (10 类分类)
    int fc_inDim = gap_C, fc_outDim = 10;
    int64_t fc_flops = 2LL * fc_inDim * fc_outDim;
    int64_t fc_mem = (fc_inDim + fc_outDim) * sizeof(float) + fc_inDim * fc_outDim * sizeof(float);

    // ---- 分配张量并运行推理 ----

    // 输入
    edgeai::Tensor input({1, 3, 224, 224});
    input.randomize();

    // Conv1 权重和输出
    edgeai::Tensor conv1Weight({conv1_outC, conv1_inC, conv1_kH, conv1_kW});
    conv1Weight.randomize();
    edgeai::Tensor conv1Output({1, conv1_outC, conv1_outH, conv1_outW});

    // 运行 Conv1（带 profiling）
    profiler.beginOp("conv1", "Convolution", "CPU");
    simulateConv2d(input, conv1Weight, conv1Output, /*stride=*/2, /*pad=*/1);
    profiler.endOp(conv1_flops, conv1_mem);

    // ReLU1
    edgeai::Tensor relu1Output({1, conv1_outC, conv1_outH, conv1_outW});
    profiler.beginOp("relu1", "ReLU", "CPU");
    simulateReLU(conv1Output, relu1Output);
    profiler.endOp(relu1_flops, relu1_mem);

    // Conv2
    edgeai::Tensor conv2Weight({conv2_outC, conv2_inC, conv2_kH, conv2_kW});
    conv2Weight.randomize();
    edgeai::Tensor conv2Output({1, conv2_outC, conv2_outH, conv2_outW});
    profiler.beginOp("conv2", "Convolution", "CPU");
    simulateConv2d(relu1Output, conv2Weight, conv2Output, /*stride=*/1, /*pad=*/1);
    profiler.endOp(conv2_flops, conv2_mem);

    // ReLU2
    edgeai::Tensor relu2Output({1, conv2_outC, conv2_outH, conv2_outW});
    profiler.beginOp("relu2", "ReLU", "CPU");
    simulateReLU(conv2Output, relu2Output);
    profiler.endOp(relu2_flops, relu2_mem);

    // Conv3
    edgeai::Tensor conv3Weight({conv3_outC, conv3_inC, conv3_kH, conv3_kW});
    conv3Weight.randomize();
    edgeai::Tensor conv3Output({1, conv3_outC, conv3_outH, conv3_outW});
    profiler.beginOp("conv3", "Convolution", "CPU");
    simulateConv2d(relu2Output, conv3Weight, conv3Output, /*stride=*/2, /*pad=*/1);
    profiler.endOp(conv3_flops, conv3_mem);

    // ReLU3
    edgeai::Tensor relu3Output({1, conv3_outC, conv3_outH, conv3_outW});
    profiler.beginOp("relu3", "ReLU", "CPU");
    simulateReLU(conv3Output, relu3Output);
    profiler.endOp(relu3_flops, relu3_mem);

    // Global Average Pooling
    edgeai::Tensor gapOutput({1, gap_C, 1, 1});
    profiler.beginOp("global_avg_pool", "Pooling", "CPU");
    simulateGlobalAvgPool(relu3Output, gapOutput);
    profiler.endOp(gap_flops, gap_mem);

    // FC
    edgeai::Tensor fcWeight({1, 1, fc_outDim, fc_inDim});
    fcWeight.randomize();
    edgeai::Tensor fcOutput({1, 1, 1, fc_outDim});
    profiler.beginOp("fc", "InnerProduct", "CPU");
    simulateFC(gapOutput, fcWeight, fcOutput);
    profiler.endOp(fc_flops, fc_mem);

    // ---- 打印分析报告 ----
    printf("   ---- 逐层性能报告 ----\n\n");

    // 手动打印详细报告（使用 OpProfiler 的 records）
    const auto& records = profiler.records();
    double totalTime = profiler.totalTimeMs();

    // 打印表头
    printf("   %-18s %-14s %10s %7s %10s %9s %10s\n",
           "算子", "类型", "耗时(ms)", "占比%", "GFLOPS", "带宽GB/s", "计算密度");
    printf("   %s\n", std::string(90, '-').c_str());

    for (const auto& rec : records) {
        double pct = (totalTime > 0) ? (rec.timeMs / totalTime * 100.0) : 0.0;
        printf("   %-18s %-14s %10.3f %6.1f%% %10.3f %9.2f %10.2f\n",
               rec.name.c_str(),
               rec.type.c_str(),
               rec.timeMs,
               pct,
               rec.gflops(),
               rec.bandwidth(),
               rec.arithmeticIntensity());
    }

    printf("   %s\n", std::string(90, '-').c_str());
    printf("   %-18s %-14s %10.3f\n\n", "总计", "", totalTime);

    // ========================================================================
    // 实验 2: 找出瓶颈算子
    // ========================================================================
    printf("2. 瓶颈分析:\n\n");

    edgeai::OpRecord bottleneck = profiler.findBottleneck();
    double boundRatio = profiler.computeBoundRatio();

    printf("   最慢算子: %s (%s)\n", bottleneck.name.c_str(), bottleneck.type.c_str());
    printf("   耗时: %.3f ms (占总耗时 %.1f%%)\n",
           bottleneck.timeMs, boundRatio * 100.0);
    printf("   GFLOPS: %.3f\n", bottleneck.gflops());
    printf("   计算密度: %.2f\n", bottleneck.arithmeticIntensity());

    if (boundRatio > 0.5) {
        printf("   --> 严重瓶颈! 优化此算子将显著提升整体性能。\n");
    } else if (boundRatio > 0.3) {
        printf("   --> 明显瓶颈，建议优先优化此算子。\n");
    } else {
        printf("   --> 算子间负载较均匀，优化任一算子收益有限。\n");
    }

    // 计算密度分析
    printf("\n   算子分类 (按计算密度):\n");
    for (const auto& rec : records) {
        double ai = rec.arithmeticIntensity();
        const char* category = "";
        if (ai > 10.0) {
            category = "计算密集 (compute-bound) --> 优化: SIMD, 多线程";
        } else if (ai < 5.0) {
            category = "内存密集 (memory-bound) --> 优化: 缓存, 数据布局";
        } else {
            category = "过渡区 --> 两者皆需优化";
        }
        printf("   %-18s 计算密度=%6.2f  %s\n", rec.name.c_str(), ai, category);
    }

    // ========================================================================
    // 实验 3: 使用 OpScopeGuard 的 RAII 风格计时
    // ========================================================================
    printf("\n3. RAII 风格计时 (OpScopeGuard):\n");
    printf("   原理: C++ RAII (Resource Acquisition Is Initialization) 惯用法:\n");
    printf("   - 在构造函数中 beginOp()，在析构函数中 endOp()\n");
    printf("   - 即使函数中途 return 或抛出异常，也能正确结束计时\n");
    printf("   - 避免忘记调用 endOp() 导致的数据错误\n\n");

    // 使用 OpScopeGuard 的示例
    {
        // 在这个作用域内，OpScopeGuard 自动管理计时
        // 构造时调用 beginOp()，作用域退出时析构调用 endOp()
        edgeai::OpScopeGuard guard(profiler, "extra_conv", "Convolution", "CPU",
                                    conv2_flops, conv2_mem);

        // 执行算子
        edgeai::Tensor extraOutput({1, conv2_outC, conv2_outH, conv2_outW});
        simulateConv2d(relu1Output, conv2Weight, extraOutput, /*stride=*/1, /*pad=*/1);

        // 作用域退出时，guard 析构自动调用 endOp()
        printf("   OpScopeGuard 将在此作用域退出时自动 endOp()\n");
    }

    printf("   RAII 计时完成，自动记录了 'extra_conv' 算子。\n\n");

    // ========================================================================
    // 实验 4: 保存 profiling 数据为 JSON（供 Python 可视化）
    // ========================================================================
    printf("4. 保存 profiling 数据为 JSON:\n");
    printf("   原理: 性能数据导出为 JSON 格式，方便用 Python 绘制 Roofline 图等。\n\n");

    // 使用 OpProfiler 的 saveJson 方法
    std::string jsonPath = "lab1_op_profile.json";
    profiler.saveJson(jsonPath);
    printf("   已保存到: %s\n", jsonPath.c_str());

    // 也打印 JSON 内容到控制台（方便查看格式）
    printf("   JSON 格式预览:\n");
    std::string jsonStr = profiler.toJson();
    // 只打印前 500 字符
    size_t previewLen = std::min(jsonStr.size(), size_t(500));
    printf("   %.*s%s\n\n", (int)previewLen, jsonStr.c_str(),
           jsonStr.size() > 500 ? "..." : "");

    // ========================================================================
    // 实验 5: NCNN 模型分析（如果有模型文件的话）
    // ========================================================================
    printf("5. NCNN 模型分析 (可选):\n");
    printf("   如果有 NCNN 模型文件 (.param + .bin)，可以使用 NcnnModelRunner:\n\n");
    printf("   // 加载模型\n");
    printf("   NcnnModelRunner runner;\n");
    printf("   runner.loadModel(\"mobilenet.param\", \"mobilenet.bin\");\n\n");
    printf("   // 带逐层 profiling 的推理\n");
    printf("   OpProfiler ncnnProfiler;\n");
    printf("   auto result = runner.forwardWithProfile(\"output\", ncnnProfiler);\n\n");
    printf("   // 打印报告\n");
    printf("   ncnnProfiler.printReport();\n\n");

    // 尝试查找模型文件
    const char* modelPaths[] = {
        "../../models/mobilenet.param",
        "../../models/squeezenet.param",
        "./mobilenet.param",
    };
    bool modelFound = false;
    for (const char* path : modelPaths) {
        std::ifstream testFile(path);
        if (testFile.good()) {
            printf("   发现模型文件: %s\n", path);
            modelFound = true;
            break;
        }
    }
    if (!modelFound) {
        printf("   未找到 NCNN 模型文件，使用上面的模拟数据演示了完整工作流。\n");
        printf("   获取模型的方法:\n");
        printf("   1. 从 NCNN 官方仓库下载: github.com/nihui/ncnn/tree/master/models\n");
        printf("   2. 使用 onnx2ncnn 工具从 ONNX 模型转换\n");
        printf("   3. 使用本项目 scripts/ 目录下的转换脚本\n\n");
    }

    // ========================================================================
    // 总结
    // ========================================================================
    printf("=== 实验完成 ===\n");
    printf("关键收获:\n");
    printf("1. 使用 OpProfiler 的 beginOp/endOp 或 OpScopeGuard 进行逐算子计时\n");
    printf("2. GFLOPS = 计算吞吐率，带宽 = 数据搬运效率\n");
    printf("3. 计算密度区分算子类型: >10 计算密集, <5 内存密集\n");
    printf("4. 瓶颈占比 >50%% 时应优先优化最慢算子\n");
    printf("5. saveJson() 导出数据供 Python 绘制 Roofline 模型图\n");

    return 0;
}
