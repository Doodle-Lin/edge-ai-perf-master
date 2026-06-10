/**
 * @file step9_custom_ncnn_op.cpp
 * @brief Lab3 Step9: NCNN 自定义算子 —— 在 NCNN 中部署优化算子
 *
 * ====== 本实验的学习目标 ======
 * 1. 掌握 NCNN 自定义算子的完整开发流程
 * 2. 实现 CustomHardSwish 层 (继承 ncnn::Layer)
 * 3. 通过 NcnnCustomLayerRegistry 注册自定义算子
 * 4. 加载 NCNN 模型并替换其中的 ReLU 层为 HardSwish
 * 5. 运行推理验证自定义算子工作正常
 *
 * ====== NCNN 自定义算子开发流程 ======
 *
 * 1. 继承 ncnn::Layer 类
 *    - 在构造函数中设置 one_blob_only、support_inplace 等属性
 *    - 实现 forward() 方法 (CPU 路径)
 *    - 可选: 实现 forward_gpu() 方法 (Vulkan 路径)
 *
 * 2. 注册到 NcnnCustomLayerRegistry
 *    - 在 load_param 之前调用 registerLayer<CustomHardSwish>("HardSwish_Custom")
 *    - 注册表会自动将 .param 文件中的该类型映射到我们的实现
 *
 * 3. 使用 NcnnModelRunner 加载模型并推理
 *    - 调用 applyAll(net) 将注册应用到 Net 实例
 *    - 正常加载模型、设置输入、执行推理
 *
 * ====== HardSwish 激活函数 ======
 *
 * HardSwish(x) = x * clip(x + 3, 0, 6) / 6
 *
 * 等价展开:
 *   当 x <= -3 时, HardSwish(x) = 0
 *   当 -3 < x < 3 时, HardSwish(x) = x * (x + 3) / 6
 *   当 x >= 3 时,   HardSwish(x) = x
 *
 * 特点:
 * - 是 Swish(x) = x * sigmoid(x) 的分段线性近似
 * - 计算效率远高于 Swish (无需计算 exp)
 * - 在 MobileNetV3 中被首次提出并广泛使用
 * - 非单调性: 在负值区域有小幅负输出, 有助于信息流动
 *
 * ====== 编译与运行 ======
 * 有 NCNN: g++ -O2 -std=c++17 -I<ncnn_include> -L<ncnn_lib> -lncnn
 *          -o step9_custom_ncnn_op step9_custom_ncnn_op.cpp
 * 无 NCNN: g++ -O2 -std=c++17 -o step9_custom_ncnn_op step9_custom_ncnn_op.cpp
 * ./step9_custom_ncnn_op
 */

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

// ============================================================================
// NCNN 条件编译
// 如果没有 NCNN, 我们提供一个独立的模拟实现来演示概念
// ============================================================================
#ifdef USE_NCNN

// 使用真正的 NCNN 头文件
#include "layer.h"
#include "mat.h"
#include "net.h"

#else

// ============================================================================
// NCNN 模拟实现 (用于没有 NCNN 的环境)
// 提供最小的接口定义, 让代码可以编译运行
// ============================================================================

namespace ncnn {

/// 模拟 ncnn::Option
class Option {
public:
    int num_threads = 4;
    bool use_vulkan_compute = false;
};

/// 模拟 ncnn::Mat — NCNN 的张量类
/// 实际 NCNN 中 Mat 更复杂, 支持多种数据类型和内存布局
class Mat {
public:
    int w = 0, h = 0, c = 0;
    float* data = nullptr;
    bool owner = true;

    Mat() = default;
    Mat(int _w) : w(_w), h(1), c(1) {
        data = new float[w * h * c]();
    }
    Mat(int _w, int _h) : w(_w), h(_h), c(1) {
        data = new float[w * h * c]();
    }
    Mat(int _w, int _h, int _c) : w(_w), h(_h), c(_c) {
        data = new float[w * h * c]();
    }
    ~Mat() { if (owner && data) delete[] data; }

    // 禁止拷贝
    Mat(const Mat&) = delete;
    Mat& operator=(const Mat&) = delete;

    int total() const { return w * h * c; }

    // 通道指针: channel(c) 返回第 c 个通道的数据起始地址
    float* channel(int cidx) {
        return data + cidx * w * h;
    }
    const float* channel(int cidx) const {
        return data + cidx * w * h;
    }
};

/// 模拟 ncnn::Layer — 所有算子的基类
/// 实际 NCNN 中 Layer 有更多方法 (load_param, load_model 等)
class Layer {
public:
    bool one_blob_only = false;   ///< 是否单输入单输出
    bool support_inplace = false;  ///< 是否支持原地计算
    bool support_vulkan = false;  ///< 是否支持 Vulkan GPU

    Layer() = default;
    virtual ~Layer() = default;

    /// 前向计算 (CPU 路径)
    /// @param bottom_blobs 输入 blob 列表
    /// @param top_blobs    输出 blob 列表
    /// @param opt          运行选项
    /// @return 0 成功, 非0 失败
    virtual int forward(const std::vector<Mat>& bottom_blobs,
                        std::vector<Mat>& top_blobs,
                        const Option& opt) const {
        (void)bottom_blobs; (void)top_blobs; (void)opt;
        return -1;  // 未实现
    }
};

/// 模拟 ncnn::Net — 网络容器
class Net {
public:
    Option opt;

    /// 注册自定义层
    void register_custom_layer(const char* type, Layer* (*)(void)) {
        printf("  [模拟] 注册自定义层: %s\n", type);
    }

    /// 加载参数文件
    int load_param(const char* path) {
        printf("  [模拟] 加载参数文件: %s\n", path);
        return 0;
    }

    /// 加载权重文件
    int load_model(const char* path) {
        printf("  [模拟] 加载权重文件: %s\n", path);
        return 0;
    }
};

} // namespace ncnn

#endif // USE_NCNN

// ============================================================================
// AVX2 条件包含
// ============================================================================
#ifdef USE_AVX2
#include <immintrin.h>
#endif

// ============================================================================
// 简易计时器
// ============================================================================
class Timer {
public:
    void start() { start_ = std::chrono::steady_clock::now(); }
    void stop()  { stop_  = std::chrono::steady_clock::now(); }
    double elapsedMs() const {
        return std::chrono::duration<double, std::milli>(stop_ - start_).count();
    }
    double elapsedUs() const {
        return std::chrono::duration<double, std::micro>(stop_ - start_).count();
    }
private:
    std::chrono::steady_clock::time_point start_, stop_;
};

// ============================================================================
// CustomHardSwish — 自定义激活函数层
//
// HardSwish(x) = x * clip(x + 3, 0, 6) / 6
//
// 继承自 ncnn::Layer, 实现 forward() 方法
// 同时提供标量 C++ 实现和 AVX2 SIMD 优化实现
// ============================================================================
class CustomHardSwish : public ncnn::Layer {
public:
    /**
     * @brief 构造函数
     *
     * 设置层属性:
     * - one_blob_only = true: 单输入单输出, NCNN 可以优化 blob 传递
     * - support_inplace = true: 支持原地计算, 减少内存拷贝
     *   对于逐元素操作 (如激活函数), inplace 是安全的,
     *   因为每个输出只依赖于同一位置的输入
     */
    CustomHardSwish() {
        one_blob_only = true;    // 单输入单输出
        support_inplace = true;  // 支持原地计算
    }

    /**
     * @brief 前向计算 (CPU 路径)
     *
     * 实现 HardSwish 激活函数的核心计算逻辑
     *
     * NCNN 的 forward() 接口约定:
     * - bottom_blobs: 输入张量列表 (one_blob_only 时只有一个元素)
     * - top_blobs:    输出张量列表 (one_blob_only 时只有一个元素)
     * - opt:          运行选项, 包含线程数等配置
     * - 返回值:       0 表示成功, 非 0 表示失败
     *
     * @param bottom_blobs 输入 blob 列表
     * @param top_blobs    输出 blob 列表
     * @param opt          NCNN 运行选项
     * @return 0 成功, 非0 失败
     */
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                        std::vector<ncnn::Mat>& top_blobs,
                        const ncnn::Option& opt) const override {
        // one_blob_only = true, 所以只有一个输入和一个输出
        const ncnn::Mat& bottom = bottom_blobs[0];
        ncnn::Mat& top = top_blobs[0];

        // 如果支持原地计算, top 和 bottom 可能指向同一内存
        // NCNN 会自动处理, 我们只需正确读写数据

        int size = bottom.w * bottom.h * bottom.c;

        // 选择优化路径
#ifdef USE_AVX2
        // AVX2 路径: 一次处理 8 个 float
        forwardAVX2(bottom.data, size);
#else
        // 标量路径: 逐元素处理
        forwardScalar(bottom.data, size);
#endif

        // 如果不是原地计算, 需要拷贝到 top
        if (bottom.data != top.data) {
            for (int i = 0; i < size; i++) {
                top.data[i] = bottom.data[i];
            }
        }

        (void)opt;  // 暂不使用多线程
        return 0;
    }

private:
    /**
     * @brief 标量 C++ 实现路径
     *
     * 使用纯 C++ 的标量运算, 逐元素计算 HardSwish
     * 适用于不支持 AVX2 的平台 (如 ARM、旧 x86 CPU)
     *
     * 使用分支消除的写法, 便于编译器自动向量化:
     *   HardSwish(x) = x * max(0, min(6, x+3)) / 6
     *   而不是 if-else 分支 (分支预测可能失败, 影响性能)
     *
     * @param ptr   输入/输出数据指针 (in-place 计算)
     * @param size  元素总数
     */
    static void forwardScalar(float* ptr, int size) {
        for (int i = 0; i < size; i++) {
            float x = ptr[i];
            // HardSwish(x) = x * clip(x+3, 0, 6) / 6
            // clip(v, lo, hi) = max(lo, min(hi, v))
            float clipped = std::max(0.0f, std::min(6.0f, x + 3.0f));
            ptr[i] = x * clipped / 6.0f;
        }
    }

    /**
     * @brief AVX2 SIMD 优化实现路径
     *
     * 使用 Intel AVX2 指令集并行处理 8 个 float:
     * - _mm256_loadu_ps:   加载 8 个非对齐 float
     * - _mm256_add_ps:     向量加法
     * - _mm256_min_ps:     向量取最小值 (实现 clip 的上界)
     * - _mm256_max_ps:     向量取最大值 (实现 clip 的下界)
     * - _mm256_mul_ps:     向量乘法
     * - _mm256_set1_ps:    广播标量到向量
     * - _mm256_storeu_ps:  存储 8 个 float
     *
     * AVX2 相比标量路径的理论加速比: 8x (8 个 float 并行)
     * 实际加速比受内存带宽限制, 通常为 3-5x
     *
     * @param ptr   输入/输出数据指针 (in-place 计算)
     * @param size  元素总数 (自动处理尾部不足 8 个的元素)
     */
    static void forwardAVX2(float* ptr, int size) {
#ifdef USE_AVX2
        __m256 six = _mm256_set1_ps(6.0f);    // 常量 6.0
        __m256 three = _mm256_set1_ps(3.0f);  // 常量 3.0
        __m256 zero = _mm256_set1_ps(0.0f);   // 常量 0.0
        __m256 inv6 = _mm256_set1_ps(1.0f / 6.0f);  // 预计算 1/6, 用乘法代替除法

        int i = 0;
        // 主循环: 每次处理 8 个元素
        for (; i + 7 < size; i += 8) {
            __m256 x = _mm256_loadu_ps(&ptr[i]);        // 加载 8 个 x
            __m256 x_plus_3 = _mm256_add_ps(x, three);  // x + 3
            __m256 clipped = _mm256_min_ps(              // min(x+3, 6)
                _mm256_max_ps(x_plus_3, zero),           // max(x+3, 0)
                six);
            __m256 result = _mm256_mul_ps(               // x * clipped / 6
                _mm256_mul_ps(x, clipped), inv6);
            _mm256_storeu_ps(&ptr[i], result);           // 存回结果
        }

        // 尾部处理: 剩余 1-7 个元素用标量
        for (; i < size; i++) {
            float x = ptr[i];
            float clipped = std::max(0.0f, std::min(6.0f, x + 3.0f));
            ptr[i] = x * clipped / 6.0f;
        }
#else
        // 回退到标量路径
        forwardScalar(ptr, size);
#endif
    }
};

// ============================================================================
// NcnnCustomLayerRegistry — 自定义算子注册表 (模拟实现)
//
// 本注册表与项目中的 NcnnCustomLayerRegistry 功能一致:
// - 统一管理 NCNN 自定义算子的注册
// - 将自定义算子批量应用到 NCNN Net 实例
// - 提供查询已注册算子的接口
//
// 设计优势:
// - 集中管理: 所有自定义算子的注册代码集中在一处
// - 解耦: 算子实现与模型加载代码分离
// - 延迟注册: 可以先注册所有算子, 在需要时才 apply 到 Net
// ============================================================================
class NcnnCustomLayerRegistry {
public:
    /// 获取注册表单例 (Meyers' Singleton)
    static NcnnCustomLayerRegistry& instance() {
        static NcnnCustomLayerRegistry reg;
        return reg;
    }

    /**
     * @brief 注册自定义算子
     * @tparam LayerT 自定义算子类型, 需继承 ncnn::Layer
     * @param layerName 算子名称, 需与 .param 文件中的 type 字段一致
     */
    template<typename LayerT>
    void registerLayer(const std::string& layerName) {
        registeredNames_.push_back(layerName);
        printf("  [注册表] 注册自定义算子: %s\n", layerName.c_str());
    }

    /**
     * @brief 将所有已注册的自定义算子应用到 NCNN Net
     * @param net 目标 NCNN Net 引用
     *
     * 必须在 net.load_param() 之前调用!
     * 否则 NCNN 解析 .param 文件时遇到未注册的算子类型会报错
     */
    void applyAll(ncnn::Net& net) {
        for (const auto& name : registeredNames_) {
            printf("  [注册表] 应用到 Net: %s\n", name.c_str());
            // 在实际实现中, 调用 net.register_custom_layer(name, creator)
            // 这里是模拟, 不实际调用
            (void)net;
        }
    }

    /// 获取已注册的算子名称列表
    const std::vector<std::string>& registeredLayers() const {
        return registeredNames_;
    }

    /// 检查指定算子是否已注册
    bool hasLayer(const std::string& name) const {
        return std::find(registeredNames_.begin(), registeredNames_.end(), name)
               != registeredNames_.end();
    }

private:
    NcnnCustomLayerRegistry() = default;
    std::vector<std::string> registeredNames_;
};

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("=== Lab3 Step9: NCNN 自定义算子 ===\n\n");

    // ================================================================
    // Step 1: 注册自定义算子
    //
    // 在实际项目中, 这通常在程序初始化时调用
    // 注册只需做一次, 之后所有 Net 实例都可以使用
    // ================================================================
    printf("--- Step 1: 注册自定义算子 ---\n");
    NcnnCustomLayerRegistry::instance().registerLayer<CustomHardSwish>("HardSwish_Custom");

    // 可以注册多个自定义算子
    // NcnnCustomLayerRegistry::instance().registerLayer<CustomGELU>("GELU_Custom");
    // NcnnCustomLayerRegistry::instance().registerLayer<CustomSiLU>("SiLU_Custom");

    printf("  已注册 %d 个自定义算子\n",
           (int)NcnnCustomLayerRegistry::instance().registeredLayers().size());

    // ================================================================
    // Step 2: 创建 NCNN Net 并应用注册
    //
    // applyAll 必须在 load_param 之前调用!
    // ================================================================
    printf("\n--- Step 2: 创建 Net 并应用注册 ---\n");
    ncnn::Net net;

    // 将注册表中的所有算子应用到 Net
    NcnnCustomLayerRegistry::instance().applyAll(net);

    // ================================================================
    // Step 3: 加载模型
    //
    // 加载 .param (网络结构) 和 .bin (权重) 文件
    // 如果模型中包含 "HardSwish_Custom" 类型的层,
    // NCNN 会自动使用我们注册的 CustomHardSwish 来处理
    // ================================================================
    printf("\n--- Step 3: 加载模型 ---\n");

    // 在实际项目中:
    //   net.load_param("model.param");
    //   net.load_model("model.bin");
    //
    // 这里使用模拟, 因为没有实际的模型文件
    printf("  [模拟] 加载模型 (实际项目中使用 net.load_param / net.load_model)\n");
    printf("  模型中如果包含 \"HardSwish_Custom\" 类型的层,\n");
    printf("  NCNN 会自动使用注册的 CustomHardSwish 来处理\n");

    // ================================================================
    // Step 4: 运行推理 (HardSwish 计算)
    //
    // 使用 Extractor 设置输入、执行推理、提取输出
    // 自定义算子会在推理过程中被自动调用
    // ================================================================
    printf("\n--- Step 4: 运行推理 (HardSwish 计算) ---\n");

    // 直接测试 CustomHardSwish 的计算功能
    // 不需要完整的 NCNN 模型, 也能验证自定义算子的正确性
    const int SIZE = 1024;
    ncnn::Mat input(SIZE);
    ncnn::Mat output(SIZE);

    // 初始化输入数据: 范围 [-5, 5]
    for (int i = 0; i < SIZE; i++) {
        input.data[i] = -5.0f + 10.0f * i / SIZE;
    }

    // 执行 HardSwish
    CustomHardSwish hardSwish;
    std::vector<ncnn::Mat> bottom = {input};
    std::vector<ncnn::Mat> top = {output};

    Timer timer;
    timer.start();
    hardSwish.forward(bottom, top, ncnn::Option());
    timer.stop();
    printf("  CustomHardSwish 计算完成 (%d 元素, %.1f us)\n",
           SIZE, timer.elapsedUs());

    // ================================================================
    // Step 5: 正确性验证
    //
    // 验证 HardSwish 的计算结果是否符合数学定义
    // ================================================================
    printf("\n--- Step 5: 正确性验证 ---\n");
    printf("  HardSwish(x) = x * clip(x+3, 0, 6) / 6\n\n");

    // 检查几个关键点
    struct TestCase {
        float x;
        float expected;
    };

    std::vector<TestCase> testCases = {
        {-5.0f, 0.0f},        // x <= -3: HardSwish = 0
        {-3.0f, 0.0f},        // x = -3: 边界值
        {-1.5f, -0.375f},     // -3 < x < 3: HardSwish = x*(x+3)/6
        { 0.0f, 0.0f},        // x = 0: HardSwish = 0
        { 1.5f, 1.125f},      // -3 < x < 3: HardSwish = x*(x+3)/6
        { 3.0f, 3.0f},        // x = 3: 边界值
        { 5.0f, 5.0f},        // x >= 3: HardSwish = x
    };

    printf("  %-8s %-12s %-12s %-10s\n", "x", "期望值", "实际值", "误差");
    printf("  %-8s %-12s %-12s %-10s\n", "------", "--------", "--------", "------");

    for (const auto& tc : testCases) {
        // 在 output 中找到最近的元素
        int idx = static_cast<int>((tc.x + 5.0f) / 10.0f * SIZE);
        idx = std::max(0, std::min(idx, SIZE - 1));
        float actual = output.data[idx];
        float diff = std::fabs(actual - tc.expected);
        printf("  %-8.1f %-12.4f %-12.4f %-10.2e %s\n",
               tc.x, tc.expected, actual, diff,
               diff < 0.01f ? "OK" : "MISMATCH");
    }

    // ================================================================
    // Step 6: 性能测试
    //
    // 对比标量实现和 AVX2 实现的性能
    // ================================================================
    printf("\n--- Step 6: 性能测试 ---\n");

    const int PERF_SIZE = 1024 * 1024;  // 1M 元素
    std::vector<float> perfData(PERF_SIZE);
    for (int i = 0; i < PERF_SIZE; i++) {
        perfData[i] = -5.0f + 10.0f * static_cast<float>(rand()) / RAND_MAX;
    }

    // 标量实现
    std::vector<float> dataCopy = perfData;
    Timer tScalar;
    tScalar.start();
    for (int iter = 0; iter < 100; iter++) {
        for (int i = 0; i < PERF_SIZE; i++) {
            float x = dataCopy[i];
            float clipped = std::max(0.0f, std::min(6.0f, x + 3.0f));
            dataCopy[i] = x * clipped / 6.0f;
        }
    }
    tScalar.stop();

    // AVX2 实现
    std::vector<float> dataAvx2 = perfData;
    Timer tAvx2;
    tAvx2.start();
    for (int iter = 0; iter < 100; iter++) {
        // 调用 AVX2 路径 (通过 CustomHardSwish)
        ncnn::Mat avx2Input(PERF_SIZE);
        memcpy(avx2Input.data, dataAvx2.data(), PERF_SIZE * sizeof(float));
        std::vector<ncnn::Mat> bot = {avx2Input};
        std::vector<ncnn::Mat> top2 = {avx2Input};
        CustomHardSwish hs;
        hs.forward(bot, top2, ncnn::Option());
        memcpy(dataAvx2.data(), avx2Input.data, PERF_SIZE * sizeof(float));
    }
    tAvx2.stop();

    printf("  %d 元素 x 100 次迭代:\n", PERF_SIZE);
    printf("    标量实现: %.2f ms\n", tScalar.elapsedMs());
    printf("    当前实现: %.2f ms\n", tAvx2.elapsedMs());

    // ================================================================
    // Step 7: 如何在 .param 文件中使用自定义算子
    // ================================================================
    printf("\n--- Step 7: .param 文件中的自定义算子用法 ---\n\n");
    printf("  NCNN .param 文件格式 (文本):\n");
    printf("  7767517\n");  // 魔数
    printf("  10 11\n");     // 层数, blob数
    printf("  Input            data     0 1 data\n");
    printf("  Convolution      conv1    1 1 data conv1 0=64 1=3 2=1 3=1 4=3\n");
    printf("  HardSwish_Custom hs1      1 1 conv1 hs1   <- 使用我们的自定义算子!\n");
    printf("  Convolution      conv2    1 1 hs1 conv2   0=128 1=3 2=1 3=1 4=64\n");
    printf("  ...\n\n");

    printf("  关键: .param 中的 type 字段 \"HardSwish_Custom\" 必须与\n");
    printf("        registerLayer<CustomHardSwish>(\"HardSwish_Custom\") 中的名称一致!\n\n");

    printf("  替换 ReLU 为 HardSwish:\n");
    printf("    原始: ReLU            relu1    1 1 conv1 relu1\n");
    printf("    替换: HardSwish_Custom hs1      1 1 conv1 hs1\n");
    printf("    只需修改 type 字段, 其他不变\n");

    // ================================================================
    // 总结
    // ================================================================
    printf("\n--- 总结 ---\n\n");
    printf("  NCNN 自定义算子开发流程:\n");
    printf("    1. 继承 ncnn::Layer, 设置 one_blob_only 和 support_inplace\n");
    printf("    2. 实现 forward() 方法 (CPU 路径)\n");
    printf("    3. 可选: 实现 forward_gpu() (Vulkan 路径)\n");
    printf("    4. 通过注册表注册: registerLayer<MyLayer>(\"MyType\")\n");
    printf("    5. 在 load_param 之前调用 applyAll(net)\n");
    printf("    6. 正常加载模型、推理\n\n");

    printf("  自定义算子的典型应用场景:\n");
    printf("    - 新激活函数 (HardSwish, GELU, SiLU, Mish 等)\n");
    printf("    - 自定义注意力机制\n");
    printf("    - 量化/反量化层\n");
    printf("    - 特殊的预处理/后处理层\n");
    printf("    - 性能优化的算子替换 (如用 Winograd 替换朴素卷积)\n\n");

    printf("  与项目架构的关系:\n");
    printf("    - CustomHardSwish 类 -> include/ncnn_ext/CustomHardSwish.h\n");
    printf("    - NcnnCustomLayerRegistry -> include/ncnn_ext/NcnnCustomLayerRegistry.h\n");
    printf("    - NcnnModelRunner -> include/ncnn_ext/NcnnModelRunner.h\n");
    printf("    - 这些组件协同工作, 实现自定义算子的无缝集成\n\n");

    printf("  生产环境注意事项:\n");
    printf("    1. Vulkan GPU 路径: 需要额外实现 create_pipeline() 和 forward_gpu()\n");
    printf("    2. 量化支持: INT8 量化需要额外的量化参数处理\n");
    printf("    3. 模型转换: 从 PyTorch/TF -> ONNX -> NCNN 的转换链\n");
    printf("    4. 单元测试: 自定义算子必须有充分的正确性测试\n");
    printf("    5. 性能回归: 定期 benchmark, 确保优化不退化\n");

    printf("\n=== Step9 完成 ===\n");
    printf("=== Lab3 全部 9 个步骤完成! ===\n\n");

    printf("回顾整个 Lab3 的优化路径:\n");
    printf("  Step1: 朴素卷积      -> 性能基线 (0.5-1 GFLOPS)\n");
    printf("  Step2: im2col+AVX2   -> SIMD 并行 (4-8x 加速)\n");
    printf("  Step3: 缓存分块      -> 缓存优化 (1.5-2x 加速)\n");
    printf("  Step4: Winograd F(2,3) -> 减少乘法 (2.25x 加速, 仅3x3)\n");
    printf("  Step5: 分块 GEMM     -> BLAS 级优化 (接近硬件峰值)\n");
    printf("  Step6: Vulkan 基础    -> GPU 计算入门\n");
    printf("  Step7: Vulkan GEMM   -> GPU 矩阵乘法\n");
    printf("  Step8: Vulkan Conv2D  -> GPU 卷积, 全面对比\n");
    printf("  Step9: NCNN 自定义算子 -> 生产部署\n\n");

    printf("核心教训:\n");
    printf("  1. 先建立基线, 再逐步优化 -- 不要猜测瓶颈\n");
    printf("  2. 每次只改一个变量 -- 确认每个优化的独立贡献\n");
    printf("  3. 正确性先于性能 -- 优化必须保持结果正确\n");
    printf("  4. CPU 和 GPU 各有优势 -- 根据场景选择\n");
    printf("  5. 生产部署需要工程化 -- 自定义算子是连接优化和部署的桥梁\n");

    return 0;
}
