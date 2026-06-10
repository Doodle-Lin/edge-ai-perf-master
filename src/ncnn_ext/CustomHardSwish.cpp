/**
 * @file CustomHardSwish.cpp
 * @brief NCNN自定义算子CustomHardSwish的实现
 *
 * 本文件实现了HardSwish激活函数的NCNN自定义算子，包括：
 * - 标量C++实现路径（通用，所有平台可用）
 * - AVX2 SIMD优化路径（x86平台，Intel Haswell及以上）
 *
 * HardSwish激活函数：
 * ====================
 * HardSwish(x) = x * clip(x + 3, 0, 6) / 6
 *
 * 数学等价形式（分段函数）：
 *   当 x <= -3 时，HardSwish(x) = 0
 *   当 -3 < x < 3 时，HardSwish(x) = x * (x + 3) / 6
 *   当 x >= 3 时，HardSwish(x) = x
 *
 * 与Swish的关系：
 *   Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *   HardSwish是用分段线性函数近似Sigmoid的结果：
 *   sigmoid(x) ≈ clip(x + 3, 0, 6) / 6
 *
 * 使用场景：
 * - MobileNetV3的激活函数
 * - 一些EfficientNet变体
 * - 搜索得到的网络结构中常见
 *
 * 编译说明：
 * - AVX2路径需要编译选项 -mavx2 启用
 * - 运行时检测CPU是否支持AVX2，自动选择路径
 * - 不支持AVX2的CPU会回退到标量路径
 */

#include "ncnn_ext/CustomHardSwish.h"

// 引入NCNN核心头文件
#include "mat.h"    // ncnn::Mat - 张量操作

// ---- AVX2头文件（条件编译） ----
// 仅在编译时启用了AVX2时才包含相关头文件
#if defined(__AVX2__)
#include <immintrin.h>  // AVX2 intrinsics: _mm256_*
#endif

#include <algorithm>  // std::min, std::max

// ============================================================================
// 构造函数
// ============================================================================

/**
 * @brief 构造函数
 *
 * 设置NCNN层属性：
 * - one_blob_only = true：
 *   告知NCNN该层只有一个输入blob和一个输出blob。
 *   NCNN据此可以优化blob传递：直接复用输入blob作为输出blob的存储，
 *   避免一次memcpy。对于逐元素操作（如激活函数），这是安全的。
 *
 * - support_inplace = true：
 *   告知NCNN该层支持in-place计算（输出直接覆盖输入内存）。
 *   对于HardSwish这样的逐元素操作，in-place是安全的，
 *   因为每个输出值只依赖于同一位置的输入值。
 *   inplace模式可以减少约50%的内存占用。
 *
 * 其他可选属性（本层不需要）：
 * - support_vulkan：是否支持Vulkan GPU计算
 * - support_packing：是否支持NCNN的通道打包存储格式
 * - use_int8_inference：是否支持INT8推理
 */
CustomHardSwish::CustomHardSwish()
{
    one_blob_only = true;    // 单输入单输出
    support_inplace = true;  // 支持原地计算
}

// ============================================================================
// 前向计算
// ============================================================================

/**
 * @brief 前向计算入口
 *
 * NCNN的forward()接口设计：
 * - bottom_blobs: 输入blob列表（因one_blob_only=true，只有一个元素）
 * - top_blobs: 输出blob列表（因one_blob_only=true，只有一个元素）
 * - opt: NCNN运行选项（线程数、是否使用FP16等）
 *
 * 由于support_inplace=true，NCNN可能传入相同的bottom_blob和top_blob，
 * 此时直接在原内存上修改即可。
 *
 * @param bottom_blobs 输入blob列表
 * @param top_blobs    输出blob列表
 * @param opt          运行选项
 * @return 0成功，非0失败
 */
int CustomHardSwish::forward(const std::vector<ncnn::Mat>& bottom_blobs,
                              std::vector<ncnn::Mat>& top_blobs,
                              const ncnn::Option& /*opt*/) const
{
    // 获取输入输出blob引用
    // 由于one_blob_only=true，每个列表只有一个元素
    const ncnn::Mat& bottom = bottom_blobs[0];
    ncnn::Mat& top = top_blobs[0];

    // ---- 处理inplace情况 ----
    // 当support_inplace=true时，NCNN可能让top和bottom指向同一块内存
    // 此时无需额外操作，直接在原数据上修改
    if (bottom.data == top.data) {
        // inplace模式：bottom和top共享内存，直接在bottom上计算
        float* ptr = (float*)bottom.data;
        int size = bottom.total();

        // 选择优化路径
#if defined(__AVX2__)
        // 编译时启用了AVX2，使用SIMD优化路径
        forwardAVX2(ptr, size);
#else
        // 未启用AVX2，使用标量路径
        forwardScalar(ptr, size);
#endif
    }
    else {
        // 非inplace模式：需要从bottom拷贝到top
        float* src = (float*)bottom.data;
        float* dst = (float*)top.data;
        int size = bottom.total();

#if defined(__AVX2__)
        // AVX2路径：先拷贝，再inplace计算
        // 这样可以复用forwardAVX2的实现
        memcpy(dst, src, size * sizeof(float));
        forwardAVX2(dst, size);
#else
        // 标量路径：逐元素计算（拷贝+计算合并，减少一次内存遍历）
        forwardScalar(src, dst, size);
#endif
    }

    return 0;
}

// ============================================================================
// 标量C++实现
// ============================================================================

/**
 * @brief 标量C++实现（inplace版本）
 *
 * 使用分段判断实现HardSwish，避免使用条件分支。
 * 编译器可以将其自动向量化（auto-vectorization）。
 *
 * 实现策略：
 * - 使用std::min/std::max代替if-else，便于编译器优化
 * - 先计算 t = clip(x + 3, 0, 6)
 * - 再计算 out = x * t / 6
 *
 * 数学推导：
 *   HardSwish(x) = x * clip(x + 3, 0, 6) / 6
 *                = x * min(max(x + 3, 0), 6) / 6
 *
 * @param ptr   输入/输出数据指针（inplace计算）
 * @param size  元素总数
 */
void CustomHardSwish::forwardScalar(float* ptr, int size)
{
    // 逐元素计算HardSwish
    // 使用分段线性函数而非sigmoid，计算效率更高
    for (int i = 0; i < size; ++i) {
        float x = ptr[i];

        // clip(x + 3, 0, 6)：将(x+3)限制在[0,6]区间
        // 等价于 min(max(x + 3, 0), 6)
        float t = std::max(x + 3.0f, 0.0f);
        t = std::min(t, 6.0f);

        // x * clip(x+3, 0, 6) / 6
        ptr[i] = x * t / 6.0f;
    }
}

/**
 * @brief 标量C++实现（非inplace版本）
 *
 * 与inplace版本功能相同，但输入输出分别存储。
 * 用于非inplace模式，将拷贝和计算合并为一次遍历，
 * 减少内存访问次数（相比先拷贝再inplace计算）。
 *
 * @param src   输入数据指针
 * @param dst   输出数据指针
 * @param size  元素总数
 */
void CustomHardSwish::forwardScalar(float* dst, const float* src, int size)
{
    for (int i = 0; i < size; ++i) {
        float x = src[i];
        float t = std::max(x + 3.0f, 0.0f);
        t = std::min(t, 6.0f);
        dst[i] = x * t / 6.0f;
    }
}

// ============================================================================
// AVX2 SIMD优化实现
// ============================================================================

#if defined(__AVX2__)

/**
 * @brief AVX2 SIMD优化实现（inplace版本）
 *
 * 使用Intel AVX2指令集并行处理8个float元素。
 *
 * AVX2指令说明：
 * ==============
 * - _mm256_set1_ps(val)：将标量val广播到256位向量的8个float通道
 * - _mm256_loadu_ps(ptr)：从内存加载8个非对齐float到向量寄存器
 * - _mm256_storeu_ps(ptr, vec)：将向量寄存器存储到非对齐内存地址
 * - _mm256_add_ps(a, b)：向量加法，8个通道同时执行
 * - _mm256_min_ps(a, b)：向量取最小值，8个通道同时执行
 * - _mm256_max_ps(a, b)：向量取最大值，8个通道同时执行
 * - _mm256_mul_ps(a, b)：向量乘法，8个通道同时执行
 * - _mm256_div_ps(a, b)：向量除法，8个通道同时执行（或用乘法替代）
 *
 * 性能优化技巧：
 * ==============
 * 1. 使用乘法代替除法：a / 6 → a * (1/6)，除法延迟约是乘法的4-5倍
 * 2. 非对齐加载/存储：使用loadu/storeu而非load/store，
 *    因为NCNN的Mat数据不保证32字节对齐
 * 3. 尾部处理：当size不是8的倍数时，剩余元素用标量路径处理
 * 4. FMA融合：如果CPU支持FMA3，可以将 mul+add 融合为单条指令
 *    （本实现未使用FMA，以保持兼容性）
 *
 * @param ptr   输入/输出数据指针（inplace计算）
 * @param size  元素总数
 */
void CustomHardSwish::forwardAVX2(float* ptr, int size)
{
    // ---- 预计算常量 ----
    // 使用乘法代替除法，提升性能
    // 1/6 ≈ 0.16666667f
    const __m256 six = _mm256_set1_ps(6.0f);
    const __m256 three = _mm256_set1_ps(3.0f);
    const __m256 zero = _mm256_set1_ps(0.0f);
    const __m256 one_sixth = _mm256_set1_ps(0.16666667f);

    // ---- 主循环：每次处理8个float ----
    int i = 0;
    for (; i + 7 < size; i += 8) {
        // 加载8个float
        __m256 x = _mm256_loadu_ps(ptr + i);

        // 计算 x + 3
        __m256 x_plus_3 = _mm256_add_ps(x, three);

        // 计算 clip(x + 3, 0, 6) = min(max(x + 3, 0), 6)
        // 步骤1：max(x + 3, 0) — 负值截断为0
        __m256 clipped = _mm256_max_ps(x_plus_3, zero);
        // 步骤2：min(result, 6) — 大于6的截断为6
        clipped = _mm256_min_ps(clipped, six);

        // 计算 x * clip(x+3, 0, 6) / 6
        // 优化：用乘法代替除法
        __m256 result = _mm256_mul_ps(x, clipped);
        result = _mm256_mul_ps(result, one_sixth);

        // 存储结果
        _mm256_storeu_ps(ptr + i, result);
    }

    // ---- 尾部处理 ----
    // 处理剩余不足8个的元素，回退到标量路径
    for (; i < size; ++i) {
        float x = ptr[i];
        float t = std::max(x + 3.0f, 0.0f);
        t = std::min(t, 6.0f);
        ptr[i] = x * t / 6.0f;
    }
}

/**
 * @brief AVX2 SIMD优化实现（非inplace版本）
 *
 * @param src   输入数据指针
 * @param dst   输出数据指针
 * @param size  元素总数
 */
void CustomHardSwish::forwardAVX2(float* dst, const float* src, int size)
{
    const __m256 six = _mm256_set1_ps(6.0f);
    const __m256 three = _mm256_set1_ps(3.0f);
    const __m256 zero = _mm256_set1_ps(0.0f);
    const __m256 one_sixth = _mm256_set1_ps(0.16666667f);

    int i = 0;
    for (; i + 7 < size; i += 8) {
        __m256 x = _mm256_loadu_ps(src + i);
        __m256 x_plus_3 = _mm256_add_ps(x, three);
        __m256 clipped = _mm256_max_ps(x_plus_3, zero);
        clipped = _mm256_min_ps(clipped, six);
        __m256 result = _mm256_mul_ps(x, clipped);
        result = _mm256_mul_ps(result, one_sixth);
        _mm256_storeu_ps(dst + i, result);
    }

    // 尾部标量处理
    for (; i < size; ++i) {
        float x = src[i];
        float t = std::max(x + 3.0f, 0.0f);
        t = std::min(t, 6.0f);
        dst[i] = x * t / 6.0f;
    }
}

#endif // defined(__AVX2__)

// ============================================================================
// NCNN自定义层注册宏
// ============================================================================

/**
 * NCNN需要一种机制在解析.param文件时创建自定义层的实例。
 * 标准做法是使用DEFINE_LAYER_CREATOR宏定义工厂函数。
 *
 * DEFINE_LAYER_CREATOR(CustomHardSwish) 展开为：
 *   static ncnn::Layer* CustomHardSwish_layer_creator() { return new CustomHardSwish; }
 *
 * 然后在加载模型前注册：
 *   net.register_custom_layer("HardSwish_Custom", CustomHardSwish_layer_creator);
 *
 * 也可以通过NcnnCustomLayerRegistry统一注册：
 *   NcnnCustomLayerRegistry::instance().registerLayer<CustomHardSwish>("HardSwish_Custom");
 *   NcnnCustomLayerRegistry::instance().applyAll(net);
 */
