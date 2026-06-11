/**
 * @file CustomHardSwish.h
 * @brief NCNN自定义算子示例：HardSwish激活函数
 *
 * 本文件实现了一个完整的NCNN自定义算子——HardSwish激活函数，
 * 用于演示NCNN自定义算子的完整开发流程。
 *
 * HardSwish激活函数定义：
 * ========================
 * HardSwish(x) = x * clip(x + 3, 0, 6) / 6
 *
 * 等价展开形式：
 *   当 x <= -3 时，HardSwish(x) = 0
 *   当 -3 < x < 3 时，HardSwish(x) = x * (x + 3) / 6
 *   当 x >= 3 时，HardSwish(x) = x
 *
 * 特点：
 * - 是Swish(x) = x * sigmoid(x)的分段线性近似
 * - 计算效率远高于Swish（无需计算exp）
 * - 在MobileNetV3中被首次提出并广泛使用
 * - 非单调性：在负值区域有小幅负输出，有助于信息流动
 *
 * NCNN自定义算子开发规范：
 * ========================
 * 1. 继承ncnn::Layer类
 * 2. 在构造函数中设置one_blob_only = true（单输入单输出）
 * 3. 实现forward()方法，支持就地（in-place）计算
 * 4. 如需支持Vulkan GPU，还需实现create_pipeline()和forward_gpu()
 * 5. 注册到NcnnCustomLayerRegistry以便自动加载
 *
 * 本实现的优化路径：
 * - 标量路径（C++）：使用分段判断，避免分支预测失败
 * - AVX2路径：使用SIMD指令批量处理8个float，大幅提升吞吐
 */

#ifndef NCNN_EXT_CUSTOM_HARD_SWISH_H
#define NCNN_EXT_CUSTOM_HARD_SWISH_H

#ifdef USE_NCNN

#include "layer.h"      // ncnn::Layer基类
#include "mat.h"         // ncnn::Mat张量类

/**
 * @class CustomHardSwish
 * @brief HardSwish激活函数的NCNN自定义算子实现
 *
 * 继承自ncnn::Layer，实现HardSwish激活函数的前向计算。
 * 同时提供标量C++实现和AVX2 SIMD优化实现。
 *
 * 内存模型：
 * - one_blob_only = true：表示该算子只有一个输入blob和一个输出blob
 *   NCNN可以据此优化内存分配（如允许in-place计算）
 * - 支持in-place模式：当bottom_blob和top_blob指向同一内存时，
 *   直接在原内存上修改，避免额外的内存分配和拷贝
 *
 * 使用方式：
 * @code
 * // 注册到NCNN（在load_param之前）
 * NcnnCustomLayerRegistry::instance().registerLayer<CustomHardSwish>("HardSwish_Custom");
 * @endcode
 */
class CustomHardSwish : public ncnn::Layer {
public:
    /**
     * @brief 构造函数
     *
     * 设置层属性：
     * - one_blob_only = true：单输入单输出，NCNN会优化blob传递
     * - support_inplace = true：支持原地计算，减少内存拷贝
     *   对于逐元素操作（如激活函数），inplace是安全的，
     *   因为每个输出只依赖于同一位置的输入
     */
    CustomHardSwish();

    /**
     * @brief 前向计算（CPU路径）
     *
     * 实现HardSwish激活函数的核心计算逻辑。
     * 根据编译时是否启用AVX2，自动选择优化路径。
     *
     * NCNN的forward()接口约定：
     * - bottom_blobs: 输入张量列表（one_blob_only时只有一个元素）
     * - top_blobs: 输出张量列表（one_blob_only时只有一个元素）
     * - opt: 运行选项，包含线程数等配置
     * - 返回值: 0表示成功，非0表示失败
     *
     * @param bottom_blobs 输入blob列表
     * @param top_blobs    输出blob列表
     * @param opt          NCNN运行选项
     * @return 0成功，非0失败
     */
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                        std::vector<ncnn::Mat>& top_blobs,
                        const ncnn::Option& opt) const override;

private:
    /**
     * @brief 标量C++实现路径
     *
     * 使用纯C++的标量运算，逐元素计算HardSwish。
     * 适用于不支持AVX2的平台（如ARM、旧x86 CPU）。
     * 使用分支消除的写法，便于编译器自动向量化。
     *
     * @param ptr     输入/输出数据指针（in-place计算）
     * @param size    元素总数
     */
    static void forwardScalar(float* ptr, int size);

    /**
     * @brief AVX2 SIMD优化实现路径
     *
     * 使用Intel AVX2指令集并行处理8个float：
     * - _mm256_loadu_ps: 加载8个非对齐float
     * - _mm256_add_ps: 向量加法
     * - _mm256_min_ps / _mm256_max_ps: 向量取极值（实现clip）
     * - _mm256_mul_ps: 向量乘法
     * - _mm256_set1_ps: 广播标量到向量
     *
     * AVX2相比标量路径的理论加速比：8x（8个float并行）
     * 实际加速比受内存带宽限制，通常为3-5x。
     *
     * @param ptr     输入/输出数据指针（in-place计算）
     * @param size    元素总数（自动处理尾部不足8个的元素）
     */
    static void forwardAVX2(float* ptr, int size);
};

#endif // USE_NCNN

#endif // NCNN_EXT_CUSTOM_HARD_SWISH_H
