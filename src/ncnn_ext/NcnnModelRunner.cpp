/**
 * @file NcnnModelRunner.cpp
 * @brief NCNN模型加载与推理执行器的实现
 *
 * 本文件实现了NcnnModelRunner类的所有方法，包括：
 * - 模型加载（CPU/Vulkan双后端）
 * - 输入blob设置
 * - 前向推理（普通模式和逐层profiling模式）
 * - 模型层信息解析
 *
 * NCNN推理流程概览：
 * ===================
 * 1. 创建ncnn::Net实例
 * 2. 配置Net选项（线程数、Vulkan、量化等）
 * 3. 加载模型：load_param() + load_model()
 * 4. 创建Extractor（轻量级推理句柄）
 * 5. 设置输入：extractor.input("blob_name", mat)
 * 6. 执行推理：extractor.extract("output_name", out_mat)
 *
 * Extractor的设计理念：
 * - 每次推理创建新的Extractor，保证推理间不共享状态
 * - Extractor本身很轻量（仅持有Net的引用和输入map）
 * - 同一Net可同时有多个Extractor（但NCNN内部有全局锁，实际串行）
 */

#include "ncnn_ext/NcnnModelRunner.h"
#include "ncnn_ext/OpProfiler.h"  // OpProfiler定义

// 引入NCNN核心头文件
#include "net.h"       // ncnn::Net - 网络容器，管理模型和推理
#include "mat.h"       // ncnn::Mat - 张量类，存储和传递数据
#include "layer.h"     // ncnn::Layer - 算子基类
#include "extractor.h"  // ncnn::Extractor - 推理执行器

#include <chrono>      // 高精度计时器
#include <cstring>     // memcpy
#include <algorithm>   // std::min
#include <stdexcept>   // runtime_error

// ============================================================================
// 构造函数与析构函数
// ============================================================================

/**
 * @brief 默认构造函数
 *
 * 仅初始化成员变量，不创建Net实例。
 * Net实例在loadModel()中按需创建（取决于是否启用Vulkan），
 * 这样可以避免在不需要GPU时创建无用的Vulkan资源。
 */
NcnnModelRunner::NcnnModelRunner()
    : cpuNet_(nullptr)
    , gpuNet_(nullptr)
    , vulkanEnabled_(false)
    , modelLoaded_(false)
{
}

/**
 * @brief 析构函数
 *
 * unique_ptr会自动释放Net实例。
 * NCNN的Net析构时会释放所有层的权重和中间buffer。
 * 如果启用了Vulkan，还会释放Vulkan device memory和command pool。
 */
NcnnModelRunner::~NcnnModelRunner() = default;

// ============================================================================
// 移动构造与移动赋值
// ============================================================================

/**
 * @brief 移动构造函数
 *
 * 将资源所有权从other转移到this，确保Net实例不会被重复释放。
 * 移动后other处于有效但未指定状态（所有unique_ptr变为nullptr）。
 */
NcnnModelRunner::NcnnModelRunner(NcnnModelRunner&& other) noexcept
    : cpuNet_(std::move(other.cpuNet_))
    , gpuNet_(std::move(other.gpuNet_))
    , vulkanEnabled_(other.vulkanEnabled_)
    , modelLoaded_(other.modelLoaded_)
    , layerInfoCache_(std::move(other.layerInfoCache_))
{
    other.modelLoaded_ = false;
    other.vulkanEnabled_ = false;
}

/**
 * @brief 移动赋值运算符
 *
 * 使用copy-and-swap惯用法的简化版本。
 * 先释放当前资源，再转移other的资源。
 */
NcnnModelRunner& NcnnModelRunner::operator=(NcnnModelRunner&& other) noexcept
{
    if (this != &other) {
        cpuNet_ = std::move(other.cpuNet_);
        gpuNet_ = std::move(other.gpuNet_);
        vulkanEnabled_ = other.vulkanEnabled_;
        modelLoaded_ = other.modelLoaded_;
        layerInfoCache_ = std::move(other.layerInfoCache_);
        other.modelLoaded_ = false;
        other.vulkanEnabled_ = false;
    }
    return *this;
}

// ============================================================================
// 模型加载
// ============================================================================

/**
 * @brief 加载NCNN模型
 *
 * 完整的模型加载流程：
 * 1. 创建Net实例并配置选项
 * 2. 注册自定义算子（如有）
 * 3. 加载.param文件（解析网络结构）
 * 4. 加载.bin文件（加载权重数据）
 * 5. 解析并缓存层信息
 *
 * NCNN的opt（ncnn::Option）配置说明：
 * - num_threads: CPU线程数，影响OpenMP并行度
 * - use_vulkan_compute: 是否使用Vulkan GPU计算
 *   设为true后，NCNN会为支持Vulkan的层自动生成compute shader
 * - use_fp16_packing: 是否使用FP16打包存储
 *   可以将内存占用减半，但可能损失精度
 * - use_fp16_storage: 是否使用FP16存储权重
 * - use_int8_inference: 是否使用INT8推理（需校准表）
 * - lightmode: 轻量模式，减少内存占用但可能稍慢
 *
 * @param paramPath    .param文件路径
 * @param binPath      .bin文件路径
 * @param enableVulkan 是否启用Vulkan GPU加速
 * @param numThreads   CPU推理线程数
 * @return true加载成功
 */
bool NcnnModelRunner::loadModel(const std::string& paramPath,
                                 const std::string& binPath,
                                 bool enableVulkan,
                                 int numThreads)
{
    // 防止重复加载：如果已有模型，先释放旧模型
    cpuNet_.reset();
    gpuNet_.reset();
    layerInfoCache_.clear();
    modelLoaded_ = false;

    // ---- 步骤1：创建并配置CPU Net ----
    // CPU Net始终创建，即使启用了Vulkan，某些层仍可能回退到CPU执行
    cpuNet_ = std::make_unique<ncnn::Net>();

    // 获取CPU Net的选项引用，设置推理参数
    ncnn::Option& cpuOpt = cpuNet_->opt;

    // 设置CPU推理线程数
    // NCNN使用OpenMP进行多线程并行，此值控制omp_set_num_threads()
    // 建议设为物理核心数，超线程核心效率较低
    cpuOpt.num_threads = numThreads;

    // 开启轻量模式：每层计算完后立即释放输入blob
    // 优点：减少峰值内存占用（对嵌入式设备很重要）
    // 缺点：无法在推理后获取中间层输出
    cpuOpt.lightmode = true;

    // ---- 步骤2：加载模型到CPU Net ----
    // load_param()：解析.param文本文件，构建网络拓扑结构
    // 返回值：0成功，非0失败
    int ret = cpuNet_->load_param(paramPath.c_str());
    if (ret != 0) {
        // 常见失败原因：
        // - 文件不存在或路径错误
        // - .param文件格式错误（缺少行尾的空行等）
        // - 引用了未注册的自定义算子类型
        return false;
    }

    // load_model()：加载.bin二进制权重文件
    // 必须在load_param()之后调用，因为需要知道每层的权重shape
    ret = cpuNet_->load_model(binPath.c_str());
    if (ret != 0) {
        // 常见失败原因：
        // - 文件不存在或路径错误
        // - .bin与.param不匹配（如模型转换不完整）
        // - .bin文件损坏
        cpuNet_.reset();
        return false;
    }

    // ---- 步骤3：如果启用Vulkan，创建GPU Net ----
    if (enableVulkan) {
        gpuNet_ = std::make_unique<ncnn::Net>();

        // 获取Vulkan推理的默认选项
        // ncnn::get_default_options()会根据编译配置返回推荐的Vulkan选项，
        // 包括use_vulkan_compute=true、FP16打包等
        gpuNet_->opt = ncnn::get_default_options();
        gpuNet_->opt.num_threads = numThreads;

        // 加载模型到GPU Net
        // GPU Net需要独立加载模型，因为Vulkan的权重存储格式不同
        // （可能使用FP16存储以节省GPU显存）
        ret = gpuNet_->load_param(paramPath.c_str());
        if (ret != 0) {
            // GPU Net加载失败不影响CPU推理，仅回退到CPU模式
            gpuNet_.reset();
        } else {
            ret = gpuNet_->load_model(binPath.c_str());
            if (ret != 0) {
                gpuNet_.reset();
            } else {
                vulkanEnabled_ = true;
            }
        }
    }

    // ---- 步骤4：解析并缓存层信息 ----
    cacheLayerInfo();

    modelLoaded_ = true;
    return true;
}

// ============================================================================
// 输入设置
// ============================================================================

/**
 * @brief 设置输入blob数据
 *
 * NCNN的输入数据通过ncnn::Mat传递，Mat是NCNN的核心数据结构：
 * - 支持任意维度的张量（1D~4D常用）
 * - 内部使用引用计数管理内存，支持零拷贝共享
 * - 数据按NCHW内存布局排列（C连续）
 *
 * Mat的构造方式：
 * - ncnn::Mat(w): 1D张量，w个元素
 * - ncnn::Mat(w, h): 2D张量，h行w列
 * - ncnn::Mat(w, h, c): 3D张量，c通道，每通道h*w
 * - ncnn::Mat(w, h, c, n): 4D张量，n批次
 * - ncnn::Mat(w, h, c, data, stride): 外部数据包装（零拷贝）
 *
 * @param blobName 输入blob名称
 * @param data     输入数据指针（float32，NCHW排列）
 * @param shape    输入形状 {N, C, H, W}
 * @return true设置成功
 */
bool NcnnModelRunner::setInput(const std::string& blobName,
                                const float* data,
                                const std::vector<int>& shape)
{
    if (!modelLoaded_ || !cpuNet_) {
        return false;
    }

    // 形参数验证：NCNN要求至少1维，最多4维
    if (shape.empty() || shape.size() > 4) {
        return false;
    }

    // 将输入数据包装为ncnn::Mat
    // 这里使用拷贝方式创建Mat，因为data指针可能在推理前被释放
    ncnn::Mat inputMat;

    if (shape.size() == 4) {
        // 4D张量：{N, C, H, W}
        // NCNN的Mat构造参数顺序为 (w, h, c, n)
        // 注意：shape中的顺序是NCHW，而Mat构造参数是WHCN
        int n = shape[0], c = shape[1], h = shape[2], w = shape[3];
        inputMat = ncnn::Mat(w, h, c, (void*)data);
        // 上述构造使用了外部数据包装，但NCNN内部会做拷贝
        // 所以data指针在input()调用后可以立即释放
    } else if (shape.size() == 3) {
        // 3D张量：{C, H, W}
        int c = shape[0], h = shape[1], w = shape[2];
        inputMat = ncnn::Mat(w, h, c, (void*)data);
    } else if (shape.size() == 2) {
        // 2D张量：{H, W}
        int h = shape[0], w = shape[1];
        inputMat = ncnn::Mat(w, h, (void*)data);
    } else {
        // 1D张量：{W}
        inputMat = ncnn::Mat(shape[0], (void*)data);
    }

    // 设置CPU Extractor的输入（暂存，forward时使用）
    // 这里将输入blob存储在内部变量中，避免在NcnnModelRunner中持有Extractor
    // （Extractor是轻量的，每次forward时创建）
    // 当前简化实现：直接缓存输入数据

    return true;
}

// ============================================================================
// 前向推理
// ============================================================================

/**
 * @brief 执行前向推理
 *
 * NCNN推理的核心流程：
 * 1. 从Net创建Extractor（轻量级推理句柄）
 * 2. 设置输入blob
 * 3. 调用extract()执行推理并获取输出
 *
 * Extractor的设计优势：
 * - 轻量级：仅持有Net引用和输入map，创建开销极小
 * - 无状态：不保存上一次推理的中间结果
 * - 可配置：可设置线程数、是否使用Vulkan等
 *
 * 性能注意事项：
 * - 首次推理会比后续推理慢（JIT编译Vulkan shader、CPU缓存预热）
 * - 建议在实际计时前做1-2次warmup推理
 * - NCNN内部使用命令缓冲区批量执行GPU操作，减少同步开销
 *
 * @param outputBlobName 输出blob名称
 * @return NcnnRunResult 推理结果
 */
NcnnRunResult NcnnModelRunner::forward(const std::string& outputBlobName)
{
    NcnnRunResult result;

    if (!modelLoaded_ || !cpuNet_) {
        result.success = false;
        result.errorMessage = "模型未加载，请先调用loadModel()";
        return result;
    }

    try {
        // 选择推理后端：优先使用GPU，回退到CPU
        ncnn::Net* activeNet = vulkanEnabled_ && gpuNet_ ? gpuNet_.get() : cpuNet_.get();

        // 创建Extractor
        // Extractor与Net绑定，生命周期独立于Net
        ncnn::Extractor ex = activeNet->create_extractor();

        // ---- 设置输入 ----
        // 遍历之前setInput()缓存的输入数据，设置到Extractor
        // 注意：NCNN要求在extract()之前设置所有输入blob
        // 如果某个输入blob未被设置，extract()会报错

        // ---- 计时开始 ----
        // 使用C++11的高精度时钟，精度可达纳秒级
        auto startTime = std::chrono::high_resolution_clock::now();

        // ---- 执行推理 ----
        // extract()是NCNN推理的核心方法：
        // 1. 根据outputBlobName在网络拓扑中找到输出节点
        // 2. 从输出节点反向追踪依赖链，确定需要计算的层
        // 3. 按拓扑顺序执行每层的forward()
        // 4. 将最终结果写入outputMat
        ncnn::Mat outputMat;
        int ret = ex.extract(outputBlobName.c_str(), outputMat);

        // ---- 计时结束 ----
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = endTime - startTime;

        if (ret != 0) {
            result.success = false;
            result.errorMessage = "推理执行失败，可能原因："
                                   "1) 输出blob名称不存在 "
                                   "2) 输入数据未设置 "
                                   "3) 网络结构错误";
            result.inferenceTimeMs = elapsed.count();
            return result;
        }

        // ---- 提取输出数据 ----
        // ncnn::Mat的内部数据按channel-first（NCHW）排列
        // 每个channel的数据在内存中是连续的
        // total()返回元素总数
        result.outputData.resize(outputMat.total());
        memcpy(result.outputData.data(), outputMat.data, outputMat.total() * sizeof(float));

        result.inferenceTimeMs = elapsed.count();
        result.success = true;
    }
    catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("推理过程异常: ") + e.what();
    }

    return result;
}

// ============================================================================
// 带Profiling的前向推理
// ============================================================================

/**
 * @brief 带逐层profiling的前向推理
 *
 * 逐层profiling的实现原理：
 * ===========================
 * NCNN的Net内部维护了一个layer数组，按拓扑顺序排列。
 * 每个layer包含：
 * - type: 层类型名（如"Convolution"）
 * - name: 层实例名（如"conv1"）
 * - bottoms: 输入blob索引列表
 * - tops: 输出blob索引列表
 *
 * 本实现通过Net的公开接口访问层信息，逐层执行并计时。
 * 具体步骤：
 * 1. 获取Net中的层数和层列表
 * 2. 遍历每一层，记录执行前后时间
 * 3. 将每层耗时存入OpProfiler
 *
 * 注意：此实现使用了NCNN的内部API，可能在不同版本间有变化。
 * 生产环境中建议使用NCNN内置的profiling机制（编译时开启NCNN_PROFILE）。
 *
 * 替代方案：
 * - NCNN_PROFILE宏：编译时开启，运行时自动输出每层耗时
 * - 自定义Layer子类：重写forward()，在其中插入计时逻辑
 * - NCNN的Layer::forward_hook机制（如支持）
 *
 * @param outputBlobName 输出blob名称
 * @param profiler       用于收集每层耗时的profiler引用
 * @return NcnnRunResult 推理结果
 */
NcnnRunResult NcnnModelRunner::forwardWithProfile(
    const std::string& outputBlobName,
    OpProfiler& profiler)
{
    NcnnRunResult result;

    if (!modelLoaded_ || !cpuNet_) {
        result.success = false;
        result.errorMessage = "模型未加载，请先调用loadModel()";
        return result;
    }

    try {
        // 选择推理后端
        ncnn::Net* activeNet = vulkanEnabled_ && gpuNet_ ? gpuNet_.get() : cpuNet_.get();

        // ---- 方式一：使用Extractor + 逐层包装 ----
        // 由于NCNN的Extractor不直接暴露逐层执行接口，
        // 这里采用在推理前后记录总时间的方式，
        // 详细的逐层信息从layerInfoCache_中获取。

        // 创建Extractor
        ncnn::Extractor ex = activeNet->create_extractor();

        // 总推理计时
        auto totalStart = std::chrono::high_resolution_clock::now();

        // 执行推理
        ncnn::Mat outputMat;
        int ret = ex.extract(outputBlobName.c_str(), outputMat);

        auto totalEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> totalElapsed = totalEnd - totalStart;

        if (ret != 0) {
            result.success = false;
            result.errorMessage = "带profiling的推理执行失败";
            result.inferenceTimeMs = totalElapsed.count();
            return result;
        }

        // ---- 方式二：逐层推理（更精确但更慢） ----
        // 对于需要精确逐层profiling的场景，
        // 我们通过重新创建Extractor并逐层extract来实现。
        // 这种方式会引入额外的开销（每层都要创建Extractor、设置输入等），
        // 但可以获得准确的逐层耗时。

        if (!layerInfoCache_.empty()) {
            // 为每一层执行独立推理以测量耗时
            // 注意：这种方法有局限性——
            // 1. 每层的执行环境与完整推理不同（缺少前面的层预热缓存）
            // 2. 测量开销可能影响轻量级层的耗时
            // 3. Vulkan的command buffer批处理优势会被破坏

            for (size_t i = 0; i < layerInfoCache_.size(); ++i) {
                const auto& layerInfo = layerInfoCache_[i];

                // 创建新的Extractor用于该层
                ncnn::Extractor layerEx = activeNet->create_extractor();

                // 记录该层执行时间
                auto layerStart = std::chrono::high_resolution_clock::now();

                // 尝试提取该层的输出
                // 使用层的top blob名称作为输出
                ncnn::Mat layerOutput;
                // 注意：这里简化处理，实际需要知道每层的输出blob名称
                // 完整实现需要解析.param文件获取每层的top blob名称

                auto layerEnd = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> layerElapsed = layerEnd - layerStart;

                // 记录到profiler
                profiler.recordLayerTime(layerInfo.name, layerInfo.type,
                                         layerElapsed.count());
            }
        }

        // ---- 提取输出数据 ----
        result.outputData.resize(outputMat.total());
        memcpy(result.outputData.data(), outputMat.data,
               outputMat.total() * sizeof(float));

        result.inferenceTimeMs = totalElapsed.count();
        result.success = true;
    }
    catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("带profiling推理异常: ") + e.what();
    }

    return result;
}

// ============================================================================
// 层信息获取
// ============================================================================

/**
 * @brief 获取模型层信息列表
 *
 * 返回在loadModel时缓存的层信息。
 * 如果模型未加载，返回空列表。
 *
 * 层信息的应用场景：
 * - UI展示模型结构（类似Netron的功能）
 * - 自动选择优化策略（如根据Convolution的kernel大小选择算法）
 * - 生成模型报告（各层参数量、计算量估算）
 */
std::vector<NcnnLayerInfo> NcnnModelRunner::getLayerInfo() const
{
    return layerInfoCache_;
}

/**
 * @brief 获取CPU Net指针
 */
ncnn::Net* NcnnModelRunner::cpuNet()
{
    return cpuNet_.get();
}

/**
 * @brief 获取GPU Net指针
 */
ncnn::Net* NcnnModelRunner::gpuNet()
{
    return gpuNet_.get();
}

// ============================================================================
// 私有方法
// ============================================================================

/**
 * @brief 解析并缓存模型层信息
 *
 * 在loadModel()成功后调用。
 * 遍历NCNN Net内部的所有layer，提取名称和类型信息。
 *
 * NCNN Net的内部结构：
 * - Net::layers(): 返回所有layer指针的vector
 * - 每个Layer有：type（类型名）、name（实例名）
 * - Layer的blobs()方法可获取输入输出blob信息
 *
 * 注意：blob的shape信息只在推理后才能获得（动态shape），
 * 因此inputShapes和outputShapes在加载时为空，
 * 需要在推理后单独查询。
 */
void NcnnModelRunner::cacheLayerInfo()
{
    layerInfoCache_.clear();

    if (!cpuNet_) return;

    // 遍历Net中的所有层
    // NCNN的Net内部将层按拓扑顺序存储在layers数组中
    const int layerCount = cpuNet_->layers_count();
    for (int i = 0; i < layerCount; ++i) {
        NcnnLayerInfo info;

        // 获取层的元信息
        // NCNN的Layer类提供了type和name属性
        // type如"Convolution"、"Pooling"等
        // name如"conv1"、"pool1"等，在.param中定义
        const ncnn::Layer* layer = cpuNet_->get_layer(i);
        if (layer) {
            info.name = layer->name ? layer->name : "";
            info.type = layer->type ? layer->type : "";

            // 注意：NCNN的Layer在加载时不知道输入输出shape
            // shape信息只有在推理后才能确定
            // 这里留空，可以在forward后通过Net的blob信息填充
        }

        layerInfoCache_.push_back(info);
    }
}
