/**
 * @file NcnnModelRunner.h
 * @brief NCNN模型加载与推理执行器
 *
 * 本头文件定义了NcnnModelRunner类，负责：
 * - 加载NCNN格式模型（.param + .bin）
 * - 支持CPU和Vulkan GPU两种推理后端
 * - 提供普通推理和逐层profiling推理两种模式
 * - 管理输入blob的设置与输出blob的提取
 *
 * NCNN是腾讯优图实验室开源的高性能神经网络前向计算框架，
 * 针对移动端和嵌入式设备做了深度优化，无第三方依赖，
 * 支持Vulkan GPU加速，适合在边缘设备上部署AI模型。
 */

#ifndef NCNN_EXT_NCNN_MODEL_RUNNER_H
#define NCNN_EXT_NCNN_MODEL_RUNNER_H

#ifdef USE_NCNN

#include <string>
#include <vector>
#include <memory>

// 前向声明ncnn类型，避免在头文件中引入ncnn头文件依赖
// 这是一种常见的C++ Pimpl（Pointer to Implementation）手法，
// 可以减少编译依赖、加快编译速度，同时避免头文件污染
namespace ncnn { class Net; class Mat; }

/**
 * @struct NcnnRunResult
 * @brief 推理执行结果
 *
 * 封装了NCNN推理的所有输出信息：
 * - outputData: 输出张量展平后的一维数据
 * - inferenceTimeMs: 推理耗时（毫秒），用于性能分析
 * - success: 推理是否成功
 * - errorMessage: 失败时的错误描述
 */
struct NcnnRunResult {
    std::vector<float> outputData;   ///< 输出张量展平数据
    double inferenceTimeMs = 0.0;    ///< 推理耗时(毫秒)
    bool success = false;             ///< 是否推理成功
    std::string errorMessage;         ///< 失败原因描述
};

/**
 * @struct NcnnLayerInfo
 * @brief 模型中单层的元信息
 *
 * 用于描述NCNN模型中每一层的名称、类型和形状信息，
 * 辅助模型结构分析和性能优化决策。
 * NCNN模型由一系列有序的layer组成，每个layer有：
 * - name: 层实例名称（在.param文件中定义，全局唯一）
 * - type: 层类型名（如Convolution、Pooling、ReLU等）
 * - inputShapes: 输入blob的维度（NCNN默认NCHW格式）
 * - outputShapes: 输出blob的维度
 */
struct NcnnLayerInfo {
    std::string name;                 ///< 层实例名称
    std::string type;                 ///< 层类型名
    std::vector<int> inputShapes;     ///< 输入blob维度
    std::vector<int> outputShapes;    ///< 输出blob维度
};

// 前向声明，避免头文件循环依赖
class OpProfiler;

/**
 * @class NcnnModelRunner
 * @brief NCNN模型加载与推理执行器
 *
 * 核心职责：
 * 1. 模型生命周期管理：加载.param/.bin文件，管理CPU和GPU两个Net实例
 * 2. 推理执行：设置输入、执行前向计算、提取输出
 * 3. 性能Profiling：支持逐层耗时统计，帮助定位推理瓶颈
 *
 * 设计要点：
 * - CPU和GPU Net分开管理：NCNN的Vulkan后端需要单独创建Net实例，
 *   且必须在load_model之前设置opt.use_vulkan_compute = true
 * - 使用unique_ptr管理Net生命周期，确保资源正确释放
 * - 线程安全：当前设计为单线程使用，如需多线程推理需每线程独立Net实例
 */
class NcnnModelRunner {
public:
    NcnnModelRunner();
    ~NcnnModelRunner();

    // 禁止拷贝，允许移动
    // NCNN的Net类内部持有GPU资源（Vulkan command buffer等），不支持拷贝
    NcnnModelRunner(const NcnnModelRunner&) = delete;
    NcnnModelRunner& operator=(const NcnnModelRunner&) = delete;
    NcnnModelRunner(NcnnModelRunner&&) noexcept;
    NcnnModelRunner& operator=(NcnnModelRunner&&) noexcept;

    /**
     * @brief 加载NCNN模型
     *
     * @param paramPath  NCNN参数文件路径（.param），描述网络结构
     *                   .param文件是文本格式，包含每层的类型、名称、
     *                   输入输出blob连接关系和超参数
     * @param binPath    NCNN权重文件路径（.bin），存储模型权重二进制数据
     *                   .bin文件是二进制格式，包含所有卷积核、BN参数等
     * @param enableVulkan 是否启用Vulkan GPU加速
     *                     true: 创建GPU后端Net，利用Vulkan Compute Shader加速
     *                     false: 仅使用CPU后端
     *                     注意：Vulkan加速需要编译时开启NCNN_VULKAN宏，
     *                     且运行时需要有支持Vulkan 1.1+的GPU驱动
     * @param numThreads CPU推理线程数
     *                   NCNN内部使用OpenMP进行多线程加速，
     *                   建议设为物理核心数，过多线程可能导致缓存抖动
     * @return true 加载成功，false 加载失败（文件不存在、格式错误等）
     */
    bool loadModel(const std::string& paramPath,
                   const std::string& binPath,
                   bool enableVulkan = false,
                   int numThreads = 4);

    /**
     * @brief 设置输入blob数据
     *
     * NCNN使用blob名称（字符串）来标识输入输出，
     * 这与TensorFlow的placeholder概念类似。
     * 输入数据必须是float32格式，shape遵循NCHW维度顺序。
     *
     * @param blobName  输入blob名称，需与.param文件中定义的名称一致
     *                  常见名称如"data"、"input"、"input.1"等
     * @param data      输入数据指针，指向float数组
     *                  数据按NCHW顺序排列（C连续内存布局）
     * @param shape     输入张量形状，{N, C, H, W}格式
     *                  - N: batch大小，NCNN通常为1
     *                  - C: 通道数（如RGB图像为3）
     *                  - H: 高度
     *                  - W: 宽度
     * @return true 设置成功，false 失败（模型未加载、blob不存在等）
     */
    bool setInput(const std::string& blobName, const float* data,
                  const std::vector<int>& shape);

    /**
     * @brief 执行前向推理
     *
     * 调用NCNN的Extractor完成一次完整的网络前向计算。
     * NCNN的Extractor是轻量级的推理句柄，每次推理创建新的Extractor，
     * 保证不同推理之间不共享内部状态。
     *
     * @param outputBlobName 输出blob名称，需与.param文件中定义的名称一致
     *                        常见名称如"output"、"prob"、"out0"等
     * @return NcnnRunResult 推理结果，包含输出数据和耗时
     */
    NcnnRunResult forward(const std::string& outputBlobName);

    /**
     * @brief 带逐层profiling的前向推理
     *
     * 与forward()功能相同，但额外记录每一层的执行时间。
     * 实现方式：使用NCNN的custom_layer回调机制，在每个layer执行前后
     * 记录时间戳，计算差值即为该层耗时。
     *
     * 逐层profiling的价值：
     * - 定位推理瓶颈：找出耗时最长的层（通常是Convolution）
     * - 指导量化决策：哪些层对精度敏感不适合INT8量化
     * - 对比CPU/GPU性能：同一层在不同后端的表现差异
     *
     * @param outputBlobName 输出blob名称
     * @param profiler       OpProfiler引用，用于收集每层耗时
     * @return NcnnRunResult 推理结果
     */
    NcnnRunResult forwardWithProfile(const std::string& outputBlobName,
                                     OpProfiler& profiler);

    /**
     * @brief 获取模型层信息列表
     *
     * 遍历NCNN Net中的所有layer，提取名称、类型和形状信息。
     * 该信息在loadModel后缓存，后续调用直接返回缓存结果。
     *
     * @return 层信息列表，按网络拓扑顺序排列
     */
    std::vector<NcnnLayerInfo> getLayerInfo() const;

    /**
     * @brief 获取CPU后端Net指针
     * @return CPU Net指针，模型未加载时返回nullptr
     */
    ncnn::Net* cpuNet();

    /**
     * @brief 获取GPU后端Net指针
     * @return GPU Net指针，未启用Vulkan时返回nullptr
     */
    ncnn::Net* gpuNet();

    /**
     * @brief 模型是否已加载
     * @return true 已加载，false 未加载
     */
    bool isLoaded() const { return modelLoaded_; }

private:
    std::unique_ptr<ncnn::Net> cpuNet_;     ///< CPU推理后端Net实例
    std::unique_ptr<ncnn::Net> gpuNet_;      ///< Vulkan GPU推理后端Net实例
    bool vulkanEnabled_ = false;             ///< 是否启用了Vulkan加速
    bool modelLoaded_ = false;               ///< 模型加载状态标志
    std::vector<NcnnLayerInfo> layerInfoCache_;  ///< 层信息缓存

    /**
     * @brief 内部方法：解析模型层信息并缓存
     *
     * 在loadModel成功后调用，遍历Net中的所有layer，
     * 提取元信息并存储到layerInfoCache_中。
     */
    void cacheLayerInfo();
};

#endif // USE_NCNN

#endif // NCNN_EXT_NCNN_MODEL_RUNNER_H
