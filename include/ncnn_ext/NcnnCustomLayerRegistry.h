/**
 * @file NcnnCustomLayerRegistry.h
 * @brief NCNN自定义算子注册表
 *
 * 本头文件定义了NcnnCustomLayerRegistry单例类，用于：
 * - 统一管理NCNN自定义算子（Custom Layer）的注册
 * - 将自定义算子批量应用到NCNN Net实例
 * - 提供查询已注册算子的接口
 *
 * NCNN自定义算子机制说明：
 * =========================
 * NCNN内置了约100种常用算子（Convolution、Pooling、ReLU等），
 * 但当模型包含NCNN不支持的算子时（如HardSwish、GELU等新激活函数），
 * 需要通过自定义算子机制扩展。
 *
 * 注册自定义算子的步骤：
 * 1. 继承ncnn::Layer类，实现forward()方法
 * 2. 在注册表中注册：registerLayer<MyCustomLayer>("MyCustom")
 * 3. 将注册应用到Net：applyAll(net)
 *
 * 本注册表的设计优势：
 * - 集中管理：所有自定义算子的注册代码集中在一处，便于维护
 * - 解耦：算子实现与模型加载代码分离，遵循开闭原则
 * - 延迟注册：可以先注册所有算子，在需要时才apply到Net
 */

#ifndef NCNN_EXT_NCNN_CUSTOM_LAYER_REGISTRY_H
#define NCNN_EXT_NCNN_CUSTOM_LAYER_REGISTRY_H

#ifdef USE_NCNN

#include <string>
#include <vector>
#include <functional>
#include <map>

namespace ncnn { class Net; class Layer; }

/**
 * @class NcnnCustomLayerRegistry
 * @brief NCNN自定义算子注册表（单例模式）
 *
 * 使用方式示例：
 * @code
 * // 1. 注册自定义算子（通常在程序初始化时调用）
 * NcnnCustomLayerRegistry::instance().registerLayer<CustomHardSwish>("HardSwish_Custom");
 * NcnnCustomLayerRegistry::instance().registerLayer<CustomGELU>("GELU_Custom");
 *
 * // 2. 在创建Net后，将所有注册的算子应用到Net
 * ncnn::Net net;
 * NcnnCustomLayerRegistry::instance().applyAll(net);
 *
 * // 3. 之后正常加载模型、执行推理
 * net.load_param("model.param");
 * net.load_model("model.bin");
 * @endcode
 *
 * 线程安全说明：
 * - registerLayer()不是线程安全的，应在程序启动阶段（单线程时）调用
 * - applyAll()是只读操作，多线程同时调用是安全的
 *   （前提是不同线程操作不同的Net实例）
 */
class NcnnCustomLayerRegistry {
public:
    /**
     * @brief 获取注册表单例实例
     *
     * 使用Meyers' Singleton模式（C++11保证线程安全的局部静态变量），
     * 无需手动管理生命周期，程序退出时自动析构。
     *
     * @return 注册表引用
     */
    static NcnnCustomLayerRegistry& instance();

    /**
     * @brief 注册自定义算子类型
     *
     * 将自定义算子类型LayerT注册到注册表中，与layerName关联。
     * 如果layerName已存在，将覆盖原有的注册（允许替换）。
     *
     * 模板参数要求：
     * - LayerT必须继承自ncnn::Layer
     * - LayerT必须实现forward(const std::vector<ncnn::Mat>&, std::vector<ncnn::Mat>&)方法
     * - LayerT必须有无参构造函数
     *
     * @tparam LayerT 自定义算子类型，需继承ncnn::Layer
     * @param layerName 算子名称，需与.param文件中该层的type字段一致
     *                  例如.param中写的是"HardSwish_Custom"，
     *                  则注册时也必须用"HardSwish_Custom"
     */
    template<typename LayerT>
    void registerLayer(const std::string& layerName) {
        creators_[layerName] = &creatorThunk<LayerT>;
    }

    /**
     * @brief 将所有已注册的自定义算子应用到NCNN Net
     *
     * 遍历注册表中所有算子，依次调用net.register_custom_layer()。
     * 此方法必须在net.load_param()之前调用，否则NCNN解析.param文件时
     * 遇到未注册的算子类型会报错退出。
     *
     * NCNN的register_custom_layer()内部机制：
     * - 将算子工厂函数注册到Net的layer_creator数组中
     * - 加载.param时，遇到该类型名称会调用工厂函数创建实例
     * - 每个Net实例需要独立注册，注册信息不会跨Net共享
     *
     * @param net 目标NCNN Net引用
     */
    void applyAll(ncnn::Net& net);

    /**
     * @brief 获取已注册的算子名称列表
     *
     * 用于调试和验证，确认所需的自定义算子都已正确注册。
     *
     * @return 算子名称列表（按注册顺序，即map的key顺序）
     */
    std::vector<std::string> registeredLayers() const;

    /**
     * @brief 检查指定算子是否已注册
     * @param layerName 算子名称
     * @return true 已注册，false 未注册
     */
    bool hasLayer(const std::string& layerName) const;

    /**
     * @brief 清除所有已注册的算子
     *
     * 主要用于单元测试中的清理操作，生产环境一般不需要调用。
     */
    void clear();

private:
    // 私有构造，禁止外部创建实例（单例模式）
    NcnnCustomLayerRegistry() = default;
    // 禁止拷贝和移动
    NcnnCustomLayerRegistry(const NcnnCustomLayerRegistry&) = delete;
    NcnnCustomLayerRegistry& operator=(const NcnnCustomLayerRegistry&) = delete;

    /// 算子创建函数类型：返回ncnn::Layer*的工厂函数
    using CreatorFn = std::function<ncnn::Layer*()>;

    /// 算子名称 -> 创建函数的映射表
    std::map<std::string, CreatorFn> creators_;

    /**
     * @brief 算子创建的模板中间函数
     *
     * 将模板化的new操作包装为类型擦除的函数指针，
     * 使得不同类型的算子可以统一存储在map中。
     * 这是Type Erasure（类型擦除）的经典手法：
     * - 模板参数LayerT在编译期确定
     * - 函数指针类型CreatorFn在运行期统一
     * - 调用时通过虚函数恢复多态性
     *
     * @tparam LayerT 具体的自定义算子类型
     * @return 新创建的LayerT实例指针（以ncnn::Layer*返回）
     */
    template<typename LayerT>
    static ncnn::Layer* creatorThunk() { return new LayerT(); }
};

#endif // USE_NCNN

#endif // NCNN_EXT_NCNN_CUSTOM_LAYER_REGISTRY_H
