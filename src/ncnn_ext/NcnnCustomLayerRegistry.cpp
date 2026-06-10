/**
 * @file NcnnCustomLayerRegistry.cpp
 * @brief NCNN自定义算子注册表的实现
 *
 * 本文件实现了NcnnCustomLayerRegistry类的所有方法，包括：
 * - 单例实例管理
 * - 自定义算子的注册与应用
 * - 查询接口
 *
 * NCNN自定义算子注册机制详解：
 * ===============================
 *
 * NCNN支持通过net.register_custom_layer()方法注册自定义算子。
 * 该方法有两个重载版本：
 *
 * 1. 基于索引的注册：
 *    int register_custom_layer(int index, layer_creator_func creator);
 *    - index: 自定义层的类型索引（>= 0x10000，与内置层区分）
 *    - creator: 工厂函数，返回ncnn::Layer*实例
 *
 * 2. 基于类型名的注册（NCNN后续版本添加）：
 *    int register_custom_layer(const char* type, layer_creator_func creator);
 *    - type: 层类型名称字符串，需与.param文件中一致
 *    - creator: 工厂函数
 *
 * 本注册表使用基于类型名的注册方式（版本2），
 * 因为它更直观，且与.param文件中的类型名直接对应。
 *
 * layer_creator_func的类型定义：
 *   typedef ncnn::Layer* (*layer_creator_func)();
 *   即一个无参函数指针，返回ncnn::Layer*。
 *
 * NCNN也提供了DEFINE_LAYER_CREATOR宏来简化工厂函数的定义：
 *   DEFINE_LAYER_CREATOR(MyCustomLayer)
 *   展开为：
 *   static ncnn::Layer* MyCustomLayer_layer_creator() { return new MyCustomLayer; }
 *
 * 但本注册表使用std::function + 模板来实现类型擦除，
 * 更灵活且支持lambda表达式。
 */

#include "ncnn_ext/NcnnCustomLayerRegistry.h"

// 引入NCNN核心头文件
#include "net.h"    // ncnn::Net - 需要调用register_custom_layer()
#include "layer.h"  // ncnn::Layer - 自定义算子的基类

#include <iostream>  // 用于日志输出
#include <utility>   // std::pair

// ============================================================================
// 单例实例获取
// ============================================================================

/**
 * @brief 获取注册表单例实例
 *
 * 使用C++11的Meyers' Singleton模式：
 * - 局部静态变量在首次调用时初始化
 * - C++11标准保证初始化是线程安全的
 * - 程序退出时自动析构，无需手动清理
 *
 * 相比传统Singleton模式的优势：
 * - 无需手动delete，避免内存泄漏
 * - 无需pthread_once等平台相关的线程安全机制
 * - 初始化顺序由C++运行时保证
 *
 * @return 注册表引用
 */
NcnnCustomLayerRegistry& NcnnCustomLayerRegistry::instance()
{
    // C++11保证：如果多个线程同时首次调用，
    // 只有一个线程会执行初始化，其他线程会等待
    static NcnnCustomLayerRegistry registry;
    return registry;
}

// ============================================================================
// 应用所有注册的算子到Net
// ============================================================================

/**
 * @brief 将所有已注册的自定义算子应用到NCNN Net
 *
 * 本方法遍历注册表中的所有算子，对每个算子调用net.register_custom_layer()。
 *
 * NCNN的register_custom_layer()内部机制：
 * ==========================================
 * 当NCNN解析.param文件时，遇到一个层声明如：
 *   HardSwish_Custom  hardswish1  1 1 input output
 * NCNN会：
 * 1. 首先在内置层类型表中查找"HardSwish_Custom"
 * 2. 如果未找到，检查自定义层注册表
 * 3. 如果注册表中存在，调用对应的creator函数创建层实例
 * 4. 如果都不存在，报错退出
 *
 * 因此，register_custom_layer()必须在load_param()之前调用！
 *
 * 多次调用的安全性：
 * - 对同一个类型名重复注册是安全的，后注册的会覆盖先注册的
 * - 对同一个Net多次调用applyAll()也是安全的，但冗余
 * - 不同Net实例需要分别调用applyAll()
 *
 * @param net 目标NCNN Net引用
 */
void NcnnCustomLayerRegistry::applyAll(ncnn::Net& net)
{
    // 遍历注册表中所有 <类型名, 创建函数> 对
    for (const auto& entry : creators_) {
        const std::string& layerName = entry.first;
        const CreatorFn& creator = entry.second;

        // 调用NCNN的注册接口
        // register_custom_layer()的第一个参数是层类型名
        // 第二个参数是layer_creator_func函数指针
        //
        // 注意：NCNN的register_custom_layer()接受的是裸函数指针，
        // 而我们的CreatorFn是std::function。
        // 由于std::function可以转换为函数指针（当不捕获时），
        // 这里通过一个中间层来桥接。
        //
        // 实际上，NCNN的register_custom_layer()第二个参数类型是
        // ncnn::Layer* (*)()，即裸函数指针。
        // std::function不能直接转为裸函数指针，
        // 因此我们使用ncnn::create_custom_layer系列机制。

        // ---- 方法：使用NCNN的DEFINE_LAYER_CREATOR模式 ----
        // NCNN提供了net.register_custom_layer(type, creator)的重载，
        // 其中creator可以是函数指针。
        // 但由于std::function无法隐式转为函数指针，
        // 我们使用一个辅助函数来完成桥接。

        // 将std::function包装的creator注册到NCNN
        // NCNN的Net类支持以下注册方式：
        //   net.register_custom_layer(index, creator_func_ptr)
        //   或通过net.register_custom_layer(type_name)配合全局注册

        // 这里使用NCNN的通用注册方式：
        // 通过create_custom_layer全局函数注册，然后Net自动识别

        // 实际实现：使用NCNN支持的自定义层注册API
        // 对于支持类型名注册的NCNN版本：
        int ret = net.register_custom_layer(
            layerName.c_str(),
            // 将std::function包装为NCNN可接受的函数指针
            // 通过lambda + 静态变量的方式实现桥接
            [](void* user_data) -> ncnn::Layer* {
                // user_data指向原始的CreatorFn
                const CreatorFn* fn = static_cast<const CreatorFn*>(user_data);
                return (*fn)();
            },
            // 将CreatorFn的指针作为user_data传递
            static_cast<void*>(const_cast<CreatorFn*>(&creator))
        );

        if (ret != 0) {
            // 注册失败：通常是类型名格式不合法或NCNN版本不支持该注册方式
            std::cerr << "[NcnnCustomLayerRegistry] 警告：注册自定义算子 '"
                      << layerName << "' 失败 (ret=" << ret << ")" << std::endl;
        } else {
            std::cout << "[NcnnCustomLayerRegistry] 已注册自定义算子: "
                      << layerName << std::endl;
        }
    }
}

// ============================================================================
// 查询接口
// ============================================================================

/**
 * @brief 获取已注册的算子名称列表
 *
 * 返回所有已注册算子的类型名称。
 * 列表按map的key顺序排列（即字典序）。
 *
 * 典型用途：
 * - 调试时确认所需算子是否已注册
 * - 日志输出中记录当前支持的算子
 * - 自动化测试中验证注册结果
 *
 * @return 算子名称列表
 */
std::vector<std::string> NcnnCustomLayerRegistry::registeredLayers() const
{
    std::vector<std::string> names;
    names.reserve(creators_.size());

    for (const auto& entry : creators_) {
        names.push_back(entry.first);
    }

    return names;
}

/**
 * @brief 检查指定算子是否已注册
 *
 * 在applyAll()之前调用，可以确认所需的算子是否都已准备好。
 *
 * @param layerName 算子类型名称
 * @return true已注册，false未注册
 */
bool NcnnCustomLayerRegistry::hasLayer(const std::string& layerName) const
{
    return creators_.find(layerName) != creators_.end();
}

/**
 * @brief 清除所有已注册的算子
 *
 * 清除注册表中的所有条目。注意：
 * - 已通过applyAll()应用到Net的注册不受影响
 *   （Net内部维护自己的注册表副本）
 * - 清除后新创建的Net将不会自动获得这些自定义算子
 *
 * 典型用途：
 * - 单元测试的teardown阶段
 * - 动态加载/卸载插件模块时
 */
void NcnnCustomLayerRegistry::clear()
{
    creators_.clear();
}
