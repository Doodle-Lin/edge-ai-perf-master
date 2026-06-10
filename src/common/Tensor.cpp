/**
 * @file Tensor.cpp
 * @brief Tensor 类的非内联方法实现
 *
 * 大部分 Tensor 方法在头文件中内联定义以减少函数调用开销。
 * 此文件包含不适合内联的方法。
 */

#include "common/Tensor.h"

namespace edgeai {

// Tensor 的核心方法（at, operator[], randomize 等）已在头文件中实现。
// 此文件保留用于将来添加需要单独编译的方法，例如：
// - 从文件加载张量
// - GPU 内存相关的拷贝方法
// - 与 NCNN Mat 的互转

} // namespace edgeai
