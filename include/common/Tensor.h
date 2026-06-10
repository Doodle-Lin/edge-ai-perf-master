/**
 * @file Tensor.h
 * @brief 张量类 —— 所有算子的统一数据容器
 *
 * ====== 学习要点 ======
 * 1. 张量 (Tensor) 是深度学习中数据的基本表示形式，可以理解为多维数组
 *    - 0维：标量 (scalar)
 *    - 1维：向量 (vector)
 *    - 2维：矩阵 (matrix)
 *    - 4维：图像批次 (batch, channels, height, width) — 本项目主要使用这种
 *
 * 2. 内存布局：采用行优先 (row-major) 连续存储
 *    - 对于 4D 张量 [N, C, H, W]，元素 (n,c,h,w) 的偏移量为：
 *      offset = n * C*H*W + c * H*W + h * W + w
 *    - 连续存储意味着最后一维（W）在内存中是相邻的，这对缓存友好
 *
 * 3. 为什么不直接用 std::vector<std::vector<...>>？
 *    - 多层嵌套 vector 导致内存不连续，每次访问都要间接寻址
 *    - 连续内存可以：① 利用 CPU 缓存行预取 ② 传给 SIMD 指令（如 AVX2 要求 32 字节对齐）
 *    - 连续内存可以：③ 直接传给 GPU（Vulkan buffer），无需额外拷贝
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <numeric>
#include <vector>

namespace edgeai {

class Tensor {
public:
    // ──────────────── 构造 / 析构 ────────────────

    /// 默认构造：空张量
    Tensor() = default;

    /// 按形状构造，元素初始化为 0
    /// @param shape 各维度大小，如 {1, 3, 224, 224}
    explicit Tensor(const std::vector<int>& shape)
        : shape_(shape)
    {
        int total = totalElements();
        data_.resize(total, 0.0f);
    }

    /// 按形状 + 填充值构造
    Tensor(const std::vector<int>& shape, float fillValue)
        : shape_(shape)
    {
        data_.resize(totalElements(), fillValue);
    }

    // ──────────────── 形状访问 ────────────────

    /// 获取形状（各维度大小）
    const std::vector<int>& shape() const { return shape_; }

    /// 获取第 dim 维的大小
    int size(int dim) const {
        assert(dim >= 0 && dim < static_cast<int>(shape_.size()));
        return shape_[dim];
    }

    /// 维度数量（如 4 维张量返回 4）
    int dims() const { return static_cast<int>(shape_.size()); }

    /// 总元素数 = shape[0] * shape[1] * ... * shape[n-1]
    int total() const { return totalElements(); }

    // ──────────────── 数据访问 ────────────────

    /// 获取原始数据指针（只读）
    const float* data() const { return data_.data(); }

    /// 获取原始数据指针（可写）
    float* data() { return data_.data(); }

    /// 按 4D 索引访问元素 (n, c, h, w)
    /// 内存偏移 = n * C*H*W + c * H*W + h * W + w
    float& at(int n, int c, int h, int w) {
        assert(shape_.size() == 4);
        assert(n >= 0 && n < shape_[0]);
        assert(c >= 0 && c < shape_[1]);
        assert(h >= 0 && h < shape_[2]);
        assert(w >= 0 && w < shape_[3]);
        return data_[n * shape_[1] * shape_[2] * shape_[3]
                   + c * shape_[2] * shape_[3]
                   + h * shape_[3]
                   + w];
    }

    const float& at(int n, int c, int h, int w) const {
        assert(shape_.size() == 4);
        assert(n >= 0 && n < shape_[0]);
        assert(c >= 0 && c < shape_[1]);
        assert(h >= 0 && h < shape_[2]);
        assert(w >= 0 && w < shape_[3]);
        return data_[n * shape_[1] * shape_[2] * shape_[3]
                   + c * shape_[2] * shape_[3]
                   + h * shape_[3]
                   + w];
    }

    /// 按一维偏移量访问（用于展平操作）
    float& operator[](int idx) {
        assert(idx >= 0 && idx < static_cast<int>(data_.size()));
        return data_[idx];
    }

    const float& operator[](int idx) const {
        assert(idx >= 0 && idx < static_cast<int>(data_.size()));
        return data_[idx];
    }

    // ──────────────── 工具方法 ────────────────

    /// 用随机值填充（均匀分布 [-1, 1]）
    /// 用于测试时生成模拟输入
    void randomize(float lo = -1.0f, float hi = 1.0f) {
        for (auto& v : data_) {
            v = lo + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (hi - lo);
        }
    }

    /// 将所有元素设为 0
    void zero() {
        std::memset(data_.data(), 0, data_.size() * sizeof(float));
    }

    /// 改变形状（总元素数必须一致）
    /// 注意：这只是重新解释内存布局，不改变实际数据
    void reshape(const std::vector<int>& newShape) {
        int newTotal = 1;
        for (int s : newShape) newTotal *= s;
        (void)newTotal; // 消除未使用变量警告
        assert(newTotal == totalElements());
        shape_ = newShape;
    }

private:
    /// 计算总元素数
    int totalElements() const {
        if (shape_.empty()) return 0;
        int total = 1;
        for (int s : shape_) total *= s;
        return total;
    }

    std::vector<int> shape_;    ///< 各维度大小
    std::vector<float> data_;   ///< 连续存储的浮点数据
};

} // namespace edgeai
