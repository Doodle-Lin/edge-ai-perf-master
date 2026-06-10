/**
 * @file VulkanConv2d.cpp
 * @brief Vulkan GPU 加速的 2D 卷积算子（融合 im2col + GEMM 方案）
 *
 * ====== 实现说明 ======
 * 1. 使用 conv2d.comp.spv 着色器直接在 GPU 上完成卷积
 * 2. 每个线程计算一个输出元素（oc, oh, ow），直接读取输入像素并累加
 * 3. 权重在 CPU 端准备好后传给 GPU，卷积参数通过 push constants 传递
 * 4. 所有代码包裹在 #ifdef ENABLE_VULKAN 中
 */

#include "operators/VulkanConv2d.h"

#ifdef ENABLE_VULKAN

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

namespace edgeai {

VulkanConv2d::VulkanConv2d(int inC, int outC, int kH, int kW,
                             int stride, int pad, int groups,
                             const std::string& spvPath)
    : inC_(inC), outC_(outC), kH_(kH), kW_(kW),
      stride_(stride), pad_(pad), groups_(groups), spvPath_(spvPath)
{
    assert(inC % groups == 0 && "inC must be divisible by groups");
    assert(outC % groups == 0 && "outC must be divisible by groups");

    int icPerGroup = inC / groups;
    weight_ = Tensor({outC, icPerGroup, kH, kW});
    bias_ = Tensor({outC});

    // Xavier uniform initialization
    float fanIn = static_cast<float>(icPerGroup * kH * kW);
    float fanOut = static_cast<float>((outC / groups) * kH * kW);
    float limit = std::sqrt(6.0f / (fanIn + fanOut));
    weight_.randomize(-limit, limit);
    bias_.zero();

    // Initialize Vulkan
    if (!vkBase_.initialize()) {
        std::cerr << "[VulkanConv2d] Vulkan initialization failed" << std::endl;
        return;
    }

    // Load conv2d compute shader
    if (!vkBase_.loadShader(spvPath_)) {
        std::cerr << "[VulkanConv2d] Failed to load shader: " << spvPath_ << std::endl;
        return;
    }

    // Create GPU buffers:
    // Binding 0: input [inC, H, W] (written each forward call)
    // Binding 1: weight [outC, icPerGroup, kH, kW] (written once)
    // Binding 2: bias [outC] (written once)
    // Binding 3: output [outC, outH, outW] (read back each forward call)
    size_t inputSize = static_cast<size_t>(inC) * 1 * 1 * sizeof(float);   // placeholder
    size_t weightSize = static_cast<size_t>(outC) * icPerGroup * kH * kW * sizeof(float);
    size_t biasSize = static_cast<size_t>(outC) * sizeof(float);
    size_t outputSize = static_cast<size_t>(outC) * 1 * 1 * sizeof(float);  // placeholder

    // Use maximum possible sizes for buffers (actual size determined at dispatch time)
    // For simplicity, we use generous buffer sizes
    const size_t MAX_SPATIAL = 1024 * 1024;  // Max spatial elements
    inputSize = static_cast<size_t>(inC) * MAX_SPATIAL * sizeof(float);
    outputSize = static_cast<size_t>(outC) * MAX_SPATIAL * sizeof(float);

    if (!vkBase_.createBuffer(inputSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        std::cerr << "[VulkanConv2d] Failed to create input buffer" << std::endl;
        return;
    }
    if (!vkBase_.createBuffer(weightSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        std::cerr << "[VulkanConv2d] Failed to create weight buffer" << std::endl;
        return;
    }
    if (!vkBase_.createBuffer(biasSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        std::cerr << "[VulkanConv2d] Failed to create bias buffer" << std::endl;
        return;
    }
    if (!vkBase_.createBuffer(outputSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        std::cerr << "[VulkanConv2d] Failed to create output buffer" << std::endl;
        return;
    }

    // Create compute pipeline
    if (!vkBase_.createPipeline()) {
        std::cerr << "[VulkanConv2d] Failed to create pipeline" << std::endl;
        return;
    }

    // Write weight and bias to GPU (they don't change between forward calls)
    vkBase_.writeBuffer(1, weight_.data(), weightSize);
    vkBase_.writeBuffer(2, bias_.data(), biasSize);

    initialized_ = true;
}

void VulkanConv2d::forward(const Tensor& input, Tensor& output)
{
    assert(initialized_ && "Vulkan not initialized");
    assert(input.dims() == 4);

    int N = input.size(0);
    int H = input.size(2);
    int W = input.size(3);

    int outH = (H + 2 * pad_ - kH_) / stride_ + 1;
    int outW = (W + 2 * pad_ - kW_) / stride_ + 1;

    output = Tensor({N, outC_, outH, outW});

    int icPerGroup = inC_ / groups_;

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < groups_; ++g) {
            int ocPerGroup = outC_ / groups_;
            int ocStart = g * ocPerGroup;
            int icStart = g * icPerGroup;

            // Prepare input for this group: [icPerGroup, H, W]
            // For simplicity, we extract the group's input channels
            // In a production implementation, the shader would handle groups directly
            size_t inputSize = static_cast<size_t>(icPerGroup) * H * W * sizeof(float);

            // Write group input to GPU buffer 0
            // We need to extract the relevant input channels
            Tensor groupInput({1, icPerGroup, H, W});
            for (int ic = 0; ic < icPerGroup; ++ic) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        groupInput.at(0, ic, h, w) = input.at(n, icStart + ic, h, w);
                    }
                }
            }
            vkBase_.writeBuffer(0, groupInput.data(), inputSize);

            // Dispatch compute: workgroup (16, 16, 1)
            // Each invocation computes one (oh, ow) for a given oc
            // Total invocations: outC * outH * outW
            // We dispatch: groupX = (outH * outW + 15) / 16, groupY = (ocPerGroup + 15) / 16
            uint32_t totalOutput = static_cast<uint32_t>(ocPerGroup * outH * outW);
            uint32_t groupX = (totalOutput + 255) / 256;  // Approximate dispatch
            uint32_t groupY = 1;

            vkBase_.dispatch(groupX, groupY, 1);

            // Read result from GPU buffer 3
            size_t outputSize = static_cast<size_t>(ocPerGroup) * outH * outW * sizeof(float);
            Tensor groupOutput({1, ocPerGroup, outH, outW});
            vkBase_.readBuffer(3, groupOutput.data(), outputSize);

            // Copy group output to final output tensor
            for (int oc = 0; oc < ocPerGroup; ++oc) {
                int ocGlobal = ocStart + oc;
                for (int oh = 0; oh < outH; ++oh) {
                    for (int ow = 0; ow < outW; ++ow) {
                        output.at(n, ocGlobal, oh, ow) = groupOutput.at(0, oc, oh, ow);
                    }
                }
            }
        }
    }
}

int64_t VulkanConv2d::flops(const std::vector<int>& inputShape,
                               const std::vector<int>& outputShape) const
{
    if (inputShape.size() < 4 || outputShape.size() < 4) return 0;

    int64_t N    = inputShape[0];
    int64_t outH = outputShape[2];
    int64_t outW = outputShape[3];
    int64_t icPerGroup = inC_ / groups_;

    int64_t macsPerElement = static_cast<int64_t>(kH_) * kW_ * icPerGroup;
    int64_t outputElements = N * outC_ * outH * outW;

    return outputElements * macsPerElement * 2;
}

} // namespace edgeai

#endif // ENABLE_VULKAN
