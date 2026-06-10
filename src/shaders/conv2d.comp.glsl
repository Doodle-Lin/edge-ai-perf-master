#version 450

/// Conv2D Compute Shader
///
/// 每个 invocation 计算一个输出元素 C[batch, oc, oh, ow]
/// 实现标准卷积: C[oc][oh][ow] = sum(IC * KH * KW 的内积)
///
/// ====== 学习要点 ======
/// 1. Conv2D 可以直接在 shader 中实现，不需要先做 im2col
///    - im2col 方法: 在 CPU/GPU 上先展开输入，再调用 GEMM
///    - 直接方法: 在 shader 中嵌套循环遍历 kernel
///    - 直接方法更省内存 (不需要 im2col 的临时矩阵)
///    - 但 GEMM 方法可以利用成熟的 GEMM 优化
///
/// 2. Push Constants 传递卷积参数
///    - 比 Uniform Buffer 更快，适合少量参数
///    - 40 bytes 的参数远低于 128 bytes 限制
///
/// 3. 为什么这个 shader 比简单的 GEMM 慢？
///    - 内层循环 (ic, kh, kw) 对 A 的访问不连续
///    - 每个 thread 要读 kH*kW*inC 个值，内存延迟高
///    - 优化方向: 使用 shared memory 做 tiling (高级版本)
///    - 本版本是教学版，先理解流程，再优化性能

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(binding = 0) readonly buffer BufferInput {
    float input[];
};

layout(binding = 1) readonly buffer BufferWeight {
    float weight[];
};

layout(binding = 2) writeonly buffer BufferOutput {
    float output[];
};

layout(push_constant) uniform PushConstants {
    uint inC;
    uint outC;
    uint inH;
    uint inW;
    uint outH;
    uint outW;
    uint kH;
    uint kW;
    uint stride;
    uint pad;
    uint groups;
} pc;

void main() {
    uint oc = gl_GlobalInvocationID.x;  // 输出通道
    uint spatial = gl_GlobalInvocationID.y;  // 空间位置 = oh * outW + ow

    if (oc >= pc.outC || spatial >= pc.outH * pc.outW) {
        return;
    }

    uint oh = spatial / pc.outW;
    uint ow = spatial % pc.outW;

    // 分组卷积: 确定当前 oc 属于哪个 group，以及对应的输入通道范围
    uint icPerGroup = pc.inC / pc.groups;
    uint ocPerGroup = pc.outC / pc.groups;
    uint g = oc / ocPerGroup;
    uint ocInGroup = oc % ocPerGroup;
    uint icStart = g * icPerGroup;

    float sum = 0.0;

    // 遍历输入通道和卷积核
    for (uint ic = 0; ic < icPerGroup; ic++) {
        for (uint kh = 0; kh < pc.kH; kh++) {
            for (uint kw = 0; kw < pc.kW; kw++) {
                // 计算输入坐标
                int ih = int(oh) * int(pc.stride) - int(pc.pad) + int(kh);
                int iw = int(ow) * int(pc.stride) - int(pc.pad) + int(kw);

                // 边界检查 (padding 区域输出为 0)
                if (ih >= 0 && ih < int(pc.inH) && iw >= 0 && iw < int(pc.inW)) {
                    // 输入: input[0, icStart+ic, ih, iw]
                    float inputValue = input[(icStart + ic) * pc.inH * pc.inW + uint(ih) * pc.inW + uint(iw)];

                    // 权重: weight[oc, ic, kh, kw]
                    float weightValue = weight[oc * icPerGroup * pc.kH * pc.kW
                                               + ic * pc.kH * pc.kW
                                               + kh * pc.kW
                                               + kw];

                    sum += inputValue * weightValue;
                }
            }
        }
    }

    // 输出: output[0, oc, oh, ow]
    output[oc * pc.outH * pc.outW + oh * pc.outW + ow] = sum;
}
