#version 450

/// GEMM Compute Shader: C[M,N] = A[M,K] * B[K,N]
///
/// 每个 workgroup 处理 16x16 的输出块
/// 每个 invocation 计算一个输出元素 C[row, col]
///
/// ====== 学习要点 ======
/// 1. Vulkan Compute Shader 是 GPU 通用计算的核心
///    - 不像图形管线有顶点/片元阶段，Compute Shader 只有"计算"
///    - 通过 gl_GlobalInvocationID 确定每个线程处理哪个数据
///
/// 2. Workgroup 大小选择 (local_size_x/y/z)
///    - 16x16 = 256 个线程/workgroup，是常见选择
///    - 太小 (如 1x1): GPU 占用率低，无法隐藏延迟
///    - 太大 (如 64x64): 超过硬件限制或占用过多寄存器
///    - 16x16 适配大多数 GPU 的 SIMD 宽度 (NVIDIA: 32, AMD: 64)
///
/// 3. 全局 invocation ID
///    - gl_GlobalInvocationID = gl_WorkGroupID * gl_WorkGroupSize + gl_LocalInvocationID
///    - gl_WorkGroupID: 哪个 workgroup (由 dispatch 指定)
///    - gl_LocalInvocationID: workgroup 内的线程编号
///    - 组合后得到全局唯一 ID，映射到输出矩阵的 (row, col)
///
/// 4. Push Constants vs Uniform Buffer
///    - Push Constants: 小量常量 (<= 128 bytes)，访问最快
///    - Uniform Buffer: 大量只读数据，需要绑定 descriptor
///    - M, N, K 只有 12 bytes，用 Push Constants 最合适
///
/// 5. Buffer 绑定 (binding = 0, 1, 2)
///    - A, B, C 三个 buffer 分别绑定到 binding 0, 1, 2
///    - A, B 只读 (readonly)，C 只写 (writeonly)
///    - 这告诉 GPU 驱动可以做访问优化

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(binding = 0) readonly buffer BufferA {
    float a[];
};

layout(binding = 1) readonly buffer BufferB {
    float b[];
};

layout(binding = 2) writeonly buffer BufferC {
    float c[];
};

layout(push_constant) uniform PushConstants {
    uint M;
    uint N;
    uint K;
} pc;

void main() {
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;

    // 边界检查: 如果 invocation 超出矩阵范围则跳过
    // 这在 M 或 N 不是 16 的倍数时发生
    if (row >= pc.M || col >= pc.N) {
        return;
    }

    // 计算 C[row, col] = sum(A[row, k] * B[k, col]) for k = 0..K-1
    // A 是 MxK 行优先: A[row, k] = a[row * K + k]
    // B 是 KxN 行优先: B[k, col] = b[k * N + col]
    float sum = 0.0;
    for (uint k = 0; k < pc.K; k++) {
        sum += a[row * pc.K + k] * b[k * pc.N + col];
    }

    // C 是 MxN 行优先: C[row, col] = c[row * N + col]
    c[row * pc.N + col] = sum;
}
