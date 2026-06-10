# 端侧 AI 推理优化 —— 从原理到实践

> 本文档是 `edge-ai-perf-master` 项目的配套学习指南，建议边读边跑实验。

---

## 第一章 性能功耗拆解 —— 先找到瓶颈在哪

> 对应代码: `include/profiler/`、`labs/lab1_profiling/`

做优化的第一步永远不是写代码，而是**测量**。不知道瓶颈在哪，优化就是盲人摸象。

### 1.1 为什么计时这么讲究？

打开 `labs/lab1_profiling/step1_timer_basics.cpp`，核心原则：

**不要用 clock()，用 steady_clock。**

| 时钟 | 问题 | 能不能用 |
|------|------|----------|
| `clock()` | 算的是 CPU 时间，多线程时会被累加，不反映真实耗时 | ✗ |
| `system_clock` | 系统时间，会被 NTP 校时跳变，可能"倒流" | ✗ |
| `steady_clock` | 单调递增，不受系统时间调整影响 | ✓ 唯一正确选择 |

正确用法（项目里 `OpProfiler` 的实现）：

```cpp
auto start = std::chrono::steady_clock::now();
// ... 执行算子 ...
auto end = std::chrono::steady_clock::now();
double ms = std::chrono::duration<double, std::milli>(end - start).count();
```

#### Warmup 坑

前几次运行特别慢，因为：
- CPU 分支预测还没建立
- 代码和数据还没进缓存
- 操作系统的 CPU 频率还没拉到最高（动态调频）

正确的测量姿势：

```
跑 5 次 warmup → 丢弃
再跑 N 次取中位数 → 报告
```

中位数比均值更好，因为偶尔的操作系统调度抖动会产生极端值，中位数不受影响。

#### 计时开销本身

```cpp
auto start = steady_clock::now();
auto end = steady_clock::now();
double overhead = duration<double, nanosecond>(end - start).count();
// 典型值: 20-100 ns，远小于 1ms 的算子耗时，可以忽略
```

但如果你的算子耗时也在 ns 级别（比如一个 8 元素的向量加），计时开销就不能忽略了。

---

### 1.2 FLOPs 和 GFLOPS —— 衡量计算密度的基本单位

**FLOPs** (Floating Point Operations) = 浮点运算次数

标准 Conv2D 的 FLOPs 公式（见 `include/common/MathUtils.h`）：

```
FLOPs = 2 × outC × outH × outW × (inC/groups) × kH × kW
```

乘 2 是因为每次"乘加"算两次运算（1 次乘法 + 1 次加法），FMA 指令一条做两个 FLOP。

**GFLOPS** (Giga FLOPS per Second) = 每秒做了多少亿次浮点运算

```
GFLOPS = FLOPs / (时间ms × 10^6)
```

衡量硬件利用率的直接指标：

| 实现 | 典型 GFLOPS | 对应峰值占比 |
|------|-------------|-------------|
| 朴素卷积 | 0.5-2 | < 1% |
| AVX2 优化 | 10-40 | 5-20% |
| 分块 GEMM | 60-120 | 30-60% |
| MKL/OpenBLAS | 120-200 | 60-80% |
| 理论峰值 (8核 AVX2) | ~256 | 100% |

如果跑出来 GFLOPS 很低，说明 CPU 大部分时间在等内存，没在算。

---

### 1.3 Roofline Model —— 最核心的分析工具

性能分析中最重要的图，没有之一。

```
          │
  峰值算力│───────────────────────────── 计算密集区
  (GFLOPS)│  "算不过来"               ╱
          │                        ╱
          │                      ╱
          │                    ╱
          │                  ╱  ← 拐点 (Ridge Point)
          │                ╱
          │              ╱
          │            ╱  内存密集区
          │          ╱    "数据搬不过来"
          │        ╱
          │      ╱
          │    ╱
          │  ╱
          │╱
          └──────────────────────────────
              Arithmetic Intensity
              (FLOPs/Byte)
```

#### 两条线的含义

**斜线（内存屋顶）**：`GFLOPS = 峰值带宽 × AI`

算子落在这条线上，说明受内存带宽限制。无论 CPU 多快，数据搬不过来就白搭。

**水平线（计算屋顶）**：`GFLOPS = 峰值算力`

算子落在这条线上，说明受计算能力限制。数据够快，但算不过来。

**拐点 (Ridge Point)**：`AI = 峰值算力 / 峰值带宽`

典型 x86 平台：峰值算力 256 GFLOPS，峰值带宽 40 GB/s → 拐点 AI ≈ 6.4

- AI < 6.4：内存密集（大部分算子落在这里）
- AI > 6.4：计算密集

#### Arithmetic Intensity 怎么算

```cpp
// 来自 OpRecord 的计算
double AI = (double)flops / (double)memBytes;  // FLOPs per Byte
```

实际例子：

| 算子 | FLOPs | 内存(bytes) | AI | 瓶颈 |
|------|-------|------------|-----|------|
| ReLU (1M 元素) | 1M | 8M | 0.125 | 严重内存密集 |
| Conv2D 3x3 (64ch) | 1.7G | 6.7M | 254 | 计算密集 |
| Conv2D 1x1 (64ch) | 0.19G | 4.4M | 43 | 计算密集 |
| FC (4096→1000) | 8.2M | 72K | 114 | 计算密集 |

**关键洞察**：大部分"重"算子是计算密集的，大部分"轻"算子是内存密集的。这直接决定了调度策略——重算子放 GPU，轻算子放 CPU。

#### 怎么画这个图

```bash
# 先跑 Lab1 Step3，生成 profiling JSON
./step3_op_profiling

# 用 Python 脚本画图
python python/scripts/plot_profiling.py profiling_result.json --type roofline

# 或用 Streamlit 交互式仪表盘
streamlit run python/edgeai/dashboard.py
```

---

### 1.4 内存带宽 —— 被忽视的瓶颈

很多时候 GFLOPS 低不是因为 CPU 慢，而是数据搬不过来。

**测试方法**（Lab1 Step2 的核心实验）：

```cpp
float* data = new float[64 * 1024 * 1024];  // 256MB
auto start = steady_clock::now();
for (int i = 0; i < 64*1024*1024; i++) {
    volatile float val = data[i];  // 强制读取
}
auto end = steady_clock::now();
// 带宽 = 256MB / 时间
```

典型结果：

```
4KB   (L1):  ~200 GB/s
32KB  (L1):  ~180 GB/s
256KB (L2):  ~80 GB/s
8MB   (L3):  ~40 GB/s
256MB (DRAM): ~20 GB/s
```

从 L1 到 DRAM，带宽掉了 10 倍！这就是缓存优化如此重要的原因。

#### 顺序 vs 随机访问

```
顺序访问: 0, 1, 2, 3, ...    → CPU 预取器可以提前加载
随机访问: 0, 64, 128, 192, ... → 预取器失效，每次都是 cache miss
```

典型差异：**顺序比随机快 5-10 倍**。这就是为什么算子优化中"让内存访问连续"是第一原则。

---

### 1.5 功耗监控 —— 移动端的生命线

在手机、IoT 设备上，功耗比性能更重要——功耗高 = 电池耗尽快 = 用户卸载。

`PowerMonitor`（`src/profiler/PowerMonitor.cpp`）的原理：

| 平台 | 方法 | 精度 |
|------|------|------|
| Windows | PDH API 读 CPU 利用率 × TDP 估算 | 中 |
| Linux | `/sys/class/powercap/intel-rapl/` 读 RAPL 计数器 | 高 |
| 移动端 | 需要厂商 SDK (高通 QSTA, 联发科等) | 高 |

关键指标不是瓦数，而是**能效比**：

```
GFLOPS/W = 每瓦特能做多少亿次运算

例子:
  1线程: 12 GFLOPS / 8W  = 1.5 GFLOPS/W
  4线程: 42 GFLOPS / 28W = 1.5 GFLOPS/W  ← 性能涨 3.5x，功耗也涨 3.5x
  2线程: 24 GFLOPS / 14W = 1.7 GFLOPS/W  ← 反而最优
```

**实践启示**：更多线程不总是更好。存在一个"甜蜜点"，性能和功耗的平衡最优。

---

## 第二章 异构并行调度 —— 让 CPU 和 GPU 都别闲着

> 对应代码: `include/scheduler/`、`labs/lab2_scheduling/`

### 2.1 先搞清楚你有什么硬件

`DeviceInfo::queryCpu()`（`src/scheduler/DeviceInfo.cpp`）用 `__cpuid` 指令查询 CPU 特性：

```cpp
int cpuInfo[4];
__cpuid(cpuInfo, 1);
bool hasAVX2 = (cpuInfo[2] & (1 << 5)) != 0;
```

输出示例：

```
CPU 信息:
  型号: Intel Core i7-10700K
  核心数: 8 / 线程数: 16
  AVX2: ✓   → 256-bit SIMD，一次处理 8 个 float
  FMA:  ✓   → 一条指令同时做乘法和加法
  L1 缓存: 32 KB    ← 分块大小的依据
  L2 缓存: 256 KB   ← 分块大小的依据
  L3 缓存: 16 MB
```

**缓存大小为什么重要？** 算子优化中的分块大小必须根据缓存大小来定——分块太大溢出缓存等于没分，太小又浪费缓存空间。

GPU 信息通过 Vulkan API 查询（`DeviceInfo::queryGpu()`），获取显存、计算单元数等。

---

### 2.2 算子图 —— 算出谁先谁后、谁能并行

深度学习模型本质上是一个**有向无环图 (DAG)**：

```
Input → Conv1 → BN1 → ReLU1 → Pool1 → Conv2 → BN2 → ReLU2 → Pool2 → FC
```

`OpGraph`（`include/scheduler/OpGraph.h`）把这个图表示为邻接表：

```cpp
int conv1 = graph.addOp("Conv1", "Convolution");
int bn1   = graph.addOp("BN1",   "BatchNorm",   {conv1});  // 依赖 Conv1
int relu1 = graph.addOp("ReLU1", "ReLU",         {bn1});    // 依赖 BN1
```

#### 拓扑排序

Kahn 算法（`src/scheduler/OpGraph.cpp`）：

```
1. 找所有入度为 0 的节点 → 放入队列
2. 取出队首节点，输出
3. 把它的所有后继的入度减 1
4. 如果某个后继入度变为 0 → 放入队列
5. 重复直到队列为空
```

#### 找并行对

两个算子可以并行，当且仅当它们之间**没有路径**（A 不会到达 B，B 也不会到达 A）。

#### 关键路径

**关键路径**是图中最长的那条路径。无论你怎么并行，总时间不可能短于关键路径的时间。

```
Conv1(12ms) → ReLU1(0.3ms) → Conv2(25ms) → ReLU2(0.3ms) → FC(5ms)
关键路径长度: 42.6ms → 即使用无限设备并行，最快也要 42.6ms
```

---

### 2.3 调度策略：三种思路的演进

#### 策略一：HeuristicDispatch —— 人写规则

最简单，也最不精确。代码在 `src/scheduler/HeuristicDispatch.cpp`：

```
规则表:
  FLOPs < 10M              → CPU  (GPU kernel launch 开销就 0.1ms)
  类型是 ReLU/BN/Add       → CPU  (内存密集，GPU 无优势)
  Conv2D 且 outC >= 64      → GPU  (计算量大，GPU 并行度高)
  Conv2D 且 outC < 64       → CPU  (小卷积，GPU 优势不大)
  默认                      → CPU  (安全兜底)
```

**GPU kernel launch 开销**是理解这些规则的关键：

```
GPU 执行一个算子的时间 = launch 开销 + 数据传输 + 实际计算

launch 开销: ~0.01-0.1ms (无论算子大小)
数据传输:   ~0.1-1ms (取决于数据量)

ReLU 只要 0.01ms 但 launch 就要 0.1ms → GPU 更慢！
大 Conv 要 10ms 而 launch 只 0.1ms   → GPU 快很多！
```

#### 策略二：ProfileGuidedDispatch —— 让数据说话

先跑一遍，记录每个算子在 CPU 和 GPU 上各花多少时间，然后选更快的：

```cpp
profileStrategy->addProfileEntry("Conv1", 12.0, 3.0);   // CPU: 12ms, GPU: 3ms → 选 GPU
profileStrategy->addProfileEntry("ReLU1", 0.3, 0.8);   // CPU: 0.3ms, GPU: 0.8ms → 选 CPU
```

**冷启动问题**：第一次跑没有 profiling 数据，回退到 HeuristicDispatch。

#### 策略三：HybridExecutor —— 混合并行执行

把图编译成分阶段方案，同阶段内 CPU 和 GPU 并行：

```
          ┌→ Conv1(GPU) → BN1(CPU) → ReLU1(CPU) ─┐
Input ────┤                                          ├→ Add(CPU) → ...
          └── shortcut(CPU) ────────────────────────┘

分阶段:
  Stage 0: GPU[Conv1] || CPU[shortcut]   ← 同时跑！
  Stage 1: CPU[BN1]
  Stage 2: CPU[ReLU1]
  Stage 3: CPU[Add]                        ← 等两条路径都完成
```

Stage 0 中 Conv1 和 shortcut 互不依赖，一个放 GPU 一个放 CPU，**同时执行**。

执行时用 `std::async`（`src/scheduler/HybridExecutor.cpp`）：

```cpp
for (int opId : stage) {
    futures.push_back(std::async(std::launch::async, [&]() {
        opRunner(opId, decision.device);
    }));
}
for (auto& f : futures) f.wait();  // 等本阶段全部完成
```

#### 三种模式对比

| 模式 | 延迟 | 原理 |
|------|------|------|
| 纯 CPU | Σ 所有算子的 CPU 时间 | 串行，CPU 独占 |
| 纯 GPU | Σ 所有算子的 GPU 时间 + launch 开销 | 串行，小算子反而更慢 |
| 混合 | ≈ 关键路径时间 | 并行，CPU 和 GPU 各干擅长的 |

---

## 第三章 高性能算子开发 —— 优化从哪里来

> 对应代码: `include/operators/`、`src/operators/`、`labs/lab3_operators/`

### 3.1 朴素卷积 —— 理解基准线

6 层嵌套循环（`src/operators/Conv2dNaive.cpp`）：

```cpp
for (int n = 0; n < batch; n++)
  for (int oc = 0; oc < outC; oc++)
    for (int oh = 0; oh < outH; oh++)
      for (int ow = 0; ow < outW; ow++) {
        float sum = 0;
        for (int ic = 0; ic < inC; ic++)
          for (int kh = 0; kh < kH; kh++)
            for (int kw = 0; kw < kW; kw++)
              sum += input[...] * weight[...];
        output[...] = sum;
      }
```

典型性能（3→64 通道, 3×3, 224×224）：

```
FLOPs:   1.73G
耗时:    ~2345 ms
GFLOPS:  0.74
利用率:  0.3%   ← 99.7% 的时间在等内存！
```

内存访问模式太差：内层 `ic` 维度跨度为 `inH*inW`，缓存根本装不下。

---

### 3.2 im2col + AVX2 GEMM —— 第一道飞跃

#### im2col 变换

核心思想：把卷积**变成矩阵乘法**，因为 GEMM 有成熟的优化。

```
input  → im2col矩阵 [3×3×3, 224×224] = [27, 50176]
kernel → 权重矩阵   [64, 27]

GEMM: [64, 27] × [27, 50176] = [64, 50176] → reshape → [64, 224, 224]
```

im2col 的代价是内存膨胀（输入从 `3×224×224` 膨胀到 `27×50176`），但 GEMM 可以用 AVX2 加速。

#### AVX2 微内核

`src/operators/Conv2dAvx2.cpp` 中的核心代码：

```cpp
__m256 sum_vec = _mm256_setzero_ps();
for (int k = 0; k < K; k += 8) {
    __m256 a = _mm256_loadu_ps(&A_row[k]);     // 加载 A 的 8 个元素
    __m256 b = _mm256_loadu_ps(&B_col[k]);     // 加载 B 的 8 个元素
    sum_vec = _mm256_fmadd_ps(a, b, sum_vec);   // FMA: sum += a * b
}
```

一个 FMA 指令同时做 8 次乘法和 8 次加法 = **16 FLOPs per cycle**！

```
朴素卷积:  0.74 GFLOPS
AVX2 GEMM: 5-15 GFLOPS    ← 7-20x 加速！
```

---

### 3.3 Cache 分块 —— 让数据待在缓存里

**核心数字**：

```
L1 缓存: ~32KB,  ~1ns
L2 缓存: ~256KB, ~4ns
L3 缓存: ~8MB,   ~10ns
DRAM:    ~GB,    ~80ns

L1 和 DRAM 差 80 倍！
```

**问题**：朴素 GEMM 访问 B 矩阵时按列遍历，跳跃 N×4 字节，缓存不友好。

**解法**：分块（`src/operators/Conv2dTiled.cpp`）

```
for ii in 0..M step MC:       ← 外层: 遍历行块
  for jj in 0..N step NC:     ← 外层: 遍历列块
    for pp in 0..K step KC:   ← 外层: 遍历 K 维块
      // A块: MC×KC×4 bytes
      // B块: KC×NC×4 bytes
      // C块: MC×NC×4 bytes
      // 要求: A块 + B块 + C块 < L2 cache size
      内层: 计算这个小块的 GEMM
```

分块大小推导：

```
16×16 输出块 + 64 通道:
  输入: (16+2)×(16+2) × 64 × 4 = ~170 KB
  权重: 16 × 64 × 3 × 3 × 4   = ~36 KB
  总计: ~206 KB < L2 (256 KB) ✓

32×32 输出块:
  总计: ~680 KB > L2 ✗  → 溢出，性能反而下降
```

---

### 3.4 Winograd 变换 —— 用代数减少乘法

最优雅的优化：用数学变换减少乘法次数。

#### 1D 原理

```
朴素: 计算 2 个输出值，3×1 卷积核 → 6 次乘法
Winograd F(2,3): 变换到 Winograd 域 → 只需 4 次乘法！
```

变换本身只有加减法（几乎免费），乘法才是瓶颈。

#### 2D 扩展

```
朴素 2D (3×3 kernel, 2×2 output): 36 次乘法
Winograd F(2,3) × F(2,3):        16 次乘法

减少 2.25 倍！
```

#### 变换矩阵

`src/operators/Conv2dWinograd.cpp` 中的定义：

```
G (权重变换):        B^T (输入变换):     A^T (输出变换):
  | 1    0    0   |    | 1  0 -1  0 |      | 1  1 |
  | 1/2  1/2  1/2 |    | 0  1  1  0 |      | 0  1 |
  | 1/2 -1/2  1/2 |    | 0 -1  1  0 |
  | 0    0    1   |    | 1  0  1  0 |
```

#### 适用场景

| 场景 | Winograd 有用？ |
|------|-----------------|
| stride=1, kernel=3×3, 大通道 | ✓ 最适用，2.25x 加速 |
| stride=2 | ✗ 变换不成立 |
| kernel=1×1 | ✗ 乘法次数已经最少 |
| kernel=5×5 | △ 收益递减 |

**数值精度**：Winograd 引入浮点误差，推理通常可接受（< 1e-6），训练需注意。

---

### 3.5 分块 GEMM —— 矩阵乘法的终极优化

`src/operators/GemmBlocked.cpp` 是 BLAS 库（MKL、OpenBLAS）的核心算法。

#### 三层分块 + Packing

```
L2 级别: MC × NC × KC 分块
  ↓ packing: 把分散数据拷贝到连续缓冲区
L1 级别: MR × NR 微内核
  ↓ AVX2: 数据保持在寄存器中
寄存器级: FMA 指令
```

Packing 的代价是额外拷贝，但收益是**完全连续的内存访问**——微内核中不再有 cache miss。

#### 微内核示例

```cpp
// 4×8 微内核 (MR=4, NR=8)
__m256 c0 = _mm256_setzero_ps();  // 8 个 C 的累加器
__m256 c1 = _mm256_setzero_ps();
__m256 c2 = _mm256_setzero_ps();
__m256 c3 = _mm256_setzero_ps();

for (int p = 0; p < KC; p++) {
    __m256 b = _mm256_loadu_ps(&B_packed[p * NR]);
    c0 = _mm256_fmadd_ps(_mm256_set1_ps(A[0*KC+p]), b, c0);
    c1 = _mm256_fmadd_ps(_mm256_set1_ps(A[1*KC+p]), b, c1);
    c2 = _mm256_fmadd_ps(_mm256_set1_ps(A[2*KC+p]), b, c2);
    c3 = _mm256_fmadd_ps(_mm256_set1_ps(A[3*KC+p]), b, c3);
}
```

#### 分块参数选择

```
MC=128, NC=256, KC=64 时:
  A 块: 128×64×4 = 32KB  → L1!
  B 块: 64×256×4  = 64KB  → L2
  C 块: 128×256×4 = 128KB → L2
  总计: 224KB < L2 (256KB) ✓
```

---

### 3.6 Vulkan GPU 计算 —— 从 CPU 到 GPU 的跨越

#### 为什么 GPU 可以快

```
CPU: 8 核 × 4 GHz × 16 FLOP/cycle (AVX2 FMA) ≈ 256 GFLOPS
GPU: 3072 核 × 1.5 GHz × 2 FLOP/cycle          ≈ 9216 GFLOPS
```

GPU 核心数是 CPU 的几百倍，并行度碾压。

#### GPU 的代价

```
CPU → GPU 传输: ~0.1-5ms
GPU kernel launch: ~0.01-0.1ms
GPU → CPU 传输: ~0.1-5ms
```

**交叉点**：数据量 < ~1MB 时，传输时间 > 计算时间，GPU 反而更慢。

#### Vulkan Compute Pipeline

`src/operators/VulkanComputeBase.cpp` 的流程：

```
1. initialize()      → 创建 GPU 环境 (Instance → Device → Queue)
2. loadShader(spv)   → 加载 SPIR-V 字节码
3. createBuffer()    → 分配 GPU 内存 (优先 HOST_VISIBLE 零拷贝)
4. writeBuffer()     → CPU 数据 → GPU 内存 (map → memcpy → unmap)
5. dispatch()        → 录制命令 → 提交队列 → 等待完成
6. readBuffer()      → GPU 内存 → CPU 数据
```

#### GLSL Compute Shader

`src/shaders/gemm.comp.glsl` 核心逻辑：

```glsl
layout(local_size_x = 16, local_size_y = 16) in;  // 16×16 workgroup

void main() {
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    if (row >= M || col >= N) return;

    float sum = 0.0;
    for (uint k = 0; k < K; k++)
        sum += a[row*K + k] * b[k*N + col];
    c[row*N + col] = sum;
}
```

#### 性能对比

| 矩阵大小 | CPU 朴素 | CPU AVX2 | GPU Vulkan | GPU 加速比 |
|----------|----------|----------|------------|-----------|
| 128×128 | 2.1 ms | 0.4 ms | 1.8 ms | 0.2x (GPU 更慢!) |
| 512×512 | 135 ms | 18 ms | 6 ms | 3x |
| 1024×1024 | 1080 ms | 140 ms | 22 ms | 6.4x |

128×128 时 GPU 更慢——数据传输时间超过了计算节省。这就是调度策略要把小算子放 CPU 的原因。

---

### 3.7 NCNN 自定义算子 —— 优化的最终归宿

所有算子优化的最终目标是**嵌入推理框架**。

#### 开发流程

```
1. 继承 ncnn::Layer
2. 设置属性: one_blob_only, support_inplace
3. 实现 forward() (含 AVX2 路径)
4. 注册: NcnnCustomLayerRegistry::instance().registerLayer<CustomHardSwish>("HardSwish_Custom")
5. 应用: registry.applyAll(net)
6. 在 .param 文件中用自定义类型名
```

#### CustomHardSwish 示例

`src/ncnn_ext/CustomHardSwish.cpp` 的 AVX2 路径：

```cpp
// HardSwish(x) = x * clip(x + 3, 0, 6) / 6
__m256 x = _mm256_loadu_ps(&input[i]);
__m256 xp3 = _mm256_add_ps(x, three);              // x + 3
__m256 clipped = _mm256_min_ps(                      // clip(x+3, 0, 6)
    _mm256_max_ps(xp3, zero), six);
__m256 result = _mm256_mul_ps(                       // x * clip / 6
    _mm256_mul_ps(x, clipped), oneSixth);            // 用乘 1/6 替代除法
_mm256_storeu_ps(&output[i], result);
```

8 个 float 同时处理，4-6x 加速。

---

## 第四章 优化路线图 —— 从哪里开始

### 4.1 推荐学习顺序

```
第 1 天: Lab1 (Step1-4)
  理解测量方法，学会用 Roofline 分析瓶颈

第 2 天: Lab3 (Step1-2)
  跑朴素卷积，看 GFLOPS 多低；跑 AVX2，看提升多大

第 3 天: Lab3 (Step3-4)
  学 Cache 分块和 Winograd，理解为什么分块大小要匹配缓存

第 4 天: Lab3 (Step5)
  分块 GEMM，这是所有算子优化的基础

第 5 天: Lab2 (Step1-5)
  学调度策略，理解什么时候用 CPU、什么时候用 GPU

第 6 天: Lab3 (Step6-8)
  Vulkan GPU 计算，画出 CPU vs GPU 的交叉点

第 7 天: Lab3 (Step9)
  把优化后的算子嵌入 NCNN，完成端到端闭环
```

### 4.2 动手练习建议

1. **改参数看效果**：修改 `Conv2dTiled` 的 tile size，观察性能变化
2. **画 Roofline 图**：用 Python 脚本可视化 profiling 数据
3. **加一个新的算子**：参照 `CustomHardSwish`，实现 `CustomSwish` 或 `CustomGELU`
4. **调调度策略**：修改 `HeuristicDispatch` 的阈值，看对端到端延迟的影响

### 4.3 性能优化检查清单

```
□ 是否做了 warmup 再测量？
□ 是否画了 Roofline 图确认瓶颈类型？
□ 内存访问是否连续（能否用 im2col + GEMM）？
□ 是否启用了 AVX2 / FMA？
□ 分块大小是否匹配缓存层级？
□ 是否可以用 Winograd（stride=1, 3×3）？
□ GPU 上跑的数据量是否足够大（> 1MB）？
□ 小算子是否留在了 CPU 上？
□ 是否有并行机会被 HybridExecutor 利用了？
□ 优化后的精度是否可接受？
```
