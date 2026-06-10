# EdgeAI Performance Master

端侧 AI 模型性能功耗拆解与瓶颈分析、异构并行调度、高性能算子开发 —— 一站式学习项目

## 项目简介

本项目通过 3 个 Lab（共 18 个 Step）的动手实验，帮助你系统掌握端侧 AI 推理优化的核心技术：

| Lab | 主题 | Steps | 学习重点 |
|-----|------|-------|----------|
| Lab1 | 性能功耗拆解 | 4 步 | 高精度计时、内存追踪、逐层 Profiling、功耗监控 |
| Lab2 | 异构并行调度 | 5 步 | 设备查询、算子图构建、规则调度、数据驱动调度、混合执行 |
| Lab3 | 高性能算子 | 9 步 | 朴素卷积→SIMD→Cache分块→Winograd→分块GEMM→Vulkan GPU→NCNN自定义算子 |

## 技术栈

- **语言**: C++17 (核心) + Python (可视化)
- **推理框架**: NCNN (腾讯开源，轻量级)
- **GPU 计算**: Vulkan Compute (跨平台)
- **SIMD**: x86 AVX2 / FMA
- **构建**: CMake 3.16+

## 快速开始

### 1. 安装依赖

```bash
# Vulkan SDK: https://vulkan.lunarg.com/sdk/home
# NCNN: 作为 git submodule 引入

git clone https://github.com/nihui/ncnn.git 3rdparty/ncnn
cd 3rdparty/ncnn && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON
cmake --build . --config Release -j
cmake --install .
```

### 2. 构建

```bash
# Windows
scripts\build.bat

# Linux/macOS
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_VULKAN=ON \
    -DENABLE_AVX2=ON \
    -DNCNN_ROOT=../3rdparty/ncnn/install
cmake --build . -j$(nproc)
```

### 3. 运行实验

```bash
# Lab1: 性能剖析
./build/step1_timer_basics
./build/step2_memory_tracking
./build/step3_op_profiling
./build/step4_power_monitoring

# Lab2: 异构调度
./build/step1_device_query
./build/step5_hybrid_execution

# Lab3: 高性能算子
./build/step1_naive_conv        # 朴素卷积基准
./build/step2_simd_conv         # AVX2 加速
./build/step4_winograd_conv     # Winograd 变换
./build/step9_custom_ncnn_op   # NCNN 自定义算子
```

### 4. 可视化

```bash
pip install -r python/requirements.txt

# 绘制 profiling 结果
python python/scripts/plot_profiling.py profiling_result.json

# 算子性能对比
python python/scripts/compare_ops.py --ops naive avx2 tiled winograd vulkan --times 2345 523 412 289 156

# Streamlit 仪表盘
streamlit run python/edgeai/dashboard.py
```

## 项目结构

```
edge-ai-perf-master/
├── include/              # C++ 头文件
│   ├── common/           # Tensor, Logger, MathUtils
│   ├── profiler/         # OpProfiler, MemoryTracker, PowerMonitor
│   ├── scheduler/        # DeviceInfo, OpGraph, DispatchStrategy, HybridExecutor
│   ├── operators/        # Naive→SIMD→Tiled→Winograd→GEMM→Vulkan 算子
│   └── ncnn_ext/         # NcnnModelRunner, CustomHardSwish
├── src/                  # C++ 实现
│   └── shaders/          # Vulkan Compute shaders (GLSL)
├── labs/                 # 18 个动手实验
│   ├── lab1_profiling/   # 性能功耗拆解 (4步)
│   ├── lab2_scheduling/  # 异构并行调度 (5步)
│   └── lab3_operators/  # 高性能算子开发 (9步)
├── python/               # Python 可视化和绑定
│   ├── edgeai/           # 工具包 (visualize, dashboard, power_analyzer)
│   ├── bindings/         # pybind11 绑定
│   └── scripts/          # CLI 绘图脚本
├── models/               # 模型下载脚本
├── cmake/                # CMake 模块
└── scripts/              # 构建脚本
```

## CMake 选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `ENABLE_VULKAN` | ON | 启用 Vulkan GPU 后端 |
| `ENABLE_AVX2` | ON | 启用 AVX2 SIMD 优化 |
| `ENABLE_PROFILING` | ON | 启用 profiling 模块 |
| `ENABLE_PYTHON` | ON | 构建 Python 绑定 |
| `BUILD_LABS` | ON | 构建实验可执行文件 |
| `BUILD_TESTS` | ON | 构建单元测试 |

## 学习路线

```
Lab1 (性能剖析)          Lab2 (异构调度)           Lab3 (算子优化)
┌──────────────┐     ┌──────────────┐        ┌──────────────┐
│ Step1 计时基础 │     │ Step1 设备查询 │        │ Step1 朴素卷积 │
│ Step2 内存追踪 │     │ Step2 算子图   │        │ Step2 SIMD    │
│ Step3 逐层Profile│   │ Step3 规则调度 │        │ Step3 Cache分块│
│ Step4 功耗监控 │     │ Step4 数据调度 │        │ Step4 Winograd│
└──────────────┘     │ Step5 混合执行 │        │ Step5 分块GEMM │
                      └──────────────┘        │ Step6 Vulkan基础│
                                              │ Step7 Vulkan GEMM│
                                              │ Step8 Vulkan Conv│
                                              │ Step9 自定义算子 │
                                              └──────────────┘
```

建议按顺序完成，每个 Step 都可以独立编译运行。
