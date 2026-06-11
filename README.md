# EdgeAI Performance Master

端侧 AI 模型性能功耗拆解与瓶颈分析、异构并行调度、高性能算子开发 —— 一站式学习项目

## 项目简介

本项目通过 3 个 Lab（共 18 个 Step）的动手实验，帮助你系统掌握端侧 AI 推理优化的核心技术：

| Lab | 主题 | Steps | 学习重点 |
|-----|------|-------|----------|
| Lab1 | 性能功耗拆解 | 4 步 | 高精度计时、内存追踪、逐层 Profiling、功耗监控 |
| Lab2 | 异构并行调度 | 5 步 | 设备查询、算子图构建、规则调度、数据驱动调度、混合执行 |
| Lab3 | 高性能算子 | 9 步 | 朴素卷积→SIMD→Cache分块→Winograd→分块GEMM→Vulkan GPU→NCNN自定义算子 |

## 前置条件

### 必需

- **CMake** 3.16 或更高版本
- **C++17 编译器**: MSVC 2019+ (Windows), GCC 9+ / Clang 10+ (Linux/macOS)

### 可选

| 依赖 | 用途 | 未安装时的影响 |
|------|------|----------------|
| Vulkan SDK | GPU 计算 (Lab 6-8) | Lab 6-8 显示 "Vulkan not enabled" 提示，其余实验正常运行 |
| NCNN | 自定义算子 (Lab 9) | Lab 9 使用模拟 NCNN 运行，演示流程完整但无真实推理 |
| pybind11 | Python 绑定 | Python 绑定不构建，可视化脚本仍可独立使用 |

## 快速开始

### 1. 最简构建（无需任何可选依赖）

**Windows:**
```bash
scripts\build.bat
```

**Linux/macOS:**
```bash
chmod +x scripts/build.sh
./scripts/build.sh
```

**手动 CMake:**
```bash
mkdir build && cd build
cmake .. -DENABLE_VULKAN=OFF -DENABLE_AVX2=ON -DBUILD_LABS=ON
cmake --build . -j
```

### 2. 启用 Vulkan GPU 加速

**Windows:**
```bash
scripts\build.bat vulkan
```

**Linux/macOS:**
```bash
./scripts/build.sh vulkan
```

### 3. 启用 NCNN（从 submodule 源码构建）

```bash
git submodule update --init --recursive
# Windows:
scripts\build.bat ncnn vulkan
# Linux/macOS:
./scripts/build.sh ncnn vulkan
```

### 4. 运行实验

```bash
# Lab1: 性能剖析
./build/step1_timer_basics
./build/step2_memory_tracking
./build/step3_op_profiling
./build/step4_power_monitoring

# Lab2: 异构调度
./build/step1_device_query
./build/step2_op_graph_build
./build/step3_heuristic_dispatch
./build/step4_profile_guided_dispatch
./build/step5_hybrid_execution

# Lab3: 高性能算子
./build/step1_naive_conv
./build/step2_simd_conv
./build/step3_cache_tiled_conv
./build/step4_winograd_conv
./build/step5_gemm_blocked
./build/step6_vulkan_compute_basics
./build/step7_vulkan_gemm
./build/step8_vulkan_conv2d
./build/step9_custom_ncnn_op
```

### 5. 可视化

```bash
pip install -r python/requirements.txt

# 绘制 profiling 结果
python python/scripts/plot_profiling.py profiling_result.json

# 算子性能对比
python python/scripts/compare_ops.py --ops naive avx2 tiled winograd vulkan --times 2345 523 412 289 156

# Streamlit 仪表盘
streamlit run python/edgeai/dashboard.py
```

## 构建配置矩阵

| NCNN | Vulkan | pybind11 | 实验覆盖 |
|------|--------|----------|----------|
| 否 | 否 | 否 | 全部 18 个实验 (Lab 6-8 显示提示信息; Lab 9 使用模拟 NCNN) |
| 否 | 是 | 否 | Lab 6-8 使用真实 Vulkan; Lab 9 使用模拟 NCNN |
| 是 | 否 | 否 | Lab 9 使用真实 NCNN; Lab 6-8 显示提示信息 |
| 是 | 是 | 是 | 完整功能 |

## 项目结构

```
edge-ai-perf-master/
├── include/              # C++ 头文件
│   ├── common/           # Tensor, Logger, MathUtils
│   ├── profiler/         # OpProfiler, MemoryTracker, PowerMonitor
│   ├── scheduler/        # DeviceInfo, OpGraph, DispatchStrategy, HybridExecutor
│   ├── operators/        # Naive→SIMD→Tiled→Winograd→GEMM→Vulkan 算子
│   └── ncnn_ext/         # NcnnModelRunner, CustomHardSwish (需 NCNN)
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
| `ENABLE_VULKAN` | ON | 启用 Vulkan GPU 后端（自动降级，Vulkan 不可用时关闭） |
| `ENABLE_AVX2` | ON | 启用 AVX2 SIMD 优化 |
| `ENABLE_PROFILING` | ON | 启用 profiling 模块 |
| `ENABLE_PYTHON` | ON | 构建 Python 绑定 |
| `BUILD_LABS` | ON | 构建实验可执行文件 |
| `BUILD_TESTS` | ON | 构建单元测试 |
| `BUILD_NCNN` | OFF | 从 3rdparty/ncnn 子模块源码构建 NCNN |

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

建议按顺序完成，每个 Step 都可以独立编译运行。详见 [LEARNING_GUIDE.md](LEARNING_GUIDE.md)。
