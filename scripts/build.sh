#!/usr/bin/env bash
# ============================================================
#  EdgeAI Performance Master - 构建脚本 (Linux/macOS)
#
#  用法:
#    ./build.sh            -- Release 构建 (无可选依赖)
#    ./build.sh debug      -- Debug 构建
#    ./build.sh clean      -- 清理构建目录
#    ./build.sh vulkan     -- 启用 Vulkan GPU 加速
#    ./build.sh ncnn       -- 从 submodule 源码构建 NCNN
#    ./build.sh vulkan ncnn -- 同时启用 Vulkan 和 NCNN
#
#  前置条件:
#    1. CMake >= 3.16
#    2. C++17 编译器 (GCC 9+, Clang 10+)
#
#  可选依赖:
#    - Vulkan SDK: 用于 GPU 计算 (Lab 6-8)
#    - NCNN: 用于自定义算子 (Lab 9)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# --- 默认值 ---
BUILD_TYPE=Release
ENABLE_VULKAN=OFF
BUILD_NCNN=OFF

# --- 参数解析 ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        debug|Debug)
            BUILD_TYPE=Debug
            ;;
        clean)
            echo "清理构建目录..."
            rm -rf build
            echo "已清理。"
            exit 0
            ;;
        vulkan)
            ENABLE_VULKAN=ON
            ;;
        ncnn)
            BUILD_NCNN=ON
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [debug] [clean] [vulkan] [ncnn]"
            exit 1
            ;;
    esac
    shift
done

# --- 自动检测 Vulkan ---
if [ -d "/usr/include/vulkan" ] || [ -n "${VULKAN_SDK:-}" ]; then
    echo "检测到 Vulkan 支持"
    ENABLE_VULKAN=ON
fi

# --- 检查 NCNN submodule ---
if [ -f "3rdparty/ncnn/CMakeLists.txt" ]; then
    if [ "$BUILD_NCNN" = "OFF" ]; then
        echo "[提示] 检测到 3rdparty/ncnn/ 源码，可用 ncnn 参数从源码构建 NCNN"
    fi
fi

# --- 创建构建目录 ---
mkdir -p build
cd build

# --- CMake 配置 ---
echo ""
echo "============================================"
echo " 配置 CMake (BUILD_TYPE=$BUILD_TYPE)"
echo " ENABLE_VULKAN=$ENABLE_VULKAN"
echo " BUILD_NCNN=$BUILD_NCNN"
echo "============================================"
echo ""

cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DENABLE_VULKAN="$ENABLE_VULKAN" \
    -DENABLE_AVX2=ON \
    -DENABLE_PYTHON=ON \
    -DBUILD_LABS=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_NCNN="$BUILD_NCNN"

# --- 编译 ---
echo ""
echo "============================================"
echo " 编译中..."
echo "============================================"
echo ""

NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
cmake --build . --config "$BUILD_TYPE" -j "$NPROC"

echo ""
echo "============================================"
echo " 构建成功!"
echo " Lab 实验: build/step1_timer_basics 等"
echo ""
echo " 运行示例:"
echo "    ./build/step1_timer_basics"
echo "    ./build/step9_custom_ncnn_op"
echo "============================================"
echo ""
