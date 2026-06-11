@echo off
REM ============================================================
REM  EdgeAI Performance Master - 构建脚本 (Windows)
REM
REM  用法:
REM    build.bat            -- Release 构建 (无可选依赖)
REM    build.bat debug      -- Debug 构建
REM    build.bat clean      -- 清理构建目录
REM    build.bat vulkan     -- 启用 Vulkan GPU 加速
REM    build.bat ncnn       -- 从 submodule 源码构建 NCNN
REM    build.bat vulkan ncnn -- 同时启用 Vulkan 和 NCNN
REM
REM  前置条件:
REM    1. CMake >= 3.16
REM    2. C++17 编译器 (MSVC 2019+, GCC 9+, Clang 10+)
REM
REM  可选依赖:
REM    - Vulkan SDK: 用于 GPU 计算 (Lab 6-8)
REM    - NCNN: 用于自定义算子 (Lab 9)
REM ============================================================

setlocal enabledelayedexpansion

set PROJECT_DIR=%~dp0..
cd /d %PROJECT_DIR%

REM --- 参数处理 ---
set BUILD_TYPE=Release
set ENABLE_VULKAN=OFF
set BUILD_NCNN=OFF

:parse_args
if "%~1"=="" goto :done_args
if /i "%~1"=="debug" set BUILD_TYPE=Debug
if /i "%~1"=="Debug" set BUILD_TYPE=Debug
if /i "%~1"=="clean" (
    echo 清理构建目录...
    if exist build rmdir /s /q build
    echo 已清理。
    exit /b 0
)
if /i "%~1"=="vulkan" set ENABLE_VULKAN=ON
if /i "%~1"=="ncnn" set BUILD_NCNN=ON
shift
goto :parse_args
:done_args

REM --- 自动检测 Vulkan SDK ---
if defined VULKAN_SDK (
    echo 检测到 VULKAN_SDK=%VULKAN_SDK%
    set ENABLE_VULKAN=ON
)

REM --- 检查 NCNN submodule ---
if exist 3rdparty\ncnn\CMakeLists.txt (
    if "%BUILD_NCNN%"=="OFF" (
        echo [提示] 检测到 3rdparty/ncnn/ 源码，可用 -DBUILD_NCNN=ON 从源码构建 NCNN
    )
)

REM --- 创建构建目录 ---
if not exist build mkdir build
cd build

REM --- CMake 配置 ---
echo.
echo ============================================
echo  配置 CMake (BUILD_TYPE=%BUILD_TYPE%)
echo  ENABLE_VULKAN=%ENABLE_VULKAN%
echo  BUILD_NCNN=%BUILD_NCNN%
echo ============================================
echo.

REM 不硬编码生成器，让 CMake 自动检测
cmake .. ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DENABLE_VULKAN=%ENABLE_VULKAN% ^
    -DENABLE_AVX2=ON ^
    -DENABLE_PYTHON=ON ^
    -DBUILD_LABS=ON ^
    -DBUILD_TESTS=OFF ^
    -DBUILD_NCNN=%BUILD_NCNN%

if errorlevel 1 (
    echo.
    echo [错误] CMake 配置失败！
    echo 可能原因:
    echo   1. CMake 未安装或版本过低 (需要 ^>= 3.16)
    echo   2. C++17 编译器未安装
    echo.
    echo 提示: 如果 Vulkan 或 NCNN 导致配置失败，可以不带参数直接运行 build.bat
    echo.
    pause
    exit /b 1
)

REM --- 编译 ---
echo.
echo ============================================
echo  编译中...
echo ============================================
echo.

cmake --build . --config %BUILD_TYPE% -j %NUMBER_OF_PROCESSORS%

if errorlevel 1 (
    echo.
    echo [错误] 编译失败！
    pause
    exit /b 1
)

echo.
echo ============================================
echo  构建成功!
echo  Lab 实验: build\%BUILD_TYPE%\step1_timer_basics.exe 等
echo.
echo  运行示例:
echo    build\%BUILD_TYPE%\step1_timer_basics.exe
echo    build\%BUILD_TYPE%\step9_custom_ncnn_op.exe
echo ============================================
echo.

endlocal
