@echo off
REM ============================================================
REM  EdgeAI Performance Master - 构建脚本 (Windows)
REM
REM  用法:
REM    build.bat          -- Release 构建
REM    build.bat debug    -- Debug 构建
REM    build.bat clean    -- 清理构建目录
REM
REM  前置条件:
REM    1. Visual Studio 2019/2022 已安装
REM    2. CMake >= 3.16
REM    3. Vulkan SDK (可选, GPU 加速)
REM    4. NCNN 已放在 3rdparty/ncnn/ (git submodule)
REM ============================================================

setlocal

set PROJECT_DIR=%~dp0..
cd /d %PROJECT_DIR%

REM --- 参数处理 ---
set BUILD_TYPE=Release
if "%1"=="debug" set BUILD_TYPE=Debug
if "%1"=="Debug" set BUILD_TYPE=Debug

if "%1"=="clean" (
    echo 清理构建目录...
    if exist build rmdir /s /q build
    echo 已清理。
    exit /b 0
)

REM --- 初始化 submodule ---
if not exist 3rdparty\ncnn\CMakeLists.txt (
    echo 初始化 git submodule...
    git submodule update --init --recursive
    if errorlevel 1 (
        echo [警告] git submodule 初始化失败
        echo 请手动执行: git submodule update --init --recursive
    )
)

REM --- 创建构建目录 ---
if not exist build mkdir build
cd build

REM --- CMake 配置 ---
echo.
echo ============================================
echo  配置 CMake (BUILD_TYPE=%BUILD_TYPE%)
echo ============================================
echo.

cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DENABLE_VULKAN=ON ^
    -DENABLE_AVX2=ON ^
    -DENABLE_PYTHON=ON ^
    -DBUILD_LABS=ON ^
    -DBUILD_TESTS=OFF

if errorlevel 1 (
    echo.
    echo [错误] CMake 配置失败！
    echo 可能原因:
    echo   1. NCNN 未放在 3rdparty/ncnn/
    echo   2. Visual Studio 未安装
    echo   3. Vulkan SDK 未安装 (设置 ENABLE_VULKAN=OFF 跳过)
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
echo  可执行文件在: build\%BUILD_TYPE%\
echo  Lab 实验:     build\%BUILD_TYPE%\step1_timer_basics.exe 等
echo ============================================
echo.

endlocal
