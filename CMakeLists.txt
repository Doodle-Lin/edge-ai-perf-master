cmake_minimum_required(VERSION 3.16)
project(edge-ai-perf-master VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_VULKAN    "Enable Vulkan GPU compute backend" ON)
option(ENABLE_AVX2      "Enable AVX2 SIMD optimizations"   ON)
option(ENABLE_PROFILING "Enable profiling module"           ON)
option(ENABLE_PYTHON    "Build Python bindings"             ON)
option(BUILD_LABS       "Build lab executables"             ON)
option(BUILD_TESTS      "Build unit tests"                  ON)

if(WIN32)
    add_definitions(-DPLATFORM_WINDOWS)
elseif(UNIX)
    add_definitions(-DPLATFORM_LINUX)
endif()

include(cmake/CompilerOptions.cmake)

if(ENABLE_VULKAN)
    include(cmake/VulkanSetup.cmake)
endif()

include(cmake/FindNCNN.cmake)

add_subdirectory(src)

if(BUILD_LABS)
    add_subdirectory(labs)
endif()

if(ENABLE_PYTHON)
    add_subdirectory(python)
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
