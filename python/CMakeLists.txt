# Python 绑定和工具

find_package(pybind11 QUIET)

if(NOT pybind11_FOUND)
    # 尝试从 submodule 查找
    set(pybind11_DIR "${CMAKE_SOURCE_DIR}/3rdparty/pybind11")
    if(EXISTS "${pybind11_DIR}/CMakeLists.txt")
        add_subdirectory(${pybind11_DIR})
    else()
        message(STATUS "pybind11 not found, skipping Python bindings")
        return()
    endif()
endif()

pybind11_add_module(_edgeai
    bindings/_edgeai.cpp
)

target_link_libraries(_edgeai PRIVATE edgeai_core)
target_include_directories(_edgeai PRIVATE ${CMAKE_SOURCE_DIR}/include)

# 安装到 python/edgeai/ 目录
install(TARGETS _edgeai
    LIBRARY DESTINATION python/edgeai
)
