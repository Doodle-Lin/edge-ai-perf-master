# 收集各模块源文件
file(GLOB_RECURSE COMMON_SOURCES     common/*.cpp)
file(GLOB_RECURSE PROFILER_SOURCES   profiler/*.cpp)
file(GLOB_RECURSE SCHEDULER_SOURCES  scheduler/*.cpp)
file(GLOB_RECURSE OPERATOR_SOURCES   operators/*.cpp)

# NCNN 扩展模块 - 仅在 NCNN 可用时编译
if(NCNN_FOUND)
    file(GLOB_RECURSE NCNN_EXT_SOURCES   ncnn_ext/*.cpp)
endif()

add_library(edgeai_core SHARED
    ${COMMON_SOURCES}
    ${PROFILER_SOURCES}
    ${SCHEDULER_SOURCES}
    ${OPERATOR_SOURCES}
    ${NCNN_EXT_SOURCES}
)

# Windows: ensure import library (.lib) is generated alongside the DLL
if(WIN32)
    set_target_properties(edgeai_core PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

target_include_directories(edgeai_core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

if(NCNN_FOUND)
    target_include_directories(edgeai_core PUBLIC ${NCNN_INCLUDE_DIRS})
endif()

# 查找线程库
find_package(Threads QUIET)
if(Threads_FOUND)
    target_link_libraries(edgeai_core PRIVATE Threads::Threads)
endif()

if(NCNN_FOUND)
    if(TARGET ncnn)
        target_link_libraries(edgeai_core PUBLIC ncnn)
    else()
        target_link_libraries(edgeai_core PUBLIC ${NCNN_LIBRARIES})
    endif()
endif()

if(WIN32)
    target_link_libraries(edgeai_core PRIVATE pdh)
endif()

if(ENABLE_VULKAN)
    target_link_libraries(edgeai_core PUBLIC Vulkan::Vulkan)
    target_include_directories(edgeai_core PRIVATE ${Vulkan_INCLUDE_DIRS})

    # 编译 GLSL shader 为 SPIR-V
    set(SHADER_DIR ${CMAKE_SOURCE_DIR}/src/shaders)
    set(SPV_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
    file(MAKE_DIRECTORY ${SPV_DIR})

    file(GLOB SHADERS "${SHADER_DIR}/*.comp.glsl")
    foreach(SHADER ${SHADERS})
        get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
        set(SPV_PATH "${SPV_DIR}/${SHADER_NAME}.spv")
        add_custom_command(
            OUTPUT ${SPV_PATH}
            COMMAND glslangValidator -V ${SHADER} -o ${SPV_PATH}
            DEPENDS ${SHADER}
            COMMENT "Compiling ${SHADER_NAME} to SPIR-V"
        )
        list(APPEND SPV_FILES ${SPV_PATH})
    endforeach()

    add_custom_target(shaders ALL DEPENDS ${SPV_FILES})
    add_dependencies(edgeai_core shaders)
endif()

# 拷贝 shader SPIR-V 到输出目录
if(ENABLE_VULKAN)
    add_custom_command(TARGET edgeai_core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_BINARY_DIR}/shaders
        $<TARGET_FILE_DIR:edgeai_core>/shaders
    )
endif()
