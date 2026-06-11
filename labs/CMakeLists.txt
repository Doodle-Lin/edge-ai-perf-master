# Lab 可执行文件构建
# 每个 lab step 是一个独立的可执行文件，链接到 edgeai_core

function(add_lab LAB_DIR STEP_NAME)
    add_executable(${STEP_NAME} ${CMAKE_SOURCE_DIR}/${LAB_DIR}/${STEP_NAME}.cpp)
    target_include_directories(${STEP_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/${LAB_DIR}
    )
    target_link_libraries(${STEP_NAME} PRIVATE edgeai_core)
    if(ENABLE_VULKAN)
        target_link_libraries(${STEP_NAME} PRIVATE Vulkan::Vulkan)
    endif()
endfunction()

# ──── Lab 1: 性能功耗拆解与瓶颈分析 ────
add_lab(labs/lab1_profiling step1_timer_basics)
add_lab(labs/lab1_profiling step2_memory_tracking)
add_lab(labs/lab1_profiling step3_op_profiling)
add_lab(labs/lab1_profiling step4_power_monitoring)

# ──── Lab 2: 异构并行调度 ────
add_lab(labs/lab2_scheduling step1_device_query)
add_lab(labs/lab2_scheduling step2_op_graph_build)
add_lab(labs/lab2_scheduling step3_heuristic_dispatch)
add_lab(labs/lab2_scheduling step4_profile_guided_dispatch)
add_lab(labs/lab2_scheduling step5_hybrid_execution)

# ──── Lab 3: 高性能算子开发 ────
add_lab(labs/lab3_operators step1_naive_conv)
add_lab(labs/lab3_operators step2_simd_conv)
add_lab(labs/lab3_operators step3_cache_tiled_conv)
add_lab(labs/lab3_operators step4_winograd_conv)
add_lab(labs/lab3_operators step5_gemm_blocked)
add_lab(labs/lab3_operators step6_vulkan_compute_basics)
add_lab(labs/lab3_operators step7_vulkan_gemm)
add_lab(labs/lab3_operators step8_vulkan_conv2d)
add_lab(labs/lab3_operators step9_custom_ncnn_op)

# NCNN 自定义算子实验 - 在 NCNN 可用时使用真实 NCNN API
if(NCNN_FOUND)
    target_compile_definitions(step9_custom_ncnn_op PRIVATE USE_NCNN)
endif()
