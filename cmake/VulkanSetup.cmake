# VulkanSetup.cmake
# Finds the Vulkan SDK and exposes an ENABLE_VULKAN option.

find_package(Vulkan QUIET)

if(Vulkan_FOUND)
    message(STATUS "VulkanSetup: Vulkan found")
    message(STATUS "  Include dirs : ${Vulkan_INCLUDE_DIRS}")
    message(STATUS "  Libraries    : ${Vulkan_LIBRARIES}")

    add_definitions(-DENABLE_VULKAN)
    set(ENABLE_VULKAN ON CACHE BOOL "Vulkan is available" FORCE)
else()
    message(WARNING "VulkanSetup: Vulkan NOT found — Vulkan-dependent features will be disabled")
    set(ENABLE_VULKAN OFF CACHE BOOL "Vulkan is not available" FORCE)
endif()
