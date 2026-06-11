# FindNCNN.cmake
# Locate the NCNN inference library.
#
# Search order:
#   1. 3rdparty/ncnn submodule (relative to project source dir)
#   2. NCNN_ROOT environment variable
#   3. System paths (via CMake default search)
#
# If NCNN is not found but the submodule source exists and BUILD_NCNN is ON,
# NCNN will be built from source via add_subdirectory.
#
# Sets:
#   NCNN_FOUND        - True if NCNN was found
#   NCNN_INCLUDE_DIRS - Include directories for NCNN headers
#   NCNN_LIBRARIES    - Libraries to link against
#   USE_NCNN          - Compile definition added when NCNN is available

# ---------------------------------------------------------------------------
# Option to build NCNN from submodule source
# ---------------------------------------------------------------------------
option(BUILD_NCNN "Build NCNN from submodule source at 3rdparty/ncnn" OFF)

# ---------------------------------------------------------------------------
# 1. Submodule path under the project tree
# ---------------------------------------------------------------------------
set(_NCNN_SUBMODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ncnn")

# ---------------------------------------------------------------------------
# 2. Environment variable hint
# ---------------------------------------------------------------------------
if(DEFINED ENV{NCNN_ROOT})
    set(_NCNN_ROOT_HINT "$ENV{NCNN_ROOT}")
else()
    set(_NCNN_ROOT_HINT "")
endif()

# ---------------------------------------------------------------------------
# Search for header
# ---------------------------------------------------------------------------
find_path(NCNN_INCLUDE_DIR
    NAMES ncnn/net.h
    PATHS
        "${_NCNN_SUBMODULE_DIR}/include"
        "${_NCNN_SUBMODULE_DIR}/build/install/include"
        "${_NCNN_ROOT_HINT}/include"
        /usr/include
        /usr/local/include
    DOC "NCNN include directory"
)

# ---------------------------------------------------------------------------
# Search for library
# ---------------------------------------------------------------------------
if(MSVC)
    set(_NCNN_LIB_NAMES ncnn)
else()
    set(_NCNN_LIB_NAMES ncnn)
endif()

find_library(NCNN_LIBRARY
    NAMES ${_NCNN_LIB_NAMES}
    PATHS
        "${_NCNN_SUBMODULE_DIR}/lib"
        "${_NCNN_SUBMODULE_DIR}/build/install/lib"
        "${_NCNN_SUBMODULE_DIR}/build/src"
        "${_NCNN_SUBMODULE_DIR}/build/Release"
        "${_NCNN_ROOT_HINT}/lib"
        /usr/lib
        /usr/local/lib
    DOC "NCNN library"
)

# ---------------------------------------------------------------------------
# Determine found status
# ---------------------------------------------------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCNN
    REQUIRED_VARS NCNN_LIBRARY NCNN_INCLUDE_DIR
)

if(NCNN_FOUND)
    set(NCNN_INCLUDE_DIRS "${NCNN_INCLUDE_DIR}")
    set(NCNN_LIBRARIES    "${NCNN_LIBRARY}")

    if(NOT TARGET NCNN::NCNN)
        add_library(NCNN::NCNN UNKNOWN IMPORTED)
        set_target_properties(NCNN::NCNN PROPERTIES
            IMPORTED_LOCATION "${NCNN_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NCNN_INCLUDE_DIR}"
        )
    endif()

    add_definitions(-DUSE_NCNN)

    message(STATUS "FindNCNN: Found NCNN")
    message(STATUS "  Include dirs : ${NCNN_INCLUDE_DIRS}")
    message(STATUS "  Libraries    : ${NCNN_LIBRARIES}")
else()
    # ---------------------------------------------------------------------------
    # Try building NCNN from submodule source
    # ---------------------------------------------------------------------------
    if(BUILD_NCNN AND EXISTS "${_NCNN_SUBMODULE_DIR}/CMakeLists.txt")
        message(STATUS "FindNCNN: NCNN not found pre-built, building from submodule source")

        # Configure NCNN build options to minimize build time
        set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
        set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
        set(NCNN_PYTHON OFF CACHE BOOL "" FORCE)
        set(NCNN_VULKAN ${ENABLE_VULKAN} CACHE BOOL "" FORCE)

        # Build NCNN as part of this project
        add_subdirectory(${_NCNN_SUBMODULE_DIR} ${CMAKE_BINARY_DIR}/ncnn-build)

        # After add_subdirectory, the ncnn target exists
        set(NCNN_FOUND TRUE)
        set(NCNN_INCLUDE_DIRS "${_NCNN_SUBMODULE_DIR}/src;${CMAKE_BINARY_DIR}/ncnn-build")
        set(NCNN_LIBRARIES ncnn)

        add_definitions(-DUSE_NCNN)

        message(STATUS "FindNCNN: NCNN built from submodule")
        message(STATUS "  Include dirs : ${NCNN_INCLUDE_DIRS}")
        message(STATUS "  Libraries    : ${NCNN_LIBRARIES}")
    else()
        message(STATUS "FindNCNN: NCNN not found - NCNN-dependent features will be disabled")
        message(STATUS "  To enable: install NCNN, set NCNN_ROOT env, or use -DBUILD_NCNN=ON")
    endif()
endif()

# ---------------------------------------------------------------------------
# Clean up internal variables
# ---------------------------------------------------------------------------
unset(_NCNN_SUBMODULE_DIR)
unset(_NCNN_ROOT_HINT)
unset(_NCNN_LIB_NAMES)
