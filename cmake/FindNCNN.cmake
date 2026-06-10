# FindNCNN.cmake
# Locate the NCNN inference library.
#
# Search order:
#   1. 3rdparty/ncnn submodule (relative to project source dir)
#   2. NCNN_ROOT environment variable
#   3. System paths (via CMake default search)
#
# Sets:
#   NCNN_FOUND        - True if NCNN was found
#   NCNN_INCLUDE_DIRS - Include directories for NCNN headers
#   NCNN_LIBRARIES    - Libraries to link against

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

    message(STATUS "FindNCNN: Found NCNN")
    message(STATUS "  Include dirs : ${NCNN_INCLUDE_DIRS}")
    message(STATUS "  Libraries    : ${NCNN_LIBRARIES}")
else()
    message(STATUS "FindNCNN: NCNN not found")
endif()

# ---------------------------------------------------------------------------
# Clean up internal variables
# ---------------------------------------------------------------------------
unset(_NCNN_SUBMODULE_DIR)
unset(_NCNN_ROOT_HINT)
unset(_NCNN_LIB_NAMES)
