# CompilerOptions.cmake
# Detects CPU feature support and sets compiler flags accordingly.

include(CheckCXXCompilerFlag)

# ---------------------------------------------------------------------------
# Helper: try to compile a snippet that uses a given intrinsic header
# ---------------------------------------------------------------------------
function(_check_avx2_support RESULT_VAR)
    if(MSVC)
        # MSVC does not have -mavx2; /arch:AVX2 enables it globally.
        # We check via the _M_IX86_FP / _M_X64 preprocessor macros instead.
        try_compile(_avx2_check
            SOURCE_FROM_CONTENT "avx2_check.cpp"
            "#include <immintrin.h>\nint main(){ __m256i v = _mm256_set1_epi32(1); (void)v; return 0; }"
            COMPILE_DEFINITIONS /arch:AVX2
        )
        set(${RESULT_VAR} ${_avx2_check} PARENT_SCOPE)
    else()
        try_compile(_avx2_check
            SOURCE_FROM_CONTENT "avx2_check.cpp"
            "#include <immintrin.h>\nint main(){ __m256i v = _mm256_set1_epi32(1); (void)v; return 0; }"
            COMPILE_DEFINITIONS -mavx2 -mfma -mf16c
        )
        set(${RESULT_VAR} ${_avx2_check} PARENT_SCOPE)
    endif()
endfunction()

function(_check_sse42_support RESULT_VAR)
    if(MSVC)
        try_compile(_sse42_check
            SOURCE_FROM_CONTENT "sse42_check.cpp"
            "#include <nmmintrin.h>\nint main(){ __m128i v = _mm_set1_epi32(1); (void)_mm_crc32_u32(0, 1); return 0; }"
        )
        set(${RESULT_VAR} ${_sse42_check} PARENT_SCOPE)
    else()
        try_compile(_sse42_check
            SOURCE_FROM_CONTENT "sse42_check.cpp"
            "#include <nmmintrin.h>\nint main(){ __m128i v = _mm_set1_epi32(1); (void)_mm_crc32_u32(0, 1); return 0; }"
            COMPILE_DEFINITIONS -msse4.2
        )
        set(${RESULT_VAR} ${_sse42_check} PARENT_SCOPE)
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Detect AVX2 / SSE4.2
# ---------------------------------------------------------------------------
_check_avx2_support(HAS_AVX2)
_check_sse42_support(HAS_SSE42)

if(HAS_AVX2)
    message(STATUS "CompilerOptions: AVX2 support detected")
elseif(HAS_SSE42)
    message(STATUS "CompilerOptions: SSE4.2 support detected (no AVX2)")
else()
    message(STATUS "CompilerOptions: No AVX2/SSE4.2 detected — using baseline flags")
endif()

# ---------------------------------------------------------------------------
# Compiler-specific flags
# ---------------------------------------------------------------------------
if(MSVC)
    # --- AVX2 ---
    if(HAS_AVX2)
        add_compile_options(/arch:AVX2)
        add_compile_definitions(USE_AVX2)
    endif()

    # --- Warnings ---
    add_compile_options(/W4)

    # --- Source charset ---
    add_compile_options(/utf-8)

    # --- Optimization ---
    # Release
    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
        add_compile_options(/O2)
    endif()
    # Debug
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/Od /Zi)
    endif()

else()
    # --- AVX2 ---
    if(HAS_AVX2)
        add_compile_options(-mavx2 -mfma -mf16c)
        add_compile_definitions(USE_AVX2)
    elseif(HAS_SSE42)
        add_compile_options(-msse4.2)
    endif()

    # --- Warnings ---
    add_compile_options(-Wall -Wextra)

    # --- Optimization ---
    # Release
    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
        add_compile_options(-O2)
    endif()
    # Debug
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O0 -g)
    endif()
endif()
