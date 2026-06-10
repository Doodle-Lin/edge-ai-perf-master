# 单元测试

find_package(GTest QUIET)

if(NOT GTest_FOUND)
    set(GTest_DIR "${CMAKE_SOURCE_DIR}/3rdparty/googletest")
    if(EXISTS "${GTest_DIR}/CMakeLists.txt")
        add_subdirectory(${GTest_DIR})
    else()
        message(STATUS "Google Test not found, skipping tests")
        return()
    endif()
endif()

# 收集测试源文件
file(GLOB TEST_SOURCES tests/*.cpp)

foreach(TEST_SRC ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SRC})
    target_include_directories(${TEST_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(${TEST_NAME} PRIVATE edgeai_core GTest::GTest GTest::Main)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
