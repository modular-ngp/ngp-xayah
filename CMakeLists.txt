cmake_minimum_required(VERSION 3.26)
if (WIN32)
    include(cmake/setup_vcpkg.cmake)
endif ()
project(ngp-xayah VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA archs" FORCE)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(ngp-xayah main.cu)
target_include_directories(ngp-xayah
        PUBLIC
        include
        ${ZSTR_INCLUDE_DIRS}
)

# args
find_package(args CONFIG REQUIRED)
target_link_libraries(ngp-xayah PRIVATE taywee::args)

# stb
find_package(Stb REQUIRED)
target_include_directories(ngp-xayah PRIVATE ${Stb_INCLUDE_DIR})

# tinyobjloader
find_package(tinyobjloader CONFIG REQUIRED)
target_link_libraries(ngp-xayah PRIVATE tinyobjloader::tinyobjloader)

# ZLIB
find_package(ZLIB REQUIRED)
find_path(ZSTR_INCLUDE_DIRS "zstr.hpp")
target_link_libraries(ngp-xayah PRIVATE ZLIB::ZLIB)

# tiny-cuda-nn
include(FetchContent)
set(TCNN_ALLOW_CUBLAS_CUSOLVER OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_NO_FWD_BWD OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_WITH_RTC ON CACHE BOOL "" FORCE)
set(TCNN_BUILD_USE_FAST_MATH ON CACHE BOOL "" FORCE)
set(TCNN_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "" FORCE)
FetchContent_Declare(
        tiny-cuda-nn
        GIT_REPOSITORY https://github.com/NVlabs/tiny-cuda-nn.git
        GIT_TAG master
)
FetchContent_MakeAvailable(tiny-cuda-nn)
target_link_libraries(ngp-xayah PUBLIC tiny-cuda-nn)