cmake_minimum_required(VERSION 4.00)
project(ngp-xayah VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API ON)

set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA archs" FORCE)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(ngp-xayah)
target_sources(ngp-xayah
        PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES
        modules/ngp.dataset.nerfsynthetic.ixx
        modules/ngp.device.ixx
)
target_sources(ngp-xayah
        PRIVATE
        cuda/ngp.device.cu
)
set_source_files_properties(cuda/ngp.device.cu PROPERTIES LANGUAGE CUDA)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(ngp-xayah PRIVATE CUDA::cudart)

include(FetchContent)
set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/_deps")
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

FetchContent_Declare(
        threadpool
        GIT_REPOSITORY https://github.com/modular-ngp/threadpool.git
        GIT_TAG master
)
FetchContent_MakeAvailable(threadpool)
target_link_libraries(ngp-xayah PRIVATE threadpool::threadpool)

FetchContent_Declare(
        args
        GIT_REPOSITORY https://github.com/Taywee/args.git
        GIT_TAG master
)
FetchContent_MakeAvailable(args)
target_link_libraries(ngp-xayah PRIVATE taywee::args)

FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG master
)
FetchContent_MakeAvailable(nlohmann_json)
target_link_libraries(ngp-xayah PRIVATE nlohmann_json::nlohmann_json)

FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG master
)
FetchContent_MakeAvailable(stb)
target_include_directories(ngp-xayah PRIVATE ${stb_SOURCE_DIR})
set(stb_impl "${CMAKE_CURRENT_BINARY_DIR}/stb_image_impl.cpp")
file(WRITE ${stb_impl} "#define STB_IMAGE_IMPLEMENTATION\n#include <stb_image.h>\n")
target_sources(ngp-xayah PRIVATE ${stb_impl})

add_executable(app main.cpp)
target_link_libraries(app PRIVATE ngp-xayah)