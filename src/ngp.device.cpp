module;
#include <tiny-cuda-nn/gpu_memory.h>
#include <iostream>

module ngp.device;

// Host-side function implemented in CUDA translation unit
void run_cuda_hello();

void say_hello() {
    std::cout << "Hello, world!\n";
    run_cuda_hello();
}