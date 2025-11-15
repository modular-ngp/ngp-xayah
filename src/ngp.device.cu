#include <cstdio>
#include <cuda_runtime.h>
#include <tiny-cuda-nn/common.h>

__global__ void hello_kernel() {
    printf("Hello from CUDA kernel!\n");
}

void run_cuda_hello() {
    hello_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}