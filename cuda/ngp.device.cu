#include "ngp.device.cuh"

#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel() {
    printf("Hello from CUDA kernel!\n");
}

void launch_kernel() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
