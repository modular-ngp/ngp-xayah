#ifndef NGP_XAYAH_NGP_CUDA_UTILS_H
#define NGP_XAYAH_NGP_CUDA_UTILS_H

#include <tiny-cuda-nn/common.h>

namespace ngp::cuda::utils {
    constexpr uint32_t N_THREADS_LINEAR = 128;

    template <typename T>
    TCNN_HOST_DEVICE T div_round_up(T val, T divisor) {
        return (val + divisor - 1) / divisor;
    }

    template <typename T>
    constexpr TCNN_HOST_DEVICE uint32_t n_blocks_linear(T n_elements, uint32_t n_threads = N_THREADS_LINEAR) {
        return (uint32_t) div_round_up(n_elements, (T) n_threads);
    }

    template <typename K, typename T, typename... Types>
    inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types... args) {
        if (n_elements <= 0) {
            return;
        }
        kernel<<<n_blocks_linear(n_elements), N_THREADS_LINEAR, shmem_size, stream>>>(n_elements, args...);
    }

    template <typename T>
    TCNN_HOST_DEVICE T next_multiple(T val, T divisor) {
        return div_round_up(val, divisor) * divisor;
    }

    template <typename F>
    __global__ void parallel_for_kernel(const size_t n_elements, F fun) {
        const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        fun(i);
    }

    template <typename F>
    inline void parallel_for_gpu(uint32_t shmem_size, cudaStream_t stream, size_t n_elements, F&& fun) {
        if (n_elements <= 0) {
            return;
        }
        parallel_for_kernel<F><<<n_blocks_linear(n_elements), N_THREADS_LINEAR, shmem_size, stream>>>(n_elements, fun);
    }

    template <typename F>
    inline void parallel_for_gpu(cudaStream_t stream, size_t n_elements, F&& fun) {
        parallel_for_gpu(0, stream, n_elements, std::forward<F>(fun));
    }

    template <typename F>
    inline void parallel_for_gpu(size_t n_elements, F&& fun) {
        parallel_for_gpu(nullptr, n_elements, std::forward<F>(fun));
    }

    template <typename F>
    __global__ void parallel_for_aos_kernel(const size_t n_elements, const uint32_t n_dims, F fun) {
        const size_t dim  = threadIdx.x;
        const size_t elem = threadIdx.y + blockIdx.x * blockDim.y;
        if (dim >= n_dims) return;
        if (elem >= n_elements) return;

        fun(elem, dim);
    }

    template <typename F>
    inline void parallel_for_gpu_aos(uint32_t shmem_size, cudaStream_t stream, size_t n_elements, uint32_t n_dims, F&& fun) {
        if (n_elements <= 0 || n_dims <= 0) {
            return;
        }

        const dim3 threads     = {n_dims, div_round_up(N_THREADS_LINEAR, n_dims), 1};
        const size_t n_threads = threads.x * threads.y;
        const dim3 blocks      = {(uint32_t) div_round_up(n_elements * n_dims, n_threads), 1, 1};

        parallel_for_aos_kernel<<<blocks, threads, shmem_size, stream>>>(
            n_elements, n_dims, fun
            );
    }

    template <typename F>
    inline void parallel_for_gpu_aos(cudaStream_t stream, size_t n_elements, uint32_t n_dims, F&& fun) {
        parallel_for_gpu_aos(0, stream, n_elements, n_dims, std::forward<F>(fun));
    }

    template <typename F>
    inline void parallel_for_gpu_aos(size_t n_elements, uint32_t n_dims, F&& fun) {
        parallel_for_gpu_aos(nullptr, n_elements, n_dims, std::forward<F>(fun));
    }

    template <typename F>
    __global__ void parallel_for_soa_kernel(const size_t n_elements, const uint32_t n_dims, F fun) {
        const size_t elem = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t dim  = blockIdx.y;
        if (elem >= n_elements) return;
        if (dim >= n_dims) return;

        fun(elem, dim);
    }

    template <typename F>
    inline void parallel_for_gpu_soa(uint32_t shmem_size, cudaStream_t stream, size_t n_elements, uint32_t n_dims, F&& fun) {
        if (n_elements <= 0 || n_dims <= 0) {
            return;
        }

        const dim3 blocks = {n_blocks_linear(n_elements), n_dims, 1};

        parallel_for_soa_kernel<<<n_blocks_linear(n_elements), N_THREADS_LINEAR, shmem_size, stream>>>(
            n_elements, n_dims, fun
            );
    }

    template <typename F>
    inline void parallel_for_gpu_soa(cudaStream_t stream, size_t n_elements, uint32_t n_dims, F&& fun) {
        parallel_for_gpu_soa(0, stream, n_elements, n_dims, std::forward<F>(fun));
    }

    template <typename F>
    inline void parallel_for_gpu_soa(size_t n_elements, uint32_t n_dims, F&& fun) {
        parallel_for_gpu_soa(nullptr, n_elements, n_dims, std::forward<F>(fun));
    }
}

#endif //NGP_XAYAH_NGP_CUDA_UTILS_H