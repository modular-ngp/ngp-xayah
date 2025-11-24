#ifndef NGP_XAYAH_NGP_CUDA_SESSION_H
#define NGP_XAYAH_NGP_CUDA_SESSION_H

#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/rtc_kernel.h>
#include <json/json.hpp>

namespace tcnn {
    template <typename T>
    class Loss;
    template <typename T>
    class Optimizer;
    template <typename T>
    class Encoding;
    template <typename T, typename PARAMS_T>
    class Network;
    template <typename T, typename PARAMS_T, typename COMPUTE_T>
    class Trainer;
    template <uint32_t N_DIMS, uint32_t RANK, typename T>
    class TrainableBuffer;
} // namespace tcnn

namespace ngp::cuda {

    struct NGPSession {
        static NGPSession& instance() {
            static NGPSession instance;
            return instance;
        }

        void reset_session(const nlohmann::json& config);
        void train(uint32_t batchsize);

    private:
        NGPSession()  = default;
        ~NGPSession() = default;

        struct NerfCounters {
            tcnn::GPUMemory<uint32_t> numsteps_counter; // number of steps each ray took
            tcnn::GPUMemory<uint32_t> numsteps_counter_compacted; // number of steps each ray took
            tcnn::GPUMemory<float> loss;

            uint32_t rays_per_batch                        = 1 << 12;
            uint32_t n_rays_total                          = 0;
            uint32_t measured_batch_size                   = 0;
            uint32_t measured_batch_size_before_compaction = 0;

            void prepare_for_training_steps(cudaStream_t stream);
            float update_after_training(uint32_t target_batch_size, bool get_loss_scalar, cudaStream_t stream);
        } m_counters_rgb;

        std::shared_ptr<tcnn::Loss<tcnn::network_precision_t>> m_loss;
        std::shared_ptr<tcnn::Optimizer<tcnn::network_precision_t>> m_optimizer;
        std::shared_ptr<tcnn::Network<float, tcnn::network_precision_t>> m_network;
        std::shared_ptr<tcnn::Encoding<tcnn::network_precision_t>> m_encoding;
        std::shared_ptr<tcnn::Trainer<float, tcnn::network_precision_t, tcnn::network_precision_t>> m_trainer;
        uint32_t m_seed = 1337;
        cudaStream_t m_stream;

        uint32_t m_training_step               = 0;
        uint32_t n_rays_since_error_map_update = 0;
        bool m_max_level_rand_training         = false;

        std::unique_ptr<tcnn::CudaRtcKernel> m_fused_kernel;
    };
}

#endif //NGP_XAYAH_NGP_CUDA_SESSION_H
