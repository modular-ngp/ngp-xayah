#include "ngp.cuda.session.cuh"

namespace ngp::cuda {
    void find_devices() {
        printf("Finding CUDA devices...\n");
        const auto cuda_device_count = tcnn::cuda_device_count();
        printf("Found %d CUDA devices.\n", cuda_device_count);
        bool m_jit_fusion = tcnn::supports_jit_fusion();
        printf("JIT fusion is %s.\n", m_jit_fusion ? "supported" : "not supported");
    }

    void reset_session(const nlohmann::json& config) {
        NGPSession::instance().reset_session(config);
    }

    void train(const uint32_t batchsize) {
        NGPSession::instance().train(batchsize);
    }
}
