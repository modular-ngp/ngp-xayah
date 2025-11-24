#include "ngp.cuda.session.cuh"

namespace ngp::cuda {
    void reset_session(const nlohmann::json& config) {
        NGPSession::instance().reset_session(config);
    }

    void train(const uint32_t batchsize) {
        NGPSession::instance().train(batchsize);
    }
}
