#ifndef NGP_XAYAH_NGP_CUDA_H
#define NGP_XAYAH_NGP_CUDA_H

#include <json/json.hpp>

namespace ngp::cuda {
    void reset_session(const nlohmann::json& config);
    void train(uint32_t batchsize);
}

#endif //NGP_XAYAH_NGP_CUDA_H