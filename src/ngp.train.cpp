module;
#include "cuda/ngp.cuda.h"
#include <json/json.hpp>
#include <fstream>
#include <print>
module ngp.train;

namespace ngp::train {
    void start_session(NGP_DATASET_TYPE type) {
        cuda::find_devices();
    }

    void session_load_tcnn_config(const std::filesystem::path& path) {
        cuda::reset_session(nlohmann::json::parse(std::ifstream(path)));
    }

    void end_session() {
    }

    void train(const size_t batchsize) {
        cuda::train(batchsize);
    }

    NGP_STATE query_ngp_state() {
        return NGP_STATE::IDLE;
    }
}
