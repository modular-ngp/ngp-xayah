module;
#include "cuda/ngp.cuda.h"
#include <json/json.hpp>
#include <fstream>
module ngp.train;

namespace ngp::train {
    void start_session(const std::filesystem::path& path) {
        cuda::reset_session(nlohmann::json::parse(std::ifstream(path)));
    }

    void end_session() {
    }

    void train(const size_t batchsize) {
        cuda::train(batchsize);
    }
}
