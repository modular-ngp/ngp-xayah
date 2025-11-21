module;
#include "ngp.train.h"
#include <json/json.hpp>
#include <fstream>
#include <print>
module ngp.train;

namespace ngp::train {
    void init_with_tcnn_network_config(const std::filesystem::path& path) {
        const auto& json             = nlohmann::json::parse(std::ifstream(path));
        const auto& encoding_config  = json["encoding"];
        const auto& loss_config      = json["loss"];
        const auto& optimizer_config = json["optimizer"];
        const auto& network_config   = json["network"];
        std::println("encoding_config: {}", encoding_config.dump(4));
        std::println("loss_config: {}", loss_config.dump(4));
        std::println("optimizer_config: {}", optimizer_config.dump(4));
        std::println("network_config: {}", network_config.dump(4));
    }

    void start_session(NGP_DATASET_TYPE type) {
        cuda::find_devices();
    }

    void end_session() {

    }

    void train(size_t epoch) {
    }

    NGP_STATE query_ngp_state() {
        return NGP_STATE::IDLE;
    }
}
