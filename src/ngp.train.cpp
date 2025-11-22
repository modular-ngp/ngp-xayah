module;
#include "ngp.train.h"
#include <json/json.hpp>
#include <fstream>
#include <print>
module ngp.train;

namespace ngp::train {
    void init_with_tcnn_network_config(const std::filesystem::path& path) {
        const auto& json             = nlohmann::json::parse(std::ifstream(path));
        const auto& loss_config      = json["loss"];
        const auto& optimizer_config = json["optimizer"];
        const auto& network_config   = json["network"];
        const auto& encoding_config  = json["encoding"];
        auto loss_config_expand = loss_config;
        loss_config_expand["otype"] = "L2";
        cuda::reset_loss(loss_config_expand);
        cuda::reset_optimizer(optimizer_config);
        cuda::reset_network(network_config);
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
