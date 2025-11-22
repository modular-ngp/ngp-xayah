#ifndef NGP_XAYAH_NGP_TRAIN_H
#define NGP_XAYAH_NGP_TRAIN_H

#include <json/json.hpp>

namespace ngp::train::cuda {
    void find_devices();
    void reset_loss(const nlohmann::json& loss_config);
    void reset_optimizer(const nlohmann::json& optimizer_config);
    void reset_encoding(const nlohmann::json& encoding_config);
    void reset_network(const nlohmann::json& network_config);
}

#endif //NGP_XAYAH_NGP_TRAIN_H
