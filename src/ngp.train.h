#ifndef NGP_XAYAH_NGP_TRAIN_H
#define NGP_XAYAH_NGP_TRAIN_H

#include <json/json.hpp>

namespace ngp::train::cuda {
    void find_devices();
    void reset_context(const nlohmann::json& config);
}

#endif //NGP_XAYAH_NGP_TRAIN_H
