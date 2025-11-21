module;
#include <filesystem>
export module ngp.train;
import ngp.dataset;
namespace ngp::train {
    export enum class NGP_DATASET_TYPE {
        NERF_SYNTHETIC,
    };

    export void start_session(NGP_DATASET_TYPE type);
    export void end_session();
    export void init_with_tcnn_network_config(const std::filesystem::path& path);
    export void train(size_t epoch);

    export enum class NGP_STATE {
        IDLE,

        DATASET_INITIALIZING,
        DATASET_INITIALIZED,

        ENCODER_INITIALIZING,
        ENCODER_INITIALIZED,

        NERF_INITIALIZING,
        NERF_INITIALIZED,

        NGP_FULLY_INITIALIZED,

        TRAINING,

        INFERENCING,
    };

    export NGP_STATE query_ngp_state();
}
