import ngp.dataset;
import ngp.train;
#include <print>

int main() {
    const ngp::dataset::NeRFSyntheticDataset& dataset = ngp::dataset::load_nerf_synthetic_dataset("C:/Users/xayah/Desktop/instant-ngp/data/nerf-synthetic/lego/transforms_train.json");
    const ngp::train::NGP_STATE& state                = ngp::train::query_ngp_state();
    ngp::train::init_with_tcnn_network_config("C:/Users/xayah/Desktop/instant-ngp/configs/nerf/base.json");
    std::println("Dataset loaded from path.");
    std::println("Current NGP State: {}", static_cast<int>(state));

    ngp::train::start_session(ngp::train::NGP_DATASET_TYPE::NERF_SYNTHETIC);
    return EXIT_SUCCESS;
}
