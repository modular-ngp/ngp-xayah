import ngp.dataset;
import ngp.train;

int main() {
    const ngp::dataset::NeRFSyntheticDataset& dataset = ngp::dataset::load_nerf_synthetic_dataset("C:/Users/xayah/Desktop/instant-ngp/data/nerf-synthetic/lego/transforms_train.json");
    ngp::train::start_session("C:/Users/xayah/Desktop/instant-ngp/configs/nerf/base.json");
    ngp::train::train(10);
    ngp::train::end_session();
    return 0;
}
