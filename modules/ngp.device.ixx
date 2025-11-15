export module ngp.device;

import ngp.dataset.nerfsynthetic;
namespace ngp::dataset {
    export void to_device(const NeRFSyntheticDataset& dataset);
} // namespace ngp::dataset
