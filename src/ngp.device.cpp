module;
#include "tiny-cuda-nn/gpu_memory.h"

module ngp.device;

namespace ngp::dataset {
    void to_device(const NeRFSyntheticDataset& dataset) {
        tcnn::GPUMemory<uint8_t> images_data_gpu_tmp;
    }
} // namespace ngp::dataset
