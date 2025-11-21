module;
#include "tiny-cuda-nn/gpu_memory.h"
#include <print>

module ngp.device;

namespace ngp::dataset {
    void to_device(const NeRFSyntheticDataset& dataset) {

        auto total_frames = dataset.image_data.size();
        std::println("Total frames: {}", total_frames);

        for (size_t i = 0; i < total_frames; ++i) {
            auto& wh               = dataset.wh[i];
            auto& comp             = dataset.comp[i];
            auto& transform_matrix = dataset.transform_matrix[i];
            auto& image_data       = dataset.image_data[i];
            std::println("Frame {}: w={}, h={}, comp={}", i, wh[0], wh[1], comp);

            tcnn::GPUMemory<uint8_t> images_data_gpu_tmp;
            images_data_gpu_tmp.copy_from_host((uint8_t*) image_data);
            void* pixels = images_data_gpu_tmp.data();
        }
    }
} // namespace ngp::dataset
