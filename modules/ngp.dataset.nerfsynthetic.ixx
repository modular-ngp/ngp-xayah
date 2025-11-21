module;
#include <array>
#include <vector>
#include <filesystem>

export module ngp.dataset.nerfsynthetic;

namespace ngp::dataset {
    export struct NeRFSyntheticDataset {
        float camera_angle_x;
        std::vector<std::filesystem::path> image_file_path;
        std::vector<float> rotation;
        std::vector<std::array<std::array<float, 4>, 4>> transform_matrix;
        std::vector<uint8_t*> image_data;
        std::vector<std::array<float, 2>> wh;
        std::vector<int> comp;
    };
    export NeRFSyntheticDataset load_nerf_synthetic_dataset(const std::filesystem::path& path);
} // namespace ngp::dataset
