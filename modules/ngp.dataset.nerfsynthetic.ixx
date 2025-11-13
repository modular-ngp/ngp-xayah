module;
#include <array>
#include <filesystem>
#include <fstream>
#include <future>
#include <stb_image.h>

#include <nlohmann/json.hpp>

export module ngp.dataset.nerfsynthetic;
import threadpool;

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

    NeRFSyntheticDataset load_nerf_synthetic_dataset(const std::filesystem::path& path) {
        if (path.extension() != ".json") throw std::runtime_error("Not a json file");
        auto json = nlohmann::json::parse(std::ifstream(path));
        NeRFSyntheticDataset ds{
            .camera_angle_x   = json["camera_angle_x"],
            .image_file_path  = json["frames"] | std::views::transform([&](auto& f) { return (path.parent_path() / f["file_path"].template get<std::string>()).concat(".png"); }) | std::ranges::to<std::vector>(),
            .rotation         = json["frames"] | std::views::transform([](auto& f) { return f["rotation"].template get<float>(); }) | std::ranges::to<std::vector>(),
            .transform_matrix = json["frames"] | std::views::transform([](auto& f) {
                std::array<std::array<float, 4>, 4> m{};
                int r = 0;
                for (auto& row : f["transform_matrix"]) {
                    int c = 0;
                    for (auto& v : row) {
                        m[r][c++] = v.template get<float>();
                    }
                    r++;
                }
                return m;
            }) | std::ranges::to<std::vector>(),
            .image_data       = {},
            .wh               = {},
            .comp             = {},
        };
        for (auto& p : ds.image_file_path) p = std::filesystem::canonical(std::filesystem::absolute(p));
        ThreadPool pool;
        ds.image_data.resize(ds.image_file_path.size());
        ds.wh.resize(ds.image_file_path.size());
        ds.comp.resize(ds.image_file_path.size());
        std::vector<std::future<void>> futures;
        futures.reserve(ds.image_file_path.size());
        for (size_t i = 0; i < ds.image_file_path.size(); ++i) {
            futures.push_back(pool.enqueue([&, i] {
                int w = 0, h = 0, comp = 0;
                ds.image_data[i] = stbi_load(ds.image_file_path[i].string().c_str(), &w, &h, &comp, 4);
                ds.wh[i]         = {static_cast<float>(w), static_cast<float>(h)};
                ds.comp[i]       = comp;
            }));
        }
        for (auto& f : futures) f.get();
        pool.wait_idle();
        return ds;
    }

} // namespace ngp::dataset
