module;
#include <filesystem>
export module ngp.train;
import ngp.dataset;
namespace ngp::train {
    export void start_session(const std::filesystem::path& path);
    export void end_session();
    export void train(size_t batchsize);
}
