#ifndef NGP_XAYAH_NGP_CUDA_ENVMAP_CUH
#define NGP_XAYAH_NGP_CUDA_ENVMAP_CUH
#include "random_val.cuh"

#include <tiny-cuda-nn/common.h>

namespace ngp::cuda {
    using vec4  = tcnn::vec4;
    using ivec2 = tcnn::ivec2;

    template <typename T>
    struct Buffer2DView {
        T* data          = nullptr;
        ivec2 resolution = 0;

        // Lookup via integer pixel position (no bounds checking)
        TCNN_HOST_DEVICE T at(const ivec2& px) const {
            return data[px.x + px.y * resolution.x];
        }

        // Lookup via UV coordinates in [0,1]^2
        TCNN_HOST_DEVICE T at(const vec2& uv) const {
            ivec2 px = clamp(ivec2(vec2(resolution) * uv), 0, resolution - 1);
            return at(px);
        }

        // Lookup via UV coordinates in [0,1]^2 and LERP the nearest texels
        TCNN_HOST_DEVICE T at_lerp(const vec2& uv) const {
            const vec2 px_float = vec2(resolution) * uv;
            const ivec2 px      = ivec2(px_float);

            const vec2 weight = px_float - vec2(px);

            auto read_val = [&](ivec2 pos) {
                return at(clamp(pos, 0, resolution - 1));
            };

            return ((1 - weight.x) * (1 - weight.y) * read_val({px.x, px.y}) +
                    (weight.x) * (1 - weight.y) * read_val({px.x + 1, px.y}) +
                    (1 - weight.x) * (weight.y) * read_val({px.x, px.y + 1}) +
                    (weight.x) * (weight.y) * read_val({px.x + 1, px.y + 1}));
        }

        TCNN_HOST_DEVICE operator bool() const {
            return data;
        }
    };

    inline __device__ vec4 read_envmap(const Buffer2DView<const vec4>& envmap, const vec3& dir) {
        auto dir_cyl = dir_to_spherical_unorm({dir.z, -dir.x, dir.y});

        auto envmap_float  = vec2{dir_cyl.y * (envmap.resolution.x - 1), dir_cyl.x * (envmap.resolution.y - 1)};
        ivec2 envmap_texel = envmap_float;

        auto weight = envmap_float - vec2(envmap_texel);

        auto read_val = [&](ivec2 pos) {
            if (pos.x < 0) {
                pos.x += envmap.resolution.x;
            } else if (pos.x >= envmap.resolution.x) {
                pos.x -= envmap.resolution.x;
            }
            pos.y = max(min(pos.y, envmap.resolution.y - 1), 0);
            return envmap.at(pos);
        };

        auto result = (
            (1 - weight.x) * (1 - weight.y) * read_val({envmap_texel.x, envmap_texel.y}) +
            (weight.x) * (1 - weight.y) * read_val({envmap_texel.x + 1, envmap_texel.y}) +
            (1 - weight.x) * (weight.y) * read_val({envmap_texel.x, envmap_texel.y + 1}) +
            (weight.x) * (weight.y) * read_val({envmap_texel.x + 1, envmap_texel.y + 1})
        );

        return result;
    }

    template <typename T, typename GRAD_T>
    __device__ void deposit_envmap_gradient(const tcnn::tvec<T, 4>& value, GRAD_T* __restrict__ envmap_gradient, const ivec2 envmap_resolution, const vec3& dir) {
        auto dir_cyl = dir_to_spherical_unorm({dir.z, -dir.x, dir.y});

        auto envmap_float  = vec2{dir_cyl.y * (envmap_resolution.x - 1), dir_cyl.x * (envmap_resolution.y - 1)};
        ivec2 envmap_texel = envmap_float;

        auto weight = envmap_float - vec2(envmap_texel);

        auto deposit_val = [&](const tcnn::tvec<T, 4>& value, T weight, ivec2 pos) {
            if (pos.x < 0) {
                pos.x += envmap_resolution.x;
            } else if (pos.x >= envmap_resolution.x) {
                pos.x -= envmap_resolution.x;
            }
            pos.y = std::max(std::min(pos.y, envmap_resolution.y - 1), 0);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
            if (std::is_same<GRAD_T, __half>::value) {
                for (uint32_t c = 0; c < 4; c += 2) {
                    atomicAdd((__half2*) &envmap_gradient[(pos.x + pos.y * envmap_resolution.x) * 4 + c], {value[c] * weight, value[c + 1] * weight});
                }
            } else
#endif
            {
                for (uint32_t c = 0; c < 4; ++c) {
                    atomicAdd(&envmap_gradient[(pos.x + pos.y * envmap_resolution.x) * 4 + c], (GRAD_T) (value[c] * weight));
                }
            }
        };

        deposit_val(value, (1 - weight.x) * (1 - weight.y), {envmap_texel.x, envmap_texel.y});
        deposit_val(value, (weight.x) * (1 - weight.y), {envmap_texel.x + 1, envmap_texel.y});
        deposit_val(value, (1 - weight.x) * (weight.y), {envmap_texel.x, envmap_texel.y + 1});
        deposit_val(value, (weight.x) * (weight.y), {envmap_texel.x + 1, envmap_texel.y + 1});
    }
}

#endif //NGP_XAYAH_NGP_CUDA_ENVMAP_CUH
