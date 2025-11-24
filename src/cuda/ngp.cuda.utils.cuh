#ifndef NGP_XAYAH_NGP_CUDA_UTILS_H
#define NGP_XAYAH_NGP_CUDA_UTILS_H

#include "random_val.cuh"

#include <tiny-cuda-nn/common.h>


namespace ngp::cuda {
    enum class ELensMode : int {
        Perspective,
        OpenCV,
        FTheta,
        LatLong,
        OpenCVFisheye,
        Equirectangular,
        Orthographic,
    };

    struct Lens {
        ELensMode mode  = ELensMode::Perspective;
        float params[7] = {};

        TCNN_HOST_DEVICE bool is_360() const {
            return mode == ELensMode::Equirectangular || mode == ELensMode::LatLong;
        }

        TCNN_HOST_DEVICE bool supports_dlss() {
            return mode == ELensMode::LatLong || mode == ELensMode::Equirectangular || mode == ELensMode::Perspective ||
                   mode == ELensMode::Orthographic || mode == ELensMode::OpenCV || mode == ELensMode::OpenCVFisheye;
        }
    };

    struct TrainingXForm {
        TCNN_HOST_DEVICE bool operator==(const TrainingXForm& other) const {
            return start == other.start && end == other.end;
        }

        tcnn::mat4x3 start;
        tcnn::mat4x3 end;
    };

    enum class ELossType : int {
        L2,
        L1,
        Mape,
        Smape,
        Huber,
        LogL1,
        RelativeL2,
    };

    enum class ENerfActivation : int {
        None,
        ReLU,
        Logistic,
        Exponential,
    };

    inline TCNN_HOST_DEVICE uint32_t binary_search(float val, const float* data, uint32_t length)
    {
        if (length == 0)
        {
            return 0;
        }

        uint32_t it;
        uint32_t count, step;
        count = length;

        uint32_t first = 0;
        while (count > 0)
        {
            it = first;
            step = count / 2;
            it += step;
            if (data[it] < val)
            {
                first = ++it;
                count -= step + 1;
            }
            else
            {
                count = step;
            }
        }

        return min(first, length - 1);
    }
}

namespace ngp::cuda {
    // The maximum depth that can be produced when rendering a frame.
    // Chosen somewhat low (rather than std::numeric_limits<float>::infinity())
    // to permit numerically stable reprojection and DLSS operation,
    // even when rendering the infinitely distant horizon.
    inline constexpr __device__ float MAX_DEPTH() {
        return 16384.0f;
    }

    template <typename T, int MAX_SIZE = 32>
    class FixedStack {
    public:
        TCNN_HOST_DEVICE void push(T val) {
            if (m_count >= MAX_SIZE - 1) {
                printf("WARNING TOO BIG\n");
            }

            m_elems[m_count++] = val;
        }

        TCNN_HOST_DEVICE T pop() {
            return m_elems[--m_count];
        }

        TCNN_HOST_DEVICE bool empty() const {
            return m_count <= 0;
        }

    private:
        T m_elems[MAX_SIZE];
        int m_count = 0;
    };

    using FixedIntStack = FixedStack<int>;

    inline TCNN_HOST_DEVICE float srgb_to_linear(float srgb) {
        if (srgb <= 0.04045f) {
            return srgb / 12.92f;
        } else {
            return pow((srgb + 0.055f) / 1.055f, 2.4f);
        }
    }

    inline TCNN_HOST_DEVICE vec3 srgb_to_linear(const vec3& x) {
        return {srgb_to_linear(x.x), srgb_to_linear(x.y), (srgb_to_linear(x.z))};
    }

    inline TCNN_HOST_DEVICE float srgb_to_linear_derivative(float srgb) {
        if (srgb <= 0.04045f) {
            return 1.0f / 12.92f;
        } else {
            return 2.4f / 1.055f * pow((srgb + 0.055f) / 1.055f, 1.4f);
        }
    }

    inline TCNN_HOST_DEVICE vec3 srgb_to_linear_derivative(const vec3& x) {
        return {srgb_to_linear_derivative(x.x), srgb_to_linear_derivative(x.y), (srgb_to_linear_derivative(x.z))};
    }

    inline TCNN_HOST_DEVICE float linear_to_srgb(float linear) {
        if (linear < 0.0031308f) {
            return 12.92f * linear;
        } else {
            return 1.055f * pow(linear, 0.41666f) - 0.055f;
        }
    }

    inline TCNN_HOST_DEVICE vec3 linear_to_srgb(const vec3& x) {
        return {linear_to_srgb(x.x), linear_to_srgb(x.y), (linear_to_srgb(x.z))};
    }

    inline TCNN_HOST_DEVICE float linear_to_srgb_derivative(float linear) {
        if (linear < 0.0031308f) {
            return 12.92f;
        } else {
            return 1.055f * 0.41666f * pow(linear, 0.41666f - 1.0f);
        }
    }

    inline TCNN_HOST_DEVICE vec3 linear_to_srgb_derivative(const vec3& x) {
        return {linear_to_srgb_derivative(x.x), linear_to_srgb_derivative(x.y), (linear_to_srgb_derivative(x.z))};
    }

    template <typename T>
    __device__ void deposit_image_gradient(const vec2& value, T* __restrict__ gradient, T* __restrict__ gradient_weight,
        const ivec2& resolution, const vec2& pos) {
        const vec2 pos_float = vec2(resolution) * pos;
        const ivec2 texel    = {pos_float};

        const vec2 weight = pos_float - vec2(texel);

        constexpr uint32_t N_DIMS = 2;

        auto deposit_val = [&](const vec2& value, T weight, ivec2 pos) {
            pos.x = max(min(pos.x, resolution.x - 1), 0);
            pos.y = max(min(pos.y, resolution.y - 1), 0);

#if defined(__CUDA_ARCH__) &&                                                                                          \
    __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
            if (std::is_same<T, __half>::value) {
                for (uint32_t c = 0; c < N_DIMS; c += 2) {
                    atomicAdd((__half2*) &gradient[(pos.x + pos.y * resolution.x) * N_DIMS + c],
                        {(T) value[c] * weight, (T) value[c + 1] * weight});
                    atomicAdd((__half2*) &gradient_weight[(pos.x + pos.y * resolution.x) * N_DIMS + c],
                        {weight, weight});
                }
            } else
#endif
            {
                for (uint32_t c = 0; c < N_DIMS; ++c) {
                    atomicAdd(&gradient[(pos.x + pos.y * resolution.x) * N_DIMS + c], (T) value[c] * weight);
                    atomicAdd(&gradient_weight[(pos.x + pos.y * resolution.x) * N_DIMS + c], weight);
                }
            }
        };

        deposit_val(value, (1 - weight.x) * (1 - weight.y), {texel.x, texel.y});
        deposit_val(value, (weight.x) * (1 - weight.y), {texel.x + 1, texel.y});
        deposit_val(value, (1 - weight.x) * (weight.y), {texel.x, texel.y + 1});
        deposit_val(value, (weight.x) * (weight.y), {texel.x + 1, texel.y + 1});
    }

    struct FoveationPiecewiseQuadratic {
        FoveationPiecewiseQuadratic() = default;
        TCNN_HOST_DEVICE FoveationPiecewiseQuadratic(float center_pixel_steepness, float center_inverse_piecewise_y,
            float center_radius) {
            float center_inverse_radius          = center_radius * center_pixel_steepness;
            float left_inverse_piecewise_switch  = center_inverse_piecewise_y - center_inverse_radius;
            float right_inverse_piecewise_switch = center_inverse_piecewise_y + center_inverse_radius;

            if (left_inverse_piecewise_switch < 0) {
                left_inverse_piecewise_switch = 0.0f;
            }

            if (right_inverse_piecewise_switch > 1) {
                right_inverse_piecewise_switch = 1.0f;
            }

            float am = center_pixel_steepness;
            float d  = (right_inverse_piecewise_switch - left_inverse_piecewise_switch) / center_pixel_steepness / 2;

            // binary search for l,r,bm since analytical is very complex
            float bm;
            float m_min = 0.0f;
            float m_max = 1.0f;
            for (uint32_t i = 0; i < 20; i++) {
                float m = (m_min + m_max) / 2.0f;
                float l = m - d;
                float r = m + d;

                bm = -((am - 1) * l * l) / (r * r - 2 * r + l * l + 1);

                float l_actual = (left_inverse_piecewise_switch - bm) / am;
                float r_actual = (right_inverse_piecewise_switch - bm) / am;
                float m_actual = (l_actual + r_actual) / 2;

                if (m_actual > m) {
                    m_min = m;
                } else {
                    m_max = m;
                }
            }

            float l = (left_inverse_piecewise_switch - bm) / am;
            float r = (right_inverse_piecewise_switch - bm) / am;

            // Full linear case. Default construction covers this.
            if ((l == 0.0f && r == 1.0f) || (am == 1.0f)) {
                return;
            }

            // write out solution
            switch_left  = l;
            switch_right = r;
            this->am     = am;
            al           = (am - 1) / (r * r - 2 * r + l * l + 1);
            bl           = (am * (r * r - 2 * r + 1) + am * l * l + (2 - 2 * am) * l) / (r * r - 2 * r + l * l + 1);
            cl           = 0;
            this->bm     = bm = -((am - 1) * l * l) / (r * r - 2 * r + l * l + 1);
            ar           = -(am - 1) / (r * r - 2 * r + l * l + 1);
            br           = (am * (r * r + 1) - 2 * r + am * l * l) / (r * r - 2 * r + l * l + 1);
            cr           = -(am * r * r - r * r + (am - 1) * l * l) / (r * r - 2 * r + l * l + 1);

            inv_switch_left  = am * switch_left + bm;
            inv_switch_right = am * switch_right + bm;
        }

        // left parabola: al * x^2 + bl * x + cl
        float al = 0.0f, bl = 0.0f, cl = 0.0f;
        // middle linear piece: am * x + bm.  am should give 1:1 pixel mapping between warped size and full size.
        float am = 1.0f, bm = 0.0f;
        // right parabola: al * x^2 + bl * x + cl
        float ar = 0.0f, br = 0.0f, cr = 0.0f;

        // points where left and right switch over from quadratic to linear
        float switch_left = 0.0f, switch_right = 1.0f;
        // same, in inverted space
        float inv_switch_left = 0.0f, inv_switch_right = 1.0f;

        TCNN_HOST_DEVICE float warp(float x) const {
            x = clamp(x, 0.0f, 1.0f);
            if (x < switch_left) {
                return al * x * x + bl * x + cl;
            } else if (x > switch_right) {
                return ar * x * x + br * x + cr;
            } else {
                return am * x + bm;
            }
        }

        TCNN_HOST_DEVICE float unwarp(float y) const {
            y = clamp(y, 0.0f, 1.0f);
            if (y < inv_switch_left) {
                return (sqrt(-4 * al * cl + 4 * al * y + bl * bl) - bl) / (2 * al);
            } else if (y > inv_switch_right) {
                return (sqrt(-4 * ar * cr + 4 * ar * y + br * br) - br) / (2 * ar);
            } else {
                return (y - bm) / am;
            }
        }

        TCNN_HOST_DEVICE float density(float x) const {
            x = clamp(x, 0.0f, 1.0f);
            if (x < switch_left) {
                return 2 * al * x + bl;
            } else if (x > switch_right) {
                return 2 * ar * x + br;
            } else {
                return am;
            }
        }
    };

    struct Foveation {
        Foveation() = default;

        TCNN_HOST_DEVICE Foveation(const vec2& center_pixel_steepness, const vec2& center_inverse_piecewise_y,
            const vec2& center_radius) :
            warp_x{center_pixel_steepness.x, center_inverse_piecewise_y.x, center_radius.x},
            warp_y{center_pixel_steepness.y, center_inverse_piecewise_y.y, center_radius.y} {
        }

        FoveationPiecewiseQuadratic warp_x, warp_y;

        TCNN_HOST_DEVICE vec2 warp(const vec2& x) const {
            return {warp_x.warp(x.x), warp_y.warp(x.y)};
        }

        TCNN_HOST_DEVICE vec2 unwarp(const vec2& y) const {
            return {warp_x.unwarp(y.x), warp_y.unwarp(y.y)};
        }

        TCNN_HOST_DEVICE float density(const vec2& x) const {
            return warp_x.density(x.x) * warp_y.density(x.y);
        }
    };

    template <typename T>
    TCNN_HOST_DEVICE inline void opencv_lens_distortion_delta(const T* extra_params, const T u, const T v, T* du, T* dv) {
        const T k1 = extra_params[0];
        const T k2 = extra_params[1];
        const T p1 = extra_params[2];
        const T p2 = extra_params[3];

        const T u2     = u * u;
        const T uv     = u * v;
        const T v2     = v * v;
        const T r2     = u2 + v2;
        const T radial = k1 * r2 + k2 * r2 * r2;
        *du            = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2);
        *dv            = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2);
    }

    template <typename T>
    TCNN_HOST_DEVICE inline void opencv_fisheye_lens_distortion_delta(const T* extra_params, const T u, const T v, T* du,
        T* dv) {
        const T k1 = extra_params[0];
        const T k2 = extra_params[1];
        const T k3 = extra_params[2];
        const T k4 = extra_params[3];

        const T r = sqrt(u * u + v * v);

        if (r > (T) std::numeric_limits<double>::epsilon()) {
            const T theta  = atan(r);
            const T theta2 = theta * theta;
            const T theta4 = theta2 * theta2;
            const T theta6 = theta4 * theta2;
            const T theta8 = theta4 * theta4;
            const T thetad = theta * (T(1) + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
            *du            = u * thetad / r - u;
            *dv            = v * thetad / r - v;
        } else {
            *du = T(0);
            *dv = T(0);
        }
    }

    template <typename T, typename F>
    TCNN_HOST_DEVICE inline void iterative_lens_undistortion(const T* params, T* u, T* v, F distortion_fun) {
        // Parameters for Newton iteration using numerical differentiation with
        // central differences, 100 iterations should be enough even for complex
        // camera models with higher order terms.
        const uint32_t kNumIterations = 100;
        const float kMaxStepNorm      = 1e-10f;
        const float kRelStepSize      = 1e-6f;

        mat2 J;
        const vec2 x0{*u, *v};
        vec2 x{*u, *v};
        vec2 dx;
        vec2 dx_0b;
        vec2 dx_0f;
        vec2 dx_1b;
        vec2 dx_1f;

        for (uint32_t i = 0; i < kNumIterations; ++i) {
            const float step0 = max(std::numeric_limits<float>::epsilon(), abs(kRelStepSize * x[0]));
            const float step1 = max(std::numeric_limits<float>::epsilon(), abs(kRelStepSize * x[1]));
            distortion_fun(params, x[0], x[1], &dx[0], &dx[1]);
            distortion_fun(params, x[0] - step0, x[1], &dx_0b[0], &dx_0b[1]);
            distortion_fun(params, x[0] + step0, x[1], &dx_0f[0], &dx_0f[1]);
            distortion_fun(params, x[0], x[1] - step1, &dx_1b[0], &dx_1b[1]);
            distortion_fun(params, x[0], x[1] + step1, &dx_1f[0], &dx_1f[1]);
            J[0][0]           = 1 + (dx_0f[0] - dx_0b[0]) / (2 * step0);
            J[1][0]           = (dx_1f[0] - dx_1b[0]) / (2 * step1);
            J[0][1]           = (dx_0f[1] - dx_0b[1]) / (2 * step0);
            J[1][1]           = 1 + (dx_1f[1] - dx_1b[1]) / (2 * step1);
            const vec2 step_x = inverse(J) * (x + dx - x0);
            x -= step_x;
            if (length2(step_x) < kMaxStepNorm) {
                break;
            }
        }

        *u = x[0];
        *v = x[1];
    }

    template <typename T>
    TCNN_HOST_DEVICE inline void iterative_opencv_lens_undistortion(const T* params, T* u, T* v) {
        iterative_lens_undistortion(params, u, v, opencv_lens_distortion_delta<T>);
    }

    template <typename T>
    TCNN_HOST_DEVICE inline void iterative_opencv_fisheye_lens_undistortion(const T* params, T* u, T* v) {
        iterative_lens_undistortion(params, u, v, opencv_fisheye_lens_distortion_delta<T>);
    }

    inline TCNN_HOST_DEVICE Ray pixel_to_ray_pinhole(uint32_t spp, const ivec2& pixel, const ivec2& resolution,
        const vec2& focal_length, const mat4x3& camera_matrix,
        const vec2& screen_center) {
        const vec2 uv = vec2(pixel) / vec2(resolution);

        vec3 dir = {(uv.x - screen_center.x) * (float) resolution.x / focal_length.x,
                    (uv.y - screen_center.y) * (float) resolution.y / focal_length.y, 1.0f};

        dir = mat3(camera_matrix) * dir;
        return {camera_matrix[3], dir};
    }

    inline TCNN_HOST_DEVICE vec3 f_theta_undistortion(const vec2& uv, const float* params, const vec3& error_direction) {
        // we take f_theta intrinsics to be: r0, r1, r2, r3, resx, resy; we rescale to whatever res the intrinsics
        // specify.
        float xpix  = uv.x * params[5];
        float ypix  = uv.y * params[6];
        float norm  = sqrtf(xpix * xpix + ypix * ypix);
        float alpha = params[0] + norm * (params[1] + norm * (params[2] + norm * (params[3] + norm * params[4])));
        float sin_alpha, cos_alpha;
        sincosf(alpha, &sin_alpha, &cos_alpha);
        if (cos_alpha <= std::numeric_limits<float>::min() || norm == 0.f) {
            return error_direction;
        }
        sin_alpha *= 1.f / norm;
        return {sin_alpha * xpix, sin_alpha * ypix, cos_alpha};
    }

    inline TCNN_HOST_DEVICE vec3 latlong_to_dir(const vec2& uv) {
        float theta = (uv.y - 0.5f) * PI();
        float phi   = (uv.x - 0.5f) * PI() * 2.0f;
        float sp, cp, st, ct;
        sincosf(theta, &st, &ct);
        sincosf(phi, &sp, &cp);
        return {sp * ct, st, cp * ct};
    }

    inline TCNN_HOST_DEVICE vec3 equirectangular_to_dir(const vec2& uv) {
        float ct  = (uv.y - 0.5f) * 2.0f;
        float st  = sqrt(max(1.0f - ct * ct, 0.0f));
        float phi = (uv.x - 0.5f) * PI() * 2.0f;
        float sp, cp;
        sincosf(phi, &sp, &cp);
        return {sp * st, ct, cp * st};
    }

    inline TCNN_HOST_DEVICE vec2 dir_to_latlong(const vec3& dir) {
        float theta = asin(dir.y);
        float phi   = atan2(dir.x, dir.z);
        return {phi / (PI() * 2.0f) + 0.5f, theta / PI() + 0.5f};
    }

    inline TCNN_HOST_DEVICE vec2 dir_to_equirectangular(const vec3& dir) {
        float ct  = dir.y;
        float phi = atan2(dir.x, dir.z);
        return {phi / (PI() * 2.0f) + 0.5f, ct / 2.0f + 0.5f};
    }

    inline TCNN_HOST_DEVICE Ray uv_to_ray(uint32_t spp, const vec2& uv, const ivec2& resolution,
        const vec2& focal_length, const mat4x3& camera_matrix,
        const vec2& screen_center, const vec3& parallax_shift = vec3(0.0f),
        float near_distance                                   = 0.0f, float focus_z = 1.0f, float aperture_size = 0.0f,
        const Foveation& foveation                            = {},
        Buffer2DView<const uint8_t> hidden_area_mask          = {}, const Lens& lens = {},
        Buffer2DView<const vec2> distortion                   = {}) {
        vec2 warped_uv = foveation.warp(uv);

        // Check the hidden area mask _after_ applying foveation, because foveation will be undone
        // before blitting to the framebuffer to which the hidden area mask corresponds.
        if (hidden_area_mask && !hidden_area_mask.at(warped_uv)) {
            return Ray::invalid();
        }

        vec3 head_pos = {parallax_shift.x, parallax_shift.y, 0.f};
        vec3 dir;
        if (lens.mode == ELensMode::FTheta) {
            dir = f_theta_undistortion(warped_uv - screen_center, lens.params, {0.f, 0.f, 0.f});
            if (dir == vec3(0.0f)) {
                return Ray::invalid();
            }
        } else if (lens.mode == ELensMode::LatLong) {
            dir = latlong_to_dir(warped_uv);
        } else if (lens.mode == ELensMode::Equirectangular) {
            dir = equirectangular_to_dir(warped_uv);
        } else if (lens.mode == ELensMode::Orthographic) {
            dir = {0.0f, 0.0f, 1.0f};
            head_pos += vec3{
                (warped_uv.x - screen_center.x) * (float) resolution.x / focal_length.x,
                (warped_uv.y - screen_center.y) * (float) resolution.y / focal_length.y,
                0.0f,
            };
        } else {
            dir = {(warped_uv.x - screen_center.x) * (float) resolution.x / focal_length.x,
                   (warped_uv.y - screen_center.y) * (float) resolution.y / focal_length.y, 1.0f};

            if (lens.mode == ELensMode::OpenCV) {
                iterative_opencv_lens_undistortion(lens.params, &dir.x, &dir.y);
            } else if (lens.mode == ELensMode::OpenCVFisheye) {
                iterative_opencv_fisheye_lens_undistortion(lens.params, &dir.x, &dir.y);
            }
        }

        if (distortion) {
            dir.xy() += distortion.at_lerp(warped_uv);
        }

        if (lens.mode != ELensMode::Orthographic && lens.mode != ELensMode::LatLong &&
            lens.mode != ELensMode::Equirectangular) {
            dir -= head_pos *
                parallax_shift.z; // we could use focus_z here in the denominator. for now, we pack m_scale in here.
        }

        dir = mat3(camera_matrix) * dir;

        vec3 origin = mat3(camera_matrix) * head_pos + camera_matrix[3];
        if (aperture_size != 0.0f) {
            vec3 lookat = origin + dir * focus_z;
            auto px     = ivec2(uv * vec2(resolution));
            vec2 blur   = aperture_size *
                          square2disk_shirley(ld_random_val_2d(spp, px.x * 19349663 + px.y * 96925573) * 2.0f - 1.0f);
            origin += mat2x3(camera_matrix) * blur;
            dir = (lookat - origin) / focus_z;
        }

        origin += dir * near_distance;
        return {origin, dir};
    }

    inline TCNN_HOST_DEVICE Ray pixel_to_ray(uint32_t spp, const ivec2& pixel, const ivec2& resolution,
        const vec2& focal_length, const mat4x3& camera_matrix,
        const vec2& screen_center, const vec3& parallax_shift = vec3(0.0f),
        bool snap_to_pixel_centers                            = false, float near_distance = 0.0f,
        float focus_z                                         = 1.0f, float aperture_size  = 0.0f,
        const Foveation& foveation                            = {},
        Buffer2DView<const uint8_t> hidden_area_mask          = {}, const Lens& lens = {},
        Buffer2DView<const vec2> distortion                   = {}) {
        return uv_to_ray(spp,
            (vec2(pixel) + ld_random_pixel_offset(snap_to_pixel_centers ? 0 : spp)) / vec2(resolution),
            resolution, focal_length, camera_matrix, screen_center, parallax_shift, near_distance, focus_z,
            aperture_size, foveation, hidden_area_mask, lens, distortion);
    }

    inline TCNN_HOST_DEVICE vec2 pos_to_uv(const vec3& pos, const ivec2& resolution, const vec2& focal_length,
        const mat4x3& camera_matrix, const vec2& screen_center,
        const vec3& parallax_shift, const Foveation& foveation = {},
        const Lens& lens                                       = {}) {
        vec3 head_pos = {parallax_shift.x, parallax_shift.y, 0.f};
        vec2 uv;

        if (lens.mode == ELensMode::Orthographic) {
            vec3 rel_pos = inverse(mat3(camera_matrix)) * (pos - camera_matrix[3]) - head_pos;
            uv           = rel_pos.xy() * focal_length / vec2(resolution) + screen_center;
        } else {
            // Express ray in terms of camera frame
            vec3 origin = mat3(camera_matrix) * head_pos + camera_matrix[3];

            vec3 dir = pos - origin;
            dir      = inverse(mat3(camera_matrix)) * dir;
            dir /= lens.is_360() ? length(dir) : dir.z;

            if (lens.mode == ELensMode::Equirectangular) {
                uv = dir_to_equirectangular(dir);
            } else if (lens.mode == ELensMode::LatLong) {
                uv = dir_to_latlong(dir);
            } else {
                // Perspective with potential distortions applied on top
                dir += head_pos * parallax_shift.z;

                float du = 0.0f, dv = 0.0f;
                if (lens.mode == ELensMode::OpenCV) {
                    opencv_lens_distortion_delta(lens.params, dir.x, dir.y, &du, &dv);
                } else if (lens.mode == ELensMode::OpenCVFisheye) {
                    opencv_fisheye_lens_distortion_delta(lens.params, dir.x, dir.y, &du, &dv);
                } else {
                    // No other type of distortion is permitted.
                    assert(lens.mode == ELensMode::Perspective);
                }

                dir.x += du;
                dir.y += dv;

                uv = dir.xy() * focal_length / vec2(resolution) + screen_center;
            }
        }

        return foveation.unwarp(uv);
    }

    inline TCNN_HOST_DEVICE vec2 pos_to_pixel(const vec3& pos, const ivec2& resolution, const vec2& focal_length,
        const mat4x3& camera_matrix, const vec2& screen_center,
        const vec3& parallax_shift, const Foveation& foveation = {},
        const Lens& lens                                       = {}) {
        return pos_to_uv(pos, resolution, focal_length, camera_matrix, screen_center, parallax_shift, foveation, lens) *
               vec2(resolution);
    }

    inline TCNN_HOST_DEVICE vec2 motion_vector(const uint32_t sample_index, const ivec2& pixel, const ivec2& resolution,
        const vec2& focal_length, const mat4x3& camera, const mat4x3& prev_camera,
        const vec2& screen_center, const vec3& parallax_shift,
        const bool snap_to_pixel_centers, const float depth,
        const Foveation& foveation = {}, const Foveation& prev_foveation = {},
        const Lens& lens           = {}) {
        vec2 pxf = vec2(pixel) + ld_random_pixel_offset(snap_to_pixel_centers ? 0 : sample_index);
        Ray ray  = uv_to_ray(sample_index, pxf / vec2(resolution), resolution, focal_length, camera, screen_center,
            parallax_shift, 0.0f, 1.0f, 0.0f, foveation, {}, // No hidden area mask
            lens);

        vec2 prev_pxf = pos_to_pixel(ray(depth), resolution, focal_length, prev_camera, screen_center, parallax_shift,
            prev_foveation, lens);

        return prev_pxf - pxf;
    }

    // Maps view-space depth (physical units) in the range [znear, zfar] hyperbolically to
    // the interval [1, 0]. This is the reverse-z-component of "normalized device coordinates",
    // which are commonly used in rasterization, where linear interpolation in screen space
    // has to be equivalent to linear interpolation in real space (which, in turn, is
    // guaranteed by the hyperbolic mapping of depth). This format is commonly found in
    // z-buffers, and hence expected by downstream image processing functions, such as DLSS
    // and VR reprojection.
    inline TCNN_HOST_DEVICE float to_ndc_depth(float z, float n, float f) {
        // View depth outside of the view frustum leads to output outside of [0, 1]
        z = clamp(z, n, f);

        float scale = n / (n - f);
        float bias  = -f * scale;
        return clamp((z * scale + bias) / z, 0.0f, 1.0f);
    }

    inline TCNN_HOST_DEVICE float fov_to_focal_length(int resolution, float degrees) {
        return 0.5f * (float) resolution / tanf(0.5f * degrees * PI() / 180.0f);
    }

    inline TCNN_HOST_DEVICE vec2 fov_to_focal_length(const ivec2& resolution, const vec2& degrees) {
        return 0.5f * vec2(resolution) / tan(0.5f * degrees * (PI() / 180.0f));
    }

    inline TCNN_HOST_DEVICE float focal_length_to_fov(int resolution, float focal_length) {
        return 2.0f * 180.0f / PI() * atanf(float(resolution) / (focal_length * 2.0f));
    }

    inline TCNN_HOST_DEVICE vec2 focal_length_to_fov(const ivec2& resolution, const vec2& focal_length) {
        return 2.0f * 180.0f / PI() * atan(vec2(resolution) / (focal_length * 2.0f));
    }

    inline TCNN_HOST_DEVICE mat4x3 camera_log_lerp(const mat4x3& a, const mat4x3& b, float t) {
        return mat_exp(mat_log(mat4(b) * inverse(mat4(a))) * t) * mat4(a);
    }

    inline TCNN_HOST_DEVICE mat4x3 camera_slerp(const mat4x3& a, const mat4x3& b, float t) {
        mat3 rot = slerp(mat3(a), mat3(b), t);
        return {rot[0], rot[1], rot[2], mix(a[3], b[3], t)};
    }

    inline TCNN_HOST_DEVICE mat4x3 get_xform_given_rolling_shutter(const TrainingXForm& training_xform,
        const vec4& rolling_shutter, const vec2& uv,
        float motionblur_time) {
        float pixel_t = rolling_shutter.x + rolling_shutter.y * uv.x + rolling_shutter.z * uv.y +
                        rolling_shutter.w * motionblur_time;
        return camera_slerp(training_xform.start, training_xform.end, pixel_t);
    }

    inline TCNN_HOST_DEVICE void apply_quilting(uint32_t* x, uint32_t* y, const ivec2& resolution, vec3& parallax_shift,
        const ivec2& quilting_dims) {
        float resx = float(resolution.x) / quilting_dims.x;
        float resy = float(resolution.y) / quilting_dims.y;
        int panelx = (int) floorf(*x / resx);
        int panely = (int) floorf(*y / resy);
        *x         = (*x - panelx * resx);
        *y         = (*y - panely * resy);
        int idx    = panelx + quilting_dims.x * panely;

        if (quilting_dims == ivec2{2, 1}) {
            // Likely VR: parallax_shift.x is the IPD in this case. The following code centers the camera matrix between
            // both eyes. idx == 0 -> left eye -> -1/2 x
            parallax_shift.x = (idx == 0) ? (-0.5f * parallax_shift.x) : (0.5f * parallax_shift.x);
        } else {
            // Likely HoloPlay lenticular display: in this case, `parallax_shift.z` is the inverse height of the head
            // above the display. The following code computes the x-offset of views as a function of this.
            const float max_parallax_angle =
                17.5f; // suggested value in https://docs.lookingglassfactory.com/keyconcepts/camera
            float parallax_angle = max_parallax_angle * PI() / 180.f *
                                   ((idx + 0.5f) * 2.f / float(quilting_dims.y * quilting_dims.x) - 1.f);
            parallax_shift.x = atanf(parallax_angle) / parallax_shift.z;
        }
    }

    template <typename T>
    __global__ void from_rgba32(const uint64_t num_pixels, const uint8_t* __restrict__ pixels, T* __restrict__ out,
        bool white_2_transparent = false, bool black_2_transparent = false,
        uint32_t mask_color      = 0) {
        const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= num_pixels) {
            return;
        }

        uint8_t rgba[4];
        *((uint32_t*) &rgba[0]) = *((uint32_t*) &pixels[i * 4]);

        float alpha = rgba[3] * (1.0f / 255.0f);
        // NSVF dataset has 'white = transparent' madness
        if (white_2_transparent && rgba[0] == 255 && rgba[1] == 255 && rgba[2] == 255) {
            alpha = 0.f;
        }
        if (black_2_transparent && rgba[0] == 0 && rgba[1] == 0 && rgba[2] == 0) {
            alpha = 0.f;
        }

        tvec<T, 4> rgba_out;
        rgba_out[0] = (T) (srgb_to_linear(rgba[0] * (1.0f / 255.0f)) * alpha);
        rgba_out[1] = (T) (srgb_to_linear(rgba[1] * (1.0f / 255.0f)) * alpha);
        rgba_out[2] = (T) (srgb_to_linear(rgba[2] * (1.0f / 255.0f)) * alpha);
        rgba_out[3] = (T) alpha;

        if (mask_color != 0 && mask_color == *((uint32_t*) &rgba[0])) {
            rgba_out[0] = rgba_out[1] = rgba_out[2] = rgba_out[3] = (T) -1.0f;
        }

        *((tvec<T, 4>*) &out[i * 4]) = rgba_out;
    }

    // Foley & van Dam p593 / http://en.wikipedia.org/wiki/HSL_and_HSV
    inline TCNN_HOST_DEVICE vec3 hsv_to_rgb(const vec3& hsv) {
        float h = hsv.x, s = hsv.y, v = hsv.z;
        if (s == 0.0f) {
            return vec3(v);
        }

        h       = fmodf(h, 1.0f) * 6.0f;
        int i   = (int) h;
        float f = h - (float) i;
        float p = v * (1.0f - s);
        float q = v * (1.0f - s * f);
        float t = v * (1.0f - s * (1.0f - f));

        switch (i) {
        case 0: return {v, t, p};
        case 1: return {q, v, p};
        case 2: return {p, v, t};
        case 3: return {p, q, v};
        case 4: return {t, p, v};
        case 5:
        default: return {v, p, q};
        }
    }

    inline TCNN_HOST_DEVICE vec3 to_rgb(const vec2& dir) {
        return hsv_to_rgb({atan2f(dir.y, dir.x) / (2.0f * PI()) + 0.5f, 1.0f, length(dir)});
    }

    enum class EImageDataType {
        None,
        Byte,
        Half,
        Float,
    };

    enum class EDepthDataType {
        UShort,
        Float,
    };

    inline TCNN_HOST_DEVICE ivec2 image_pos(const vec2& pos, const ivec2& resolution) {
        return clamp(ivec2(pos * vec2(resolution)), 0, resolution - 1);
    }

    inline TCNN_HOST_DEVICE uint64_t pixel_idx(const ivec2& px, const ivec2& resolution, uint32_t img) {
        return px.x + px.y * resolution.x + img * (uint64_t) resolution.x * resolution.y;
    }

    inline TCNN_HOST_DEVICE uint64_t pixel_idx(const vec2& uv, const ivec2& resolution, uint32_t img) {
        return pixel_idx(image_pos(uv, resolution), resolution, img);
    }

    // inline TCNN_HOST_DEVICE vec3 composit_and_lerp(vec2 pos, const ivec2& resolution, uint32_t img, const __half*
    // training_images, const vec3& background_color, const vec3& exposure_scale = vec3(1.0f)) {
    //	pos = (pos.cwiseProduct(vec2(resolution)) - 0.5f).cwiseMax(0.0f).cwiseMin(vec2(resolution) - (1.0f + 1e-4f));

    //	const ivec2 pos_int = pos.cast<int>();
    //	const vec2 weight = pos - pos_int.cast<float>();

    //	const ivec2 idx = pos_int.cwiseMin(resolution - 2).cwiseMax(0);

    //	auto read_val = [&](const ivec2& p) {
    //		__half val[4];
    //		*(uint64_t*)&val[0] = ((uint64_t*)training_images)[pixel_idx(p, resolution, img)];
    //		return vec3{val[0], val[1], val[2]} * exposure_scale + background_color * (1.0f - (float)val[3]);
    //	};

    //	return (
    //		(1 - weight.x) * (1 - weight.y) * read_val({idx.x, idx.y}) +
    //		(weight.x) * (1 - weight.y) * read_val({idx.x+1, idx.y}) +
    //		(1 - weight.x) * (weight.y) * read_val({idx.x, idx.y+1}) +
    //		(weight.x) * (weight.y) * read_val({idx.x+1, idx.y+1})
    //	);
    // }

    // inline TCNN_HOST_DEVICE vec3 composit(vec2 pos, const ivec2& resolution, uint32_t img, const __half*
    // training_images, const vec3& background_color, const vec3& exposure_scale = vec3(1.0f)) {
    //	auto read_val = [&](const ivec2& p) {
    //		__half val[4];
    //		*(uint64_t*)&val[0] = ((uint64_t*)training_images)[pixel_idx(p, resolution, img)];
    //		return vec3{val[0], val[1], val[2]} * exposure_scale + background_color * (1.0f - (float)val[3]);
    //	};

    //	return read_val(image_pos(pos, resolution));
    // }

    inline TCNN_HOST_DEVICE uint32_t rgba_to_rgba32(const vec4& rgba) {
        return ((uint32_t) (clamp(rgba.r, 0.0f, 1.0f) * 255.0f + 0.5f) << 0) |
               ((uint32_t) (clamp(rgba.g, 0.0f, 1.0f) * 255.0f + 0.5f) << 8) |
               ((uint32_t) (clamp(rgba.b, 0.0f, 1.0f) * 255.0f + 0.5f) << 16) |
               ((uint32_t) (clamp(rgba.a, 0.0f, 1.0f) * 255.0f + 0.5f) << 24);
    }

    inline TCNN_HOST_DEVICE float rgba32_to_a(uint32_t rgba32) {
        return ((rgba32 & 0xFF000000) >> 24) * (1.0f / 255.0f);
    }

    inline TCNN_HOST_DEVICE vec3 rgba32_to_rgb(uint32_t rgba32) {
        return vec3{
            ((rgba32 & 0x000000FF) >> 0) * (1.0f / 255.0f),
            ((rgba32 & 0x0000FF00) >> 8) * (1.0f / 255.0f),
            ((rgba32 & 0x00FF0000) >> 16) * (1.0f / 255.0f),
        };
    }

    inline TCNN_HOST_DEVICE vec4 rgba32_to_rgba(uint32_t rgba32) {
        return vec4{
            ((rgba32 & 0x000000FF) >> 0) * (1.0f / 255.0f),
            ((rgba32 & 0x0000FF00) >> 8) * (1.0f / 255.0f),
            ((rgba32 & 0x00FF0000) >> 16) * (1.0f / 255.0f),
            ((rgba32 & 0xFF000000) >> 24) * (1.0f / 255.0f),
        };
    }

    inline TCNN_HOST_DEVICE vec4 read_rgba(ivec2 px, const ivec2& resolution, const void* pixels,
        EImageDataType image_data_type, uint32_t img = 0) {
        switch (image_data_type) {
        default:
            // This should never happen. Bright red to indicate this.
            return vec4{5.0f, 0.0f, 0.0f, 1.0f};
        case EImageDataType::Byte: {
            uint32_t val = ((uint32_t*) pixels)[pixel_idx(px, resolution, img)];
            if (val == 0x00FF00FF) {
                return vec4(-1.0f);
            }

            vec4 result  = rgba32_to_rgba(val);
            result.rgb() = srgb_to_linear(result.rgb()) * result.a;
            return result;
        }
        case EImageDataType::Half: {
            __half val[4];
            *(uint64_t*) &val[0] = ((uint64_t*) pixels)[pixel_idx(px, resolution, img)];
            return vec4{(float) val[0], (float) val[1], (float) val[2], (float) val[3]};
        }
        case EImageDataType::Float: return ((vec4*) pixels)[pixel_idx(px, resolution, img)];
        }
    }

    inline TCNN_HOST_DEVICE vec4 read_rgba(vec2 pos, const ivec2& resolution, const void* pixels,
        EImageDataType image_data_type, uint32_t img = 0) {
        return read_rgba(image_pos(pos, resolution), resolution, pixels, image_data_type, img);
    }

    inline TCNN_HOST_DEVICE float read_depth(vec2 pos, const ivec2& resolution, const float* depth, uint32_t img = 0) {
        auto read_val = [&](const ivec2& p) {
            return depth[pixel_idx(p, resolution, img)];
        };

        return read_val(image_pos(pos, resolution));
    }

    inline __device__ int float_to_ordered_int(float f) {
        int i = __float_as_int(f);
        return (i >= 0) ? i : i ^ 0x7FFFFFFF;
    }

    inline __device__ float ordered_int_to_float(int i) {
        return __int_as_float(i >= 0 ? i : i ^ 0x7FFFFFFF);
    }

    inline __device__ vec3 colormap_turbo(float x) {
        const vec4 kRedVec4   = {0.13572138f, 4.61539260f, -42.66032258f, 132.13108234f};
        const vec4 kGreenVec4 = {0.09140261f, 2.19418839f, 4.84296658f, -14.18503333f};
        const vec4 kBlueVec4  = {0.10667330f, 12.64194608f, -60.58204836f, 110.36276771f};
        const vec2 kRedVec2   = {-152.94239396f, 59.28637943f};
        const vec2 kGreenVec2 = {4.27729857f, 2.82956604f};
        const vec2 kBlueVec2  = {-89.90310912f, 27.34824973f};

        x       = __saturatef(x);
        vec4 v4 = {1.0f, x, x * x, x * x * x};
        vec2 v2 = {v4.w * x, v4.w * v4.z};
        return {
            dot(v4, kRedVec4) + dot(v2, kRedVec2),
            dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
            dot(v4, kBlueVec4) + dot(v2, kBlueVec2),
        };
    }
}


namespace ngp::cuda {
    // size of the density/occupancy grid in number of cells along an axis.
    inline constexpr TCNN_HOST_DEVICE uint32_t NERF_GRIDSIZE() {
        return 128;
    }

    inline constexpr TCNN_HOST_DEVICE uint32_t NERF_GRID_N_CELLS() {
        return NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE();
    }

    inline constexpr TCNN_HOST_DEVICE float NERF_RENDERING_NEAR_DISTANCE() {
        return 0.05f;
    }

    inline constexpr TCNN_HOST_DEVICE uint32_t NERF_STEPS() {
        return 1024;
    } // finest number of steps per unit length
    inline constexpr TCNN_HOST_DEVICE uint32_t NERF_CASCADES() {
        return 8;
    }

    inline constexpr TCNN_HOST_DEVICE float SQRT3() {
        return 1.73205080757f;
    }

    inline constexpr TCNN_HOST_DEVICE float STEPSIZE() {
        return (SQRT3() / NERF_STEPS());
    } // for nerf raymarch
    inline constexpr TCNN_HOST_DEVICE float MIN_CONE_STEPSIZE() {
        return STEPSIZE();
    }

    // Maximum step size is the width of the coarsest gridsize cell.
    inline constexpr TCNN_HOST_DEVICE float MAX_CONE_STEPSIZE() {
        return STEPSIZE() * (1 << (NERF_CASCADES() - 1)) * NERF_STEPS() / NERF_GRIDSIZE();
    }

    // Used to index into the PRNG stream. Must be larger than the number of
    // samples consumed by any given training ray.
    inline constexpr TCNN_HOST_DEVICE uint32_t N_MAX_RANDOM_SAMPLES_PER_RAY() {
        return 16;
    }

    // Any alpha below this is considered "invisible" and is thus culled away.
    inline constexpr TCNN_HOST_DEVICE float NERF_MIN_OPTICAL_THICKNESS() {
        return 0.01f;
    }

    struct TrainingImageMetadata {
        // Camera intrinsics and additional data associated with a NeRF training image
        // the memory to back the pixels and rays is held by GPUMemory objects in the NerfDataset and copied here.
        const void* pixels             = nullptr;
        EImageDataType image_data_type = EImageDataType::Half;

        const float* depth = nullptr;
        const Ray* rays    = nullptr;

        Lens lens            = {};
        ivec2 resolution     = ivec2(0);
        vec2 principal_point = vec2(0.5f);
        vec2 focal_length    = vec2(1000.f);
        vec4 rolling_shutter = vec4(0.0f);
        vec3 light_dir       = vec3(0.f); // TODO: replace this with more generic float[] of task-specific metadata.
    };

    struct LossAndGradient {
        vec3 loss;
        vec3 gradient;

        TCNN_HOST_DEVICE LossAndGradient operator*(float scalar) {
            return {loss * scalar, gradient * scalar};
        }

        TCNN_HOST_DEVICE LossAndGradient operator/(float scalar) {
            return {loss / scalar, gradient / scalar};
        }
    };

    inline TCNN_HOST_DEVICE LossAndGradient l2_loss(const vec3& target, const vec3& prediction) {
        vec3 difference = prediction - target;
        return {
            difference * difference,
            2.0f * difference
        };
    }

    inline TCNN_HOST_DEVICE LossAndGradient relative_l2_loss(const vec3& target, const vec3& prediction) {
        vec3 difference = prediction - target;
        vec3 denom      = prediction * prediction + 1e-2f;
        return {
            difference * difference / denom,
            2.0f * difference / denom
        };
    }

    inline TCNN_HOST_DEVICE LossAndGradient l1_loss(const vec3& target, const vec3& prediction) {
        vec3 difference = prediction - target;
        return {
            abs(difference),
            copysign(vec3(1.0f), difference),
        };
    }

    inline TCNN_HOST_DEVICE LossAndGradient huber_loss(const vec3& target, const vec3& prediction, float alpha = 1) {
        vec3 difference = prediction - target;
        vec3 abs_diff   = abs(difference);
        vec3 square     = 0.5f / alpha * difference * difference;
        return {
            {
                abs_diff.x > alpha ? (abs_diff.x - 0.5f * alpha) : square.x,
                abs_diff.y > alpha ? (abs_diff.y - 0.5f * alpha) : square.y,
                abs_diff.z > alpha ? (abs_diff.z - 0.5f * alpha) : square.z,
            },
            {
                abs_diff.x > alpha ? (difference.x > 0 ? 1.0f : -1.0f) : (difference.x / alpha),
                abs_diff.y > alpha ? (difference.y > 0 ? 1.0f : -1.0f) : (difference.y / alpha),
                abs_diff.z > alpha ? (difference.z > 0 ? 1.0f : -1.0f) : (difference.z / alpha),
            },
        };
    }

    inline TCNN_HOST_DEVICE LossAndGradient log_l1_loss(const vec3& target, const vec3& prediction) {
        vec3 difference = prediction - target;
        vec3 divisor    = abs(difference) + 1.0f;
        return {
            log(divisor),
            copysign(vec3(1.0f) / divisor, difference),
        };
    }

    inline TCNN_HOST_DEVICE LossAndGradient smape_loss(const vec3& target, const vec3& prediction) {
        vec3 difference = prediction - target;
        vec3 denom      = 0.5f * (abs(prediction) + abs(target)) + 1e-2f;
        return {
            abs(difference) / denom,
            copysign(vec3(1.0f) / denom, difference),
        };
    }

    inline TCNN_HOST_DEVICE LossAndGradient mape_loss(const vec3& target, const vec3& prediction) {
        vec3 difference = prediction - target;
        vec3 denom      = abs(prediction) + 1e-2f;
        return {
            abs(difference) / denom,
            copysign(vec3(1.0f) / denom, difference),
        };
    }

    struct NerfPayload {
        vec3 origin;
        vec3 dir;
        float t;
        float max_weight;
        uint32_t idx;
        uint16_t n_steps;
        bool alive;
    };

    //#define TRIPLANAR_COMPATIBLE_POSITIONS   // if this is defined, then positions are stored as [x,y,z,x] so that it can be split as [x,y] [y,z] [z,x] by the input encoding

    struct NerfPosition {
        TCNN_HOST_DEVICE NerfPosition(const vec3& pos, float dt)
            :
            p{pos}
#ifdef TRIPLANAR_COMPATIBLE_POSITIONS
        , x{pos.x}
#endif
        {
        }

        vec3 p;
#ifdef TRIPLANAR_COMPATIBLE_POSITIONS
        float x;
#endif
    };

    struct NerfDirection {
        TCNN_HOST_DEVICE NerfDirection(const vec3& dir, float dt) : d{dir} {
        }

        vec3 d;
    };

    struct NerfCoordinate {
        TCNN_HOST_DEVICE NerfCoordinate(const vec3& pos, const vec3& dir, float dt) : pos{pos, dt}, dt{dt}, dir{dir, dt} {
        }

        TCNN_HOST_DEVICE void set_with_optional_extra_dims(const vec3& pos, const vec3& dir, float dt, const float* extra_dims, uint32_t stride_in_bytes) {
            this->dt  = dt;
            this->pos = NerfPosition(pos, dt);
            this->dir = NerfDirection(dir, dt);
            copy_extra_dims(extra_dims, stride_in_bytes);
        }

        inline TCNN_HOST_DEVICE const float* get_extra_dims() const {
            return (const float*) (this + 1);
        }

        inline TCNN_HOST_DEVICE float* get_extra_dims() {
            return (float*) (this + 1);
        }

        TCNN_HOST_DEVICE void copy(const NerfCoordinate& inp, uint32_t stride_in_bytes) {
            *this = inp;
            copy_extra_dims(inp.get_extra_dims(), stride_in_bytes);
        }

        TCNN_HOST_DEVICE inline void copy_extra_dims(const float* extra_dims, uint32_t stride_in_bytes) {
            if (stride_in_bytes >= sizeof(NerfCoordinate)) {
                float* dst             = get_extra_dims();
                const uint32_t n_extra = (stride_in_bytes - sizeof(NerfCoordinate)) / sizeof(float);
                for (uint32_t i = 0; i < n_extra; ++i) dst[i] = extra_dims[i];
            }
        }

        NerfPosition pos;
        float dt;
        NerfDirection dir;
    };

    inline TCNN_HOST_DEVICE float network_to_rgb(float val, ENerfActivation activation) {
        switch (activation) {
        case ENerfActivation::None: return val;
        case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
        case ENerfActivation::Logistic: return logistic(val);
        case ENerfActivation::Exponential: return expf(clamp(val, -10.0f, 10.0f));
        default: assert(false);
        }
        return 0.0f;
    }

    inline TCNN_HOST_DEVICE float network_to_rgb_derivative(float val, ENerfActivation activation) {
        switch (activation) {
        case ENerfActivation::None: return 1.0f;
        case ENerfActivation::ReLU: return val > 0.0f ? 1.0f : 0.0f;
        case ENerfActivation::Logistic: {
            float density = logistic(val);
            return density * (1 - density);
        };
        case ENerfActivation::Exponential: return expf(clamp(val, -10.0f, 10.0f));
        default: assert(false);
        }
        return 0.0f;
    }

    template <typename T>
    TCNN_HOST_DEVICE vec3 network_to_rgb_derivative_vec(const T& val, ENerfActivation activation) {
        return {
            network_to_rgb_derivative(float(val[0]), activation),
            network_to_rgb_derivative(float(val[1]), activation),
            network_to_rgb_derivative(float(val[2]), activation),
        };
    }

    inline TCNN_HOST_DEVICE float network_to_density(float val, ENerfActivation activation) {
        switch (activation) {
        case ENerfActivation::None: return val;
        case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
        case ENerfActivation::Logistic: return logistic(val);
        case ENerfActivation::Exponential: return expf(val);
        default: assert(false);
        }
        return 0.0f;
    }

    inline TCNN_HOST_DEVICE float network_to_density_derivative(float val, ENerfActivation activation) {
        switch (activation) {
        case ENerfActivation::None: return 1.0f;
        case ENerfActivation::ReLU: return val > 0.0f ? 1.0f : 0.0f;
        case ENerfActivation::Logistic: {
            float density = logistic(val);
            return density * (1 - density);
        };
        case ENerfActivation::Exponential: return expf(clamp(val, -15.0f, 15.0f));
        default: assert(false);
        }
        return 0.0f;
    }

    template <typename T>
    TCNN_HOST_DEVICE vec3 network_to_rgb_vec(const T& val, ENerfActivation activation) {
        return {
            network_to_rgb(float(val[0]), activation),
            network_to_rgb(float(val[1]), activation),
            network_to_rgb(float(val[2]), activation),
        };
    }

    inline TCNN_HOST_DEVICE vec3 warp_position(const vec3& pos, const BoundingBox& aabb) {
        // return {logistic(pos.x - 0.5f), logistic(pos.y - 0.5f), logistic(pos.z - 0.5f)};
        // return pos;

        return aabb.relative_pos(pos);
    }

    inline TCNN_HOST_DEVICE vec3 unwarp_position(const vec3& pos, const BoundingBox& aabb) {
        // return {logit(pos.x) + 0.5f, logit(pos.y) + 0.5f, logit(pos.z) + 0.5f};
        // return pos;

        return aabb.min + pos * aabb.diag();
    }

    inline TCNN_HOST_DEVICE vec3 unwarp_position_derivative(const vec3& pos, const BoundingBox& aabb) {
        // return {logit(pos.x) + 0.5f, logit(pos.y) + 0.5f, logit(pos.z) + 0.5f};
        // return pos;

        return aabb.diag();
    }

    inline TCNN_HOST_DEVICE vec3 warp_position_derivative(const vec3& pos, const BoundingBox& aabb) {
        return vec3(1.0f) / unwarp_position_derivative(pos, aabb);
    }

    inline TCNN_HOST_DEVICE vec3 warp_direction(const vec3& dir) {
        return (dir + 1.0f) * 0.5f;
    }

    inline TCNN_HOST_DEVICE vec3 unwarp_direction(const vec3& dir) {
        return dir * 2.0f - 1.0f;
    }

    inline TCNN_HOST_DEVICE vec3 warp_direction_derivative(const vec3& dir) {
        return vec3(0.5f);
    }

    inline TCNN_HOST_DEVICE vec3 unwarp_direction_derivative(const vec3& dir) {
        return vec3(2.0f);
    }

    inline TCNN_HOST_DEVICE float warp_dt(float dt) {
        float max_stepsize = MIN_CONE_STEPSIZE() * (1 << (NERF_CASCADES() - 1));
        return (dt - MIN_CONE_STEPSIZE()) / (max_stepsize - MIN_CONE_STEPSIZE());
    }

    inline TCNN_HOST_DEVICE float unwarp_dt(float dt) {
        float max_stepsize = MIN_CONE_STEPSIZE() * (1 << (NERF_CASCADES() - 1));
        return dt * (max_stepsize - MIN_CONE_STEPSIZE()) + MIN_CONE_STEPSIZE();
    }

    inline TCNN_HOST_DEVICE uint32_t cascaded_grid_idx_at(vec3 pos, uint32_t mip) {
        float mip_scale = scalbnf(1.0f, -mip);
        pos -= vec3(0.5f);
        pos *= mip_scale;
        pos += vec3(0.5f);

        ivec3 i = pos * (float) NERF_GRIDSIZE();
        if (i.x < 0 || i.x >= NERF_GRIDSIZE() || i.y < 0 || i.y >= NERF_GRIDSIZE() || i.z < 0 || i.z >= NERF_GRIDSIZE()) {
            return 0xFFFFFFFF;
        }

        return morton3D(i.x, i.y, i.z);
    }

    inline TCNN_HOST_DEVICE uint32_t grid_mip_offset(uint32_t mip) {
        return NERF_GRID_N_CELLS() * mip;
    }

    inline TCNN_HOST_DEVICE bool density_grid_occupied_at(const vec3& pos, const uint8_t* density_grid_bitfield, uint32_t mip) {
        uint32_t idx = cascaded_grid_idx_at(pos, mip);
        if (idx == 0xFFFFFFFF) {
            return false;
        }
        return density_grid_bitfield[idx / 8 + grid_mip_offset(mip) / 8] & (1 << (idx % 8));
    }

    inline TCNN_HOST_DEVICE float cascaded_grid_at(vec3 pos, const float* cascaded_grid, uint32_t mip) {
        uint32_t idx = cascaded_grid_idx_at(pos, mip);
        if (idx == 0xFFFFFFFF) {
            return 0.0f;
        }
        return cascaded_grid[idx + grid_mip_offset(mip)];
    }

    inline TCNN_HOST_DEVICE float& cascaded_grid_at(vec3 pos, float* cascaded_grid, uint32_t mip) {
        uint32_t idx = cascaded_grid_idx_at(pos, mip);
        if (idx == 0xFFFFFFFF) {
            idx = 0;
            printf("WARNING: invalid cascaded grid access.");
        }
        return cascaded_grid[idx + grid_mip_offset(mip)];
    }

    inline TCNN_HOST_DEVICE float distance_to_next_voxel(const vec3& pos, const vec3& dir, const vec3& idir, float res) {
        // dda like step
        vec3 p   = res * (pos - 0.5f);
        float tx = (floorf(p.x + 0.5f + 0.5f * sign(dir.x)) - p.x) * idir.x;
        float ty = (floorf(p.y + 0.5f + 0.5f * sign(dir.y)) - p.y) * idir.y;
        float tz = (floorf(p.z + 0.5f + 0.5f * sign(dir.z)) - p.z) * idir.z;
        float t  = min(min(tx, ty), tz);

        return fmaxf(t / res, 0.0f);
    }

    inline TCNN_HOST_DEVICE float calc_cone_angle(float cosine, const vec2& focal_length, float cone_angle_constant) {
        // Pixel size. Doesn't always yield a good performance vs. quality
        // trade off. Especially if training pixels have a much different
        // size than rendering pixels.
        // return cosine*cosine / focal_length.mean();

        return cone_angle_constant;
    }

    inline TCNN_HOST_DEVICE float to_stepping_space(float t, float cone_angle) {
        if (cone_angle <= 1e-5f) {
            return t / MIN_CONE_STEPSIZE();
        }

        float log1p_c = logf(1.0f + cone_angle);

        float a = (logf(MIN_CONE_STEPSIZE()) - logf(log1p_c)) / log1p_c;
        float b = (logf(MAX_CONE_STEPSIZE()) - logf(log1p_c)) / log1p_c;

        float at = expf(a * log1p_c);
        float bt = expf(b * log1p_c);

        if (t <= at) {
            return (t - at) / MIN_CONE_STEPSIZE() + a;
        } else if (t <= bt) {
            return logf(t) / log1p_c;
        } else {
            return (t - bt) / MAX_CONE_STEPSIZE() + b;
        }
    }

    inline TCNN_HOST_DEVICE float from_stepping_space(float n, float cone_angle) {
        if (cone_angle <= 1e-5f) {
            return n * MIN_CONE_STEPSIZE();
        }

        float log1p_c = logf(1.0f + cone_angle);

        float a = (logf(MIN_CONE_STEPSIZE()) - logf(log1p_c)) / log1p_c;
        float b = (logf(MAX_CONE_STEPSIZE()) - logf(log1p_c)) / log1p_c;

        float at = expf(a * log1p_c);
        float bt = expf(b * log1p_c);

        if (n <= a) {
            return (n - a) * MIN_CONE_STEPSIZE() + at;
        } else if (n <= b) {
            return expf(n * log1p_c);
        } else {
            return (n - b) * MAX_CONE_STEPSIZE() + bt;
        }
    }

    inline TCNN_HOST_DEVICE float advance_n_steps(float t, float cone_angle, float n) {
        return from_stepping_space(to_stepping_space(t, cone_angle) + n, cone_angle);
    }

    inline TCNN_HOST_DEVICE float calc_dt(float t, float cone_angle) {
        return advance_n_steps(t, cone_angle, 1.0f) - t;
    }

    inline TCNN_HOST_DEVICE float advance_to_next_voxel(float t, float cone_angle, const vec3& pos, const vec3& dir, const vec3& idir, uint32_t mip) {
        float res = scalbnf(NERF_GRIDSIZE(), -(int) mip);

        float t_target = t + distance_to_next_voxel(pos, dir, idir, res);

        // Analytic stepping in multiples of 1 in the "log-space" of our exponential stepping routine
        t        = to_stepping_space(t, cone_angle);
        t_target = to_stepping_space(t_target, cone_angle);

        return from_stepping_space(t + ceilf(fmaxf(t_target - t, 0.5f)), cone_angle);
    }

    inline TCNN_HOST_DEVICE uint32_t mip_from_pos(const vec3& pos, uint32_t max_cascade = NERF_CASCADES() - 1) {
        int exponent;
        float maxval = max(abs(pos - 0.5f));
        frexpf(maxval, &exponent);
        return (uint32_t) clamp(exponent + 1, 0, (int) max_cascade);
    }

    inline TCNN_HOST_DEVICE uint32_t mip_from_dt(float dt, const vec3& pos, uint32_t max_cascade = NERF_CASCADES() - 1) {
        uint32_t mip = mip_from_pos(pos, max_cascade);
        dt *= 2 * NERF_GRIDSIZE();
        if (dt < 1.0f) {
            return mip;
        }

        int exponent;
        frexpf(dt, &exponent);
        return (uint32_t) clamp((int) mip, exponent, (int) max_cascade);
    }

    template <bool MIP_FROM_DT = false>
    TCNN_HOST_DEVICE float if_unoccupied_advance_to_next_occupied_voxel(
        float t,
        float cone_angle,
        const Ray& ray,
        const vec3& idir,
        const uint8_t* __restrict__ density_grid,
        uint32_t min_mip,
        uint32_t max_mip,
        BoundingBox aabb,
        mat3 aabb_to_local = mat3::identity()
        ) {
        while (true) {
            vec3 pos = ray(t);
            if (t >= MAX_DEPTH() || !aabb.contains(aabb_to_local * pos)) {
                return MAX_DEPTH();
            }

            uint32_t mip = clamp(MIP_FROM_DT ? mip_from_dt(calc_dt(t, cone_angle), pos) : mip_from_pos(pos), min_mip, max_mip);

            if (!density_grid || density_grid_occupied_at(pos, density_grid, mip)) {
                return t;
            }

            // Find largest empty voxel surrounding us, such that we can advance as far as possible in the next step.
            // Other places that do voxel stepping don't need this, because they don't rely on thread coherence as
            // much as this one here.
            while (mip < max_mip && !density_grid_occupied_at(pos, density_grid, mip + 1)) {
                ++mip;
            }

            t = advance_to_next_voxel(t, cone_angle, pos, ray.d, idir, mip);
        }
    }

    static constexpr float UNIFORM_SAMPLING_FRACTION = 0.5f;

    inline TCNN_HOST_DEVICE vec2 sample_cdf_2d(vec2 sample, uint32_t img, const ivec2& res, const float* __restrict__ cdf_x_cond_y, const float* __restrict__ cdf_y, float* __restrict__ pdf) {
        if (sample.x < UNIFORM_SAMPLING_FRACTION) {
            sample.x /= UNIFORM_SAMPLING_FRACTION;
            return sample;
        }

        sample.x = (sample.x - UNIFORM_SAMPLING_FRACTION) / (1.0f - UNIFORM_SAMPLING_FRACTION);

        cdf_y += img * res.y;

        // First select row according to cdf_y
        uint32_t y  = binary_search(sample.y, cdf_y, res.y);
        float prev  = y > 0 ? cdf_y[y - 1] : 0.0f;
        float pmf_y = cdf_y[y] - prev;
        sample.y    = (sample.y - prev) / pmf_y;

        cdf_x_cond_y += img * res.y * res.x + y * res.x;

        // Then, select col according to x
        uint32_t x  = binary_search(sample.x, cdf_x_cond_y, res.x);
        prev        = x > 0 ? cdf_x_cond_y[x - 1] : 0.0f;
        float pmf_x = cdf_x_cond_y[x] - prev;
        sample.x    = (sample.x - prev) / pmf_x;

        if (pdf) {
            *pdf = pmf_x * pmf_y * product(res);
        }

        return {((float) x + sample.x) / (float) res.x, ((float) y + sample.y) / (float) res.y};
    }

    inline TCNN_HOST_DEVICE float pdf_2d(vec2 sample, uint32_t img, const ivec2& res, const float* __restrict__ cdf_x_cond_y, const float* __restrict__ cdf_y) {
        ivec2 p = clamp(ivec2(sample * vec2(res)), 0, res - 1);

        cdf_y += img * res.y;
        cdf_x_cond_y += img * res.y * res.x + p.y * res.x;

        float pmf_y = cdf_y[p.y];
        if (p.y > 0) {
            pmf_y -= cdf_y[p.y - 1];
        }

        float pmf_x = cdf_x_cond_y[p.x];
        if (p.x > 0) {
            pmf_x -= cdf_x_cond_y[p.x - 1];
        }

        // Probability mass of picking the pixel
        float pmf = pmf_x * pmf_y;

        // To convert to probability density, divide by area of pixel
        return UNIFORM_SAMPLING_FRACTION + pmf * product(res) * (1.0f - UNIFORM_SAMPLING_FRACTION);
    }

    inline __device__ vec2 nerf_random_image_pos_training(default_rng_t& rng, const ivec2& resolution, bool snap_to_pixel_centers, const float* __restrict__ cdf_x_cond_y, const float* __restrict__ cdf_y, const ivec2& cdf_res, uint32_t img, float* __restrict__ pdf = nullptr) {
        vec2 uv = random_val_2d(rng);

        if (cdf_x_cond_y) {
            uv = sample_cdf_2d(uv, img, cdf_res, cdf_x_cond_y, cdf_y, pdf);
        } else {
            // // Warp-coherent tile
            // uv.x = __shfl_sync(0xFFFFFFFF, uv.x, 0);
            // uv.y = __shfl_sync(0xFFFFFFFF, uv.y, 0);

            // const ivec2 TILE_SIZE = {8, 4};
            // uv = (uv * vec2(resolution - TILE_SIZE) + vec2(tcnn::lane_id() % TILE_SIZE.x, tcnn::lane_id() / threadIdx.x)) / vec2(resolution);

            if (pdf) {
                *pdf = 1.0f;
            }
        }

        if (snap_to_pixel_centers) {
            uv = (vec2(clamp(ivec2(uv * vec2(resolution)), 0, resolution - 1)) + 0.5f) / vec2(resolution);
        }

        return uv;
    }

    inline TCNN_HOST_DEVICE uint32_t image_idx(uint32_t base_idx, uint32_t n_rays, uint32_t n_rays_total, uint32_t n_training_images, const float* __restrict__ cdf = nullptr, float* __restrict__ pdf = nullptr) {
        if (cdf) {
            float sample = ld_random_val(base_idx/* + n_rays_total*/, 0xdeadbeef);
            // float sample = random_val(base_idx/* + n_rays_total*/);
            uint32_t img = binary_search(sample, cdf, n_training_images);

            if (pdf) {
                float prev = img > 0 ? cdf[img - 1] : 0.0f;
                *pdf       = (cdf[img] - prev) * n_training_images;
            }

            return img;
        }

        // return ((base_idx/* + n_rays_total*/) * 56924617 + 96925573) % n_training_images;

        // Neighboring threads in the warp process the same image. Increases locality.
        if (pdf) {
            *pdf = 1.0f;
        }
        return (((base_idx/* + n_rays_total*/) * n_training_images) / n_rays) % n_training_images;
    }

    inline TCNN_HOST_DEVICE LossAndGradient loss_and_gradient(const vec3& target, const vec3& prediction, ELossType loss_type) {
        switch (loss_type) {
        case ELossType::RelativeL2: return relative_l2_loss(target, prediction);
            break;
        case ELossType::L1: return l1_loss(target, prediction);
            break;
        case ELossType::Mape: return mape_loss(target, prediction);
            break;
        case ELossType::Smape: return smape_loss(target, prediction);
            break;
        // Note: we divide the huber loss by a factor of 5 such that its L2 region near zero
        // matches with the L2 loss and error numbers become more comparable. This allows reading
        // off dB numbers of ~converged models and treating them as approximate PSNR to compare
        // with other NeRF methods. Self-normalizing optimizers such as Adam are agnostic to such
        // constant factors; optimization is therefore unaffected.
        case ELossType::Huber: return huber_loss(target, prediction, 0.1f) / 5.0f;
            break;
        case ELossType::LogL1: return log_l1_loss(target, prediction);
            break;
        default: case ELossType::L2: return l2_loss(target, prediction);
            break;
        }
    }
}
#endif //NGP_XAYAH_NGP_CUDA_UTILS_H
