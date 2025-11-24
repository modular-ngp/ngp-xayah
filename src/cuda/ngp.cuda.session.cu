#include "ngp.cuda.session.cuh"
#include "ngp.cuda.nerfnetwork.cuh"
#include "ngp.cuda.envmap.cuh"

#include <tiny-cuda-nn/encodings/multi_level_interface.h>

namespace ngp::cuda {
    using namespace tcnn;

    static constexpr float LOSS_SCALE() {
        return default_loss_scale<network_precision_t>();
    }

    __global__ void generate_training_samples_nerf(
        const uint32_t n_rays, BoundingBox aabb, const uint32_t max_samples, const uint32_t n_rays_total,
        default_rng_t rng, uint32_t* __restrict__ ray_counter, uint32_t* __restrict__ numsteps_counter,
        uint32_t* __restrict__ ray_indices_out, Ray* __restrict__ rays_out_unnormalized,
        uint32_t* __restrict__ numsteps_out, PitchedPtr<NerfCoordinate> coords_out, const uint32_t n_training_images,
        const TrainingImageMetadata* __restrict__ metadata, const TrainingXForm* training_xforms,
        const uint8_t* __restrict__ density_grid, uint32_t max_mip, bool max_level_rand_training,
        float* __restrict__ max_level_ptr, bool snap_to_pixel_centers, bool train_envmap, float cone_angle_constant,
        Buffer2DView<const vec2> distortion, const float* __restrict__ cdf_x_cond_y, const float* __restrict__ cdf_y,
        const float* __restrict__ cdf_img, const ivec2 cdf_res, const float* __restrict__ extra_dims_gpu,
        uint32_t n_extra_dims) {
        const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_rays) {
            return;
        }

        uint32_t img     = image_idx(i, n_rays, n_rays_total, n_training_images, cdf_img);
        ivec2 resolution = metadata[img].resolution;

        rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY());
        vec2 uv =
            nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, cdf_res, img);

        // Negative values indicate masked-away regions
        size_t pix_idx = pixel_idx(uv, resolution, 0);
        if (read_rgba(uv, resolution, metadata[img].pixels, metadata[img].image_data_type).x < 0.0f) {
            return;
        }

        float max_level = max_level_rand_training
                              ? (random_val(rng) * 2.0f)
                              : 1.0f; // Multiply by 2 to ensure 50% of training is at max level

        float motionblur_time = random_val(rng);

        const vec2 focal_length    = metadata[img].focal_length;
        const vec2 principal_point = metadata[img].principal_point;
        const float* extra_dims    = extra_dims_gpu + img * n_extra_dims;
        const Lens lens            = metadata[img].lens;

        const mat4x3 xform =
            get_xform_given_rolling_shutter(training_xforms[img], metadata[img].rolling_shutter, uv, motionblur_time);

        Ray ray_unnormalized;
        const Ray* rays_in_unnormalized = metadata[img].rays;
        if (rays_in_unnormalized) {
            // Rays have been explicitly supplied. Read them.
            ray_unnormalized = rays_in_unnormalized[pix_idx];

            /* DEBUG - compare the stored rays to the computed ones
            const mat4x3 xform = get_xform_given_rolling_shutter(training_xforms[img], metadata[img].rolling_shutter,
            uv, 0.f); Ray ray2; ray2.o = xform[3]; ray2.d = f_theta_distortion(uv, principal_point, lens); ray2.d =
            (xform.block<3, 3>(0, 0) * ray2.d).normalized(); if (i==1000) { printf("\n%d uv %0.3f,%0.3f pixel
            %0.2f,%0.2f transform from [%0.5f %0.5f %0.5f] to [%0.5f %0.5f %0.5f]\n" " origin    [%0.5f %0.5f %0.5f] vs
            [%0.5f %0.5f %0.5f]\n" " direction [%0.5f %0.5f %0.5f] vs [%0.5f %0.5f %0.5f]\n" , img,uv.x, uv.y,
            uv.x*resolution.x, uv.y*resolution.y,
                    training_xforms[img].start[3].x,training_xforms[img].start[3].y,training_xforms[img].start[3].z,
                    training_xforms[img].end[3].x,training_xforms[img].end[3].y,training_xforms[img].end[3].z,
                    ray_unnormalized.o.x,ray_unnormalized.o.y,ray_unnormalized.o.z,
                    ray2.o.x,ray2.o.y,ray2.o.z,
                    ray_unnormalized.d.x,ray_unnormalized.d.y,ray_unnormalized.d.z,
                    ray2.d.x,ray2.d.y,ray2.d.z);
            }
            */
        } else {
            ray_unnormalized = uv_to_ray(0, uv, resolution, focal_length, xform, principal_point, vec3(0.0f), 0.0f,
                1.0f, 0.0f, {}, {}, lens, distortion);
            if (!ray_unnormalized.is_valid()) {
                ray_unnormalized = {xform[3], xform[2]};
            }
        }

        vec3 ray_d_normalized = normalize(ray_unnormalized.d);

        vec2 tminmax     = aabb.ray_intersect(ray_unnormalized.o, ray_d_normalized);
        float cone_angle = calc_cone_angle(dot(ray_d_normalized, xform[2]), focal_length, cone_angle_constant);

        // The near distance prevents learning of camera-specific fudge right in front of the camera
        tminmax.x = fmaxf(tminmax.x, 0.0f);

        float startt = advance_n_steps(tminmax.x, cone_angle, random_val(rng));
        vec3 idir    = vec3(1.0f) / ray_d_normalized;

        // first pass to compute an accurate number of steps
        uint32_t j = 0;
        float t    = startt;
        vec3 pos;

        while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < NERF_STEPS()) {
            float dt     = calc_dt(t, cone_angle);
            uint32_t mip = mip_from_dt(dt, pos, max_mip);
            if (density_grid_occupied_at(pos, density_grid, mip)) {
                ++j;
                t += dt;
            } else {
                t = advance_to_next_voxel(t, cone_angle, pos, ray_d_normalized, idir, mip);
            }
        }
        if (j == 0 && !train_envmap) {
            return;
        }
        uint32_t numsteps = j;
        uint32_t base     = atomicAdd(numsteps_counter, numsteps); // first entry in the array is a counter
        if (base + numsteps > max_samples) {
            return;
        }

        coords_out += base;

        uint32_t ray_idx = atomicAdd(ray_counter, 1);

        ray_indices_out[ray_idx]       = i;
        rays_out_unnormalized[ray_idx] = ray_unnormalized;
        numsteps_out[ray_idx * 2 + 0]  = numsteps;
        numsteps_out[ray_idx * 2 + 1]  = base;

        vec3 warped_dir = warp_direction(ray_d_normalized);
        t               = startt;
        j               = 0;
        while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < numsteps) {
            float dt     = calc_dt(t, cone_angle);
            uint32_t mip = mip_from_dt(dt, pos, max_mip);
            if (density_grid_occupied_at(pos, density_grid, mip)) {
                coords_out(j)->set_with_optional_extra_dims(warp_position(pos, aabb), warped_dir, warp_dt(dt),
                    extra_dims, coords_out.stride_in_bytes);
                ++j;
                t += dt;
            } else {
                t = advance_to_next_voxel(t, cone_angle, pos, ray_d_normalized, idir, mip);
            }
        }

        if (max_level_rand_training) {
            max_level_ptr += base;
            for (j = 0; j < numsteps; ++j) {
                max_level_ptr[j] = max_level;
            }
        }
    }


    __global__ void compute_loss_kernel_train_nerf(
        const uint32_t n_rays, BoundingBox aabb, const uint32_t n_rays_total, default_rng_t rng,
        const uint32_t max_samples_compacted, const uint32_t* __restrict__ rays_counter, float loss_scale,
        int padded_output_width, Buffer2DView<const vec4> envmap, float* __restrict__ envmap_gradient,
        const ivec2 envmap_resolution, ELossType envmap_loss_type, vec3 background_color, EColorSpace color_space,
        bool train_with_random_bg_color, bool train_in_linear_colors, const uint32_t n_training_images,
        const TrainingImageMetadata* __restrict__ metadata, const network_precision_t* network_output,
        uint32_t* __restrict__ numsteps_counter, const uint32_t* __restrict__ ray_indices_in,
        const Ray* __restrict__ rays_in_unnormalized, uint32_t* __restrict__ numsteps_in,
        PitchedPtr<const NerfCoordinate> coords_in, PitchedPtr<NerfCoordinate> coords_out,
        network_precision_t* dloss_doutput, ELossType loss_type, ELossType depth_loss_type,
        float* __restrict__ loss_output, bool max_level_rand_training, float* __restrict__ max_level_compacted_ptr,
        ENerfActivation rgb_activation, ENerfActivation density_activation, bool snap_to_pixel_centers,
        float* __restrict__ error_map, const float* __restrict__ cdf_x_cond_y, const float* __restrict__ cdf_y,
        const float* __restrict__ cdf_img, const ivec2 error_map_res, const ivec2 error_map_cdf_res,
        const float* __restrict__ sharpness_data, ivec2 sharpness_resolution, float* __restrict__ sharpness_grid,
        float* __restrict__ density_grid, const float* __restrict__ mean_density_ptr, uint32_t max_mip,
        const vec3* __restrict__ exposure, vec3* __restrict__ exposure_gradient, float depth_supervision_lambda,
        float near_distance) {
        const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= *rays_counter) {
            return;
        }

        // grab the number of samples for this ray, and the first sample
        uint32_t numsteps = numsteps_in[i * 2 + 0];
        uint32_t base     = numsteps_in[i * 2 + 1];

        coords_in += base;
        network_output += base * padded_output_width;

        float T = 1.f;

        float EPSILON = 1e-4f;

        vec3 rgb_ray  = vec3(0.0f);
        vec3 hitpoint = vec3(0.0f);

        float depth_ray             = 0.f;
        uint32_t compacted_numsteps = 0;
        vec3 ray_o                  = rays_in_unnormalized[i].o;
        for (; compacted_numsteps < numsteps; ++compacted_numsteps) {
            if (T < EPSILON) {
                break;
            }

            const tvec<network_precision_t, 4> local_network_output = *(tvec<network_precision_t, 4>*) network_output;
            const vec3 rgb                                          = network_to_rgb_vec(local_network_output, rgb_activation);
            const vec3 pos                                          = unwarp_position(coords_in.ptr->pos.p, aabb);
            const float dt                                          = unwarp_dt(coords_in.ptr->dt);
            float cur_depth                                         = distance(pos, ray_o);
            float density                                           = network_to_density(float(local_network_output[3]), density_activation);


            const float alpha  = 1.f - __expf(-density * dt);
            const float weight = alpha * T;
            rgb_ray += weight * rgb;
            hitpoint += weight * pos;
            depth_ray += weight * cur_depth;
            T *= (1.f - alpha);

            network_output += padded_output_width;
            coords_in += 1;
        }
        hitpoint /= (1.0f - T);

        // Must be same seed as above to obtain the same
        // background color.
        uint32_t ray_idx = ray_indices_in[i];
        rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY());

        float img_pdf    = 1.0f;
        uint32_t img     = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img, &img_pdf);
        ivec2 resolution = metadata[img].resolution;

        float uv_pdf = 1.0f;
        vec2 uv      = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y,
            error_map_cdf_res, img, &uv_pdf);
        float max_level = max_level_rand_training
                              ? (random_val(rng) * 2.0f)
                              : 1.0f; // Multiply by 2 to ensure 50% of training is at max level
        rng.advance(1); // motionblur_time

        if (train_with_random_bg_color) {
            background_color = random_val_3d(rng);
        }
        vec3 pre_envmap_background_color = background_color = srgb_to_linear(background_color);

        // Composit background behind envmap
        vec4 envmap_value;
        vec3 dir;
        if (envmap) {
            dir              = normalize(rays_in_unnormalized[i].d);
            envmap_value     = read_envmap(envmap, dir);
            background_color = envmap_value.rgb() + background_color * (1.0f - envmap_value.a);
        }

        vec3 exposure_scale = exp(0.6931471805599453f * exposure[img]);
        // vec3 rgbtarget = composit_and_lerp(uv, resolution, img, training_images, background_color, exposure_scale);
        // vec3 rgbtarget = composit(uv, resolution, img, training_images, background_color, exposure_scale);
        vec4 texsamp = read_rgba(uv, resolution, metadata[img].pixels, metadata[img].image_data_type);

        vec3 rgbtarget;
        if (train_in_linear_colors || color_space == EColorSpace::Linear) {
            rgbtarget = exposure_scale * texsamp.rgb() + (1.0f - texsamp.a) * background_color;

            if (!train_in_linear_colors) {
                rgbtarget        = linear_to_srgb(rgbtarget);
                background_color = linear_to_srgb(background_color);
            }
        } else if (color_space == EColorSpace::SRGB) {
            background_color = linear_to_srgb(background_color);
            if (texsamp.a > 0) {
                rgbtarget = linear_to_srgb(exposure_scale * texsamp.rgb() / texsamp.a) * texsamp.a +
                            (1.0f - texsamp.a) * background_color;
            } else {
                rgbtarget = background_color;
            }
        }

        if (compacted_numsteps == numsteps) {
            // support arbitrary background colors
            rgb_ray += T * background_color;
        }

        // Step again, this time computing loss
        network_output -= padded_output_width * compacted_numsteps; // rewind the pointer
        coords_in -= compacted_numsteps;

        uint32_t compacted_base =
            atomicAdd(numsteps_counter, compacted_numsteps); // first entry in the array is a counter
        compacted_numsteps =
            min(max_samples_compacted - min(max_samples_compacted, compacted_base), compacted_numsteps);
        numsteps_in[i * 2 + 0] = compacted_numsteps;
        numsteps_in[i * 2 + 1] = compacted_base;
        if (compacted_numsteps == 0) {
            return;
        }

        max_level_compacted_ptr += compacted_base;
        coords_out += compacted_base;

        dloss_doutput += compacted_base * padded_output_width;

        LossAndGradient lg = loss_and_gradient(rgbtarget, rgb_ray, loss_type);
        lg.loss /= img_pdf * uv_pdf;

        float target_depth = length(rays_in_unnormalized[i].d) *
                             ((depth_supervision_lambda > 0.0f && metadata[img].depth)
                                  ? read_depth(uv, resolution, metadata[img].depth)
                                  : -1.0f);
        LossAndGradient lg_depth  = loss_and_gradient(vec3(target_depth), vec3(depth_ray), depth_loss_type);
        float depth_loss_gradient = target_depth > 0.0f ? depth_supervision_lambda * lg_depth.gradient.x : 0;

        // Note: dividing the gradient by the PDF would cause unbiased loss estimates.
        // Essentially: variance reduction, but otherwise the same optimization.
        // We _dont_ want that. If importance sampling is enabled, we _do_ actually want
        // to change the weighting of the loss function. So don't divide.
        // lg.gradient /= img_pdf * uv_pdf;

        float mean_loss = mean(lg.loss);
        if (loss_output) {
            loss_output[i] = mean_loss / (float) n_rays;
        }

        if (error_map) {
            const vec2 pos      = clamp(uv * vec2(error_map_res) - 0.5f, 0.0f, vec2(error_map_res) - (1.0f + 1e-4f));
            const ivec2 pos_int = pos;
            const vec2 weight   = pos - vec2(pos_int);

            ivec2 idx = clamp(pos_int, 0, resolution - 2);

            auto deposit_val = [&](int x, int y, float val) {
                atomicAdd(&error_map[img * product(error_map_res) + y * error_map_res.x + x], val);
            };

            if (sharpness_data && aabb.contains(hitpoint)) {
                ivec2 sharpness_pos = clamp(ivec2(uv * vec2(sharpness_resolution)), 0, sharpness_resolution - 1);
                float sharp         = sharpness_data[img * product(sharpness_resolution) +
                                             sharpness_pos.y * sharpness_resolution.x + sharpness_pos.x] +
                              1e-6f;

                // The maximum value of positive floats interpreted in uint format is the same as the maximum value of
                // the floats.
                float grid_sharp = __uint_as_float(
                    atomicMax((uint32_t*) &cascaded_grid_at(hitpoint, sharpness_grid, mip_from_pos(hitpoint, max_mip)),
                        __float_as_uint(sharp)));
                grid_sharp =
                    fmaxf(sharp, grid_sharp); // atomicMax returns the old value, so compute the new one locally.

                mean_loss *= fmaxf(sharp / grid_sharp, 0.01f);
            }

            deposit_val(idx.x, idx.y, (1 - weight.x) * (1 - weight.y) * mean_loss);
            deposit_val(idx.x + 1, idx.y, weight.x * (1 - weight.y) * mean_loss);
            deposit_val(idx.x, idx.y + 1, (1 - weight.x) * weight.y * mean_loss);
            deposit_val(idx.x + 1, idx.y + 1, weight.x * weight.y * mean_loss);
        }

        loss_scale /= n_rays;

        const float output_l2_reg         = rgb_activation == ENerfActivation::Exponential ? 1e-4f : 0.0f;
        const float output_l1_reg_density = *mean_density_ptr < NERF_MIN_OPTICAL_THICKNESS() ? 1e-4f : 0.0f;

        // now do it again computing gradients
        vec3 rgb_ray2    = {0.f, 0.f, 0.f};
        float depth_ray2 = 0.f;
        T                = 1.f;
        for (uint32_t j = 0; j < compacted_numsteps; ++j) {
            if (max_level_rand_training) {
                max_level_compacted_ptr[j] = max_level;
            }
            // Compact network inputs
            NerfCoordinate* coord_out      = coords_out(j);
            const NerfCoordinate* coord_in = coords_in(j);
            coord_out->copy(*coord_in, coords_out.stride_in_bytes);

            const vec3 pos = unwarp_position(coord_in->pos.p, aabb);
            float depth    = distance(pos, ray_o);

            float dt                                                = unwarp_dt(coord_in->dt);
            const tvec<network_precision_t, 4> local_network_output = *(tvec<network_precision_t, 4>*) network_output;
            const vec3 rgb                                          = network_to_rgb_vec(local_network_output, rgb_activation);
            const float density                                     = network_to_density(float(local_network_output[3]), density_activation);
            const float alpha                                       = 1.f - __expf(-density * dt);
            const float weight                                      = alpha * T;
            rgb_ray2 += weight * rgb;
            depth_ray2 += weight * depth;
            T *= (1.f - alpha);

            // we know the suffix of this ray compared to where we are up to. note the suffix depends on this step's
            // alpha as suffix = (1-alpha)*(somecolor), so dsuffix/dalpha = -somecolor = -suffix/(1-alpha)
            const vec3 suffix        = rgb_ray - rgb_ray2;
            const vec3 dloss_by_drgb = weight * lg.gradient;

            tvec<network_precision_t, 4> local_dL_doutput;

            // chain rule to go from dloss/drgb to dloss/dmlp_output
            local_dL_doutput[0] = loss_scale *
                                  (dloss_by_drgb.x * network_to_rgb_derivative(local_network_output[0], rgb_activation) +
                                   fmaxf(0.0f, output_l2_reg * (float) local_network_output[0])); // Penalize way too large color values
            local_dL_doutput[1] = loss_scale *
                                  (dloss_by_drgb.y * network_to_rgb_derivative(local_network_output[1], rgb_activation) +
                                   fmaxf(0.0f, output_l2_reg * (float) local_network_output[1]));
            local_dL_doutput[2] = loss_scale *
                                  (dloss_by_drgb.z * network_to_rgb_derivative(local_network_output[2], rgb_activation) +
                                   fmaxf(0.0f, output_l2_reg * (float) local_network_output[2]));

            float density_derivative =
                network_to_density_derivative(float(local_network_output[3]), density_activation);
            const float depth_suffix      = depth_ray - depth_ray2;
            const float depth_supervision = depth_loss_gradient * (T * depth - depth_suffix);

            float dloss_by_dmlp = density_derivative * (dt * (dot(lg.gradient, T * rgb - suffix) + depth_supervision));

            // static constexpr float mask_supervision_strength = 1.f; // we are already 'leaking' mask information into
            // the nerf via the random bg colors; setting this to eg between 1 and  100 encourages density towards 0 in
            // such regions. dloss_by_dmlp += (texsamp.a<0.001f) ? mask_supervision_strength * weight : 0.f;

            local_dL_doutput[3] = loss_scale * dloss_by_dmlp +
                                  (float(local_network_output[3]) < 0.0f ? -output_l1_reg_density : 0.0f) +
                                  (float(local_network_output[3]) > -10.0f && depth < near_distance ? 1e-4f : 0.0f);;

            *(tvec<network_precision_t, 4>*) dloss_doutput = local_dL_doutput;

            dloss_doutput += padded_output_width;
            network_output += padded_output_width;
        }

        if (exposure_gradient) {
            // Assume symmetric loss
            vec3 dloss_by_dgt = -lg.gradient / uv_pdf;

            if (!train_in_linear_colors) {
                dloss_by_dgt /= srgb_to_linear_derivative(rgbtarget);
            }

            // 2^exposure * log(2)
            vec3 dloss_by_dexposure = loss_scale * dloss_by_dgt * exposure_scale * 0.6931471805599453f;
            atomicAdd(&exposure_gradient[img].x, dloss_by_dexposure.x);
            atomicAdd(&exposure_gradient[img].y, dloss_by_dexposure.y);
            atomicAdd(&exposure_gradient[img].z, dloss_by_dexposure.z);
        }

        if (compacted_numsteps == numsteps && envmap_gradient) {
            vec3 loss_gradient = lg.gradient;
            if (envmap_loss_type != loss_type) {
                loss_gradient = loss_and_gradient(rgbtarget, rgb_ray, envmap_loss_type).gradient;
            }

            vec3 dloss_by_dbackground = T * loss_gradient;
            if (!train_in_linear_colors) {
                dloss_by_dbackground /= srgb_to_linear_derivative(background_color);
            }

            tvec<network_precision_t, 4> dL_denvmap;
            dL_denvmap[0] = loss_scale * dloss_by_dbackground.x;
            dL_denvmap[1] = loss_scale * dloss_by_dbackground.y;
            dL_denvmap[2] = loss_scale * dloss_by_dbackground.z;

            float dloss_by_denvmap_alpha = -dot(dloss_by_dbackground, pre_envmap_background_color);

            // dL_denvmap[3] = loss_scale * dloss_by_denvmap_alpha;
            dL_denvmap[3] = (network_precision_t) 0;

            deposit_envmap_gradient(dL_denvmap, envmap_gradient, envmap_resolution, dir);
        }
    }
}


void ngp::cuda::NGPSession::reset_session(const nlohmann::json& config) {
    const auto& loss_config         = config["loss"];
    const auto& optimizer_config    = config["optimizer"];
    const auto& network_config      = config["network"];
    const auto& encoding_config     = config["encoding"];
    const auto& dir_encoding_config = config["dir_encoding"];
    const auto& rgb_network_config  = config["rgb_network"];

    auto loss_config_expand     = loss_config;
    loss_config_expand["otype"] = "L2";
    uint32_t n_pos              = 3;
    uint32_t n_input            = 7;
    uint32_t n_output           = 4;
    uint32_t n_dir_dims         = 3;
    uint32_t n_extra_dims       = 0;

    m_loss.reset(tcnn::create_loss<tcnn::network_precision_t>(loss_config_expand));
    m_optimizer.reset(tcnn::create_optimizer<tcnn::network_precision_t>(optimizer_config));
    m_network  = std::make_shared<NerfNetwork<tcnn::network_precision_t>>(n_pos, n_dir_dims, n_extra_dims, n_pos + 1, encoding_config, dir_encoding_config, network_config, rgb_network_config);
    m_encoding = dynamic_cast<NerfNetwork<tcnn::network_precision_t>*>(m_network.get())->m_pos_encoding;
    m_trainer  = std::make_shared<tcnn::Trainer<float, tcnn::network_precision_t, tcnn::network_precision_t>>(m_network, m_optimizer, m_loss, m_seed);

    auto optimizer_config_expand          = optimizer_config;
    nlohmann::json* leaf_optimizer_config = &optimizer_config_expand;
    while (leaf_optimizer_config->contains("nested")) leaf_optimizer_config = &(*leaf_optimizer_config)["nested"];
    (*leaf_optimizer_config)["optimize_matrix_params"]     = true;
    (*leaf_optimizer_config)["optimize_non_matrix_params"] = true;
    m_optimizer->update_hyperparams(optimizer_config_expand);
}

void ngp::cuda::NGPSession::train(const uint32_t batchsize) {
    counters.prepare_for_training_steps(m_stream);


    const uint32_t padded_output_width = m_network->padded_output_width();
    const uint32_t max_samples         = batchsize * 16;
    const uint32_t floats_per_coord    = 7; // pos + dir + dt
    const uint32_t extra_stride        = 0;

    tcnn::GPUMemoryArena::Allocation alloc;
    auto scratch = tcnn::allocate_workspace_and_distribute<
        uint32_t, // ray_indices
        tcnn::Ray, // rays
        uint32_t, // numsteps
        float, // coords
        float, // max_level
        tcnn::network_precision_t, // mlp_out
        tcnn::network_precision_t, // dloss_dmlp_out
        float, // coords_compacted
        float, // coords_gradient
        float, // max_level_compacted
        uint32_t // ray_counter
    >(
        m_stream, &alloc, counters.rays_per_batch, counters.rays_per_batch, counters.rays_per_batch * 2,
        max_samples * floats_per_coord, max_samples, std::max(batchsize, max_samples) * padded_output_width,
        batchsize * padded_output_width, batchsize * floats_per_coord,
        batchsize * floats_per_coord, batchsize, 1);

    uint32_t* ray_indices                     = std::get<0>(scratch);
    tcnn::Ray* rays_unnormalized              = std::get<1>(scratch);
    uint32_t* numsteps                        = std::get<2>(scratch);
    float* coords                             = std::get<3>(scratch);
    float* max_level                          = std::get<4>(scratch);
    tcnn::network_precision_t* mlp_out        = std::get<5>(scratch);
    tcnn::network_precision_t* dloss_dmlp_out = std::get<6>(scratch);
    float* coords_compacted                   = std::get<7>(scratch);
    float* coords_gradient                    = std::get<8>(scratch);
    float* max_level_compacted                = std::get<9>(scratch);
    uint32_t* ray_counter                     = std::get<10>(scratch);

    uint32_t max_inference;
    if (counters.measured_batch_size_before_compaction == 0) {
        counters.measured_batch_size_before_compaction = max_inference = max_samples;
    } else {
        max_inference = tcnn::next_multiple(std::min(counters.measured_batch_size_before_compaction, max_samples), tcnn::BATCH_SIZE_GRANULARITY);
    }


    tcnn::GPUMatrix<float> compacted_coords_matrix((float*) coords_compacted, floats_per_coord, batchsize);
    tcnn::GPUMatrix<tcnn::network_precision_t> compacted_rgbsigma_matrix(mlp_out, padded_output_width, batchsize);
    tcnn::GPUMatrix<tcnn::network_precision_t> gradient_matrix(dloss_dmlp_out, padded_output_width, batchsize);


    if (m_training_step == 0) counters.n_rays_total = 0;
    uint32_t n_rays_total = counters.n_rays_total;
    counters.n_rays_total += counters.rays_per_batch;
    n_rays_since_error_map_update += counters.rays_per_batch;


    CUDA_CHECK_THROW(cudaMemsetAsync(ray_counter, 0, sizeof(uint32_t), m_stream));

    auto hg_enc = dynamic_cast<tcnn::MultiLevelEncoding<tcnn::network_precision_t>*>(m_encoding.get());

    // {
    //     // TODO:: 1125 START FROM HERE
    //     bool sample_focal_plane_proportional_to_error = false;
    //     bool sample_image_proportional_to_error       = false;
    //     bool include_sharpness_in_error               = false;
    //     linear_kernel(
    //         generate_training_samples_nerf, 0, m_stream, counters.rays_per_batch, m_aabb, max_inference, n_rays_total,
    //         m_rng, ray_counter, counters.numsteps_counter.data(), ray_indices, rays_unnormalized, numsteps,
    //         PitchedPtr<NerfCoordinate>((NerfCoordinate*) coords, 1, 0, extra_stride),
    //         m_nerf.training.n_images_for_training, m_nerf.training.dataset.metadata_gpu.data(),
    //         m_nerf.training.transforms_gpu.data(), m_nerf.density_grid_bitfield.data(), m_nerf.max_cascade,
    //         m_max_level_rand_training, max_level, m_nerf.training.snap_to_pixel_centers,
    //         m_nerf.training.train_envmap, m_nerf.cone_angle_constant, m_distortion.view(),
    //         sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
    //         sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
    //         sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
    //         m_nerf.training.error_map.cdf_resolution, m_nerf.training.extra_dims_gpu.data(),
    //         0);
    //
    //     if (hg_enc) {
    //         hg_enc->set_max_level_gpu(m_max_level_rand_training ? max_level : nullptr);
    //     }
    //
    //     GPUMatrix<float> coords_matrix((float*) coords, floats_per_coord, max_inference);
    //     GPUMatrix<network_precision_t> rgbsigma_matrix(mlp_out, padded_output_width, max_inference);
    //     m_network->inference_mixed_precision(m_stream, coords_matrix, rgbsigma_matrix, false);
    //
    //     if (hg_enc) {
    //         hg_enc->set_max_level_gpu(m_max_level_rand_training ? max_level_compacted : nullptr);
    //     }
    //
    //     linear_kernel(
    //         compute_loss_kernel_train_nerf, 0, m_stream, counters.rays_per_batch, m_aabb, n_rays_total, m_rng,
    //         target_batch_size, ray_counter, LOSS_SCALE(), padded_output_width, m_envmap.view(), envmap_gradient,
    //         m_envmap.resolution, m_envmap.loss_type, m_background_color.rgb(), m_color_space,
    //         m_nerf.training.random_bg_color, m_nerf.training.linear_colors, m_nerf.training.n_images_for_training,
    //         m_nerf.training.dataset.metadata_gpu.data(), mlp_out, counters.numsteps_counter_compacted.data(),
    //         ray_indices, rays_unnormalized, numsteps,
    //         PitchedPtr<const NerfCoordinate>((NerfCoordinate*) coords, 1, 0, extra_stride),
    //         PitchedPtr<NerfCoordinate>((NerfCoordinate*) coords_compacted, 1, 0, extra_stride), dloss_dmlp_out,
    //         m_nerf.training.loss_type, m_nerf.training.depth_loss_type, counters.loss.data(),
    //         m_max_level_rand_training, max_level_compacted, m_nerf.rgb_activation, m_nerf.density_activation,
    //         m_nerf.training.snap_to_pixel_centers,
    //         accumulate_error ? m_nerf.training.error_map.data.data() : nullptr,
    //         sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
    //         sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
    //         sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
    //         m_nerf.training.error_map.resolution, m_nerf.training.error_map.cdf_resolution,
    //         include_sharpness_in_error ? m_nerf.training.dataset.sharpness_data.data() : nullptr,
    //         m_nerf.training.dataset.sharpness_resolution, m_nerf.training.sharpness_grid.data(),
    //         m_nerf.density_grid.data(), m_nerf.density_grid_mean.data(), m_nerf.max_cascade,
    //         m_nerf.training.cam_exposure_gpu.data(),
    //         m_nerf.training.optimize_exposure ? m_nerf.training.cam_exposure_gradient_gpu.data() : nullptr,
    //         m_nerf.training.depth_supervision_lambda, m_nerf.training.near_distance);
    // }
}

void ngp::cuda::NGPSession::NerfCounters::prepare_for_training_steps(cudaStream_t stream) {
    numsteps_counter.enlarge(1);
    numsteps_counter_compacted.enlarge(1);
    loss.enlarge(rays_per_batch);
    CUDA_CHECK_THROW(cudaMemsetAsync(numsteps_counter.data(), 0, sizeof(uint32_t), stream)); // clear the counter in the first slot
    CUDA_CHECK_THROW(cudaMemsetAsync(numsteps_counter_compacted.data(), 0, sizeof(uint32_t), stream)); // clear the counter in the first slot
    CUDA_CHECK_THROW(cudaMemsetAsync(loss.data(), 0, sizeof(float) * rays_per_batch, stream));
}

float ngp::cuda::NGPSession::NerfCounters::update_after_training(uint32_t target_batch_size, bool get_loss_scalar, cudaStream_t stream) {
    std::vector<uint32_t> counter_cpu(1);
    std::vector<uint32_t> compacted_counter_cpu(1);
    numsteps_counter.copy_to_host(counter_cpu);
    numsteps_counter_compacted.copy_to_host(compacted_counter_cpu);
    measured_batch_size                   = 0;
    measured_batch_size_before_compaction = 0;

    if (counter_cpu[0] == 0 || compacted_counter_cpu[0] == 0) {
        return 0.f;
    }

    measured_batch_size_before_compaction = counter_cpu[0];
    measured_batch_size                   = compacted_counter_cpu[0];

    float loss_scalar = 0.0;
    if (get_loss_scalar) {
        loss_scalar =
            tcnn::reduce_sum(loss.data(), rays_per_batch, stream) * (float) measured_batch_size / (float) target_batch_size;
    }

    rays_per_batch = (uint32_t) ((float) rays_per_batch * (float) target_batch_size / (float) measured_batch_size);
    rays_per_batch = std::min(tcnn::next_multiple(rays_per_batch, tcnn::BATCH_SIZE_GRANULARITY), 1u << 18);

    return loss_scalar;
}
