#include "ngp.cuda.session.cuh"
#include "ngp.cuda.nerfnetwork.cuh"
#include "ngp.cuda.envmap.cuh"
#include "ngp.cuda.utils.cuh"

#include <tiny-cuda-nn/encodings/multi_level_interface.h>

namespace ngp::cuda {
    __global__ void generate_training_samples_nerf(
        const uint32_t n_rays,
        BoundingBox aabb,
        const uint32_t max_samples,
        const uint32_t n_rays_total,
        tcnn::pcg32 rng,
        uint32_t* __restrict__ ray_counter,
        uint32_t* __restrict__ numsteps_counter,
        uint32_t* __restrict__ ray_indices_out,
        tcnn::Ray* __restrict__ rays_out_unnormalized,
        uint32_t* __restrict__ numsteps_out,
        tcnn::PitchedPtr<NerfCoordinate> coords_out,
        const uint32_t n_training_images,
        const TrainingImageMetadata* __restrict__ metadata,
        const TrainingXForm* training_xforms,
        const uint8_t* __restrict__ density_grid,
        uint32_t max_mip,
        bool max_level_rand_training,
        float* __restrict__ max_level_ptr,
        bool snap_to_pixel_centers,
        bool train_envmap,
        float cone_angle_constant,
        Buffer2DView<const vec2> distortion,
        const float* __restrict__ cdf_x_cond_y,
        const float* __restrict__ cdf_y,
        const float* __restrict__ cdf_img,
        const ivec2 cdf_res,
        const float* __restrict__ extra_dims_gpu,
        uint32_t n_extra_dims
        ) {
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

        tcnn::Ray ray_unnormalized;
        const tcnn::Ray* rays_in_unnormalized = metadata[img].rays;
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
    m_counters_rgb.prepare_for_training_steps(m_stream);


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
        m_stream, &alloc, m_counters_rgb.rays_per_batch, m_counters_rgb.rays_per_batch, m_counters_rgb.rays_per_batch * 2,
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
    if (m_counters_rgb.measured_batch_size_before_compaction == 0) {
        m_counters_rgb.measured_batch_size_before_compaction = max_inference = max_samples;
    } else {
        max_inference = tcnn::next_multiple(std::min(m_counters_rgb.measured_batch_size_before_compaction, max_samples), tcnn::BATCH_SIZE_GRANULARITY);
    }


    tcnn::GPUMatrix<float> compacted_coords_matrix((float*) coords_compacted, floats_per_coord, batchsize);
    tcnn::GPUMatrix<tcnn::network_precision_t> compacted_rgbsigma_matrix(mlp_out, padded_output_width, batchsize);
    tcnn::GPUMatrix<tcnn::network_precision_t> gradient_matrix(dloss_dmlp_out, padded_output_width, batchsize);


    if (m_training_step == 0) m_counters_rgb.n_rays_total = 0;
    uint32_t n_rays_total = m_counters_rgb.n_rays_total;
    m_counters_rgb.n_rays_total += m_counters_rgb.rays_per_batch;
    n_rays_since_error_map_update += m_counters_rgb.rays_per_batch;


    CUDA_CHECK_THROW(cudaMemsetAsync(ray_counter, 0, sizeof(uint32_t), m_stream));

    auto hg_enc = dynamic_cast<tcnn::MultiLevelEncoding<tcnn::network_precision_t>*>(m_encoding.get());

    {
        // tcnn::linear_kernel(
        //     generate_training_samples_nerf,
        //     0,
        //     m_stream,
        //     m_counters_rgb.rays_per_batch,
        //     m_aabb,
        //     max_inference,
        //     n_rays_total,
        //     m_rng,
        //     ray_counter,
        //     m_counters_rgb.numsteps_counter.data(),
        //     ray_indices,
        //     rays_unnormalized,
        //     numsteps,
        //     tcnn::PitchedPtr<NerfCoordinate>((NerfCoordinate*) coords, 1, 0, extra_stride),
        //     m_nerf.training.n_images_for_training,
        //     m_nerf.training.dataset.metadata_gpu.data(),
        //     m_nerf.training.transforms_gpu.data(),
        //     m_nerf.density_grid_bitfield.data(),
        //     m_nerf.max_cascade,
        //     m_max_level_rand_training,
        //     max_level,
        //     m_nerf.training.snap_to_pixel_centers,
        //     m_nerf.training.train_envmap,
        //     m_nerf.cone_angle_constant,
        //     m_distortion.view(),
        //     nullptr,
        //     nullptr,
        //     nullptr,
        //     m_nerf.training.error_map.cdf_resolution,
        //     m_nerf.training.extra_dims_gpu.data(),
        //     m_nerf_network->n_extra_dims());

        // if (hg_enc) {
        //     hg_enc->set_max_level_gpu(m_max_level_rand_training ? max_level : nullptr);
        // }

        //     tcnn::GPUMatrix<float> coords_matrix((float*) coords, floats_per_coord, max_inference);
        //     tcnn::GPUMatrix<tcnn::network_precision_t> rgbsigma_matrix(mlp_out, padded_output_width, max_inference);
        //     m_network->inference_mixed_precision(m_stream, coords_matrix, rgbsigma_matrix, false);
        //
        //     if (hg_enc) {
        //         hg_enc->set_max_level_gpu(m_max_level_rand_training ? max_level_compacted : nullptr);
        //     }
        //
        //     tcnn::linear_kernel(
        //         compute_loss_kernel_train_nerf, 0, m_stream, m_counters_rgb.rays_per_batch, m_aabb, n_rays_total, m_rng,
        //         target_batch_size, ray_counter, LOSS_SCALE(), padded_output_width, m_envmap.view(), envmap_gradient,
        //         m_envmap.resolution, m_envmap.loss_type, m_background_color.rgb(), m_color_space,
        //         m_nerf.training.random_bg_color, m_nerf.training.linear_colors, m_nerf.training.n_images_for_training,
        //         m_nerf.training.dataset.metadata_gpu.data(), mlp_out, m_counters_rgb.numsteps_counter_compacted.data(),
        //         ray_indices, rays_unnormalized, numsteps,
        //         PitchedPtr<const NerfCoordinate>((NerfCoordinate*) coords, 1, 0, extra_stride),
        //         PitchedPtr<NerfCoordinate>((NerfCoordinate*) coords_compacted, 1, 0, extra_stride), dloss_dmlp_out,
        //         m_nerf.training.loss_type, m_nerf.training.depth_loss_type, m_counters_rgb.loss.data(),
        //         m_max_level_rand_training, max_level_compacted, m_nerf.rgb_activation, m_nerf.density_activation,
        //         m_nerf.training.snap_to_pixel_centers,
        //         accumulate_error ? m_nerf.training.error_map.data.data() : nullptr,
        //         nullptr,
        //         nullptr,
        //         nullptr,
        //         m_nerf.training.error_map.resolution, m_nerf.training.error_map.cdf_resolution,
        //         nullptr,
        //         m_nerf.training.dataset.sharpness_resolution, m_nerf.training.sharpness_grid.data(),
        //         m_nerf.density_grid.data(), m_nerf.density_grid_mean.data(), m_nerf.max_cascade,
        //         m_nerf.training.cam_exposure_gpu.data(),
        //         m_nerf.training.optimize_exposure ? m_nerf.training.cam_exposure_gradient_gpu.data() : nullptr,
        //         m_nerf.training.depth_supervision_lambda, m_nerf.training.near_distance);
    }


    m_trainer->optimizer_step(m_stream, tcnn::default_loss_scale<tcnn::network_precision_t>());
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
