#include "ngp.cuda.session.cuh"
#include "ngp.cuda.nerfnetwork.cuh"
#include "ngp.cuda.boundingbox.cuh"
#include "ngp.cuda.envmap.cuh"

#include <tiny-cuda-nn/encodings/multi_level_interface.h>


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

    // {
    //     tcnn::linear_kernel(
    //         generate_training_samples_nerf, 0, m_stream, m_counters_rgb.rays_per_batch, m_aabb, max_inference, n_rays_total,
    //         m_rng, ray_counter, m_counters_rgb.numsteps_counter.data(), ray_indices, rays_unnormalized, numsteps,
    //         PitchedPtr<NerfCoordinate>((NerfCoordinate*) coords, 1, 0, extra_stride),
    //         m_nerf.training.n_images_for_training, m_nerf.training.dataset.metadata_gpu.data(),
    //         m_nerf.training.transforms_gpu.data(), m_nerf.density_grid_bitfield.data(), m_nerf.max_cascade,
    //         m_max_level_rand_training, max_level, m_nerf.training.snap_to_pixel_centers,
    //         m_nerf.training.train_envmap, m_nerf.cone_angle_constant, m_distortion.view(),
    //         nullptr,
    //         nullptr,
    //         nullptr,
    //         m_nerf.training.error_map.cdf_resolution, m_nerf.training.extra_dims_gpu.data(),
    //         m_nerf_network->n_extra_dims());
    //
    //     if (hg_enc) {
    //         hg_enc->set_max_level_gpu(m_max_level_rand_training ? max_level : nullptr);
    //     }
    //
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
    // }


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
