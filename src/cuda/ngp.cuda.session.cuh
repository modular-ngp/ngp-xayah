#ifndef NGP_XAYAH_NGP_CUDA_SESSION_H
#define NGP_XAYAH_NGP_CUDA_SESSION_H

#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/rtc_kernel.h>
#include <json/json.hpp>
#include <pcg32/pcg32.h>

#include "ngp.cuda.boundingbox.cuh"
#include "ngp.cuda.utils.cuh"

namespace tcnn {
    template <typename T>
    class Loss;
    template <typename T>
    class Optimizer;
    template <typename T>
    class Encoding;
    template <typename T, typename PARAMS_T>
    class Network;
    template <typename T, typename PARAMS_T, typename COMPUTE_T>
    class Trainer;
    template <uint32_t N_DIMS, uint32_t RANK, typename T>
    class TrainableBuffer;
}

namespace ngp::cuda {

    struct NGPSession {
        static NGPSession& instance() {
            static NGPSession instance;
            return instance;
        }

        void reset_session(const nlohmann::json& config);
        void train(uint32_t batchsize);

    private:
        NGPSession()  = default;
        ~NGPSession() = default;

        // struct Nerf
        //         {
        //     struct Training
        //     {
        //         NerfDataset dataset;
        //         int n_images_for_training =
        //             0; // how many images to train from, as a high watermark compared to the dataset size
        //         int n_images_for_training_prev = 0; // how many images we saw last time we updated the density grid
        //
        //         struct ErrorMap
        //         {
        //             tcnn::GPUMemory<float> data;
        //             tcnn::GPUMemory<float> cdf_x_cond_y;
        //             tcnn::GPUMemory<float> cdf_y;
        //             tcnn::GPUMemory<float> cdf_img;
        //             std::vector<float> pmf_img_cpu;
        //             tcnn::ivec2 resolution = {16, 16};
        //             tcnn::ivec2 cdf_resolution = {16, 16};
        //             bool is_cdf_valid = false;
        //         } error_map;
        //
        //         std::vector<TrainingXForm> transforms;
        //         tcnn::GPUMemory<TrainingXForm> transforms_gpu;
        //
        //         std::vector<tcnn::vec3> cam_pos_gradient;
        //         tcnn::GPUMemory<tcnn::vec3> cam_pos_gradient_gpu;
        //
        //         std::vector<tcnn::vec3> cam_rot_gradient;
        //         tcnn::GPUMemory<tcnn::vec3> cam_rot_gradient_gpu;
        //
        //         tcnn::GPUMemory<tcnn::vec3> cam_exposure_gpu;
        //         std::vector<tcnn::vec3> cam_exposure_gradient;
        //         tcnn::GPUMemory<tcnn::vec3> cam_exposure_gradient_gpu;
        //
        //         tcnn::vec2 cam_focal_length_gradient = tcnn::vec2(0.0f);
        //         tcnn::GPUMemory<tcnn::vec2> cam_focal_length_gradient_gpu;
        //
        //         std::vector<AdamOptimizer<tcnn::vec3>> cam_exposure;
        //         std::vector<AdamOptimizer<tcnn::vec3>> cam_pos_offset;
        //         std::vector<RotationAdamOptimizer> cam_rot_offset;
        //         AdamOptimizer<tcnn::vec2> cam_focal_length_offset = AdamOptimizer<tcnn::vec2>(0.0f);
        //
        //         tcnn::GPUMemory<float>
        //             extra_dims_gpu; // if the model demands a latent code per training image, we put them in here.
        //         tcnn::GPUMemory<float> extra_dims_gradient_gpu;
        //         std::vector<VarAdamOptimizer> extra_dims_opt;
        //
        //         std::vector<float> get_extra_dims_cpu(int trainview) const;
        //
        //         float extrinsic_l2_reg = 1e-4f;
        //         float extrinsic_learning_rate = 1e-3f;
        //
        //         float intrinsic_l2_reg = 1e-4f;
        //         float exposure_l2_reg = 0.0f;
        //
        //         NerfCounters counters_rgb;
        //
        //         bool random_bg_color = true;
        //         bool linear_colors = false;
        //         ELossType loss_type = ELossType::L2;
        //         ELossType depth_loss_type = ELossType::L1;
        //         bool snap_to_pixel_centers = true;
        //         bool train_envmap = false;
        //
        //         bool optimize_distortion = false;
        //         bool optimize_extrinsics = false;
        //         bool optimize_extra_dims = false;
        //         bool optimize_focal_length = false;
        //         bool optimize_exposure = false;
        //         bool render_error_overlay = false;
        //         float error_overlay_brightness = 0.125f;
        //         uint32_t n_steps_between_cam_updates = 16;
        //         uint32_t n_steps_since_cam_update = 0;
        //
        //         bool sample_focal_plane_proportional_to_error = false;
        //         bool sample_image_proportional_to_error = false;
        //         bool include_sharpness_in_error = false;
        //         uint32_t n_steps_between_error_map_updates = 128;
        //         uint32_t n_steps_since_error_map_update = 0;
        //         uint32_t n_rays_since_error_map_update = 0;
        //
        //         float near_distance = 0.1f;
        //         float density_grid_decay = 0.95f;
        //         default_rng_t density_grid_rng;
        //         int view = 0;
        //
        //         ETrainMode train_mode = ETrainMode::RflRelax;
        //
        //         float depth_supervision_lambda = 0.f;
        //
        //         tcnn::GPUMemory<float> sharpness_grid;
        //
        //         std::unique_ptr<CudaRtcKernel> fused_kernel;
        //
        //         void set_camera_intrinsics(int frame_idx, float fx, float fy = 0.0f, float cx = -0.5f, float cy = -0.5f,
        //                                    float k1 = 0.0f, float k2 = 0.0f, float p1 = 0.0f, float p2 = 0.0f,
        //                                    float k3 = 0.0f, float k4 = 0.0f, bool is_fisheye = false);
        //         void set_camera_extrinsics_rolling_shutter(int frame_idx, tcnn::mat4x3 camera_to_world_start,
        //                                                    tcnn::mat4x3 camera_to_world_end, const tcnn::vec4& rolling_shutter,
        //                                                    bool convert_to_ngp = true);
        //         void set_camera_extrinsics(int frame_idx, tcnn::mat4x3 camera_to_world, bool convert_to_ngp = true);
        //         tcnn::mat4x3 get_camera_extrinsics(int frame_idx);
        //         void update_transforms(int first = 0, int last = -1);
        //         void update_extra_dims();
        //
        //         void reset_camera_extrinsics();
        //         void export_camera_extrinsics(const fs::path& path, bool export_extrinsics_in_quat_format = true);
        //     } training = {};
        //
        //     tcnn::GPUMemory<float> density_grid; // NERF_GRIDSIZE()^3 grid of EMA smoothed densities from the network
        //     tcnn::GPUMemory<uint8_t> density_grid_bitfield;
        //     uint8_t* get_density_grid_bitfield_mip(uint32_t mip);
        //     tcnn::GPUMemory<float> density_grid_mean;
        //     uint32_t density_grid_ema_step = 0;
        //
        //     uint32_t max_cascade = 0;
        //
        //     ENerfActivation rgb_activation = ENerfActivation::Exponential;
        //     ENerfActivation density_activation = ENerfActivation::Exponential;
        //
        //     tcnn::vec3 light_dir = tcnn::vec3(0.5f);
        //     // which training image's latent code should be used for rendering
        //     int rendering_extra_dims_from_training_view = 0;
        //     tcnn::GPUMemory<float> rendering_extra_dims;
        //
        //     void reset_extra_dims(default_rng_t& rng);
        //     const float* get_rendering_extra_dims(cudaStream_t stream) const;
        //
        //     int show_accel = -1;
        //
        //     float sharpen = 0.f;
        //
        //     float cone_angle_constant = 1.f / 256.f;
        //
        //     bool surface_rendering = false;
        //     float surface_rendering_threshold = 0.5f;
        //
        //     bool visualize_cameras = false;
        //
        //     float render_min_transmittance = 0.01f;
        //     bool render_gbuffer_hard_edges = false;
        //
        //     int find_closest_training_view(tcnn::mat4x3 pose) const;
        //     void set_rendering_extra_dims_from_training_view(int trainview);
        //     void set_rendering_extra_dims(const std::vector<float>& vals);
        //     std::vector<float> get_rendering_extra_dims_cpu() const;
        // } m_nerf;

        struct NerfCounters {
            tcnn::GPUMemory<uint32_t> numsteps_counter; // number of steps each ray took
            tcnn::GPUMemory<uint32_t> numsteps_counter_compacted; // number of steps each ray took
            tcnn::GPUMemory<float> loss;

            uint32_t rays_per_batch                        = 1 << 12;
            uint32_t n_rays_total                          = 0;
            uint32_t measured_batch_size                   = 0;
            uint32_t measured_batch_size_before_compaction = 0;

            void prepare_for_training_steps(cudaStream_t stream);
            float update_after_training(uint32_t target_batch_size, bool get_loss_scalar, cudaStream_t stream);
        } counters;

        BoundingBox m_aabb = {tcnn::vec3(0.0f), tcnn::vec3(1.0f)};

        std::shared_ptr<tcnn::Loss<tcnn::network_precision_t>> m_loss;
        std::shared_ptr<tcnn::Optimizer<tcnn::network_precision_t>> m_optimizer;
        std::shared_ptr<tcnn::Network<float, tcnn::network_precision_t>> m_network;
        std::shared_ptr<tcnn::Encoding<tcnn::network_precision_t>> m_encoding;
        std::shared_ptr<tcnn::Trainer<float, tcnn::network_precision_t, tcnn::network_precision_t>> m_trainer;
        uint32_t m_seed = 1337;
        cudaStream_t m_stream;
        tcnn::pcg32 m_rng;

        uint32_t m_training_step               = 0;
        uint32_t n_rays_since_error_map_update = 0;
        bool m_max_level_rand_training         = false;

        std::unique_ptr<tcnn::CudaRtcKernel> m_fused_kernel;
    };
}

#endif //NGP_XAYAH_NGP_CUDA_SESSION_H
