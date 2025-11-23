#include "ngp.train.h"

#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/trainer.h>
#include <memory>
#include <iostream>

namespace ngp::train::cuda::impl {
    constexpr uint32_t N_THREADS_LINEAR = 128;

    template <typename T>
    TCNN_HOST_DEVICE T div_round_up(T val, T divisor) {
        return (val + divisor - 1) / divisor;
    }

    template <typename T>
    constexpr TCNN_HOST_DEVICE uint32_t n_blocks_linear(T n_elements, uint32_t n_threads = N_THREADS_LINEAR) {
        return (uint32_t) div_round_up(n_elements, (T) n_threads);
    }

    template <typename K, typename T, typename... Types>
    inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types... args) {
        if (n_elements <= 0) {
            return;
        }
        kernel<<<n_blocks_linear(n_elements), N_THREADS_LINEAR, shmem_size, stream>>>(n_elements, args...);
    }

    template <typename T>
    TCNN_HOST_DEVICE T next_multiple(T val, T divisor) {
        return div_round_up(val, divisor) * divisor;
    }

    template <typename F>
    __global__ void parallel_for_kernel(const size_t n_elements, F fun) {
        const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        fun(i);
    }

    template <typename F>
    inline void parallel_for_gpu(uint32_t shmem_size, cudaStream_t stream, size_t n_elements, F&& fun) {
        if (n_elements <= 0) {
            return;
        }
        parallel_for_kernel<F><<<n_blocks_linear(n_elements), N_THREADS_LINEAR, shmem_size, stream>>>(n_elements, fun);
    }

    template <typename F>
    inline void parallel_for_gpu(cudaStream_t stream, size_t n_elements, F&& fun) {
        parallel_for_gpu(0, stream, n_elements, std::forward<F>(fun));
    }

    template <typename F>
    inline void parallel_for_gpu(size_t n_elements, F&& fun) {
        parallel_for_gpu(nullptr, n_elements, std::forward<F>(fun));
    }

    template <typename F>
    __global__ void parallel_for_aos_kernel(const size_t n_elements, const uint32_t n_dims, F fun) {
        const size_t dim  = threadIdx.x;
        const size_t elem = threadIdx.y + blockIdx.x * blockDim.y;
        if (dim >= n_dims) return;
        if (elem >= n_elements) return;

        fun(elem, dim);
    }

    template <typename F>
    inline void parallel_for_gpu_aos(uint32_t shmem_size, cudaStream_t stream, size_t n_elements, uint32_t n_dims, F&& fun) {
        if (n_elements <= 0 || n_dims <= 0) {
            return;
        }

        const dim3 threads     = {n_dims, div_round_up(N_THREADS_LINEAR, n_dims), 1};
        const size_t n_threads = threads.x * threads.y;
        const dim3 blocks      = {(uint32_t) div_round_up(n_elements * n_dims, n_threads), 1, 1};

        parallel_for_aos_kernel<<<blocks, threads, shmem_size, stream>>>(
            n_elements, n_dims, fun
            );
    }

    template <typename F>
    inline void parallel_for_gpu_aos(cudaStream_t stream, size_t n_elements, uint32_t n_dims, F&& fun) {
        parallel_for_gpu_aos(0, stream, n_elements, n_dims, std::forward<F>(fun));
    }

    template <typename F>
    inline void parallel_for_gpu_aos(size_t n_elements, uint32_t n_dims, F&& fun) {
        parallel_for_gpu_aos(nullptr, n_elements, n_dims, std::forward<F>(fun));
    }

    template <typename F>
    __global__ void parallel_for_soa_kernel(const size_t n_elements, const uint32_t n_dims, F fun) {
        const size_t elem = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t dim  = blockIdx.y;
        if (elem >= n_elements) return;
        if (dim >= n_dims) return;

        fun(elem, dim);
    }

    template <typename F>
    inline void parallel_for_gpu_soa(uint32_t shmem_size, cudaStream_t stream, size_t n_elements, uint32_t n_dims, F&& fun) {
        if (n_elements <= 0 || n_dims <= 0) {
            return;
        }

        const dim3 blocks = {n_blocks_linear(n_elements), n_dims, 1};

        parallel_for_soa_kernel<<<n_blocks_linear(n_elements), N_THREADS_LINEAR, shmem_size, stream>>>(
            n_elements, n_dims, fun
            );
    }

    template <typename F>
    inline void parallel_for_gpu_soa(cudaStream_t stream, size_t n_elements, uint32_t n_dims, F&& fun) {
        parallel_for_gpu_soa(0, stream, n_elements, n_dims, std::forward<F>(fun));
    }

    template <typename F>
    inline void parallel_for_gpu_soa(size_t n_elements, uint32_t n_dims, F&& fun) {
        parallel_for_gpu_soa(nullptr, n_elements, n_dims, std::forward<F>(fun));
    }

    template <typename T>
    __global__ void extract_density(const uint32_t n_elements, const uint32_t density_stride,
        const uint32_t rgbd_stride, const T* __restrict__ density, T* __restrict__ rgbd) {
        const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        rgbd[i * rgbd_stride] = density[i * density_stride];
    }

    template <typename T>
    __global__ void extract_rgb(const uint32_t n_elements, const uint32_t rgb_stride, const uint32_t output_stride,
        const T* __restrict__ rgbd, T* __restrict__ rgb) {
        const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        const uint32_t elem_idx = i / 3;
        const uint32_t dim_idx  = i - elem_idx * 3;

        rgb[elem_idx * rgb_stride + dim_idx] = rgbd[elem_idx * output_stride + dim_idx];
    }

    template <typename T>
    __global__ void add_density_gradient(const uint32_t n_elements, const uint32_t rgbd_stride,
        const T* __restrict__ rgbd, const uint32_t density_stride,
        T* __restrict__ density) {
        const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        density[i * density_stride] += rgbd[i * rgbd_stride + 3];
    }

    template <typename T>
    class NerfNetwork final : public tcnn::Network<float, T> {
    public:
        NerfNetwork(uint32_t n_pos_dims, uint32_t n_dir_dims, uint32_t n_extra_dims, uint32_t dir_offset, const nlohmann::json& pos_encoding, const nlohmann::json& dir_encoding, const nlohmann::json& density_network, const nlohmann::json& rgb_network);
        ~NerfNetwork() override = default;
        tcnn::json hyperparams() const override;
        void set_params_impl(T* params, T* inference_params, T* gradients) override;
        void initialize_params(tcnn::pcg32& rnd, float* params_full_precision, float scale) override;
        size_t n_params() const override;
        std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override;
        void inference_mixed_precision_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_params) override;
        std::unique_ptr<tcnn::Context> forward_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>* output, bool use_inference_params, bool prepare_input_gradients) override;
        void backward_impl(cudaStream_t stream, const tcnn::Context& ctx, const tcnn::GPUMatrixDynamic<float>& input, const tcnn::GPUMatrixDynamic<T>& output, const tcnn::GPUMatrixDynamic<T>& dL_doutput, tcnn::GPUMatrixDynamic<float>* dL_dinput, bool use_inference_params, tcnn::GradientMode param_gradients_mode) override;
        uint32_t input_width() const override;
        uint32_t padded_output_width() const override;
        uint32_t output_width() const override;
        uint32_t required_input_alignment() const override;
        uint32_t width(uint32_t layer) const override;
        uint32_t num_forward_activations() const override;
        std::pair<const T*, tcnn::MatrixLayout> forward_activations(const tcnn::Context& ctx, uint32_t layer) const override;

    private:
        std::shared_ptr<tcnn::Network<T>> m_density_network;
        std::shared_ptr<tcnn::Network<T>> m_rgb_network;
        std::shared_ptr<tcnn::Encoding<T>> m_pos_encoding;
        std::shared_ptr<tcnn::Encoding<T>> m_dir_encoding;
        std::shared_ptr<tcnn::NetworkWithInputEncoding<T>> m_density_model;

        uint32_t m_rgb_network_input_width;
        uint32_t m_dir_offset;
        uint32_t m_n_dir_dims;
        uint32_t m_n_extra_dims;
        uint32_t m_n_pos_dims;

        struct ForwardContext final : public tcnn::Context {
            tcnn::GPUMatrixDynamic<T> density_network_input;
            tcnn::GPUMatrixDynamic<T> density_network_output;
            tcnn::GPUMatrixDynamic<T> rgb_network_input;
            tcnn::GPUMatrix<T> rgb_network_output;

            std::unique_ptr<Context> pos_encoding_ctx;
            std::unique_ptr<Context> dir_encoding_ctx;

            std::unique_ptr<Context> density_network_ctx;
            std::unique_ptr<Context> rgb_network_ctx;
        };
    };

    template <typename T>
    NerfNetwork<T>::NerfNetwork(const uint32_t n_pos_dims, const uint32_t n_dir_dims, const uint32_t n_extra_dims, const uint32_t dir_offset, const nlohmann::json& pos_encoding, const nlohmann::json& dir_encoding, const nlohmann::json& density_network, const nlohmann::json& rgb_network)
        : m_dir_offset{dir_offset}, m_n_dir_dims{n_dir_dims}, m_n_extra_dims{n_extra_dims}, m_n_pos_dims{n_pos_dims} {
        m_pos_encoding.reset(
            tcnn::create_encoding<T>(n_pos_dims, pos_encoding,
                density_network.contains("otype") &&
                (tcnn::equals_case_insensitive(density_network["otype"], "FullyFusedMLP") ||
                 tcnn::equals_case_insensitive(density_network["otype"], "MegakernelMLP"))
                    ? 16u
                    : 8u));
        uint32_t rgb_alignment = tcnn::minimum_alignment(rgb_network);
        m_dir_encoding.reset(tcnn::create_encoding<T>(m_n_dir_dims + m_n_extra_dims, dir_encoding, rgb_alignment));

        nlohmann::json local_density_network_config  = density_network;
        local_density_network_config["n_input_dims"] = m_pos_encoding->padded_output_width();
        if (!density_network.contains("n_output_dims")) {
            local_density_network_config["n_output_dims"] = 16;
        }
        m_density_network.reset(tcnn::create_network<T>(local_density_network_config));

        m_rgb_network_input_width = next_multiple(
            m_dir_encoding->padded_output_width() + m_density_network->padded_output_width(), rgb_alignment);

        nlohmann::json local_rgb_network_config   = rgb_network;
        local_rgb_network_config["n_input_dims"]  = m_rgb_network_input_width;
        local_rgb_network_config["n_output_dims"] = 3;
        m_rgb_network.reset(tcnn::create_network<T>(local_rgb_network_config));

        m_density_model = std::make_shared<tcnn::NetworkWithInputEncoding<T>>(m_pos_encoding, m_density_network);
    }

    template <typename T>
    tcnn::json NerfNetwork<T>::hyperparams() const {
        nlohmann::json density_network_hyperparams   = m_density_network->hyperparams();
        density_network_hyperparams["n_output_dims"] = m_density_network->padded_output_width();
        return {
            {"otype", "NerfNetwork"},
            {"pos_encoding", m_pos_encoding->hyperparams()},
            {"dir_encoding", m_dir_encoding->hyperparams()},
            {"density_network", density_network_hyperparams},
            {"rgb_network", m_rgb_network->hyperparams()},
        };
    }

    template <typename T>
    void NerfNetwork<T>::set_params_impl(T* params, T* inference_params, T* gradients) {
        m_density_model->set_params(params, inference_params, gradients);

        size_t offset = 0;
        m_density_network->set_params(params + offset, inference_params + offset, gradients + offset);
        offset += m_density_network->n_params();

        m_rgb_network->set_params(params + offset, inference_params + offset, gradients + offset);
        offset += m_rgb_network->n_params();

        m_pos_encoding->set_params(params + offset, inference_params + offset, gradients + offset);
        offset += m_pos_encoding->n_params();

        m_dir_encoding->set_params(params + offset, inference_params + offset, gradients + offset);
        offset += m_dir_encoding->n_params();
    }

    template <typename T>
    void NerfNetwork<T>::initialize_params(tcnn::pcg32& rnd, float* params_full_precision, float scale) {
        m_density_network->initialize_params(rnd, params_full_precision, scale);
        params_full_precision += m_density_network->n_params();

        m_rgb_network->initialize_params(rnd, params_full_precision, scale);
        params_full_precision += m_rgb_network->n_params();

        m_pos_encoding->initialize_params(rnd, params_full_precision, scale);
        params_full_precision += m_pos_encoding->n_params();

        m_dir_encoding->initialize_params(rnd, params_full_precision, scale);
        params_full_precision += m_dir_encoding->n_params();
    }

    template <typename T>
    size_t NerfNetwork<T>::n_params() const {
        return m_pos_encoding->n_params() + m_density_network->n_params() + m_dir_encoding->n_params() + m_rgb_network->n_params();
    }

    template <typename T>
    std::vector<std::pair<uint32_t, uint32_t>> NerfNetwork<T>::layer_sizes() const {
        auto layers     = m_density_network->layer_sizes();
        auto rgb_layers = m_rgb_network->layer_sizes();
        layers.insert(layers.end(), rgb_layers.begin(), rgb_layers.end());
        return layers;
    }

    template <typename T>
    void NerfNetwork<T>::inference_mixed_precision_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_params) {
        uint32_t batch_size = input.n();
        tcnn::GPUMatrixDynamic<T> density_network_input{m_pos_encoding->padded_output_width(), batch_size, stream,
                                                        m_pos_encoding->preferred_output_layout()};
        tcnn::GPUMatrixDynamic<T> rgb_network_input{m_rgb_network_input_width, batch_size, stream,
                                                    m_dir_encoding->preferred_output_layout()};

        tcnn::GPUMatrixDynamic<T> density_network_output =
            rgb_network_input.slice_rows(0, m_density_network->padded_output_width());
        tcnn::GPUMatrixDynamic<T> rgb_network_output{output.data(), m_rgb_network->padded_output_width(), batch_size,
                                                     output.layout()};

        m_pos_encoding->inference_mixed_precision(stream, input.slice_rows(0, m_pos_encoding->input_width()),
            density_network_input, use_inference_params);

        m_density_network->inference_mixed_precision(stream, density_network_input, density_network_output,
            use_inference_params);

        auto dir_out = rgb_network_input.slice_rows(m_density_network->padded_output_width(),
            m_dir_encoding->padded_output_width());
        m_dir_encoding->inference_mixed_precision(
            stream, input.slice_rows(m_dir_offset, m_dir_encoding->input_width()), dir_out, use_inference_params);

        m_rgb_network->inference_mixed_precision(stream, rgb_network_input, rgb_network_output,
            use_inference_params);

        linear_kernel(extract_density<T>, 0, stream, batch_size,
            density_network_output.layout() == tcnn::AoS ? density_network_output.stride() : 1,
            output.layout() == tcnn::AoS ? padded_output_width() : 1, density_network_output.data(),
            output.data() + 3 * (output.layout() == tcnn::AoS ? 1 : batch_size));
    }

    template <typename T>
    std::unique_ptr<tcnn::Context> NerfNetwork<T>::forward_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>* output, bool use_inference_params, bool prepare_input_gradients) {
        // Make sure our temporary buffers have the correct size for the given batch size
        uint32_t batch_size = input.n();

        auto forward = std::make_unique<ForwardContext>();

        forward->density_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size,
                                                                   stream, m_pos_encoding->preferred_output_layout()};
        forward->rgb_network_input = tcnn::GPUMatrixDynamic<T>{m_rgb_network_input_width, batch_size, stream,
                                                               m_dir_encoding->preferred_output_layout()};

        forward->pos_encoding_ctx =
            m_pos_encoding->forward(stream, input.slice_rows(0, m_pos_encoding->input_width()),
                &forward->density_network_input, use_inference_params, prepare_input_gradients);

        forward->density_network_output =
            forward->rgb_network_input.slice_rows(0, m_density_network->padded_output_width());
        forward->density_network_ctx =
            m_density_network->forward(stream, forward->density_network_input, &forward->density_network_output,
                use_inference_params, prepare_input_gradients);

        auto dir_out = forward->rgb_network_input.slice_rows(m_density_network->padded_output_width(),
            m_dir_encoding->padded_output_width());
        forward->dir_encoding_ctx =
            m_dir_encoding->forward(stream, input.slice_rows(m_dir_offset, m_dir_encoding->input_width()), &dir_out,
                use_inference_params, prepare_input_gradients);

        if (output) {
            forward->rgb_network_output = tcnn::GPUMatrixDynamic<T>{output->data(), m_rgb_network->padded_output_width(),
                                                                    batch_size, output->layout()};
        }

        forward->rgb_network_ctx = m_rgb_network->forward(stream, forward->rgb_network_input,
            output ? &forward->rgb_network_output : nullptr,
            use_inference_params, prepare_input_gradients);

        if (output) {
            linear_kernel(
                extract_density<T>, 0, stream, batch_size,
                m_dir_encoding->preferred_output_layout() == tcnn::AoS ? forward->density_network_output.stride() : 1,
                padded_output_width(), forward->density_network_output.data(), output->data() + 3);
        }

        return forward;
    }

    template <typename T>
    void NerfNetwork<T>::backward_impl(cudaStream_t stream, const tcnn::Context& ctx, const tcnn::GPUMatrixDynamic<float>& input, const tcnn::GPUMatrixDynamic<T>& output, const tcnn::GPUMatrixDynamic<T>& dL_doutput, tcnn::GPUMatrixDynamic<float>* dL_dinput, bool use_inference_params, tcnn::GradientMode param_gradients_mode) {
        const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

        // Make sure our teporary buffers have the correct size for the given batch size
        uint32_t batch_size = input.n();

        tcnn::GPUMatrix<T> dL_drgb{m_rgb_network->padded_output_width(), batch_size, stream};
        CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb.data(), 0, dL_drgb.n_bytes(), stream));
        linear_kernel(extract_rgb<T>, 0, stream, batch_size * 3, dL_drgb.m(), dL_doutput.m(), dL_doutput.data(),
            dL_drgb.data());

        const tcnn::GPUMatrixDynamic<T> rgb_network_output{static_cast<T*>(output.data()), m_rgb_network->padded_output_width(),
                                                           batch_size, output.layout()};
        tcnn::GPUMatrixDynamic<T> dL_drgb_network_input{m_rgb_network_input_width, batch_size, stream,
                                                        m_dir_encoding->preferred_output_layout()};
        m_rgb_network->backward(stream, *forward.rgb_network_ctx, forward.rgb_network_input, rgb_network_output,
            dL_drgb, &dL_drgb_network_input, use_inference_params, param_gradients_mode);

        // Backprop through dir encoding if it is trainable or if we need input gradients
        if (m_dir_encoding->n_params() > 0 || dL_dinput) {
            tcnn::GPUMatrixDynamic<T> dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(
                m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
            tcnn::GPUMatrixDynamic<float> dL_ddir_encoding_input;
            if (dL_dinput) {
                dL_ddir_encoding_input = dL_dinput->slice_rows(m_dir_offset, m_dir_encoding->input_width());
            }

            m_dir_encoding->backward(stream, *forward.dir_encoding_ctx,
                input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
                forward.rgb_network_input.slice_rows(m_density_network->padded_output_width(),
                    m_dir_encoding->padded_output_width()),
                dL_ddir_encoding_output, dL_dinput ? &dL_ddir_encoding_input : nullptr,
                use_inference_params, param_gradients_mode);
        }

        tcnn::GPUMatrixDynamic<T> dL_ddensity_network_output =
            dL_drgb_network_input.slice_rows(0, m_density_network->padded_output_width());
        linear_kernel(add_density_gradient<T>, 0, stream, batch_size, dL_doutput.m(), dL_doutput.data(),
            dL_ddensity_network_output.layout() == tcnn::RM ? 1 : dL_ddensity_network_output.stride(),
            dL_ddensity_network_output.data());

        tcnn::GPUMatrixDynamic<T> dL_ddensity_network_input;
        if (m_pos_encoding->n_params() > 0 || dL_dinput) {
            dL_ddensity_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size,
                                                                  stream, m_pos_encoding->preferred_output_layout()};
        }

        m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input,
            forward.density_network_output, dL_ddensity_network_output,
            dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr,
            use_inference_params, param_gradients_mode);

        // Backprop through pos encoding if it is trainable or if we need input gradients
        if (dL_ddensity_network_input.data()) {
            tcnn::GPUMatrixDynamic<float> dL_dpos_encoding_input;
            if (dL_dinput) {
                dL_dpos_encoding_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
            }

            m_pos_encoding->backward(
                stream, *forward.pos_encoding_ctx, input.slice_rows(0, m_pos_encoding->input_width()),
                forward.density_network_input, dL_ddensity_network_input,
                dL_dinput ? &dL_dpos_encoding_input : nullptr, use_inference_params, param_gradients_mode);
        }
    }

    template <typename T>
    uint32_t NerfNetwork<T>::input_width() const {
        return m_dir_offset + m_n_dir_dims + m_n_extra_dims;
    }

    template <typename T>
    uint32_t NerfNetwork<T>::padded_output_width() const {
        return std::max(m_rgb_network->padded_output_width(), static_cast<uint32_t>(4));
    }

    template <typename T>
    uint32_t NerfNetwork<T>::output_width() const {
        return m_density_network->padded_output_width();
    }

    template <typename T>
    uint32_t NerfNetwork<T>::required_input_alignment() const {
        return 1; // No alignment required due to encoding
    }

    template <typename T>
    uint32_t NerfNetwork<T>::width(uint32_t layer) const {
        if (layer == 0) {
            return m_pos_encoding->padded_output_width();
        } else if (layer < m_density_network->num_forward_activations() + 1) {
            return m_density_network->width(layer - 1);
        } else if (layer == m_density_network->num_forward_activations() + 1) {
            return m_rgb_network_input_width;
        } else {
            return m_rgb_network->width(layer - 2 - m_density_network->num_forward_activations());
        }
    }

    template <typename T>
    uint32_t NerfNetwork<T>::num_forward_activations() const {
        return m_density_network->num_forward_activations() + m_rgb_network->num_forward_activations() + 2;
    }

    template <typename T>
    std::pair<const T*, tcnn::MatrixLayout> NerfNetwork<T>::forward_activations(const tcnn::Context& ctx, uint32_t layer) const {
        const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
        if (layer == 0) {
            return {forward.density_network_input.data(), m_pos_encoding->preferred_output_layout()};
        } else if (layer < m_density_network->num_forward_activations() + 1) {
            return m_density_network->forward_activations(*forward.density_network_ctx, layer - 1);
        } else if (layer == m_density_network->num_forward_activations() + 1) {
            return {forward.rgb_network_input.data(), m_dir_encoding->preferred_output_layout()};
        } else {
            return m_rgb_network->forward_activations(*forward.rgb_network_ctx,
                layer - 2 - m_density_network->num_forward_activations());
        }
    }

    struct NGPContext {
        static NGPContext& instance() {
            static NGPContext instance;
            return instance;
        }

        void reset_session(const nlohmann::json& config);

    private:
        NGPContext()  = default;
        ~NGPContext() = default;

        std::shared_ptr<tcnn::Loss<tcnn::network_precision_t>> m_loss;
        std::shared_ptr<tcnn::Optimizer<tcnn::network_precision_t>> m_optimizer;
        std::shared_ptr<tcnn::Network<float, tcnn::network_precision_t>> m_network;
        std::shared_ptr<tcnn::Trainer<float, tcnn::network_precision_t, tcnn::network_precision_t>> m_trainer;
        uint32_t m_seed = 1337;
    };

    void NGPContext::reset_session(const nlohmann::json& config) {
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
        m_network = std::make_shared<NerfNetwork<tcnn::network_precision_t>>(n_pos, n_dir_dims, n_extra_dims, n_pos + 1, encoding_config, dir_encoding_config, network_config, rgb_network_config);
        m_trainer = std::make_shared<tcnn::Trainer<float, tcnn::network_precision_t, tcnn::network_precision_t>>(m_network, m_optimizer, m_loss, m_seed);

        auto optimizer_config_expand          = optimizer_config;
        nlohmann::json* leaf_optimizer_config = &optimizer_config_expand;
        while (leaf_optimizer_config->contains("nested")) leaf_optimizer_config = &(*leaf_optimizer_config)["nested"];
        (*leaf_optimizer_config)["optimize_matrix_params"]     = true;
        (*leaf_optimizer_config)["optimize_non_matrix_params"] = true;
        m_optimizer->update_hyperparams(optimizer_config_expand);
    }
}

namespace ngp::train::cuda {
    void find_devices() {
        printf("Finding CUDA devices...\n");
    }

    void reset_context(const nlohmann::json& config) {
        impl::NGPContext::instance().reset_session(config);
    }
}
