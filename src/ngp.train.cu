#include "ngp.train.h"

#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/trainer.h>
#include <memory>

namespace ngp::train::cuda {
    struct NGPContext {
        static NGPContext& instance() {
            static NGPContext instance;
            return instance;
        }

    private:
        NGPContext();
        ~NGPContext();
        std::shared_ptr<tcnn::Loss<tcnn::network_precision_t>> m_loss;
        std::shared_ptr<tcnn::Optimizer<tcnn::network_precision_t>> m_optimizer;
        std::shared_ptr<tcnn::Encoding<tcnn::network_precision_t>> m_encoding;
        std::shared_ptr<tcnn::Network<float, tcnn::network_precision_t>> m_network;
        std::shared_ptr<tcnn::Trainer<float, tcnn::network_precision_t, tcnn::network_precision_t>> m_trainer;
    };

    NGPContext::NGPContext()  = default;
    NGPContext::~NGPContext() = default;

    void find_devices() {
        printf("Finding CUDA devices...\n");
    }
}
