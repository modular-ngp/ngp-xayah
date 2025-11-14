export module ngp.device;

extern "C" void launch_kernel();

export void run_kernel() {
    launch_kernel();
}

// import ngp.dataset.nerfsynthetic;
// namespace ngp::device {
//     export void to_device(const ngp::dataset::NeRFSyntheticDataset& dataset);
//
//     void to_device(const ngp::dataset::NeRFSyntheticDataset& dataset) {
//         tcnn::GPUMemory<uint8_t> images_data_gpu_tmp;
//     }
// } // namespace ngp::device
