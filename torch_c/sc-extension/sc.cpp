#include <torch/extension.h>

std::vector<torch::Tensor> linear_generic_general(
    torch::Tensor input,
    torch::Tensor weight,
    at::IntArrayRef prog_load,
    int bit_length,
    int lfsr_length,
    int bin_config,
    int gen_config,
    bool xnor,
    bool mux);

torch::Tensor conv2d_generic_general(torch::Tensor input, 
        torch::Tensor weight, 
        int bit_length, 
        int lfsr_length, 
        int z_units, 
        at::IntArrayRef padding, 
        at::IntArrayRef stride, 
        at::IntArrayRef prog_load, 
        int bin_config, 
        int gen_config, 
        bool xnor, 
        bool mux);

torch::Tensor test_throughput(
    torch::Tensor activation_test,
    torch::Tensor weight_pos_test,
    torch::Tensor weight_neg_test,
    torch::Tensor output_pos_test,
    torch::Tensor output_neg_test);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_generic_general", &conv2d_generic_general, "SC forward generic generalized SC");
    m.def("linear_generic_general", &linear_generic_general, "SC forward generic version");
}