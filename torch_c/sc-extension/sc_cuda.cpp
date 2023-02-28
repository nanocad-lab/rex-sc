#include <torch/extension.h>
#include <iostream>

namespace F = torch::nn::functional;

/*
 * Accelerated GPU implementation. Helper functions
 * Supported functions right now:
 * OR-n implementations. n=1 is normal OR
 */

std::vector<torch::Tensor> conv2d_generic_cuda_general_gemm(
    torch::Tensor input,
    torch::Tensor weight_pos,
    torch::Tensor weight_neg,
    at::IntArrayRef stride,
    at::IntArrayRef prog_load,
    int bit_length,
    int lfsr_length,
    int z_units,
    int bin_config, //0 = full or; 1 = 1d_bin; 2 = yz_bin; 3 = z_bin
    int gen_config, //0 = rand; 1 = lfsr; 2 = lfsr_acc(WIP); 3 = rand_acc
    bool xnor,
    bool mux);

std::vector<torch::Tensor> linear_generic_cuda_general(
    torch::Tensor input,
    torch::Tensor weight_pos,
    torch::Tensor weight_neg,
    at::IntArrayRef prog_load,
    int bit_length,
    int lfsr_length,
    int bin_config,
    int gen_config,
    bool xnor,
    bool mux);

torch::Tensor or_approx_n_forward_acc(
    torch::Tensor input,
    int or_n);

torch::Tensor or_approx_n_backward_acc(
    torch::Tensor grad_output,
    torch::Tensor input,
    int or_n);

torch::Tensor or_approx_n_forward_bias_std_acc(
    torch::Tensor input,
    torch::Tensor bias_coef,
    torch::Tensor std_coef,
    int or_n);

torch::Tensor or_approx_n_backward_bias_std_acc(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor bias_coef,
    int or_n);

torch::Tensor or_approx_n_forward_bias_std_both_acc(
    torch::Tensor input_pos,
    torch::Tensor input_neg,
    torch::Tensor bias_coef,
    torch::Tensor std_coef,
    int or_n);

std::vector<torch::Tensor> or_approx_n_backward_bias_std_both_acc(
    torch::Tensor grad_output,
    torch::Tensor input_pos,
    torch::Tensor input_neg,
    torch::Tensor bias_coef,
    int or_n);

at::Tensor conv2d_generic_general_acc(torch::Tensor input, torch::Tensor weight, int bit_length, int lfsr_length, int z_units, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef prog_load, int bin_config, int gen_config, bool xnor, bool mux) {
    auto input_pad = F::pad(input, F::PadFuncOptions({padding.data()[0], padding.data()[0], padding.data()[1], padding.data()[1]})/*.mode(torch::kReplicate)*/);
    auto compare_type = torch::kInt32;
    int lfsr_bit_length = bit_length; //Scaling factor for the floating-point weights/activations
    if(lfsr_length>0) lfsr_bit_length = (1<<lfsr_length);
    auto input_split = (input_pad*lfsr_bit_length).clamp(0, lfsr_bit_length-1).ceil().to(compare_type).clone();
    auto w_pos_split = (weight*lfsr_bit_length).clamp(0, lfsr_bit_length-1).ceil().to(compare_type);
    auto w_neg_split = (-(weight*lfsr_bit_length).clamp(1-lfsr_bit_length, 0)).ceil().to(compare_type);

    std::vector<torch::Tensor> output_split;
    output_split = conv2d_generic_cuda_general_gemm(input_split, w_pos_split, w_neg_split, stride, prog_load, bit_length, lfsr_length, z_units, bin_config, gen_config, xnor, mux);
    return output_split[0]-output_split[1];
}

std::vector<at::Tensor> conv2d_generic_general_split_acc(torch::Tensor input, torch::Tensor weight, int bit_length, int lfsr_length, int z_units, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef prog_load, int bin_config, int gen_config, bool xnor, bool mux) {
    auto input_pad = F::pad(input, F::PadFuncOptions({padding.data()[0], padding.data()[0], padding.data()[1], padding.data()[1]})/*.mode(torch::kReplicate)*/);
    auto compare_type = torch::kInt32;
    int lfsr_bit_length = bit_length; //Scaling factor for the floating-point weights/activations
    if(lfsr_length>0) lfsr_bit_length = (1<<lfsr_length);
    auto input_split = (input_pad*lfsr_bit_length).clamp(0, lfsr_bit_length-1).ceil().to(compare_type).clone();
    auto w_pos_split = (weight*lfsr_bit_length).clamp(0, lfsr_bit_length-1).ceil().to(compare_type);
    auto w_neg_split = (-(weight*lfsr_bit_length).clamp(1-lfsr_bit_length, 0)).ceil().to(compare_type);

    std::vector<torch::Tensor> output_split;
    output_split = conv2d_generic_cuda_general_gemm(input_split, w_pos_split, w_neg_split, stride, prog_load, bit_length, lfsr_length, z_units, bin_config, gen_config, xnor, mux);
    return output_split;
}

std::vector<at::Tensor> linear_generic_general_acc(
    torch::Tensor input,
    torch::Tensor weight,
    at::IntArrayRef prog_load,
    int bit_length,
    int lfsr_length,
    int bin_config,
    int gen_config,
    bool xnor,
    bool mux) {
    auto compare_type = torch::kInt32;
    int lfsr_bit_length = bit_length;
    if(lfsr_length>0) lfsr_bit_length = (1<<lfsr_length);
    auto input_split = (input*lfsr_bit_length).clamp(0, lfsr_bit_length-1).ceil().to(compare_type).clone();
    auto w_pos_split = (weight*lfsr_bit_length).clamp(0, lfsr_bit_length-1).ceil().to(compare_type);
    auto w_neg_split = (-(weight*lfsr_bit_length).clamp(1-lfsr_bit_length, 0)).ceil().to(compare_type);

    auto output_split = linear_generic_cuda_general(input_split, w_pos_split, w_neg_split, prog_load, bit_length, lfsr_length, bin_config, gen_config, xnor, mux);
    return output_split;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_generic_general_acc", &conv2d_generic_general_acc, "General SC forward conv2d for generalized SC");
    m.def("conv2d_generic_general_split_acc", &conv2d_generic_general_split_acc, "General SC forward conv2d for generalized SC with split pos and neg");
    m.def("linear_generic_general_acc", &linear_generic_general_acc, "General SC forward linear");
    m.def("or_approx_n_forward_acc", &or_approx_n_forward_acc, "OR forward n");
    m.def("or_approx_n_backward_acc", &or_approx_n_backward_acc, "OR backward n");
    m.def("or_approx_n_forward_bias_std_acc", &or_approx_n_forward_bias_std_acc, "OR forward bias n");
    m.def("or_approx_n_backward_bias_std_acc", &or_approx_n_backward_bias_std_acc, "OR backward bias n");
    m.def("or_approx_n_forward_bias_std_both_acc", &or_approx_n_forward_bias_std_both_acc, "OR forward pos+neg n");
    m.def("or_approx_n_backward_bias_std_both_acc", &or_approx_n_backward_bias_std_both_acc, "OR backward pos+neg");
}
