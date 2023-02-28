#include <torch/extension.h>
#include <iostream>

namespace F = torch::nn::functional;

/*
 * Accelerated GPU implementation. Function definitions
 * input: input tensor in NCHW format (default of PyTorch)
 * weight: weight tensor in NCHW format
 * add_config: 0 => fixed-point add
 *             >=1 -> simulated SC add. You can ignore these configs and only use add_config=1
 * precision_in: input and weight precision
 * precision_out: precision of intermediate addition results. Currently all additions are quantized
 */

torch::Tensor conv2d_saturate_cuda(torch::Tensor input,
        torch::Tensor weight_pos,
        torch::Tensor weight_neg,
        at::IntArrayRef stride,
        int add_config,
        int precision_in,
        int precision_out);

at::Tensor conv2d_saturate_acc(torch::Tensor input, torch::Tensor weight, int add_config, int precision_in, int precision_out, at::IntArrayRef padding, at::IntArrayRef stride) {
    auto input_pad = F::pad(input, F::PadFuncOptions({padding[0], padding[0], padding[1], padding[1]})/*.mode(torch::kReplicate)*/);
    auto compare_type = torch::kInt32;
    int precision_2 = 1<<precision_in;
    auto input_split = (input_pad*precision_2).clamp(0, precision_2-1).round().to(compare_type).clone();
    auto w_pos_split = (weight*precision_2).clamp(0, precision_2-1).round().to(compare_type);
    auto w_neg_split = (-(weight*precision_2).clamp(1-precision_2, 0)).round().to(compare_type);

    return conv2d_saturate_cuda(input_split, w_pos_split, w_neg_split, stride, add_config, precision_in, precision_out);
    // return conv2d_generic_cuda(input_split, w_pos_split, w_neg_split, stride, prog_load, bit_length, lfsr_length, z_units, bin_config, gen_config, xnor, mux);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_saturate_acc", &conv2d_saturate_acc, "General fixed point forward conv2d for saturating/truncating fixed-point network");
}
