#include "sc_device.cuh"

// Stream generation for weights
__global__
void stream_generation_or_general(
    const int32_t* __restrict__ weight_pos,
    const int32_t* __restrict__ weight_neg,
    int32_t* __restrict__ weight_pos_stream,
    int32_t* __restrict__ weight_neg_stream,
    int bit_length,
    int lfsr_length,
    int gen_config,
    bool gen_mult, //Use lfsr_mult
    int c_outs,
    int c_ins,
    int w_ins,
    int h_ins,
    int total_width,
    int load_width,
    int load_wait,
    bool channels_last_weight,
    bool xnor);

// Stream generation for activations
__global__
void activation_generation_or_general(
    const int32_t* __restrict__ input_bin,
    int32_t* __restrict__ input_stream,
    int bit_length,
    int lfsr_length,
    int gen_config,
    bool gen_mult,
    int batches,
    int c_ins,
    int w_ins,
    int h_ins,
    const int total_width,
    const int load_width,
    const int load_wait,
    bool channels_last_activation,
    bool xnor);