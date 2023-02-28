#include "sc_cuda_conv.cuh"

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
    bool xnor) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    int z_packs = (c_ins + COMPUTE_CINS-1) / COMPUTE_CINS;
    int seed_mult = seed_mult_select(lfsr_length);

    int (*lfsr)(int) = lfsr_select(lfsr_length, xnor);

    for(int i=index_gen; i<c_outs*w_ins*h_ins*z_packs; i+=stride_gen) {

        int c_out = i/(w_ins*h_ins*z_packs);
        int w_in = (i%(w_ins*h_ins*z_packs)) / (h_ins*z_packs);
        int h_in = (i%(h_ins*z_packs)) / z_packs;
        int z_pack = i % z_packs;

        __shared__ int8_t pos_seed_shared [COMPUTE_CINS][THREADS_GENERAL];
        __shared__ int8_t neg_seed_shared [COMPUTE_CINS][THREADS_GENERAL];
        __shared__ int8_t weight_pos_shared [COMPUTE_CINS][THREADS_GENERAL];
        __shared__ int8_t weight_neg_shared [COMPUTE_CINS][THREADS_GENERAL];

        // Load seeds and weights
        for(int compute_cin=0; compute_cin<COMPUTE_CINS; compute_cin++) {
            int c_in = z_pack*COMPUTE_CINS + compute_cin;
            int seed_ind = c_in*w_ins*h_ins + w_in*h_ins + h_in;
            int weight_ind;
            if (channels_last_weight) weight_ind = weight_ind_ohwi(c_out, c_in, w_in, h_in, c_outs, c_ins, w_ins, h_ins);
            else weight_ind = weight_ind_oihw(c_out, c_in, w_in, h_in, c_outs, c_ins, w_ins, h_ins);
            if (c_in<c_ins) {
                if (gen_mult) {
                    pos_seed_shared[compute_cin][threadIdx.x] = seed_mult;
                    neg_seed_shared[compute_cin][threadIdx.x] = seed_mult;
                }
                else {
                    pos_seed_shared[compute_cin][threadIdx.x] = (POS_SEED + seed_ind)%((1<<lfsr_length)-1) + 1 - int(xnor);
                    neg_seed_shared[compute_cin][threadIdx.x] = (NEG_SEED + seed_ind)%((1<<lfsr_length)-1) + 1 - int(xnor);
                }
                weight_pos_shared[compute_cin][threadIdx.x] = weight_pos[weight_ind];
                weight_neg_shared[compute_cin][threadIdx.x] = weight_neg[weight_ind];
            }
            else {
                pos_seed_shared[compute_cin][threadIdx.x] = int(0);
                neg_seed_shared[compute_cin][threadIdx.x] = int(0);
                weight_pos_shared[compute_cin][threadIdx.x] = int(0);
                weight_neg_shared[compute_cin][threadIdx.x] = int(0);
            }
        }

        // Generation
        int cur_width = 0;
        for(int bit=0; bit<bit_length; bit++) {
            cur_width = (bit/load_wait + 1)*load_width - 1;
            if (cur_width > total_width) cur_width=total_width;
            int weight_pos_stream_c = 0;
            int weight_neg_stream_c = 0;
            for(int compute_cin=0; compute_cin<COMPUTE_CINS; compute_cin++) {
                int weight_pos_actual = (weight_pos_shared[compute_cin][threadIdx.x] >> (total_width-cur_width)) << (total_width-cur_width);
                int weight_neg_actual = (weight_neg_shared[compute_cin][threadIdx.x] >> (total_width-cur_width)) << (total_width-cur_width);
                int pos_seed_cur = pos_seed_shared[compute_cin][threadIdx.x];
                int neg_seed_cur = neg_seed_shared[compute_cin][threadIdx.x];
                pos_seed_cur = (*lfsr)(pos_seed_cur);
                neg_seed_cur = (*lfsr)(neg_seed_cur);

                weight_pos_stream_c += int(weight_pos_actual>pos_seed_cur) << compute_cin;
                weight_neg_stream_c += int(weight_neg_actual>neg_seed_cur) << compute_cin;
                pos_seed_shared[compute_cin][threadIdx.x] = pos_seed_cur;
                neg_seed_shared[compute_cin][threadIdx.x] = neg_seed_cur;
            }
            for(int gen_and=0; gen_and<gen_config; gen_and++) {
                weight_pos_stream[(bit*gen_config+gen_and)*c_outs*w_ins*h_ins*z_packs + i] = weight_pos_stream_c;
                weight_neg_stream[(bit*gen_config+gen_and)*c_outs*w_ins*h_ins*z_packs + i] = weight_neg_stream_c;
            }
        }
    }
}

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
    bool xnor) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    int z_packs = (c_ins + COMPUTE_CINS-1) / COMPUTE_CINS;

    int (*lfsr)(int) = lfsr_select(lfsr_length, xnor);
    
    for(int i=index_gen; i<batches*w_ins*h_ins*z_packs; i+=stride_gen) {
        int batch = i/(w_ins*h_ins*z_packs);
        int w_in = (i%(w_ins*h_ins*z_packs)) / (h_ins*z_packs);
        int h_in = (i%(h_ins*z_packs)) / z_packs;
        int z_pack = i % z_packs;

        __shared__ int8_t seed_shared [COMPUTE_CINS][THREADS_GENERAL];
        __shared__ int8_t input_shared [COMPUTE_CINS][THREADS_GENERAL];

        // Load seeds and inputs
        for(int compute_cin=0; compute_cin<COMPUTE_CINS; compute_cin++) {
            int c_in = z_pack*COMPUTE_CINS + compute_cin;
            int seed_ind = c_in*w_ins*h_ins + w_in*h_ins + h_in;
            int input_ind;
            if (channels_last_activation) input_ind = weight_ind_ohwi(batch, c_in, w_in, h_in, batches, c_ins, w_ins, h_ins);
            else input_ind = weight_ind_oihw(batch, c_in, w_in, h_in, batches, c_ins, w_ins, h_ins);
            if (c_in<c_ins) {
                if (gen_mult) seed_shared[compute_cin][threadIdx.x] = 1;
                else seed_shared[compute_cin][threadIdx.x] = (0 + seed_ind)%((1<<lfsr_length)-1) + 1 - int(xnor);
                input_shared[compute_cin][threadIdx.x] = input_bin[input_ind];
            }
            else {
                seed_shared[compute_cin][threadIdx.x] = int(0);
                input_shared[compute_cin][threadIdx.x] = int(0);
            }
        }

        // Generation
        int cur_width = 0;
        for(int bit=0; bit<bit_length; bit++) {
            cur_width = (bit/load_wait + 1)*load_width - 1;
            if (cur_width > total_width) cur_width = total_width;
            int input_stream_c = 0;
            for(int compute_cin=0; compute_cin<COMPUTE_CINS; compute_cin++) {
                int input_actual = (input_shared[compute_cin][threadIdx.x] >> (total_width-cur_width)) << (total_width-cur_width);
                int seed_cur = seed_shared[compute_cin][threadIdx.x];
                seed_cur = (*lfsr)(seed_cur);
                input_stream_c += int(input_actual > seed_cur) <<compute_cin;
                seed_shared[compute_cin][threadIdx.x] = seed_cur;
            }
            for(int gen_and=0; gen_and<gen_config; gen_and++) {
                input_stream[((bit/gen_config)*gen_config*gen_config + gen_and*gen_config + bit%gen_config)*batches*w_ins*h_ins*z_packs + i] = input_stream_c;
            }
        }
    }
}