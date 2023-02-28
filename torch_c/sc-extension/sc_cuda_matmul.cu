#include "sc_device.cuh"

using namespace nvcuda;

// Stream generation for weights
__global__
void stream_generation_linear_general(
    const int32_t* __restrict__ weight_pos,
    const int32_t* __restrict__ weight_neg,
    int32_t* __restrict__ weight_pos_stream,
    int32_t* __restrict__ weight_neg_stream,
    int bit_length,
    int lfsr_length,
    int gen_config,
    bool gen_mult,
    int c_outs,
    int c_ins,
    int total_width,
    int load_width,
    int load_wait) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    int z_packs = (c_ins + COMPUTE_CINS-1) / COMPUTE_CINS;
    int seed_mult = 1;

    int (*lfsr)(int);
    switch(lfsr_length) {
    case 1:
        lfsr=&d_lfsr_1;
        break;
    case 2:
        lfsr=&d_lfsr_2;
        break;
    case 3:
        lfsr=&d_lfsr_3;
        break;
    case 4:
        lfsr=&d_lfsr_4;
        seed_mult = SEED_4;
        break;
    case 5:
        lfsr=&d_lfsr_5;
        seed_mult = SEED_5;
        break;
    case 6:
        lfsr=&d_lfsr_6;
        seed_mult = SEED_6;
        break;
    case 7:
        lfsr=&d_lfsr_7;
        seed_mult = SEED_7;
        break;
    case 8:
        lfsr=&d_lfsr_8;
        break;
    }

    for(int i=index_gen; i<c_outs*z_packs; i+=stride_gen) {
        int c_out = i/z_packs;
        int z_pack = i%z_packs;

        __shared__ int8_t pos_seed_shared [COMPUTE_CINS][THREADS_GENERAL];
        __shared__ int8_t neg_seed_shared [COMPUTE_CINS][THREADS_GENERAL];
        __shared__ int8_t weight_pos_shared [COMPUTE_CINS][THREADS_GENERAL];
        __shared__ int8_t weight_neg_shared [COMPUTE_CINS][THREADS_GENERAL];

        for(int compute_cin=0; compute_cin<COMPUTE_CINS; compute_cin++) {
            int c_in = z_pack*COMPUTE_CINS + compute_cin;
            int weight_ind = c_out*c_ins + c_in;
            if (c_in<c_ins) {
                if (gen_mult) {
                    pos_seed_shared[compute_cin][threadIdx.x] = seed_mult;
                    neg_seed_shared[compute_cin][threadIdx.x] = seed_mult;
                }
                else {
                    pos_seed_shared[compute_cin][threadIdx.x] = (POS_SEED + c_in)%((1<<lfsr_length)-1) + 1;
                    neg_seed_shared[compute_cin][threadIdx.x] = (NEG_SEED + c_in)%((1<<lfsr_length)-1) + 1;
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
                weight_pos_stream[(bit*gen_config+gen_and)*c_outs*z_packs + i] = weight_pos_stream_c;
                weight_neg_stream[(bit*gen_config+gen_and)*c_outs*z_packs + i] = weight_neg_stream_c;
            }
        }
    }
}

// Stream generation for activations
__global__
void activation_generation_linear_general(
    const int32_t* __restrict__ input_bin,
    int32_t* __restrict__ input_stream,
    int bit_length,
    int lfsr_length,
    int gen_config,
    bool gen_mult,
    int batches,
    int c_ins,
    const int total_width,
    const int load_width,
    const int load_wait) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    int z_packs = (c_ins + COMPUTE_CINS-1) / COMPUTE_CINS;

    int (*lfsr)(int);
    switch(lfsr_length) {
    case 1:
        lfsr=&d_lfsr_1;
        break;
    case 2:
        lfsr=&d_lfsr_2;
        break;
    case 3:
        lfsr=&d_lfsr_3;
        break;
    case 4:
        lfsr=&d_lfsr_4;
        break;
    case 5:
        lfsr=&d_lfsr_5;
        break;
    case 6:
        lfsr=&d_lfsr_6;
        break;
    case 7:
        lfsr=&d_lfsr_7;
        break;
    case 8:
        lfsr=&d_lfsr_8;
        break;
    }

    for(int i=index_gen; i<batches*z_packs; i+=stride_gen) {
        int batch = i/z_packs;
        int z_pack = i%z_packs;

        __shared__ int8_t seed_shared [COMPUTE_CINS][THREADS_GENERAL+1];
        __shared__ int8_t input_shared [COMPUTE_CINS][THREADS_GENERAL+1];

        for(int compute_cin=0; compute_cin<COMPUTE_CINS; compute_cin++) {
            int c_in = z_pack*COMPUTE_CINS + compute_cin;
            int input_ind = batch*c_ins + c_in;
            if (c_in<c_ins) {
                if (gen_mult) seed_shared[compute_cin][threadIdx.x] = 1;
                else seed_shared[compute_cin][threadIdx.x] = (0 + c_in)%((1<<lfsr_length)-1) + 1;
                
                input_shared[compute_cin][threadIdx.x] = input_bin[input_ind];
            }
            else {
                seed_shared[compute_cin][threadIdx.x] = int(0);
                input_shared[compute_cin][threadIdx.x] = int(0);
            }
        }

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
                input_stream[((bit/gen_config)*gen_config*gen_config + gen_and*gen_config + bit%gen_config)*batches*z_packs + i] = input_stream_c;
            }
        }
    }
}

// Old version of OR-n linear kernel
__global__
void stream_compute_linear_general(
    const int32_t* __restrict__ input_stream,
    const int32_t* __restrict__ weight_pos_stream,
    const int32_t* __restrict__ weight_neg_stream,
    int32_t* __restrict__ output_pos_stream,
    int32_t* __restrict__ output_neg_stream,
    int bit_length,
    int bin_config,
    int batches,
    int c_ins,
    int c_outs) {
    int index_gen = threadIdx.x;
    int stride_gen = blockDim.x;
    int index_block = blockIdx.x;
    int stride_block = gridDim.x;

    bool or_no=false;
    if (bin_config>=0) bin_config+=1;
    else {
        bin_config = -bin_config+1;
        or_no = true;
    }

    int z_packs = (c_ins + COMPUTE_CINS-1) / COMPUTE_CINS;
    int m_blocks = (batches + 2*M_GENERAL-1) / (2*M_GENERAL);
    int n_blocks = (c_outs + 2*N_GENERAL-1) / (2*N_GENERAL);

    int index_i_c = index_gen / N_GENERAL;
    int index_w_c = index_gen % N_GENERAL;

    #if __CUDA_ARCH__ >= 800
    wmma::fragment<wmma::matrix_a, WMMA_M_bin, WMMA_N_bin, WMMA_K_bin, wmma::experimental::precision::b1, wmma::row_major> a_frag_0;
    wmma::fragment<wmma::matrix_a, WMMA_M_bin, WMMA_N_bin, WMMA_K_bin, wmma::experimental::precision::b1, wmma::row_major> a_frag_1;
    wmma::fragment<wmma::matrix_b, WMMA_M_bin, WMMA_N_bin, WMMA_K_bin, wmma::experimental::precision::b1, wmma::col_major> b_frag_pos;
    wmma::fragment<wmma::matrix_b, WMMA_M_bin, WMMA_N_bin, WMMA_K_bin, wmma::experimental::precision::b1, wmma::col_major> b_frag_neg;
    wmma::fragment<wmma::accumulator, WMMA_M_bin, WMMA_N_bin, WMMA_K_bin, int> acc_frag_pos_0;
    wmma::fragment<wmma::accumulator, WMMA_M_bin, WMMA_N_bin, WMMA_K_bin, int> acc_frag_pos_1;
    wmma::fragment<wmma::accumulator, WMMA_M_bin, WMMA_N_bin, WMMA_K_bin, int> acc_frag_neg_0;
    wmma::fragment<wmma::accumulator, WMMA_M_bin, WMMA_N_bin, WMMA_K_bin, int> acc_frag_neg_1;
    int index_warp = index_gen / WMMA_INT_WIDTH;
    int warp_m = index_warp / N_GENERAL_WMMA;
    int warp_n = index_warp % N_GENERAL_WMMA;
    __shared__ int output_s_pos [2*M_GENERAL*2*N_GENERAL];
    __shared__ int output_s_neg [2*M_GENERAL*2*N_GENERAL];
    #endif

    for (int block=index_block; block<m_blocks*n_blocks; block+=stride_block) {
        int m_block = (block/n_blocks)*2*M_GENERAL;
        int n_block = (block%n_blocks)*2*N_GENERAL;
        const int32_t* input_mblock = input_stream + m_block*z_packs;
        const int32_t* weight_pos_nblock = weight_pos_stream + n_block*z_packs;
        const int32_t* weight_neg_nblock = weight_neg_stream + n_block*z_packs;
        int32_t* output_pos_batch = output_pos_stream + m_block*c_outs + n_block;
        int32_t* output_neg_batch = output_neg_stream + m_block*c_outs + n_block;

        for (int bit=0; bit<bit_length; bit++) {
            const int32_t* input_bit = input_mblock + bit*batches*z_packs;
            const int32_t* weight_pos_bit = weight_pos_nblock + bit*c_outs*z_packs;
            const int32_t* weight_neg_bit = weight_neg_nblock + bit*c_outs*z_packs;
            __shared__ uint weight_pos_s [2*N_GENERAL*K_GENERAL_STORE];
            __shared__ uint weight_neg_s [2*N_GENERAL*K_GENERAL_STORE];
            __shared__ uint input_s [2*M_GENERAL*K_GENERAL_STORE];
            #if __CUDA_ARCH__ >= 800
            wmma::fill_fragment(acc_frag_pos_0, int(0));
            wmma::fill_fragment(acc_frag_pos_1, int(0));
            wmma::fill_fragment(acc_frag_neg_0, int(0));
            wmma::fill_fragment(acc_frag_neg_1, int(0));
            #else
            uint output_pos_c_00 = 0;
            uint output_pos_c_01 = 0;
            uint output_pos_c_10 = 0;
            uint output_pos_c_11 = 0;
            uint output_neg_c_00 = 0;
            uint output_neg_c_01 = 0;
            uint output_neg_c_10 = 0;
            uint output_neg_c_11 = 0;
            #endif
            for (int inner=0; inner<z_packs; inner+=K_GENERAL) {
                // Load weights
                for (int index_w=index_gen; index_w<2*N_GENERAL*K_GENERAL; index_w+=stride_gen) {
                    int n = index_w / K_GENERAL;
                    int k = index_w % K_GENERAL;
                    int weight_index_s = n*K_GENERAL_STORE+k;
                    if ((inner+k<z_packs) && (n_block+n<c_outs)) {
                        int weight_index = n*z_packs + inner+k;
                        weight_pos_s[weight_index_s] = weight_pos_bit[weight_index];
                        weight_neg_s[weight_index_s] = weight_neg_bit[weight_index];
                    }
                    else {
                        weight_pos_s[weight_index_s] = 0;
                        weight_neg_s[weight_index_s] = 0;
                    }
                }
                // Load inputs
                for (int index_i=index_gen; index_i<2*M_GENERAL*K_GENERAL; index_i+=stride_gen) {
                    int m = index_i / K_GENERAL;
                    int k = index_i % K_GENERAL;
                    int input_index_s = m*K_GENERAL_STORE+k;
                    if ((inner+k<z_packs) && (m_block+m<batches)) {
                        int input_index = m*z_packs + inner+k;
                        input_s[input_index_s] = input_bit[input_index];
                    }
                    else input_s[input_index_s] = 0;
                }
                __syncthreads();
                // Compute
                #if __CUDA_ARCH__ >= 800
                for (int i=0; i<K_GENERAL*WMMA_INT_WIDTH/WMMA_K_bin; i++) {
                    wmma::load_matrix_sync(a_frag_0, input_s+(warp_m*2+0)*WMMA_M_bin*K_GENERAL_STORE+i*WMMA_K_bin/WMMA_INT_WIDTH, K_GENERAL_STORE*WMMA_INT_WIDTH);
                    wmma::load_matrix_sync(a_frag_1, input_s+(warp_m*2+1)*WMMA_M_bin*K_GENERAL_STORE+i*WMMA_K_bin/WMMA_INT_WIDTH, K_GENERAL_STORE*WMMA_INT_WIDTH);
                    wmma::load_matrix_sync(b_frag_pos, weight_pos_s+warp_n*WMMA_N_bin*K_GENERAL_STORE+i*WMMA_K_bin/WMMA_INT_WIDTH, K_GENERAL_STORE*WMMA_INT_WIDTH);
                    wmma::load_matrix_sync(b_frag_neg, weight_neg_s+warp_n*WMMA_N_bin*K_GENERAL_STORE+i*WMMA_K_bin/WMMA_INT_WIDTH, K_GENERAL_STORE*WMMA_INT_WIDTH);
                    wmma::bmma_sync(acc_frag_pos_0, a_frag_0, b_frag_pos, acc_frag_pos_0, wmma::experimental::bmmaBitOpAND, wmma::experimental::bmmaAccumulateOpPOPC);
                    wmma::bmma_sync(acc_frag_pos_1, a_frag_1, b_frag_pos, acc_frag_pos_1, wmma::experimental::bmmaBitOpAND, wmma::experimental::bmmaAccumulateOpPOPC);
                    wmma::bmma_sync(acc_frag_neg_0, a_frag_0, b_frag_neg, acc_frag_neg_0, wmma::experimental::bmmaBitOpAND, wmma::experimental::bmmaAccumulateOpPOPC);
                    wmma::bmma_sync(acc_frag_neg_1, a_frag_1, b_frag_neg, acc_frag_neg_1, wmma::experimental::bmmaBitOpAND, wmma::experimental::bmmaAccumulateOpPOPC);
                }
                #else
                for (int i=0; i<K_GENERAL; i++) {
                    int32_t input_s_0 = input_s[(index_i_c*2+0)*K_GENERAL_STORE+i];
                    int32_t input_s_1 = input_s[(index_i_c*2+1)*K_GENERAL_STORE+i];
                    int32_t weight_pos_s_0 = weight_pos_s[(index_w_c*2+0)*K_GENERAL_STORE+i];
                    int32_t weight_pos_s_1 = weight_pos_s[(index_w_c*2+1)*K_GENERAL_STORE+i];
                    int32_t weight_neg_s_0 = weight_neg_s[(index_w_c*2+0)*K_GENERAL_STORE+i];
                    int32_t weight_neg_s_1 = weight_neg_s[(index_w_c*2+1)*K_GENERAL_STORE+i];

                    
                    output_pos_c_00 += __popc(input_s_0 & weight_pos_s_0);
                    output_pos_c_01 += __popc(input_s_0 & weight_pos_s_1);
                    output_pos_c_10 += __popc(input_s_1 & weight_pos_s_0);
                    output_pos_c_11 += __popc(input_s_1 & weight_pos_s_1);

                    output_neg_c_00 += __popc(input_s_0 & weight_neg_s_0);
                    output_neg_c_01 += __popc(input_s_0 & weight_neg_s_1);
                    output_neg_c_10 += __popc(input_s_1 & weight_neg_s_0);
                    output_neg_c_11 += __popc(input_s_1 & weight_neg_s_1);
                }
                #endif
                __syncthreads();
            }
            int batch_0 = m_block+index_i_c*2+0;
            int batch_1 = m_block+index_i_c*2+1;
            int cout_0 = n_block+index_w_c*2+0;
            int cout_1 = n_block+index_w_c*2+1;
            #if __CUDA_ARCH__ >= 800
            wmma::store_matrix_sync(output_s_pos + (warp_m*2+0)*WMMA_M_bin*2*N_GENERAL + warp_n*WMMA_N_bin, acc_frag_pos_0, 2*N_GENERAL, wmma::mem_row_major);
            wmma::store_matrix_sync(output_s_pos + (warp_m*2+1)*WMMA_M_bin*2*N_GENERAL + warp_n*WMMA_N_bin, acc_frag_pos_1, 2*N_GENERAL, wmma::mem_row_major);
            wmma::store_matrix_sync(output_s_neg + (warp_m*2+0)*WMMA_M_bin*2*N_GENERAL + warp_n*WMMA_N_bin, acc_frag_neg_0, 2*N_GENERAL, wmma::mem_row_major);
            wmma::store_matrix_sync(output_s_neg + (warp_m*2+1)*WMMA_M_bin*2*N_GENERAL + warp_n*WMMA_N_bin, acc_frag_neg_1, 2*N_GENERAL, wmma::mem_row_major);
            __syncthreads();
            if (or_no) {
                if ((batch_0<batches) && (cout_0<c_outs)) {
                    int index_o = (index_i_c*2+0)*c_outs + (index_w_c*2+0);
                    output_pos_batch[index_o] += int(output_s_pos[(index_i_c*2+0)*(2*N_GENERAL)+(index_w_c*2+0)]>=bin_config);
                    output_neg_batch[index_o] += int(output_s_neg[(index_i_c*2+0)*(2*N_GENERAL)+(index_w_c*2+0)]>=bin_config);
                }
                if ((batch_0<batches) && (cout_1<c_outs)) {
                    int index_o = (index_i_c*2+0)*c_outs + (index_w_c*2+1);
                    output_pos_batch[index_o] += int(output_s_pos[(index_i_c*2+0)*(2*N_GENERAL)+(index_w_c*2+1)]>=bin_config);
                    output_neg_batch[index_o] += int(output_s_neg[(index_i_c*2+0)*(2*N_GENERAL)+(index_w_c*2+1)]>=bin_config);
                }
                if ((batch_1<batches) && (cout_0<c_outs)) {
                    int index_o = (index_i_c*2+1)*c_outs + (index_w_c*2+0);
                    output_pos_batch[index_o] += int(output_s_pos[(index_i_c*2+1)*(2*N_GENERAL)+(index_w_c*2+0)]>=bin_config);
                    output_neg_batch[index_o] += int(output_s_neg[(index_i_c*2+1)*(2*N_GENERAL)+(index_w_c*2+0)]>=bin_config);
                }
                if ((batch_1<batches) && (cout_1<c_outs)) {
                    int index_o = (index_i_c*2+1)*c_outs + (index_w_c*2+1);
                    output_pos_batch[index_o] += int(output_s_pos[(index_i_c*2+1)*(2*N_GENERAL)+(index_w_c*2+1)]>=bin_config);
                    output_neg_batch[index_o] += int(output_s_neg[(index_i_c*2+1)*(2*N_GENERAL)+(index_w_c*2+1)]>=bin_config);
                }
            }
            else {
                if ((batch_0<batches) && (cout_0<c_outs)) {
                    int index_o = (index_i_c*2+0)*c_outs + (index_w_c*2+0);
                    output_pos_batch[index_o] += min(output_s_pos[(index_i_c*2+0)*(2*N_GENERAL)+(index_w_c*2+0)], bin_config);
                    output_neg_batch[index_o] += min(output_s_neg[(index_i_c*2+0)*(2*N_GENERAL)+(index_w_c*2+0)], bin_config);
                }
                if ((batch_0<batches) && (cout_1<c_outs)) {
                    int index_o = (index_i_c*2+0)*c_outs + (index_w_c*2+1);
                    output_pos_batch[index_o] += min(output_s_pos[(index_i_c*2+0)*(2*N_GENERAL)+(index_w_c*2+1)], bin_config);
                    output_neg_batch[index_o] += min(output_s_neg[(index_i_c*2+0)*(2*N_GENERAL)+(index_w_c*2+1)], bin_config);
                }
                if ((batch_1<batches) && (cout_0<c_outs)) {
                    int index_o = (index_i_c*2+1)*c_outs + (index_w_c*2+0);
                    output_pos_batch[index_o] += min(output_s_pos[(index_i_c*2+1)*(2*N_GENERAL)+(index_w_c*2+0)], bin_config);
                    output_neg_batch[index_o] += min(output_s_neg[(index_i_c*2+1)*(2*N_GENERAL)+(index_w_c*2+0)], bin_config);
                }
                if ((batch_1<batches) && (cout_1<c_outs)) {
                    int index_o = (index_i_c*2+1)*c_outs + (index_w_c*2+1);
                    output_pos_batch[index_o] += min(output_s_pos[(index_i_c*2+1)*(2*N_GENERAL)+(index_w_c*2+1)], bin_config);
                    output_neg_batch[index_o] += min(output_s_neg[(index_i_c*2+1)*(2*N_GENERAL)+(index_w_c*2+1)], bin_config);
                }
            }
            #else
            if (or_no) {
                if ((batch_0<batches) && (cout_0<c_outs)) {
                    int index_o = (index_i_c*2+0)*c_outs + (index_w_c*2+0);
                    output_pos_batch[index_o] += int(output_pos_c_00>=bin_config);
                    output_neg_batch[index_o] += int(output_neg_c_00>=bin_config);
                }
                if ((batch_0<batches) && (cout_1<c_outs)) {
                    int index_o = (index_i_c*2+0)*c_outs + (index_w_c*2+1);
                    output_pos_batch[index_o] += int(output_pos_c_01>=bin_config);
                    output_neg_batch[index_o] += int(output_neg_c_01>=bin_config);
                }
                if ((batch_1<batches) && (cout_0<c_outs)) {
                    int index_o = (index_i_c*2+1)*c_outs + (index_w_c*2+0);
                    output_pos_batch[index_o] += int(output_pos_c_10>=bin_config);
                    output_neg_batch[index_o] += int(output_neg_c_10>=bin_config);
                }
                if ((batch_1<batches) && (cout_1<c_outs)) {
                    int index_o = (index_i_c*2+1)*c_outs + (index_w_c*2+1);
                    output_pos_batch[index_o] += int(output_pos_c_11>=bin_config);
                    output_neg_batch[index_o] += int(output_neg_c_11>=bin_config);
                }
            }
            else {
                if ((batch_0<batches) && (cout_0<c_outs)) {
                    int index_o = (index_i_c*2+0)*c_outs + (index_w_c*2+0);
                    output_pos_batch[index_o] += min(output_pos_c_00, bin_config);
                    output_neg_batch[index_o] += min(output_neg_c_00, bin_config);
                }
                if ((batch_0<batches) && (cout_1<c_outs)) {
                    int index_o = (index_i_c*2+0)*c_outs + (index_w_c*2+1);
                    output_pos_batch[index_o] += min(output_pos_c_01, bin_config);
                    output_neg_batch[index_o] += min(output_neg_c_01, bin_config);
                }
                if ((batch_1<batches) && (cout_0<c_outs)) {
                    int index_o = (index_i_c*2+1)*c_outs + (index_w_c*2+0);
                    output_pos_batch[index_o] += min(output_pos_c_10, bin_config);
                    output_neg_batch[index_o] += min(output_neg_c_10, bin_config);
                }
                if ((batch_1<batches) && (cout_1<c_outs)) {
                    int index_o = (index_i_c*2+1)*c_outs + (index_w_c*2+1);
                    output_pos_batch[index_o] += min(output_pos_c_11, bin_config);
                    output_neg_batch[index_o] += min(output_neg_c_11, bin_config);
                }
            }
            #endif
            __syncthreads();
        }
    }
}

// Cut input and weight sections for gemm computation
__global__ void
prepare_data_gemm_linear(
    const int32_t* __restrict__ input_stream,
    const int32_t* __restrict__ weight_pos_stream,
    const int32_t* __restrict__ weight_neg_stream,
    int32_t* __restrict__ input_temp,
    int32_t* __restrict__ weight_pos_temp,
    int32_t* __restrict__ weight_neg_temp,
    const int m_offset,
    const int n_offset,
    const int k_offset,
    const int m_total,
    const int n_total,
    const int k_total,
    const int k_store,
    const int m_input,
    const int n_input,
    const int z_packs,
    const int bit_length
) {
    int index_gen = threadIdx.x;
    int stride_gen = blockDim.x;
    int index_block = blockIdx.x;
    int stride_block = gridDim.x;

    int index_warp = index_gen/32;
    int num_warp = stride_gen/32;
    int id_warp = index_gen%32;
    int k_unit = (z_packs+7)/8;
    int k_unit_total = k_unit*8;
    int k_unit_compute = (K_UNIT_BLOCK/k_unit_total)*k_unit_total;

    // Prepare inputs
    for (int m_c=index_block; m_c<m_total; m_c+=stride_block) {
        int id_new = (id_warp + 4*(m_c%8))%32;
        int32_t* input_temp_m = input_temp + m_c*k_store;
        int m_c_o = m_c+m_offset;
        if (m_c_o<m_input) {
            const int32_t* input_stream_m = input_stream+m_c_o*z_packs;
            for (int k_c=0; k_c<k_store; k_c+=stride_gen) {
                int k_c_s = k_c+32*index_warp;
                int k_c_o = k_c_s+k_offset+id_warp;
                int bit = k_c_o/k_unit_total;
                int inner_unit = k_c_o%k_unit_total;
                int input_c = 0;
                if ((bit<bit_length) & (inner_unit<z_packs) & (k_c_s+id_warp<k_total)) {
                    input_c = input_stream_m[bit*m_input*z_packs+inner_unit];
                }
                if ((k_c_s+id_new)<k_store) input_temp_m[k_c_s+id_new] = input_c;
            }
        }
    }

    // Prepare weights
    for (int n_c=index_block; n_c<n_total; n_c+=stride_block) {
        int id_new = (id_warp + 4*(n_c%8))%32;
        int32_t* weight_pos_temp_n = weight_pos_temp + n_c*k_store;
        int32_t* weight_neg_temp_n = weight_neg_temp + n_c*k_store;
        int n_c_o = n_c+n_offset;
        if (n_c_o<n_input) {
            const int32_t* weight_pos_stream_n = weight_pos_stream + n_c_o*z_packs;
            const int32_t* weight_neg_stream_n = weight_neg_stream + n_c_o*z_packs;
            for (int k_c=0; k_c<k_store; k_c+=stride_gen) {
                int k_c_s = k_c + 32*index_warp;
                int k_c_o = k_c_s + k_offset + id_warp;
                int bit = k_c_o / k_unit_total;
                int inner_unit = k_c_o%k_unit_total;
                int weight_pos_c = 0;
                int weight_neg_c = 0;
                if ((bit<bit_length) & (inner_unit<z_packs) & (k_c_s+id_warp<k_total)) {
                    weight_pos_c = weight_pos_stream_n[bit*n_input*z_packs+inner_unit];
                    weight_neg_c = weight_neg_stream_n[bit*n_input*z_packs+inner_unit];
                }
                if ((k_c_s+id_new)<k_store) {
                    weight_pos_temp_n[k_c_s+id_new] = weight_pos_c;
                    weight_neg_temp_n[k_c_s+id_new] = weight_neg_c;
                }
            }
        }
    }
}

// Save computation results to output tensor
__global__ void
save_data_gemm_linear(
    const int16_t* __restrict__ output_pos_temp,
    const int16_t* __restrict__ output_neg_temp,
    int16_t* __restrict__ output_pos,
    int16_t* __restrict__ output_neg,
    const int m_offset,
    const int n_offset,
    const int m_total,
    const int n_total,
    const int m_input,
    const int n_input
) {
    int index_gen = threadIdx.x;
    int stride_gen = blockDim.x;
    int index_block = blockIdx.x;
    int stride_block = gridDim.x;

    int index_warp = index_gen/32;
    int num_warp = stride_gen/32;
    int id_warp = index_gen%32;

    for (int n_c=index_block; n_c<n_total; n_c+=stride_block) {
        const int16_t* output_pos_temp_n = output_pos_temp + n_c*m_total;
        const int16_t* output_neg_temp_n = output_neg_temp + n_c*m_total;
        int n_c_o = n_c+n_offset;
        if (n_c_o<n_input) {
            int16_t *output_pos_n = output_pos+n_c_o;
            int16_t *output_neg_n = output_neg+n_c_o;

            for (int m_c=index_gen; m_c<m_total; m_c+=stride_gen) {
                const int16_t* output_pos_temp_m = output_pos_temp_n + m_c;
                const int16_t* output_neg_temp_m = output_neg_temp_n + m_c;
                int m_c_o = m_c+m_offset;
                if (m_c_o<m_input) {
                    int16_t *output_pos_m = output_pos_n + m_c_o*n_input;
                    int16_t *output_neg_m = output_neg_n + m_c_o*n_input;
                    *output_pos_m = *output_pos_temp_m;
                    *output_neg_m = *output_neg_temp_m;
                }
            }
        }
    }
}

// Gemm version of OR-n linear
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
    bool mux) {
    auto weight_size = weight_pos.sizes();
    auto input_size = input.sizes();
    auto device = weight_pos.device();
    int device_index = device.index();
    cudaSetDevice(int(device_index));
    int z_packs = (weight_size[1] + COMPUTE_CINS-1) / COMPUTE_CINS;

    int compute_length = bit_length*gen_config;

    if(lfsr_length<0) {
        switch(bit_length) {
            case 2:
            lfsr_length = 1;
            case 4:
            lfsr_length = 2;
            case 8:
            lfsr_length = 3;
            break;
            case 16:
            lfsr_length = 4;
            break;
            case 32:
            lfsr_length = 5;
            break;
            case 64:
            lfsr_length = 6;
            break;
            case 128:
            lfsr_length = 7;
            break;
            case 256:
            lfsr_length = 8;
            break;
        }
    }

    const int threads = THREADS_GENERAL;
    bool gen_mult = false;

    if (!Init_Temp) {
        cudaMalloc(&Input_Stream, 1073741824UL);
        cudaMalloc(&Weight_Pos_Stream, 2048*2048*4*sizeof(int32_t));
        cudaMalloc(&Weight_Neg_Stream, 2048*2048*4*sizeof(int32_t));
        cudaMalloc(&Weight_Pos_Temp, N_UNIT_BLOCK*2*W_UNROLL_TEST*N_GENERAL_TEST*K_UNIT_BLOCK*sizeof(int32_t));
        cudaMalloc(&Weight_Neg_Temp, N_UNIT_BLOCK*2*W_UNROLL_TEST*N_GENERAL_TEST*K_UNIT_BLOCK*sizeof(int32_t));
        cudaMalloc(&Input_Temp, M_UNIT_BLOCK*2*I_UNROLL_TEST*M_GENERAL_TEST*K_UNIT_BLOCK*sizeof(int32_t));
        cudaMalloc(&Output_Pos_Temp, N_UNIT_BLOCK*2*W_UNROLL_TEST*N_GENERAL_TEST*M_UNIT_BLOCK*2*I_UNROLL_TEST*M_GENERAL_TEST*sizeof(int16_t));
        cudaMalloc(&Output_Neg_Temp, N_UNIT_BLOCK*2*W_UNROLL_TEST*N_GENERAL_TEST*M_UNIT_BLOCK*2*I_UNROLL_TEST*M_GENERAL_TEST*sizeof(int16_t));
        Init_Temp = true;
    }
    stream_generation_linear_general <<<10000, threads>>>(
        weight_pos.data_ptr<int32_t>(),
        weight_neg.data_ptr<int32_t>(),
        Weight_Pos_Stream,
        Weight_Neg_Stream,
        bit_length,
        lfsr_length,
        gen_config,
        gen_mult,
        weight_size[0],
        weight_size[1],
        prog_load[0],
        prog_load[1],
        prog_load[2]);
    activation_generation_linear_general <<<10000, threads>>>(
        input.data_ptr<int32_t>(),
        Input_Stream,
        bit_length,
        lfsr_length,
        gen_config,
        gen_mult,
        input_size[0],
        input_size[1],
        prog_load[0],
        prog_load[1],
        prog_load[2]);

    auto output_tensor_pos = torch::zeros({input_size[0], weight_size[0]}, at::TensorOptions().dtype(torch::kInt16).device(device));
    auto output_tensor_neg = torch::zeros({input_size[0], weight_size[0]}, at::TensorOptions().dtype(torch::kInt16).device(device));

    // Calculate gemm mapping parameters
    // Limitation: the size of the dot product cannot be larger than 512x32=16384
    int m_block_compute = (input_size[0]+2*I_UNROLL_TEST*M_GENERAL_TEST-1)/(2*I_UNROLL_TEST*M_GENERAL_TEST);
    int n_block_compute = (weight_size[0]+2*W_UNROLL_TEST*N_GENERAL_TEST-1)/(2*W_UNROLL_TEST*N_GENERAL_TEST);
    int m_block_compute_sw = ((m_block_compute+M_SW-1)/M_SW)*M_SW;
    int n_block_compute_sw = ((n_block_compute+N_SW-1)/N_SW)*N_SW;
    int m_total_compute = m_block_compute*(2*I_UNROLL_TEST*M_GENERAL_TEST);
    int n_total_compute = n_block_compute*(2*W_UNROLL_TEST*N_GENERAL_TEST);

    int k_unit = (z_packs+7)/8;
    int k_unit_total = k_unit*8;
    int k_unit_compute = (K_UNIT_BLOCK/k_unit_total)*k_unit_total;
    int k_unit_store = ((k_unit_compute+31)/32)*32;
    int k_total = compute_length*k_unit_total;

    int m_compute = std::min(m_total_compute, M_UNIT_BLOCK*2*I_UNROLL_TEST*M_GENERAL_TEST);
    int n_compute = std::min(n_total_compute, N_UNIT_BLOCK*2*W_UNROLL_TEST*N_GENERAL_TEST);
    int k_compute = std::min(k_unit_compute, k_total);

    for (int m_c=0; m_c<int(input_size[0]); m_c+=m_compute) {
        for (int n_c=0; n_c<int(weight_size[0]); n_c+=n_compute) {
            for (int k_c=0; k_c<k_total; k_c+=k_compute) {
                prepare_data_gemm_linear <<<10000, 128>>> (
                    Input_Stream,
                    Weight_Pos_Stream,
                    Weight_Neg_Stream,
                    Input_Temp,
                    Weight_Pos_Temp,
                    Weight_Neg_Temp,
                    m_c,
                    n_c,
                    k_c,
                    m_compute,
                    n_compute,
                    k_compute,
                    k_unit_store,
                    int(input_size[0]),
                    int(weight_size[0]),
                    z_packs,
                    bit_length
                );
                stream_compute_matmul_orn <<<m_block_compute_sw*n_block_compute_sw, M_GENERAL_TEST*N_GENERAL_TEST>>> (
                    Input_Temp,
                    Weight_Pos_Temp,
                    Weight_Neg_Temp,
                    Output_Pos_Temp,
                    Output_Neg_Temp,
                    k_unit_store,
                    std::min(k_unit_store, ((k_total-k_c+31)/32)*32),
                    m_compute/(2*I_UNROLL_TEST*M_GENERAL_TEST),
                    n_compute/(2*W_UNROLL_TEST*N_GENERAL_TEST),
                    k_unit_total,
                    bin_config+1,
                    (k_c==0)
                );
            }
            save_data_gemm_linear <<<10000, 128>>> (
                Output_Pos_Temp,
                Output_Neg_Temp,
                output_tensor_pos.data_ptr<int16_t>(),
                output_tensor_neg.data_ptr<int16_t>(),
                m_c,
                n_c,
                m_compute,
                n_compute,
                int(input_size[0]),
                int(weight_size[0])
            );
        }
    }

    return {output_tensor_pos, output_tensor_neg};
}