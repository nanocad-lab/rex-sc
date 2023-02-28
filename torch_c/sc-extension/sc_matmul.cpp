#include "sc_cpu.hpp"

void stream_generation_linear(
    const int32_t* weight_pos,
    const int32_t* weight_neg,
    int32_t* weight_pos_stream,
    int32_t* weight_neg_stream,
    int bit_length,
    int lfsr_length,
    int gen_config,
    bool gen_mult,
    int c_outs,
    int c_ins,
    int total_width,
    int load_width,
    int load_wait) {

    int z_packs = (c_ins + COMPUTE_CINS-1) / COMPUTE_CINS;
    int seed_mult = 1;

    int (*lfsr)(int);
    switch(lfsr_length) {
    case 3:
        lfsr=&lfsr_3_s;
        break;
    case 4:
        lfsr=&lfsr_4_s;
        seed_mult = SEED_4;
        break;
    case 5:
        lfsr=&lfsr_5_s;
        seed_mult = SEED_5;
        break;
    case 6:
        lfsr=&lfsr_6_s;
        seed_mult = SEED_6;
        break;
    case 7:
        lfsr=&lfsr_7_s;
        seed_mult = SEED_7;
        break;
    case 8:
        lfsr=&lfsr_8_s;
        break;
    }

    #ifdef __AVX512F__
    __m512i (*lfsr_512)(__m512i);
    switch(lfsr_length) {
    case 3:
        lfsr_512=&lfsr_3_512;
        break;
    case 4:
        lfsr_512=&lfsr_4_512;
        seed_mult = SEED_4;
        break;
    case 5:
        lfsr_512=&lfsr_5_512;
        seed_mult = SEED_5;
        break;
    case 6:
        lfsr_512=&lfsr_6_512;
        seed_mult = SEED_6;
        break;
    case 7:
        lfsr_512=&lfsr_7_512;
        seed_mult = SEED_7;
        break;
    }
    #endif

    #ifdef __AVX512F__
    #pragma omp parallel for
    for (int i=0; i<c_outs*z_packs; i++) {
        int c_out = i/z_packs;
        int z_pack = i%z_packs;
        int weight_ind = c_out*c_ins+z_pack*32;

        // For AVX512, 2 registers are needed to fill 32x32
        __m512i pos_seed_0;
        __m512i pos_seed_1;
        __m512i neg_seed_0;
        __m512i neg_seed_1;
        __m512i weight_pos_reg_0;
        __m512i weight_pos_reg_1;
        __m512i weight_neg_reg_0;
        __m512i weight_neg_reg_1;
        int32_t pos_seed[COMPUTE_CINS];
        int32_t neg_seed[COMPUTE_CINS];
        // Load seeds and weights

        if (gen_mult) {
            pos_seed_0 = _mm512_set1_epi32(seed_mult);
            pos_seed_1 = _mm512_set1_epi32(seed_mult);
            neg_seed_0 = _mm512_set1_epi32(seed_mult);
            neg_seed_1 = _mm512_set1_epi32(seed_mult);
        }
        else {
            for(int index_seed=0; index_seed<COMPUTE_CINS; index_seed++) {
                pos_seed[index_seed] = (POS_SEED+z_pack*COMPUTE_CINS+index_seed)%((1<<lfsr_length)-1) + 1;
                neg_seed[index_seed] = (NEG_SEED+z_pack*COMPUTE_CINS+index_seed)%((1<<lfsr_length)-1) + 1;
            }
            pos_seed_0 = _mm512_loadu_si512(&(pos_seed[0]));
            pos_seed_1 = _mm512_loadu_si512(&(pos_seed[16]));
            neg_seed_0 = _mm512_loadu_si512(&(neg_seed[0]));
            neg_seed_1 = _mm512_loadu_si512(&(neg_seed[16]));
        }
        if (z_pack*32+31<c_ins) {
            weight_pos_reg_0 = _mm512_loadu_si512(&(weight_pos[weight_ind+0]));
            weight_pos_reg_1 = _mm512_loadu_si512(&(weight_pos[weight_ind+16]));
            weight_neg_reg_0 = _mm512_loadu_si512(&(weight_neg[weight_ind+0]));
            weight_neg_reg_1 = _mm512_loadu_si512(&(weight_neg[weight_ind+16]));
        }
        else {
            int32_t weight_pos_cur[COMPUTE_CINS] = {0};
            int32_t weight_neg_cur[COMPUTE_CINS] = {0};
            for (int index_cin=0; index_cin<std::min(COMPUTE_CINS, c_ins-z_pack*COMPUTE_CINS); index_cin++) {
                weight_pos_cur[index_cin] = weight_pos[weight_ind+index_cin];
                weight_neg_cur[index_cin] = weight_neg[weight_ind+index_cin];
            }
            weight_pos_reg_0 = _mm512_loadu_si512(&(weight_pos_cur[0]));
            weight_pos_reg_1 = _mm512_loadu_si512(&(weight_pos_cur[16]));
            weight_neg_reg_0 = _mm512_loadu_si512(&(weight_neg_cur[0]));
            weight_neg_reg_1 = _mm512_loadu_si512(&(weight_neg_cur[16]));
        }

        int cur_width = 0;
        for(int bit=0; bit<bit_length; bit++) {
            cur_width = (bit/load_wait + 1)*load_width - 1;
            if (cur_width > total_width) cur_width=total_width;
            uint weight_pos_stream_c = 0;
            uint weight_neg_stream_c = 0;

            int shift_cur = total_width - cur_width;
            __m512i weight_pos_actual_0 = _mm512_slli_epi32(_mm512_srli_epi32(weight_pos_reg_0, shift_cur), shift_cur);
            __m512i weight_neg_actual_0 = _mm512_slli_epi32(_mm512_srli_epi32(weight_neg_reg_0, shift_cur), shift_cur);
            __m512i weight_pos_actual_1 = _mm512_slli_epi32(_mm512_srli_epi32(weight_pos_reg_1, shift_cur), shift_cur);
            __m512i weight_neg_actual_1 = _mm512_slli_epi32(_mm512_srli_epi32(weight_neg_reg_1, shift_cur), shift_cur);

            pos_seed_0 = (*lfsr_512)(pos_seed_0);
            pos_seed_1 = (*lfsr_512)(pos_seed_1);
            neg_seed_0 = (*lfsr_512)(neg_seed_0);
            neg_seed_1 = (*lfsr_512)(neg_seed_1);

            weight_pos_stream_c += reinterpret_cast<uint16_t>(_mm512_cmp_epi32_mask(weight_pos_actual_0, pos_seed_0, 6));
            weight_pos_stream_c += uint(reinterpret_cast<uint16_t>(_mm512_cmp_epi32_mask(weight_pos_actual_1, pos_seed_1, 6))) << 16;
            weight_neg_stream_c += reinterpret_cast<uint16_t>(_mm512_cmp_epi32_mask(weight_neg_actual_0, neg_seed_0, 6));
            weight_neg_stream_c += uint(reinterpret_cast<uint16_t>(_mm512_cmp_epi32_mask(weight_neg_actual_1, neg_seed_1, 6))) << 16;

            for(int gen_and=0; gen_and<gen_config; gen_and++) {
                weight_pos_stream[(bit*gen_config+gen_and)*c_outs*z_packs + i] = weight_pos_stream_c;
                weight_neg_stream[(bit*gen_config+gen_and)*c_outs*z_packs + i] = weight_neg_stream_c;
            }
        }
    }
    #else
    #pragma omp parallel for
    for(int i=0; i<c_outs*z_packs; i++) {
        int c_out = i/z_packs;
        int z_pack = i%z_packs;
        int weight_ind = i*COMPUTE_CINS;

        int32_t pos_seed[COMPUTE_CINS];
        int32_t neg_seed[COMPUTE_CINS];
        int32_t weight_pos_shared[COMPUTE_CINS];
        int32_t weight_neg_shared[COMPUTE_CINS];

        for(int compute_cin=0; compute_cin<COMPUTE_CINS; compute_cin++) {
            int c_in = z_pack*COMPUTE_CINS+compute_cin;
            if (c_in<c_ins) {
                if (gen_mult) {
                    pos_seed[compute_cin] = seed_mult;
                    neg_seed[compute_cin] = seed_mult;
                }
                else {
                    pos_seed[compute_cin] = (POS_SEED+c_in)%((1<<lfsr_length)-1) + 1;
                    neg_seed[compute_cin] = (NEG_SEED+c_in)%((1<<lfsr_length)-1) + 1;
                }
                weight_pos_shared[compute_cin] = weight_pos[weight_ind+compute_cin];
                weight_neg_shared[compute_cin] = weight_neg[weight_ind+compute_cin];
            }
            else {
                pos_seed[compute_cin] = int(0);
                neg_seed[compute_cin] = int(0);
                weight_pos_shared[compute_cin] = int(0);
                weight_neg_shared[compute_cin] = int(0);
            }
        }

        int cur_width = 0;
        for(int bit=0; bit<bit_length;bit++){
            cur_width = (bit/load_wait + 1)*load_width - 1;
            if (cur_width > total_width) cur_width=total_width;
            int weight_pos_stream_c = 0;
            int weight_neg_stream_c = 0;
            for(int compute_cin=0; compute_cin<COMPUTE_CINS; compute_cin++) {
                int weight_pos_actual = (weight_pos_shared[compute_cin] >> (total_width-cur_width)) << (total_width-cur_width);
                int weight_neg_actual = (weight_neg_shared[compute_cin] >> (total_width-cur_width)) << (total_width-cur_width);
                int pos_seed_cur = pos_seed[compute_cin];
                int neg_seed_cur = neg_seed[compute_cin];
                pos_seed_cur = (*lfsr)(pos_seed_cur);
                neg_seed_cur = (*lfsr)(neg_seed_cur);

                weight_pos_stream_c += int(weight_pos_actual>pos_seed_cur) << compute_cin;
                weight_neg_stream_c += int(weight_neg_actual>neg_seed_cur) << compute_cin;

                pos_seed[compute_cin] = pos_seed_cur;
                neg_seed[compute_cin] = neg_seed_cur;
            }
            for(int gen_and=0; gen_and<gen_config; gen_and++) {

                weight_pos_stream[(bit*gen_config+gen_and)*c_outs*z_packs + i] = weight_pos_stream_c;
                weight_neg_stream[(bit*gen_config+gen_and)*c_outs*z_packs + i] = weight_neg_stream_c;
            }
        }
    }
    #endif
}

void activation_general_linear(
    const int32_t* input_bin,
    int32_t* input_stream,
    int bit_length,
    int lfsr_length,
    int gen_config,
    bool gen_mult,
    int batches,
    int c_ins,
    const int total_width,
    const int load_width,
    const int load_wait) {
    int z_packs = (c_ins + COMPUTE_CINS-1) / COMPUTE_CINS;
    int seed_mult = 1;

    int (*lfsr)(int);
    switch(lfsr_length) {
    case 3:
        lfsr=&lfsr_3_s;
        break;
    case 4:
        lfsr=&lfsr_4_s;
        break;
    case 5:
        lfsr=&lfsr_5_s;
        break;
    case 6:
        lfsr=&lfsr_6_s;
        break;
    case 7:
        lfsr=&lfsr_7_s;
        break;
    case 8:
        lfsr=&lfsr_8_s;
        break;
    }

    #ifdef __AVX512F__
    __m512i (*lfsr_512)(__m512i);
    switch(lfsr_length) {
    case 3:
        lfsr_512=&lfsr_3_512;
        break;
    case 4:
        lfsr_512=&lfsr_4_512;
        seed_mult = SEED_4;
        break;
    case 5:
        lfsr_512=&lfsr_5_512;
        seed_mult = SEED_5;
        break;
    case 6:
        lfsr_512=&lfsr_6_512;
        seed_mult = SEED_6;
        break;
    case 7:
        lfsr_512=&lfsr_7_512;
        seed_mult = SEED_7;
        break;
    }
    #endif

    #ifdef __AVX512F__
    #pragma omp parallel for
    for (int i=0; i<batches*z_packs; i++) {
        int batch = i/z_packs;
        int z_pack = i%z_packs;
        int input_ind = batch*c_ins+z_pack*32;
        // int input_ind = i*32;

        __m512i seed_0;
        __m512i seed_1;
        __m512i input_reg_0;
        __m512i input_reg_1;
        int32_t seed[COMPUTE_CINS];

        if (gen_mult) {
            seed_0 = _mm512_set1_epi32(1);
            seed_1 = _mm512_set1_epi32(1);
        }
        else {
            for(int index_seed=0; index_seed<COMPUTE_CINS; index_seed++) {
                seed[index_seed] = (0+z_pack*COMPUTE_CINS+index_seed)%((1<<lfsr_length)-1) + 1;
            }
            seed_0 = _mm512_loadu_si512(&(seed[0]));
            seed_1 = _mm512_loadu_si512(&(seed[16]));
        }
        if (z_pack*32+31<c_ins) {
            input_reg_0 = _mm512_loadu_si512(input_bin+input_ind);
            input_reg_1 = _mm512_loadu_si512(input_bin+input_ind+16);
        }
        else {
            int32_t input_cur[COMPUTE_CINS] = {0};
            for (int index_cin=0; index_cin<std::min(COMPUTE_CINS, c_ins-z_pack*COMPUTE_CINS); index_cin++) {
                input_cur[index_cin] = input_bin[input_ind+index_cin];
            }
            input_reg_0 = _mm512_loadu_si512(&(input_cur[0]));
            input_reg_1 = _mm512_loadu_si512(&(input_cur[16]));
        }

        int cur_width = 0;
        for(int bit=0; bit<bit_length; bit++) {
            cur_width = (bit/load_wait + 1)*load_width - 1;
            if (cur_width > total_width) cur_width=total_width;
            uint input_stream_c = 0;

            int shift_cur = total_width - cur_width;
            __m512i input_actual_0 = _mm512_slli_epi32(_mm512_srli_epi32(input_reg_0, shift_cur), shift_cur);
            __m512i input_actual_1 = _mm512_slli_epi32(_mm512_srli_epi32(input_reg_1, shift_cur), shift_cur);

            seed_0 = (*lfsr_512)(seed_0);
            seed_1 = (*lfsr_512)(seed_1);

            input_stream_c += reinterpret_cast<uint16_t>(_mm512_cmp_epi32_mask(seed_0, input_actual_0, 1));
            input_stream_c += uint(reinterpret_cast<uint16_t>(_mm512_cmp_epi32_mask(seed_1, input_actual_1, 1))) << 16;

            for(int gen_and=0; gen_and<gen_config; gen_and++) {
                input_stream[((bit/gen_config)*gen_config*gen_config + gen_and*gen_config + bit%gen_config)*batches*z_packs+i] = input_stream_c;
            }
        }
    }
    #else
    #pragma omp parallel for
    for(int i=0; i<batches*z_packs; i++) {
        int batch = i/z_packs;
        int z_pack = i%z_packs;
        int input_ind = i*COMPUTE_CINS;

        int32_t seed[COMPUTE_CINS];
        int32_t input_shared[COMPUTE_CINS];

        for(int compute_cin=0; compute_cin<COMPUTE_CINS; compute_cin++) {
            int c_in = z_pack*COMPUTE_CINS+compute_cin;
            if (c_in<c_ins) {
                if (gen_mult) seed[compute_cin] = int(1);
                else seed[compute_cin] = (0+c_in)%((1<<lfsr_length)-1) + 1;
                input_shared[compute_cin] = input_bin[input_ind+compute_cin];
            }
            else {
                seed[compute_cin] = int(0);
                input_shared[compute_cin] = int(0);
            }
        }

        int cur_width = 0;
        for(int bit=0; bit<bit_length; bit++){
            cur_width = (bit/load_wait + 1)*load_width - 1;
            if (cur_width > total_width) cur_width=total_width;
            int input_stream_c = 0;
            for(int compute_cin=0; compute_cin<COMPUTE_CINS; compute_cin++) {
                int input_actual = (input_shared[compute_cin] >> (total_width-cur_width)) << (total_width-cur_width);
                int seed_cur = seed[compute_cin];
                seed_cur = (*lfsr)(seed_cur);

                input_stream_c += int(input_actual>seed_cur) << compute_cin;
                seed[compute_cin] = seed_cur;
            }
            for(int gen_and=0; gen_and<gen_config; gen_and++) {
                input_stream[((bit/gen_config)*gen_config*gen_config + gen_and*gen_config + bit%gen_config)*batches*z_packs+i] = input_stream_c;
            }
        }
    }
    #endif
}

// Blocking configuration

const int M_GENERAL = 64;
const int N_GENERAL = 64;

void stream_compute_linear_general(
    const int32_t* input_stream,
    const int32_t* weight_pos_stream,
    const int32_t* weight_neg_stream,
    int32_t* output_pos_stream,
    int32_t* output_neg_stream,
    int bit_length,
    int batches,
    int c_ins,
    int c_outs,
    int bin_config) {
    bool or_no = false;
    if (bin_config>=0) bin_config+=1;
    else {
        bin_config = -bin_config+1;
        or_no = true;
    }

    int z_packs = (c_ins + COMPUTE_CINS-1) / COMPUTE_CINS;
    int batch_packs = (batches + 2*M_GENERAL-1) / (2*M_GENERAL);
    int cout_packs = (c_outs + 2*N_GENERAL-1) / (2*N_GENERAL);

    #pragma omp parallel for
    for (int block=0; block<batch_packs*cout_packs; block++) {
        int batch_pack = block/cout_packs;
        int cout_pack = block%cout_packs;
        const int32_t* input_batch = input_stream + batch_pack*2*M_GENERAL*z_packs;
        const int32_t* weight_pos_cout = weight_pos_stream + cout_pack*2*N_GENERAL*z_packs;
        const int32_t* weight_neg_cout = weight_neg_stream + cout_pack*2*N_GENERAL*z_packs;
        int32_t* output_pos_block = output_pos_stream + (batch_pack*2*M_GENERAL)*c_outs + cout_pack*2*N_GENERAL;
        int32_t* output_neg_block = output_neg_stream + (batch_pack*2*M_GENERAL)*c_outs + cout_pack*2*N_GENERAL;

        for (int bit=0; bit<bit_length; bit++) {
            const int32_t* input_bit = input_batch + bit*batches*z_packs;
            const int32_t* weight_pos_bit = weight_pos_cout + bit*c_outs*z_packs;
            const int32_t* weight_neg_bit = weight_neg_cout + bit*c_outs*z_packs;

            for (int m=0; m<M_GENERAL; m++) {
                const int32_t* input_batch_c_0 = input_bit + (m+0*M_GENERAL)*z_packs;
                const int32_t* input_batch_c_1 = input_bit + (m+1*M_GENERAL)*z_packs;
                int32_t* output_pos_m_0 = output_pos_block + (m+0*M_GENERAL)*c_outs;
                int32_t* output_neg_m_0 = output_neg_block + (m+0*M_GENERAL)*c_outs;
                int32_t* output_pos_m_1 = output_pos_block + (m+1*M_GENERAL)*c_outs;
                int32_t* output_neg_m_1 = output_neg_block + (m+1*M_GENERAL)*c_outs;
                for (int n=0; n<N_GENERAL; n++) {
                    const int32_t* weight_pos_cout_c_0 = weight_pos_bit + (n+0*N_GENERAL)*z_packs;
                    const int32_t* weight_neg_cout_c_0 = weight_neg_bit + (n+0*N_GENERAL)*z_packs;
                    const int32_t* weight_pos_cout_c_1 = weight_pos_bit + (n+1*N_GENERAL)*z_packs;
                    const int32_t* weight_neg_cout_c_1 = weight_neg_bit + (n+1*N_GENERAL)*z_packs;

                    int32_t output_pos_c_00 = 0;
                    int32_t output_neg_c_00 = 0;
                    int32_t output_pos_c_01 = 0;
                    int32_t output_neg_c_01 = 0;
                    int32_t output_pos_c_10 = 0;
                    int32_t output_neg_c_10 = 0;
                    int32_t output_pos_c_11 = 0;
                    int32_t output_neg_c_11 = 0;

                    size_t z_pack=0;
                    #ifdef __AVX512F__
                    #ifdef __AVX512VPOPCNTDQ__
                    __m512i output_pos_acc_00 = _mm512_setzero_si512();
                    __m512i output_pos_acc_01 = _mm512_setzero_si512();
                    __m512i output_pos_acc_10 = _mm512_setzero_si512();
                    __m512i output_pos_acc_11 = _mm512_setzero_si512();
                    __m512i output_neg_acc_00 = _mm512_setzero_si512();
                    __m512i output_neg_acc_01 = _mm512_setzero_si512();
                    __m512i output_neg_acc_10 = _mm512_setzero_si512();
                    __m512i output_neg_acc_11 = _mm512_setzero_si512();
                    // #endif
                    for (; z_pack+15<z_packs; z_pack+=16) {
                        __m512i input_c_0 = _mm512_loadu_si512(input_batch_c_0+z_pack);
                        __m512i input_c_1 = _mm512_loadu_si512(input_batch_c_1+z_pack);
                        // printf("Z_PACK offset %d, Total offset %d\n", z_pack, cout_pack*N_GENERAL*z_packs+bit*c_outs*z_packs+n*z_packs);
                        __m512i weight_pos_c_0 = _mm512_loadu_si512(&(weight_pos_cout_c_0[z_pack]));
                        __m512i weight_neg_c_0 = _mm512_loadu_si512(&(weight_neg_cout_c_0[z_pack]));
                        __m512i weight_pos_c_1 = _mm512_loadu_si512(&(weight_pos_cout_c_1[z_pack]));
                        __m512i weight_neg_c_1 = _mm512_loadu_si512(&(weight_neg_cout_c_1[z_pack]));
                        __m512i prod_neg_00 = _mm512_and_si512(input_c_0, weight_neg_c_0);
                        __m512i prod_pos_00 = _mm512_and_si512(input_c_0, weight_pos_c_0);
                        __m512i prod_neg_01 = _mm512_and_si512(input_c_0, weight_neg_c_1);
                        __m512i prod_pos_01 = _mm512_and_si512(input_c_0, weight_pos_c_1);
                        __m512i prod_neg_10 = _mm512_and_si512(input_c_1, weight_neg_c_0);
                        __m512i prod_pos_10 = _mm512_and_si512(input_c_1, weight_pos_c_0);
                        __m512i prod_neg_11 = _mm512_and_si512(input_c_1, weight_neg_c_1);
                        __m512i prod_pos_11 = _mm512_and_si512(input_c_1, weight_pos_c_1);
                        prod_pos_00 = _mm512_popcnt_epi64(prod_pos_00);
                        prod_neg_00 = _mm512_popcnt_epi64(prod_neg_00);
                        prod_pos_01 = _mm512_popcnt_epi64(prod_pos_01);
                        prod_neg_01 = _mm512_popcnt_epi64(prod_neg_01);
                        prod_pos_10 = _mm512_popcnt_epi64(prod_pos_10);
                        prod_neg_10 = _mm512_popcnt_epi64(prod_neg_10);
                        prod_pos_11 = _mm512_popcnt_epi64(prod_pos_11);
                        prod_neg_11 = _mm512_popcnt_epi64(prod_neg_11);
                        output_pos_acc_00 = _mm512_add_epi64(output_pos_acc_00, prod_pos_00);
                        output_neg_acc_00 = _mm512_add_epi64(output_neg_acc_00, prod_neg_00);
                        output_pos_acc_01 = _mm512_add_epi64(output_pos_acc_01, prod_pos_01);
                        output_neg_acc_01 = _mm512_add_epi64(output_neg_acc_01, prod_neg_01);
                        output_pos_acc_10 = _mm512_add_epi64(output_pos_acc_10, prod_pos_10);
                        output_neg_acc_10 = _mm512_add_epi64(output_neg_acc_10, prod_neg_10);
                        output_pos_acc_11 = _mm512_add_epi64(output_pos_acc_11, prod_pos_11);
                        output_neg_acc_11 = _mm512_add_epi64(output_neg_acc_11, prod_neg_11);
                    }
                    output_pos_c_00 += _mm512_reduce_add_epi64(output_pos_acc_00);
                    output_neg_c_00 += _mm512_reduce_add_epi64(output_neg_acc_00);
                    output_pos_c_01 += _mm512_reduce_add_epi64(output_pos_acc_01);
                    output_neg_c_01 += _mm512_reduce_add_epi64(output_neg_acc_01);
                    output_pos_c_10 += _mm512_reduce_add_epi64(output_pos_acc_10);
                    output_neg_c_10 += _mm512_reduce_add_epi64(output_neg_acc_10);
                    output_pos_c_11 += _mm512_reduce_add_epi64(output_pos_acc_11);
                    output_neg_c_11 += _mm512_reduce_add_epi64(output_neg_acc_11);
                    #endif
                    #endif
                    for (; z_pack<z_packs; z_pack++) {
                        int input_c_0 = input_batch_c_0[z_pack];
                        int weight_pos_c_0 = weight_pos_cout_c_0[z_pack];
                        int weight_neg_c_0 = weight_neg_cout_c_0[z_pack];
                        int input_c_1 = input_batch_c_1[z_pack];
                        int weight_pos_c_1 = weight_pos_cout_c_1[z_pack];
                        int weight_neg_c_1 = weight_neg_cout_c_1[z_pack];

                        int prod_pos_00 = input_c_0 & weight_pos_c_0;
                        int prod_neg_00 = input_c_0 & weight_neg_c_0;
                        int prod_pos_01 = input_c_0 & weight_pos_c_1;
                        int prod_neg_01 = input_c_0 & weight_neg_c_1;
                        int prod_pos_10 = input_c_1 & weight_pos_c_0;
                        int prod_neg_10 = input_c_1 & weight_neg_c_0;
                        int prod_pos_11 = input_c_1 & weight_pos_c_1;
                        int prod_neg_11 = input_c_1 & weight_neg_c_1;
                        
                        output_pos_c_00 += __builtin_popcount(prod_pos_00);
                        output_neg_c_00 += __builtin_popcount(prod_neg_00);
                        output_pos_c_01 += __builtin_popcount(prod_pos_01);
                        output_neg_c_01 += __builtin_popcount(prod_neg_01);
                        output_pos_c_10 += __builtin_popcount(prod_pos_10);
                        output_neg_c_10 += __builtin_popcount(prod_neg_10);
                        output_pos_c_11 += __builtin_popcount(prod_pos_11);
                        output_neg_c_11 += __builtin_popcount(prod_neg_11);
                    }
                    if ((batch_pack*2*M_GENERAL+m+0*M_GENERAL<batches) && (cout_pack*2*N_GENERAL+n+0*N_GENERAL<c_outs)) {
                        if (or_no) {
                            output_pos_m_0[n+0*N_GENERAL] += int(output_pos_c_00>=bin_config);
                            output_neg_m_0[n+0*N_GENERAL] += int(output_neg_c_00>=bin_config);
                        }
                        else {
                            output_pos_m_0[n+0*N_GENERAL] += std::min(output_pos_c_00, bin_config);
                            output_neg_m_0[n+0*N_GENERAL] += std::min(output_neg_c_00, bin_config);
                        }
                    }
                    if ((batch_pack*2*M_GENERAL+m+0*M_GENERAL<batches) && (cout_pack*2*N_GENERAL+n+1*N_GENERAL<c_outs)) {
                        if (or_no) {
                            output_pos_m_0[n+1*N_GENERAL] += int(output_pos_c_01>=bin_config);
                            output_neg_m_0[n+1*N_GENERAL] += int(output_neg_c_01>=bin_config);
                        }
                        else {
                            output_pos_m_0[n+1*N_GENERAL] += std::min(output_pos_c_01, bin_config);
                            output_neg_m_0[n+1*N_GENERAL] += std::min(output_neg_c_01, bin_config);
                        }
                    }
                    if ((batch_pack*2*M_GENERAL+m+1*M_GENERAL<batches) && (cout_pack*2*N_GENERAL+n+0*N_GENERAL<c_outs)) {
                        if (or_no) {
                            output_pos_m_1[n+0*N_GENERAL] += int(output_pos_c_10>=bin_config);
                            output_neg_m_1[n+0*N_GENERAL] += int(output_neg_c_10>=bin_config);
                        }
                        else {
                            output_pos_m_1[n+0*N_GENERAL] += std::min(output_pos_c_10, bin_config);
                            output_neg_m_1[n+0*N_GENERAL] += std::min(output_neg_c_10, bin_config);
                        }
                    }
                    if ((batch_pack*2*M_GENERAL+m+1*M_GENERAL<batches) && (cout_pack*2*N_GENERAL+n+1*N_GENERAL<c_outs)) {
                        if (or_no) {
                            output_pos_m_1[n+1*N_GENERAL] += int(output_pos_c_11>=bin_config);
                            output_neg_m_1[n+1*N_GENERAL] += int(output_neg_c_11>=bin_config);
                        }
                        else {
                            output_pos_m_1[n+1*N_GENERAL] += std::min(output_pos_c_11, bin_config);
                            output_neg_m_1[n+1*N_GENERAL] += std::min(output_neg_c_11, bin_config);
                        }
                    }
                }
            }
        }
    }
}

std::vector<torch::Tensor> linear_generic_general(torch::Tensor input,
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
    auto weight_pos = (weight*lfsr_bit_length).clamp(0, lfsr_bit_length-1).ceil().to(compare_type);
    auto weight_neg = (-(weight*lfsr_bit_length).clamp(1-lfsr_bit_length, 0)).ceil().to(compare_type);

    auto weight_size = weight_pos.sizes();
    auto input_size = input_split.sizes();

    int z_packs = (input.size(1) + COMPUTE_CINS-1) / COMPUTE_CINS;

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

    int32_t* weight_pos_stream = (int32_t*) malloc(compute_length*weight_pos.size(0) * z_packs * sizeof(int32_t));
    int32_t* weight_neg_stream = (int32_t*) malloc(compute_length*weight_pos.size(0) * z_packs * sizeof(int32_t));
    int32_t* input_stream = (int32_t*) malloc(compute_length*input_split.size(0) * z_packs * sizeof(int32_t));

    stream_generation_linear(
        weight_pos.data_ptr<int32_t>(),
        weight_neg.data_ptr<int32_t>(),
        weight_pos_stream,
        weight_neg_stream,
        bit_length,
        lfsr_length,
        gen_config,
        false,
        weight_size.data()[0],
        weight_size.data()[1],
        prog_load.data()[0],
        prog_load.data()[1],
        prog_load.data()[2]);
    activation_general_linear(
        input_split.data_ptr<int32_t>(),
        input_stream,
        bit_length,
        lfsr_length,
        gen_config,
        false,
        input_size.data()[0],
        input_size.data()[1],
        prog_load.data()[0],
        prog_load.data()[1],
        prog_load.data()[2]);

    auto output_tensor_pos = torch::zeros({input_size.data()[0], weight_size.data()[0]}, at::TensorOptions().dtype(torch::kInt32));
    auto output_tensor_neg = torch::zeros({input_size.data()[0], weight_size.data()[0]}, at::TensorOptions().dtype(torch::kInt32));

    stream_compute_linear_general(
        input_stream,
        weight_pos_stream,
        weight_neg_stream,
        output_tensor_pos.data_ptr<int32_t>(),
        output_tensor_neg.data_ptr<int32_t>(),
        bit_length,
        input_size.data()[0],
        weight_size.data()[1],
        weight_size.data()[0],
        bin_config);
    delete [] weight_pos_stream;
    delete [] weight_neg_stream;
    delete [] input_stream;
    return {output_tensor_pos, output_tensor_neg};
}

void
stream_compute_matmul(
    const int32_t* input_stream,
    const int32_t* weight_pos_stream,
    const int32_t* weight_neg_stream,
    int16_t* output_pos_stream,
    int16_t* output_neg_stream,
    int k_total,
    int m_total,
    int n_total) {
    #pragma omp parallel for
    for (int n=0; n<n_total; n++) {
        const int32_t* weight_pos_n = weight_pos_stream + n*k_total;
        const int32_t* weight_neg_n = weight_neg_stream + n*k_total;
        for (int m=0; m<m_total; m++) {
            int output_pos_c = output_pos_stream[m*n_total+n];
            int output_neg_c = output_neg_stream[m*n_total+n];
            const int32_t* input_m = input_stream + m*k_total;
            for (int k=0; k<k_total; k++) {
                output_pos_c += __builtin_popcount(input_m[k] & weight_pos_n[k]);
                output_neg_c += __builtin_popcount(input_m[k] & weight_neg_n[k]);
            }
            output_pos_stream[m*n_total+n]=output_pos_c;
            output_neg_stream[m*n_total+n]=output_neg_c;
        }
    }
}
