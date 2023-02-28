#include "sc_device.cuh"

bool Init_Temp = false;

// Compute a binary gemm between (I_UNROLL_TEST*WMMA_M_UNROLL*16, 256) and (W_UNROLL_TEST*WMMA_N_UNROLL*16, 256) for each warp
// Using ptx as the intrinsics do not support ldmatrix with non-uniform stride and 16x8x256 binary matmul
__device__ void 
compute_unit_gemm(
    int32_t* weight_pos_s,
    int32_t* weight_neg_s,
    int32_t* input_s,
    int (&acc_frag_pos)[I_UNROLL_TEST][W_UNROLL_TEST][WMMA_M_UNROLL][WMMA_N_UNROLL][2],
    int (&acc_frag_neg)[I_UNROLL_TEST][W_UNROLL_TEST][WMMA_M_UNROLL][WMMA_N_UNROLL][2],
    int (&output_value_pos)[2*I_UNROLL_TEST][2*W_UNROLL_TEST],
    int (&output_value_neg)[2*I_UNROLL_TEST][2*W_UNROLL_TEST],
    struct Gemm_Unit_Param gemm_param,
    int i_8
) {
    #if __CUDA_ARCH__ >= 800
    int a_frag[I_UNROLL_TEST][WMMA_M_UNROLL][WMMA_K_UNROLL];
    int b_frag_pos[W_UNROLL_TEST][WMMA_N_UNROLL][WMMA_K_UNROLL];
    int b_frag_neg[W_UNROLL_TEST][WMMA_N_UNROLL][WMMA_K_UNROLL];
    #pragma unroll
    for (int index_i=0; index_i<I_UNROLL_TEST; index_i++) {
        // WMMA_M_bin_TEST number of rows
        // Stride v2
        int32_t* start_addr = input_s+(gemm_param.warp_m*I_UNROLL_TEST+index_i)*WMMA_M_bin_TEST*K_GENERAL_UNIT_STORE
                                +((gemm_param.id_warp%16)/8)*8*K_GENERAL_UNIT_STORE+(gemm_param.id_warp%8)*32+((gemm_param.id_warp%8)*4+(gemm_param.id_warp/16)*4+i_8*8)%32;
        asm volatile ("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];" : 
                    "=r"(a_frag[index_i][0][0]), "=r"(a_frag[index_i][1][0]),
                    "=r"(a_frag[index_i][0][1]), "=r"(a_frag[index_i][1][1]):
                    "l"(start_addr));
    }
    #pragma unroll
    for (int index_w=0; index_w<W_UNROLL_TEST; index_w++) {
        // Stride v2
        int start_addr_offset = (gemm_param.warp_n*W_UNROLL_TEST+index_w)*WMMA_N_bin_TEST*K_GENERAL_UNIT_STORE
                                +(gemm_param.id_warp%8)*32+((gemm_param.id_warp%8)*4+((gemm_param.id_warp%16)/8)*4+i_8*8)%32;
        int32_t* start_addr_pos = weight_pos_s + start_addr_offset;
        int32_t* start_addr_neg = weight_neg_s + start_addr_offset;
        asm volatile ("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1}, [%2];" :
                    "=r"(b_frag_pos[index_w][0][0]),"=r"(b_frag_pos[index_w][0][1]):
                    "l"(start_addr_pos));
        asm volatile ("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1}, [%2];" :
                    "=r"(b_frag_neg[index_w][0][0]),"=r"(b_frag_neg[index_w][0][1]):
                    "l"(start_addr_neg));
    }
    #pragma unroll
    for (int index_i=0; index_i<I_UNROLL_TEST; index_i++) {
        #pragma unroll
        for (int index_w=0; index_w<W_UNROLL_TEST; index_w++) {
            asm volatile ("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
            : "=r"(acc_frag_pos[index_i][index_w][0][0][0]), "=r"(acc_frag_pos[index_i][index_w][0][0][1]),
                "=r"(acc_frag_pos[index_i][index_w][1][0][0]), "=r"(acc_frag_pos[index_i][index_w][1][0][1])
            : "r"(a_frag[index_i][0][0]), "r"(a_frag[index_i][1][0]),
                "r"(a_frag[index_i][0][1]), "r"(a_frag[index_i][1][1]),
                "r"(b_frag_pos[index_w][0][0]), "r"(b_frag_pos[index_w][0][1]),
                "r"(acc_frag_pos[index_i][index_w][0][0][0]), "r"(acc_frag_pos[index_i][index_w][0][0][1]),
                "r"(acc_frag_pos[index_i][index_w][1][0][0]), "r"(acc_frag_pos[index_i][index_w][1][0][1]));
            asm volatile ("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
            : "=r"(acc_frag_neg[index_i][index_w][0][0][0]), "=r"(acc_frag_neg[index_i][index_w][0][0][1]),
                "=r"(acc_frag_neg[index_i][index_w][1][0][0]), "=r"(acc_frag_neg[index_i][index_w][1][0][1])
            : "r"(a_frag[index_i][0][0]), "r"(a_frag[index_i][1][0]),
                "r"(a_frag[index_i][0][1]), "r"(a_frag[index_i][1][1]),
                "r"(b_frag_neg[index_w][0][0]), "r"(b_frag_neg[index_w][0][1]),
                "r"(acc_frag_neg[index_i][index_w][0][0][0]), "r"(acc_frag_neg[index_i][index_w][0][0][1]),
                "r"(acc_frag_neg[index_i][index_w][1][0][0]), "r"(acc_frag_neg[index_i][index_w][1][0][1]));
        }
    }
    #else
    for (int i=0; i<8; i++) {
        int32_t input_s_a[I_UNROLL_TEST][2];
        int32_t weight_pos_s_a[W_UNROLL_TEST][2];
        int32_t weight_neg_s_a[W_UNROLL_TEST][2];
        for (int index_i=0; index_i<I_UNROLL_TEST; index_i++) {
            for (int index_i_2=0; index_i_2<2; index_i_2++) {
                int i_correct = (i_8*8+i+gemm_param.index_i_c*4)%32;
                input_s_a[index_i][index_i_2] = input_s[(gemm_param.warp_m*I_UNROLL_TEST*WMMA_M_bin_TEST+index_i*WMMA_M_bin_TEST+gemm_param.index_i_c+index_i_2*8)*K_GENERAL_UNIT_STORE+i_correct];
            }
        }
        for (int index_w=0; index_w<W_UNROLL_TEST; index_w++) {
            for (int index_w_2=0; index_w_2<2; index_w_2++) {
                int i_correct = (i_8*8+i+(gemm_param.index_w_c+index_w_2)*4)%32;
                weight_pos_s_a[index_w][index_w_2] = weight_pos_s[(gemm_param.warp_n*W_UNROLL_TEST*WMMA_N_bin_TEST+index_w*WMMA_N_bin_TEST+gemm_param.index_w_c+index_w_2)*K_GENERAL_UNIT_STORE+i_correct];
                weight_neg_s_a[index_w][index_w_2] = weight_neg_s[(gemm_param.warp_n*W_UNROLL_TEST*WMMA_N_bin_TEST+index_w*WMMA_N_bin_TEST+gemm_param.index_w_c+index_w_2)*K_GENERAL_UNIT_STORE+i_correct];
            }
        }
        for (int index_i=0; index_i<I_UNROLL_TEST; index_i++) {
            for (int index_w=0; index_w<W_UNROLL_TEST; index_w++) {
                for (int index_i_2=0; index_i_2<2; index_i_2++) {
                    for (int index_w_2=0; index_w_2<2; index_w_2++) {
                        output_value_pos[index_i*2+index_i_2][index_w*2+index_w_2] += __popc(input_s_a[index_i][index_i_2] & weight_pos_s_a[index_w][index_w_2]);
                        output_value_neg[index_i*2+index_i_2][index_w*2+index_w_2] += __popc(input_s_a[index_i][index_i_2] & weight_neg_s_a[index_w][index_w_2]);
                    }
                }
            }
        }
    }
    #endif
}

// Perform gemm computation and OR-n update
__device__ void
compute_shared_gemm_orn(
    int32_t* weight_pos_s,
    int32_t* weight_neg_s,
    int32_t* input_s,
    int (&acc_frag_pos)[I_UNROLL_TEST][W_UNROLL_TEST][WMMA_M_UNROLL][WMMA_N_UNROLL][2],
    int (&acc_frag_neg)[I_UNROLL_TEST][W_UNROLL_TEST][WMMA_M_UNROLL][WMMA_N_UNROLL][2],
    int (&output_compute_pos)[2*I_UNROLL_TEST][2*W_UNROLL_TEST],
    int (&output_compute_neg)[2*I_UNROLL_TEST][2*W_UNROLL_TEST],
    int k_sum,
    int index_k,
    int bin_config
) {
    int id_warp = threadIdx.x % WMMA_INT_WIDTH;
    int index_warp = threadIdx.x / WMMA_INT_WIDTH;
    
    int warp_m = index_warp / N_GENERAL_TEST_WMMA;
    int warp_n = index_warp % N_GENERAL_TEST_WMMA;
    int index_i_c = (id_warp / (WMMA_N_bin_TEST/2));
    int index_w_c = (id_warp % (WMMA_N_bin_TEST/2))*2;
    struct Gemm_Unit_Param gemm_param = {warp_m, warp_n, id_warp, index_i_c, index_w_c};

    int (*or_act)(int, int);

    for (int i=0; i<K_GENERAL_UNIT*WMMA_INT_WIDTH/WMMA_K_bin_TEST; i++) {
        compute_unit_gemm(weight_pos_s, weight_neg_s, input_s, acc_frag_pos, acc_frag_neg, output_compute_pos, output_compute_neg, gemm_param, i);
        if (((i+1)*WMMA_K_bin_TEST/WMMA_INT_WIDTH+index_k-K_GENERAL_UNIT)%k_sum==0) {
            #if __CUDA_ARCH__ >= 800
            for (int index_i=0; index_i<I_UNROLL_TEST; index_i++) {
                for (int index_w=0; index_w<W_UNROLL_TEST; index_w++) {
                    for (int m=0; m<WMMA_M_UNROLL; m++) {
                        for (int n=0; n<WMMA_N_UNROLL; n++) {
                            acc_frag_pos[index_i][index_w][m][n][0] = or_act_split_update(acc_frag_pos[index_i][index_w][m][n][0], or_act, bin_config);
                            acc_frag_pos[index_i][index_w][m][n][1] = or_act_split_update(acc_frag_pos[index_i][index_w][m][n][1], or_act, bin_config);
                            acc_frag_neg[index_i][index_w][m][n][0] = or_act_split_update(acc_frag_neg[index_i][index_w][m][n][0], or_act, bin_config);
                            acc_frag_neg[index_i][index_w][m][n][1] = or_act_split_update(acc_frag_neg[index_i][index_w][m][n][1], or_act, bin_config);
                        }
                    }
                }
            }
            #else
            for (int index_i=0; index_i<2*I_UNROLL_TEST; index_i++) {
                for (int index_w=0; index_w<2*W_UNROLL_TEST; index_w++) {
                    output_compute_pos[index_i][index_w] = or_act_split_update(output_compute_pos[index_i][index_w], or_act, bin_config);
                    output_compute_neg[index_i][index_w] = or_act_split_update(output_compute_neg[index_i][index_w], or_act, bin_config);
                }
            }
            #endif
        }
    }
}

// async load data from L2/global to shared
__device__ void
load_data_shared_gemm(
    int32_t* weight_pos_s,
    int32_t* weight_neg_s,
    int32_t* input_s,
    const int32_t* weight_pos_o,
    const int32_t* weight_neg_o,
    const int32_t* input_o,
    int k_offset,
    struct Gemm_Param gemm_param
){
    namespace cg = cooperative_groups;
    auto block = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);
    int num_group32 = M_GENERAL_TEST*N_GENERAL_TEST/32;
    int index_group32 = threadIdx.x/32;
    int index_new = (threadIdx.x%32 + index_group32*4)%32;

    // Load weights
    for (int index_w=0; index_w<2*W_UNROLL_TEST*N_GENERAL_TEST; index_w+=num_group32) {
        cg::memcpy_async(tile32, weight_pos_s+(index_w+index_group32)*K_GENERAL_UNIT_STORE, weight_pos_o+(index_w+index_group32)*gemm_param.k_total+k_offset, sizeof(int32_t)*32);
        cg::memcpy_async(tile32, weight_neg_s+(index_w+index_group32)*K_GENERAL_UNIT_STORE, weight_neg_o+(index_w+index_group32)*gemm_param.k_total+k_offset, sizeof(int32_t)*32);
    }
    // Load inputs
    for (int index_i=0; index_i<2*I_UNROLL_TEST*M_GENERAL_TEST; index_i+=num_group32) {
        cg::memcpy_async(tile32, input_s+(index_i+index_group32)*K_GENERAL_UNIT_STORE, input_o+(index_i+index_group32)*gemm_param.k_total+k_offset, sizeof(int32_t)*32);
    }
}

// Load data + compute + store data for OR-n gemm
__global__ void 
__launch_bounds__(int(M_GENERAL_TEST*N_GENERAL_TEST), 2)
stream_compute_matmul_orn(
    const int32_t* __restrict__ input_stream,
    const int32_t* __restrict__ weight_pos_stream,
    const int32_t* __restrict__ weight_neg_stream,
    int16_t* __restrict__ output_pos_stream,
    int16_t* __restrict__ output_neg_stream,
    const int k_total,
    const int k_compute,
    const int m_total,
    const int n_total,
    const int k_sum,
    const int bin_config,
    const bool init) {
    int index_gen = threadIdx.x;
    int stride_gen = blockDim.x;
    int index_block = blockIdx.x;
    int stride_block = gridDim.x;
    namespace cg = cooperative_groups;
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    cg::thread_block_tile<4> tile4 = cg::tiled_partition<4>(block);
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);

    int (*or_act)(int, int) = &or_act_n;

    // Simple thread block swizzle to improve L2 hit rate
    int m_sws = (m_total + M_SW-1)/M_SW;
    int n_sws = (n_total + N_SW-1)/N_SW;
    int index_sw_o = index_block / (M_SW*N_SW);
    int index_sw_i = index_block % (M_SW*N_SW);
    int index_sw_m = index_sw_o / n_sws;
    int index_sw_n = index_sw_o % n_sws;
    int index_i_m = index_sw_i / N_SW;
    int index_i_n = index_sw_i % N_SW;
    int index_block_m = index_sw_m*M_SW + index_i_m;
    int index_block_n = index_sw_n*N_SW + index_i_n;
    if ((index_block_m>=m_total) || (index_block_n>=n_total)) return;
    
    int index_warp = index_gen / WMMA_INT_WIDTH;
    int id_warp = index_gen % WMMA_INT_WIDTH;
    int warp_m = index_warp / N_GENERAL_TEST_WMMA;
    int warp_n = index_warp % N_GENERAL_TEST_WMMA;
    int index_i_c = (id_warp / (WMMA_N_bin_TEST/2));
    int index_w_c = (id_warp % (WMMA_N_bin_TEST/2))*2;

    __shared__ int32_t weight_pos_s_0[2*W_UNROLL_TEST*N_GENERAL_TEST*K_GENERAL_UNIT_STORE];
    __shared__ int32_t weight_neg_s_0[2*W_UNROLL_TEST*N_GENERAL_TEST*K_GENERAL_UNIT_STORE];
    __shared__ int32_t input_s_0[2*I_UNROLL_TEST*M_GENERAL_TEST*K_GENERAL_UNIT_STORE];
    __shared__ int32_t weight_pos_s_1[2*W_UNROLL_TEST*N_GENERAL_TEST*K_GENERAL_UNIT_STORE];
    __shared__ int32_t weight_neg_s_1[2*W_UNROLL_TEST*N_GENERAL_TEST*K_GENERAL_UNIT_STORE];
    __shared__ int32_t input_s_1[2*I_UNROLL_TEST*M_GENERAL_TEST*K_GENERAL_UNIT_STORE];

    const int32_t* input_block = input_stream + index_block_m*2*I_UNROLL_TEST*M_GENERAL_TEST*k_total;
    const int32_t* weight_pos_block = weight_pos_stream + index_block_n*2*W_UNROLL_TEST*N_GENERAL_TEST*k_total;
    const int32_t* weight_neg_block = weight_neg_stream + index_block_n*2*W_UNROLL_TEST*N_GENERAL_TEST*k_total;

    int16_t* output_pos_block = output_pos_stream + index_block_n*2*W_UNROLL_TEST*N_GENERAL_TEST*m_total*2*I_UNROLL_TEST*M_GENERAL_TEST + index_block_m*2*I_UNROLL_TEST*M_GENERAL_TEST;
    int16_t* output_neg_block = output_neg_stream + index_block_n*2*W_UNROLL_TEST*N_GENERAL_TEST*m_total*2*I_UNROLL_TEST*M_GENERAL_TEST + index_block_m*2*I_UNROLL_TEST*M_GENERAL_TEST;
    int16_t* output_pos_warp = output_pos_block + warp_n*W_UNROLL_TEST*WMMA_N_bin_TEST*m_total*2*I_UNROLL_TEST*M_GENERAL_TEST + warp_m*I_UNROLL_TEST*WMMA_M_bin_TEST;
    int16_t* output_neg_warp = output_neg_block + warp_n*W_UNROLL_TEST*WMMA_N_bin_TEST*m_total*2*I_UNROLL_TEST*M_GENERAL_TEST + warp_m*I_UNROLL_TEST*WMMA_M_bin_TEST;
    int16_t output_value_pos[2*I_UNROLL_TEST][2*W_UNROLL_TEST];
    int16_t output_value_neg[2*I_UNROLL_TEST][2*W_UNROLL_TEST];

    if (init) {
        for (int index_i=0; index_i<I_UNROLL_TEST; index_i++) {
            for (int index_w=0; index_w<W_UNROLL_TEST; index_w++) {
                for (int index_i_2=0; index_i_2<2; index_i_2++) {
                    for (int index_w_2=0; index_w_2<2; index_w_2++) {
                        output_value_pos[index_i*2+index_i_2][index_w*2+index_w_2] = 0;
                        output_value_neg[index_i*2+index_i_2][index_w*2+index_w_2] = 0;
                    }
                }
            }
        }
    }
    else {
        for (int index_i=0; index_i<I_UNROLL_TEST; index_i++) {
            for (int index_w=0; index_w<W_UNROLL_TEST; index_w++) {
                for (int index_i_2=0; index_i_2<2; index_i_2++) {
                    for (int index_w_2=0; index_w_2<2; index_w_2++) {
                        output_value_pos[index_i*2+index_i_2][index_w*2+index_w_2] = output_pos_warp[
                                                                        (index_w*WMMA_N_bin_TEST+index_w_2+index_w_c)*m_total*2*I_UNROLL_TEST*M_GENERAL_TEST
                                                                        +index_i*WMMA_M_bin_TEST
                                                                        +index_i_2*8+index_i_c];
                        output_value_neg[index_i*2+index_i_2][index_w*2+index_w_2] = output_neg_warp[
                                                                        (index_w*WMMA_N_bin_TEST+index_w_2+index_w_c)*m_total*2*I_UNROLL_TEST*M_GENERAL_TEST
                                                                        +index_i*WMMA_M_bin_TEST
                                                                        +index_i_2*8+index_i_c];
                    }
                }
            }
        }
    }

    int acc_frag_pos[I_UNROLL_TEST][W_UNROLL_TEST][WMMA_M_UNROLL][WMMA_N_UNROLL][2];
    int acc_frag_neg[I_UNROLL_TEST][W_UNROLL_TEST][WMMA_M_UNROLL][WMMA_N_UNROLL][2];
    int32_t output_compute_pos[2*I_UNROLL_TEST][2*W_UNROLL_TEST];
    int32_t output_compute_neg[2*I_UNROLL_TEST][2*W_UNROLL_TEST];
    #if __CUDA_ARCH__ >= 800

    for (int i=0; i<I_UNROLL_TEST; i++) {
        for (int j=0; j<W_UNROLL_TEST; j++) {
            for (int m=0; m<WMMA_M_UNROLL; m++) {
                for (int n=0; n<WMMA_N_UNROLL; n++) {
                    acc_frag_pos[i][j][m][n][0] = 0;
                    acc_frag_pos[i][j][m][n][1] = 0;
                    acc_frag_neg[i][j][m][n][0] = 0;
                    acc_frag_neg[i][j][m][n][1] = 0;
                }
            }
        }
    }
    #else
    for (int index_i=0; index_i<2*I_UNROLL_TEST; index_i++) {
        for (int index_w=0; index_w<2*W_UNROLL_TEST; index_w++) {
            output_compute_pos[index_i][index_w] = 0;
            output_compute_neg[index_i][index_w] = 0;
        }
    }
    #endif
    struct Gemm_Param gemm_param = {k_total};
    load_data_shared_gemm(weight_pos_s_0, weight_neg_s_0, input_s_0, weight_pos_block, weight_neg_block, input_block, 0, gemm_param);
    // synchronize
    cg::wait(block);

    for (int index_k=K_GENERAL_UNIT; index_k<k_compute; index_k+=K_GENERAL_UNIT) {
        int32_t *weight_pos_s_c, *weight_neg_s_c, *input_s_c, *weight_pos_s_s, *weight_neg_s_s, *input_s_s;
        if ((index_k/K_GENERAL_UNIT)%2==1) {
            weight_pos_s_c = weight_pos_s_0;
            weight_neg_s_c = weight_neg_s_0;
            weight_pos_s_s = weight_pos_s_1;
            weight_neg_s_s = weight_neg_s_1;
            input_s_c = input_s_0;
            input_s_s = input_s_1;
        }
        else {
            weight_pos_s_c = weight_pos_s_1;
            weight_neg_s_c = weight_neg_s_1;
            weight_pos_s_s = weight_pos_s_0;
            weight_neg_s_s = weight_neg_s_0;
            input_s_c = input_s_1;
            input_s_s = input_s_0;
        }
        // Load data
        load_data_shared_gemm(weight_pos_s_s, weight_neg_s_s, input_s_s, weight_pos_block, weight_neg_block, input_block, index_k, gemm_param);
        // Compute
        compute_shared_gemm_orn(weight_pos_s_c, weight_neg_s_c, input_s_c, acc_frag_pos, acc_frag_neg, output_compute_pos, output_compute_neg, k_sum, index_k, bin_config);
        // synchronize
        cg::wait(block);
    }
    int32_t *weight_pos_s_c, *weight_neg_s_c, *input_s_c;
    if ((k_compute/K_GENERAL_UNIT)%2==1) {
        weight_pos_s_c = weight_pos_s_0;
        weight_neg_s_c = weight_neg_s_0;
        input_s_c = input_s_0;
    }
    else {
        weight_pos_s_c = weight_pos_s_1;
        weight_neg_s_c = weight_neg_s_1;
        input_s_c = input_s_1;
    }
    // Compute
    compute_shared_gemm_orn(weight_pos_s_c, weight_neg_s_c, input_s_c, acc_frag_pos, acc_frag_neg, output_compute_pos, output_compute_neg, k_sum, k_compute, bin_config);
    // Store output
    #if __CUDA_ARCH__ >= 800
    for (int index_i=0; index_i<I_UNROLL_TEST; index_i++) {
        for (int index_w=0; index_w<W_UNROLL_TEST; index_w++) {
            // stmatrix does not work in this case (because it assumes 16-bit elements). Store directly to register
            for (int index_i_2=0; index_i_2<2; index_i_2++) {
                for (int index_w_2=0; index_w_2<2; index_w_2++) {
                    output_value_pos[index_i*2+index_i_2][index_w*2+index_w_2] += acc_frag_pos[index_i][index_w][index_i_2][0][index_w_2]>>16;
                    output_value_neg[index_i*2+index_i_2][index_w*2+index_w_2] += acc_frag_neg[index_i][index_w][index_i_2][0][index_w_2]>>16;
                }
            }
        }
    }
    #else
    for (int index_i=0; index_i<2*I_UNROLL_TEST; index_i++) {
        for (int index_w=0; index_w<2*W_UNROLL_TEST; index_w++) {
            output_value_pos[index_i][index_w] += output_compute_pos[index_i][index_w]>>16;
            output_value_neg[index_i][index_w] += output_compute_neg[index_i][index_w]>>16;
        }
    }
    #endif
    for (int index_i=0; index_i<I_UNROLL_TEST; index_i++) {
        for (int index_w=0; index_w<W_UNROLL_TEST; index_w++) {
            for (int index_i_2=0; index_i_2<2; index_i_2++) {
                for (int index_w_2=0; index_w_2<2; index_w_2++) {
                    output_pos_warp[(index_w*WMMA_N_bin_TEST+index_w_2+index_w_c)*m_total*2*I_UNROLL_TEST*M_GENERAL_TEST
                                    +index_i*WMMA_M_bin_TEST
                                    +index_i_c+index_i_2*8] = int16_t(output_value_pos[index_i*2+index_i_2][index_w*2+index_w_2]);
                    output_neg_warp[(index_w*WMMA_N_bin_TEST+index_w_2+index_w_c)*m_total*2*I_UNROLL_TEST*M_GENERAL_TEST
                                    +index_i*WMMA_M_bin_TEST
                                    +index_i_c+index_i_2*8] = int16_t(output_value_neg[index_i*2+index_i_2][index_w*2+index_w_2]);
                }
            }
        }
    }
}