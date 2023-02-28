#include "sc_device.cuh"
#include "sc_cuda_conv.cuh"

using namespace nvcuda;

__device__ curandStatePhilox4_32_10_t *rand_states_conv;
bool rand_init_conv = false;

/*
 * Accelerated GPU implementation. Kernel functions
 */

// Old version of OR-n stream computation
const int I_UNROLL=2;
const int W_UNROLL=2;
const int I_UNROLL_PAD=2;
const int W_OUT_BLOCK=4*I_UNROLL_PAD;
const int H_OUT_BLOCK=8;

const int W_OUT_BLOCK_33 = 4;
const int H_OUT_BLOCK_33 = 4;

__global__ void
__launch_bounds__(M_GENERAL*N_GENERAL, 2)
stream_compute_or_general(
    const int32_t* __restrict__ input_stream,
    const int32_t* __restrict__ weight_pos_stream,
    const int32_t* __restrict__ weight_neg_stream,
    int16_t* __restrict__ output_pos_stream,
    int16_t* __restrict__ output_neg_stream,
    int stride_w,
    int stride_h,
    int bin_config,
    struct Compute_Param c_param,
    bool channels_last_activation) {
    int index_gen = threadIdx.x;
    int stride_gen = blockDim.x;
    int index_block = blockIdx.x;
    int stride_block = gridDim.x;

    bool or_no = false;
    if (bin_config>=0) bin_config+=1;
    else {
        bin_config = -bin_config+1;
        or_no = true;
    }
    int (*or_act)(int, int);
    if (or_no) or_act = &or_act_no;
    else or_act = &or_act_n;

    int z_packs = (c_param.c_ins + COMPUTE_CINS-1) / COMPUTE_CINS;

    int cout_step = (c_param.c_outs + C_GENERAL-1) / C_GENERAL;
    int i_flatten_step = ((c_param.i_w_ins-c_param.w_w_ins)/stride_w+1)*((c_param.i_h_ins-c_param.w_h_ins)/stride_h+1);

    int inner_size = c_param.w_w_ins*c_param.w_h_ins*z_packs;
    int inner_packs = (inner_size+K_GENERAL-1)/K_GENERAL;

    int index_i_c = index_gen / N_GENERAL;
    int index_w_c = index_gen % N_GENERAL;
    int (*output_ind)(int, int, int);
    if (channels_last_activation) output_ind=&output_ind_nhwc;
    else output_ind=&output_ind_nchw;
    int stride_out = i_flatten_step;
    if (channels_last_activation) stride_out = c_param.c_outs;

    #if __CUDA_ARCH__ >= 800
    wmma::fragment<wmma::matrix_a, WMMA_M_bin, WMMA_N_bin, WMMA_K_bin, wmma::experimental::precision::b1, wmma::row_major> a_frag[I_UNROLL];
    wmma::fragment<wmma::matrix_b, WMMA_M_bin, WMMA_N_bin, WMMA_K_bin, wmma::experimental::precision::b1, wmma::col_major> b_frag_pos[W_UNROLL];
    wmma::fragment<wmma::matrix_b, WMMA_M_bin, WMMA_N_bin, WMMA_K_bin, wmma::experimental::precision::b1, wmma::col_major> b_frag_neg[W_UNROLL];
    
    wmma::fragment<wmma::accumulator, WMMA_M_bin, WMMA_N_bin, WMMA_K_bin, int> acc_frag_pos[I_UNROLL][W_UNROLL];
    wmma::fragment<wmma::accumulator, WMMA_M_bin, WMMA_N_bin, WMMA_K_bin, int> acc_frag_neg[I_UNROLL][W_UNROLL];

    int index_warp = index_gen / WMMA_INT_WIDTH;
    int warp_m = index_warp / N_GENERAL_WMMA;
    int warp_n = index_warp % N_GENERAL_WMMA;
    __shared__ int output_s [M_GENERAL*N_GENERAL_STORE];
    #endif
    
    __shared__ uint weight_pos_s [2*W_UNROLL*N_GENERAL*K_GENERAL_STORE];
    __shared__ uint weight_neg_s [2*W_UNROLL*N_GENERAL*K_GENERAL_STORE];
    __shared__ uint input_s [I_UNROLL*M_GENERAL*K_GENERAL_STORE];
    for (int block=index_block; block<c_param.batches*cout_step; block+=stride_block) {
        int batch = block/cout_step;
        int cout_offset = (block%cout_step)*C_GENERAL;
        const int32_t* input_stream_batch = input_stream + batch*c_param.i_w_ins*c_param.i_h_ins*z_packs;
        int16_t* output_pos_batch = output_pos_stream + batch*c_param.c_outs*i_flatten_step;
        int16_t* output_neg_batch = output_neg_stream + batch*c_param.c_outs*i_flatten_step;

        for (int cin=0; cin<i_flatten_step; cin+=I_UNROLL*M_GENERAL) {
            for (int cout=0; cout<C_GENERAL; cout+=2*W_UNROLL*N_GENERAL) {
                int cout_c[2*W_UNROLL];
                for (int index_w=0; index_w<2*W_UNROLL; index_w++) cout_c[index_w]=cout_offset+cout+(index_w_c*2*W_UNROLL+index_w);
                int cin_c[I_UNROLL];
                for (int index_i=0; index_i<I_UNROLL; index_i++) cin_c[index_i] = cin+(index_i_c*I_UNROLL+index_i);
                bool valid[I_UNROLL][2*W_UNROLL];
                int index[I_UNROLL][2*W_UNROLL];
                int16_t *output_pos[I_UNROLL][2*W_UNROLL];
                int16_t *output_neg[I_UNROLL][2*W_UNROLL];
                uint output_value_pos[I_UNROLL][2*W_UNROLL];
                uint output_value_neg[I_UNROLL][2*W_UNROLL];
                for (int index_i=0; index_i<I_UNROLL; index_i++) {
                    for (int index_w=0; index_w<2*W_UNROLL; index_w++) {
                        valid[index_i][index_w] = (cout_c[index_w]<c_param.c_outs)  & (cin_c[index_i]<i_flatten_step) & (cout+(index_w_c*2*W_UNROLL+index_w)<C_GENERAL);
                        index[index_i][index_w] = output_ind(cout_c[index_w], cin_c[index_i], stride_out);
                        output_pos[index_i][index_w] = output_pos_batch + index[index_i][index_w];
                        output_neg[index_i][index_w] = output_neg_batch + index[index_i][index_w];

                        output_value_pos[index_i][index_w] = 0;
                        output_value_neg[index_i][index_w] = 0;
                    }
                }
                for (int bit=0; bit<c_param.bit_length; bit++) {
                    #if __CUDA_ARCH__ >= 800
                    for (int i=0; i<I_UNROLL; i++) {
                        for (int j=0; j<W_UNROLL; j++) {
                            wmma::fill_fragment(acc_frag_pos[i][j], int(0));
                            wmma::fill_fragment(acc_frag_neg[i][j], int(0));
                        }
                    }
                    #endif
                    const int32_t* input_bit = input_stream_batch + bit*c_param.batches*c_param.i_w_ins*c_param.i_h_ins*z_packs;
                    const int32_t* weight_pos_bit = weight_pos_stream + bit*c_param.c_outs*c_param.w_w_ins*c_param.w_h_ins*z_packs;
                    const int32_t* weight_neg_bit = weight_neg_stream + bit*c_param.c_outs*c_param.w_w_ins*c_param.w_h_ins*z_packs;

                    for (int inner_i=0; inner_i<inner_packs; inner_i++) {
                        int inner = inner_i*K_GENERAL;
                        // Load weights
                        for (int index_w=index_gen; index_w<2*W_UNROLL*N_GENERAL*K_GENERAL; index_w+=stride_gen) {
                            int n = index_w / K_GENERAL;
                            int k = index_w % K_GENERAL;
                            if ((inner+k<inner_size) & (cout_offset+cout+n<c_param.c_outs) & (cout+n<C_GENERAL)) {
                                int weight_index = (cout_offset+cout+n) * c_param.w_w_ins * c_param.w_h_ins * z_packs + (inner+k);
                                weight_pos_s[n*K_GENERAL_STORE+k] = weight_pos_bit[weight_index];
                                weight_neg_s[n*K_GENERAL_STORE+k] = weight_neg_bit[weight_index];
                            }
                            else {
                                weight_pos_s[n*K_GENERAL_STORE+k] = 0;
                                weight_neg_s[n*K_GENERAL_STORE+k] = 0;
                            }
                        }
                        // Load inputs
                        for (int index_i=index_gen; index_i<I_UNROLL*M_GENERAL*K_GENERAL; index_i+=stride_gen) {
                            int m = index_i / K_GENERAL;
                            int k = index_i % K_GENERAL;
                            int w_in_i_c = ((cin+m) / ((c_param.i_h_ins - c_param.w_h_ins)/stride_h+1))*stride_w;
                            int h_in_i_c = ((cin+m) % ((c_param.i_h_ins - c_param.w_h_ins)/stride_h+1))*stride_h;
                            int z_pack_c = (inner+k) % z_packs;
                            int w_in_w_c = (inner+k) / (c_param.w_h_ins*z_packs);
                            int h_in_w_c = ((inner+k) % (c_param.w_h_ins*z_packs)) / z_packs;
                            if ((inner+k<inner_size) & (cin+m<i_flatten_step)) {
                                int input_index = (w_in_i_c + w_in_w_c) * c_param.i_h_ins * z_packs
                                                +(h_in_i_c + h_in_w_c) * z_packs
                                                +z_pack_c;
                                input_s[m*K_GENERAL_STORE+k] = input_bit[input_index];
                            }
                            else input_s[m*K_GENERAL_STORE+k] = 0;
                        }
                        __syncthreads();
                        
                        // Compute
                        #if __CUDA_ARCH__ >= 800
                        for (int i=0; i<K_GENERAL*WMMA_INT_WIDTH/WMMA_K_bin; i++) {
                            #pragma unroll
                            for (int index_i=0; index_i<I_UNROLL; index_i++) wmma::load_matrix_sync(a_frag[index_i], input_s+(warp_m*I_UNROLL*WMMA_M_bin+index_i)*K_GENERAL_STORE+i*WMMA_K_bin/WMMA_INT_WIDTH, I_UNROLL*K_GENERAL_STORE*WMMA_INT_WIDTH);
                            #pragma unroll
                            for (int index_w=0; index_w<W_UNROLL; index_w++) {
                                wmma::load_matrix_sync(b_frag_pos[index_w], weight_pos_s+(warp_n*W_UNROLL*WMMA_N_bin+index_w)*K_GENERAL_STORE+i*WMMA_K_bin/WMMA_INT_WIDTH, W_UNROLL*K_GENERAL_STORE*WMMA_INT_WIDTH);
                                wmma::load_matrix_sync(b_frag_neg[index_w], weight_neg_s+(warp_n*W_UNROLL*WMMA_N_bin+index_w)*K_GENERAL_STORE+i*WMMA_K_bin/WMMA_INT_WIDTH, W_UNROLL*K_GENERAL_STORE*WMMA_INT_WIDTH);
                            }

                            #pragma unroll
                            for (int index_i=0; index_i<I_UNROLL; index_i++) {
                                #pragma unroll
                                for (int index_w=0; index_w<W_UNROLL; index_w++) {
                                    wmma::bmma_sync(acc_frag_pos[index_i][index_w], a_frag[index_i], b_frag_pos[index_w], acc_frag_pos[index_i][index_w], wmma::experimental::bmmaBitOpAND, wmma::experimental::bmmaAccumulateOpPOPC);
                                    wmma::bmma_sync(acc_frag_neg[index_i][index_w], a_frag[index_i], b_frag_neg[index_w], acc_frag_neg[index_i][index_w], wmma::experimental::bmmaBitOpAND, wmma::experimental::bmmaAccumulateOpPOPC);
                                }
                            }
                        }
                        #else
                        for (int i=0; i<K_GENERAL; i++) {
                            int32_t input_s_a[I_UNROLL];
                            int32_t weight_pos_s_a[2*W_UNROLL];
                            int32_t weight_neg_s_a[2*W_UNROLL];
                            for (int index_i=0; index_i<I_UNROLL; index_i++) input_s_a[index_i] = input_s[(index_i_c*I_UNROLL+index_i)*K_GENERAL_STORE+i];
                            for (int index_w=0; index_w<2*W_UNROLL; index_w++) {
                                weight_pos_s_a[index_w] = weight_pos_s[(index_w_c*2*W_UNROLL+index_w)*K_GENERAL_STORE+i];
                                weight_neg_s_a[index_w] = weight_neg_s[(index_w_c*2*W_UNROLL+index_w)*K_GENERAL_STORE+i];
                            }
                            for (int index_i=0; index_i<I_UNROLL; index_i++) {
                                for (int index_w=0; index_w<2*W_UNROLL; index_w++) {
                                    output_value_pos[index_i][index_w] += __popc(input_s_a[index_i] & weight_pos_s_a[index_w]);
                                    output_value_neg[index_i][index_w] += __popc(input_s_a[index_i] & weight_neg_s_a[index_w]);
                                }
                            }
                        }
                        #endif
                        __syncthreads();
                    }
                    #if __CUDA_ARCH__ >= 800
                    for (int index_i=0; index_i<I_UNROLL; index_i++) {
                        for (int index_w=0; index_w<W_UNROLL; index_w++) {
                            wmma::store_matrix_sync(output_s + warp_m*WMMA_M_bin*N_GENERAL_STORE + warp_n*WMMA_N_bin, acc_frag_pos[index_i][index_w], N_GENERAL_STORE, wmma::mem_row_major);
                            __syncthreads();
                            output_value_pos[index_i][index_w+W_UNROLL*0] += output_s[index_i_c*(N_GENERAL_STORE)+(index_w_c*2+0)];
                            output_value_pos[index_i][index_w+W_UNROLL*1] += output_s[index_i_c*(N_GENERAL_STORE)+(index_w_c*2+1)];
                            __syncthreads();
                            wmma::store_matrix_sync(output_s + warp_m*WMMA_M_bin*N_GENERAL_STORE + warp_n*WMMA_N_bin, acc_frag_neg[index_i][index_w], N_GENERAL_STORE, wmma::mem_row_major);
                            __syncthreads();
                            output_value_neg[index_i][index_w+W_UNROLL*0] += output_s[index_i_c*(N_GENERAL_STORE)+(index_w_c*2+0)];
                            output_value_neg[index_i][index_w+W_UNROLL*1] += output_s[index_i_c*(N_GENERAL_STORE)+(index_w_c*2+1)];
                            __syncthreads();
                        }
                    }
                    #endif
                    for (int index_i=0; index_i<I_UNROLL; index_i++) {
                        for (int index_w=0; index_w<2*W_UNROLL; index_w++) {
                            output_value_pos[index_i][index_w] = or_act_split_update(output_value_pos[index_i][index_w], or_act, bin_config);
                            output_value_neg[index_i][index_w] = or_act_split_update(output_value_neg[index_i][index_w], or_act, bin_config);
                        }
                    }
                }
                for (int index_i=0; index_i<I_UNROLL; index_i++) {
                    for (int index_w=0; index_w<2*W_UNROLL; index_w++) {
                        if (valid[index_i][index_w]) {
                            *output_pos[index_i][index_w] = output_value_pos[index_i][index_w]>>16;
                            *output_neg[index_i][index_w] = output_value_neg[index_i][index_w]>>16;
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
}

// Old version of OR-n convolution. Performance is slightly lower than the gemm version
std::vector<torch::Tensor> conv2d_generic_cuda_general(torch::Tensor input,
        torch::Tensor weight_pos,
        torch::Tensor weight_neg,
        at::IntArrayRef stride,
        at::IntArrayRef prog_load,
        int bit_length,
        int lfsr_length,
        int z_units,
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
    auto input_stride = input.strides();
    auto weight_stride = weight_pos.strides();
    bool channels_last_activation = false;
    bool channels_last_weight = false;
    if ((input_stride[1]==1) && (input_stride[1]*input_stride[2]*input_stride[3]!=1)) channels_last_activation = true;
    if ((weight_stride[1]==1) && (weight_stride[1]*weight_stride[2]*weight_stride[3]!=1)) channels_last_weight = true;
    // printf("channels_last_activation, %d\n", channels_last_activation);

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

    int32_t *weight_pos_stream, *weight_neg_stream, *input_stream;
    cudaMalloc(&weight_pos_stream, compute_length*weight_size[0]*weight_size[2]*weight_size[3]*z_packs*sizeof(int32_t));
    cudaMalloc(&weight_neg_stream, compute_length*weight_size[0]*weight_size[2]*weight_size[3]*z_packs*sizeof(int32_t));
    cudaMalloc(&input_stream, compute_length*input_size[0]*input_size[2]*input_size[3]*z_packs*sizeof(int32_t));

    const int threads = THREADS_GENERAL;

    bool gen_mult = 0;
    // if (bin_config>1000) gen_mult=1;
    stream_generation_or_general <<<10000, threads>>>(
        weight_pos.data_ptr<int32_t>(),
        weight_neg.data_ptr<int32_t>(),
        weight_pos_stream,
        weight_neg_stream,
        bit_length,
        lfsr_length,
        gen_config,
        gen_mult,
        weight_size[0],
        weight_size[1],
        weight_size[2],
        weight_size[3],
        prog_load[0],
        prog_load[1],
        prog_load[2],
        channels_last_weight,
        xnor);
    activation_generation_or_general <<<10000, threads>>>(
        input.data_ptr<int32_t>(),
        input_stream,
        bit_length,
        lfsr_length,
        gen_config,
        gen_mult,
        input_size[0],
        input_size[1],
        input_size[2],
        input_size[3],
        prog_load[0],
        prog_load[1],
        prog_load[2],
        channels_last_activation,
        xnor);

    torch::MemoryFormat output_format;
    if (channels_last_activation) output_format=torch::MemoryFormat::ChannelsLast;
    else output_format=torch::MemoryFormat::Contiguous;
    auto output_tensor_pos = torch::zeros({input_size[0], weight_size[0], (input_size[2]-weight_size[2])/stride[0]+1, (input_size[3]-weight_size[3])/stride[1]+1}, at::TensorOptions().dtype(torch::kInt16).device(device)).to(output_format);
    auto output_tensor_neg = torch::zeros({input_size[0], weight_size[0], (input_size[2]-weight_size[2])/stride[0]+1, (input_size[3]-weight_size[3])/stride[1]+1}, at::TensorOptions().dtype(torch::kInt16).device(device)).to(output_format);
    struct Compute_Param c_param = {int(compute_length), int(0), int(input_size[0]), int(input_size[1]), int(input_size[2]), int(input_size[3]), int(weight_size[0]), int(weight_size[2]), int(weight_size[3])};

    // printf("Here\n");
    stream_compute_or_general <<<10000, M_GENERAL*N_GENERAL>>> (
        input_stream,
        weight_pos_stream,
        weight_neg_stream,
        output_tensor_pos.data_ptr<int16_t>(),
        output_tensor_neg.data_ptr<int16_t>(),
        stride[0],
        stride[1],
        bin_config,
        c_param,
        channels_last_activation);


    cudaFree(input_stream);
    cudaFree(weight_pos_stream);
    cudaFree(weight_neg_stream);
    return {output_tensor_pos, output_tensor_neg};
}

// Transform 4d input and weight tensors into 2d matrix for gemm computation
__global__ void
prepare_data_gemm_conv(
    const int32_t* __restrict__ input_stream,
    const int32_t* __restrict__ weight_pos_stream,
    const int32_t* __restrict__ weight_neg_stream,
    int32_t* __restrict__ input_temp,
    int32_t* __restrict__ weight_pos_temp,
    int32_t* __restrict__ weight_neg_temp,
    struct Compute_Param c_param,
    const int stride_w,
    const int stride_h,
    const int m_offset,
    const int n_offset,
    const int k_offset,
    const int m_total,
    const int n_total,
    const int k_total,
    const int k_store
) {
    int index_gen = threadIdx.x;
    int stride_gen = blockDim.x;
    int index_block = blockIdx.x;
    int stride_block = gridDim.x;

    int index_warp = index_gen/32;
    int num_warp = stride_gen/32;
    int id_warp = index_gen%32;
    int z_packs = (c_param.c_ins+31)/32;
    int inner_size = c_param.w_w_ins*c_param.w_h_ins*z_packs;
    int k_unit = (inner_size+7)/8;
    int k_unit_total = k_unit*8;
    int k_unit_compute = (K_UNIT_BLOCK/k_unit_total)*k_unit_total;
    
    //Prepare inputs
    int o_w_limit = (c_param.i_w_ins-c_param.w_w_ins)/stride_w+1;
    int o_h_limit = (c_param.i_h_ins-c_param.w_h_ins)/stride_h+1;

    for (int m_c=index_block; m_c<m_total; m_c+=stride_block) {
        // K-dimension shuffling to prevent bank conflict during gemm
        int id_new = (id_warp + 4*(m_c%8))%32;
        int32_t* input_temp_m = input_temp + m_c*k_store;
        int m_c_o = m_c+m_offset;
        int batch_c = m_c_o/(o_w_limit*o_h_limit);
        int o_w_c = ((m_c_o%(o_w_limit*o_h_limit))/o_h_limit);
        int o_h_c = m_c_o%o_h_limit;
        if (batch_c<c_param.batches) {
            const int32_t* input_stream_m = 
                input_stream+
                batch_c*c_param.i_w_ins*c_param.i_h_ins*z_packs
                +o_w_c*stride_w*c_param.i_h_ins*z_packs
                +o_h_c*stride_h*z_packs;
            for (int k_c=0; k_c<k_store; k_c+=stride_gen) {
                int k_c_s = k_c+32*index_warp;
                int k_c_o = k_c_s+k_offset+id_warp;
                int bit = k_c_o/k_unit_total;
                int inner_unit = k_c_o%k_unit_total;
                int input_c = 0;
                if ((bit<c_param.bit_length) & (inner_unit<inner_size) & (k_c_s+id_warp<k_total)) {
                    int w_w_c = inner_unit/(c_param.w_h_ins*z_packs);
                    int w_h_c = (inner_unit%(c_param.w_h_ins*z_packs))/z_packs;
                    int z_c = inner_unit%z_packs;
                    input_c = input_stream_m[
                        bit*c_param.batches*c_param.i_w_ins*c_param.i_h_ins*z_packs+
                        w_w_c*c_param.i_h_ins*z_packs+
                        w_h_c*z_packs+
                        z_c
                    ];
                }
                if ((k_c_s+id_new)<k_store) input_temp_m[k_c_s+id_new]=input_c;
            }
        }
    }
    //Prepare weights
    for (int n_c=index_block; n_c<n_total; n_c+=stride_block) {
        int id_new = (id_warp + 4*(n_c%8))%32;
        int32_t* weight_pos_temp_n = weight_pos_temp + n_c*k_store;
        int32_t* weight_neg_temp_n = weight_neg_temp + n_c*k_store;
        int n_c_o = n_c+n_offset;
        if (n_c_o<c_param.c_outs) {
            const int32_t* weight_pos_stream_n = 
                weight_pos_stream+
                n_c_o*inner_size;
            const int32_t* weight_neg_stream_n = 
                weight_neg_stream+
                n_c_o*inner_size;
            for (int k_c=0; k_c<k_store; k_c+=stride_gen) {
                int k_c_s = k_c+32*index_warp;
                int k_c_o = k_c_s+k_offset+id_warp;
                int bit = k_c_o/k_unit_total;
                int inner_unit = k_c_o%k_unit_total;
                int weight_pos_c = 0;
                int weight_neg_c = 0;
                if ((bit<c_param.bit_length) & (inner_unit<inner_size) & (k_c_s+id_warp<k_total)) {
                    int w_w_c = inner_unit/(c_param.w_h_ins*z_packs);
                    int w_h_c = (inner_unit%(c_param.w_h_ins*z_packs))/z_packs;
                    int z_c = inner_unit%z_packs;
                    weight_pos_c = weight_pos_stream_n[
                        bit*c_param.c_outs*c_param.w_w_ins*c_param.w_h_ins*z_packs+
                        w_w_c*c_param.w_h_ins*z_packs+
                        w_h_c*z_packs+
                        z_c
                    ];
                    weight_neg_c = weight_neg_stream_n[
                        bit*c_param.c_outs*c_param.w_w_ins*c_param.w_h_ins*z_packs+
                        w_w_c*c_param.w_h_ins*z_packs+
                        w_h_c*z_packs+
                        z_c
                    ];
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
save_data_gemm_conv(
    const int16_t* __restrict__ output_pos_temp,
    const int16_t* __restrict__ output_neg_temp,
    int16_t* __restrict__ output_pos,
    int16_t* __restrict__ output_neg,
    struct Compute_Param c_param,
    const int stride_w,
    const int stride_h,
    const int m_offset,
    const int n_offset,
    const int m_total,
    const int n_total,
    const bool channels_last
) {
    int index_gen = threadIdx.x;
    int stride_gen = blockDim.x;
    int index_block = blockIdx.x;
    int stride_block = gridDim.x;
    
    int index_warp = index_gen/32;
    int num_warp = stride_gen/32;
    int id_warp = index_gen%32;

    int o_w_limit = (c_param.i_w_ins-c_param.w_w_ins)/stride_w+1;
    int o_h_limit = (c_param.i_h_ins-c_param.w_h_ins)/stride_h+1;

    for (int n_c=index_block; n_c<n_total; n_c+=stride_block) {
        const int16_t* output_pos_temp_n = output_pos_temp + n_c*m_total;
        const int16_t* output_neg_temp_n = output_neg_temp + n_c*m_total;
        int n_c_o = n_c+n_offset;
        if (n_c_o<c_param.c_outs) {
            int16_t *output_pos_n, *output_neg_n;
            if (channels_last) {
                output_pos_n = output_pos + n_c_o;
                output_neg_n = output_neg + n_c_o;
            }
            else {
                output_pos_n = output_pos + n_c_o*o_w_limit*o_h_limit;
                output_neg_n = output_neg + n_c_o*o_w_limit*o_h_limit;
            }
            for (int m_c=index_gen; m_c<m_total; m_c+=stride_gen) {
                const int16_t* output_pos_temp_m = output_pos_temp_n + m_c;
                const int16_t* output_neg_temp_m = output_neg_temp_n + m_c;
                int m_c_o = m_c+m_offset;
                int batch_c = m_c_o/(o_w_limit*o_h_limit);
                int o_w_c = ((m_c_o%(o_w_limit*o_h_limit))/o_h_limit);
                int o_h_c = m_c_o%o_h_limit;
                if (batch_c<c_param.batches) {
                    int16_t *output_pos_m, *output_neg_m;
                    if (channels_last) {
                        output_pos_m = output_pos_n+
                                       batch_c*o_w_limit*o_h_limit*c_param.c_outs+
                                       o_w_c*o_h_limit*c_param.c_outs+
                                       o_h_c*c_param.c_outs;
                        output_neg_m = output_neg_n+
                                       batch_c*o_w_limit*o_h_limit*c_param.c_outs+
                                       o_w_c*o_h_limit*c_param.c_outs+
                                       o_h_c*c_param.c_outs;
                    }
                    else {
                        output_pos_m = output_pos_n+
                                       batch_c*c_param.c_outs*o_w_limit*o_h_limit+
                                       o_w_c*o_h_limit+
                                       o_h_c;
                        output_neg_m = output_neg_n+
                                       batch_c*c_param.c_outs*o_w_limit*o_h_limit+
                                       o_w_c*o_h_limit+
                                       o_h_c;
                    }
                    *output_pos_m = *output_pos_temp_m;
                    *output_neg_m = *output_neg_temp_m;
                }
            }
        }
    }
}

// Gemm version of OR-n convolution
std::vector<torch::Tensor> conv2d_generic_cuda_general_gemm(
    torch::Tensor input,
    torch::Tensor weight_pos,
    torch::Tensor weight_neg,
    at::IntArrayRef stride,
    at::IntArrayRef prog_load,
    int bit_length,
    int lfsr_length,
    int z_units,
    int bin_config,
    int gen_config,
    bool xnor,
    bool mux
) {
    auto weight_size = weight_pos.sizes();
    auto input_size = input.sizes();
    auto device = weight_pos.device();
    int device_index = device.index();
    cudaSetDevice(int(device_index));
    int z_packs = (weight_size[1] + COMPUTE_CINS-1) / COMPUTE_CINS;
    auto input_stride = input.strides();
    auto weight_stride = weight_pos.strides();
    bool channels_last_activation = false;
    bool channels_last_weight = false;
    if ((input_stride[1]==1) && (input_stride[1]*input_stride[2]*input_stride[3]!=1)) channels_last_activation = true;
    if ((weight_stride[1]==1) && (weight_stride[1]*weight_stride[2]*weight_stride[3]!=1)) channels_last_weight = true;

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

    // Reuse buffers between layers
    if (!Init_Temp) {
        cudaMalloc(&Input_Stream, 1073741824UL);
        cudaMalloc(&Weight_Pos_Stream, 3*3*512*512*4*sizeof(int32_t));
        cudaMalloc(&Weight_Neg_Stream, 3*3*512*512*4*sizeof(int32_t));
        cudaMalloc(&Weight_Pos_Temp, N_UNIT_BLOCK*2*W_UNROLL_TEST*N_GENERAL_TEST*K_UNIT_BLOCK*sizeof(int32_t));
        cudaMalloc(&Weight_Neg_Temp, N_UNIT_BLOCK*2*W_UNROLL_TEST*N_GENERAL_TEST*K_UNIT_BLOCK*sizeof(int32_t));
        cudaMalloc(&Input_Temp, M_UNIT_BLOCK*2*I_UNROLL_TEST*M_GENERAL_TEST*K_UNIT_BLOCK*sizeof(int32_t));
        cudaMalloc(&Output_Pos_Temp, N_UNIT_BLOCK*2*W_UNROLL_TEST*N_GENERAL_TEST*M_UNIT_BLOCK*2*I_UNROLL_TEST*M_GENERAL_TEST*sizeof(int16_t));
        cudaMalloc(&Output_Neg_Temp, N_UNIT_BLOCK*2*W_UNROLL_TEST*N_GENERAL_TEST*M_UNIT_BLOCK*2*I_UNROLL_TEST*M_GENERAL_TEST*sizeof(int16_t));
        Init_Temp = true;
    }
    const int threads = THREADS_GENERAL;

    bool gen_mult = 0;
    stream_generation_or_general <<<10000, threads>>>(
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
        weight_size[2],
        weight_size[3],
        prog_load[0],
        prog_load[1],
        prog_load[2],
        channels_last_weight,
        xnor);
    activation_generation_or_general <<<10000, threads>>>(
        input.data_ptr<int32_t>(),
        Input_Stream,
        bit_length,
        lfsr_length,
        gen_config,
        gen_mult,
        input_size[0],
        input_size[1],
        input_size[2],
        input_size[3],
        prog_load[0],
        prog_load[1],
        prog_load[2],
        channels_last_activation,
        xnor);

    torch::MemoryFormat output_format;
    if (channels_last_activation) output_format=torch::MemoryFormat::ChannelsLast;
    else output_format=torch::MemoryFormat::Contiguous;
    const int o_w_limit = (input_size[2]-weight_size[2])/stride[0]+1;
    const int o_h_limit = (input_size[3]-weight_size[3])/stride[1]+1;

    auto output_tensor_pos = torch::zeros({input_size[0], weight_size[0], o_w_limit, o_h_limit}, at::TensorOptions().dtype(torch::kInt16).device(device)).to(output_format);
    auto output_tensor_neg = torch::zeros({input_size[0], weight_size[0], o_w_limit, o_h_limit}, at::TensorOptions().dtype(torch::kInt16).device(device)).to(output_format);

    // Calculate gemm mapping parameters
    // Limitation: the size of the dot product cannot be larger than 512x32=16384
    int m_total = input_size[0]*((input_size[2]-weight_size[2])/stride[0]+1)*((input_size[3]-weight_size[3])/stride[1]+1);
    int n_total = weight_size[0];
    int m_block_compute = ((m_total+2*I_UNROLL_TEST*M_GENERAL_TEST-1)/(2*I_UNROLL_TEST*M_GENERAL_TEST));
    int n_block_compute = ((n_total+2*W_UNROLL_TEST*N_GENERAL_TEST-1)/(2*W_UNROLL_TEST*N_GENERAL_TEST));
    int m_block_compute_sw = ((m_block_compute+M_SW-1)/M_SW)*M_SW;
    int n_block_compute_sw = ((n_block_compute+N_SW-1)/N_SW)*N_SW;
    int m_total_compute = m_block_compute*(2*I_UNROLL_TEST*M_GENERAL_TEST);
    int n_total_compute = n_block_compute*(2*W_UNROLL_TEST*N_GENERAL_TEST);
    int k_unit = (weight_size[2]*weight_size[3]*z_packs+7)/8;
    int k_unit_total = k_unit*8;
    int k_unit_compute = (K_UNIT_BLOCK/k_unit_total)*k_unit_total;
    // Need to be a multiple of 32
    int k_unit_store = ((k_unit_compute+31)/32)*32;
    int k_total = compute_length*k_unit_total;
    // Reduce the size of memory allocation in case of small problem size
    // Cannot reduce to the actual problem size. Need to be a multiple of a single block
    // 2*I_UNROLL_TEST*M_GENERAL_TEST (64) for inputs, 2*W_UNROLL_TEST*N_GENERAL_TEST (64) for weights
    int m_compute = std::min(m_total_compute, M_UNIT_BLOCK*2*I_UNROLL_TEST*M_GENERAL_TEST);
    int n_compute = std::min(n_total_compute, N_UNIT_BLOCK*2*W_UNROLL_TEST*N_GENERAL_TEST);
    int k_compute = std::min(k_unit_compute, k_total);
    struct Compute_Param c_param = {int(compute_length), int(0), int(input_size[0]), int(input_size[1]), int(input_size[2]), int(input_size[3]), int(weight_size[0]), int(weight_size[2]), int(weight_size[3])};
    
    for (int m_c=0; m_c<m_total; m_c+=m_compute) {
        for (int n_c=0; n_c<n_total; n_c+=n_compute) {
            for (int k_c=0; k_c<k_total; k_c+=k_compute) {
                prepare_data_gemm_conv <<<10000,128>>> (
                    Input_Stream,
                    Weight_Pos_Stream,
                    Weight_Neg_Stream,
                    Input_Temp,
                    Weight_Pos_Temp,
                    Weight_Neg_Temp,
                    c_param,
                    stride[0],
                    stride[1],
                    m_c,
                    n_c,
                    k_c,
                    m_compute,
                    n_compute,
                    k_compute,
                    k_unit_store
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
            save_data_gemm_conv <<<10000, 128>>> (
                Output_Pos_Temp,
                Output_Neg_Temp,
                output_tensor_pos.data_ptr<int16_t>(),
                output_tensor_neg.data_ptr<int16_t>(),
                c_param,
                stride[0],
                stride[1],
                m_c,
                n_c,
                m_compute,
                n_compute,
                channels_last_activation
            );
        }
    }
    return {output_tensor_pos, output_tensor_neg};
}