#include <torch/extension.h>
//
#include <cuda.h>
#include <cuda_fp16.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <mma.h>
#include "sc_device.cuh"

using namespace nvcuda;

// OR_2 simulation in fixed-point
struct OR2_Value {
    int value1;
    int value2;
};

// Add and saturation functions
__device__ __forceinline__ int saturate_shift(int value, int max_value, int precision_diff) {
    int value_saturate = min(value, max_value);
    int round_bit = (value_saturate>>max(0, precision_diff-1))%2;
    return ((value_saturate>>precision_diff)+round_bit)<<precision_diff;
}
__device__ __forceinline__ int fixed_add(int a, int b, int max_value, int precision_in, int precision_out) {
    int sum_full = a+b;
    return saturate_shift(sum_full, max_value, max(0,precision_in*2-precision_out));
}
__device__ __forceinline__ int fixed_add_or(int a, int b, int max_value, int precision_in, int precision_out) {
    int sum_full = a+b-((a*b)>>(precision_in*2));
    return saturate_shift(sum_full, max_value, max(0,precision_in*2-precision_out));
}

__device__ __forceinline__ struct OR2_Value fixed_add_or2(struct OR2_Value a, struct OR2_Value b, int max_value, int precision_in, int precision_out) {
    int value1_out = a.value1 + b.value1 - int((long(a.value1)*long(a.value2)*long(b.value2))>>(precision_in*4)) - int((long(a.value2)*long(b.value1)*long(b.value2))>>(precision_in*4)) + int((long(a.value1)*long(a.value2)*long(b.value1)*long(b.value2))>>(precision_in*6));
    int value2_out = a.value2 + b.value2 - int((long(a.value1)*long(a.value2)*long(b.value1))>>(precision_in*4)) - int((long(a.value1)*long(b.value1)*long(b.value2))>>(precision_in*4)) + int((long(a.value1)*long(a.value2)*long(b.value1)*long(b.value2))>>(precision_in*6));
    value1_out = saturate_shift(value1_out, max_value, std::max(0,precision_in*2-precision_out));
    value2_out = saturate_shift(value2_out, max_value, std::max(0,precision_in*2-precision_out));
    struct OR2_Value output = {value1_out, value2_out};
    return output;
}

__device__ __forceinline__ int mult_normal(int a, int b) {
    return a*b;
}

const int Bit_In_Max = 8;

__device__ __forceinline__ int mult_2x2(int a, int b) {
    int sum = 0;
    for(int bit_a=0; bit_a<Bit_In_Max; bit_a+=2) {
        for(int bit_b=0; bit_b<Bit_In_Max; bit_b+=2) {
            int a_2 = (a>>bit_a)%4;
            int b_2 = (b>>bit_b)%4;
            sum += max(a_2*b_2, 7)<<(bit_a+bit_b);
        }
    }
    return sum;
}

/*
 * Accelerated GPU implementation. Kernel functions
 */

#define BLOCK_INPUT 16
#define BLOCK_WEIGHT 64
#define BATCH_SIZE 1024
#define SHARE_SIZE 96

#define BLOCK_BATCH 16
#define BLOCK_CIN 64

#define SHARE_SIZE_BOTH 96
#define BLOCK_BOTH_INPUT 16
#define BLOCK_BOTH_WEIGHT 64

#define DEFAULT_THREADS 1024
#define VARSEED_THREADS 64

// Normal: 32x16 = 512 threads
// WMMA: (32/8) * (16/8) * 32
__global__
void compute_saturate_general(
    const int32_t* __restrict__ input_stream,
    const int32_t* __restrict__ weight_pos_stream,
    const int32_t* __restrict__ weight_neg_stream,
    int32_t* __restrict__ output_stream,
    int stride_w,
    int stride_h,
    struct Compute_Param c_param,
    int add_config,
    int precision_in,
    int precision_out) {
    int index_gen = threadIdx.x;
    int stride_gen = blockDim.x;
    int index_block = blockIdx.x;
    int stride_block = gridDim.x;

    int cout_step = (c_param.c_outs + C_GENERAL-1) / C_GENERAL;
    int i_flatten_step = ((c_param.i_w_ins-c_param.w_w_ins)/stride_w+1)*((c_param.i_h_ins-c_param.w_h_ins)/stride_h+1);
    int cin_step = (i_flatten_step*c_param.batches + I_GENERAL-1) / I_GENERAL;

    int inner_size = c_param.c_ins*c_param.w_w_ins*c_param.w_h_ins;

    int index_i_c = index_gen / N_GENERAL;
    int index_w_c = index_gen % N_GENERAL;

    int precision_max = min(31, max(precision_in*2, precision_out));
    int max_value = ((1<<precision_max)-1);
    int precision_diff = max(0,precision_in*2-precision_out);
    max_value = (max_value>>precision_diff)<<precision_diff;
    int (*add)(int, int, int, int, int);
    switch(add_config) {
        case 0:
        add=&fixed_add;
        break;
        case 1:
        add=&fixed_add_or;
        break;
    }

    for (int block=index_block; block<c_param.batches*cout_step; block+=stride_block) {
        int batch = block/cout_step;
        int cout_offset = (block%cout_step)*C_GENERAL;
        const int32_t* input_batch = input_stream + batch*c_param.c_ins*c_param.i_w_ins*c_param.i_h_ins;
        const int32_t* weight_pos_couts = weight_pos_stream + cout_offset*c_param.c_ins*c_param.w_w_ins*c_param.w_h_ins;
        const int32_t* weight_neg_couts = weight_neg_stream + cout_offset*c_param.c_ins*c_param.w_w_ins*c_param.w_h_ins;
        int32_t* output_batch = output_stream + batch*c_param.c_outs*i_flatten_step;

        __shared__ uint weight_pos_s [2*N_GENERAL*K_GENERAL_STORE];
        __shared__ uint weight_neg_s [2*N_GENERAL*K_GENERAL_STORE];
        __shared__ uint input_s [2*M_GENERAL*K_GENERAL_STORE];
        for (int cin=0; cin<i_flatten_step; cin+=2*M_GENERAL) {
            for (int cout=0; cout<C_GENERAL; cout+=2*N_GENERAL) {
                uint output_pos_c_00 = 0;
                uint output_pos_c_01 = 0;
                uint output_pos_c_10 = 0;
                uint output_pos_c_11 = 0;
                uint output_neg_c_00 = 0;
                uint output_neg_c_01 = 0;
                uint output_neg_c_10 = 0;
                uint output_neg_c_11 = 0;

                struct OR2_Value sum_pos_c_00 = {0,0};
                struct OR2_Value sum_pos_c_01 = {0,0};
                struct OR2_Value sum_pos_c_10 = {0,0};
                struct OR2_Value sum_pos_c_11 = {0,0};
                struct OR2_Value sum_neg_c_00 = {0,0};
                struct OR2_Value sum_neg_c_01 = {0,0};
                struct OR2_Value sum_neg_c_10 = {0,0};
                struct OR2_Value sum_neg_c_11 = {0,0};

                for (int inner=0; inner<inner_size; inner+=K_GENERAL) {
                    // Load weights
                    for (int index_w=index_gen; index_w<2*N_GENERAL*K_GENERAL; index_w+=stride_gen) {
                        int n = index_w / K_GENERAL;
                        int k = index_w % K_GENERAL;
                        if ((inner+k<inner_size) & (cout_offset+cout+n<c_param.c_outs) & (cout+n<C_GENERAL)) {
                            int weight_index = (cout+n) * c_param.c_ins * c_param.w_w_ins * c_param.w_h_ins + (inner+k);
                            weight_pos_s[n*K_GENERAL_STORE+k] = weight_pos_couts[weight_index];
                            weight_neg_s[n*K_GENERAL_STORE+k] = weight_neg_couts[weight_index];
                        }
                        else {
                            weight_pos_s[n*K_GENERAL_STORE+k] = 0;
                            weight_neg_s[n*K_GENERAL_STORE+k] = 0;
                        }
                    }
                    // __syncthreads();
                    // Load inputs
                    for (int index_i=index_gen; index_i<2*M_GENERAL*K_GENERAL; index_i+=stride_gen) {
                        int m = index_i / K_GENERAL;
                        int k = index_i % K_GENERAL;
                        int w_in_i_c = ((cin+m) / ((c_param.i_h_ins - c_param.w_h_ins)/stride_h+1))*stride_w;
                        int h_in_i_c = ((cin+m) % ((c_param.i_h_ins - c_param.w_h_ins)/stride_h+1))*stride_h;
                        int c_in = (inner+k) / (c_param.w_w_ins*c_param.w_h_ins);
                        int w_in_w_c = ((inner+k) % (c_param.w_w_ins*c_param.w_h_ins)) / c_param.w_h_ins;
                        int h_in_w_c = (inner+k) % c_param.w_h_ins;
                        if ((inner+k<inner_size) & (cin+m<i_flatten_step)) {
                            int input_index = c_in * c_param.i_w_ins * c_param.i_h_ins
                                             +(w_in_i_c+w_in_w_c) * c_param.i_h_ins
                                             +(h_in_i_c+h_in_w_c);
                            input_s[m*K_GENERAL_STORE+k] = input_batch[input_index];
                        }
                        else input_s[m*K_GENERAL_STORE+k] = 0;
                    }
                    __syncthreads();
                    // Compute
                    if (add_config<2) {
                        for (int i=0; i<K_GENERAL; i++) {
                            int32_t input_s_0 = input_s[(index_i_c*2+0)*K_GENERAL_STORE+i];
                            int32_t input_s_1 = input_s[(index_i_c*2+1)*K_GENERAL_STORE+i];
                            int32_t weight_pos_s_0 = weight_pos_s[(index_w_c*2+0)*K_GENERAL_STORE+i];
                            int32_t weight_pos_s_1 = weight_pos_s[(index_w_c*2+1)*K_GENERAL_STORE+i];
                            int32_t weight_neg_s_0 = weight_neg_s[(index_w_c*2+0)*K_GENERAL_STORE+i];
                            int32_t weight_neg_s_1 = weight_neg_s[(index_w_c*2+1)*K_GENERAL_STORE+i];

                            output_pos_c_00 = (*add)(output_pos_c_00, input_s_0 * weight_pos_s_0, max_value, precision_in, precision_out);
                            output_pos_c_01 = (*add)(output_pos_c_01, input_s_0 * weight_pos_s_1, max_value, precision_in, precision_out);
                            output_pos_c_10 = (*add)(output_pos_c_10, input_s_1 * weight_pos_s_0, max_value, precision_in, precision_out);
                            output_pos_c_11 = (*add)(output_pos_c_11, input_s_1 * weight_pos_s_1, max_value, precision_in, precision_out);

                            output_neg_c_00 = (*add)(output_neg_c_00, input_s_0 * weight_neg_s_0, max_value, precision_in, precision_out);
                            output_neg_c_01 = (*add)(output_neg_c_01, input_s_0 * weight_neg_s_1, max_value, precision_in, precision_out);
                            output_neg_c_10 = (*add)(output_neg_c_10, input_s_1 * weight_neg_s_0, max_value, precision_in, precision_out);
                            output_neg_c_11 = (*add)(output_neg_c_11, input_s_1 * weight_neg_s_1, max_value, precision_in, precision_out);
                        }
                    }
                    else {
                        int i=0;
                        for (; i+1<K_GENERAL; i+=2) {
                            int32_t input_s_0_0 = input_s[(index_i_c*2+0)*K_GENERAL_STORE+i+0];
                            int32_t input_s_1_0 = input_s[(index_i_c*2+1)*K_GENERAL_STORE+i+0];
                            int32_t input_s_0_1 = input_s[(index_i_c*2+0)*K_GENERAL_STORE+i+1];
                            int32_t input_s_1_1 = input_s[(index_i_c*2+1)*K_GENERAL_STORE+i+1];
                            int32_t weight_pos_s_0_0 = weight_pos_s[(index_w_c*2+0)*K_GENERAL_STORE+i+0];
                            int32_t weight_pos_s_1_0 = weight_pos_s[(index_w_c*2+1)*K_GENERAL_STORE+i+0];
                            int32_t weight_neg_s_0_0 = weight_neg_s[(index_w_c*2+0)*K_GENERAL_STORE+i+0];
                            int32_t weight_neg_s_1_0 = weight_neg_s[(index_w_c*2+1)*K_GENERAL_STORE+i+0];
                            int32_t weight_pos_s_0_1 = weight_pos_s[(index_w_c*2+0)*K_GENERAL_STORE+i+1];
                            int32_t weight_pos_s_1_1 = weight_pos_s[(index_w_c*2+1)*K_GENERAL_STORE+i+1];
                            int32_t weight_neg_s_0_1 = weight_neg_s[(index_w_c*2+0)*K_GENERAL_STORE+i+1];
                            int32_t weight_neg_s_1_1 = weight_neg_s[(index_w_c*2+1)*K_GENERAL_STORE+i+1];

                            struct OR2_Value product_pos_s_00 = {input_s_0_0*weight_pos_s_0_0, input_s_0_1*weight_pos_s_0_1};
                            struct OR2_Value product_pos_s_01 = {input_s_0_0*weight_pos_s_1_0, input_s_0_1*weight_pos_s_1_1};
                            struct OR2_Value product_pos_s_10 = {input_s_1_0*weight_pos_s_0_0, input_s_1_1*weight_pos_s_0_1};
                            struct OR2_Value product_pos_s_11 = {input_s_1_0*weight_pos_s_1_0, input_s_1_1*weight_pos_s_1_1};
                            struct OR2_Value product_neg_s_00 = {input_s_0_0*weight_neg_s_0_0, input_s_0_1*weight_neg_s_0_1};
                            struct OR2_Value product_neg_s_01 = {input_s_0_0*weight_neg_s_1_0, input_s_0_1*weight_neg_s_1_1};
                            struct OR2_Value product_neg_s_10 = {input_s_1_0*weight_neg_s_0_0, input_s_1_1*weight_neg_s_0_1};
                            struct OR2_Value product_neg_s_11 = {input_s_1_0*weight_neg_s_1_0, input_s_1_1*weight_neg_s_1_1};

                            sum_pos_c_00 = fixed_add_or2(sum_pos_c_00, product_pos_s_00, max_value, precision_in, precision_out);
                            sum_pos_c_01 = fixed_add_or2(sum_pos_c_01, product_pos_s_01, max_value, precision_in, precision_out);
                            sum_pos_c_10 = fixed_add_or2(sum_pos_c_10, product_pos_s_10, max_value, precision_in, precision_out);
                            sum_pos_c_11 = fixed_add_or2(sum_pos_c_11, product_pos_s_11, max_value, precision_in, precision_out);
                            sum_neg_c_00 = fixed_add_or2(sum_neg_c_00, product_neg_s_00, max_value, precision_in, precision_out);
                            sum_neg_c_01 = fixed_add_or2(sum_neg_c_01, product_neg_s_01, max_value, precision_in, precision_out);
                            sum_neg_c_10 = fixed_add_or2(sum_neg_c_10, product_neg_s_10, max_value, precision_in, precision_out);
                            sum_neg_c_11 = fixed_add_or2(sum_neg_c_11, product_neg_s_11, max_value, precision_in, precision_out);
                        }
                        if (i<K_GENERAL) {
                            int32_t input_s_0 = input_s[(index_i_c*2+0)*K_GENERAL_STORE+i];
                            int32_t input_s_1 = input_s[(index_i_c*2+1)*K_GENERAL_STORE+i];
                            int32_t weight_pos_s_0 = weight_pos_s[(index_w_c*2+0)*K_GENERAL_STORE+i];
                            int32_t weight_pos_s_1 = weight_pos_s[(index_w_c*2+1)*K_GENERAL_STORE+i];
                            int32_t weight_neg_s_0 = weight_neg_s[(index_w_c*2+0)*K_GENERAL_STORE+i];
                            int32_t weight_neg_s_1 = weight_neg_s[(index_w_c*2+1)*K_GENERAL_STORE+i];

                            struct OR2_Value product_pos_s_00 = {input_s_0*weight_pos_s_0, 0};
                            struct OR2_Value product_pos_s_01 = {input_s_0*weight_pos_s_1, 0};
                            struct OR2_Value product_pos_s_10 = {input_s_1*weight_pos_s_0, 0};
                            struct OR2_Value product_pos_s_11 = {input_s_1*weight_pos_s_1, 0};
                            struct OR2_Value product_neg_s_00 = {input_s_0*weight_neg_s_0, 0};
                            struct OR2_Value product_neg_s_01 = {input_s_0*weight_neg_s_1, 0};
                            struct OR2_Value product_neg_s_10 = {input_s_1*weight_neg_s_0, 0};
                            struct OR2_Value product_neg_s_11 = {input_s_1*weight_neg_s_1, 0};
                            
                            sum_pos_c_00 = fixed_add_or2(sum_pos_c_00, product_pos_s_00, max_value, precision_in, precision_out);
                            sum_pos_c_01 = fixed_add_or2(sum_pos_c_01, product_pos_s_01, max_value, precision_in, precision_out);
                            sum_pos_c_10 = fixed_add_or2(sum_pos_c_10, product_pos_s_10, max_value, precision_in, precision_out);
                            sum_pos_c_11 = fixed_add_or2(sum_pos_c_11, product_pos_s_11, max_value, precision_in, precision_out);
                            sum_neg_c_00 = fixed_add_or2(sum_neg_c_00, product_neg_s_00, max_value, precision_in, precision_out);
                            sum_neg_c_01 = fixed_add_or2(sum_neg_c_01, product_neg_s_01, max_value, precision_in, precision_out);
                            sum_neg_c_10 = fixed_add_or2(sum_neg_c_10, product_neg_s_10, max_value, precision_in, precision_out);
                            sum_neg_c_11 = fixed_add_or2(sum_neg_c_11, product_neg_s_11, max_value, precision_in, precision_out);
                        }
                    }
                    __syncthreads();
                }
                int cout_c_0 = cout_offset+cout+(index_w_c*2+0);
                int cout_c_1 = cout_offset+cout+(index_w_c*2+1);
                int cin_c_0 = cin+(index_i_c*2+0);
                int cin_c_1 = cin+(index_i_c*2+1);
                if (add_config<2) {
                    if ((cout_c_0<c_param.c_outs)  & (cin_c_0<i_flatten_step) & (cout+(index_w_c*2+0)<C_GENERAL)) {
                        int index_o = cout_c_0*i_flatten_step+cin_c_0;
                        output_batch[index_o] += output_pos_c_00 - output_neg_c_00;
                    }
                    if ((cout_c_1<c_param.c_outs)  & (cin_c_0<i_flatten_step) & (cout+(index_w_c*2+1)<C_GENERAL)) {
                        int index_o = cout_c_1*i_flatten_step+cin_c_0;
                        output_batch[index_o] += output_pos_c_01 - output_neg_c_01;
                    }
                    if ((cout_c_0<c_param.c_outs)  & (cin_c_1<i_flatten_step) & (cout+(index_w_c*2+0)<C_GENERAL)) {
                        int index_o = cout_c_0*i_flatten_step+cin_c_1;
                        output_batch[index_o] += output_pos_c_10 - output_neg_c_10;
                    }
                    if ((cout_c_1<c_param.c_outs)  & (cin_c_1<i_flatten_step) & (cout+(index_w_c*2+1)<C_GENERAL)) {
                        int index_o = cout_c_1*i_flatten_step+cin_c_1;
                        output_batch[index_o] += output_pos_c_11 - output_neg_c_11;
                    }
                }
                else {
                    if ((cout_c_0<c_param.c_outs)  & (cin_c_0<i_flatten_step) & (cout+(index_w_c*2+0)<C_GENERAL)) {
                        int index_o = cout_c_0*i_flatten_step+cin_c_0;
                        output_batch[index_o] += sum_pos_c_00.value1+sum_pos_c_00.value2 - sum_neg_c_00.value1-sum_neg_c_00.value2;
                    }
                    if ((cout_c_1<c_param.c_outs)  & (cin_c_0<i_flatten_step) & (cout+(index_w_c*2+1)<C_GENERAL)) {
                        int index_o = cout_c_1*i_flatten_step+cin_c_0;
                        output_batch[index_o] += sum_pos_c_01.value1+sum_pos_c_01.value2 - sum_neg_c_01.value1-sum_neg_c_01.value2;
                    }
                    if ((cout_c_0<c_param.c_outs)  & (cin_c_1<i_flatten_step) & (cout+(index_w_c*2+0)<C_GENERAL)) {
                        int index_o = cout_c_0*i_flatten_step+cin_c_1;
                        output_batch[index_o] += sum_pos_c_10.value1+sum_pos_c_10.value2 - sum_neg_c_10.value1-sum_neg_c_10.value2;
                    }
                    if ((cout_c_1<c_param.c_outs)  & (cin_c_1<i_flatten_step) & (cout+(index_w_c*2+1)<C_GENERAL)) {
                        int index_o = cout_c_1*i_flatten_step+cin_c_1;
                        output_batch[index_o] += sum_pos_c_11.value1+sum_pos_c_11.value2 - sum_neg_c_11.value1-sum_neg_c_11.value2;
                    }
                }
                __syncthreads();
            }
        }
    }
}

torch::Tensor conv2d_saturate_cuda(torch::Tensor input,
        torch::Tensor weight_pos,
        torch::Tensor weight_neg,
        at::IntArrayRef stride,
        int add_config,
        int precision_in,
        int precision_out) {
    auto weight_size = weight_pos.sizes();
    auto input_size = input.sizes();
    auto device = weight_pos.device();
    int device_index = device.index();
    cudaSetDevice(int(device_index));

    const int threads = THREADS_GENERAL;

    auto output_tensor = torch::zeros({input_size[0], weight_size[0], (input_size[2]-weight_size[2])/stride[0]+1, (input_size[3]-weight_size[3])/stride[1]+1}, at::TensorOptions().dtype(torch::kInt32).device(device));
    struct Compute_Param c_param = {0, 0, input_size[0], input_size[1], input_size[2], input_size[3], weight_size[0], weight_size[2], weight_size[3]};
    compute_saturate_general <<<10000, M_GENERAL*N_GENERAL>>> (
        input.data_ptr<int32_t>(),
        weight_pos.data_ptr<int32_t>(),
        weight_neg.data_ptr<int32_t>(),
        output_tensor.data_ptr<int32_t>(),
        stride[0],
        stride[1],
        c_param,
        add_config,
        precision_in,
        precision_out
        );
    return output_tensor;
}