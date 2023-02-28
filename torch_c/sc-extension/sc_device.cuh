#include <torch/extension.h>
//
#include <cuda.h>
#include <cuda_fp16.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <mma.h>
#define _CG_ABI_EXPERIMENTAL
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#ifndef LOADED_DEVICE_HPP
#define LOADED_DEVICE_HPP

constexpr int BLOCK_INPUT=16;
constexpr int BLOCK_WEIGHT=64;
constexpr int BATCH_SIZE=1024;
constexpr int SHARE_SIZE=96;

constexpr int BLOCK_BATCH=16;
constexpr int BLOCK_CIN=64;

constexpr int SHARE_SIZE_BOTH=96;
constexpr int BLOCK_BOTH_INPUT=16;
constexpr int BLOCK_BOTH_WEIGHT=64;

constexpr int DEFAULT_THREADS=1024;
constexpr int VARSEED_THREADS=64;

// constexpr int POS_SEED=67;
// constexpr int NEG_SEED=37;
constexpr int POS_SEED=13;
constexpr int NEG_SEED=13;

constexpr int COMPUTE_CINS = 32;
constexpr int THREADS_GENERAL = 96;
#if __CUDA_ARCH__ >= 800
constexpr int THREADS_TOTAL = 1536;
#else
constexpr int THREADS_TOTAL = 1024;
#endif
constexpr int SEED_4 = 8;
constexpr int SEED_5 = 20;
constexpr int SEED_6 = 36;
constexpr int SEED_7 = 68;

// Blocking configuration
constexpr int I_GENERAL = 64;
// COUT block
constexpr int C_GENERAL = 64;
constexpr int C_GENERAL_33 = 32;

// Matmul block

constexpr int M_GENERAL = 32;
constexpr int N_GENERAL = 16;
constexpr int K_GENERAL = 8;

#if __CUDA_ARCH__ >= 800
constexpr int K_GENERAL_STORE = K_GENERAL+4;
constexpr int N_GENERAL_STORE = 2*N_GENERAL+4;
constexpr int M_GENERAL_STORE = M_GENERAL+4;
#else
constexpr int K_GENERAL_STORE = K_GENERAL+4;
#endif

// #if __CUDA_ARCH__ >= 800
// 2X unrolling
// const int M_GENERAL_WMMA = (M_GENERAL+WMMA_M_bin-1)/WMMA_M_bin;
constexpr int WMMA_M_bin = 8;
constexpr int WMMA_N_bin = 8;
constexpr int WMMA_K_bin = 128;
constexpr int WMMA_INT_WIDTH = 32;
constexpr int N_GENERAL_WMMA = (2*N_GENERAL+WMMA_N_bin-1)/WMMA_N_bin;
constexpr int M_GENERAL_WMMA = (M_GENERAL+WMMA_M_bin-1)/WMMA_M_bin;
// #endif 

struct Compute_Param {
    int bit_length;
    int bit_packs;
    int batches;
    int c_ins;
    int i_w_ins;
    int i_h_ins;
    int c_outs;
    int w_w_ins;
    int w_h_ins;
};
struct Matmul_Param {
    int pw_size;
    int i_pw_step;
    int w_pw_step;
    int i_flatten_step;
    int w_flatten_step;
    int i_kernel_size;
    int w_kernel_size;
    int inner_size;
    int o_cout_step;
};
struct Batch_Steps {
    int o_batch_step;
    int i_bin_batch_step;
    int i_stream_batch_step;
    int i_point_batch_step;
};

__device__ __forceinline__ int d_lfsr_8_xnor(int value) {
    return 1-((value/128)+(value/32)%2+(value/16)%2+(value/8)%2)%2+2*(value%128);
}

__device__ __forceinline__ int d_lfsr_8(int value) {
    return ((value/128)+(value/32)%2+(value/16)%2+(value/8)%2)%2+2*(value%128);
}

__device__ __forceinline__ int d_lfsr_8_acc(int value) {
    switch(value) {
        case 1:
        return 0;
        case 0:
        return 2;
        default:
        return d_lfsr_8(value);
    }
}

__device__ __forceinline__ int d_lfsr_8_1(int value) {
    return ((value/128)+(value/8)%2+(value/4)%2+(value/2)%2)%2+2*(value%128);
}

__device__ __forceinline__ int d_lfsr_7(int value) {
    return ((value/64)+(value/32)%2)%2+2*(value%64);
}

__device__ __forceinline__ int d_lfsr_7_acc(int value) {
    switch(value) {
        case 1:
        return 0;
        case 0:
        return 2;
        default:
        return d_lfsr_7(value);
    }
}

__device__ __forceinline__ int d_lfsr_7_xnor(int value) {
    return 1-((value/64)+(value/32)%2)%2+2*(value%64);
}

__device__ __forceinline__ int d_lfsr_7_1(int value) {
    return ((value/64)+(value/8)%2)%2+2*(value%64);
}

__device__ __forceinline__ int d_lfsr_6(int value) {
    return ((value/32)+(value/16)%2)%2+2*(value%32);
}

__device__ __forceinline__ int d_lfsr_6_acc(int value) {
    switch(value) {
        case 1:
        return 0;
        case 0:
        return 2;
        default:
        return d_lfsr_6(value);
    }
}

__device__ __forceinline__ int d_lfsr_6_xnor(int value) {
    return 1-((value/32)+(value/16)%2)%2+2*(value%32);
}

__device__ __forceinline__ int d_lfsr_6_1(int value) {
    return ((value/32)+(value/16)%2+(value/4)%2+(value/2)%2)+2*(value%32);
}

__device__ __forceinline__ int d_lfsr_5(int value) {
    return ((value/16)+(value/4)%2)%2+2*(value%16);
}

__device__ __forceinline__ int d_lfsr_5_acc(int value) {
    switch(value) {
        case 1:
        return 0;
        case 0:
        return 2;
        default:
        return d_lfsr_5(value);
    }
}

__device__ __forceinline__ int d_lfsr_5_xnor(int value) {
    return 1-((value/16)+(value/4)%2)%2+2*(value%16);
}

__device__ __forceinline__ int d_lfsr_5_1(int value) {
    return ((value/16)+(value/2)%2)%2+2*(value%16);
}

__device__ __forceinline__ int d_lfsr_5_2(int value) {
    return ((value/16)+(value/8)%2+(value/4)%2+(value/2)%2)%2+2*(value%16);
}

__device__ __forceinline__ int d_lfsr_4(int value) {
    return ((value/8)+(value/4)%2)%2+2*(value%8);
}

__device__ __forceinline__ int d_lfsr_4_acc(int value) {
    switch(value) {
        case 1:
        return 0;
        case 0:
        return 2;
        default:
        return d_lfsr_4(value);
    }
}

__device__ __forceinline__ int d_lfsr_4_xnor(int value) {
    return 1-((value/8)+(value/4)%2)%2+2*(value%8);
}

__device__ __forceinline__ int d_lfsr_3(int value) {
    return ((value/4)+(value/2)%2)%2+2*(value%4);
}

__device__ __forceinline__ int d_lfsr_3_acc(int value) {
    switch(value) {
        case 1:
        return 0;
        case 0:
        return 2;
        default:
        return d_lfsr_3(value);
    }
}

__device__ __forceinline__ int d_lfsr_3_xnor(int value) {
    return 1-((value/4)+(value/2)%2)%2+2*(value%4);
}

__device__ __forceinline__ int d_lfsr_2(int value) {
    return (value+1)%4;
}

__device__ __forceinline__ int d_lfsr_1(int value) {
    return 1-value;
}

__device__ __forceinline__
int (*lfsr_select(int lfsr_length, bool xnor))(int) {
    int (*lfsr)(int);
    switch(lfsr_length) {
    case 1:
        lfsr=&d_lfsr_1;
        break;
    case 2:
        lfsr=&d_lfsr_2;
        break;
    case 3:
        if (xnor) lfsr = &d_lfsr_3_xnor;
        else lfsr=&d_lfsr_3;
        break;
    case 4:
        if (xnor) lfsr = &d_lfsr_4_xnor;
        else lfsr=&d_lfsr_4;
        break;
    case 5:
        if (xnor) lfsr = &d_lfsr_5_xnor;
        else lfsr=&d_lfsr_5;
        break;
    case 6:
        if (xnor) lfsr=&d_lfsr_6_xnor;
        else lfsr=&d_lfsr_6;
        break;
    case 7:
        if (xnor) lfsr=&d_lfsr_7_xnor;
        else lfsr=&d_lfsr_7;
        break;
    case 8:
        if (xnor) lfsr=&d_lfsr_8_xnor;
        else lfsr=&d_lfsr_8;
        break;
    }
    return lfsr;
}

__device__ __forceinline__
int seed_mult_select(int lfsr_length) {
    int seed_mult = 1;
    switch(lfsr_length) {
        case 4:
        seed_mult = SEED_4;
        break;
        case 5:
        seed_mult = SEED_5;
        break;
        case 6:
        seed_mult = SEED_6;
        break;
        case 7:
        seed_mult = SEED_7;
        break;
    }
    return seed_mult;
}

__device__ __forceinline__ int d_mux_gen(int value, int rand, int prec) {
    int prior = 0;
    for(int p=0; p<prec; p++) {
        int value_c = (value % (1 << (p+1))) /  (1 << p);
        int rand_c = (rand % (1 << (p+1))) / (1 << p);
        prior = value_c * rand_c + prior * (1-rand_c);
    }
    return prior;
}

__device__ __forceinline__ int d_comp_gen(int value, int rand, int prec) {
    return (value>rand);
}

__device__ __forceinline__ int weight_ind_oihw(int c_out, int c_in, int w_in, int h_in, int c_outs, int c_ins, int w_ins, int h_ins) {
    return c_out*c_ins*w_ins*h_ins + c_in*w_ins*h_ins + w_in*h_ins + h_in;
}

__device__ __forceinline__ int weight_ind_ohwi(int c_out, int c_in, int w_in, int h_in, int c_outs, int c_ins, int w_ins, int h_ins) {
    return c_out*c_ins*w_ins*h_ins + w_in*h_ins*c_ins + h_in*c_ins + c_in;
}

__device__ __forceinline__ int output_ind_nchw(int c_out, int c_in, int stride) {
    return c_out*stride+c_in;
}

__device__ __forceinline__ int output_ind_nhwc(int c_out, int c_in, int stride) {
    return c_in*stride+c_out;
}

__device__ __forceinline__ int or_act_n(int value, int n) {
    return min(value, n);
}

__device__ __forceinline__ int or_act_no(int value, int n) {
    return int(value>=n);
}

__device__ __forceinline__ int or_act_or_21(int value, int n) {
    return max(0,value-n+2);
}

__device__ __forceinline__ int or_act_or_24(int value, int n) {
    return min(value, 1) + max(0, value-n+1);
}

__device__ __forceinline__ void or_act_store(bool valid, int32_t* ptr_pos, int32_t* ptr_neg, int (*or_act)(int, int), int value_pos, int value_neg, int n) {
    if (valid) {
        // printf("Valid!\n");
        *ptr_pos += or_act(value_pos, n);
        *ptr_neg += or_act(value_neg, n);
    }
}

__device__ __forceinline__ void or_act_store_ns(bool valid, int32_t value, int32_t* ptr, int (*or_act)(int, int), int value_pos, int value_neg, int n) {
    if (valid) {
        *ptr = value + or_act_n(value_pos, n) - or_act_n(value_neg, n);
    }
}

__device__ __forceinline__ void or_act_store_ns_ptr(bool valid, int32_t* ptr, int (*or_act)(int, int), int value_pos, int value_neg, int n) {
    if (valid) {
        int value = value_pos >> 16;
        value_pos = (value_pos << 16) >> 16;
        *ptr = value + or_act_n(value_pos, n) - or_act_n(value_neg, n);
    }
}

__device__ __forceinline__ int32_t or_act_update(int32_t value, int (*or_act)(int, int), int value_pos, int value_neg, int n) {
    return value + or_act_n(value_pos, n) - or_act_n(value_neg, n);
}

__device__ __forceinline__ int32_t or_act_split_update(uint value, int (*or_act)(int, int), int n) {
    uint old_value = value>>16;
    uint new_value = (value<<16)>>16;
    return (old_value + or_act_n(new_value, n))<<16;
}

__device__ __forceinline__ int32_t or_act_analog_update(uint value, int prec) {
    uint max_value = (1u<<prec)-1;
    uint old_value = value>>16;
    uint new_value = (value<<16)>>16;
    return (old_value + min(new_value, max_value))<<16;
}

__device__ __forceinline__
float or_approx_1_forward_scalar(
    float input_c) {
    float input_exp = __expf(-input_c);
    return 1-input_exp;
}

__device__ __forceinline__
float or_approx_2_forward_scalar(
    float input_c) {
    float input_exp = __expf(-input_c);
    float x = fmaf(-2.f,input_exp,2.f);
    return fmaf(-input_c,input_exp,x);
}

__device__ __forceinline__
float or_approx_3_forward_scalar(
    float input_c) {
    float input_exp = __expf(-input_c);
    float input_c_2 = input_c*input_c;
    float x = fmaf(-3.f,input_exp,3.f);
    x = fmaf(-2.f*input_c, input_exp, x);
    x = fmaf(-0.5f*input_c_2, input_exp, x);
    return x;
}

__device__ __forceinline__
float or_approx_4_forward_scalar(
    float input_c) {
    float input_exp = __expf(-input_c);
    float input_c_2 = input_c*input_c;
    float x = fmaf(-4.f, input_exp, 4.f);
    x = fmaf(-3.f*input_c, input_exp, x);
    float input_c_3 = input_c_2*input_c;
    x = fmaf(-input_c_2, input_exp, x);
    x = fmaf((-1.f/6.f)*input_c_3, input_exp, x);
    return x;
}

__device__ __forceinline__
float ar_approx_1_forward_scalar(
    float input_c,
    int n) {
    float output = 1.f;
    float input_exp = __expf(-input_c);
    for (int i=0; i<n+1; i++) {
        // output -= powf(input_c, i)*(1/)
        float sum_pow = 1.f;
        float factorial = 1.f;
        for (int j=0; j<i; j++) {
            sum_pow *= input_c;
            factorial *= float(j+1);
        }
        output -= sum_pow*(1.f/factorial)*input_exp;
    }
    return output;
}

__device__ __forceinline__
float ar_approx_1_backward_scalar(
    float grad_output,
    float sum,
    int n) {
    float sum_exp = __expf(-sum);
    float sum_pow = 1.f;
    float factorial = 1.f;
    for (int j=0; j<n; j++) {
        sum_pow = sum*sum_pow;
        factorial = factorial*(j+1);
    }
    return sum_pow*(1.f/factorial)*sum_exp*grad_output;
}

__device__ __forceinline__
float or_approx_1_backward_scalar(
    float grad_output,
    float sum) {
    float sum_exp = __expf(-sum);
    return sum_exp*grad_output;
}

__device__ __forceinline__
float or_approx_2_backward_scalar(
    float grad_output,
    float sum) {
    float sum_exp = __expf(-sum);
    float x_or = fmaf(sum, sum_exp, sum_exp);
    return x_or*grad_output;
}

__device__ __forceinline__
float or_approx_3_backward_scalar(
    float grad_output,
    float sum) {
    float sum_exp = __expf(-sum);
    float sum_2 = sum*sum;
    float x_or = fmaf(sum, sum_exp, sum_exp);
    x_or = fmaf(0.5f*sum_2, sum_exp, x_or);
    return x_or*grad_output;
}

__device__ __forceinline__
float or_approx_4_backward_scalar(
    float grad_output,
    float sum) {
    float sum_exp = __expf(-sum);
    float sum_2 = sum*sum;
    float x_or = fmaf(sum, sum_exp, sum_exp);
    float sum_3 = sum_2*sum;
    x_or = fmaf(0.5f*sum_2, sum_exp, x_or);
    x_or = fmaf((1.f/6.f)*sum_3, sum_exp, x_or);
    return x_or*grad_output;
}

__device__ __forceinline__
float bias_correct_forward_scalar(
    float x,
    float x_rand,
    float* __restrict__ bias_coef,
    float* __restrict__ std_coef) {
    float x_2 = x*x;
    float x_bias = fmaf(bias_coef[1], x, bias_coef[0]);
    float x_3 = x_2*x;
    x_bias = fmaf(bias_coef[2], x_2, x_bias);
    float x_4 = x_3*x;
    x_bias = fmaf(bias_coef[3], x_3, x_bias);
    float x_5 = x_4*x;
    x_bias = fmaf(bias_coef[4], x_4, x_bias);
    x_bias = fmaf(bias_coef[5], x_5, x_bias);
    float x_std = fmaf(std_coef[1], x, std_coef[0]);
    x_std = fmaf(std_coef[2], x_2, x_std);
    x_std = fmaf(std_coef[3], x_3, x_std);
    x_std = fmaf(std_coef[4], x_4, x_std);
    x_std = fmaf(std_coef[5], x_5, x_std);
    return x + x_bias + x_std*x_rand;
}

__device__ __forceinline__
float bias_correct_backward_scalar(
    float x,
    float* __restrict__ bias_coef) {
    float x_2 = x*x;
    float x_dot = fmaf(2.f*bias_coef[2], x, 1.f+bias_coef[1]);
    float x_3 = x_2*x;
    x_dot = fmaf(3.f*bias_coef[3], x_2, x_dot);
    float x_4 = x_3*x;
    x_dot = fmaf(4.f*bias_coef[4], x_3, x_dot);
    x_dot = fmaf(5.f*bias_coef[5], x_4, x_dot);
    return x_dot;
}

__device__ __forceinline__
float or_approx_1_bias_correct_forward_scalar(
    float input_c,
    float x_rand,
    float* __restrict__ bias_coef,
    float* __restrict__ std_coef) {
    float x=or_approx_1_forward_scalar(input_c);
    return bias_correct_forward_scalar(x, x_rand, bias_coef, std_coef);
}

__device__ __forceinline__
float or_approx_2_bias_correct_forward_scalar(
    float input_c,
    float x_rand,
    float* __restrict__ bias_coef,
    float* __restrict__ std_coef) {
    float x=or_approx_2_forward_scalar(input_c);
    return bias_correct_forward_scalar(x, x_rand, bias_coef, std_coef);
}

__device__ __forceinline__
float or_approx_3_bias_correct_forward_scalar(
    float input_c,
    float x_rand,
    float* __restrict__ bias_coef,
    float* __restrict__ std_coef) {
    float x=or_approx_3_forward_scalar(input_c);
    return bias_correct_forward_scalar(x, x_rand, bias_coef, std_coef);
}

__device__ __forceinline__
float or_approx_4_bias_correct_forward_scalar(
    float input_c,
    float x_rand,
    float* __restrict__ bias_coef,
    float* __restrict__ std_coef) {
    float x=or_approx_4_forward_scalar(input_c);
    return bias_correct_forward_scalar(x, x_rand, bias_coef, std_coef);
}

__device__ __forceinline__
float or_approx_1_bias_correct_backward_scalar(
    float grad_output,
    float input_c,
    float* __restrict__ bias_coef) {
    // OR_1 forward
    float input_exp = __expf(-input_c);
    float x = 1-input_exp;
    // Bias backward
    float x_dot = bias_correct_backward_scalar(x, bias_coef);
    x_dot = x_dot * grad_output;
    // OR_2 backward
    return input_exp*x_dot;
}

__device__ __forceinline__
float or_approx_2_bias_correct_backward_scalar(
    float grad_output,
    float input_c,
    float* __restrict__ bias_coef) {
    // OR_2 forward
    float input_exp = __expf(-input_c);
    float x = fmaf(-2.f,input_exp,2.f);
    x = fmaf(-input_c,input_exp,x);
    // Bias backward
    float x_dot = bias_correct_backward_scalar(x, bias_coef);
    x_dot = x_dot * grad_output;
    // OR_2 backward
    float x_or = fmaf(input_c, input_exp, input_exp);
    return x_or*x_dot;
}

__device__ __forceinline__
float or_approx_3_bias_correct_backward_scalar(
    float grad_output,
    float input_c,
    float* __restrict__ bias_coef) {
    // OR_3 forward
    float input_exp = __expf(-input_c);
    float input_c_2 = input_c*input_c;
    float x = fmaf(-3.f,input_exp,3.f);
    x = fmaf(-2.f*input_c, input_exp, x);
    x = fmaf(-0.5f*input_c_2, input_exp, x);
    // Bias backward
    float x_dot = bias_correct_backward_scalar(x, bias_coef);
    x_dot = x_dot * grad_output;
    // OR_3 backward
    float x_or = fmaf(input_c, input_exp, input_exp);
    x_or = fmaf(0.5f*input_c_2, input_exp, x_or);
    return x_or*x_dot;
}

__device__ __forceinline__
float or_approx_4_bias_correct_backward_scalar(
    float grad_output,
    float input_c,
    float* __restrict__ bias_coef) {
    // OR_4 forward
    float input_exp = __expf(-input_c);
    float input_c_2 = input_c*input_c;
    float x = fmaf(-4.f, input_exp, 4.f);
    x = fmaf(-3.f*input_c, input_exp, x);
    float input_c_3 = input_c_2*input_c;
    x = fmaf(-input_c_2, input_exp, x);
    x = fmaf((-1.f/6.f)*input_c_3, input_exp, x);
    // Bias backward
    float x_dot = bias_correct_backward_scalar(x, bias_coef);
    x_dot = x_dot * grad_output;
    // OR_3 backward
    float x_or = fmaf(input_c, input_exp, input_exp);
    x_or = fmaf(0.5f*input_c_2, input_exp, x_or);
    x_or = fmaf((1.f/6.f)*input_c_3, input_exp, x_or);
    return x_or*x_dot;
}

__device__ __forceinline__
uint generate_mask_scalar(
    float* __restrict__ rand_32,
    float thres) {
    uint mask = 0;
    for (int i=0; i<32; i++) {
        mask += uint(rand_32[i]<thres) << i;
    }
    return mask;
}

__device__ __forceinline__
void generate_rand_array(
    float* __restrict__ rand_32,
    curandStatePhilox4_32_10_t* __restrict__ rand_state) {
    for (int i=0; i<8; i++) {
        float4 x_rand_4 = curand_uniform4(rand_state);
        rand_32[i*4+0] = x_rand_4.w;
        rand_32[i*4+1] = x_rand_4.x;
        rand_32[i*4+2] = x_rand_4.y;
        rand_32[i*4+3] = x_rand_4.z;
    }
}

constexpr int I_UNROLL_TEST = 2;
constexpr int W_UNROLL_TEST = 2;
constexpr int M_GENERAL_TEST = 16;
constexpr int N_GENERAL_TEST = 16;
constexpr int K_GENERAL_TEST = 256;
constexpr int K_GENERAL_UNIT = 32;

constexpr int WMMA_M_bin_TEST = 16;
constexpr int WMMA_N_bin_TEST = 8;
constexpr int WMMA_K_bin_TEST = 256;
constexpr int WMMA_M_UNROLL = WMMA_M_bin_TEST / 8;
constexpr int WMMA_N_UNROLL = WMMA_N_bin_TEST / 8;
constexpr int WMMA_K_UNROLL = WMMA_K_bin_TEST / 128;

constexpr int M_GENERAL_TEST_STORE = M_GENERAL_TEST+4;
constexpr int N_GENERAL_TEST_STORE = N_GENERAL_TEST+4;
constexpr int K_GENERAL_TEST_STORE = K_GENERAL_TEST+4;
constexpr int K_GENERAL_UNIT_STORE = K_GENERAL_UNIT;
constexpr int N_GENERAL_TEST_WMMA = 2*N_GENERAL_TEST/WMMA_N_bin_TEST;

constexpr int M_SW = 8;
constexpr int N_SW = 8;
constexpr int K_SW = 8*K_GENERAL_UNIT;

constexpr int M_UNIT_BLOCK=512;
constexpr int N_UNIT_BLOCK=16;
constexpr int K_UNIT_BLOCK=512;

__device__ int32_t *Input_Stream, *Weight_Pos_Stream, *Weight_Neg_Stream;
__device__ int32_t *Weight_Pos_Temp, *Weight_Neg_Temp, *Input_Temp;
__device__ int16_t *Output_Pos_Temp, *Output_Neg_Temp;
extern bool Init_Temp;

struct Gemm_Unit_Param {
    int warp_m;
    int warp_n;
    int id_warp;
    int index_i_c;
    int index_w_c;
};

__device__ void 
compute_unit_gemm(
    int32_t* weight_pos_s,
    int32_t* weight_neg_s,
    int32_t* input_s,
    int (&acc_frag_pos)[I_UNROLL_TEST][W_UNROLL_TEST][WMMA_M_UNROLL][WMMA_N_UNROLL][2],
    int (&acc_frag_neg)[I_UNROLL_TEST][W_UNROLL_TEST][WMMA_M_UNROLL][WMMA_N_UNROLL][2],
    int16_t (&output_value_pos)[2*I_UNROLL_TEST][2*W_UNROLL_TEST],
    int16_t (&output_value_neg)[2*I_UNROLL_TEST][2*W_UNROLL_TEST],
    struct Gemm_Unit_Param gemm_param,
    int i_8
);

__device__ void
compute_shared_gemm_orn(
    int32_t* weight_pos_s,
    int32_t* weight_neg_s,
    int32_t* input_s,
    int (&acc_frag_pos)[I_UNROLL_TEST][W_UNROLL_TEST][WMMA_M_UNROLL][WMMA_N_UNROLL][2],
    int (&acc_frag_neg)[I_UNROLL_TEST][W_UNROLL_TEST][WMMA_M_UNROLL][WMMA_N_UNROLL][2],
    int16_t (&output_value_pos)[2*I_UNROLL_TEST][2*W_UNROLL_TEST],
    int16_t (&output_value_neg)[2*I_UNROLL_TEST][2*W_UNROLL_TEST],
    int k_sum,
    int index_k,
    int bin_config
);

struct Gemm_Param {
    int k_total;
};

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
);

__global__ void 
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
    const bool init);

#endif