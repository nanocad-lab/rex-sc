// #include <ATen/ATen.h>
#include <torch/extension.h>
#ifdef __AVX512F__
#include <immintrin.h>
#endif
#include <omp.h>
#include <algorithm>
#pragma once

#define POS_SEED 13
#define NEG_SEED 13

const int COMPUTE_CINS = 32;
const int SEED_4 = 8;
const int SEED_5 = 20;
const int SEED_6 = 36;
const int SEED_7 = 68;

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

// LFSR generator functions
template <class T>
inline T lfsr_8_s(T value) {
    return ((value/128)+(value/32)%2+(value/16)%2+(value/8)%2)%2+2*(value%128);
}

template <class T>
inline T lfsr_8_s_acc(T value) {
    switch(value) {
        case 1:
        return 0;
        case 0:
        return 2;
        default:
        return lfsr_8_s(value);
    }
}

template <class T>
inline T lfsr_7_s(T value) {
    return ((value/32)%2+value/64)%2+2*(value%64);
}

template <class T>
inline T lfsr_7_s_acc(T value) {
    switch(value) {
        case 1:
        return 0;
        case 0:
        return 2;
        default:
        return lfsr_7_s(value);
    }
}

template <class T>
inline T lfsr_6_s(T value) {
    return ((value/32)+(value/16)%2)%2+2*(value%32);
}

template <class T>
inline T lfsr_6_s_acc(T value) {
    switch(value) {
        case 1:
        return 0;
        case 0:
        return 2;
        default:
        return lfsr_6_s(value);
    }
}

template <class T>
inline T lfsr_5_s(T value) {
    return ((value/16)+(value/4)%2)%2+2*(value%16);
}

template <class T>
inline T lfsr_5_s_acc(T value) {
    switch(value) {
        case 1:
        return 0;
        case 0:
        return 2;
        default:
        return lfsr_5_s(value);
    }
}

template <class T>
inline T lfsr_4_s(T value) {
    return ((value/8)+(value/4)%2)%2+2*(value%8);
}

template <class T>
inline T lfsr_4_s_acc(T value) {
    switch(value) {
        case 1:
        return 0;
        case 0:
        return 2;
        default:
        return lfsr_4_s(value);
    }
}

template <class T>
inline T lfsr_3_s(T value) {
    return ((value/4)+(value/2)%2)%2+2*(value%4);
}

#ifdef __AVX512F__
inline __m512i extract_add(__m512i value_0, __m512i value_1, __m512i value, int width) {
    __m512i value_fb = _mm512_add_epi32(value_0, value_1);
    __m512i value_res = _mm512_srli_epi32(_mm512_slli_epi32(value_fb, 31), 31);
    __m512i value_fwd = _mm512_srli_epi32(_mm512_slli_epi32(value, 32-width+1), 32-width);
    return _mm512_add_epi32(value_res, value_fwd);
}

inline __m512i lfsr_7_512(__m512i value) {
    __m512i value_64 = _mm512_srli_epi32(value, 6);
    __m512i value_32 = _mm512_srli_epi32(value, 5);
    return extract_add(value_64, value_32, value, 7);
}

inline __m512i lfsr_6_512(__m512i value) {
    __m512i value_32 = _mm512_srli_epi32(value, 5);
    __m512i value_16 = _mm512_srli_epi32(value, 4);
    return extract_add(value_32, value_16, value, 6);
}

inline __m512i lfsr_5_512(__m512i value) {
    __m512i value_16 = _mm512_srli_epi32(value, 4);
    __m512i value_4 = _mm512_srli_epi32(value, 2);
    return extract_add(value_16, value_4, value, 5);
}

inline __m512i lfsr_4_512(__m512i value) {
    __m512i value_8 = _mm512_srli_epi32(value, 3);
    __m512i value_4 = _mm512_srli_epi32(value, 2);
    return extract_add(value_8, value_4, value, 4);
}

inline __m512i lfsr_3_512(__m512i value) {
    __m512i value_4 = _mm512_srli_epi32(value, 2);
    __m512i value_2 = _mm512_srli_epi32(value, 1);
    return extract_add(value_4, value_2, value, 3);
}
#endif



template <class T>
inline T lfsr_3_s_acc(T value) {
    switch(value) {
        case 1:
        return 0;
        case 0:
        return 2;
        default:
        return lfsr_3_s(value);
    }
}

inline float or_approx_1_forward_s(
    float input_c) {
    float input_exp = expf(-input_c);
    return 1-input_exp;
}

inline float or_approx_2_forward_s(
    float input_c) {
    float input_exp = expf(-input_c);
    float x = fmaf(-2.f,input_exp,2.f);
    return fmaf(-input_c,input_exp,x);
}

inline float or_approx_3_forward_s(
    float input_c) {
    float input_exp = expf(-input_c);
    float input_c_2 = input_c*input_c;
    float x = fmaf(-3.f,input_exp,3.f);
    x = fmaf(-2.f*input_c, input_exp, x);
    x = fmaf(-0.5f*input_c_2, input_exp, x);
    return x;
}

inline float or_approx_4_forward_s(
    float input_c) {
    float input_exp = expf(-input_c);
    float input_c_2 = input_c*input_c;
    float x = fmaf(-4.f, input_exp, 4.f);
    x = fmaf(-3.f*input_c, input_exp, x);
    float input_c_3 = input_c_2*input_c;
    x = fmaf(-input_c_2, input_exp, x);
    x = fmaf((-1.f/6.f)*input_c_3, input_exp, x);
    return x;
}

inline float or_approx_1_backward_s(
    float grad_output,
    float sum) {
    float sum_exp = expf(-sum);
    return sum_exp*grad_output;
}

inline float or_approx_2_backward_s(
    float grad_output,
    float sum) {
    float sum_exp = expf(-sum);
    float x_or = fmaf(sum, sum_exp, sum_exp);
    return x_or*grad_output;
}

inline float or_approx_3_backward_s(
    float grad_output,
    float sum) {
    float sum_exp = expf(-sum);
    float sum_2 = sum*sum;
    float x_or = fmaf(sum, sum_exp, sum_exp);
    x_or = fmaf(0.5f*sum_2, sum_exp, x_or);
    return x_or*grad_output;
}

inline float or_approx_4_backward_s(
    float grad_output,
    float sum) {
    float sum_exp = expf(-sum);
    float sum_2 = sum*sum;
    float x_or = fmaf(sum, sum_exp, sum_exp);
    float sum_3 = sum_2*sum;
    x_or = fmaf(0.5f*sum_2, sum_exp, x_or);
    x_or = fmaf((1.f/6.f)*sum_3, sum_exp, x_or);
    return x_or*grad_output;
}

inline float bias_correct_forward_s(
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

inline float bias_correct_backward_s(
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

inline float or_approx_1_bias_correct_forward_s(
    float input_c,
    float x_rand,
    float* __restrict__ bias_coef,
    float* __restrict__ std_coef) {
    float x=or_approx_1_forward_s(input_c);
    return bias_correct_forward_s(x, x_rand, bias_coef, std_coef);
}

inline float or_approx_2_bias_correct_forward_s(
    float input_c,
    float x_rand,
    float* __restrict__ bias_coef,
    float* __restrict__ std_coef) {
    float x=or_approx_2_forward_s(input_c);
    return bias_correct_forward_s(x, x_rand, bias_coef, std_coef);
}

inline float or_approx_3_bias_correct_forward_s(
    float input_c,
    float x_rand,
    float* __restrict__ bias_coef,
    float* __restrict__ std_coef) {
    float x=or_approx_3_forward_s(input_c);
    return bias_correct_forward_s(x, x_rand, bias_coef, std_coef);
}

inline float or_approx_4_bias_correct_forward_s(
    float input_c,
    float x_rand,
    float* __restrict__ bias_coef,
    float* __restrict__ std_coef) {
    float x=or_approx_4_forward_s(input_c);
    return bias_correct_forward_s(x, x_rand, bias_coef, std_coef);
}

inline float or_approx_1_bias_correct_backward_s(
    float grad_output,
    float input_c,
    float* __restrict__ bias_coef) {
    // OR_1 forward
    float input_exp = expf(-input_c);
    float x = 1-input_exp;
    // Bias backward
    float x_dot = bias_correct_backward_s(x, bias_coef);
    x_dot = x_dot * grad_output;
    // OR_2 backward
    return input_exp*x_dot;
}

inline float or_approx_2_bias_correct_backward_s(
    float grad_output,
    float input_c,
    float* __restrict__ bias_coef) {
    // OR_2 forward
    float input_exp = expf(-input_c);
    float x = fmaf(-2.f,input_exp,2.f);
    x = fmaf(-input_c,input_exp,x);
    // Bias backward
    float x_dot = bias_correct_backward_s(x, bias_coef);
    x_dot = x_dot * grad_output;
    // OR_2 backward
    float x_or = fmaf(input_c, input_exp, input_exp);
    return x_or*x_dot;
}

inline float or_approx_3_bias_correct_backward_s(
    float grad_output,
    float input_c,
    float* __restrict__ bias_coef) {
    // OR_3 forward
    float input_exp = expf(-input_c);
    float input_c_2 = input_c*input_c;
    float x = fmaf(-3.f,input_exp,3.f);
    x = fmaf(-2.f*input_c, input_exp, x);
    x = fmaf(-0.5f*input_c_2, input_exp, x);
    // Bias backward
    float x_dot = bias_correct_backward_s(x, bias_coef);
    x_dot = x_dot * grad_output;
    // OR_3 backward
    float x_or = fmaf(input_c, input_exp, input_exp);
    x_or = fmaf(0.5f*input_c_2, input_exp, x_or);
    return x_or*x_dot;
}

inline float or_approx_4_bias_correct_backward_s(
    float grad_output,
    float input_c,
    float* __restrict__ bias_coef) {
    // OR_4 forward
    float input_exp = expf(-input_c);
    float input_c_2 = input_c*input_c;
    float x = fmaf(-4.f, input_exp, 4.f);
    x = fmaf(-3.f*input_c, input_exp, x);
    float input_c_3 = input_c_2*input_c;
    x = fmaf(-input_c_2, input_exp, x);
    x = fmaf((-1.f/6.f)*input_c_3, input_exp, x);
    // Bias backward
    float x_dot = bias_correct_backward_s(x, bias_coef);
    x_dot = x_dot * grad_output;
    // OR_3 backward
    float x_or = fmaf(input_c, input_exp, input_exp);
    x_or = fmaf(0.5f*input_c_2, input_exp, x_or);
    x_or = fmaf((1.f/6.f)*input_c_3, input_exp, x_or);
    return x_or*x_dot;
}