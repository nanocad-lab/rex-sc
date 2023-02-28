#include "sc_device.cuh"

using namespace nvcuda;

// Random number state. Using Philox4 as it's the fastest.
using curand_choice = curandStatePhilox4_32_10_t;
__device__ curand_choice *rand_states;
bool rand_init = false;

const int Global_Blocks = 10000;
const int Global_Threads = 32;
const int Local_Blocks = 1;
const int Local_Size = Global_Threads*Local_Blocks;

__global__
void or_approx_1_forward_cuda(
    float* __restrict__ input,
    float* __restrict__ output,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    for(int i=index_gen; i<size; i+=stride_gen) {
        float input_c = input[i];
        output[i] = or_approx_1_forward_scalar(input_c);
    }
}

__global__
void or_approx_2_forward_cuda(
    float* __restrict__ input,
    float* __restrict__ output,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    for(int i=index_gen; i<size; i+=stride_gen) {
        float input_c = input[i];
        output[i] = or_approx_2_forward_scalar(input_c);
    }
}

__global__
void or_approx_3_forward_cuda(
    float* __restrict__ input,
    float* __restrict__ output,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    for(int i=index_gen; i<size; i+=stride_gen) {
        float input_c = input[i];
        output[i] = or_approx_3_forward_scalar(input_c);
    }
}

__global__
void or_approx_4_forward_cuda(
    float* __restrict__ input,
    float* __restrict__ output,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    for(int i=index_gen; i<size; i+=stride_gen) {
        float input_c = input[i];
        output[i] = or_approx_4_forward_scalar(input_c);
    }
}

__global__
void ar_approx_1_forward_cuda(
    float* __restrict__ input,
    float* __restrict__ output,
    int size,
    int n) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    for (int i=index_gen; i<size; i+=stride_gen) {
        float input_c = input[i];
        output[i] = ar_approx_1_forward_scalar(input_c, n);
    }
}

__global__
void or_approx_1_backward_cuda(
    float* __restrict__ grad_output,
    half* __restrict__ sum,
    float* __restrict__ grad_input,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    for(int i=index_gen; i<size; i+=stride_gen) {
        float x = __half2float(sum[i]);
        grad_input[i] = or_approx_1_backward_scalar(grad_output[i],x);
    }
}

__global__
void or_approx_2_backward_cuda(
    float* __restrict__ grad_output,
    half* __restrict__ sum,
    float* __restrict__ grad_input,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    for(int i=index_gen; i<size; i+=stride_gen) {
        float x = __half2float(sum[i]);
        grad_input[i] = or_approx_2_backward_scalar(grad_output[i],x);
    }
}

__global__
void or_approx_3_backward_cuda(
    float* __restrict__ grad_output,
    half* __restrict__ sum,
    float* __restrict__ grad_input,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    for(int i=index_gen; i<size; i+=stride_gen) {
        float x = __half2float(sum[i]);
        grad_input[i] = or_approx_3_backward_scalar(grad_output[i],x);
    }
}

__global__
void or_approx_4_backward_cuda(
    float* __restrict__ grad_output,
    half* __restrict__ sum,
    float* __restrict__ grad_input,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    for(int i=index_gen; i<size; i+=stride_gen) {
        float x = __half2float(sum[i]);
        grad_input[i] = or_approx_4_backward_scalar(grad_output[i],x);
    }
}

__global__
void ar_approx_1_backward_cuda(
    float* __restrict__ grad_output,
    half* __restrict__ sum,
    float* __restrict__ grad_input,
    int size,
    int n) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    for(int i=index_gen; i<size; i+=stride_gen) {
        float x = __half2float(sum[i]);
        grad_input[i] = ar_approx_1_backward_scalar(grad_output[i], x, n);
    }
}

__global__
void or_approx_n_forward_cuda(
    float* __restrict__ input,
    int or_n,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;
    
    float (*or_approx_forward_scalar)(float);
    switch(or_n) {
        case 1:
        or_approx_forward_scalar=&or_approx_1_forward_scalar;
        break;
        case 2:
        or_approx_forward_scalar=&or_approx_2_forward_scalar;
        break;
        case 3:
        or_approx_forward_scalar=&or_approx_3_forward_scalar;
        break;
        case 4:
        or_approx_forward_scalar=&or_approx_4_forward_scalar;
        break;
    }
    for(int i=index_gen; i<size; i+=stride_gen) {
        float input_c = input[i];
        input[i] = or_approx_forward_scalar(input_c);
    }
}

__global__
void or_approx_2_bias_correct_forward_cuda(
    half* __restrict__ input,
    half* __restrict__ output,
    float* __restrict__ bias_coef,
    float* __restrict__ std_coef,
    curand_choice* __restrict__ states,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    int i=index_gen*4;
    for (; i+3<size; i+=stride_gen*4) {
        float4 x_rand_4 = curand_normal4(states+index_gen);
        float* x_rand_4_ptr = reinterpret_cast<float*>(&x_rand_4);
        for (int index_rand=0; index_rand<4; index_rand++) {
            float input_c = __half2float(input[i+index_rand]);
            float x_rand = x_rand_4_ptr[index_rand];
            output[i+index_rand] = __float2half(or_approx_2_bias_correct_forward_scalar(input_c, x_rand, bias_coef, std_coef));
        }
    }
    for(; i<size; i++) {
        float input_c = __half2float(input[i]);
        float x_rand = curand_normal(states+index_gen);
        output[i] = __float2half(or_approx_2_bias_correct_forward_scalar(input_c, x_rand, bias_coef, std_coef));
    }
}

__global__
void or_approx_1_bias_correct_forward_both_cuda(
    float* __restrict__ input_pos,
    float* __restrict__ input_neg,
    float* __restrict__ output,
    float* __restrict__ bias_coef,
    float* __restrict__ std_coef,
    curand_choice* __restrict__ states,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    int i=index_gen*4;
    for (; i+3<size; i+=stride_gen*4) {
        float4 x_rand_4_pos = curand_normal4(states+index_gen);
        float* x_rand_4_pos_ptr = reinterpret_cast<float*>(&x_rand_4_pos);
        float4 x_rand_4_neg = curand_normal4(states+index_gen);
        float* x_rand_4_neg_ptr = reinterpret_cast<float*>(&x_rand_4_neg);
        for (int index_rand=0; index_rand<4; index_rand++) {
            float input_pos_c = input_pos[i+index_rand];
            float input_neg_c = input_neg[i+index_rand];
            float x_rand_pos = x_rand_4_pos_ptr[index_rand];
            float x_rand_neg = x_rand_4_neg_ptr[index_rand];
            float output_pos_c = or_approx_1_bias_correct_forward_scalar(input_pos_c, x_rand_pos, bias_coef, std_coef);
            float output_neg_c = or_approx_1_bias_correct_forward_scalar(input_neg_c, x_rand_neg, bias_coef, std_coef);
            output[i+index_rand] = output_pos_c - output_neg_c;
        }
    }
    for(; i<size; i++) {
        float input_pos_c = input_pos[i];
        float input_neg_c = input_neg[i];
        float x_rand_pos = curand_normal(states+index_gen);
        float x_rand_neg = curand_normal(states+index_gen);
        float output_pos_c = or_approx_1_bias_correct_forward_scalar(input_pos_c, x_rand_pos, bias_coef, std_coef);
        float output_neg_c = or_approx_1_bias_correct_forward_scalar(input_neg_c, x_rand_neg, bias_coef, std_coef);
        output[i] = output_pos_c - output_neg_c;
    }
}

__global__
void or_approx_2_bias_correct_forward_both_cuda(
    float* __restrict__ input_pos,
    float* __restrict__ input_neg,
    float* __restrict__ output,
    float* __restrict__ bias_coef,
    float* __restrict__ std_coef,
    curand_choice* __restrict__ states,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    int i=index_gen*4;
    for (; i+3<size; i+=stride_gen*4) {
        float4 x_rand_4_pos = curand_normal4(states+index_gen);
        float* x_rand_4_pos_ptr = reinterpret_cast<float*>(&x_rand_4_pos);
        float4 x_rand_4_neg = curand_normal4(states+index_gen);
        float* x_rand_4_neg_ptr = reinterpret_cast<float*>(&x_rand_4_neg);
        for (int index_rand=0; index_rand<4; index_rand++) {
            float input_pos_c = input_pos[i+index_rand];
            float input_neg_c = input_neg[i+index_rand];
            float x_rand_pos = x_rand_4_pos_ptr[index_rand];
            float x_rand_neg = x_rand_4_neg_ptr[index_rand];
            float output_pos_c = or_approx_2_bias_correct_forward_scalar(input_pos_c, x_rand_pos, bias_coef, std_coef);
            float output_neg_c = or_approx_2_bias_correct_forward_scalar(input_neg_c, x_rand_neg, bias_coef, std_coef);
            output[i+index_rand] = output_pos_c - output_neg_c;
        }
    }
    for(; i<size; i++) {
        float input_pos_c = input_pos[i];
        float input_neg_c = input_neg[i];
        float x_rand_pos = curand_normal(states+index_gen);
        float x_rand_neg = curand_normal(states+index_gen);
        float output_pos_c = or_approx_2_bias_correct_forward_scalar(input_pos_c, x_rand_pos, bias_coef, std_coef);
        float output_neg_c = or_approx_2_bias_correct_forward_scalar(input_neg_c, x_rand_neg, bias_coef, std_coef);
        output[i] = output_pos_c - output_neg_c;
    }
}

__global__
void or_approx_3_bias_correct_forward_both_cuda(
    float* __restrict__ input_pos,
    float* __restrict__ input_neg,
    float* __restrict__ output,
    float* __restrict__ bias_coef,
    float* __restrict__ std_coef,
    curand_choice* __restrict__ states,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    int i=index_gen*4;
    for (; i+3<size; i+=stride_gen*4) {
        float4 x_rand_4_pos = curand_normal4(states+index_gen);
        float* x_rand_4_pos_ptr = reinterpret_cast<float*>(&x_rand_4_pos);
        float4 x_rand_4_neg = curand_normal4(states+index_gen);
        float* x_rand_4_neg_ptr = reinterpret_cast<float*>(&x_rand_4_neg);
        for (int index_rand=0; index_rand<4; index_rand++) {
            float input_pos_c = input_pos[i+index_rand];
            float input_neg_c = input_neg[i+index_rand];
            float x_rand_pos = x_rand_4_pos_ptr[index_rand];
            float x_rand_neg = x_rand_4_neg_ptr[index_rand];
            float output_pos_c = or_approx_3_bias_correct_forward_scalar(input_pos_c, x_rand_pos, bias_coef, std_coef);
            float output_neg_c = or_approx_3_bias_correct_forward_scalar(input_neg_c, x_rand_neg, bias_coef, std_coef);
            output[i+index_rand] = output_pos_c - output_neg_c;
        }
    }
    for(; i<size; i++) {
        float input_pos_c = input_pos[i];
        float input_neg_c = input_neg[i];
        float x_rand_pos = curand_normal(states+index_gen);
        float x_rand_neg = curand_normal(states+index_gen);
        float output_pos_c = or_approx_3_bias_correct_forward_scalar(input_pos_c, x_rand_pos, bias_coef, std_coef);
        float output_neg_c = or_approx_3_bias_correct_forward_scalar(input_neg_c, x_rand_neg, bias_coef, std_coef);
        output[i] = output_pos_c - output_neg_c;
    }
}

__global__
void or_approx_4_bias_correct_forward_both_cuda(
    float* __restrict__ input_pos,
    float* __restrict__ input_neg,
    float* __restrict__ output,
    float* __restrict__ bias_coef,
    float* __restrict__ std_coef,
    curand_choice* __restrict__ states,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    int i=index_gen*4;
    for (; i+3<size; i+=stride_gen*4) {
        float4 x_rand_4_pos = curand_normal4(states+index_gen);
        float* x_rand_4_pos_ptr = reinterpret_cast<float*>(&x_rand_4_pos);
        float4 x_rand_4_neg = curand_normal4(states+index_gen);
        float* x_rand_4_neg_ptr = reinterpret_cast<float*>(&x_rand_4_neg);
        for (int index_rand=0; index_rand<4; index_rand++) {
            float input_pos_c = input_pos[i+index_rand];
            float input_neg_c = input_neg[i+index_rand];
            float x_rand_pos = x_rand_4_pos_ptr[index_rand];
            float x_rand_neg = x_rand_4_neg_ptr[index_rand];
            float output_pos_c = or_approx_4_bias_correct_forward_scalar(input_pos_c, x_rand_pos, bias_coef, std_coef);
            float output_neg_c = or_approx_4_bias_correct_forward_scalar(input_neg_c, x_rand_neg, bias_coef, std_coef);
            output[i+index_rand] = output_pos_c - output_neg_c;
        }
    }
    for(; i<size; i++) {
        float input_pos_c = input_pos[i];
        float input_neg_c = input_neg[i];
        float x_rand_pos = curand_normal(states+index_gen);
        float x_rand_neg = curand_normal(states+index_gen);
        float output_pos_c = or_approx_4_bias_correct_forward_scalar(input_pos_c, x_rand_pos, bias_coef, std_coef);
        float output_neg_c = or_approx_4_bias_correct_forward_scalar(input_neg_c, x_rand_neg, bias_coef, std_coef);
        output[i] = output_pos_c - output_neg_c;
    }
}

__global__
void or_approx_n_bias_correct_forward_cuda(
    half* __restrict__ input,
    half* __restrict__ output,
    float* __restrict__ bias_coef,
    float* __restrict__ std_coef,
    curand_choice* __restrict__ states,
    int or_n,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    float (*or_approx_bias_correct_forward_scalar)(float, float, float*, float*);
    switch(or_n) {
        case 1:
        or_approx_bias_correct_forward_scalar=&or_approx_1_bias_correct_forward_scalar;
        break;
        case 2:
        or_approx_bias_correct_forward_scalar=&or_approx_2_bias_correct_forward_scalar;
        break;
        case 3:
        or_approx_bias_correct_forward_scalar=&or_approx_3_bias_correct_forward_scalar;
        break;
        case 4:
        or_approx_bias_correct_forward_scalar=&or_approx_4_bias_correct_forward_scalar;
        break;
    }

    int i=index_gen*4;
    for (; i+3<size; i+=stride_gen*4) {
        float4 x_rand_4 = curand_normal4(states+index_gen);
        float* x_rand_4_ptr = reinterpret_cast<float*>(&x_rand_4);
        for (int index_rand=0; index_rand<4; index_rand++) {
            float input_c = __half2float(input[i+index_rand]);
            float x_rand = x_rand_4_ptr[index_rand];
            output[i+index_rand] = __float2half(or_approx_bias_correct_forward_scalar(input_c, x_rand, bias_coef, std_coef));
        }
    }
    for(; i<size; i++) {
        float input_c = __half2float(input[i]);
        float x_rand = curand_normal(states+index_gen);
        output[i] = __float2half(or_approx_bias_correct_forward_scalar(input_c, x_rand, bias_coef, std_coef));
    }
}

__global__
void or_approx_n_bias_correct_forward_both_cuda(
    half* __restrict__ input_pos,
    half* __restrict__ input_neg,
    half* __restrict__ output,
    float* __restrict__ bias_coef,
    float* __restrict__ std_coef,
    curand_choice* __restrict__ states,
    int or_n,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    float (*or_approx_bias_correct_forward_scalar)(float, float, float*, float*);
    switch(or_n) {
        case 1:
        or_approx_bias_correct_forward_scalar=&or_approx_1_bias_correct_forward_scalar;
        break;
        case 2:
        or_approx_bias_correct_forward_scalar=&or_approx_2_bias_correct_forward_scalar;
        break;
        case 3:
        or_approx_bias_correct_forward_scalar=&or_approx_3_bias_correct_forward_scalar;
        break;
        case 4:
        or_approx_bias_correct_forward_scalar=&or_approx_4_bias_correct_forward_scalar;
        break;
    }

    int i=index_gen*4;
    for (; i+3<size; i+=stride_gen*4) {
        float4 x_rand_4_pos = curand_normal4(states+index_gen);
        float* x_rand_4_pos_ptr = reinterpret_cast<float*>(&x_rand_4_pos);
        float4 x_rand_4_neg = curand_normal4(states+index_gen);
        float* x_rand_4_neg_ptr = reinterpret_cast<float*>(&x_rand_4_neg);
        for (int index_rand=0; index_rand<4; index_rand++) {
            float input_pos_c = __half2float(input_pos[i+index_rand]);
            float input_neg_c = __half2float(input_neg[i+index_rand]);
            float x_rand_pos = x_rand_4_pos_ptr[index_rand];
            float x_rand_neg = x_rand_4_neg_ptr[index_rand];
            float output_pos_c = or_approx_bias_correct_forward_scalar(input_pos_c, x_rand_pos, bias_coef, std_coef);
            float output_neg_c = or_approx_bias_correct_forward_scalar(input_neg_c, x_rand_neg, bias_coef, std_coef);
            output[i+index_rand] = __float2half(output_pos_c - output_neg_c);
        }
    }
    for(; i<size; i++) {
        float input_pos_c = __half2float(input_pos[i]);
        float input_neg_c = __half2float(input_neg[i]);
        float x_rand_pos = curand_normal(states+index_gen);
        float x_rand_neg = curand_normal(states+index_gen);
        float output_pos_c = or_approx_bias_correct_forward_scalar(input_pos_c, x_rand_pos, bias_coef, std_coef);
        float output_neg_c = or_approx_bias_correct_forward_scalar(input_neg_c, x_rand_neg, bias_coef, std_coef);
        output[i] = __float2half(output_pos_c - output_neg_c);
    }
}

__global__
void or_approx_2_bias_correct_backward_cuda(
    half* __restrict__ grad_output,
    half* __restrict__ input,
    half* __restrict__ grad_input,
    float* __restrict__ bias_coef,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    for (int i=index_gen; i<size; i+=stride_gen) {
        // OR_2 forward
        float input_c = __half2float(input[i]);
        grad_input[i] = __float2half(or_approx_2_bias_correct_backward_scalar(__half2float(grad_output[i]), input_c, bias_coef));
    }
}

__global__
void or_approx_n_bias_correct_backward_cuda(
    float* __restrict__ grad_output,
    half* __restrict__ input,
    float* __restrict__ grad_input,
    float* __restrict__ bias_coef,
    int or_n,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    float (*or_approx_bias_correct_backward_scalar)(float, float, float*);
    switch(or_n) {
        case 1:
        or_approx_bias_correct_backward_scalar=&or_approx_1_bias_correct_backward_scalar;
        break;
        case 2:
        or_approx_bias_correct_backward_scalar=&or_approx_2_bias_correct_backward_scalar;
        break;
        case 3:
        or_approx_bias_correct_backward_scalar=&or_approx_3_bias_correct_backward_scalar;
        break;
        case 4:
        or_approx_bias_correct_backward_scalar=&or_approx_4_bias_correct_backward_scalar;
        break;
    }

    for (int i=index_gen; i<size; i+=stride_gen) {
        // OR_2 forward
        float input_c = __half2float(input[i]);
        grad_input[i] = or_approx_bias_correct_backward_scalar(grad_output[i], input_c, bias_coef);
    }
}

__global__
void or_approx_1_bias_correct_backward_both_cuda(
    float* __restrict__ grad_output,
    half* __restrict__ input_pos,
    half* __restrict__ input_neg,
    float* __restrict__ grad_input_pos,
    float* __restrict__ grad_input_neg,
    float* __restrict__ bias_coef,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    for (int i=index_gen; i<size; i+=stride_gen) {
        // OR_2 forward
        float input_pos_c = __half2float(input_pos[i]);
        float input_neg_c = __half2float(input_neg[i]);
        float grad_output_c = grad_output[i];
        float grad_input_pos_c = or_approx_1_bias_correct_backward_scalar(grad_output_c, input_pos_c, bias_coef);
        float grad_input_neg_c = or_approx_1_bias_correct_backward_scalar(-grad_output_c, input_neg_c, bias_coef);
        grad_input_pos[i] = grad_input_pos_c;
        grad_input_neg[i] = grad_input_neg_c;
    }
}

__global__
void or_approx_2_bias_correct_backward_both_cuda(
    float* __restrict__ grad_output,
    half* __restrict__ input_pos,
    half* __restrict__ input_neg,
    float* __restrict__ grad_input_pos,
    float* __restrict__ grad_input_neg,
    float* __restrict__ bias_coef,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    for (int i=index_gen; i<size; i+=stride_gen) {
        // OR_2 forward
        float input_pos_c = __half2float(input_pos[i]);
        float input_neg_c = __half2float(input_neg[i]);
        float grad_output_c = grad_output[i];
        float grad_input_pos_c = or_approx_2_bias_correct_backward_scalar(grad_output_c, input_pos_c, bias_coef);
        float grad_input_neg_c = or_approx_2_bias_correct_backward_scalar(-grad_output_c, input_neg_c, bias_coef);
        grad_input_pos[i] = grad_input_pos_c;
        grad_input_neg[i] = grad_input_neg_c;
    }
}

__global__
void or_approx_3_bias_correct_backward_both_cuda(
    float* __restrict__ grad_output,
    half* __restrict__ input_pos,
    half* __restrict__ input_neg,
    float* __restrict__ grad_input_pos,
    float* __restrict__ grad_input_neg,
    float* __restrict__ bias_coef,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    for (int i=index_gen; i<size; i+=stride_gen) {
        // OR_2 forward
        float input_pos_c = __half2float(input_pos[i]);
        float input_neg_c = __half2float(input_neg[i]);
        float grad_output_c = grad_output[i];
        float grad_input_pos_c = or_approx_3_bias_correct_backward_scalar(grad_output_c, input_pos_c, bias_coef);
        float grad_input_neg_c = or_approx_3_bias_correct_backward_scalar(-grad_output_c, input_neg_c, bias_coef);
        grad_input_pos[i] = grad_input_pos_c;
        grad_input_neg[i] = grad_input_neg_c;
    }
}

__global__
void or_approx_4_bias_correct_backward_both_cuda(
    float* __restrict__ grad_output,
    half* __restrict__ input_pos,
    half* __restrict__ input_neg,
    float* __restrict__ grad_input_pos,
    float* __restrict__ grad_input_neg,
    float* __restrict__ bias_coef,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    for (int i=index_gen; i<size; i+=stride_gen) {
        // OR_2 forward
        float input_pos_c = __half2float(input_pos[i]);
        float input_neg_c = __half2float(input_neg[i]);
        float grad_output_c = grad_output[i];
        float grad_input_pos_c = or_approx_4_bias_correct_backward_scalar(grad_output_c, input_pos_c, bias_coef);
        float grad_input_neg_c = or_approx_4_bias_correct_backward_scalar(-grad_output_c, input_neg_c, bias_coef);
        grad_input_pos[i] = grad_input_pos_c;
        grad_input_neg[i] = grad_input_neg_c;
    }
}

__global__
void or_approx_n_bias_correct_backward_both_cuda(
    half* __restrict__ grad_output,
    half* __restrict__ input_pos,
    half* __restrict__ input_neg,
    half* __restrict__ grad_input_pos,
    half* __restrict__ grad_input_neg,
    float* __restrict__ bias_coef,
    int or_n,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    float (*or_approx_bias_correct_backward_scalar)(float, float, float*);
    switch(or_n) {
        case 1:
        or_approx_bias_correct_backward_scalar=&or_approx_1_bias_correct_backward_scalar;
        break;
        case 2:
        or_approx_bias_correct_backward_scalar=&or_approx_2_bias_correct_backward_scalar;
        break;
        case 3:
        or_approx_bias_correct_backward_scalar=&or_approx_3_bias_correct_backward_scalar;
        break;
        case 4:
        or_approx_bias_correct_backward_scalar=&or_approx_4_bias_correct_backward_scalar;
        break;
    }

    for (int i=index_gen; i<size; i+=stride_gen) {
        // OR_2 forward
        float input_pos_c = __half2float(input_pos[i]);
        float input_neg_c = __half2float(input_neg[i]);
        float grad_output_c = __half2float(grad_output[i]);
        float grad_input_pos_c = or_approx_bias_correct_backward_scalar(grad_output_c, input_pos_c, bias_coef);
        float grad_input_neg_c = or_approx_bias_correct_backward_scalar(-grad_output_c, input_neg_c, bias_coef);
        grad_input_pos[i] = __float2half(grad_input_pos_c);
        grad_input_neg[i] = __float2half(grad_input_neg_c);
    }
}

__global__
void bias_correct_forward_cuda(
    float* __restrict__ input,
    float* __restrict__ bias_coef,
    float* __restrict__ std_coef,
    curand_choice* __restrict__ states,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    for(int i=index_gen; i<size; i+=stride_gen) {
        float input_c = input[i];
        float x_bias = bias_coef[0] + bias_coef[1]*input_c + bias_coef[2]*powf(input_c, 2.f) + bias_coef[3]*powf(input_c,3.f) + bias_coef[4]*powf(input_c,4.f) + bias_coef[5]*powf(input_c,5.f);
        float x_std = std_coef[0] + std_coef[1]*input_c + std_coef[2]*powf(input_c, 2.f) + std_coef[3]*powf(input_c,3.f) + std_coef[4]*powf(input_c,4.f) + std_coef[5]*powf(input_c,5.f);
        float x_rand = curand_normal(states+index_gen);
        input[i] = input_c + x_bias+x_std*x_rand;
    }
}

__global__
void bias_correct_backward_cuda(
    float* __restrict__ grad_output,
    float* __restrict__ or_sum,
    float* __restrict__ bias_coef,
    int size) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    for(int i=index_gen; i<size; i+=stride_gen) {
        float x = or_sum[i];
        float x_dot = 1.f+bias_coef[1] + bias_coef[2]*2.f*x + bias_coef[3]*3.f*powf(x,2.f) + bias_coef[4]*4.f*powf(x,3.f) + bias_coef[5]*5.f*powf(x,4.f);
        or_sum[i] = x_dot*grad_output[i];
    }
}

__global__
void rand_state_init(
    curand_choice* __restrict__ states) {
    int index_gen = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_gen = blockDim.x * gridDim.x;

    curand_init(0, index_gen, 0, states+index_gen);
}

torch::Tensor or_approx_n_forward_acc(
    torch::Tensor input,
    int or_n) {
    auto device = input.device();
    int device_index = device.index();
    cudaSetDevice(int(device_index));

    torch::Tensor input_f32 = input.to(torch::kFloat32);
    torch::Tensor output = torch::empty_like(input_f32);
    int input_size = input.numel();

    switch(or_n) {
        case 1:
        or_approx_1_forward_cuda <<<Global_Blocks,Global_Threads>>>(
            input_f32.data_ptr<float>(),
            output.data_ptr<float>(),
            input_size);
        break;
        case 2:
        or_approx_2_forward_cuda <<<Global_Blocks,Global_Threads>>>(
            input_f32.data_ptr<float>(),
            output.data_ptr<float>(),
            input_size);
        break;
        case 3:
        or_approx_3_forward_cuda <<<Global_Blocks,Global_Threads>>>(
            input_f32.data_ptr<float>(),
            output.data_ptr<float>(),
            input_size);
        break;
        case 4:
        or_approx_4_forward_cuda <<<Global_Blocks,Global_Threads>>>(
            input_f32.data_ptr<float>(),
            output.data_ptr<float>(),
            input_size);
        break;
    }
    return output;
}

torch::Tensor ar_approx_1_forward_acc(
    torch::Tensor input,
    int or_n) {
    auto device = input.device();
    int device_index = device.index();
    cudaSetDevice(int(device_index));

    torch::Tensor input_f32 = input.to(torch::kFloat32);
    torch::Tensor output = torch::empty_like(input_f32);
    int input_size = input.numel();

    ar_approx_1_forward_cuda <<<Global_Blocks,Global_Threads>>>(
        input_f32.data_ptr<float>(),
        output.data_ptr<float>(),
        input_size,
        or_n);
    return output;
}

torch::Tensor or_approx_n_backward_acc(
    torch::Tensor grad_output,
    torch::Tensor input,
    int or_n) {
    auto device = input.device();
    int device_index = device.index();
    cudaSetDevice(int(device_index));

    torch::Tensor input_f16 = input.to(torch::kFloat16);
    torch::Tensor grad_input = grad_output.clone();
    int input_size = grad_output.numel();

    switch(or_n) {
        case 1:
        or_approx_1_backward_cuda <<<Global_Blocks,Global_Threads>>>(
            grad_output.data_ptr<float>(),
            reinterpret_cast<half*>(input_f16.data_ptr()),
            grad_input.data_ptr<float>(),
            input_size);
        break;
        case 2:
        or_approx_2_backward_cuda <<<Global_Blocks,Global_Threads>>>(
            grad_output.data_ptr<float>(),
            reinterpret_cast<half*>(input_f16.data_ptr()),
            grad_input.data_ptr<float>(),
            input_size);
        break;
        case 3:
        or_approx_3_backward_cuda <<<Global_Blocks,Global_Threads>>>(
            grad_output.data_ptr<float>(),
            reinterpret_cast<half*>(input_f16.data_ptr()),
            grad_input.data_ptr<float>(),
            input_size);
        break;
        case 4:
        or_approx_4_backward_cuda <<<Global_Blocks,Global_Threads>>>(
            grad_output.data_ptr<float>(),
            reinterpret_cast<half*>(input_f16.data_ptr()),
            grad_input.data_ptr<float>(),
            input_size);
        break;
    }

    return grad_input;
}

torch::Tensor ar_approx_1_backward_acc(
    torch::Tensor grad_output,
    torch::Tensor input,
    int or_n) {
    auto device = input.device();
    int device_index = device.index();
    cudaSetDevice(int(device_index));

    torch::Tensor input_f16 = input.to(torch::kFloat16);
    torch::Tensor grad_input = grad_output.clone();
    int input_size = grad_output.numel();

    ar_approx_1_backward_cuda <<<Global_Blocks,Global_Threads>>>(
        grad_output.data_ptr<float>(),
        reinterpret_cast<half*>(input_f16.data_ptr()),
        grad_input.data_ptr<float>(),
        input_size,
        or_n);
    return grad_input;
}

torch::Tensor or_approx_n_forward_bias_std_acc(
    torch::Tensor input,
    torch::Tensor bias_coef,
    torch::Tensor std_coef,
    int or_n) {
    auto device = input.device();
    int device_index = device.index();
    cudaSetDevice(int(device_index));

    torch::Tensor input_fp = input.to(torch::kFloat16);
    int input_size = input.numel();
    torch::Tensor output_fp = torch::empty_like(input_fp);

    if (!rand_init) {
        cudaMalloc(&rand_states, Global_Blocks*Global_Threads*sizeof(curand_choice));
        rand_state_init <<<Global_Blocks,Global_Threads>>> (rand_states);
        rand_init = true;
    }
    or_approx_n_bias_correct_forward_cuda <<<Global_Blocks, Global_Threads>>> (
        reinterpret_cast<half*>(input_fp.data_ptr()),
        reinterpret_cast<half*>(output_fp.data_ptr()),
        bias_coef.data_ptr<float>(),
        std_coef.data_ptr<float>(),
        rand_states,
        or_n,
        input_size);
    return output_fp;
}

torch::Tensor or_approx_n_forward_bias_std_both_acc(
    torch::Tensor input_pos,
    torch::Tensor input_neg,
    torch::Tensor bias_coef,
    torch::Tensor std_coef,
    int or_n) {
    auto device = input_pos.device();
    int device_index = device.index();
    cudaSetDevice(int(device_index));

    torch::Tensor input_pos_fp = input_pos.to(torch::kFloat32);
    torch::Tensor input_neg_fp = input_neg.to(torch::kFloat32);
    int input_size = input_pos.numel();
    torch::Tensor output_fp = torch::empty_like(input_pos_fp);

    if (!rand_init) {
        cudaMalloc(&rand_states, Global_Blocks*Global_Threads*sizeof(curand_choice));
        rand_state_init <<<Global_Blocks,Global_Threads>>> (rand_states);
        rand_init = true;
    }

    switch (or_n) {
        case 1:
        or_approx_1_bias_correct_forward_both_cuda <<<Global_Blocks, Global_Threads>>> (
            input_pos_fp.data_ptr<float>(),
            input_neg_fp.data_ptr<float>(),
            output_fp.data_ptr<float>(),
            bias_coef.data_ptr<float>(),
            std_coef.data_ptr<float>(),
            rand_states,
            input_size);
        break;
        case 2:
        or_approx_2_bias_correct_forward_both_cuda <<<Global_Blocks, Global_Threads>>> (
            input_pos_fp.data_ptr<float>(),
            input_neg_fp.data_ptr<float>(),
            output_fp.data_ptr<float>(),
            bias_coef.data_ptr<float>(),
            std_coef.data_ptr<float>(),
            rand_states,
            input_size);
        break;
        case 3:
        or_approx_3_bias_correct_forward_both_cuda <<<Global_Blocks, Global_Threads>>> (
            input_pos_fp.data_ptr<float>(),
            input_neg_fp.data_ptr<float>(),
            output_fp.data_ptr<float>(),
            bias_coef.data_ptr<float>(),
            std_coef.data_ptr<float>(),
            rand_states,
            input_size);
        break;
        case 4:
        or_approx_4_bias_correct_forward_both_cuda <<<Global_Blocks, Global_Threads>>> (
            input_pos_fp.data_ptr<float>(),
            input_neg_fp.data_ptr<float>(),
            output_fp.data_ptr<float>(),
            bias_coef.data_ptr<float>(),
            std_coef.data_ptr<float>(),
            rand_states,
            input_size);
        break;
    }
    return output_fp;
}

torch::Tensor or_approx_n_backward_bias_std_acc(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor bias_coef,
    int or_n) {
    auto device = input.device();
    int device_index = device.index();
    cudaSetDevice(int(device_index));

    torch::Tensor input_f16 = input.to(torch::kFloat16);
    torch::Tensor grad_input = torch::empty_like(grad_output);
    int input_size = grad_output.numel();

    or_approx_n_bias_correct_backward_cuda <<<Global_Blocks,Global_Threads>>> (
        grad_output.data_ptr<float>(),
        reinterpret_cast<half*>(input_f16.data_ptr()),
        grad_input.data_ptr<float>(),
        bias_coef.data_ptr<float>(),
        or_n,
        input_size);
    return grad_input;
}

std::vector<torch::Tensor> or_approx_n_backward_bias_std_both_acc(
    torch::Tensor grad_output,
    torch::Tensor input_pos,
    torch::Tensor input_neg,
    torch::Tensor bias_coef,
    int or_n) {
    auto device = input_pos.device();
    int device_index = device.index();
    cudaSetDevice(int(device_index));

    torch::Tensor input_pos_f16 = input_pos.to(torch::kFloat16);
    torch::Tensor input_neg_f16 = input_neg.to(torch::kFloat16);
    torch::Tensor grad_input_pos = torch::empty_like(grad_output);
    torch::Tensor grad_input_neg = torch::empty_like(grad_output);
    int input_size = grad_output.numel();

    switch (or_n) {
        case 1:
        or_approx_1_bias_correct_backward_both_cuda <<<Global_Blocks,Global_Threads>>> (
            grad_output.data_ptr<float>(),
            reinterpret_cast<half*>(input_pos_f16.data_ptr()),
            reinterpret_cast<half*>(input_neg_f16.data_ptr()),
            grad_input_pos.data_ptr<float>(),
            grad_input_neg.data_ptr<float>(),
            bias_coef.data_ptr<float>(),
            input_size);
        break;
        case 2:
        or_approx_2_bias_correct_backward_both_cuda <<<Global_Blocks,Global_Threads>>> (
            grad_output.data_ptr<float>(),
            reinterpret_cast<half*>(input_pos_f16.data_ptr()),
            reinterpret_cast<half*>(input_neg_f16.data_ptr()),
            grad_input_pos.data_ptr<float>(),
            grad_input_neg.data_ptr<float>(),
            bias_coef.data_ptr<float>(),
            input_size);
        break;
        case 3:
        or_approx_3_bias_correct_backward_both_cuda <<<Global_Blocks,Global_Threads>>> (
            grad_output.data_ptr<float>(),
            reinterpret_cast<half*>(input_pos_f16.data_ptr()),
            reinterpret_cast<half*>(input_neg_f16.data_ptr()),
            grad_input_pos.data_ptr<float>(),
            grad_input_neg.data_ptr<float>(),
            bias_coef.data_ptr<float>(),
            input_size);
        break;
        case 4:
        or_approx_4_bias_correct_backward_both_cuda <<<Global_Blocks,Global_Threads>>> (
            grad_output.data_ptr<float>(),
            reinterpret_cast<half*>(input_pos_f16.data_ptr()),
            reinterpret_cast<half*>(input_neg_f16.data_ptr()),
            grad_input_pos.data_ptr<float>(),
            grad_input_neg.data_ptr<float>(),
            bias_coef.data_ptr<float>(),
            input_size);
        break;
    }
    return {grad_input_pos, grad_input_neg};
}