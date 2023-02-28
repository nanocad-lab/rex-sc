#include <ATen/ATen.h>
#include <torch/extension.h>
#include <immintrin.h>
#include <omp.h>

namespace F = torch::nn::functional;

/*
 * Accelerated CPU implementation.
 * input: input tensor in NCHW format (default of PyTorch)
 * weight: weight tensor in NCHW format
 * add_config: 0 => fixed-point add
 *             >=1 -> simulated SC add. You can ignore these configs and only use add_config=1
 * precision_in: input and weight precision
 * precision_out: precision of intermediate addition results. Currently all additions are quantized
 */

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

struct OR2_Value {
    int value1;
    int value2;
};

//6-bit in, 8-bit out -> 12-bit multiplication, take 8 bits -> right shift 4 bits -> right shift=(precision_in*2-precision_out)
//prec_diff = precision_in*2-precision_out
//max value = 1<<(precision_in*2-precision_out)

int saturate_shift(int value, int max_value, int precision_diff) {
    int value_saturate = std::min(value, max_value);
    int round_bit = (value_saturate>>std::max(0, precision_diff-1))%2;
    return ((value_saturate>>precision_diff)+round_bit)<<precision_diff;
}
int fixed_add(int a, int b, int max_value, int precision_in, int precision_out) {
    int sum_full = a+b;
    return saturate_shift(sum_full, max_value, std::max(0,precision_in*2-precision_out));
}

// fixed_add_or and fixed_add_or2 are for SC testing purposes. You can ignore them
int fixed_add_or(int a, int b, int max_value, int precision_in, int precision_out) {
    int sum_full = a+b-((a*b)>>(precision_in*2));
    return saturate_shift(sum_full, max_value, std::max(0,precision_in*2-precision_out));
}
struct OR2_Value fixed_add_or2(struct OR2_Value a, struct OR2_Value b, int max_value, int precision_in, int precision_out) {
    int value1_out = a.value1 + b.value1 - int((long(a.value1)*long(a.value2)*long(b.value2))>>(precision_in*4)) - int((long(a.value2)*long(b.value1)*long(b.value2))>>(precision_in*4)) + int((long(a.value1)*long(a.value2)*long(b.value1)*long(b.value2))>>(precision_in*6));
    int value2_out = a.value2 + b.value2 - int((long(a.value1)*long(a.value2)*long(b.value1))>>(precision_in*4)) - int((long(a.value1)*long(b.value1)*long(b.value2))>>(precision_in*4)) + int((long(a.value1)*long(a.value2)*long(b.value1)*long(b.value2))>>(precision_in*6));
    value1_out = saturate_shift(value1_out, max_value, std::max(0,precision_in*2-precision_out));
    value2_out = saturate_shift(value2_out, max_value, std::max(0,precision_in*2-precision_out));
    struct OR2_Value output = {value1_out, value2_out};
    return output;
}

void compute_saturate_general(
        const int32_t* input_stream,
        const int32_t* weight_pos_stream,
        const int32_t* weight_neg_stream,
        int32_t* output_stream,
        int stride_w,
        int stride_h,
        struct Compute_Param c_param,
        struct Batch_Steps b_steps,
        int add_config, //0 = fixed; 1 = simulated or
        int precision_in,
        int precision_out
    ) {
    // printf("Entered 1din\n");
    
    int precision_max = std::min(31, std::max(precision_in*2, precision_out));
    int max_value = ((1<<precision_max)-1);
    int precision_diff = std::max(0,precision_in*2-precision_out);
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
    #pragma omp parallel for
    for(int batch=0; batch<c_param.batches; batch++) {
        const int32_t* input_stream_batch = input_stream + batch*b_steps.i_bin_batch_step;
        int32_t* output_stream_batch = output_stream + batch*b_steps.o_batch_step;
        int c_out=0;

        for(; c_out<c_param.c_outs; c_out++) {
            const int32_t* weight_pos_cout = weight_pos_stream + c_out*c_param.c_ins*c_param.w_w_ins*c_param.w_h_ins;
            const int32_t* weight_neg_cout = weight_neg_stream + c_out*c_param.c_ins*c_param.w_w_ins*c_param.w_h_ins;
            int32_t* output_stream_cout = output_stream_batch + c_out*((c_param.i_w_ins-c_param.w_w_ins)/stride_w+1)*((c_param.i_h_ins-c_param.w_h_ins)/stride_h+1);
            for(int win_c=0; win_c<(c_param.i_w_ins-c_param.w_w_ins)/stride_w+1; win_c++) {
                int32_t* output_stream_win = output_stream_cout + win_c*((c_param.i_h_ins-c_param.w_h_ins)/stride_h+1);
                for(int hin_c=0; hin_c<(c_param.i_h_ins-c_param.w_h_ins)/stride_h+1; hin_c++) {

                    if (add_config<2) {
                        int sum_pos = 0;
                        int sum_neg = 0;

                        // You might want to start from here
                        // Input channel loop
                        for(int c_in=0; c_in<c_param.c_ins; c_in++) {
                            const int32_t* input_stream_cin = input_stream_batch + c_in*c_param.i_w_ins*c_param.i_h_ins;
                            const int32_t* weight_pos_cin = weight_pos_cout + c_in*c_param.w_w_ins*c_param.w_h_ins;
                            const int32_t* weight_neg_cin = weight_neg_cout + c_in*c_param.w_w_ins*c_param.w_h_ins;
                            // Weight width loop (actually the H in NCHW)
                            for(int wk_c=0; wk_c<c_param.w_w_ins; wk_c++) {
                                const int32_t* input_stream_wk = input_stream_cin + (win_c*stride_w+wk_c)*c_param.i_h_ins + hin_c*stride_h;
                                const int32_t* weight_pos_wk = weight_pos_cin + wk_c*c_param.w_h_ins;
                                const int32_t* weight_neg_wk = weight_neg_cin + wk_c*c_param.w_h_ins;
                                // Weight height loop (actually the W in NCHW)
                                for(int hk_c=0; hk_c<c_param.w_h_ins; hk_c++) {
                                    int input_c = input_stream_wk[hk_c];
                                    int weight_pos_c = weight_pos_wk[hk_c];
                                    int weight_neg_c = weight_neg_wk[hk_c];
                                    sum_pos = (*add)(sum_pos, input_c*weight_pos_c, max_value, precision_in, precision_out);
                                    sum_neg = (*add)(sum_neg, input_c*weight_neg_c, max_value, precision_in, precision_out);
                                }
                            }
                        }
                        output_stream_win[hin_c] = sum_pos - sum_neg;
                    }
                    // Ignore these
                    else {
                        struct OR2_Value sum_pos = {0,0};
                        struct OR2_Value sum_neg = {0,0};

                        int cin_wk_hk_c = 0;
                        for(; cin_wk_hk_c+1<c_param.c_ins*c_param.w_w_ins*c_param.w_h_ins; cin_wk_hk_c+=2) {
                            int cin_0 = (cin_wk_hk_c+0)/(c_param.w_w_ins*c_param.w_h_ins);
                            int cin_1 = (cin_wk_hk_c+1)/(c_param.w_w_ins*c_param.w_h_ins);
                            int wk_c_0 = ((cin_wk_hk_c+0)%(c_param.w_w_ins*c_param.w_h_ins))/c_param.w_h_ins;
                            int wk_c_1 = ((cin_wk_hk_c+1)%(c_param.w_w_ins*c_param.w_h_ins))/c_param.w_h_ins;
                            int hk_c_0 = (cin_wk_hk_c+0)%c_param.w_h_ins;
                            int hk_c_1 = (cin_wk_hk_c+1)%c_param.w_h_ins;
                            struct OR2_Value input_c;
                            struct OR2_Value weight_pos_c;
                            struct OR2_Value weight_neg_c;

                            input_c = {input_stream_batch[cin_0*c_param.i_w_ins*c_param.i_h_ins+(win_c*stride_w+wk_c_0)*c_param.i_h_ins+hin_c*stride_h+hk_c_0],
                                       input_stream_batch[cin_1*c_param.i_w_ins*c_param.i_h_ins+(win_c*stride_w+wk_c_1)*c_param.i_h_ins+hin_c*stride_h+hk_c_1]};
                            weight_pos_c = {weight_pos_cout[cin_wk_hk_c+0], weight_pos_cout[cin_wk_hk_c+1]};
                            weight_neg_c = {weight_neg_cout[cin_wk_hk_c+0], weight_neg_cout[cin_wk_hk_c+1]};
                            struct OR2_Value product_pos = {input_c.value1*weight_pos_c.value1, input_c.value2*weight_pos_c.value2};
                            struct OR2_Value product_neg = {input_c.value1*weight_neg_c.value1, input_c.value2*weight_neg_c.value2};
                            sum_pos = fixed_add_or2(sum_pos, product_pos, max_value, precision_in, precision_out);
                            sum_neg = fixed_add_or2(sum_neg, product_neg, max_value, precision_in, precision_out);
                            // printf("Sum_pos_0 %d, Sum_pos_1 %d, produce_pos_0 %d, product_pos_1 %d\n", sum_pos.value1, sum_pos.value2, product_pos.value1, product_pos.value2);
                            // printf("Sum_neg_0 %d, Sum_neg_1 %d, produce_neg_0 %d, product_neg_1 %d\n", sum_neg.value1, sum_neg.value2, product_neg.value1, product_neg.value2);
                        }
                        if(cin_wk_hk_c<c_param.c_ins*c_param.w_w_ins*c_param.w_h_ins) {
                            int c_in = cin_wk_hk_c/(c_param.w_w_ins*c_param.w_h_ins);
                            int wk_c = (cin_wk_hk_c%(c_param.w_w_ins*c_param.w_h_ins)) / c_param.w_h_ins;
                            int hk_c = cin_wk_hk_c % c_param.w_h_ins;
                            int input_c = input_stream_batch[c_in*c_param.i_w_ins*c_param.i_h_ins+(win_c*stride_w+wk_c)*c_param.i_h_ins+hin_c*stride_h+hk_c];
                            int weight_pos_c = weight_pos_cout[cin_wk_hk_c];
                            int weight_neg_c = weight_neg_cout[cin_wk_hk_c];
                            struct OR2_Value product_pos = {input_c*weight_pos_c, 0};
                            struct OR2_Value product_neg = {input_c*weight_neg_c, 0};
                            sum_pos = fixed_add_or2(sum_pos, product_pos, max_value, precision_in, precision_out);
                            sum_neg = fixed_add_or2(sum_neg, product_neg, max_value, precision_in, precision_out);
                        }
                        output_stream_win[hin_c] = sum_pos.value1+sum_pos.value2 - sum_neg.value1-sum_neg.value2;
                    }
                }
            }
        }
    }
}

torch::Tensor conv2d_saturate(torch::Tensor input, 
        torch::Tensor weight, 
        int add_config,
        int precision_in,
        int precision_out,
        at::IntArrayRef padding, 
        at::IntArrayRef stride) {
    // printf("Inside func\n");
    auto input_pad = F::pad(input, F::PadFuncOptions({padding[0], padding[0], padding[1], padding[1]})/*.mode(torch::kReplicate)*/);
    auto compare_type = torch::kInt32;
    int precision_2 = 1<<precision_in;
    auto input_split = (input_pad*precision_2).clamp(0, precision_2-1).round().to(compare_type).clone();
    auto weight_pos = (weight*precision_2).clamp(0, precision_2-1).round().to(compare_type);
    auto weight_neg = (-(weight*precision_2).clamp(1-precision_2, 0)).round().to(compare_type);

    auto weight_size = weight_pos.sizes();
    auto input_size = input_split.sizes();
    
    //Output tensor preparation
    auto output_tensor = torch::zeros({input_size[0], weight_size[0], (input_size[2]-weight_size[2])/stride[0]+1, (input_size[3]-weight_size[3])/stride[1]+1}, at::TensorOptions().dtype(torch::kInt32));
    const int o_batch_step = weight_size[0] * ((input_size[2]-weight_size[2])/stride[0]+1) * ((input_size[3]-weight_size[3])/stride[1]+1);
    const int i_bin_batch_step = input_size[1] * input_size[2] * input_size[3];
    struct Compute_Param c_param = {0, 0, input_size[0], input_size[1], input_size[2], input_size[3], weight_size[0], weight_size[2], weight_size[3]};
    struct Batch_Steps b_steps = {o_batch_step, i_bin_batch_step, 0, 0};
    
    // printf("Inside func\n");
    //Direct conv
    compute_saturate_general (
        input_split.data_ptr<int32_t>(),
        weight_pos.data_ptr<int32_t>(),
        weight_neg.data_ptr<int32_t>(),
        output_tensor.data_ptr<int32_t>(),
        stride[0],
        stride[1],
        c_param,
        b_steps,
        add_config,
        precision_in,
        precision_out
    );
    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_saturate", &conv2d_saturate, "Fixed forward generic version");
}
