#include "sc_cpu.hpp"

namespace F = torch::nn::functional;

/*
 * Accelerated CPU implementation. Only y-dimension fixed-point accumulation with LFSR generator is supported right now
 */

int weight_ind_oihw_host(int c_out, int c_in, int w_in, int h_in, int c_outs, int c_ins, int w_ins, int h_ins) {
    return c_out*c_ins*w_ins*h_ins + c_in*w_ins*h_ins + w_in*h_ins + h_in;
}

int weight_ind_ohwi_host(int c_out, int c_in, int w_in, int h_in, int c_outs, int c_ins, int w_ins, int h_ins) {
    return c_out*c_ins*w_ins*h_ins + w_in*h_ins*c_ins + h_in*c_ins + c_in;
}

int output_ind_nchw_host(int c_out, int c_in, int stride) {
    return c_out*stride+c_in;
}

int output_ind_nhwc_host(int c_out, int c_in, int stride) {
    return c_in*stride+c_out;
}

int or_act_n_host(int value, int n) {
    return std::min(value, n);
}

int32_t or_act_split_update_host(uint value, int (*or_act)(int, int), int n) {
    uint old_value = value>>16;
    uint new_value = (value<<16)>>16;
    return (old_value + or_act_n_host(new_value, n))<<16;
}

 int32_t or_act_analog_update_host(uint value, int prec) {
    uint max_value = (1u<<prec)-1;
    uint old_value = value>>16;
    uint new_value = (value<<16)>>16;
    return (old_value + std::min(new_value, max_value))<<16;
}

void stream_generation_or_general(
    const int32_t* __restrict__ weight_pos,
    const int32_t* __restrict__ weight_neg,
    int32_t* __restrict__ weight_pos_stream,
    int32_t* __restrict__ weight_neg_stream,
    int bit_length,
    int lfsr_length,
    int (*lfsr)(int),
    int gen_config,
    bool gen_mult, //Use lfsr_mult
    int c_outs,
    int c_ins,
    int w_ins,
    int h_ins,
    int total_width,
    int load_width,
    int load_wait,
    bool channels_last_weight) {

    int z_packs = (c_ins + COMPUTE_CINS-1) / COMPUTE_CINS;
    int seed_mult = 1;

    #pragma omp parallel for
    for(int i=0; i<c_outs*w_ins*h_ins*z_packs; i++) {

        int c_out = i/(w_ins*h_ins*z_packs);
        int w_in = (i%(w_ins*h_ins*z_packs)) / (h_ins*z_packs);
        int h_in = (i%(h_ins*z_packs)) / z_packs;
        int z_pack = i % z_packs;

        int pos_seed_shared [COMPUTE_CINS];
        int neg_seed_shared [COMPUTE_CINS];
        int weight_pos_shared [COMPUTE_CINS];
        int weight_neg_shared [COMPUTE_CINS];

        // Load seeds and weights
        for(int compute_cin=0; compute_cin<COMPUTE_CINS; compute_cin++) {
            int c_in = z_pack*COMPUTE_CINS + compute_cin;
            int seed_ind = c_in*w_ins*h_ins + w_in*h_ins + h_in;
            int weight_ind;
            if (channels_last_weight) weight_ind = weight_ind_ohwi_host(c_out, c_in, w_in, h_in, c_outs, c_ins, w_ins, h_ins);
            else weight_ind = weight_ind_oihw_host(c_out, c_in, w_in, h_in, c_outs, c_ins, w_ins, h_ins);
            // int weight_ind = c_out*c_ins*w_ins*h_ins + c_in*w_ins*h_ins + w_in*h_ins + h_in;
            if (c_in<c_ins) {    
                // printf("Weight ind %d, pos %d, neg %d\n", weight_ind, weight_pos[weight_ind], weight_neg[weight_ind]);
                if (gen_mult) {
                    pos_seed_shared[compute_cin] = seed_mult;
                    neg_seed_shared[compute_cin] = seed_mult;
                }
                else {
                    pos_seed_shared[compute_cin] = (POS_SEED + seed_ind)%((1<<lfsr_length)-1) + 1;
                    neg_seed_shared[compute_cin] = (NEG_SEED + seed_ind)%((1<<lfsr_length)-1) + 1;
                }
                weight_pos_shared[compute_cin] = weight_pos[weight_ind];
                weight_neg_shared[compute_cin] = weight_neg[weight_ind];
            }
            else {
                pos_seed_shared[compute_cin] = int(0);
                neg_seed_shared[compute_cin] = int(0);
                weight_pos_shared[compute_cin] = int(0);
                weight_neg_shared[compute_cin] = int(0);
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
                int weight_pos_actual = (weight_pos_shared[compute_cin] >> (total_width-cur_width)) << (total_width-cur_width);
                int weight_neg_actual = (weight_neg_shared[compute_cin] >> (total_width-cur_width)) << (total_width-cur_width);
                int pos_seed_cur = pos_seed_shared[compute_cin];
                int neg_seed_cur = neg_seed_shared[compute_cin];
                pos_seed_cur = lfsr(pos_seed_cur);
                neg_seed_cur = lfsr(neg_seed_cur);

                weight_pos_stream_c += int(weight_pos_actual>pos_seed_cur) << compute_cin;
                weight_neg_stream_c += int(weight_neg_actual>neg_seed_cur) << compute_cin;
                // if((c_out==28) && (compute_cin==0)) printf("Generating bit %d, pos %d, neg %d, pos value %d, neg value %d pos bit %d, neg_bit %d\n", bit, weight_pos_stream_c, weight_neg_stream_c, weight_pos_actual, weight_neg_actual, int(weight_pos_actual>pos_seed_cur) << compute_cin, int(weight_neg_actual>neg_seed_cur) << compute_cin);
                pos_seed_shared[compute_cin] = pos_seed_cur;
                neg_seed_shared[compute_cin] = neg_seed_cur;
            }
            for(int gen_and=0; gen_and<gen_config; gen_and++) {
                weight_pos_stream[(bit*gen_config+gen_and)*c_outs*w_ins*h_ins*z_packs + i] = weight_pos_stream_c;
                weight_neg_stream[(bit*gen_config+gen_and)*c_outs*w_ins*h_ins*z_packs + i] = weight_neg_stream_c;
            }
        }
    }
}

// Also transposes the vector to NHWC from NCHW. Doing direct convolution with NCHW is a pain
void activation_generation_or_general(
    const int32_t* __restrict__ input_bin,
    int32_t* __restrict__ input_stream,
    int bit_length,
    int lfsr_length,
    int (*lfsr)(int),
    int gen_config,
    bool gen_mult,
    int batches,
    int c_ins,
    int w_ins,
    int h_ins,
    const int total_width,
    const int load_width,
    const int load_wait,
    bool channels_last_activation) {
    int z_packs = (c_ins + COMPUTE_CINS-1) / COMPUTE_CINS;

    #pragma omp parallel for
    for(int i=0; i<batches*w_ins*h_ins*z_packs; i++) {
        int batch = i/(w_ins*h_ins*z_packs);
        int w_in = (i%(w_ins*h_ins*z_packs)) / (h_ins*z_packs);
        int h_in = (i%(h_ins*z_packs)) / z_packs;
        int z_pack = i % z_packs;

        int seed_shared[COMPUTE_CINS];
        int input_shared[COMPUTE_CINS];

        // Load seeds and inputs
        for(int compute_cin=0; compute_cin<COMPUTE_CINS; compute_cin++) {
            int c_in = z_pack*COMPUTE_CINS + compute_cin;
            int seed_ind = c_in*w_ins*h_ins + w_in*h_ins + h_in;
            int input_ind;
            if (channels_last_activation) input_ind = weight_ind_ohwi_host(batch, c_in, w_in, h_in, batches, c_ins, w_ins, h_ins);
            else input_ind = weight_ind_oihw_host(batch, c_in, w_in, h_in, batches, c_ins, w_ins, h_ins);
            if (c_in<c_ins) {
                if (gen_mult) seed_shared[compute_cin] = 1;
                else seed_shared[compute_cin] = (0 + seed_ind)%((1<<lfsr_length)-1) + 1;
                input_shared[compute_cin] = input_bin[input_ind];
            }
            else {
                seed_shared[compute_cin] = int(0);
                input_shared[compute_cin] = int(0);
            }
        }

        // Generation
        int cur_width = 0;
        for(int bit=0; bit<bit_length; bit++) {
            cur_width = (bit/load_wait + 1)*load_width - 1;
            if (cur_width > total_width) cur_width = total_width;
            int input_stream_c = 0;
            for(int compute_cin=0; compute_cin<COMPUTE_CINS; compute_cin++) {
                int input_actual = (input_shared[compute_cin] >> (total_width-cur_width)) << (total_width-cur_width);
                int seed_cur = seed_shared[compute_cin];
                seed_cur = (*lfsr)(seed_cur);
                input_stream_c += int(input_actual > seed_cur) <<compute_cin;
                seed_shared[compute_cin] = seed_cur;
            }
            for(int gen_and=0; gen_and<gen_config; gen_and++) {
                input_stream[((bit/gen_config)*gen_config*gen_config + gen_and*gen_config + bit%gen_config)*batches*w_ins*h_ins*z_packs + i] = input_stream_c;
            }
        }
    }
}

const int C_GENERAL=32;
const int K_GENERAL=8;
void stream_compute_or_general(
    const int32_t* input_stream,
    const int32_t* weight_pos_stream,
    const int32_t* weight_neg_stream,
    int16_t* output_pos_stream,
    int16_t* output_neg_stream,
    int stride_w,
    int stride_h,
    int bin_config,
    struct Compute_Param c_param,
    bool channels_last_activation) {

    bool or_no = false;
    if (bin_config>=0) bin_config+=1;
    else {
        bin_config = -bin_config+1;
        or_no = true;
    }
    int (*or_act)(int, int);
    if (or_no) or_act = &or_act_n_host;
    else or_act = &or_act_n_host;

    int z_packs = (c_param.c_ins + COMPUTE_CINS-1) / COMPUTE_CINS;

    int cout_step = (c_param.c_outs + C_GENERAL-1) / C_GENERAL;
    int i_flatten_step = ((c_param.i_w_ins-c_param.w_w_ins)/stride_w+1)*((c_param.i_h_ins-c_param.w_h_ins)/stride_h+1);

    int inner_size = c_param.w_w_ins*c_param.w_h_ins*z_packs;
    int inner_packs = (inner_size+K_GENERAL-1)/K_GENERAL;

    int (*output_ind)(int, int, int);
    if (channels_last_activation) output_ind=&output_ind_nhwc_host;
    else output_ind=&output_ind_nchw_host;
    int stride_out = i_flatten_step;
    if (channels_last_activation) stride_out = c_param.c_outs;

    #pragma omp parallel for 
    for (int block=0; block<c_param.batches*cout_step; block++) {
        int batch = block/cout_step;
        int cout_offset = (block%cout_step)*C_GENERAL;
        const int32_t* input_stream_batch = input_stream + batch*c_param.i_w_ins*c_param.i_h_ins*z_packs;
        int16_t* output_pos_batch = output_pos_stream + batch*c_param.c_outs*i_flatten_step;
        int16_t* output_neg_batch = output_neg_stream + batch*c_param.c_outs*i_flatten_step;

        for (int cin=0; cin<i_flatten_step; cin++) {
            int w_in_i_c = (cin / ((c_param.i_h_ins - c_param.w_h_ins)/stride_h+1))*stride_w;
            int h_in_i_c = (cin % ((c_param.i_h_ins - c_param.w_h_ins)/stride_h+1))*stride_h;
            for (int cout=0; cout<C_GENERAL; cout++) {
                int cout_c = cout_offset+cout;
                int cin_c = cin;
                bool valid_00 = (cout_c<c_param.c_outs)  & (cin_c<i_flatten_step) & (cout<C_GENERAL);
                int index_00 = output_ind(cout_c, cin_c, stride_out);
                int16_t* output_pos_00 = output_pos_batch+index_00;
                int16_t* output_neg_00 = output_neg_batch+index_00;
                uint output_value_pos_00 = 0;
                uint output_value_neg_00 = 0;
                if (valid_00) {
                    for (int bit=0; bit<c_param.bit_length; bit++) {
                        const int32_t* input_bit = input_stream_batch + bit*c_param.batches*c_param.i_w_ins*c_param.i_h_ins*z_packs;
                        const int32_t* weight_pos_bit = weight_pos_stream + bit*c_param.c_outs*c_param.w_w_ins*c_param.w_h_ins*z_packs;
                        const int32_t* weight_neg_bit = weight_neg_stream + bit*c_param.c_outs*c_param.w_w_ins*c_param.w_h_ins*z_packs;

                        for (int inner_i=0; inner_i<inner_packs; inner_i++) {
                            int inner = inner_i*K_GENERAL;
                            for (int i=0; i<K_GENERAL; i++) {
                                int32_t weight_pos_c=0;
                                int32_t weight_neg_c=0;
                                int32_t input_c=0;
                                if ((inner+i<inner_size) & (cout_offset+cout<c_param.c_outs)) {
                                    int weight_index = (cout_offset+cout) * c_param.w_w_ins * c_param.w_h_ins * z_packs + (inner+i);
                                    weight_pos_c = weight_pos_bit[weight_index];
                                    weight_neg_c = weight_neg_bit[weight_index];
                                }
                                int z_pack_c = (inner+i) % z_packs;
                                int w_in_w_c = (inner+i) / (c_param.w_h_ins*z_packs);
                                int h_in_w_c = ((inner+i) % (c_param.w_h_ins*z_packs)) / z_packs;
                                if (inner+i<inner_size) {
                                    int input_index = (w_in_i_c + w_in_w_c) * c_param.i_h_ins * z_packs
                                                    +(h_in_i_c + h_in_w_c) * z_packs
                                                    +z_pack_c;
                                    input_c = input_bit[input_index];
                                }
                                output_value_pos_00 += __builtin_popcount(input_c & weight_pos_c);
                                output_value_neg_00 += __builtin_popcount(input_c & weight_neg_c);
                            }
                        }
                        output_value_pos_00 = or_act_split_update_host(output_value_pos_00, or_act, bin_config);
                        output_value_neg_00 = or_act_split_update_host(output_value_neg_00, or_act, bin_config);
                    }
                *output_pos_00 = (output_value_pos_00>>16);
                *output_neg_00 = (output_value_neg_00>>16);
                }
            }
        }
    }
}

torch::Tensor conv2d_generic_general(torch::Tensor input, 
        torch::Tensor weight, 
        int bit_length, 
        int lfsr_length, 
        int z_units, 
        at::IntArrayRef padding, 
        at::IntArrayRef stride, 
        at::IntArrayRef prog_load, 
        int bin_config, 
        int gen_config, 
        bool xnor, 
        bool mux) {
    // printf("Inside func\n");
    auto input_pad = F::pad(input, F::PadFuncOptions({padding.data()[0], padding.data()[0], padding.data()[1], padding.data()[1]})/*.mode(torch::kReplicate)*/);
    auto compare_type = torch::kInt32;
    int lfsr_bit_length = bit_length; //Scaling factor for the floating-point weights/activations
    if(lfsr_length>0) lfsr_bit_length = (1<<lfsr_length);
    auto input_split = (input_pad*lfsr_bit_length).clamp(0, lfsr_bit_length-1).ceil().to(compare_type).clone();
    auto weight_pos = (weight*lfsr_bit_length).clamp(0, lfsr_bit_length-1).ceil().to(compare_type);
    auto weight_neg = (-(weight*lfsr_bit_length).clamp(1-lfsr_bit_length, 0)).ceil().to(compare_type);

    auto weight_size = weight_pos.sizes().data();
    auto input_size = input_split.sizes().data();
    int z_packs = (weight_size[1] + COMPUTE_CINS-1) / COMPUTE_CINS;

    auto input_stride = input.strides().data();
    auto weight_stride = weight.strides().data();
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

    int (*lfsr)(int);
    int bit_unit = 32;
    if (bit_length<32) bit_unit=bit_length;
    switch(lfsr_length) {
        case 3:
        switch(gen_config) {
            case 1:
            lfsr=&lfsr_3_s;
            break;
            case 2:
            lfsr=&lfsr_3_s_acc;
            break;
        }
        break;
        case 4:
        switch(gen_config) {
            case 1:
            lfsr=&lfsr_4_s;
            break;
            case 2:
            lfsr=&lfsr_4_s_acc;
            break;
        }
        break;
        case 5:
        switch(gen_config) {
            case 1:
            lfsr=&lfsr_5_s;
            break;
            case 2:
            lfsr=&lfsr_5_s_acc;
            break;
        }
        break;
        case 6:
        switch(gen_config) {
            case 1:
            lfsr=&lfsr_6_s;
            break;
            case 2:
            lfsr=&lfsr_6_s_acc;
            break;
        }
        break;
        case 7:
        switch(gen_config) {
            case 1:
            lfsr=&lfsr_7_s;
            break;
            case 2:
            lfsr=&lfsr_7_s_acc;
            break;
        }
        break;
        case 8:
        switch(gen_config) {
            case 1:
            lfsr=&lfsr_8_s;
            break;
            case 2:
            lfsr=&lfsr_8_s_acc;
            break;
        }
    }

    int32_t *weight_pos_stream = (int32_t*) malloc(compute_length*weight_size[0]*weight_size[2]*weight_size[3]*z_packs*sizeof(int32_t));
    int32_t *weight_neg_stream = (int32_t*) malloc(compute_length*weight_size[0]*weight_size[2]*weight_size[3]*z_packs*sizeof(int32_t));
    int32_t *input_stream = (int32_t*) malloc(compute_length*input_size[0]*input_size[2]*input_size[3]*z_packs*sizeof(int32_t));

    bool gen_mult = 0;
    stream_generation_or_general (
        weight_pos.data_ptr<int32_t>(),
        weight_neg.data_ptr<int32_t>(),
        weight_pos_stream,
        weight_neg_stream,
        bit_length,
        lfsr_length,
        lfsr,
        gen_config,
        gen_mult,
        weight_size[0],
        weight_size[1],
        weight_size[2],
        weight_size[3],
        prog_load[0],
        prog_load[1],
        prog_load[2],
        channels_last_weight);
    activation_generation_or_general (
        input_split.data_ptr<int32_t>(),
        input_stream,
        bit_length,
        lfsr_length,
        lfsr,
        gen_config,
        gen_mult,
        input_size[0],
        input_size[1],
        input_size[2],
        input_size[3],
        prog_load[0],
        prog_load[1],
        prog_load[2],
        channels_last_activation);

    torch::MemoryFormat output_format;
    if (channels_last_activation) output_format=torch::MemoryFormat::ChannelsLast;
    else output_format=torch::MemoryFormat::Contiguous;
    auto output_tensor_pos = torch::zeros({input_size[0], weight_size[0], (input_size[2]-weight_size[2])/stride[0]+1, (input_size[3]-weight_size[3])/stride[1]+1}, at::TensorOptions().dtype(torch::kInt16)).to(output_format);
    auto output_tensor_neg = torch::zeros({input_size[0], weight_size[0], (input_size[2]-weight_size[2])/stride[0]+1, (input_size[3]-weight_size[3])/stride[1]+1}, at::TensorOptions().dtype(torch::kInt16)).to(output_format);
    struct Compute_Param c_param = {int(compute_length), int(0), int(input_size[0]), int(input_size[1]), int(input_size[2]), int(input_size[3]), int(weight_size[0]), int(weight_size[2]), int(weight_size[3])};    

    stream_compute_or_general (
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
    delete [] weight_pos_stream;
    delete [] weight_neg_stream;
    delete [] input_stream;
    return output_tensor_pos-output_tensor_neg;
}