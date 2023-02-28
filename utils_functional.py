import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# from launch_param import *

import config

import os
import time

'''
Accelerated kernels
'''
import sc_extension
import sc_extension_cuda
import fixed_extension
import fixed_extension_cuda

'''
File containing SC-specific function implementations
'''
# Using torch.float16 for compute_type and compare_type improves performance when using legacy computation on 
# GPUs supporting half precision, but may cause issue with CPU implementation and older GPUs
compute_type = torch.float16
compare_type = torch.float16
# Use true or for training instead of approximation using activation function. Has high performance penalties
# Default precision is 8-bit integer, but one bit is used for sign, so 7 bits are left
prec_default = 7

Seed_4 = 8
Seed_5 = 20
Seed_6 = 36
Seed_7 = 68

Print_Ctr = 0

Scale_Offset = 0.1
Grad_Scale = 10

# Always use maximal-length LFSR
LFSR_Length = -1
XNOR = True #Use XNOR for LFSR
MUX = False #Use comparator for stream generation

'''
Helper functions
'''

def setattr_default(object:nn.Module, kwargs:dict, att_string:str, default_value:any):
    try:
        setattr(object, att_string, kwargs[att_string])
    except:
        setattr(object, att_string, default_value)
    else:
        del kwargs[att_string]

def setdict_default(dict:dict, kwargs:dict, att_string:str, default_value:any):
    try:
        dict[att_string] = kwargs[att_string]
    except:
        dict[att_string] = default_value
    else:
        del kwargs[att_string]

def setvalue_default(kwargs:dict, att_string:str, default_value:any):
    try:
        value = kwargs[att_string]
    except:
        value = default_value
    else:
        del kwargs[att_string]
    return value

def update_prune(model, prune):
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d) and hasattr(mod, "prune"):
            mod.prune = torch.ones_like(mod.prune)*prune

def update_com_gen(model, compute, generator):
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            mod.compute=compute
            mod.generator=generator
def update_sync(model, sync=1):
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            if hasattr(mod, 'run_args'):
                mod.run_args['sync'] = sync

def quantize(input, prec=8, prune=0, monitor=False):
    '''
    Quantize values between 0 and 1
    '''
    prec_2 = 2**prec
    dtype = input.dtype
    if monitor:
        print("Before quantize", input.min(), input.max())
    if prec<=16:
        input = (input * prec_2).round().to(dtype=dtype).clamp(-prec_2+1, prec_2-1)/prec_2
    else:
        input = input.clamp(-1,1)
    if prune>0:
        input_flat = torch.abs(input.view(-1))
        input_size = input_flat.size(0)
        input_sorted, _ = torch.sort(input_flat)
        thres = input_sorted[int(prune*input_size)-1]

        input_flag = torch.ones_like(input)
        input_flag[input.data.ge(-thres) * input.data.le(thres)] = 0
        return input, input_flag
    else:
        return input

class QuantizeGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, prec):
        return quantize(x, prec=prec)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def quantize_shift(tensor, scale=None, scale_only=False):
    '''
    Quantize values with a shift to adjust range
    '''
    if scale_only:
        scale = torch.mean(tensor)*2
        scale = 2**torch.ceil(torch.log2(scale))
        tensor_quant = torch.ones_like(tensor)*scale
        return tensor_quant, scale
    else:
        if scale is None:
            scale = torch.mean(tensor)*3
            scale = 2**torch.ceil(torch.log2(scale))
        tensor_quant = tensor / scale
        tensor_quant = (tensor_quant * 128).round().clamp(-127, 127)/128
        tensor_quant = tensor_quant * scale
        return tensor_quant, scale  

'''
Generator functions
'''

WEIGHT_POS_SEED = 67
WEIGHT_NEG_SEED = 37
INPUT_SEED = 0
def lfsr_init(w_size, a_size=None, device=torch.device('cpu'), prec=128):
    '''
    Initialize generator for LFSR
    '''
    weight_split_size_flat = np.array(w_size).prod()
    weight_seed_pos = np.arange(WEIGHT_POS_SEED, weight_split_size_flat+WEIGHT_POS_SEED)%(prec-1)+1
    weight_seed_neg = np.arange(WEIGHT_NEG_SEED, weight_split_size_flat+WEIGHT_NEG_SEED)%(prec-1)+1
    rand_weight_pos = torch.from_numpy(weight_seed_pos).reshape(w_size).to(device)
    rand_weight_neg = torch.from_numpy(weight_seed_neg).reshape(w_size).to(device)
    if a_size is not None:
        input_split_size_flat = np.array(a_size).prod()
        input_seed = np.arange(INPUT_SEED, input_split_size_flat+INPUT_SEED)%(prec-1)+1
        rand_input = torch.from_numpy(input_seed).reshape(a_size).to(device)
        return rand_input, rand_weight_pos, rand_weight_neg
    else:
        return None, rand_weight_pos, rand_weight_neg

def rand_init(w_size, a_size=None, device=torch.device('cpu'), prec=128):
    '''
    Initialize generator for simulated true random generation
    '''
    rand_weight_pos = torch.randint(prec, w_size, dtype=compare_type, device=device)
    rand_weight_neg = torch.randint(prec, w_size, dtype=compare_type, device=device)
    if a_size is not None:
        rand_input = torch.randint(prec, a_size, dtype=compare_type, device=device)
        return rand_input, rand_weight_pos, rand_weight_neg
    else:
        return None, rand_weight_pos, rand_weight_neg
    
def lfsr_3(value):
    return ((value//4)+(value//2)%2)%2+2*(value%4)

def lfsr_4(rand_in):
    '''
    4-bit LFSR
    '''
    rand_out = ((rand_in//8)+(rand_in//4)%2)%2+2*(rand_in%8)
    return rand_out

def lfsr_4_xnor(rand_in):
    '''
    4-bit LFSR using xnor instead of xor
    '''
    rand_out = 1-((rand_in//8)+(rand_in//4)%2)%2+2*(rand_in%8)
    return rand_out
    
def lfsr_5(rand_in):
    '''
    5-bit LFSR
    '''
    rand_out = ((rand_in//16)+(rand_in//4)%2)%2+2*(rand_in%16)
    return rand_out

def lfsr_5_2(rand_in):
    '''
    5-bit LFSR, version 2
    '''
    rand_out = ((rand_in//16)+(rand_in//2)%2)%2+2*(rand_in%16)
    return rand_out

def lfsr_5_xnor(rand_in):
    '''
    5-bit LFSR using xnor instead of xor
    '''
    rand_out = ((rand_in//16)+(rand_in//4)%2)%2+2*(rand_in%16)
    return rand_out

def lfsr_6(rand_in):
    '''
    6-bit LFSR
    '''
    rand_out = ((rand_in//32)+(rand_in//16)%2)%2+2*(rand_in%32)
    return rand_out

def lfsr_6_xnor(rand_in):
    '''
    6-bit LFSR using xnor instead of xor
    '''
    rand_out = 1-((rand_in//32)+(rand_in//16)%2)%2+2*(rand_in%32)
    return rand_out
    
def lfsr_7(rand_in):
    '''
    7-bit LFSR
    '''
    rand_out = ((rand_in//32)%2+rand_in//64)%2+2*(rand_in%64)
    return rand_out

def lfsr_7_xnor(rand_in):
    '''
    7-bit LFSR using xnor instead of xor
    '''
    rand_out = 1-((rand_in//32)%2+rand_in//64)%2+2*(rand_in%64)
    return rand_out

def lfsr_8(rand_in):
    '''
    8-bit LFSR
    '''
    rand_out = ((rand_in//128)+(rand_in//32)%2+(rand_in//16)%2+(rand_in//8)%2)%2+2*(rand_in%128)
    return rand_out
    
def lfsr_cont(rand_input, rand_weight_pos, rand_weight_neg, bit_length=128):
    '''
    Continue generation using LFSR
    '''
    if bit_length==128:
        lfsr_gen = lfsr_7
    elif bit_length==32:
        lfsr_gen = lfsr_5
    elif bit_length==64:
        lfsr_gen = lfsr_6
    elif bit_length==256:
        lfsr_gen = lfsr_8
    elif bit_length==16:
        lfsr_gen = lfsr_4
    elif bit_length==8:
        lfsr_gen = lfsr_3
    if rand_input is not None:
        rand_input = lfsr_gen(rand_input).to(compare_type)
    rand_weight_pos = lfsr_gen(rand_weight_pos).to(compare_type)
    rand_weight_neg = lfsr_gen(rand_weight_neg).to(compare_type)
    return rand_input, rand_weight_pos, rand_weight_neg
'''
Forward functions for training
'''

'''
Wrapping multiple point-wise operations in JIT improves performance
Doesn't work for arbitrary or_n value?
'''
def or_approx_2_1(sum:torch.Tensor, n:float):
    return (sum/n)**n + ((sum/n)**(n-1))*(1-sum/n)*n
def or_approx_2_4(sum:torch.Tensor, n:float):
    return 1-torch.exp(-sum) + (sum/n)**n

@torch.jit.script
def or_approx_1_forward(sum:torch.Tensor):
    return 1-torch.exp(-sum)

@torch.jit.script
def or_approx_1_backward(grad_output:torch.Tensor, sum:torch.Tensor):
    return torch.exp(-sum)*grad_output

@torch.jit.script
def or_approx_2_forward(sum:torch.Tensor):
    sum_exp = torch.exp(-sum)
    return 2-2*sum_exp-sum*sum_exp

@torch.jit.script
def or_approx_2_backward(grad_output:torch.Tensor, sum:torch.Tensor):
    sum_exp = torch.exp(-sum)
    return (sum_exp + sum*sum_exp)*grad_output

@torch.jit.script
def or_approx_1_forward_bias_std(sum:torch.Tensor, bias_coef:torch.Tensor, std_coef:torch.Tensor):
    sum_exp = torch.exp(-sum)
    x = 1-sum_exp
    x_bias = bias_coef[0]+bias_coef[1]*x+bias_coef[2]*x**2+bias_coef[3]*x**3+bias_coef[4]*x**4+bias_coef[5]*x**5
    x_rand = std_coef[0]+std_coef[1]*x+std_coef[2]*x**2+std_coef[3]*x**3+std_coef[4]*x**4+std_coef[5]*x**5
    x_with_rand = x + x_rand*torch.randn_like(x)
    return x_with_rand+x_bias

@torch.jit.script
def or_approx_2_forward_bias_std(sum:torch.Tensor, bias_coef:torch.Tensor, std_coef:torch.Tensor):
    sum_exp = torch.exp(-sum)
    x = 2-2*sum_exp-sum*sum_exp
    x_bias = bias_coef[0]+bias_coef[1]*x+bias_coef[2]*x**2+bias_coef[3]*x**3+bias_coef[4]*x**4+bias_coef[5]*x**5
    x_rand = std_coef[0]+std_coef[1]*x+std_coef[2]*x**2+std_coef[3]*x**3+std_coef[4]*x**4+std_coef[5]*x**5
    x_with_rand = x + x_rand*torch.randn_like(x)
    return x_with_rand+x_bias

@torch.jit.script
def or_approx_3_forward_bias_std(sum:torch.Tensor, bias_coef:torch.Tensor, std_coef:torch.Tensor):
    sum_exp = torch.exp(-sum)
    x = 3-3*sum_exp-2*sum*sum_exp-0.5*(sum**2)*sum_exp
    x_bias = bias_coef[0]+bias_coef[1]*x+bias_coef[2]*x**2+bias_coef[3]*x**3+bias_coef[4]*x**4+bias_coef[5]*x**5
    x_rand = std_coef[0]+std_coef[1]*x+std_coef[2]*x**2+std_coef[3]*x**3+std_coef[4]*x**4+std_coef[5]*x**5
    x_with_rand = x + x_rand*torch.randn_like(x)
    return x_with_rand+x_bias

@torch.jit.script
def or_approx_4_forward_bias_std(sum:torch.Tensor, bias_coef:torch.Tensor, std_coef:torch.Tensor):
    sum_exp = torch.exp(-sum)
    x = 4-4*sum_exp-3*sum*sum_exp-(sum**2)*sum_exp-(1/6)*(sum**3)*sum_exp
    x_bias = bias_coef[0]+bias_coef[1]*x+bias_coef[2]*x**2+bias_coef[3]*x**3+bias_coef[4]*x**4+bias_coef[5]*x**5
    x_rand = std_coef[0]+std_coef[1]*x+std_coef[2]*x**2+std_coef[3]*x**3+std_coef[4]*x**4+std_coef[5]*x**5
    x_with_rand = x + x_rand*torch.randn_like(x)
    # x_with_rand = x + x_rand
    return x_with_rand+x_bias

@torch.jit.script
def or_approx_1_forward_bias_std_both(sum_pos:torch.Tensor, sum_neg:torch.Tensor, bias_coef:torch.Tensor, std_coef:torch.Tensor):
    x_pos = or_approx_1_forward_bias_std(sum_pos, bias_coef, std_coef)
    x_neg = or_approx_1_forward_bias_std(sum_neg, bias_coef, std_coef)
    return x_pos - x_neg

@torch.jit.script
def or_approx_2_forward_bias_std_both(sum_pos:torch.Tensor, sum_neg:torch.Tensor, bias_coef:torch.Tensor, std_coef:torch.Tensor):
    x_pos = or_approx_2_forward_bias_std(sum_pos, bias_coef, std_coef)
    x_neg = or_approx_2_forward_bias_std(sum_neg, bias_coef, std_coef)
    return x_pos - x_neg

@torch.jit.script
def or_approx_3_forward_bias_std_both(sum_pos:torch.Tensor, sum_neg:torch.Tensor, bias_coef:torch.Tensor, std_coef:torch.Tensor):
    x_pos = or_approx_3_forward_bias_std(sum_pos, bias_coef, std_coef)
    x_neg = or_approx_3_forward_bias_std(sum_neg, bias_coef, std_coef)
    return x_pos - x_neg

@torch.jit.script
def or_approx_4_forward_bias_std_both(sum_pos:torch.Tensor, sum_neg:torch.Tensor, bias_coef:torch.Tensor, std_coef:torch.Tensor):
    x_pos = or_approx_4_forward_bias_std(sum_pos, bias_coef, std_coef)
    x_neg = or_approx_4_forward_bias_std(sum_neg, bias_coef, std_coef)
    return x_pos - x_neg

@torch.jit.script
def or_approx_1_backward_bias(grad_output:torch.Tensor, sum:torch.Tensor, bias_coef:torch.Tensor):
    sum_exp = torch.exp(-sum)
    x = 1-sum_exp
    x_dot = 1+bias_coef[1] + bias_coef[2]*2*x + bias_coef[3]*3*x**2 + bias_coef[4]*4*x**3 + bias_coef[5]*5*x**4
    grad_output = x_dot*grad_output
    return sum_exp*grad_output

@torch.jit.script
def or_approx_2_backward_bias(grad_output:torch.Tensor, sum:torch.Tensor, bias_coef:torch.Tensor):
    sum_exp = torch.exp(-sum)
    x = 2-2*sum_exp-sum*sum_exp
    x_dot = 1+bias_coef[1] + bias_coef[2]*2*x + bias_coef[3]*3*x**2 + bias_coef[4]*4*x**3 + bias_coef[5]*5*x**4
    grad_output = x_dot*grad_output
    return (1+sum)*sum_exp*grad_output

@torch.jit.script
def or_approx_3_backward_bias(grad_output:torch.Tensor, sum:torch.Tensor, bias_coef:torch.Tensor):
    sum_exp = torch.exp(-sum)
    x = 3-3*sum_exp-2*sum*sum_exp-0.5*(sum**2)*sum_exp
    x_dot = 1+bias_coef[1] + bias_coef[2]*2*x + bias_coef[3]*3*x**2 + bias_coef[4]*4*x**3 + bias_coef[5]*5*x**4
    grad_output = x_dot*grad_output
    return (1+sum+0.5*(sum**2))*sum_exp*grad_output

@torch.jit.script
def or_approx_4_backward_bias(grad_output:torch.Tensor, sum:torch.Tensor, bias_coef:torch.Tensor):
    sum_exp = torch.exp(-sum)
    x = 4-4*sum_exp-3*sum*sum_exp-(sum**2)*sum_exp-(1/6)*(sum**3)*sum_exp
    x_dot = 1+bias_coef[1] + bias_coef[2]*2*x + bias_coef[3]*3*x**2 + bias_coef[4]*4*x**3 + bias_coef[5]*5*x**4
    grad_output = x_dot*grad_output
    return (1+sum+0.5*(sum**2)+(1/6)*(sum**3))*sum_exp*grad_output

@torch.jit.script
def or_approx_1_backward_bias_both(grad_output:torch.Tensor, sum_pos:torch.Tensor, sum_neg:torch.Tensor, bias_coef:torch.Tensor):
    grad_pos = or_approx_1_backward_bias(grad_output, sum_pos, bias_coef)
    grad_neg = or_approx_1_backward_bias(-grad_output, sum_neg, bias_coef)
    return grad_pos, grad_neg

@torch.jit.script
def or_approx_2_backward_bias_both(grad_output:torch.Tensor, sum_pos:torch.Tensor, sum_neg:torch.Tensor, bias_coef:torch.Tensor):
    grad_pos = or_approx_2_backward_bias(grad_output, sum_pos, bias_coef)
    grad_neg = or_approx_2_backward_bias(-grad_output, sum_neg, bias_coef)
    return grad_pos, grad_neg

@torch.jit.script
def or_approx_3_backward_bias_both(grad_output:torch.Tensor, sum_pos:torch.Tensor, sum_neg:torch.Tensor, bias_coef:torch.Tensor):
    grad_pos = or_approx_3_backward_bias(grad_output, sum_pos, bias_coef)
    grad_neg = or_approx_3_backward_bias(-grad_output, sum_neg, bias_coef)
    return grad_pos, grad_neg

@torch.jit.script
def or_approx_4_backward_bias_both(grad_output:torch.Tensor, sum_pos:torch.Tensor, sum_neg:torch.Tensor, bias_coef:torch.Tensor):
    grad_pos = or_approx_4_backward_bias(grad_output, sum_pos, bias_coef)
    grad_neg = or_approx_4_backward_bias(-grad_output, sum_neg, bias_coef)
    return grad_pos, grad_neg

@torch.jit.script
def or_approx_3_forward(sum:torch.Tensor):
    sum_exp = torch.exp(-sum)
    return 3-3*sum_exp-2*sum*sum_exp-0.5*(sum**2)*sum_exp

@torch.jit.script
def or_approx_3_backward(grad_output:torch.Tensor, sum:torch.Tensor):
    sum_exp = torch.exp(-sum)
    return (1+sum+0.5*(sum**2))*sum_exp*grad_output

@torch.jit.script
def or_approx_4_forward(sum:torch.Tensor):
    sum_exp = torch.exp(-sum)
    return 4-4*sum_exp-3*sum*sum_exp-(sum**2)*sum_exp-(1/6)*(sum**3)*sum_exp

@torch.jit.script
def or_approx_4_backward(grad_output:torch.Tensor, sum:torch.Tensor):
    sum_exp = torch.exp(-sum)
    return (1+sum+0.5*(sum**2)+(1/6)*(sum**3))*sum_exp*grad_output

@torch.jit.script
def bias_correct_forward(x:torch.Tensor, bias_coef:torch.Tensor, std_coef:torch.Tensor):
    x_bias = bias_coef[0]+bias_coef[1]*x+bias_coef[2]*x**2+bias_coef[3]*x**3+bias_coef[4]*x**4+bias_coef[5]*x**5
    x_rand = std_coef[0]+std_coef[1]*x+std_coef[2]*x**2+std_coef[3]*x**3+std_coef[4]*x**4+std_coef[5]*x**5
    x_with_rand = x + x_rand*torch.randn_like(x)
    # x_with_rand = x + x_rand
    return x_with_rand, x_with_rand+x_bias

@torch.jit.script
def rand_only_forward(x:torch.Tensor, std_coef:torch.Tensor):
    x_rand = std_coef[0]+std_coef[1]*x+std_coef[2]*x**2+std_coef[3]*x**3+std_coef[4]*x**4+std_coef[5]*x**5
    x_with_rand = x + x_rand*torch.randn_like(x)
    # x_with_rand = x + x_rand
    return x_with_rand

@torch.jit.script
def bias_correct_backward(grad_output:torch.Tensor, x:torch.Tensor, bias_coef:torch.Tensor):
    x_dot = 1+bias_coef[1] + bias_coef[2]*2*x + bias_coef[3]*3*x**2 + bias_coef[4]*4*x**3 + bias_coef[5]*5*x**4
    return x_dot*grad_output


'''
Defining custom fwd/bwd like this reduces memory consumption
Fewer tensors are saved for backward
Ideally, only sum needs to be stored along the previous conv2d/linear
But conv2d_backward is not exposed, so...
'''
class Or_Approx_n_Grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sum:torch.Tensor, n:int):
        ctx.n = n
        ctx.save_for_backward(sum)
        if n<=4:
            sum = sc_extension_cuda.or_approx_n_forward_acc(sum, n)
            return sum
        else:
            # Higher order or_n cannot be jit compiled, since n is unknown?
            sum_exp = torch.exp(-sum)
            output = n
            for i in range(n):
                output = output - (n-i)*(sum**i)*(1/np.math.factorial(i))*sum_exp
            return output
    @staticmethod
    def backward(ctx, grad_output):
        n = ctx.n
        sum, = ctx.saved_tensors
        if n<=4:
            grad_output = sc_extension_cuda.or_approx_n_backward_acc(grad_output, sum, n)
            return grad_output, None
        else:
            sum_exp = torch.exp(-sum)
            grad_input = 0
            for i in range(n):
                grad_input = grad_input + (1/np.math.factorial(i))*(sum**i)*sum_exp
            return grad_input*grad_output, None

class Or_Approx_N_Bias_Correct_Grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sum:torch.Tensor, n:int, bias_coef:torch.Tensor, std_coef:torch.Tensor, order:int):
        ctx.n = n
        ctx.order = order
        ctx.save_for_backward(sum, bias_coef)
        if n==1:
            x = or_approx_1_forward(sum)
        elif n==2:
            x = or_approx_2_forward(sum)
        elif n==3:
            x = or_approx_3_forward(sum)
        elif n==4:
            x = or_approx_4_forward(sum)
        else:
            # Higher order or_n cannot be jit compiled, since n is unknown?
            sum_exp = torch.exp(-sum)
            output = n
            for i in range(n):
                output = output - (n-i)*(sum**i)*(1/np.math.factorial(i))*sum_exp
            x = output
        _, x_output = bias_correct_forward(x, bias_coef, std_coef)
        return x_output
    @staticmethod
    def backward(ctx, grad_output:torch.Tensor):
        # Replay or_n activation and rand add
        sum, bias_coef = ctx.saved_tensors
        n = ctx.n
        if n==1:
            x = or_approx_1_forward(sum)
        elif n==2:
            x = or_approx_2_forward(sum)
        elif n==3:
            x = or_approx_3_forward(sum)
        elif n==4:
            x = or_approx_4_forward(sum)
        else:
            # Higher order or_n cannot be jit compiled, since n is unknown?
            sum_exp = torch.exp(-sum)
            output = n
            for i in range(n):
                output = output - (n-i)*(sum**i)*(1/np.math.factorial(i))*sum_exp
            x = output
        grad_output = bias_correct_backward(grad_output, x, bias_coef)
        # OR_N BWD
        if n==1:
            return or_approx_1_backward(grad_output, sum), None, None, None, None
        elif n==2:
            return or_approx_2_backward(grad_output, sum), None, None, None, None
        elif n==3:
            return or_approx_3_backward(grad_output, sum), None, None, None, None
        elif n==4:
            return or_approx_4_backward(grad_output, sum), None, None, None, None
        else:
            sum_exp = torch.exp(-sum)
            grad_input = 0
            for i in range(n):
                grad_input = grad_input + (1/np.math.factorial(i))*(sum**i)*sum_exp
            # for i in range(n):
            #     grad_input = grad_input + (n-i)*(1/np.math.factorial(i))*i*(sum**(i-1))*sum_exp
            return grad_input*grad_output, None, None, None, None

class Or_Approx_N_Bias_Correct_Grad_Both(torch.autograd.Function):
    '''
    Fused version: positive OR_N + A+E + negative OR_N + A+E + subtraction
    '''
    @staticmethod
    def forward(ctx, sum_pos:torch.Tensor, sum_neg :torch.Tensor, n:int, bias_coef:torch.Tensor, std_coef:torch.Tensor, order:int):
        ctx.n = n
        ctx.order = order
        ctx.save_for_backward(sum_pos, sum_neg, bias_coef)
        sum = sc_extension_cuda.or_approx_n_forward_bias_std_both_acc(sum_pos, sum_neg, bias_coef, std_coef, n)
        return sum
    @staticmethod
    def backward(ctx, grad_output:torch.Tensor):
        sum_pos, sum_neg, bias_coef = ctx.saved_tensors
        n = ctx.n
        grad_input_pos, grad_input_neg = sc_extension_cuda.or_approx_n_backward_bias_std_both_acc(grad_output, sum_pos, sum_neg, bias_coef, n)
        return grad_input_pos, grad_input_neg, None, None, None, None

class Bias_Correct_Grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:torch.Tensor, bias_coef:torch.Tensor, std_coef:torch.Tensor, order:int):
        ctx.order = order
        _, x_output = bias_correct_forward(x, bias_coef, std_coef)
        ctx.save_for_backward(x, bias_coef)
        return x_output
    @staticmethod
    def backward(ctx, grad_output:torch.Tensor):
        x, bias_coef = ctx.saved_tensors
        return bias_correct_backward(grad_output, x, bias_coef), None, None, None

# @torch.jit.script
def or_approx_no(sum, n):
    '''
    or_n with only the nth bit
    '''
    sum_exp = torch.exp(-sum)
    output = 1
    for i in range(n):
        output = output - (sum**i)*(1/np.math.factorial(i))*sum_exp
    return output

def linear_or_approx(activation, weight, true_or=False):
    '''
    Floating-point forward function to guide back propagation for linear layers
    '''
    if true_or:
        mult_result = activation.unsqueeze(1)*weight
        return 1-torch.prod(1-mult_result, dim=-1)
    else:
        return 1-torch.exp(-F.linear(activation, weight))
def conv2d_or_approx(activation, weight, stride=1, padding=0, true_or=False, groups=1, dump=-1, scale=1, n=1, input_scale=1):
    '''
    Floating-point forward function to guide back propagation for conv2d layers using full or accumulation
    '''
    if true_or:
        # True or is achieved by first performing an im2col transformation
        kernel_size = weight.size()[-2:]
        activation_col = F.unfold(activation, kernel_size, dilation=1, padding=padding, stride=stride)
        weight_col = weight.view(weight.data.size(0),-1)
        a_size = list(activation_col.size())
        a_size.insert(1,1)
        w_size = list(weight_col.size())
        w_size.append(1)
        activation_col = activation_col.view(a_size)
        weight_col = weight_col.view(w_size)
    
        mult_result = activation_col*weight_col
        add_res = 1-torch.prod(1-mult_result, dim=2)
        size_out = np.sqrt(add_res.size(-1)).astype(int)
        return F.fold(add_res, (size_out, size_out), (1,1))
    else:
        conv_normal = F.conv2d(activation, weight, padding=padding, stride=stride, groups=groups)
        if input_scale != 1:
            conv_normal = conv_normal*input_scale
        conv_exp = Or_Approx_n_Grad.apply(conv_normal, n)
        if dump==0:
            if not os.path.exists("conv_normal_pos.npy"):
                np.save("conv_normal_pos.npy", conv_normal.data.cpu().numpy().reshape(-1))
            else:
                conv_normal_old = np.load("conv_normal_pos.npy")
                conv_normal_old = np.concatenate((conv_normal_old, conv_normal.data.cpu().numpy().reshape(-1)))
                np.save("conv_normal_pos.npy", conv_normal_old)
            if not os.path.exists("conv_exp_pos.npy"):
                np.save("conv_exp_pos.npy", conv_exp.data.cpu().numpy().reshape(-1))
            else:
                conv_exp_old = np.load("conv_exp_pos.npy")
                conv_exp_old = np.concatenate((conv_exp_old, conv_exp.data.cpu().numpy().reshape(-1)))
                np.save("conv_exp_pos.npy", conv_exp_old)
        if dump==1:
            if not os.path.exists("conv_normal_neg.npy"):
                np.save("conv_normal_neg.npy", conv_normal.data.cpu().numpy().reshape(-1))
            else:
                conv_normal_old = np.load("conv_normal_neg.npy")
                conv_normal_old = np.concatenate((conv_normal_old, conv_normal.data.cpu().numpy().reshape(-1)))
                np.save("conv_normal_neg.npy", conv_normal_old)
            if not os.path.exists("conv_exp_neg.npy"):
                np.save("conv_exp_neg.npy", conv_exp.data.cpu().numpy().reshape(-1))
            else:
                conv_exp_old = np.load("conv_exp_neg.npy")
                conv_exp_old = np.concatenate((conv_exp_old, conv_exp.data.cpu().numpy().reshape(-1)))
                np.save("conv_exp_neg.npy", conv_exp_old)
        if config.Use_Approx:
            err = torch.randn_like(conv_exp) * torch.sqrt(conv_exp*(1-conv_exp) * (1./config.Stream_Length))
            conv_exp.data += err
        return conv_exp

def conv2d_or_approx_variable(activation, w_pos, w_neg, padding, stride, bin_count=5):
    result_pos_value = bin_count*(conv2d_or_approx(activation, w_pos, padding=padding, stride=stride, scale=bin_count))
    result_neg_value = bin_count*(conv2d_or_approx(activation, w_neg, padding=padding, stride=stride, scale=bin_count))
    return result_pos_value, result_neg_value
        
def conv2d_or_bin_2d(activation, w_pos, w_neg, padding, stride):
    '''
    Floating-point forward function to guide back propagation for conv2d layers using mixed accumulation
    Accumulation is done using fixed-point adders for x and y dimensions, and simulated using multiple convolutions
    '''
    i_x = activation.size(2)
    i_y = activation.size(3)
    f_x = w_pos.size(2)
    f_y = w_pos.size(3)
    result_pos_value = []
    result_neg_value = []
    for i in range(f_x):
        for j in range(f_y):
            a_sec = activation[...,i:i_x-f_x+i+1,j:i_y-f_y+j+1].clone()
            w_pos_sec = w_pos[...,i:i+1,j:j+1].clone()
            w_neg_sec = w_neg[...,i:i+1,j:j+1].clone()
            result_pos_value.append(conv2d_or_approx(a_sec, w_pos_sec, padding=padding, stride=stride))
            result_neg_value.append(conv2d_or_approx(a_sec, w_neg_sec, padding=padding, stride=stride))
    
    result_pos_value = torch.stack(result_pos_value, 0).sum(0)
    result_neg_value = torch.stack(result_neg_value, 0).sum(0)
    return result_pos_value, result_neg_value

def conv2d_or_bin_1d(activation, w_pos, w_neg, padding, stride, true_or=False, n=1):
    '''
    Floating-point forward function to guide back propagation for conv2d layers using mixed accumulation
    Accumulation is done using fixed-point adders for the y dimension, and simulated using multiple convolutions
    '''
    i_y = activation.size(3)
    f_y = w_pos.size(3)
    result_pos_value = []
    result_neg_value = []
    a_comb = []
    w_pos_comb = []
    w_neg_comb = []
    for j in range(f_y):
        a_sec = activation[...,j:i_y-f_y+j+1].clone()
        w_pos_sec = w_pos[...,j:j+1].clone()
        w_neg_sec = w_neg[...,j:j+1].clone()

        result_pos_value.append(conv2d_or_approx(a_sec, w_pos_sec, padding=padding, stride=stride, n=n))
        result_neg_value.append(conv2d_or_approx(a_sec, w_neg_sec, padding=padding, stride=stride, n=n))
    result_pos_value = torch.stack(result_pos_value, 0).sum(0)
    result_neg_value = torch.stack(result_neg_value, 0).sum(0)
    return result_pos_value, result_neg_value

def conv2d_or_bin_1d_2(activation, w_pos, w_neg, padding, stride):
    '''
    Floating-point forward function to guide back prop
    '''
    i_y = activation.size(3)
    f_y = w_pos.size(3)
    s_y = (f_y+1)//2
    s_y_c = f_y - s_y
    result_pos_value = []
    result_neg_value = []

    # First half
    a_sec = activation[...,:i_y-s_y_c].clone()
    w_pos_sec = w_pos[...,:s_y].clone()
    w_neg_sec = w_neg[...,:s_y].clone()
    # print(a_sec.size(), w_pos_sec.size(), w_neg_sec.size())
    result_pos_value.append(s_y*(conv2d_or_approx(a_sec, w_pos_sec, padding=padding, stride=stride))*(1./s_y))
    result_neg_value.append(s_y*(conv2d_or_approx(a_sec, w_neg_sec, padding=padding, stride=stride))*(1./s_y))

    # Second half
    a_sec = activation[...,s_y:].clone()
    w_pos_sec = w_pos[...,s_y:].clone()
    w_neg_sec = w_neg[...,s_y:].clone()
    # print(a_sec.size(), w_pos_sec.size(), w_neg_sec.size())
    result_pos_value.append(s_y_c*(conv2d_or_approx(a_sec, w_pos_sec, padding=padding, stride=stride))*(1./s_y_c))
    result_neg_value.append(s_y_c*(conv2d_or_approx(a_sec, w_neg_sec, padding=padding, stride=stride))*(1./s_y_c))

    result_pos_value = result_pos_value[0] + result_pos_value[1]
    result_neg_value = result_neg_value[0] + result_neg_value[1]
    return result_pos_value, result_neg_value
    

def conv2d_or_bin_z(activation, w_pos, w_neg, padding, stride):
    '''
    Floating-point forward function to guide back propagation for conv2d layers using mixed accumulation
    Accumulation is done using fixed-point adders for the z dimension, and simulated using multiple convolutions
    '''
    c_in = activation.size(1)
    result_pos_value = []
    result_neg_value = []
    for c in range(c_in):
        a_sec = activation[:,c:c+1].clone()
        w_pos_sec = w_pos[:,c:c+1].clone()
        w_neg_sec = w_neg[:,c:c+1].clone()
        result_pos_value.append(conv2d_or_approx(a_sec, w_pos_sec, padding=padding, stride=stride))
        result_neg_value.append(conv2d_or_approx(a_sec, w_neg_sec, padding=padding, stride=stride))
    result_pos_value = torch.stack(result_pos_value, 0).sum(0)
    result_neg_value = torch.stack(result_neg_value, 0).sum(0)
    return result_pos_value, result_neg_value

def conv2d_or_bin_xy(activation, w_pos, w_neg, padding, stride, z_unit):
    '''
    Floating-point forward function to guide back propagation for conv2d layers using mixed accumulation
    Accumulation is done using fixed-point adders for x and y dimensions, and simulated using multiple convolutions
    '''
    if np.random.rand()>0.3:
        bin_count = w_pos.size(-1)*w_pos.size(-2)*((w_pos.size(1)+z_unit-1)//z_unit)
        return conv2d_or_approx_variable(activation, w_pos, w_neg, padding, stride, bin_count=bin_count)
    else:
        i_x = activation.size(2)
        i_y = activation.size(3)
        f_x = w_pos.size(2)
        f_y = w_pos.size(3)
        c_in = activation.size(1)
        result_pos_value = []
        result_neg_value = []
        for c in range(0,c_in,z_unit):
            if c+z_unit<=c_in:
                c_end = c+z_unit
            else:
                c_end = c_in
            for i in range(f_x):
                for j in range(f_y):
                    a_sec = activation[:,c:c_end,i:i_x-f_x+i+1,j:i_y-f_y+j+1].clone()
                    w_pos_sec = w_pos[:,c:c_end,i:i+1,j:j+1].clone()
                    w_neg_sec = w_neg[:,c:c_end,i:i+1,j:j+1].clone()
                    result_pos_value.append(conv2d_or_approx(a_sec, w_pos_sec, padding=padding, stride=stride))
                    result_neg_value.append(conv2d_or_approx(a_sec, w_neg_sec, padding=padding, stride=stride))
        result_pos_value = torch.stack(result_pos_value, 0).sum(0)
        result_neg_value = torch.stack(result_neg_value, 0).sum(0)
        return result_pos_value, result_neg_value

def conv2d_or_bin_z_partial(activation, w_pos, w_neg, padding, stride, z_unit):
    '''
    Floating-point forward function to guide back propagation for conv2d layers using mixed accumulation
    Accumulation is done using fixed-point adders for the z dimension, and simulated using multiple convolutions
    '''
    c_in = activation.size(1)
    result_pos_value = []
    result_neg_value = []
    for c in range(0,c_in,z_unit):
        if c+z_unit<=c_in:
            c_end = c+z_unit
        else:
            c_end = c_in
        a_sec = activation[:,c:c_end].clone()
        w_pos_sec = w_pos[:,c:c_end].clone()
        w_neg_sec = w_neg[:,c:c_end].clone()
        result_pos_value.append(conv2d_or_approx(a_sec, w_pos_sec, padding=padding, stride=stride))
        result_neg_value.append(conv2d_or_approx(a_sec, w_neg_sec, padding=padding, stride=stride))
    # print(torch.stack(result_pos_value, 0).size())
    result_pos_value = torch.stack(result_pos_value, 0).sum(0)
    result_neg_value = torch.stack(result_neg_value, 0).sum(0)
    return result_pos_value, result_neg_value

def conv2d_or_bin_yz(activation, w_pos, w_neg, padding, stride, z_unit):
    '''
    Floating-point forward function to guide back propagation for conv2d layers using mixed accumulation
    Accumulation is done using fixed-point adders for y and z dimension, and simulated using multiple convolutions
    y dimension is accumulated using fixed-point adders only, while z dimension performs some accumulations using OR to reduce cost, specified by z_unit argument
    '''
    c_in = activation.size(1)
    i_y = activation.size(3)
    f_y = w_pos.size(3)
    result_pos_value = []
    result_neg_value = []# SC
    a_comb = []
    w_pos_comb = []
    w_neg_comb = []
    z_packs = (c_in + z_unit-1) // z_unit
    for c in range(0,c_in,z_unit):
        if c+z_unit<=c_in:
            c_end=c+z_unit
        else:
            c_end=c_in
        for j in range(f_y):
            a_sec = activation[:,c:c_end,:,j:i_y-f_y+j+1].clone()
            w_pos_sec = w_pos[:,c:c_end,:,j:j+1].clone()
            w_neg_sec = w_neg[:,c:c_end,:,j:j+1].clone()

            a_comb.append(a_sec)
            w_pos_comb.append(w_pos_sec)
            w_neg_comb.append(w_neg_sec)
    a_comb = torch.cat(a_comb, 1).contiguous()
    w_pos_comb = torch.cat(w_pos_comb, 0).contiguous()
    w_neg_comb = torch.cat(w_neg_comb, 0).contiguous()
    result_pos = conv2d_or_approx(a_comb, w_pos_comb, padding=padding, stride=stride, groups=f_y*z_packs)
    result_neg = conv2d_or_approx(a_comb, w_neg_comb, padding=padding, stride=stride, groups=f_y*z_packs)
    result_pos = result_pos.view(result_pos.size(0),f_y*z_packs,-1, result_pos.size(2), result_pos.size(3)).sum(1)
    result_neg = result_neg.view(result_neg.size(0),f_y*z_packs,-1, result_neg.size(2), result_neg.size(3)).sum(1)
    return result_pos, result_neg

def conv2d_or_bin_yz_2(activation, w_pos, w_neg, padding, stride, z_unit):
    c_in = activation.size(1)
    i_y = activation.size(3)
    f_y = w_pos.size(3)
    s_y = (f_y+1)//2
    s_y_c = f_y - s_y
    z_pack = (c_in + z_unit-1)//z_unit
    s_z_pack = (z_pack+1)//2
    s_z_pack_c = z_pack - s_z_pack
    result_pos_value = 0
    result_neg_value = 0

    #00
    a_sec = activation[:,:s_z_pack*z_unit,:,:i_y-s_y_c].clone()
    w_pos_sec = w_pos[:,:s_z_pack*z_unit,:,:s_y].clone()
    w_neg_sec = w_neg[:,:s_z_pack*z_unit,:,:s_y].clone()
    result_pos_value = result_pos_value + s_z_pack*s_y*(conv2d_or_approx(a_sec, w_pos_sec, stride=stride))*(1./(s_z_pack*s_z_pack))
    result_neg_value = result_neg_value + s_z_pack*s_y*(conv2d_or_approx(a_sec, w_neg_sec, stride=stride))*(1./(s_z_pack*s_z_pack))

    #01
    a_sec = activation[:,:s_z_pack*z_unit,:,s_y:].clone()
    w_pos_sec = w_pos[:,:s_z_pack*z_unit,:,s_y:].clone()
    w_neg_sec = w_neg[:,:s_z_pack*z_unit,:,s_y:].clone()
    result_pos_value = result_pos_value + s_z_pack*s_y_c*(conv2d_or_approx(a_sec, w_pos_sec, stride=stride))*(1./(s_z_pack*s_z_pack))
    result_neg_value = result_neg_value + s_z_pack*s_y_c*(conv2d_or_approx(a_sec, w_neg_sec, stride=stride))*(1./(s_z_pack*s_z_pack))

    if s_z_pack_c>0:
        #10
        a_sec = activation[:,s_z_pack*z_unit:,:,:i_y-s_y_c].clone()
        w_pos_sec = w_pos[:,s_z_pack*z_unit:,:,:s_y].clone()
        w_neg_sec = w_neg[:,s_z_pack*z_unit:,:,:s_y].clone()
        result_pos_value = result_pos_value + s_z_pack_c*s_y*(conv2d_or_approx(a_sec, w_pos_sec, stride=stride))*(1./(s_z_pack*s_z_pack))
        result_neg_value = result_neg_value + s_z_pack_c*s_y*(conv2d_or_approx(a_sec, w_neg_sec, stride=stride))*(1./(s_z_pack*s_z_pack))

        #11
        a_sec = activation[:,s_z_pack*z_unit:,:,s_y:].clone()
        w_pos_sec = w_pos[:,s_z_pack*z_unit:,:,s_y:].clone()
        w_neg_sec = w_neg[:,s_z_pack*z_unit:,:,s_y:].clone()
        result_pos_value = result_pos_value + s_z_pack_c*s_y_c*(conv2d_or_approx(a_sec, w_pos_sec, stride=stride))*(1./(s_z_pack*s_z_pack))
        result_neg_value = result_neg_value + s_z_pack_c*s_y_c*(conv2d_or_approx(a_sec, w_neg_sec, stride=stride))*(1./(s_z_pack*s_z_pack))
    return result_pos_value, result_neg_value

def compute_bias_std_coef(or_n, a_bit, a_act, order):
    if int(os.environ.get("Weight_Prec", "-1"))>0:
        # For analog SC
        or_n = 1
        # order = 0
    if or_n<0:
        or_n = a_bit.max().item()
        bias_coef = torch.zeros(order+1, device=a_bit.device)
        std_coef = torch.zeros(order+1, device=a_bit.device)
        return bias_coef, std_coef
    output_range = np.linspace(0,a_bit.max().item(),30)
    bias_bins = []
    std_bins = []
    index_valid = []

    # print("Bit average", a_bit.mean(), "Act average", a_act.mean())

    for i,start_pos in enumerate(output_range):
        index_range = (a_act>=start_pos) * (a_act<start_pos+0.1)
        a_bit_range = a_bit[index_range]
        a_act_range = a_act[index_range]
        err = a_bit_range - a_act_range
        if torch.numel(err)>1:
            bias_bins.append(err.mean())
            std_bins.append(err.std())
            index_valid.append(i)
    bias_bins = torch.tensor(bias_bins).to(a_bit.device)
    std_bins = torch.tensor(std_bins).to(a_bit.device)
    # print(bias_bins)
    # print(std_bins)
    output_range = output_range[index_valid]
    range_stack = []
    # order = min(order, len(index_valid)//5)
    # Prevent overfitting
    # print(len(index_valid))
    for i in np.arange(0, order+1):
        range_stack.append(output_range**i)
    range_reshape = torch.from_numpy(np.stack(range_stack, axis=1)).to(a_bit.device).to(bias_bins.dtype)

    # bias_coef = torch.zeros(order+1,device=a_bit.device)
    # std_coef = torch.zeros(order+1,device=a_bit.device)

    if order>=len(index_valid)//5:
        bias_coef = torch.zeros(order+1,device=a_bit.device)
        # std_coef = torch.zeros(order+1,device=a_bit.device)
    else:
        bias_coef = torch.linalg.lstsq(range_reshape.float(), bias_bins.float())[0]
    std_coef = torch.linalg.lstsq(range_reshape.float(), std_bins.float())[0]
    # print(bias_coef, std_coef, a_bit.max().item())
    return bias_coef, std_coef

'''
Generic functional layers. All other configurations should be derived from this
'''

def linear_generic(activation, weight, **kwargs):
    '''
    Generic linear layer
    Arguments:
    bit_length: stream length to use
    prec: weight and activation quantization precision specified using number of allowed discrete values
    share: allow limited sharing of stream generators to reduce cost and improve accuracy
    generator: stream generator to use
    forward: sc computation to use
    '''
    bit_length = setvalue_default(kwargs, 'bit_length', 128)
    prec = setvalue_default(kwargs, 'prec', 128)
    share = setvalue_default(kwargs, 'share', True)
    generator = setvalue_default(kwargs, 'generator', None)
    forward = setvalue_default(kwargs, 'forward', 'full_or')
    
    device = activation.device
    bit_range = prec-1
    
    dtype = activation.dtype
    if os.environ.get("Global_Sync", "1")=="1":
        # Quantization precision is tied to stream length for LFSR generator. E.g.: 5-bit precision is used for 
        # 32-bit streams (+1 bit precision for sign)
        if generator=='lfsr' or 'lfsr_split':
            prec = bit_length
        input_split = (activation.data*prec).to(compare_type)

        if generator=='lfsr_split':
            bit_range_l = 8
            weight_m = weight.data // (1/16) * (1/16)
            weight_l = weight.data.sign() * (torch.abs(weight.data)%(1/16)) * 16
            w_pos_split = (weight_m.data*prec).clamp(0,bit_range).to(compare_type)
            w_neg_split = -(weight_m.data*prec).clamp(-bit_range,0).to(compare_type)
            w_l_pos_split = (weight_l.data*bit_range_l).clamp(0,bit_range_l-1).to(compare_type)
            w_l_neg_split = -(weight_l.data*bit_range_l).clamp(-bit_range_l+1,0).to(compare_type)
        else:
            w_pos_split = (weight.data*prec).clamp(0,bit_range).to(compare_type)
            w_neg_split = -(weight.data*prec).clamp(-bit_range,0).to(compare_type)

        # Share stream generator between different filters and different inputs if permitted
        if share:
            a_size = [activation.size(-1)]
            w_size = [weight.size(-1)]
        else:
            a_size = activation.size()
            w_size = weight.size()

        # Only LFSR and true random generator is implemented for FC layers (for now)
        if generator=='lfsr':
            rand_input, rand_weight_pos, rand_weight_neg = lfsr_init(w_size, a_size, device, prec)
        elif generator=='lfsr_split':
            rand_input, rand_weight_pos, rand_weight_neg = lfsr_init(w_size, a_size, device, prec)
            rand_input_l, rand_weight_l_pos, rand_weight_l_neg = lfsr_init(w_size, a_size, device, bit_range_l)
        else:
            rand_input, rand_weight_pos, rand_weight_neg = rand_init(w_size, a_size, device, prec)

        if generator=='lfsr_split':
            result_pos = []
            result_l_pos = []
            result_neg = []
            result_l_neg = []
        else:
            result_pos = []
            result_neg = []

        for k in range(bit_length):
            # SC computation is simulated as sum of normal FC layers on single bits
            if generator=='lfsr':
                rand_input, rand_weight_pos, rand_weight_neg = lfsr_cont(rand_input, rand_weight_pos, rand_weight_neg, bit_length=bit_length)
            else:
                rand_input, rand_weight_pos, rand_weight_neg = rand_init(w_size, a_size, device, prec)
            a_bit = (input_split > rand_input).to(compute_type)
            w_pos_bit = (w_pos_split > rand_weight_pos).to(compute_type)
            w_neg_bit = (w_neg_split > rand_weight_neg).to(compute_type)
            # For OR accumulation, having one 1 in the entire accumulation means output is one, so taking the sign of 
            # normal accumulation is equivalent to doing OR accumulation
            if forward == 'full_or':
                result_pos.append(F.linear(a_bit, w_pos_bit).sign())
                result_neg.append(F.linear(a_bit, w_neg_bit).sign())
            elif forward == 'full_bin':
                result_pos.append(F.linear(a_bit, w_pos_bit))
                result_neg.append(F.linear(a_bit, w_neg_bit))
        
        result_pos = torch.stack(result_pos, 0)
        result_neg = torch.stack(result_neg, 0)
        if generator=='lfsr_split':
            for k in range(bit_range_l):
                rand_input_l, rand_weight_l_pos, rand_weight_l_neg = lfsr_cont(rand_input_l, rand_weight_l_pos, rand_weight_l_neg, bit_length=bit_range_l)
                a_bit = (input_split > rand_input_l).to(compute_type)
                w_pos_bit = (w_l_pos_split > rand_weight_l_pos).to(compute_type)
                w_neg_bit = (w_l_neg_split > rand_weight_l_neg).to(compute_type)
                if forward == 'full_or':
                    result_l_pos.append(F.linear(a_bit, w_pos_bit).sign())
                    result_l_neg.append(F.linear(a_bit, w_neg_bit).sign())
                elif forward == 'full_bin':
                    result_l_pos.append(F.linear(a_bit, w_pos_bit))
                    result_l_neg.append(F.linear(a_bit, w_neg_bit))
            result_pos_l = torch.stack(result_l_pos, 0)
            result_neg_l = torch.stack(result_l_neg, 0)
    
    w_pos = weight.clamp(0,100)
    w_neg = -(weight.clamp(-100,0))
    activation = activation.to(w_pos.dtype)
    
    # Floating point forward pass to guide backpropagation
    if forward == 'full_or':
        result_pos_value = linear_or_approx(activation, w_pos)
        result_neg_value = linear_or_approx(activation, w_neg)
    elif forward == 'full_bin':
        result_pos_value = F.linear(activation, w_pos)
        result_neg_value = F.linear(activation, w_neg)
        
    device = str(result_pos_value.device)[-1]
        
    # Result from SC computation overwrites floating point forward pass
    if os.environ.get("Global_Sync", "1")=="1":
        if (forward=='full_or') and (generator=='lfsr_seed'):
            result_pos_value.data = result_pos/bit_length
            result_neg_value.data = result_neg/bit_length
        else:
            result_pos_value.data = result_pos.mean(0)
            result_neg_value.data = result_neg.mean(0)
        if generator=='lfsr_split':
            result_pos_value.data = result_pos_value.data + result_pos_l.mean(0)*(bit_range_l/bit_length)
            result_neg_value.data = result_neg_value.data + result_neg_l.mean(0)*(bit_range_l/bit_length)
        
    return result_pos_value - result_neg_value

def conv2d_generic(activation, weight, padding, stride, **kwargs):
    global Print_Ctr
    '''
    Generic conv2d layer
    Arguments:
    bit_length: stream length to use
    prec: weight and activation quantization precision specified using number of allowed discrete values
    share: allow limited sharing of stream generators to reduce cost and improve accuracy
    generator: stream generator to use
    forward: sc computation to use
    legacy: use older implementation without optimization
    load_unit: number of bits to load each time for progressive loading
    load_wait_w: number of cycles to wait between loading weights for progressive loading
    load_wait_a: number of cycles to wait between loading activations for progressive loading
    z_unit: number of input channels to accumulate using OR
    '''
    # print("At beginning", activation.min().item(), activation.max().item(), weight.min().item(), weight.max().item())
    bit_length = setvalue_default(kwargs, 'bit_length', 128)
    prec = setvalue_default(kwargs, 'prec', 128)
    acc = setvalue_default(kwargs, 'acc', False)
    approx = setvalue_default(kwargs, 'approx', 0)
    generator = setvalue_default(kwargs, 'generator', None)        
    share = setvalue_default(kwargs, 'share', True)
    forward = setvalue_default(kwargs, 'forward', 'full_or')
    legacy = setvalue_default(kwargs, 'legacy', False)
    load_unit = setvalue_default(kwargs, 'load_unit', 8)
    load_wait_w = setvalue_default(kwargs, 'load_wait_w', 1)
    load_wait_a = setvalue_default(kwargs, 'load_wait_a', 1)
    prec_out = setvalue_default(kwargs, 'prec_out', 24)
    split_output = setvalue_default(kwargs, 'split_output', False)
    order = setvalue_default(kwargs, 'order', 10)
    output_scale = setvalue_default(kwargs, 'scale', None)
    '''
    Conv2d specific
    '''
    z_unit = setvalue_default(kwargs, 'z_unit', 16)
    mult_lfsr = setvalue_default(kwargs, 'mult_lfsr', False)
    im2col = setvalue_default(kwargs, 'im2col', False)
    bias_coef = setvalue_default(kwargs, 'bias_coef', torch.zeros(order+1))
    std_coef = setvalue_default(kwargs, 'std_coef', torch.zeros(order+1))
    arn_calibrate = setvalue_default(kwargs, 'arn_calibrate', False)
    '''
    End of Conv2d specific
    '''
    or_compute = (forward=='full') or (forward=='z_bin')
    sync_override = setvalue_default(kwargs, 'sync', 1)
    # if sync_override==2:
    #     legacy = True
    device = activation.device
    
    cout = weight.size(0)
    
    # Quantization precision is tied to stream length for LFSR generator. E.g.: 5-bit precision is used for 
    # 32-bit streams (+1 bit precision for sign)
    if (generator=='lfsr') or (generator=='lfsr_tight') or (generator=='lfsr_mult') or (generator=='rand_mult') or (generator[:4]=='and_'):
        prec = bit_length
    if generator=='lfsr_tight':
        # seeds = np.load("seed_{0}.npy".format(bit_length))
        # seeds_arr = torch.tensor(seeds)#.to(device=activation.device)
        seeds_1 = np.load("lfsr_{0}_1.npy".format(bit_length)).astype(np.int32)
        seeds_2 = np.load("lfsr_{0}_2.npy".format(bit_length)).astype(np.int32)
        seeds_arr_1 = torch.tensor(seeds_1)#.to(device=activation.device)
        seeds_arr_2 = torch.tensor(seeds_2)#.to(device=activation.device)
    bit_range = prec-1

    dtype = activation.dtype
    if torch.is_autocast_enabled():
        dtype = torch.float16
    else:
        dtype = activation.dtype
    # dtype = compute_type
    activation = F.pad(activation, (padding[0], padding[0], padding[1], padding[1]))
    or_n = -1
    weight_prec = int(os.environ.get("Weight_Prec", "-1"))
    if os.environ.get("Global_Sync", "1")=="1" and (not config.Use_Approx) and (sync_override>0) and (sync_override<3):
        with torch.no_grad():
            if (((forward[:3]=='or_') or (forward=='full_or') or (forward=='full_bin')) and ((generator=='lfsr') or (generator[:4]=='and_') or (generator=='lfsr_mult') or (generator=='lfsr_split')) and (not legacy)):
                if forward[:3]=='or_':
                    if len(forward)==4:
                        or_n = int(forward[-1])
                        bin_config = or_n-1
                    elif len(forward)==5:
                        or_n = int(forward[3])
                        bin_config = -or_n+1
                    elif len(forward)==6:
                        or_n = int(forward[-1])
                        bin_config = or_n
                elif forward=='full_or':
                    bin_config = 0
                elif forward=='full_bin':
                    bin_config = 51200
                if (generator=='lfsr') or (generator=='lfsr_mult') or (generator=='lfsr_split'):
                    gen_config = 1
                elif generator[:4]=='and_':
                    and_n = int(generator[-1])
                    gen_config = and_n
                result_pos, result_neg = sc_extension_cuda.conv2d_generic_general_split_acc(activation.data, weight.data, bit_length, LFSR_Length, z_unit, (0,0), stride, (prec_default, load_unit, load_wait_w), bin_config, gen_config, XNOR, MUX)
                # print(result_pos.min().item(), result_pos.max().item(), result_neg.min().item(), result_neg.max().item())
                result_pos = result_pos.to(dtype=dtype)
                result_neg = result_neg.to(dtype=dtype)
            if (generator=='fixed'):
                result_pos = fixed_extension_cuda.conv2d_saturate_acc(activation.data, weight.data, 0, int(np.log2(bit_length)), 24, (0,0), stride).to(dtype=dtype)
                result_neg = torch.zeros_like(result_pos)
                result_pos = result_pos / bit_length
                result_neg = result_neg / bit_length
            if (generator=='analog'):
                if weight_prec<0:
                    print("ASC requires overriding Weight_Prec variable")
                    exit()
                result_pos,result_neg = sc_extension_cuda.conv2d_generic_analog_acc(activation.data, weight.data, bit_length, weight_prec, LFSR_Length, z_unit, (0,0), stride, (prec_default, load_unit, load_wait_w), 0, 1, XNOR, MUX)
                result_pos = (result_pos.to(dtype=dtype)) / (2**weight_prec)
                result_neg = (result_neg.to(dtype=dtype)) / (2**weight_prec)
    else:
        config.Stream_Length = bit_length
    
                                
    w_pos = weight.clamp(0,100)
    w_neg = -(weight.clamp(-100,0))
    activation = activation.to(w_pos.dtype)

    if (sync_override>=0):
        # Use normal approximationg method
        # Floating point forward pass
        if (forward == 'full_or') or (forward == 'and_2'):
            result_pos_value = conv2d_or_approx(activation, w_pos, padding=(0,0), stride=stride)
            result_neg_value = conv2d_or_approx(activation, w_neg, padding=(0,0), stride=stride)
        elif forward[:3]== 'or_':
            conv_pos_normal = F.conv2d(activation, w_pos, padding=(0,0), stride=stride)
            conv_neg_normal = F.conv2d(activation, w_neg, padding=(0,0), stride=stride)
            if len(forward)==4:
                or_n = int(forward[-1])
                if sync_override==3:
                    # Fuse OR_N approximation and bias correction
                    # result_pos_value = Or_Approx_N_Bias_Correct_Grad.apply(conv_pos_normal, or_n, bias_coef, std_coef, order)
                    # result_neg_value = Or_Approx_N_Bias_Correct_Grad.apply(conv_neg_normal, or_n, bias_coef, std_coef, order)
                    
                    # result_pos_value = Or_Approx_n_Grad.apply(conv_pos_normal, or_n)
                    # result_neg_value = Or_Approx_n_Grad.apply(conv_neg_normal, or_n)
                    # result_value = result_pos_value - result_neg_value
                    # print(conv_pos_normal.min().item(), conv_pos_normal.max().item(), result_value.min().item(), result_value.max().item())
                    # print(torch.sqrt(torch.mean(result_pos_value**2)).item(), torch.sqrt(torch.mean(conv_pos_normal**2)).item())
                    # return (result_pos_value - result_neg_value).to(dtype)
                    # result = Or_Approx_N_Bias_Correct_Grad_Both.apply(conv_pos_normal, conv_neg_normal, or_n, bias_coef, std_coef, order).to(dtype)
                    # if activation.max().item()<0.1:
                    #     print("")
                    #     print("Activation Data", activation.min().item(), activation.max().item())
                    #     print("Weight data", weight.min().item(), weight.max().item())
                    #     print("Pre act Data", conv_pos_normal.min().item(), conv_pos_normal.max().item())
                    #     print("Post act data", result.min().item(), result.max().item())
                    return Or_Approx_N_Bias_Correct_Grad_Both.apply(conv_pos_normal, conv_neg_normal, or_n, bias_coef, std_coef, order).to(dtype)
                else:
                    result_pos_value = Or_Approx_n_Grad.apply(conv_pos_normal, or_n)
                    result_neg_value = Or_Approx_n_Grad.apply(conv_neg_normal, or_n)
            elif len(forward)==5:
                or_n = int(forward[3])
                result_pos_value = or_approx_no(conv_pos_normal, or_n)
                result_neg_value = or_approx_no(conv_neg_normal, or_n)
            elif len(forward)==6:
                if int(forward[-1])==1:

                    result_pos_value = or_approx_2_1(conv_pos_normal, weight.size(1)*weight.size(2)*weight.size(3))
                    result_neg_value = or_approx_2_1(conv_neg_normal, weight.size(1)*weight.size(2)*weight.size(3))
                elif int(forward[-1])==4:
                    result_pos_value = or_approx_2_4(conv_pos_normal, weight.size(1)*weight.size(2)*weight.size(3))
                    result_neg_value = or_approx_2_4(conv_neg_normal, weight.size(1)*weight.size(2)*weight.size(3))
        elif forward=='full_bin':
            result_pos_value = F.conv2d(activation, weight.clamp(-100,100), padding=(0,0), stride=stride)
            result_neg_value = torch.zeros_like(result_pos_value)

    # Result from SC computation overwrites floating point forward pass
    if os.environ.get("Global_Sync", "1")=="1" and (not config.Use_Approx) and (sync_override==1):
        if forward == 'prog_search':
            result_pos_value.data = result_pos / torch.tensor([16.,32.,64.,128.]).to(device=device,dtype=torch.float32)
            result_neg_value.data = result_neg / torch.tensor([16.,32.,64.,128.]).to(device=device,dtype=torch.float32)
        else:
            logging_level = os.environ.get("LOGGING_LEVEL", "0")
            if logging_level == '3':
                conv_pos_normal = F.conv2d(activation, w_pos, padding=(0,0), stride=stride)
                conv_neg_normal = F.conv2d(activation, w_neg, padding=(0,0), stride=stride)
                result_pos_bit = result_pos / bit_length
                result_neg_bit = result_neg / bit_length
                dump_dir = os.environ.get("DUMP_DIR", "dump")
                dump_prefix = os.path.join(dump_dir, "w{0}_{1}_{2}_{3}_a{4}_{5}_{6}_{7}".format(weight.size(0), weight.size(1), weight.size(2), weight.size(3), activation.size(0), activation.size(1), activation.size(2), activation.size(3)))
                np.save(dump_prefix+'_pos_acc.npy', conv_pos_normal.detach().cpu().numpy())
                np.save(dump_prefix+'_neg_acc.npy', conv_neg_normal.detach().cpu().numpy())
                np.save(dump_prefix+'_pos_bit.npy', result_pos_bit.detach().cpu().numpy())
                np.save(dump_prefix+'_neg_bit.npy', result_neg_bit.detach().cpu().numpy())
                np.save(dump_prefix+'_pos_act.npy', result_pos_value.detach().cpu().numpy())
                np.save(dump_prefix+'_neg_act.npy', result_neg_value.detach().cpu().numpy())
            # print(forward, generator)
            result_pos_value.data = result_pos / bit_length
            result_neg_value.data = result_neg / bit_length
            # print(result_pos_value.min().item(), result_pos_value.max().item(), result_neg_value.min().item(), result_neg_value.max().item())
            # Idealized CeMux emulation. Overwrite floating-point and fixed-point forward pass
            if os.environ.get("CEMUX", "-1")=="1":
                prec_cemux = int(np.log2(bit_length))
                weight_total = weight.abs().sum((1,2,3)).data
                weight_scale = weight * (1/weight_total.view(-1,1,1,1))
                weight_scale = QuantizeGrad.apply(weight_scale, prec_cemux)
                result_pos_value = F.conv2d(activation, weight_scale, padding=(0,0), stride=stride)
                result_neg_value = torch.zeros_like(result_pos_value)

                result_pos_value = QuantizeGrad.apply(result_pos_value, prec_cemux)
                result_neg_value = QuantizeGrad.apply(result_neg_value, prec_cemux)
                result_pos_value = result_pos_value * weight_total.view(-1,1,1)
                result_neg_value = result_neg_value * weight_total.view(-1,1,1)

    elif sync_override==2:
        # Perform approximation calibration
        # Multi output value version
        with torch.no_grad():
            result_pos_bit = result_pos/bit_length
            result_neg_bit = result_neg/bit_length
            result_bit = torch.cat([result_pos_bit.reshape(-1), result_neg_bit.reshape(-1)])
            result_act = torch.cat([result_pos_value.reshape(-1), result_neg_value.reshape(-1)])

            bias_coef_new, std_coef_new = compute_bias_std_coef(or_n, result_bit, result_act, order)
            bias_coef[:bias_coef_new.size(0)] = bias_coef_new
            std_coef[:std_coef_new.size(0)] = std_coef_new

        result_pos_value = Bias_Correct_Grad.apply(result_pos_value, bias_coef, std_coef, order)
        result_neg_value = Bias_Correct_Grad.apply(result_neg_value, bias_coef, std_coef, order)
    elif sync_override==3:
        # Fused version will have returned by now
        pass

    result_pos_value.data = result_pos_value.data.to(dtype)
    result_neg_value.data = result_neg_value.data.to(dtype)
    if split_output:
        return result_pos_value, result_neg_value
    elif sync_override==2:
        return result_pos_value-result_neg_value, bias_coef, std_coef
    else:
        
        return result_pos_value-result_neg_value
