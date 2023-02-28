import torch
import torch.nn as nn
import torch.nn.functional as F

import utils_functional as utils_f
import os
import numpy as np

# from launch_param import *
   
'''
Custom layers for SC
'''
class Conv2d_Add_Partial(nn.Conv2d):
    '''
    SC Conv2d using partial binary add
    '''
    def __init__(self, *kargs, **kwargs):
        self.run_args = {}
        utils_f.setdict_default(self.run_args, kwargs, 'approx', False)
        utils_f.setdict_default(self.run_args, kwargs, 'forward', "or_2")
        utils_f.setdict_default(self.run_args, kwargs, 'generator', "lfsr")
        utils_f.setdict_default(self.run_args, kwargs, 'err', 6)
        utils_f.setdict_default(self.run_args, kwargs, 'load_unit', 8)
        utils_f.setdict_default(self.run_args, kwargs, 'load_wait_w', 1)
        utils_f.setdict_default(self.run_args, kwargs, 'load_wait_a', 1)
        utils_f.setdict_default(self.run_args, kwargs, 'prec_out', 24)
        utils_f.setdict_default(self.run_args, kwargs, 'sync', 1)
        utils_f.setdict_default(self.run_args, kwargs, 'prec', 7)
        utils_f.setdict_default(self.run_args, kwargs, 'split_output', False)
        utils_f.setdict_default(self.run_args, kwargs, 'order', 5)
        utils_f.setdict_default(self.run_args, kwargs, 'scale_offset', 0)
        # print(self.run_args)
        self.run_args['bit_length'] = 2**self.run_args['err']
        super(Conv2d_Add_Partial, self).__init__(*kargs, **kwargs)
        self.register_buffer("weight_flag", self.weight.data.clone())
        self.register_buffer('prune', torch.tensor([0.]))
        self.register_buffer("bias_coef", torch.zeros(self.run_args['order']+1))
        self.register_buffer("std_coef", torch.zeros(self.run_args['order']+1))

    def forward(self, input):
        '''
        Arguments:
        prec: weight and activation precision to quantize to
        err: stream length in the form of 2**err
        forward: sc compute. Specifically how accumulation is done
        generator: stream generator
        z_unit: number of input channles to sum using OR accumulation when forward==yz_bin
        legacy: disable accelerated kernels
        load_unit: number of bits to load each time for progressive loading
        load_wait_w: number of cycles to wait between loading weights for progressive loading
        load_wait_a: number of cycles to wait between loading activations for progressive loading
        '''
        input = utils_f.QuantizeGrad.apply(input, self.run_args['prec'])
        g_in = input.size(1)//self.groups
        g_wgt = self.weight.size(0)//self.groups

        if os.environ.get("Weight_Prec", "-1")!="-1":
            weight_prec = int(os.environ.get("Weight_Prec"))
        else:
            weight_prec = self.run_args['prec']
        weight = utils_f.QuantizeGrad.apply(self.weight, weight_prec)
        out_list = []
        for g in range(self.groups):
            if self.run_args['sync']==2:
                out, self.bias_coef, self.std_coef = utils_f.conv2d_generic(input[:,g*g_in:(g+1)*g_in], weight[g*g_wgt:(g+1)*g_wgt], padding=self.padding, stride=self.stride, **self.run_args)
            else:
                self.run_args['bias_coef'] = self.bias_coef
                self.run_args['std_coef'] = self.std_coef
                out = utils_f.conv2d_generic(input[:,g*g_in:(g+1)*g_in], weight[g*g_wgt:(g+1)*g_wgt], padding=self.padding, stride=self.stride, **self.run_args)
            out_list.append(out)

        out = torch.stack(out_list, dim=2)
        out = out.reshape(input.size(0), -1, out.size(3), out.size(4))

        logging_level = os.environ.get("LOGGING_LEVEL", "0")
        if logging_level == '2':
            out_max = out.max().item()
            out_min = out.min().item()
            if (out_max<0.01) or (out_min>-0.01):
                print("Small output value, max {0}, min {1}".format(out_max, out_min), self.weight.size(), input.size())
            if torch.isnan(input).any():
                print("Detected nan input", input.size(), self.weight.size())
            if torch.isnan(self.weight).any():
                print("Detected nan weight", self.weight.size())
        return out

class BatchNorm_Clamp(nn.Module):
    '''
    More like poor man's fake quantization
    '''
    def __init__(self, **kwargs):
        super(BatchNorm_Clamp, self).__init__()
        self.momentum = utils_f.setvalue_default(kwargs, 'momentum', 0.9)
        self.register_buffer('scale_float', torch.tensor(1.))
        self.register_buffer('scale_range', torch.tensor(1.))
    def forward(self, x):
        with torch.no_grad():
            if self.training:
                max_value = torch.max(torch.abs(x))
                scale_float = self.momentum*self.scale_float + (1-self.momentum)*max_value
                scale_range = torch.log2(scale_float).ceil()
                scale_float = (scale_float*2**(7-scale_range)).ceil()*2**(scale_range-7)
                self.scale_range = scale_range
                self.scale_float = scale_float
        x = x*(1/self.scale_float)
        return x

class BatchNorm2d_fixed(nn.BatchNorm2d):
    '''
    Quantized 2d batchnorm
    '''
    def __init__(self, *kargs, **kwargs):
        super(BatchNorm2d_fixed, self).__init__(*kargs, **kwargs)
        self.register_buffer('scale', torch.tensor(1.))
    def forward(self, x):
        out = F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=self.training)
        if self.training:
            mean = x.mean(dim=(0,2,3))
            var = x.var(dim=(0,2,3), unbiased=False)
        else:
            mean = self.running_mean
            var = self.running_var
        if self.affine:
            weight = self.weight.data
            bias = self.bias.data
        else:
            weight = 1
            bias = 0
        w_n = weight/torch.sqrt(var + self.eps)
        b_n = bias - mean*weight/torch.sqrt(var + self.eps)
        w_n, self.scale = utils_f.quantize_shift(w_n.detach())
        b_n, _ = utils_f.quantize_shift(b_n.detach(), self.scale)
        w_n = w_n.reshape(w_n.size(0),1,1)
        b_n = b_n.reshape(b_n.size(0),1,1)
        out.data = (x.data*w_n + b_n).to(out.dtype)
        return out
    
class BatchNorm1d_fixed(nn.BatchNorm1d):
    '''
    Quantized 1d batchnorm
    '''
    def __init__(self, *kargs, **kwargs):
        super(BatchNorm1d_fixed, self).__init__(*kargs, **kwargs)
        self.register_buffer('scale', torch.tensor(1.))
    def forward(self, x):
        if self.weight.size(0)==1:
            # print("Doing 1d bn")
            num_classes = x.size(1)
            out = F.batch_norm(x, self.running_mean.repeat(num_classes), self.running_var.repeat(num_classes), self.weight.repeat(num_classes), self.bias.repeat(num_classes), training=self.training)
            if self.training:
                mean = x.mean()
                var = x.var(unbiased=False)
            else:
                mean = self.running_mean
                var = self.running_var
            if self.affine:
                weight = self.weight.data
                bias = self.bias.data
            else:
                weight = 1
                bias = 0
        else:
            out = F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=self.training)
            if self.training:
                mean = x.mean(dim=(0))
                var = x.var(dim=(0), unbiased=False)
            else:
                mean = self.running_mean
                var = self.running_var
            if self.affine:
                weight = self.weight.data
                bias = self.bias.data
            else:
                weight = 1
                bias = 0
        w_n = weight/torch.sqrt(var + self.eps)
        b_n = bias - mean*weight/torch.sqrt(var + self.eps)
        w_n, self.scale = utils_f.quantize_shift(w_n)
        b_n, _ = utils_f.quantize_shift(b_n, self.scale)
        out.data = (x.data*w_n + b_n).to(out.dtype)
        return out
class Linear_Or(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(Linear_Or, self).__init__(*kargs, **kwargs)
        self.register_buffer("weight_flag", self.weight.data.clone())
        self.add_or = False
    
    def forward(self, input, add_or=False, prec=7, err=7, add_full=1, true_or=False, add_count=False, generator='lfsr', prune=0, fixed=False):
        if prec is not None:
            quant=True
        else:
            quant=False
            prec=8
        add_or = self.add_or or add_or
        input.data = utils_f.quantize(input.data, prec=prec)
        if prune>0:
            weight, weight_flag = utils_f.quantize(self.weight, prec=prec, prune=prune)
            if not fixed:
                self.weight_flag = weight_flag
            weight.data = weight.data * self.weight_flag
            # print(self.weight_flag.sum(), (self.weight==0).sum(), self.weight_flag.size())
            out = utils_f.linear_generic(input, weight, bit_length=2**err, forward='full_or', generator=generator)
        else:
            # weight = utils_f.quantize(self.weight, prec=prec)
            weight = utils_f.QuantizeGrad.apply(self.weight, prec)
            out = utils_f.linear_generic(input, weight, bit_length=2**err, forward='full_or', generator=generator)
        return out