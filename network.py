import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import utils_class
import utils_functional as utils_f
import os

from utils import *
# from launch_param import *

'''
File containing all model definitions
'''

def generate_default_runargs(**kwargs):
    run_args = {}
    utils_f.setdict_default(run_args, kwargs, "generator", 'lfsr')
    utils_f.setdict_default(run_args, kwargs, "forward", "or_2")
    utils_f.setdict_default(run_args, kwargs, "approx", False)
    utils_f.setdict_default(run_args, kwargs, "prec_out", 24)
    utils_f.setdict_default(run_args, kwargs, "prec", 7)
    utils_f.setdict_default(run_args, kwargs, "err", 6)
    run_args['load_unit'] = 8
    run_args['load_wait_w'] = 1
    run_args['load_wait_a'] = 1
    utils_f.setdict_default(run_args, kwargs, "sync", 1)
    utils_f.setdict_default(run_args, kwargs, 'scale_offset', 0)
    return run_args

class VGG16_add_partial(nn.Module):
    '''
    VGG16 modified for SC
    Stream length (err)/load unit/load wait values are specified using a string of 6 integers. Pooling layers automatically use half the stream length. Stream length = 2**{value specified}
    The first value is used for the layer1-2, second for layer3-4, third for layer5-7, fourth for layer8-10, fifth for layer11-13, sixth for FC layers
    E.g.: to achieve load unit=2, load wait=2, stream length=64 for all layers without pooling and 32 for all layers with pooling, err=[666666], load_unit=[222222], load_wait=[222222]
    '''
    def __init__(self, num_classes=10, uniform=False, sc_compute='or_1', generator='lfsr', legacy=False, approx=False, err=[5,5,5,5,5,7], sync=1):
        super(VGG16_add_partial, self).__init__()
        conv1_args = generate_default_runargs(err=int(err[0]), generator=generator, forward=sc_compute, approx=approx, sync=sync)
        conv2_args = generate_default_runargs(err=int(err[1]), generator=generator, forward=sc_compute, approx=approx, sync=sync)
        conv3_args = generate_default_runargs(err=int(err[2]), generator=generator, forward=sc_compute, approx=approx, sync=sync)
        conv4_args = generate_default_runargs(err=int(err[3]), generator=generator, forward=sc_compute, approx=approx, sync=sync)
        conv5_args = generate_default_runargs(err=int(err[4]), generator=generator, forward=sc_compute, approx=approx, sync=sync)
        fc1_args = generate_default_runargs(err=int(err[5]), generator=generator, forward=sc_compute, approx=approx, sync=sync)
        self.conv1 = utils_class.Conv2d_Add_Partial(3, 64, kernel_size=3, padding=1, bias=False, **conv1_args)
        self.conv2 = utils_class.Conv2d_Add_Partial(64, 64, kernel_size=3, padding=1, bias=False, **conv1_args)

        self.conv3 = utils_class.Conv2d_Add_Partial(64, 128, kernel_size=3, padding=1, bias=False, **conv2_args)
        self.conv4 = utils_class.Conv2d_Add_Partial(128, 128, kernel_size=3, padding=1, bias=False, **conv2_args)

        self.conv5 = utils_class.Conv2d_Add_Partial(128, 256, kernel_size=3, padding=1, bias=False, **conv3_args)
        self.conv6 = utils_class.Conv2d_Add_Partial(256, 256, kernel_size=3, padding=1, bias=False, **conv3_args)
        self.conv7 = utils_class.Conv2d_Add_Partial(256, 256, kernel_size=3, padding=1, bias=False, **conv3_args)

        self.conv8 = utils_class.Conv2d_Add_Partial(256, 512, kernel_size=3, padding=1, bias=False, **conv4_args)
        self.conv9 = utils_class.Conv2d_Add_Partial(512, 512, kernel_size=3, padding=1, bias=False, **conv4_args)
        self.conv10 = utils_class.Conv2d_Add_Partial(512, 512, kernel_size=3, padding=1, bias=False, **conv4_args)

        self.conv11 = utils_class.Conv2d_Add_Partial(512, 512, kernel_size=3, padding=1, bias=False, **conv5_args)
        self.conv12 = utils_class.Conv2d_Add_Partial(512, 512, kernel_size=3, padding=1, bias=False, **conv5_args)
        self.conv13 = utils_class.Conv2d_Add_Partial(512, 512, kernel_size=3, padding=1, bias=False, **conv5_args)

        self.fc1 = utils_class.Conv2d_Add_Partial(512, 10, kernel_size=1, padding=0, bias=False, **fc1_args)
        self.pool = nn.AvgPool2d(2)

        self.bn1 = utils_class.BatchNorm2d_fixed(64)
        self.bn2 = utils_class.BatchNorm2d_fixed(64)
        self.bn3 = utils_class.BatchNorm2d_fixed(128)
        self.bn4 = utils_class.BatchNorm2d_fixed(128)
        self.bn5 = utils_class.BatchNorm2d_fixed(256)
        self.bn6 = utils_class.BatchNorm2d_fixed(256)
        self.bn7 = utils_class.BatchNorm2d_fixed(256)
        self.bn8 = utils_class.BatchNorm2d_fixed(512)
        self.bn9 = utils_class.BatchNorm2d_fixed(512)
        self.bn10 = utils_class.BatchNorm2d_fixed(512)
        self.bn11 = utils_class.BatchNorm2d_fixed(512)
        self.bn12 = utils_class.BatchNorm2d_fixed(512)
        self.bn13 = utils_class.BatchNorm2d_fixed(512)
        self.bnfc1 = utils_class.BatchNorm1d_fixed(10)
        self.tanh = nn.Hardtanh(0,1)
        self.relu = nn.ReLU()

        self.layer1 = nn.Sequential(self.conv1, self.bn1, self.tanh,
                                    self.conv2, self.bn2, self.relu, self.pool, self.tanh)
        self.layer2 = nn.Sequential(self.conv3, self.bn3, self.tanh,
                                    self.conv4, self.bn4, self.relu, self.pool, self.tanh)
        self.layer3 = nn.Sequential(self.conv5, self.bn5, self.tanh,
                                    self.conv6, self.bn6, self.tanh,
                                    self.conv7, self.bn7, self.relu, self.pool, self.tanh)
        self.layer4 = nn.Sequential(self.conv8, self.bn8, self.tanh,
                                    self.conv9, self.bn9, self.tanh,
                                    self.conv10,self.bn10,self.relu, self.pool, self.tanh)
        self.layer5 = nn.Sequential(self.conv11,self.bn11,self.tanh,
                                    self.conv12,self.bn12,self.tanh,
                                    self.conv13,self.bn13,self.relu, self.pool, self.tanh)

        self.num_classes = num_classes
        self.prune = 0
        self.sync = sync
        
        if uniform:
            for mod in self.modules():
                if isinstance(mod, nn.Conv2d):
                    mod.weight.data *= 0.5
    def forward(self, x):
        utils_f.update_prune(self, self.prune)
        utils_f.update_sync(self, self.sync)
        # print("Start", self.training)
        x = self.layer1(x)
        # print(x.min(), x.max())
        x = self.layer2(x)
        # print(x.min(), x.max())
        x = self.layer3(x)
        # print(x.min(), x.max())
        x = self.layer4(x)
        # print(x.min(), x.max())
        x = self.layer5(x)
        # print(x.min(), x.max())
        x = self.fc1(x)
        x = x.view(-1,10)
        x = self.bnfc1(x)
        # print(x.min(), x.max())
        return x

class VGG11_add_partial(nn.Module):
    '''
    VGG16 modified for SC
    Stream length (err)/load unit/load wait values are specified using a string of 6 integers. Pooling layers automatically use half the stream length. Stream length = 2**{value specified}
    The first value is used for the layer1-2, second for layer3-4, third for layer5-7, fourth for layer8-10, fifth for layer11-13, sixth for FC layers
    E.g.: to achieve load unit=2, load wait=2, stream length=64 for all layers without pooling and 32 for all layers with pooling, err=[666666], load_unit=[222222], load_wait=[222222]
    '''
    def __init__(self, num_classes=10, uniform=False, sc_compute='or_1', generator='lfsr', legacy=False, approx=False, err=[5,5,5,5,5,7], sync=1):
        super(VGG11_add_partial, self).__init__()
        conv1_args = generate_default_runargs(err=int(err[0]), generator=generator, forward=sc_compute, approx=approx, sync=sync)
        conv2_args = generate_default_runargs(err=int(err[1]), generator=generator, forward=sc_compute, approx=approx, sync=sync)
        conv3_args = generate_default_runargs(err=int(err[2]), generator=generator, forward=sc_compute, approx=approx, sync=sync)
        conv4_args = generate_default_runargs(err=int(err[3]), generator=generator, forward=sc_compute, approx=approx, sync=sync)
        conv5_args = generate_default_runargs(err=int(err[4]), generator=generator, forward=sc_compute, approx=approx, sync=sync)
        fc1_args = generate_default_runargs(err=int(err[5]), generator=generator, forward=sc_compute, approx=approx, sync=sync)
        self.conv1 = utils_class.Conv2d_Add_Partial(3, 64, kernel_size=3, padding=1, bias=False, **conv1_args)

        self.conv3 = utils_class.Conv2d_Add_Partial(64, 128, kernel_size=3, padding=1, bias=False, **conv2_args)

        self.conv5 = utils_class.Conv2d_Add_Partial(128, 256, kernel_size=3, padding=1, bias=False, **conv3_args)
        self.conv7 = utils_class.Conv2d_Add_Partial(256, 256, kernel_size=3, padding=1, bias=False, **conv3_args)

        self.conv8 = utils_class.Conv2d_Add_Partial(256, 512, kernel_size=3, padding=1, bias=False, **conv4_args)
        self.conv10 = utils_class.Conv2d_Add_Partial(512, 512, kernel_size=3, padding=1, bias=False, **conv4_args)

        self.conv11 = utils_class.Conv2d_Add_Partial(512, 512, kernel_size=3, padding=1, bias=False, **conv5_args)
        self.conv13 = utils_class.Conv2d_Add_Partial(512, 512, kernel_size=3, padding=1, bias=False, **conv5_args)

        self.fc1 = utils_class.Conv2d_Add_Partial(512, 10, kernel_size=1, padding=0, bias=False, **fc1_args)
        self.pool = nn.AvgPool2d(2)

        self.bn1 = utils_class.BatchNorm2d_fixed(64)
        self.bn3 = utils_class.BatchNorm2d_fixed(128)
        self.bn5 = utils_class.BatchNorm2d_fixed(256)
        self.bn7 = utils_class.BatchNorm2d_fixed(256)
        self.bn8 = utils_class.BatchNorm2d_fixed(512)
        self.bn10 = utils_class.BatchNorm2d_fixed(512)
        self.bn11 = utils_class.BatchNorm2d_fixed(512)
        self.bn13 = utils_class.BatchNorm2d_fixed(512)
        self.bnfc1 = utils_class.BatchNorm1d_fixed(10)
        self.tanh = nn.Hardtanh(0,1)
        self.relu = nn.ReLU()

        self.layer1 = nn.Sequential(self.conv1, self.bn1, self.relu, self.pool, self.tanh)
        self.layer2 = nn.Sequential(self.conv3, self.bn3, self.relu, self.pool, self.tanh)
        self.layer3 = nn.Sequential(self.conv5, self.bn5, self.tanh,
                                    self.conv7, self.bn7, self.relu, self.pool, self.tanh)
        self.layer4 = nn.Sequential(self.conv8, self.bn8, self.tanh,
                                    self.conv10,self.bn10,self.relu, self.pool, self.tanh)
        self.layer5 = nn.Sequential(self.conv11,self.bn11,self.tanh,
                                    self.conv13,self.bn13,self.relu, self.pool, self.tanh)

        self.num_classes = num_classes
        self.prune = 0
        self.sync = sync
        
        if uniform:
            for mod in self.modules():
                if isinstance(mod, nn.Conv2d):
                    mod.weight.data *= 4
    def forward(self, x):
        utils_f.update_prune(self, self.prune)
        utils_f.update_sync(self, self.sync)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.fc1(x)
        x = x.view(-1,10)
        x = self.bnfc1(x)
        return x

def generate_default_runargs(**kwargs):
    run_args = {}
    utils_f.setdict_default(run_args, kwargs, "generator", 'lfsr')
    utils_f.setdict_default(run_args, kwargs, "forward", "or_2")
    utils_f.setdict_default(run_args, kwargs, "approx", False)
    utils_f.setdict_default(run_args, kwargs, "prec_out", 24)
    utils_f.setdict_default(run_args, kwargs, "prec", 7)
    utils_f.setdict_default(run_args, kwargs, "err", 6)
    run_args['load_unit'] = 8
    run_args['load_wait_w'] = 1
    run_args['load_wait_a'] = 1
    utils_f.setdict_default(run_args, kwargs, "sync", 1)
    utils_f.setdict_default(run_args, kwargs, 'scale_offset', 0)
    return run_args

class CONV_tiny_add_partial(nn.Module):
    '''
    4-layer CNN for SC
    Stream length (err)/load unit/load wait values are specified using a string of 4 integers, one for each layer.
    E.g.: to achieve load unit=2, load wait=2, stream length=32 for all conv layers 128 for the last fc layer, err=[5557], load_unit=[2222], load_wait=[2222]
    '''
    def __init__(self, num_classes=10, uniform=False, generator='lfsr', sc_compute='or_1', legacy=False, approx=False, err=[6,6,6,7], sync=1):
        super(CONV_tiny_add_partial, self).__init__()
        conv1_args = generate_default_runargs(err=int(err[0]), generator=generator, forward=sc_compute, approx=approx, sync=sync)
        conv2_args = generate_default_runargs(err=int(err[1]), generator=generator, forward=sc_compute, approx=approx, sync=sync)
        conv3_args = generate_default_runargs(err=int(err[2]), generator=generator, forward=sc_compute, approx=approx, sync=sync)
        fc1_args = generate_default_runargs(err=int(err[3]), generator=generator, forward=sc_compute, approx=approx, sync=sync)

        self.conv1 = utils_class.Conv2d_Add_Partial(3, 32, 5, padding=2, bias=False, **conv1_args)
        self.bn1 = utils_class.BatchNorm2d_fixed(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2)
        self.tanh1 = nn.Hardtanh(0,1)

        self.conv2 = utils_class.Conv2d_Add_Partial(32, 32, 5, padding=2, bias=False, **conv2_args)
        self.bn2 = utils_class.BatchNorm2d_fixed(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(2)
        self.tanh2 = nn.Hardtanh(0,1)

        self.conv3 = utils_class.Conv2d_Add_Partial(32, 64, 5, padding=2, bias=False, **conv3_args)
        self.bn3 = utils_class.BatchNorm2d_fixed(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AvgPool2d(2)
        self.tanh3 = nn.Hardtanh(0,1)

        self.fc1   = utils_class.Conv2d_Add_Partial(64, num_classes, 4, padding=0, bias=False, **fc1_args)
        self.bn4 = utils_class.BatchNorm1d_fixed(10)

        self.layer1 = nn.Sequential(self.conv1, self.bn1, self.relu1, self.pool1, self.tanh1)
        self.layer2 = nn.Sequential(self.conv2, self.bn2, self.relu2, self.pool2, self.tanh2)
        self.layer3 = nn.Sequential(self.conv3, self.bn3, self.relu3, self.pool3, self.tanh3)
        self.layer4 = nn.Sequential(self.fc1, nn.Flatten(), self.bn4)
        
        self.num_classes = num_classes
        self.prune = 0
        self.sync = sync
        
        # Scale up weights for low precision and stream length. Other underflow prevents effective training
        if uniform:
            self.conv1.weight.data*= 4
            self.conv2.weight.data*= 4
            self.conv3.weight.data*= 4
            self.fc1.weight.data  *= 4
                             
    def forward(self, x, target=None):
        utils_f.update_prune(self, self.prune)
        utils_f.update_sync(self, self.sync)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x