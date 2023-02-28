import argparse
import os
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time

from utils import *

import utils_own

import torch.optim as optim

from torch.cuda.amp import GradScaler
'''
File controlling the overall training and validation procedure
Default parameters train a 4-layer CNN using LFSR generator with 128-bit stream length for all layers. Accumulation is done using OR for x and z dimensions of the filter, while y dimension is accumulated using fixed-point adders.
E.g.: for a 32x5x5 filter, the "32" and the first "5" dimension are accumulated using OR, and the last "5" dimension is accumulated using fixed-point adders, so only 4 integer additions is needed.
'''

parser = argparse.ArgumentParser(description='PyTorch small CNN Training for SC')

parser.add_argument('--save_dir', metavar='SAVE_DIR', default='./', type=str, help='save dir. Default translates to ../training_data_sc/cifar_mid')
parser.add_argument('--dataset', metavar='DATASET', default='CIFAR10', type=str, help='dataset to use. Choose between CIFAR10, SVHN, and MNIST')
parser.add_argument('--seed', metavar='SEED', default=0, type=int, help='seed to use for this run')
parser.add_argument('--device', metavar='DEVICE', default=-1, type=int, help='the device to use. Device>=0 specifies the GPU ID, device=-1 specifies automatic device placement')
parser.add_argument('--optim', metavar='OPTIM', default='Adam', type=str, help='optimizer to use. Choose between Adam, RMSprop, SGD (with momentum) and Adabound')
parser.add_argument('--lr', metavar='LR', default=8e-3, type=float, help='leaning rate to use')
parser.add_argument('--tune_lr', default=3e-3, type=float, help='Learning rate for the tuning steps')
parser.add_argument('--load_unit', metavar='LUNIT', default='8888', type=str, help='Number of bits to load each time for progressive loading')
parser.add_argument('--load_wait', metavar='LWAIT', default='1111', type=str, help='Number of cycles to wait between loads for progressive loading')
parser.add_argument('-b','--batch', metavar='BATCH', default=256, type=int, help='Batch size to use')
parser.add_argument('-e','--epoch', metavar='EPOCH', default=100, type=int, help='Default epoch to use')
parser.add_argument('--legacy', metavar='LEGACY', default=0, type=int, help='Use legacy computation. Disabling it uses accelerated CUDA kernels when possible, and some functions are only available in accelerated versions')
# parser.add_argument('--half', default=0, type=int, help='Use half')
parser.add_argument('--dtype', type=str, default='tf32', help='Data type to use for training')

parser.add_argument('--setup', metavar='SETUP', default='nothing', type=str, help='See main_conv.py for more details')
parser.add_argument('--run', metavar='RUN', default='nothing', type=str, help='See main_conv.py for more details.')
parser.add_argument('--end_lr', default=1e-5, type=float, help='end learning rate')

'''
setup parameters:
dataset, c=cifar10, s=svhn, m=mnist
size, 0=TinyConv, 1=VGG, 3=VGG-11
err(stream length), 5=32, 6=64
z_unit, 5=32, 6=64
generator, f=fixed, l=lfsr, r=random, s=lfsr_split
compute, fo=full or, 1d=1d_bin, 2d=2d_bin, yz=yz_bin, zb=z_bin, o{x}=or_{x}, fb=full bin, d{x}=or_{x}+1d_bin
relu, b=relu before pooling, a=relu after pooling
pooling, m=max_pooling, a=average pooling
prec_out, two digits, output precision 
'''
'''
run parameters:
mode, o=overwrite, r=resume, v=validate, t=tune, c=calibrate
monitor, 0=don't monitor, 1=monitor
initialization, n=no uniform, u=uniform (Increases weight magnitude to improve training stability)
prune, 3 digits, percentage to prune. 000=not pruning anything, 100=pruning anything, 090=pruning 90%
approx_most, d=don't do approximate training, a=do approximate training (deprecated)
half, h=half precision, f=full precision
seed, seed value
'''

gpu_map = [0,1,2]
gpu_prior = [0,1,2]

def main():
    global args, best_prec1, gpu_map, gpu_prior
    args = parser.parse_args()
    run = args.run
    if len(run)!=9:
        exit("Wrong total number of runtime parameters")

    run_mode = run[0]
    run_monitor = run[1]
    run_init = run[2]
    run_prune = run[3:6]
    run_approx = run[6]
    run_half = run[7]
    run_seed = run[8]

    if run_mode not in ['o', 'r', 'v', 't', 'c']:
        exit("Wrong run mode")
    else:
        overwrite = False
        resume = False
        val = False
        tune = False
        calibrate = False
        if run_mode=='o':
            overwrite = True
        elif run_mode=='r':
            resume = True
        elif run_mode=='v':
            val = True
        elif run_mode=='t':
            tune = True
        elif run_mode=='c':
            overwrite = True
            calibrate = True
    if run_monitor not in ['0', '1']:
        exit("Wrong monitor mode")
    else:
        if run_monitor=='0':
            monitor = False
        elif run_monitor=='1':
            monitor = True
    if run_init not in ["n", "u"]:
        exit("Wrong initialization mode")
    else:
        if run_init=='n':
            uniform = False
        elif run_init=='u':
            uniform = True
    try:
        prune = int(run_prune)
    except:
        exit("Wrong pruning parameter")
    else:
        prune = float(prune)/100
    if run_approx not in ['d', 'a']:
        exit("Wrong approximation parameter")
    else:
        if run_approx=='d':
            approx_most = False
        elif run_approx=='a':
            approx_most = True
    if run_half not in ['h', 'f']:
        exit("Wrong training precision parameter")
    else:
        if run_half=='f':
            half = False
        elif run_half=='h':
            half = True
    try:
        seed = int(run_seed)
    except:
        exit("Wrong seed parameter")
    else:
        pass

    optim_choice = args.optim
    lr = args.lr
    
    # Setting seed allows reproducible results
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=True

    device = args.device
    
    epoch_limit = args.epoch
    net, trainloader, testloader = get_model_and_dataset(args, args.setup)

    if args.dtype=='fp32':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif args.dtype=='tf32':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.dtype=='fp16':
        scaler = GradScaler()
    else:
        scaler = None

    # Default save directory is one level up in the hierarchy
    save_dir = os.path.join('./', args.save_dir)
    ckpt_file = os.path.join(save_dir, 'model.pth')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file = os.path.join(save_dir, 'CONV_small')

    # Prevents overwriting logging files if already exists
    if (not val) and (prune==0) and (not resume) and (not tune):
        if os.path.exists(save_file + '_log.txt') and (not overwrite):
            print("Save file already exists. Delete manually if want to overwrite")
            return 0
    setup_logging(save_file + '_log.txt', mode='a')
    logging.info("saving to %s", save_file)
    logging.info("Setup parameters {0}".format(args.setup))
    logging.info("Runtime parameters {0}".format(run))
    result_dic = save_file + '_result.pt'
    save_file += '.pt'

    if resume or tune:
        if os.path.exists(ckpt_file):
            ckpt = torch.load(ckpt_file, map_location='cpu')
            resume_from_ckpt = True
            resume_from_save = False
        elif os.path.exists(save_file):
            model_dict = torch.load(save_file, map_location='cpu')
            resume_from_ckpt = False
            resume_from_save = True
        else:
            resume_from_ckpt = False
            resume_from_save = False    
    
    torch.cuda.empty_cache()
    net.prune = prune

    if device==-2:
        pass
    else:
        if device<0:
            device = utils_own.get_device(gpu_prior, gpu_map)
        torch.cuda.set_device(device)
        net.cuda(device)
        
    if val:
        saved_state_dict = torch.load(save_file, map_location="cpu")
        net.load_state_dict(saved_state_dict, strict=False)
        print("Loaded checkpoint")
        net.eval()
    
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothLoss(0.1)

    # model specifies the parameters to train on. Since all parameters are trained here, there is no difference between
    # model and net
    model = net
   
    # Switches between different optimizers
    if optim_choice=='Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    elif optim_choice=='RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=0.0001)
    elif optim_choice=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0, momentum=0.01)
    
    # Learning rate scheduler that anneals learning rate if loss fails decrease for some epochs
    num_steps_epoch = len(trainloader)
    num_epochs_decay = args.epoch

    scheduler = PolynomialWarmup(optimizer, decay_steps=num_epochs_decay*num_steps_epoch,
                                warmup_steps=2*num_steps_epoch,
                                end_lr=args.end_lr, power=2.0, last_epoch=-1)
    
    best_prec1 = 0
    val_prec1 = 0

    if prune>0:
        net.prune = prune
    
    if val:
        # Validate accuracy without training
        val_loss, val_prec1, val_prec5 = utils_own.validate(
            testloader, net, criterion, 0, verbal=True, monitor=monitor)
        return 0
    else:
        # Prevents overwriting existing save files
        if os.path.exists(save_file) and (not overwrite) and (not resume) and (not tune):
            print("Save file already exists. Delete manually if want to overwrite")
            return 0


    start_epoch = 0
    if resume or tune:
        if resume_from_ckpt:
            start_epoch = min(ckpt['epoch'], 1000)
            print(start_epoch)
            if 'model_state_dict' in ckpt.keys():
                mismatch_bn = False
                if hasattr(model, "bn4"):
                    if isinstance(model.bn4, nn.Identity):
                        print("Mismatched bn parameters")
                        del ckpt['model_state_dict']['bn4.weight'], ckpt['model_state_dict']['bn4.bias'], ckpt['model_state_dict']['bn4.running_mean'], ckpt['model_state_dict']['bn4.running_var']
                    elif model.bn4.weight.data.size(0)!=ckpt['model_state_dict']['bn4.weight'].size(0):
                        print("Mismatched bn parameters, ", model.bn4.weight.data.size(0), ckpt['model_state_dict']['bn4.weight'].size(0))
                        del ckpt['model_state_dict']['bn4.weight'], ckpt['model_state_dict']['bn4.bias'], ckpt['model_state_dict']['bn4.running_mean'], ckpt['model_state_dict']['bn4.running_var']
                model.load_state_dict(ckpt['model_state_dict'], strict=False)
            if not tune:
                if 'optimizer_state_dict' in ckpt.keys():
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if 'scheduler_state_dict' in ckpt.keys():
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            else:
                total_steps = (args.epoch-start_epoch)*num_steps_epoch
                warmup_steps = int(total_steps * 0.02)
                decay_steps = int(total_steps - warmup_steps)
                scheduler = PolynomialWarmup(optimizer, decay_steps=decay_steps,
                                            warmup_steps=warmup_steps, end_lr=args.end_lr, power=2.0, last_epoch=-1)
        elif resume_from_save:
            print('Resume from save file')
            model.load_state_dict(model_dict, strict=False)

    save_every = int(os.environ.get("SAVE_EVERY", "0"))
    print(start_epoch, epoch_limit)
    end = time.time()
    for epoch in range(start_epoch, epoch_limit):  # loop over the dataset multiple times
        if prune>0:
            if epoch==int(0.05*epoch_limit):
                net.fixed=True

        # Turn off calibration for the last few epochs
        if calibrate and (epoch==epoch_limit-20):
            calibrate = False
            net.sync = 1
            total_steps = (epoch_limit-epoch)*num_steps_epoch
            warmup_steps = int(total_steps * 0.02)
            decay_steps = int(total_steps - warmup_steps)
            optimizer = optim.Adam(model.parameters(), lr=args.tune_lr, weight_decay=0.0001)
            scheduler = PolynomialWarmup(optimizer, decay_steps=decay_steps,
                                         warmup_steps=warmup_steps, end_lr=args.end_lr, power=2.0, last_epoch=-1)

        # train for one epoch
        train_loss, train_prec1, train_prec5 = utils_own.train(
            trainloader, net, criterion, epoch, optimizer, scaler=scaler, monitor=monitor, dtype=args.dtype, scheduler=scheduler, calibrate=calibrate)
        if calibrate:
            net.sync = 1
        # evaluate on validation set
        val_loss, val_prec1, val_prec5 = utils_own.validate(
            testloader, net, criterion, epoch, verbal=False, monitor=monitor)

        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)

        if not tune:
            torch.save({'epoch': epoch+1, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, ckpt_file)
        if (save_every>0) and (epoch%save_every==save_every-1):
            torch.save({'epoch': epoch+1, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, os.path.join('../', args.save_dir, 'model_{0}.pth'.format(epoch+1)))
        if is_best:
            torch.save(net.state_dict(), save_file)
            logging.info('Epoch: {0}\t'
                        'Training Prec {train_prec1:.3f} \t'
                        'Validation Prec {val_prec1:.3f} \t'
                        'Time {time:.3f}s \t best'
                        .format(epoch+1, train_prec1=train_prec1, val_prec1=val_prec1, time=time.time()-end))
        else:
            logging.info('Epoch: {0}\t'
                        'Training Prec {train_prec1:.3f} \t'
                        'Validation Prec {val_prec1:.3f} \t'
                        'Time {time:.3f}s \t not best'
                        .format(epoch+1, train_prec1=train_prec1, val_prec1=val_prec1, time=time.time()-end))
        end = time.time()
        # scheduler.step(val_loss)
    
    logging.info('\nTraining finished!. Validation accuracy {0}'.format(best_prec1))
    return 0
        
if __name__ == '__main__':
    main()
