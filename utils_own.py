import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import time

from utils import *
# from launch_param import *

from sys import exit

scale_factor = 256.0

'''
Auxilliary functions mostly for half-precision training
'''          
def set_data(model, model_with_data):
    for param, param_w_data in zip(model.parameters(), model_with_data.parameters()):
        if param_w_data.data is not None:
            param.data.copy_(param_w_data.data)
            
def get_device(gpu_prior, gpu_map):
    '''
    Get idle GPU device
    '''
    import nvidia_smi as ns
    ns.nvmlInit()
    gpu_count = ns.nvmlDeviceGetCount()
    if gpu_count < 1:
        exit("No GPU")
    if gpu_count < len(gpu_prior):
        gpu_prior = list(range(gpu_count))
    if gpu_count < len(gpu_map):
        gpu_map = list(range(gpu_count))
    handles = []
    for i in gpu_prior:
        handles.append(ns.nvmlDeviceGetHandleByIndex(i))
    device_free = False
    while not device_free:
        for d, handle in enumerate(handles):
            res_handle = ns.nvmlDeviceGetUtilizationRates(handle)
            if (res_handle.gpu<70) and (res_handle.memory<50):
                device_cur = gpu_map[d]
                free_count=1
                for i in range(10):
                    time.sleep(0.5)
                    res_handle = ns.nvmlDeviceGetUtilizationRates(handle)
                    if (res_handle.gpu<70) and (res_handle.memory<50):
                        free_count+=1
                if free_count>10:
                    device_free = True
                    break
        if not device_free:
            # Prevent too frequent queries
            time.sleep(5)
    return device_cur
        
def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None, verbal=False, monitor=False, scaler=None, dtype="fp32", scheduler=None, dali=False, acc_limit=128, channels_last=False, transfer=False, model_base=None, calibrate=False):
    '''
    Copied from BinarizedNN.pytorch rep
    '''
    global scale_factor
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    torch.cuda.empty_cache()
    end = time.time()
    acc = 0
    
    if (dtype=="fp32") or (dtype=='tf32'):
        torch_type = torch.float32
    elif dtype=="fp16":
        torch_type = torch.float16
    elif dtype=="bf16":
        torch_type = torch.bfloat16

    try:
        device_cur = model.fc1.weight.device
    except:
        device_cur = model.fc.weight.device
    else:
        pass

    if monitor and training:
        print_freq = max(len(data_loader)//5, 1)
        # print_freq = 2
    else:
        print_freq = 2000
        # print_freq = 2

    for i, data in enumerate(data_loader):
        logging_level = os.environ.get("LOGGING_LEVEL", "0")
        if logging_level == '3':
            if i<1:
                continue
            if i>1:
                break
            dump_dir = os.environ.get("DUMP_DIR", "dump")
            np.save(os.path.join(dump_dir, "target.npy"), target.numpy())
        if dali:
            inputs = data[0]["data"]#.to(device_cur, non_blocking=True)
            target = data[0]["label"].squeeze(-1).long()#.to(device_cur, non_blocking=True)
        else:
            inputs = data[0].to(device_cur, non_blocking=True)
            target = data[1].to(device_cur, non_blocking=True)
        if channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)
        # measure data loading time
        data_time.update(time.time() - end)
        
        # Modifying calibration parameters
        if calibrate:
            if i%max(len(data_loader)//5, 100)==99:
                model.sync=2
                # Calibration step requires an additional gradient storage
                # Use half batch size to prevent overflow
                inputs = inputs[:inputs.size(0)//2]
                target = target[:target.size(0)//2]
            else:
                model.sync=3
        # compute output
        if torch_type!=torch.float32:
            with torch.cuda.amp.autocast(dtype=torch_type):
                output = model(inputs)
        else:
            output = model(inputs)
        
        # output = output.float()
        loss = criterion(output, target)
        if transfer:
            model_base.eval()
            with torch.no_grad():
                target_layer1 = model_base.layer1(inputs)
                target_layer2 = model_base.layer2(target_layer1)
                target_layer3 = model_base.layer3(target_layer2)
                target_layer4 = model_base.layer4(target_layer3)
            output_layer1 = model.layer1(inputs)
            output_layer2 = model.layer2(output_layer1.clone().detach())
            output_layer3 = model.layer3(output_layer2.clone().detach())
            output_layer4 = model.layer4(output_layer3.clone().detach())
            loss_distill = F.mse_loss(output_layer1, target_layer1) + F.mse_loss(output_layer2, target_layer2) + F.mse_loss(output_layer3, target_layer3) + F.mse_loss(output_layer4, target_layer4)
            loss = loss+loss_distill*0.5
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        
        acc += inputs.size(0)
            
        if training:
            # compute gradient and do SGD step
            if scaler is not None:
                loss = scaler.scale(loss)
            # with torch.cuda.amp.autocast(dtype=torch_type):
            loss.backward(retain_graph=True)
            if acc>=acc_limit:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pass
                    optimizer.step()
                optimizer.zero_grad()
                acc = 0
        if scheduler is not None:
            scheduler.step()
        
        # Skip the first few iterations
        if i>10:
            batch_time.update(time.time() - end)
        end = time.time()

        if monitor and (i%print_freq==print_freq-1):
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.6f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i+1, len(data_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))     
    if training:
        for p in model.modules():
            if isinstance(p, nn.Linear) or isinstance(p, nn.Conv2d):
                p.weight.data.clamp_(-1,0.999)
    if not training:
        if verbal:
            print('Epoch: [{0}]\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                epoch, loss=losses, top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer, monitor=False, scaler=None, dtype='fp32', scheduler=None, dali=False, acc_limit=128, channels_last=False, transfer=False, model_base=None, calibrate=False):
    '''
    Copied from BinarizedNN.pytorch rep. Iteration specifies the number of batches to run (to save time)
    '''
    # switch to train mode
    model.train()
    optimizer.zero_grad()
    return forward(data_loader, model, criterion, epoch, training=True, optimizer=optimizer, monitor=monitor, scaler=scaler, dtype=dtype, scheduler=scheduler, dali=dali, acc_limit=acc_limit, channels_last=channels_last, transfer=transfer, model_base=model_base, calibrate=calibrate)


def validate(data_loader, model, criterion, epoch, verbal=False, monitor=False, dtype='fp32', dali=False, channels_last=False):
    '''
    Copied from BinarizedNN.pytorch rep. Verbal allows training information to be displayed. Iteration specifies the number of batches to run (to save time)
    '''
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        return forward(data_loader, model, criterion, epoch, training=False, optimizer=None, dtype=dtype, verbal=verbal, monitor=monitor, dali=dali, channels_last=channels_last)
