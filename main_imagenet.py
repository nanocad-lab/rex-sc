import argparse
import os

import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torch.optim as optim
from utils import *
import utils_own
import numpy as np
import time

import torchvision.models as models

import resnet_sc

from torch.cuda.amp import GradScaler

parser = argparse.ArgumentParser(description='PyTorch imageNet Training for SC Darpa')

parser.add_argument('--save_dir', metavar='SAVE_DIR', default='./', type=str, help='save dir')
parser.add_argument('--architecture', metavar='ARCH', default='resnet34', type=str, help='network architecture to use')
parser.add_argument('--seed', metavar='SEED', default=0, type=int, help='seed to used for this run')
parser.add_argument('--device', metavar='DEVICE', default=-1, type=int, help='the device to use')
parser.add_argument('--dtype', default='fp16', type=str, help='Training datatype')
parser.add_argument('--lr', metavar='LR', default=1e-3, type=float, help='initial learning rate to use')
parser.add_argument('--tune_lr', default=3e-4, type=float, help='Learning rate for the tuning steps')
parser.add_argument('--end_lr', type=float, default=0, help='End learning rate')
parser.add_argument('--workers', metavar='WORKER', default=4, type=int, help='Number of workers to use')
parser.add_argument('--data_dir', metavar='DATADIR', default='/data/imagenet/', type=str, help='data directory to use')
parser.add_argument('-b', metavar='BATCH', default=128, type=int, help='Batch size to use')
parser.add_argument('--l2', metavar='L2', default=0, type=float, help='L2 loss to use')
parser.add_argument('-p', '--parallel', metavar='PARALLEL', default=0, type=int, help='Do data parallel training')
parser.add_argument('--add_or', metavar='ADD_OR', default=0, type=int, help='simulate addition using OR')
parser.add_argument('--monitor', metavar='MONITOR', default=1, type=int, help='Monitor loss and computation time during training')
parser.add_argument('--val', metavar='VAL', default=0, type=int, help='Evaluate')
parser.add_argument('--warmup', default=0, type=int, help='Reduce learning rate in the beginning')
parser.add_argument('--resume', default=0, type=int, help='resume from training')
parser.add_argument('--float', default=0, type=int, help='train the floating-point model instead')
parser.add_argument('--epoch', default=35, type=int, help='Number of epochs to train')
parser.add_argument('--optim', default='adam', type=str, help='Default optimizer to use')
parser.add_argument('--sc_compute', default='or_2', type=str, help="SC Compute to use")

parser.add_argument('--setup', metavar='SETUP', default='nothing', type=str, help='See main_imagenet_prod.py for more details')
parser.add_argument('--label-smoothing', type=float, default=0.1, help='label smoothing (default 0.1)')
parser.add_argument('--dali', default=False, action="store_true", help="Use Dali for data loading")
parser.add_argument('--acc_limit', default=128, type=int, help="Gradient accumulation steps to perform")
parser.add_argument('--channels_last', default=False, action="store_true", help="Use channels_last format")
parser.add_argument('--uniform', default=False, action="store_true", help='Use larger weights')
parser.add_argument('--prec', default=7, type=int)
parser.add_argument('--err', default=6, type=int)
parser.add_argument('--logging_level', default='0', type=str)
parser.add_argument('--calibrate', default=False, action="store_true")

# import imagenet_data
GPU_MAP = [0,1,2]
GPU_PRIOR = [0,1,2]
    
def main():
    global args
    
    args = parser.parse_args()
    seed = args.seed
    save_dir = args.save_dir
    arch = args.architecture
    lr = args.lr
    device = args.device
    torch.cuda.set_device(device)
    workers = args.workers
    data_dir = args.data_dir
    b = args.b
    l2 = args.l2
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=True
    # torch.autograd.set_detect_anomaly(True)
    if args.resume == 1:
        resume = True
    else:
        resume = False
    # If logging=0, just do normal logging
    # If logging=1, dump
    # If logging=2, find abnormalities
    # os.environ["LOGGING_LEVEL"]=args.logging_level
    dali_cpu = False
    if args.float == 1:
        if arch=='resnet34':
            model = models.resnet.resnet34(pretrained=False)
        elif arch=='resnet18':
            model = models.resnet.resnet18(pretrained=False)
        elif arch=='resnet50':
            model = models.resnet.resnet50(pretrained=False)
            dali_cpu = True
        print("Using pretrained model")
    else:
        dali_cpu = True
        if arch=='resnet34':
            model = resnet_sc.resnet34(pretrained=True, sc_compute=args.sc_compute, uniform=args.uniform, prec=args.prec, err=args.err)
        elif arch=='resnet18':
            model = resnet_sc.resnet18(pretrained=True, sc_compute=args.sc_compute, uniform=args.uniform, prec=args.prec, err=args.err)
        elif arch=='resnet50':
            model = resnet_sc.resnet50(pretrained=True, sc_compute=args.sc_compute, uniform=args.uniform, prec=args.prec, err=args.err)
        print("SC using pretrained model")
    errs = []
    
    if args.dtype=='fp32':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.dtype=='tf32':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.dtype=='fp16':
        scaler = GradScaler()
    else:
        scaler = None

    calibrate = args.calibrate
    
    save_dir = os.path.join('./', save_dir)
    ckpt_file = os.path.join(save_dir, 'model.pth')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = os.path.join(save_dir, arch)
    if resume or args.val==1:
        setup_logging(save_file+'_log.txt', mode='a')
    else:
        setup_logging(save_file+'_log.txt', mode='w')
    logging.info("saving to %s", save_file)
    result_dic = save_file + '_result.pt'
    save_file += '.pt'
    
    
    valdir = os.path.join(data_dir, 'val')
    traindir = os.path.join(data_dir, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    trainset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            normalize
        ]))
    valset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]))
    
    if device<0:
        device = utils_own.get_device(list(range(torch.cuda.device_count())),list(range(torch.cuda.device_count())))
    model.cuda(device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    
    if args.dali:
        pipe = create_dali_pipeline(batch_size=b,
                                    num_threads=workers,
                                    device_id=device,
                                    seed=args.seed,
                                    data_dir=traindir,
                                    crop=224,
                                    size=256,
                                    dali_cpu=dali_cpu,
                                    shard_id=0,
                                    num_shards=1,
                                    is_training=True)
        pipe.build()
        train_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
        pipe = create_dali_pipeline(batch_size=b,
                                    num_threads=workers,
                                    device_id=device,
                                    seed=args.seed,
                                    data_dir=valdir,
                                    crop=224,
                                    size=256,
                                    dali_cpu=dali_cpu,
                                    shard_id=0,
                                    num_shards=1,
                                    is_training=False)
        pipe.build()
        val_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
    else:
        # pin_memory=True does not work in WSL
        if in_wsl():
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=b, shuffle=True, num_workers=workers, pin_memory=False)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=b, shuffle=False, num_workers=workers, pin_memory=False)
        else:
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=b, shuffle=True, num_workers=workers, pin_memory=True, pin_memory_device='cuda:{0}'.format(device))
            val_loader = torch.utils.data.DataLoader(valset, batch_size=b, shuffle=False, num_workers=workers, pin_memory=True, pin_memory_device='cuda:{0}'.format(device))
    criterion = LabelSmoothLoss(args.label_smoothing)

    if args.val==1:
        print("Evaluating", approx_most)
        if args.float != 1:
            ckpt = torch.load(ckpt_file, map_location='cpu')
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
        model.eval()
        val_loss, val_prec1, val_prec5 = utils_own.validate(
            val_loader, model, criterion, 0, verbal=True, monitor=args.monitor, dali=args.dali, channels_last=args.channels_last)
        return 0

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    best_prec1 = 0
    model.add_or = False
    for mod in model.modules():
        if hasattr(mod, "add_or"):
            mod.add_or = False

    num_steps_epoch = len(train_loader)
    if args.warmup>0:
        scheduler = PolynomialWarmup(optimizer, decay_steps=args.epoch*num_steps_epoch,
                                            warmup_steps=args.warmup*num_steps_epoch,
                                            end_lr=args.end_lr, power=2.0, last_epoch=-1)
    else:
        gamma = np.exp(np.log(((args.end_lr+1e-5)/args.lr))/(args.epoch*num_steps_epoch-1))
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)


    start_epoch = 0
    if resume:
        if os.path.exists(ckpt_file):
            ckpt = torch.load(ckpt_file, map_location='cpu')
            start_epoch = min(ckpt['epoch'], 300)
            print("Starting from epoch", start_epoch)
            if 'model_state_dict' in ckpt.keys():
                model.load_state_dict(ckpt['model_state_dict'], strict=False)
            if 'epoch' in ckpt.keys():
                start_epoch = ckpt['epoch']
            if 'optimizer_state_dict' in ckpt.keys():
                pass
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt.keys():
                pass
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        elif os.path.exists(save_file):
            model_dict = torch.load(save_file, map_location='cpu')
            model.load_state_dict(model_dict, strict=False)
            print("Starting from save file")

    epoch_limit = args.epoch
    print(start_epoch, epoch_limit)
    epoch_end = time.time()
    for epoch in range(start_epoch, epoch_limit):  # loop over the dataset multiple times
        
        if calibrate and (epoch==epoch_limit-5):
            calibrate = False
            model.sync = 1
            total_steps = (epoch_limit-epoch)*num_steps_epoch
            warmup_steps = 0
            decay_steps = int(total_steps-warmup_steps)
            optimizer = optim.Adam(model.parameters(), lr=args.tune_lr, weight_decay=l2)
            scheduler = PolynomialWarmup(optimizer, decay_steps=decay_steps,
                                         warmup_steps=warmup_steps, end_lr=args.end_lr, power=2.0, last_epoch=-1)

        if calibrate and (epoch>epoch_limit-5):
            calibrate = False
            model.sync = 1
        # train for one epoch
        train_loss, train_prec1, train_prec5 = utils_own.train(
            train_loader, model, criterion, epoch, optimizer, monitor=args.monitor, scaler=scaler, dtype=args.dtype, scheduler=scheduler, dali=args.dali, acc_limit=args.acc_limit, channels_last=args.channels_last, calibrate=calibrate)

        val_loss, val_prec1, val_prec5 = utils_own.validate(
            val_loader, model, criterion, epoch, verbal=True, monitor=args.monitor, dali=args.dali, channels_last=args.channels_last)
        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)
        if args.parallel==1:
            torch.save({'epoch': epoch+1, 'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, ckpt_file)
        else:
            if args.calibrate and not calibrate:
                torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, ckpt_file+'c')
            else:
                torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, ckpt_file)
        if is_best:
            if args.parallel==1:
                torch.save(model.module.state_dict(), save_file)
            else:
                torch.save(model.state_dict(), save_file)
        logging.info('Epoch: {0}\t'
                        'T Prec1 {train_prec1:.3f} \t'
                        'T Prec5 {train_prec5: 3f} \t'
                        'V Prec1 {val_prec1:.3f} \t'
                        'V Prec5 {val_prec5:.3f} \t'
                        'Epoch Time {epoch_time:.3f}s \t'
                        'Is best? '
                        .format(epoch+1, train_prec1=train_prec1, val_prec1=val_prec1,
                            train_prec5=train_prec5, val_prec5=val_prec5, epoch_time=time.time()-epoch_end)+str(is_best))
        epoch_end = time.time()
        
if __name__ == '__main__':
    main()
