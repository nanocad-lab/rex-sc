import os
import torch
import torch.nn.functional as F
import logging.config
import shutil
import pandas as pd
from bokeh.io import output_file, save, show
from bokeh.plotting import figure
from bokeh.layouts import column

from torch.optim.lr_scheduler import _LRScheduler
from platform import uname

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
import numpy as np

import torchvision
import network
import torchvision.transforms as transforms

def in_wsl() -> bool:
    return 'microsoft-standard' in uname().release

def setup_logging(log_file='log.txt', name='', mode='w'):
    """Setup logging configuration
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode=mode)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger(name).addHandler(console)


class ResultsLog(object):

    def __init__(self, path='results.csv', plot_path=None):
        self.path = path
        self.plot_path = plot_path or (self.path + '.html')
        self.figures = []
        self.results = None

    def add(self, **kwargs):
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        if self.results is None:
            self.results = df
        else:
            self.results = self.results.append(df, ignore_index=True)

    def save(self, title='Training Results'):
        if len(self.figures) > 0:
            if os.path.isfile(self.plot_path):
                os.remove(self.plot_path)
            output_file(self.plot_path, title=title)
            plot = column(*self.figures)
            save(plot)
            self.figures = []
        self.results.to_csv(self.path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.path
        if os.path.isfile(path):
            self.results.read_csv(path)

    def show(self):
        if len(self.figures) > 0:
            plot = column(*self.figures)
            show(plot)

    #def plot(self, *kargs, **kwargs):
    #    line = Line(data=self.results, *kargs, **kwargs)
    #    self.figures.append(line)

    def image(self, *kargs, **kwargs):
        fig = figure()
        fig.image(*kargs, **kwargs)
        self.figures.append(fig)


def save_checkpoint(state, is_best, path='.', filename='checkpoint.pth.tar', save_all=False):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))
    if save_all:
        shutil.copyfile(filename, os.path.join(
            path, 'checkpoint_epoch_%s.pth.tar' % state['epoch']))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

__optimizers = {
    'SGD': torch.optim.SGD,
    'ASGD': torch.optim.ASGD,
    'Adam': torch.optim.Adam,
    'Adamax': torch.optim.Adamax,
    'Adagrad': torch.optim.Adagrad,
    'Adadelta': torch.optim.Adadelta,
    'Rprop': torch.optim.Rprop,
    'RMSprop': torch.optim.RMSprop
}


def adjust_optimizer(optimizer, epoch, config):
    """Reconfigures the optimizer according to epoch and config dict"""
    def modify_optimizer(optimizer, setting):
        if 'optimizer' in setting:
            optimizer = __optimizers[setting['optimizer']](
                optimizer.param_groups)
            logging.debug('OPTIMIZER - setting method = %s' %
                          setting['optimizer'])
        for param_group in optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    logging.debug('OPTIMIZER - setting %s = %s' %
                                  (key, setting[key]))
                    param_group[key] = setting[key]
        return optimizer

    if callable(config):
        optimizer = modify_optimizer(optimizer, config(epoch))
    else:
        for e in range(epoch + 1):  # run over all epochs - sticky setting
            if e in config:
                optimizer = modify_optimizer(optimizer, config[e])

    return optimizer


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.float().topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class PolynomialDecay(_LRScheduler):
    def __init__(self, optimizer, decay_steps, end_lr=0.0001, power=1.0, last_epoch=-1):
        self.decay_steps = decay_steps
        self.end_lr = end_lr
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        return [
            (base_lr - self.end_lr) * ((1 - min(self.last_epoch, self.decay_steps) /
                                        self.decay_steps) ** self.power) + self.end_lr
            for base_lr in self.base_lrs
        ]

"""
Warmup Scheduler and Label Smoothing is from https://github.com/NUS-HPC-AI-Lab/LARS-ImageNet-PyTorch/blob/main/utils.py
"""

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, after_scheduler, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()

        return [self.last_epoch / self.warmup_epochs * lr for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
        else:
            return super().step(epoch)


class PolynomialWarmup(WarmupScheduler):
    def __init__(self, optimizer, decay_steps, warmup_steps=0, end_lr=0.0001, power=1.0, last_epoch=-1):
        base_scheduler = PolynomialDecay(
            optimizer, decay_steps - warmup_steps, end_lr=end_lr, power=power, last_epoch=last_epoch)
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)

class LabelSmoothLoss(torch.nn.Module):
    
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

"""
DALI pipeline is from https://github.com/NVIDIA/DALI/blob/main/docs/examples/use_cases/pytorch/resnet50/main.py
"""
@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
                                            #    random_aspect_ratio=[0.8, 1.25],
                                            #    random_area=[0.1, 1.0],
                                            #    num_attempts=100,
                                               )
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_LINEAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_LINEAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT16,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels

def get_model_and_dataset(args, setup):
    # setup = args.setup
    if len(setup)<11:
        exit("Wrong total number of parameters")
    setup_dataset = setup[0]
    setup_size = setup[1]
    setup_err = setup[2]
    setup_zunit = setup[3]
    setup_generator = setup[4]
    setup_compute = setup[5:7]
    setup_relu = setup[7]
    setup_pooling = setup[8]
    setup_precout = setup[9:11]
    if len(setup)>=12:
        setup_sync = int(setup[11])
    else:
        setup_sync = 1

    b = args.batch

    if setup_dataset not in ['c', 's']:
        exit("Wrong dataset parameter")
    else:
        if setup_dataset=='c':
            dataset='CIFAR10'
        elif setup_dataset=='s':
            dataset='SVHN'
    if setup_size not in ['0', '1', '3']:
        exit("Wrong model parameter")
    else:
        if setup_size=='0':
            model='TinyConv'
        elif setup_size=='1':
            model='VGG-16'
        elif setup_size=='3':
            model='VGG-11'
    if setup_err not in ['4','5','6','7','8','9']:
        exit("Wrong stream length parameter")
    else:
        if model=='TinyConv':
            errs = [int(setup_err), int(setup_err), int(setup_err), np.maximum(7, int(setup_err))]
        elif model=='VGG-16':
            errs = [int(setup_err), int(setup_err), int(setup_err), int(setup_err), int(setup_err), np.maximum(7, int(setup_err))]
        elif model=='VGG-11':
            errs = [int(setup_err), int(setup_err), int(setup_err), int(setup_err), int(setup_err), np.maximum(7, int(setup_err))]
    if setup_zunit not in ['3','4','5','6','7','8']:
        exit("Wrong zunit parameter")
    else:
        z_unit_in = int(setup_zunit)
        if model=='TinyConv':
            z_units = [z_unit_in, z_unit_in, z_unit_in, z_unit_in]
        elif model=='VGG-16':
            z_units = [z_unit_in, z_unit_in, z_unit_in, z_unit_in, z_unit_in, z_unit_in]
        elif model=='VGG-11':
            z_units = [z_unit_in, z_unit_in, z_unit_in, z_unit_in, z_unit_in, z_unit_in]
    if setup_generator not in ['f', 'l', 'r', 's']:
        exit("Wrong generator setup")
    else:
        if setup_generator=='f':
            generator = 'fixed'
        elif setup_generator=='l':
            generator = 'lfsr'
        elif setup_generator=='r':
            generator = 'random'
        elif setup_generator=='s':
            generator = 'lfsr_split'
    if (setup_compute not in ['fb']) and setup_compute[0]!='o' and setup_compute[0]!='d':
        exit("Wrong compute setup")
    else:
        if setup_compute[0]=='o':
            compute = 'or_{0}'.format(setup_compute[1])
        elif setup_compute=='fb':
            compute = 'full_bin'
    if setup_relu not in ['b', 'a']:
        exit("Wrong relu parameter")
    else:
        if setup_relu=='b':
            relu = True
        elif setup_relu=='a':
            relu = False
    if setup_pooling not in ['m', 'a']:
        exit("Wrong pooling parameter")
    else:
        if setup_pooling=='m':
            max_pool = True
        elif setup_pooling=='a':
            max_pool = False
    try:
        prec_out = int(setup_precout)
    except:
        exit("Wrong output precision parameter")
    else:
        pass

    uniform = True

    if (dataset=='CIFAR10') or (dataset=='SVHN'):
        if dataset=='CIFAR10':
            trainset = torchvision.datasets.CIFAR10(root='/data', train=True, download=True,
                                                    transform=transforms.Compose([transforms.RandomCrop(32, 4),
                                                                                transforms.RandomHorizontalFlip(),
                                                                                transforms.ToTensor()]))     
            testset = torchvision.datasets.CIFAR10(root='/data', train=False, download=True, transform=transforms.ToTensor())    
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=b, shuffle=True, num_workers=2)
            testloader = torch.utils.data.DataLoader(testset, batch_size=b, shuffle=False, num_workers=2)
        elif dataset=='SVHN':
            trainset = torchvision.datasets.SVHN(root='/data', split='train', download=True, transform=transforms.Compose([transforms.RandomCrop(32,4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]))
            testset = torchvision.datasets.SVHN(root='/data', split='test', download=True, transform=transforms.Compose([transforms.ToTensor()]))
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=b, shuffle=True, num_workers=2)
            testloader = torch.utils.data.DataLoader(testset, batch_size=b, shuffle=False, num_workers=2)
        if model=='TinyConv':
            net = network.CONV_tiny_add_partial(uniform=uniform, sc_compute=compute, generator=generator, legacy=args.legacy, err=errs, approx=(args.run[6]=='a'), sync=setup_sync)
            layers = 4
        elif model=='VGG-16':
            net = network.VGG16_add_partial(uniform=uniform, sc_compute=compute, generator=generator, legacy=args.legacy, err=errs, approx=(args.run[6]=='a'), sync=setup_sync)
            layers = 6
        elif model=='VGG-11':
            net = network.VGG11_add_partial(uniform=uniform, sc_compute=compute, generator=generator, legacy=args.legacy, err=errs, approx=(args.run[6]=='a'), sync=setup_sync)
            layers = 6
        if (layers != len(errs)):
            print('Mismatch')
            return -1

    return net, trainloader, testloader