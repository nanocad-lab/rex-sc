# REX-SC
This is a software library used for training of stochastic computing(SC)-based neural networks tuned for REX-SC.

## Directory Structure
```
.
├── main_conv.py              # Train a small CNN in using REX-SC
├── main_imagenet.py          # Train a large CNN for imagenet
├── utils.py                  # Miscellaneous functions
├── network.py                # Model definitions
├── utils_class.py            # Layer definitions
├── utils_functional.py       # SC-specific function definitions
├── utils_own.py              # Training function definitions
├── resnet{_sc}.py            # Original and SC version of Resnet (https://arxiv.org/abs/1512.03385) models
├── prepare_env.sh            # Create conda environment and install prerequisites
├── torch_c                   # C++-based torch extension
    ├── sc-extension          # SC-specific torch extension
        ├── sc.cpp            # SC kernel definitions for CPU
        ├── sc_conv.cpp       # SC convolution kernel for CPU
        ├── sc_matmul.cpp     # SC linear kernel for CPU
        ├── sc_cpu.hpp        # Shared SC functions for CPU
        ├── sc_cuda.cpp       # SC kernel definitions for GPU
        ├── sc_cuda_conv.cu   # SC shared GPU functions for convolution
        ├── sc_cuda_conv.cuh  # SC shared GPU function definitions for convolution
        ├── sc_cuda_kernel.cu # SC convolution kernels for GPU
        ├── sc_cuda_matmul.cu # SC linear kernels for GPU
        ├── sc_cuda_pw.cu     # point-wise (OR-n activation, activation calibration, error injection) functions for GPU
        ├── sc_device.cuh     # Shared SC function definitions and inline functions for GPU
        ├── fixed.cpp         # fixed-point convolution with saturation for CPU
        ├── fixed_cuda.cpp    # fixed-point convolution with saturation definition for GPU
        ├── fixed_cuda_kernel.cu
                              # fixed-point convolution implementation for GPU
        ├── setup.py          # Compilation file for CPU kernels
        ├── setup_cuda.py     # Compilation file for GPU kernels
        ├── setup_fixed.py    # Compilation file for CPU fixed-point kernels
        ├── setup_cuda_fixed.py
                              # Compilation file for GPU fixed-point kernels
        ├── compile_all.sh    # Clean up and compile the SC kernels
        ├── compile_fixed.sh  # Clean up and compile the fixed-point kernels
```

## Prerequisites
Ubuntu 18.04 or 20.04 LTS (other distros not tested)

NVIDIA driver >= 470

CUDA >= 11.3

conda >= 4.8.4

With conda installed, to create an environment with the required dependencies, execute:
```
chmod +x prepare_env.sh
./prepare_env.sh
```
By default the custom kernels are compiled for cc 7.5 (Turing) and 8.6 (Ampere consumer) GPUs. To compile for other architectures, modify the respective lines in torch_c/sc-extension/{setup_cuda.py, setup_cuda_fixed.py}

## Example Usage
To train TinyConv using OR-1, 32-bit stream (5-bit unipolar precision), automatically select gpu, execute:
```
python main_conv.py --setup c055lo1ba24 --run o0n000df0
```
OR-1, 64-bit stream (6-bit unipolar precision)
```
python main_conv.py --setup c065lo1ba24 --run o0n000df0
```
OR-2, 64-bit stream
```
python main_conv.py --setup c065lo2ba24 --run o0n000df0
```
OR-2, 64-bit stream with error injection
```
python main_conv.py --setup c065lo2ba24 --run c0n000df0
```
OR-3, 64-bit stream with error injection, manually assign to GPU 0
```
python main_conv.py --setup c065lo3ba24 --run c0n000df0 --device 0
```
FXP6 (5-bit split-unipolar), 24-bit saturation
```
python main_conv.py --setup c055ffbba24 --run o0n000df0
```
CeMux (Idealized, https://arxiv.org/abs/2108.12326), 128-bit stream
```
CEMUX=1 python main_conv.py --setup c075ffbba24 --run o0n000df0
```
To train VGG-16 using OR-1, 32-bit stream, automatically select gpu, execute:
```
python main_conv.py --setup c155lo1ba24 --run o0n000df0
```
To train Resnet-18 on Imagenet using OR-2, 32-bit stream (5-bit unipolar precision) for 35 epochs, execute:
```
python main_imagenet.py --architecture resnet18 --channels_last --dali --err 5 --calibrate
```
70 epochs (setup used to get accuracy results):
```
python main_imagenet.py --architecture resnet18 --channels_last --dali --err 5 --epoch 70 --calibrate
```
Resnet-34, 64-bit streams, OR-3
```
python main_imagenet.py --architecture resnet34 --channels_last --dali --err 6 --calibrate
```
For Imagenet training, please first download and decompress the ImageNet2012 dataset. By default, main_imagenet.py assumes that the data is in /data/imagenet/train and /data/imagenet/val. The root directory of the dataset can be changed using --data_dir parameter:
```
python main_imagenet.py --channels_last --dali --err 5 --data_dir /path/to/imagenet/dataset
```
