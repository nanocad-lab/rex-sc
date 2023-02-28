from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Change the following line if you want to compile for other architectures
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.6"

setup(
    name='sc_extension_cuda',
    ext_modules=[
        CUDAExtension(name='sc_extension_cuda', 
                      sources=['sc_cuda.cpp','sc_cuda_kernel.cu', 'sc_cuda_matmul.cu', 'sc_cuda_pw.cu', 'sc_device.cu', 'sc_cuda_conv.cu'],
                      extra_compile_args={'nvcc': ['-t=0']}
                      )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
