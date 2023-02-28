from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Change the following line if you want to compile for other architectures
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.6"

setup(
    name='fixed_extension_cuda',
    ext_modules=[
        CUDAExtension(name='fixed_extension_cuda', 
                      sources=['fixed_cuda.cpp','fixed_cuda_kernel.cu'],
                      extra_compile_args={'nvcc': ['-maxrregcount=64']}
                      )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
