from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

os.environ['CFLAGS'] = '-march=native -fopenmp'

setup(name='sc_extension',
      ext_modules=[CppExtension('sc_extension', ['sc.cpp', 'sc_conv.cpp', 'sc_matmul.cpp'])],
      cmdclass={'build_ext': BuildExtension})
