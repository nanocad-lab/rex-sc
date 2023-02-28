from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

os.environ['CFLAGS'] = '-march=native -fopenmp'

setup(name='fixed_extension',
      ext_modules=[CppExtension('fixed_extension', ['fixed.cpp'])],
      cmdclass={'build_ext': BuildExtension})
