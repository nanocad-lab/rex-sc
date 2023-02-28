rm -rf build
rm -rf dist
rm -rf fixed_extension_cuda.egg-info
rm -rf fixed_extension.egg-info
python setup_fixed.py install
python setup_cuda_fixed.py install
