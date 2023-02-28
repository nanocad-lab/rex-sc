#!/bin/bash

# Create conda environment
ENV_NAME=pytorch_sc
CONDA_BIN_DIR=$(which conda)
CONDA_PARENT_DIR=${CONDA_BIN_DIR::-9}
CONDA_SH_DIR="$CONDA_PARENT_DIR/etc/profile.d/conda.sh"
source $CONDA_SH_DIR
conda create -n $ENV_NAME -y python=3.7
conda activate $ENV_NAME
# Install dependencies
conda install -y pandas pillow bokeh matplotlib scipy h5py absl-py boto3
# Install pytorch
conda install -y pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cudatoolkit=11.3 -c pytorch
# Install ninja. Helps with compilation speed of the kernels
conda install -y -c conda-forge ninja
pip install nvidia-ml-py3 gdown
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

#Compile binaries
CUR_DIR=$PWD
cd torch_c/sc-extension
./compile_all.sh
./compile_fixed.sh
cd $CUR_DIR