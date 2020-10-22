#!/bin/bash
module load anaconda3/2020.7
pip install --user virtualenv
mkdir jax-gpu
virtualenv jax-gpu
source jax-gpu/bin/activate
pip install numpy scipy cython six

git clone https://github.com/google/jax
cd jax
module load cudatoolkit/11.0 cudnn/cuda-11.0/8.0.2 rh/devtoolset/8
python build/build.py --enable_cuda --cuda_path /usr/local/cuda-11.0 \
                      --cudnn_path /usr/local/cudnn/cuda-11.0/8.0.2 \
                      --cuda_compute_capabilities 6.0 \
                      --enable_march_native --enable_mkl_dnn
pip install -e build
pip install -e .

deactivate
exit
