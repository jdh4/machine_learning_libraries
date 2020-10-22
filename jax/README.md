# JAX

## About

[JAX](https://github.com/google/jax) is [Autograd](https://github.com/hips/autograd) and [XLA](https://www.tensorflow.org/xla), brought
together for high-performance machine learning research.

## Installation

### CPU (Della, Perseus)

Here are the installation directions for the CPU-only clusters:

```
$ module load anaconda3
$ conda create --name jax-cpu --channel conda-forge --override-channels jax "libblas=*=*mkl"
```

### GPU (TigerGPU, Traverse, Adroit)

JAX must be built from source to use on the GPU clusters as [described here](https://jax.readthedocs.io/en/latest/developer.html). Below is the build procedure for TigerGPU (for Traverse and Adroit see notes below):

$ ssh <YourNetID>@tigergpu.princeton.edu
$ cd software  # or another directory
$ wget https://github.com/google/jax/archive/jaxlib-v0.1.55.tar.gz  # check for a newer release
$ tar zxf jaxlib-v0.1.55.tar.gz
$ cd jax-jaxlib-v0.1.55
$ cd jax
$ module load anaconda3/2019.10 cudatoolkit/10.1 cudnn/cuda-10.1/7.6.3 rh/devtoolset/8
$ conda create --name jax-gpu python=3.7 numpy scipy cython six
$ conda activate jax-gpu
$ python build/build.py --enable_cuda --cudnn_path /usr/local/cudnn/cuda-10.1/7.6.3 --enable_march_native --enable_mkl_dnn
$ pip install -e build
$ pip install -e .
```

For Traverse and Adroit, use `--cuda_compute_capabilities 7.0` instead of 6.0.

If you do a pip install on TigerGPU then you will encounter the following error when you try to import jax:

```
ImportError: /lib64/libm.so.6: version `GLIBC_2.23' not found
```

Follow the directions above to build from source.

## Example Job

```
#SBATCH --gres=gpu:1
module load cudatoolkit/10.1 cudnn/cuda-10.1/7.6.3 anaconda3/2019.10
conda activate jax-gpu
```
