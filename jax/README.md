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

JAX must be built from source to use on the GPU clusters:

```
$ ssh <YourNetID>@tigergpu.princeton.edu  # or adroit
$ cd software  # or another directory
$ git clone https://github.com/google/jax
$ cd jax
$ module load anaconda3/2019.10 cudatoolkit/10.1 cudnn/cuda-10.1/7.6.3 rh/devtoolset/8
$ conda create --name jax-gpu python=3.7 numpy scipy cython six
$ conda activate jax-gpu
$ python build/build.py --enable_cuda --cudnn_path /usr/local/cudnn/cuda-10.1/7.6.3 --enable_march_native --enable_mkl_dnn
$ pip install -e build
$ pip install -e .
```

## Example Job

```
#SBATCH --gres=gpu:1
module load cudatoolkit/10.1 cudnn/cuda-10.1/7.6.3 anaconda3/2019.10
conda activate jax-gpu
```
