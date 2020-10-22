# JAX

<img src="https://raw.githubusercontent.com/google/jax/master/images/jax_logo_250px.png" alt="logo"></img>

[JAX](https://github.com/google/jax) is [Autograd](https://github.com/hips/autograd) and [XLA](https://www.tensorflow.org/xla), brought
together for high-performance machine learning research.

## Installation

### CPU-Only Version (Della, Perseus)

Here are the installation directions for the CPU-only clusters:

```
$ module load anaconda3
$ conda create --name jax-cpu --channel conda-forge --override-channels jax "libblas=*=*mkl"
```

See [this page](https://researchcomputing.princeton.edu/python) for Slurm scripts. Be sure to take advantage of the parallelism of the CPU version which uses MKL and OpenMP. For the MNIST example, one finds as `cpus-per-task` increases from 1, 2, 4, the run time decreases as 139 s, 87 s, 58 s.

### GPU Version (TigerGPU, Traverse, Adroit)

JAX must be built from source to use on the GPU clusters as [described here](https://jax.readthedocs.io/en/latest/developer.html). Below is the build procedure for TigerGPU (for Traverse and Adroit see notes below):

```
$ ssh <YourNetID>@tigergpu.princeton.edu
$ cd software  # or another directory
$ wget
$ bash install_jax_tigergpu.sh | tee jax.log
```

For Traverse and Adroit, use `--cuda_compute_capabilities 7.0` instead of 6.0. You also may need to use different modules. On Adroit use `#SBATCH --gres=gpu:tesla_v100:1`.

If you do a pip install instead of building from source on TigerGPU then you will encounter the following error when you try to import jax:

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
