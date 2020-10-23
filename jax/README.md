# JAX

<img src="https://raw.githubusercontent.com/google/jax/master/images/jax_logo_250px.png" alt="logo"></img>

[JAX](https://github.com/google/jax) is [Autograd](https://github.com/hips/autograd) and [XLA](https://www.tensorflow.org/xla), brought
together for high-performance machine learning research.

## Installation

### GPU Version (TigerGPU, Traverse, Adroit)

JAX must be built from source to use on the GPU clusters as [described here](https://jax.readthedocs.io/en/latest/developer.html). Below is the build procedure for TigerGPU (for Traverse and Adroit see notes below):

```
$ ssh <YourNetID>@tigergpu.princeton.edu
$ cd software  # or another directory
$ wget
$ bash install_jax_tigergpu.sh | tee jax.log
```

For Traverse and Adroit, use `--cuda_compute_capabilities 7.0` instead of 6.0. You also may need to use different modules. On Adroit use `#SBATCH --gres=gpu:tesla_v100:1`. On Traverse it may be necessary to use stable releases instead of the master branch on github.

If you do a pip install instead of building from source on TigerGPU then you will encounter the following error when you try to import jax:

```
ImportError: /lib64/libm.so.6: version `GLIBC_2.23' not found
```

Follow the directions above to build from source.

### CPU-Only Version (Della, Perseus)

Here are the installation directions for the CPU-only clusters:

```
$ module load anaconda3
$ conda create --name jax-cpu --channel conda-forge --override-channels jax "libblas=*=*mkl"
```

See [this page](https://researchcomputing.princeton.edu/python) for Slurm scripts. Be sure to take advantage of the parallelism of the CPU version which uses MKL and OpenMP. For the MNIST example, one finds as `cpus-per-task` increases from 1, 2, 4, the run time decreases as 139 s, 87 s, 58 s.

## Example Job for GPU Version

```
#!/bin/bash
#SBATCH --job-name=jax-gpu       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email when job begins, ends and fails
#SBATCH --mail-user=<YourNetID>@princeton.edu

module purge
module load anaconda3/2020.7 cudatoolkit/11.0 cudnn/cuda-11.0/8.0.2
source $HOME/software/jax-gpu/bin/activate

python mnist_classifier.py
```

The example below is from `https://github.com/google/jax/blob/master/examples/mnist_classifier.py`.

```
$ wget https://raw.githubusercontent.com/google/jax/master/examples/mnist_classifier.py
```

Next make the appropriate lines in the file `software/jax/examples/datasets.py` in the function `mnist_raw()` look like this:

```python
#for filename in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
#                   "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
#    _download(base_url + filename, filename)
 
  _DATA = os.getcwd() + "/data"
  train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
  train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
  test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
  test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))
```
