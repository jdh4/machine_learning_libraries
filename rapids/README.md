# NVIDIA Rapids

Built based on the Apache Arrow columnar memory format, cuDF is a GPU DataFrame library for loading, joining, aggregating, filtering, and otherwise manipulating data.

cuDF provides a pandas-like API that will be familiar to data engineers & data scientists, so they can use it to easily accelerate their workflows without going into the details of CUDA programming.

cuML is a suite of libraries that implement machine learning algorithms and mathematical primitives functions that share compatible APIs with other RAPIDS projects.

cuML enables data scientists, researchers, and software engineers to run traditional tabular ML tasks on GPUs without going into the details of CUDA programming. In most cases, cuML's Python API matches the API from scikit-learn.

For large datasets, these GPU-based implementations can complete 10-50x faster than their CPU equivalents.

<p align="center"><img src="https://github.com/rapidsai/cudf/blob/branch-0.13/img/rapids_arrow.png" width="80%"/></p>

[Getting started](https://rapids.ai/start.html)

## Installation

Install `cuml` and its dependencies `cudf` and `dask-cudf`:

```bash
# ~/.condarc should directing the install to /scratch/network or /scratch/gpfs
$ module load anaconda3
$ conda create -n rapids-0.16 -c rapidsai -c nvidia -c conda-forge -c defaults cuml=0.16 python=3.8 cudatoolkit=10.2
```

Or install all components of Rapids:

```bash
# ~/.condarc should directing the install to /scratch/network or /scratch/gpfs
$ module load anaconda3
$ conda create -n rapids-0.16 -c rapidsai -c nvidia -c conda-forge -c defaults rapids=0.16 python=3.8 cudatoolkit=10.2
```

## Using cuDF

Note that Rapids requires a GPU with compute capability (CC) of 6.0 and greater. This means the K40c GPUs on `adroit-h11g4` cannot be used (they are CC 3.5). On Adroit we mut request a V100 GPU (CC 7.0). TigerGPU is CC 6.0.

Below is a simple interactive session on Adroit checking the installation:

```bash
$ salloc -N 1 -n 1 -t 5 --gres=gpu:tesla_v100:1
$ module load anaconda3
$ conda activate rapids-0.16
$ python
>>> import cudf
>>> s = cudf.Series([1, 2, 3, None, 4])
>>> s
0       1
1       2
2       3
3    <NA>
4       4
dtype: int64
>>> exit()
$ exit
```

See this [guide](https://rapidsai.github.io/projects/cudf/en/0.12.0/10min.html) for a 10-minute introduction to cuDF and Dask-cuDF.

Submitting a job to the Slurm scheduler:

```bash
#!/bin/bash
#SBATCH --job-name=rapids        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # total memory per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:tesla_v100:1

module purge
module load anaconda3
conda activate rapids-0.16

python rapids.py
```

```python
import cudf
s = cudf.Series([1, 2, 3, None, 4])
print(s)
```

The output is

```
0       1
1       2
2       3
3    null
4       4
dtype: int64
```

## cuDF with Multiple GPUs

```bash
#!/bin/bash
#SBATCH --job-name=rapids        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # total memory per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:tesla_v100:2

module purge
module load anaconda3
conda activate rapids-0.16

python rapids.py
```

```python
import cudf
import dask_cudf

df = cudf.DataFrame({'a':list(range(20, 40)), 'b':list(range(20))})

ddf = dask_cudf.from_cudf(df, npartitions=2)
print(ddf.compute())
```

```
     a   b
0   20   0
1   21   1
2   22   2
3   23   3
4   24   4
5   25   5
6   26   6
7   27   7
8   28   8
9   29   9
10  30  10
11  31  11
12  32  12
13  33  13
14  34  14
15  35  15
16  36  16
17  37  17
18  38  18
19  39  19
```

## Machine Learning Example

```bash
$ conda activate /scratch/network/$USER/rapids-env
$ conda install scikit-learn ipywidgets
$ wget https://raw.githubusercontent.com/rapidsai/notebooks/branch-0.13/cuml/svm_demo.ipynb
$ salloc -N 1 -n 1 -t 5 --gres=gpu:tesla_v100:1
```

This [example](https://github.com/rapidsai/notebooks/blob/branch-0.13/cuml/svm_demo.ipynb) can be ran as follows:

```bash
$ ipython svm_demo.py 
CPU times: user 1.99 s, sys: 786 ms, total: 2.78 s
Wall time: 3.55 s
CPU times: user 48.1 s, sys: 258 ms, total: 48.4 s
Wall time: 48.5 s
CPU times: user 196 ms, sys: 27 ms, total: 223 ms
Wall time: 230 ms
CPU times: user 8.91 s, sys: 0 ns, total: 8.91 s
Wall time: 8.93 s
Accuracy: cumlSVC 96.46000000000001%, sklSVC 96.46000000000001%
CPU times: user 851 ms, sys: 68.8 ms, total: 920 ms
Wall time: 942 ms
CPU times: user 477 ms, sys: 318 ms, total: 796 ms
Wall time: 798 ms
CPU times: user 13 s, sys: 70.8 ms, total: 13.1 s
Wall time: 13.1 s
CPU times: user 6.71 ms, sys: 991 µs, total: 7.7 ms
Wall time: 7.79 ms
CPU times: user 2.72 s, sys: 0 ns, total: 2.72 s
Wall time: 2.72 s
R2 score: cumlSVR 0.940174859165765, sklSVR 0.9401745984283644
CPU times: user 198 ms, sys: 105 ms, total: 303 ms
Wall time: 304 ms
```

## Traverse

cuDF and cuML are available on the IBM conda channel. You can make an environment like this:

```
$ ssh <YourNetID>@traverse.princeton.edu
$ module load anaconda3
$ conda create --name rapids-env --channel https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda cudf cuml
# accept the license agreement
```

There are also dask-based packages available like dask-cudf.

## Useful Links

[Example Notebooks](https://github.com/rapidsai/notebooks)  
[NVIDIA Rapids on GitHub](https://github.com/rapidsai)
