# NVIDIA Rapids

"Pandas and Scikit-Learn" on GPUs.

[Getting started](https://rapids.ai/start.html)

## Install to /scratch/network

You can request more space in `/home/<YourNetID>` for installing software but because Conda environment are easy to remake we will do the installation on `/scratch/network/<YourNetID>`. Note that this filesystem is not backed up but it provides a vast amount of space (100 GB by default).

```bash
$ cd
$ vim .condarc
```

Edit your file as follows:

```
pkgs_dirs:
 - /scratch/network/<YourNetID>/conda/pkgs
```

Check that it is correct with:

```bash
$ module load anaconda3
$ conda config --show
```

Now perform the installation by copying and pasting the following two lines:

```
conda create --prefix /scratch/network/$USER/rapids-env -c rapidsai -c nvidia -c conda-forge \
-c defaults cuml=0.12 python=3.7 cudatoolkit=10.0
```

After the installation one can recover space by deleting the index cache, lock files, unused cache packages, and tarballs:

```
$ conda clean -a
```

## Using Rapids

Note that Rapids requires a GPU with compute capability of 6.0+. This means the K40c GPUs on adroit-h11g4 cannot be used (they are CC 3.5).

```bash
$ salloc -N 1 -n 1 -t 5 --gres=gpu:tesla_v100:1
$ module load anaconda3
$ conda activate /scratch/network/$USER/rapids-env
$ python
>>> import cudf
>>> s = cudf.Series([1, 2, 3, None, 4])
>>> s
0       1
1       2
2       3
3    null
4       4
dtype: int64
>>> exit()
$ exit
```
