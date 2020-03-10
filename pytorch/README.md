# PyTorch

## An Aside on Installing Python Packages

Anaconda maintains roughly 700 popular packages for scientfic computing and data science. Try building your environment from Anaconda Cloud:

A good starting point for installing scientific software is try :

```
$ module load anaconda3
$ conda create --name myenv <package-1> <package-2> ... <package-N>
$ conda activate torch-env
```

In the above case, the implied channel is `anaconda`. If one or more packages are not found then try also looking in the community channel `conda-forge`:

```
$ module load anaconda3
$ conda create --name myenv <package-1> <package-2> ... <package-N> --channel conda-forge
$ conda activate torch-env
```

If still no luck then try adding a specialized channel that you have knowledge of:

```
$ module load anaconda3
$ conda create --name myenv <package-1> <package-2> ... <package-N> --channel conda-forge --channel <channel>
$ conda activate torch-env
```

In the above, `<channel>` corresponds to a special channel like `bioconda`, `r`, `intel`, `pytorch`, etc. Lastly, if packages are not found via conda then turn to pip:

```
$ module load anaconda3
$ conda create --name myenv <package-1> <package-2> ... <package-N> --channel conda-forge --channel <channel>
$ conda activate torch-env
$ pip install <package-a> <package-b> ... <package-M>
```

You [should not](https://www.anaconda.com/using-pip-in-a-conda-environment/) do additional conda installs after the pip install. In your script, you should import your conda-installed modules before the pip-installed ones (since you may get a mixing of libstdc++ versions otherwise).

When in doubt make a new environment.

Don't be afraid to delete all your environments and start over.

## An Aside of Where to Store Your Conda Packages and Environments

You can request more space in `/home/<YourNetID>` for installing software but because Conda environment are easy to remake we will do the installation on `/scratch/network/<YourNetID>`. Note that this filesystem is not backed up but it provides a vast amount of space (100 GB by default).

```bash
$ ssh adroit
$ cd
$ vim .condarc
```

Edit your file as follows:

```
pkgs_dirs:
 - /scratch/network/<YourNetID>/conda-pkgs
envs_dirs:
 - /scratch/network/<YourNetID>/conda-envs
```

Check that it is correct with:

```bash
$ module load anaconda3
$ conda config --show
```





```
$ module load anaconda3
$ conda create --name torch-env pytorch torchvision cudatoolkit=10.1 matplotlib --channel pytorch
$ conda activate torch-env
```

Please see [https://github.com/PrincetonUniversity/install_pytorch](https://github.com/PrincetonUniversity/install_pytorch).

```
$ mla
$ conda create --name torch-env pytorch torchvision cudatoolkit=10.1 matplotlib --channel pytorch
 _libgcc_mutex      pkgs/main/linux-64::_libgcc_mutex-0.1-main
  blas               pkgs/main/linux-64::blas-1.0-mkl
  ca-certificates    pkgs/main/linux-64::ca-certificates-2020.1.1-0
  certifi            pkgs/main/linux-64::certifi-2019.11.28-py37_0
  cudatoolkit        pkgs/main/linux-64::cudatoolkit-10.1.243-h6bb024c_0
  cycler             pkgs/main/linux-64::cycler-0.10.0-py37_0
  dbus               pkgs/main/linux-64::dbus-1.13.12-h746ee38_0
  expat              pkgs/main/linux-64::expat-2.2.6-he6710b0_0
  fontconfig         pkgs/main/linux-64::fontconfig-2.13.0-h9420a91_0
  freetype           pkgs/main/linux-64::freetype-2.9.1-h8a8886c_1
  glib               pkgs/main/linux-64::glib-2.63.1-h5a9c865_0
  gst-plugins-base   pkgs/main/linux-64::gst-plugins-base-1.14.0-hbbd80ab_1
  gstreamer          pkgs/main/linux-64::gstreamer-1.14.0-hb453b48_1
  icu                pkgs/main/linux-64::icu-58.2-h9c2bf20_1
  intel-openmp       pkgs/main/linux-64::intel-openmp-2020.0-166
  jpeg               pkgs/main/linux-64::jpeg-9b-h024ee3a_2
  kiwisolver         pkgs/main/linux-64::kiwisolver-1.1.0-py37he6710b0_0
  ld_impl_linux-64   pkgs/main/linux-64::ld_impl_linux-64-2.33.1-h53a641e_7
  libedit            pkgs/main/linux-64::libedit-3.1.20181209-hc058e9b_0
  libffi             pkgs/main/linux-64::libffi-3.2.1-hd88cf55_4
  libgcc-ng          pkgs/main/linux-64::libgcc-ng-9.1.0-hdf63c60_0
  libgfortran-ng     pkgs/main/linux-64::libgfortran-ng-7.3.0-hdf63c60_0
  libpng             pkgs/main/linux-64::libpng-1.6.37-hbc83047_0
  libstdcxx-ng       pkgs/main/linux-64::libstdcxx-ng-9.1.0-hdf63c60_0
  libtiff            pkgs/main/linux-64::libtiff-4.1.0-h2733197_0
  libuuid            pkgs/main/linux-64::libuuid-1.0.3-h1bed415_2
  libxcb             pkgs/main/linux-64::libxcb-1.13-h1bed415_1
  libxml2            pkgs/main/linux-64::libxml2-2.9.9-hea5a465_1
  matplotlib         pkgs/main/linux-64::matplotlib-3.1.3-py37_0
  matplotlib-base    pkgs/main/linux-64::matplotlib-base-3.1.3-py37hef1b27d_0
  mkl                pkgs/main/linux-64::mkl-2020.0-166
  mkl-service        pkgs/main/linux-64::mkl-service-2.3.0-py37he904b0f_0
  mkl_fft            pkgs/main/linux-64::mkl_fft-1.0.15-py37ha843d7b_0
  mkl_random         pkgs/main/linux-64::mkl_random-1.1.0-py37hd6b4f25_0
  ncurses            pkgs/main/linux-64::ncurses-6.2-he6710b0_0
  ninja              pkgs/main/linux-64::ninja-1.9.0-py37hfd86e86_0
  numpy              pkgs/main/linux-64::numpy-1.18.1-py37h4f9e942_0
  numpy-base         pkgs/main/linux-64::numpy-base-1.18.1-py37hde5b4d6_1
  olefile            pkgs/main/linux-64::olefile-0.46-py37_0
  openssl            pkgs/main/linux-64::openssl-1.1.1d-h7b6447c_4
  pcre               pkgs/main/linux-64::pcre-8.43-he6710b0_0
  pillow             pkgs/main/linux-64::pillow-7.0.0-py37hb39fc2d_0
  pip                pkgs/main/linux-64::pip-20.0.2-py37_1
  pyparsing          pkgs/main/noarch::pyparsing-2.4.6-py_0
  pyqt               pkgs/main/linux-64::pyqt-5.9.2-py37h05f1152_2
  python             pkgs/main/linux-64::python-3.7.6-h0371630_2
  python-dateutil    pkgs/main/noarch::python-dateutil-2.8.1-py_0
  pytorch            pytorch/linux-6::pytorch-1.4.0-py3.7_cuda10.1.243_cudnn7.6.3_0
  qt                 pkgs/main/linux-64::qt-5.9.7-h5867ecd_1
  readline           pkgs/main/linux-64::readline-7.0-h7b6447c_5
  setuptools         pkgs/main/linux-64::setuptools-45.2.0-py37_0
  sip                pkgs/main/linux-64::sip-4.19.8-py37hf484d3e_0
  six                pkgs/main/linux-64::six-1.14.0-py37_0
  sqlite             pkgs/main/linux-64::sqlite-3.31.1-h7b6447c_0
  tk                 pkgs/main/linux-64::tk-8.6.8-hbc83047_0
  torchvision        pytorch/linux-64::torchvision-0.5.0-py37_cu101
  tornado            pkgs/main/linux-64::tornado-6.0.3-py37h7b6447c_3
  wheel              pkgs/main/linux-64::wheel-0.34.2-py37_0
  xz                 pkgs/main/linux-64::xz-5.2.4-h14c3975_4
  zlib               pkgs/main/linux-64::zlib-1.2.11-h7b6447c_3
  zstd               pkgs/main/linux-64::zstd-1.3.7-h0b5b093_0

```
