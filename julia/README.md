# Julia

## Installing flux on TigerGPU

```bash
$ ssh tigergpu
$ module load julia/1.2.0 cudatoolkit/10.2 cudnn/cuda-10.2/7.6.5
$ julia
$ ]
$ activate flux-env
$ add CuArrays
$ #backspace
$ exit()
$ salloc -t 5:00 --gres=gpu:1
$ julia
$ ]
$ activate flux-env
$ build CuArrays
$ build CUDAdrv
$ Backspace (or delete on Mac)
$ exit()

```
