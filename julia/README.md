# Julia and Machine Learning

## About Julia

According to the [Julia](https://docs.julialang.org/en/v1/) website:

> [Julia] is a flexible dynamic language, appropriate for scientific and numerical computing, with performance comparable to traditional statically-typed languages. Once you understand how Julia works, it's easy to write code that's nearly as fast as C. Julia features optional typing, multiple dispatch, and good performance, achieved using type inference and just-in-time (JIT) compilation, implemented using LLVM. It is multi-paradigm, combining features of imperative, functional, and object-oriented programming.

## Flux with CPUs

Here is a 60-minute [introduction](https://github.com/FluxML/model-zoo/blob/master/tutorials/60-minute-blitz.jl) to Flux.

```bash
$ ssh adroit
$ module load julia/1.2.0
$ julia
julia> ]
(v1.2) pkg> add Flux, Zygote, Metalhead, Images
# press the backspace or delete key
julia> exit()
```

Now run the script:

```bash
$ cd intro_machine_learning_libs/julia
$ wget https://raw.githubusercontent.com/FluxML/model-zoo/master/tutorials/60-minute-blitz.jl
$ cd cpu
$ sbatch job.slurm  # this will take about 50 minutes to run
```

Here is the output:

```
WARNING: using Images.data in module Main conflicts with an existing identifier.
[ Info: CUDAdrv.jl failed to initialize, GPU functionality unavailable (set JULIA_CUDA_SILENT or JULIA_CUDA_VERBOSE to silence or expand this message)
AbstractFloat[0.0f0 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0; -0.023309880974436 -0.026529296389295614 -0.05422134914540073 -0.06377514815120719 -0.05472705065848379 -0.010623560387355268 -0.09121674023495918 -0.04655826905684877 -0.07661404230926636 -0.04440761868867722; 0.017096034409426576 0.019457232082252102 0.03976725800238591 0.04677424687217634 0.04013815180823797 0.0077915779206264805 0.06690057883533744 0.03414696843420867 0.056190604533799975 0.03256962907596699; 0.011762327563885759 0.013386866904748617 0.027360468734416256 0.03218138195611239 0.02761564922431061 0.0053607222322514326 0.0460285995938865 0.02349362538808211 0.038659976969564844 0.02240839230029551; 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0]
AbstractFloat[0.0f0, -0.09390856835412051, 0.06887483113632147, 0.04738691472719748, 0.0f0]
[-0.0 -0.054558614835297004 -0.10010597347806593 -0.04262705892595774 -0.0; 0.0 0.054558631459904014 0.10010600398145156 0.04262707191488909 0.0]
[-0.09780522225474342, 0.09780525205706497]
accuracy(valX, valY) = 0.14
accuracy(valX, valY) = 0.296
accuracy(valX, valY) = 0.31
accuracy(valX, valY) = 0.385
accuracy(valX, valY) = 0.397
accuracy(valX, valY) = 0.447
accuracy(valX, valY) = 0.482
accuracy(valX, valY) = 0.483
accuracy(valX, valY) = 0.486
accuracy(valX, valY) = 0.501
```

The script took 50 minutes to run and required 2.4 GB of memory.

## Flux with GPUs

First we need to add the GPU packages:

```bash
$ module load julia/1.2.0 module load cudatoolkit/10.1 cudnn/cuda-10.1/7.6.3
$ julia
julia> ]
(v1.2) pkg> add CuArrays
$ # press the backspace or delete key
julia> exit()
```

```bash
$ cd intro_machine_learning_libs/julia/gpu
$ sbatch job.slurm  # this will take about 50 minutes to run
```

Below is an appropriate Slurm script (`job.slurm`):

```bash
#!/bin/bash
#SBATCH --job-name=flux-gpu      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:tesla_v100:1  # number of gpus per node
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)

module purge
module load julia/1.2.0 cudatoolkit/10.1 cudnn/cuda-10.1/7.6.3

julia ../60-minute-blitz.jl
```

Submit the job with: sbatch job.slurm

Here's the output on Adroit:

```
┌ Warning: Some registries failed to update:
│     — /home/jdh4/.julia/registries/General — failed to fetch from repo
└ @ Pkg.Types /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.2/Pkg/src/Types.jl:1171
[ Info: Building the CUDAnative run-time library for your sm_35 device, this might take a while...
Activating environment at `~/flux-env/Project.toml`
  Updating registry at `~/.julia/registries/General`
  Updating git-repo `https://github.com/JuliaRegistries/General.git`
Float32[2.0, 4.0, 6.0]
Float32[-0.835879, 0.3685953, 1.0108142, -0.29181987, 0.31272212]
```

Note that there are no GPUs on the head node of TigerGPU and no internet connection on the compute nodes. To run MNIST for example you will need to download the data first.

## Knet


## TensorFlow

There is a Julia package called [TensorFlow.jl](https://github.com/malmaud/TensorFlow.jl) that provides an interface to TensorFlow. It can be used with up to version 1.12 of TensorFlow. It appears that the number of commits is decreasing with time on this repo.

## More Info

[Julia Documentation](https://docs.julialang.org/en/v1/)  
[Flux website](https://fluxml.ai/)  
[Flux documentation](https://fluxml.ai/Flux.jl/stable/)  
[Flux on GitHub](https://github.com/FluxML/Flux.jl)  
[Knet documentation](https://denizyuret.github.io/Knet.jl/latest/)  
[Knet on GitHub](https://github.com/denizyuret/Knet.jl)  

