# Julia and Machine Learning

According to the Julia website:

> [Julia] is a flexible dynamic language, appropriate for scientific and numerical computing, with performance comparable to traditional statically-typed languages. Once you understand how Julia works, it's easy to write code that's nearly as fast as C. Julia features optional typing, multiple dispatch, and good performance, achieved using type inference and just-in-time (JIT) compilation, implemented using LLVM. It is multi-paradigm, combining features of imperative, functional, and object-oriented programming.

## Installing flux on TigerGPU

```bash
$ ssh tigergpu
$ module load julia/1.2.0 cudatoolkit/10.2 cudnn/cuda-10.2/7.6.5
$ julia
$ ]
$ activate flux-env
$ add CuArrays Flux DifferentialEquations DiffEqFlux
$ # press the backspace or delete key
$ exit()
```

Note that this will place your Julia environment in your home directory. You will need to point to this when activating it in scripts. For instance, if your Julia script is in /home/NetID/myjob then do: activate "../flux-env". 

Below is a sample Julia script (myscript.jl):

```bash
using Pkg
Pkg.activate("../flux-env")
Pkg.instantiate()

using Flux, CuArrays

z = CuArrays.cu([1, 2, 3])
println(2 * z)

m = Dense(10,5) |> gpu
x = rand(10) |> gpu
println(m(x))
```

Below is an appropriate Slurm script (job.slurm):

```bash
#!/bin/bash
#SBATCH --job-name=myjob         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=1G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)

module purge
module load julia/1.2.0 cudatoolkit/10.2 cudnn/cuda-10.2/7.6.5

julia myscript.jl
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

## More Info

[Julia Documentation](https://docs.julialang.org/en/v1/)
