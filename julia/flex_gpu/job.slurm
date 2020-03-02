#!/bin/bash
#SBATCH --job-name=flux-gpu      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:tesla_v100:1  # number of gpus per node

module purge
module load julia/1.2.0 cudatoolkit/10.2 cudnn/cuda-10.1/7.6.3

julia ../60-minute-blitz.jl
