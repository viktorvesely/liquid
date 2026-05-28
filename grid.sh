#!/bin/bash
#SBATCH --mem=40GB
#SBATCH --time=6:00:00
#SBATCH --job-name=grid
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --nodes=1
#SBATCH --output=grid.out
#SBATCH --error=grid.err

module purge
module load CUDA/13.2.0

source .venv/bin/activate
cd liquid_jax
srun python grid.py