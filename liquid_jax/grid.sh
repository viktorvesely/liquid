#!/bin/bash
#SBATCH --mem=32GB
#SBATCH --time=8:00:00
#SBATCH --job-name=grid
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=v100:1
#SBATCH --nodes=1

module purge
module load CUDA/13.2.0

source .venv/bin/activate
cd liquid_jax
srun python grid.py