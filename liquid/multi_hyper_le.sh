#!/bin/bash
#SBATCH --mem=32GB
#SBATCH --time=8:00:00
#SBATCH --job-name=hyperle
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --cpus-per-task=15

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

source .venv/bin/activate
srun python -m liquid.hyper --algorithm le --cpu 10