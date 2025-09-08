#!/bin/bash
#SBATCH --mem=18GB
#SBATCH --time=42:00:00
#SBATCH --job-name=rfhyper
#SBATCH --partition=regular
#SBATCH --nodes=1

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

source .venv/bin/activate
srun python -m liquid.hyper --algorithm rf