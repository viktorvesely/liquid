#!/bin/bash

TIME_LIMIT=${1:-"6:00:00"}
shift
TASKS=("$@")

if [ ${#TASKS[@]} -eq 0 ]; then
    echo "Error: No tasks specified."
    exit 1
fi

sbatch --time="$TIME_LIMIT" <<EOD
#!/bin/bash
#SBATCH --mem=20GB
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
srun python grid.py --tasks ${TASKS[@]}
EOD