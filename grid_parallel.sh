#!/bin/bash

TIME_LIMIT=${1:-"6:00:00"}
shift
TASKS=("$@")

if [ ${#TASKS[@]} -eq 0 ]; then
    echo "Error: No tasks specified."
    exit 1
fi

for TASK in "${TASKS[@]}"; do
    sbatch --time="$TIME_LIMIT" --job-name="grid_${TASK}" --output="grid_${TASK}.out" --error="grid_${TASK}.err" <<EOD
#!/bin/bash
#SBATCH --mem=20GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --nodes=1

module purge
module load CUDA/13.2.0

source .venv/bin/activate
cd liquid_jax
srun python grid.py --tasks $TASK
EOD
done