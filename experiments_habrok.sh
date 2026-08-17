#!/bin/bash

set -e

TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
MEM="${MEM:-20GB}"
CPUS="${CPUS:-1}"
GPUS="${GPUS:-1}"
PARTITION="${PARTITION:-gpu}"
CUDA_MODULE="${CUDA_MODULE:-CUDA/13.2.0}"
DRY_RUN=0

# Keep in sync with the argparse choices in liquid_jax/experiment.py
VALID_EXPERIMENTS=(agg gradient scaling)
VALID_TASKS=(Cifar10 Svhn Bikes Energy)


usage() {
    cat <<'USAGE'
Usage: experiments_habrok.sh [options] <experiment_name> <task1> [task2 ...]

  experiment_name : agg | gradient | scaling
  task            : Cifar10 | Svhn | Bikes | Energy

Options:
  -t, --time       Wall clock limit per job      (default 12:00:00)
  -m, --mem        Host RAM per job              (default 20GB)
  -c, --cpus       CPU cores per job             (default 1)
  -g, --gpus       GPUs per job, any type        (default 1)
  -p, --partition  Slurm partition               (default gpu)
      --dry-run    Print the sbatch scripts instead of submitting
  -h, --help       Show this message
USAGE
}


while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -t|--time)      TIME_LIMIT="$2"; shift 2 ;;
        -m|--mem)       MEM="$2";        shift 2 ;;
        -c|--cpus)      CPUS="$2";       shift 2 ;;
        -g|--gpus)      GPUS="$2";       shift 2 ;;
        -p|--partition) PARTITION="$2";  shift 2 ;;
        --dry-run)      DRY_RUN=1;       shift   ;;
        -h|--help)      usage; exit 0 ;;
        --) shift; break ;;
        -*) echo "Unknown option: $1"; echo; usage; exit 1 ;;
        *)  break ;;
    esac
done

if [ "$#" -lt 2 ]; then
    usage
    exit 1
fi

EXPERIMENT_NAME="$1"
shift

TASKS=("$@")
NUM_TASKS=${#TASKS[@]}


contains() {
    local needle="$1"
    shift

    local item
    for item in "$@"; do
        [[ "$item" == "$needle" ]] && return 0
    done

    return 1
}

if ! contains "$EXPERIMENT_NAME" "${VALID_EXPERIMENTS[@]}"; then
    echo "Error: unknown experiment '$EXPERIMENT_NAME'."
    echo "Valid  : ${VALID_EXPERIMENTS[*]}"
    exit 1
fi

for TASK in "${TASKS[@]}"; do
    if ! contains "$TASK" "${VALID_TASKS[@]}"; then
        echo "Error: unknown task '$TASK'."
        echo "Valid : ${VALID_TASKS[*]}"
        exit 1
    fi
done


REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="${REPO_ROOT}/logs"

mkdir -p "$LOG_ROOT"


echo
echo "Experiment  : $EXPERIMENT_NAME"
echo "Tasks       : ${TASKS[*]} ($NUM_TASKS job(s))"
echo "Per job     : ${GPUS} GPU(s) any type, ${CPUS} cpu(s), ${MEM} ram, ${TIME_LIMIT} wall"
echo "Partition   : $PARTITION"
echo "Repo        : $REPO_ROOT"

for TASK in "${TASKS[@]}"; do

    JOB_NAME="${EXPERIMENT_NAME}_${TASK}"

    SESSION_LOG_PATH="${LOG_ROOT}/${JOB_NAME}"
    STDOUT_FILE="${SESSION_LOG_PATH}/log.out"
    STDERR_FILE="${SESSION_LOG_PATH}/log.err"

    mkdir -p "$SESSION_LOG_PATH"

    # Unquoted delimiter bakes in the loop variables; runtime ones are escaped
    JOB_SCRIPT=$(cat <<EOD
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=${GPUS}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=1
#SBATCH --output=${STDOUT_FILE}
#SBATCH --error=${STDERR_FILE}

set -e

echo "Job \${SLURM_JOB_ID} started on \$(hostname) at \$(date)"
nvidia-smi -L || true

module purge
module load ${CUDA_MODULE}

cd "${REPO_ROOT}"
source .venv/bin/activate

cd liquid_jax
srun python -u experiment.py '${EXPERIMENT_NAME}' '${TASK}'

echo "Job \${SLURM_JOB_ID} finished at \$(date)"
EOD
)

    if [ "$DRY_RUN" -eq 1 ]; then
        echo
        echo "would submit: ${JOB_NAME}"
        echo "$JOB_SCRIPT"
        continue
    fi

    JOB_ID=$(sbatch --parsable <<< "$JOB_SCRIPT")

    echo
    echo "Task       : $TASK"
    echo "Experiment : $EXPERIMENT_NAME"
    echo "Job        : $JOB_NAME (id $JOB_ID)"
    echo "stdout     : $STDOUT_FILE"
    echo "stderr     : $STDERR_FILE"

done

if [ "$DRY_RUN" -eq 0 ]; then
    echo
    echo "Follow with : squeue -u \$USER"
    echo "Tail a log  : tail -f ${LOG_ROOT}/${EXPERIMENT_NAME}_<task>/log.out"
fi
