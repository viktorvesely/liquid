#!/bin/bash

set -euo pipefail

# Consider a MIG unused if at least this percentage of its
# nvidia-smi-reported memory is free.
MIN_FREE_PERCENT=99

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage:
    $0 [--dry-run|-n] <experiment_name> <task[=resume_dir]> [task[=resume_dir] ...]

Examples:
    $0 my_experiment task1 task2 task3

    $0 my_experiment \
        task1=/path/to/task1/checkpoint \
        task2 \
        task3=/path/to/task3/checkpoint

    $0 --dry-run my_experiment \
        task1=/path/to/task1/checkpoint \
        task2

Options:
    -n, --dry-run
        Validate arguments, resume directories, tmux session names,
        and available MIG instances, and print the commands that would
        be launched without starting anything.

    -h, --help
        Show this help message.
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# 1. Arguments
# ---------------------------------------------------------------------------

DRY_RUN=0

while (( $# > 0 )); do
    case "$1" in
        -n|--dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            die "Unknown option: $1"
            ;;
        *)
            break
            ;;
    esac
done

if (( $# < 2 )); then
    usage
    exit 1
fi

EXPERIMENT_NAME="$1"
shift

TASK_ARGS=("$@")
NUM_TASKS=${#TASK_ARGS[@]}

if [[ -z "$EXPERIMENT_NAME" ]]; then
    die "Experiment name cannot be empty."
fi

# ---------------------------------------------------------------------------
# 2. Validate tasks and resume directories
#
# Parsed values are stored separately so we only have to parse once.
# Resume directories are converted to absolute paths because experiment.py
# is launched after `cd liquid_jax`.
# ---------------------------------------------------------------------------

declare -a TASKS
declare -a RESUME_DIRS

declare -A SEEN_TASKS

echo
echo "Validating tasks:"
echo "-------------------------------------"

for ((i = 0; i < NUM_TASKS; i++)); do
    TASK_ARG="${TASK_ARGS[$i]}"

    if [[ "$TASK_ARG" == *=* ]]; then
        TASK="${TASK_ARG%%=*}"
        RESUME_DIR="${TASK_ARG#*=}"

        if [[ -z "$TASK" ]]; then
            die "Invalid task argument '$TASK_ARG': task name is empty."
        fi

        if [[ -z "$RESUME_DIR" ]]; then
            die "Invalid task argument '$TASK_ARG': resume directory is empty."
        fi

        if [[ ! -e "$RESUME_DIR" ]]; then
            die "Resume path does not exist for task '$TASK': $RESUME_DIR"
        fi

        if [[ ! -d "$RESUME_DIR" ]]; then
            die "Resume path is not a directory for task '$TASK': $RESUME_DIR"
        fi

        if [[ ! -r "$RESUME_DIR" ]]; then
            die "Resume directory is not readable for task '$TASK': $RESUME_DIR"
        fi

        # Resolve before changing into liquid_jax later.
        RESUME_DIR="$(cd "$RESUME_DIR" && pwd -P)"
    else
        TASK="$TASK_ARG"
        RESUME_DIR=""

        if [[ -z "$TASK" ]]; then
            die "Task name cannot be empty."
        fi
    fi

    if [[ -n "${SEEN_TASKS[$TASK]:-}" ]]; then
        die "Task '$TASK' was specified more than once."
    fi
    SEEN_TASKS["$TASK"]=1

    TASKS+=("$TASK")
    RESUME_DIRS+=("$RESUME_DIR")

    if [[ -n "$RESUME_DIR" ]]; then
        echo "  $TASK"
        echo "    resume: $RESUME_DIR"
    else
        echo "  $TASK"
        echo "    resume: none"
    fi
done

# ---------------------------------------------------------------------------
# 3. Verify local environment
# ---------------------------------------------------------------------------

command -v nvidia-smi >/dev/null 2>&1 ||
    die "nvidia-smi was not found."

command -v tmux >/dev/null 2>&1 ||
    die "tmux was not found."

if [[ ! -d ".venv" ]]; then
    die "Virtual environment '.venv' does not exist."
fi

if [[ ! -f ".venv/bin/activate" ]]; then
    die "Virtual environment activation script '.venv/bin/activate' does not exist."
fi

if [[ ! -d "liquid_jax" ]]; then
    die "Directory 'liquid_jax' does not exist."
fi

if [[ ! -f "liquid_jax/experiment.py" ]]; then
    die "File 'liquid_jax/experiment.py' does not exist."
fi

# Store absolute paths before entering any tmux sessions.
ROOT_DIR="$(pwd -P)"
VENV_ACTIVATE="$ROOT_DIR/.venv/bin/activate"
LIQUID_JAX_DIR="$ROOT_DIR/liquid_jax"
LOG_ROOT="$ROOT_DIR/logs"

# ---------------------------------------------------------------------------
# 4. Check experiment tmux session names before doing anything
# ---------------------------------------------------------------------------

for TASK in "${TASKS[@]}"; do
    SESSION_NAME="${EXPERIMENT_NAME}_${TASK}"

    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        die "tmux session '$SESSION_NAME' already exists."
    fi
done

# ---------------------------------------------------------------------------
# 5. Kerberos revive session
#
# In dry-run mode we intentionally do not modify the existing session.
# ---------------------------------------------------------------------------

REVIVE_SESSION="kerberos_revive"

if (( DRY_RUN )); then
    echo
    echo "[DRY RUN] Kerberos revive session:"
    echo "  Would replace/start tmux session '$REVIVE_SESSION'"
    echo "  Command: while true; do kinit -R; sleep 8h; done"
else
    if tmux has-session -t "$REVIVE_SESSION" 2>/dev/null; then
        echo
        echo "Killing existing '$REVIVE_SESSION' session..."
        tmux kill-session -t "$REVIVE_SESSION"
    fi

    echo
    echo "Starting new '$REVIVE_SESSION' session..."

    tmux new-session -d -s "$REVIVE_SESSION" \
        "while true; do kinit -R; sleep 8h; done"
fi

# ---------------------------------------------------------------------------
# 6. Discover MIG names + UUIDs from `nvidia-smi -L`
# ---------------------------------------------------------------------------

declare -A MIG_NAMES
declare -A MIG_UUIDS

CURRENT_GPU=""

while IFS= read -r line; do
    if [[ "$line" =~ ^GPU\ ([0-9]+): ]]; then
        CURRENT_GPU="${BASH_REMATCH[1]}"
    fi

    # Example:
    #
    # MIG 1g.20gb Device 0: (UUID: MIG-f467...)
    #
    if [[ "$line" =~ MIG\ +(.+)\ +Device\ +([0-9]+):\ +\(UUID:\ +(MIG-[A-Za-z0-9-]+)\) ]]; then
        KEY="$CURRENT_GPU:${BASH_REMATCH[2]}"

        MIG_NAMES["$KEY"]="${BASH_REMATCH[1]}"
        MIG_UUIDS["$KEY"]="${BASH_REMATCH[3]}"
    fi
done < <(nvidia-smi -L)

if (( ${#MIG_UUIDS[@]} == 0 )); then
    die "No MIG instances were discovered from 'nvidia-smi -L'."
fi

# ---------------------------------------------------------------------------
# 7. Find unused MIG instances
#
# Candidate format:
#
#   TOTAL_MEMORY|GPU|MIG|NAME|FREE|TOTAL|UUID
#
# Sorting numerically by TOTAL_MEMORY means smaller MIGs are consumed first:
#
#   20 GB -> 40 GB -> 80 GB
# ---------------------------------------------------------------------------

declare -a CANDIDATES=()

while read -r GPU GI CI MIG USED_RAW SLASH TOTAL_RAW REST; do
    # Skip headers / BAR1 / anything that isn't a GPU row.
    if ! [[ "$GPU" =~ ^[0-9]+$ ]]; then
        continue
    fi

    USED="${USED_RAW%MiB}"
    TOTAL="${TOTAL_RAW%MiB}"

    # Ensure parsed values are numeric.
    if ! [[ "$USED" =~ ^[0-9]+$ && "$TOTAL" =~ ^[0-9]+$ ]]; then
        continue
    fi

    if (( TOTAL <= 0 )); then
        continue
    fi

    FREE=$((TOTAL - USED))

    KEY="$GPU:$MIG"
    NAME="${MIG_NAMES[$KEY]:-}"
    UUID="${MIG_UUIDS[$KEY]:-}"

    if [[ -z "$UUID" ]]; then
        continue
    fi

    # Avoid floating point arithmetic:
    #
    #     FREE / TOTAL >= MIN_FREE_PERCENT / 100
    #
    if (( FREE * 100 >= TOTAL * MIN_FREE_PERCENT )); then
        CANDIDATES+=(
            "$TOTAL|$GPU|$MIG|$NAME|$FREE|$TOTAL|$UUID"
        )
    fi
done < <(nvidia-smi | grep "MiB /" | tr -d '|')

# ---------------------------------------------------------------------------
# 8. Prioritize smaller MIGs
#
# In your setup this should be approximately:
#
#   19938 MiB -> 40103 MiB -> 81153 MiB
#
# i.e. 1g.20gb first.
# ---------------------------------------------------------------------------

if (( ${#CANDIDATES[@]} > 0 )); then
    mapfile -t CANDIDATES < <(
        printf '%s\n' "${CANDIDATES[@]}" |
            sort -t'|' -k1,1n
    )
fi

# ---------------------------------------------------------------------------
# 9. Make sure enough unused MIGs exist
# ---------------------------------------------------------------------------

NUM_AVAILABLE=${#CANDIDATES[@]}

if (( NUM_AVAILABLE < NUM_TASKS )); then
    echo
    echo "Error: Not enough unused MIG instances."
    echo "Requested tasks : $NUM_TASKS"
    echo "Available MIGs  : $NUM_AVAILABLE"
    echo "Free threshold  : ${MIN_FREE_PERCENT}%"
    echo
    exit 1
fi

# ---------------------------------------------------------------------------
# 10. Select only as many MIGs as needed
# ---------------------------------------------------------------------------

declare -a SELECTED_UUIDS
declare -a SELECTED_LABELS

echo
echo "Selecting $NUM_TASKS MIG instance(s):"
echo "-------------------------------------"

for ((i = 0; i < NUM_TASKS; i++)); do
    IFS='|' read -r TOTAL_SORT GPU MIG NAME FREE TOTAL UUID \
        <<< "${CANDIDATES[$i]}"

    SELECTED_UUIDS+=("$UUID")

    LABEL="$(
        printf \
            "GPU %s [MIG %s] %-10s : %5s / %5s MiB Free (%s)" \
            "$GPU" \
            "$MIG" \
            "$NAME" \
            "$FREE" \
            "$TOTAL" \
            "$UUID"
    )"

    SELECTED_LABELS+=("$LABEL")

    echo "  $LABEL"
done

# ---------------------------------------------------------------------------
# 11. Launch one tmux session per task
# ---------------------------------------------------------------------------

if (( DRY_RUN )); then
    echo
    echo "DRY RUN -- no sessions will be launched."
else
    mkdir -p "$LOG_ROOT"
fi

echo
echo "Experiments:"
echo "-------------------------------------"

for ((i = 0; i < NUM_TASKS; i++)); do
    TASK="${TASKS[$i]}"
    RESUME_DIR="${RESUME_DIRS[$i]}"
    UUID="${SELECTED_UUIDS[$i]}"

    SESSION_NAME="${EXPERIMENT_NAME}_${TASK}"
    SESSION_LOG_PATH="${LOG_ROOT}/${EXPERIMENT_NAME}_${TASK}"

    STDOUT_FILE="${SESSION_LOG_PATH}/log.out"
    STDERR_FILE="${SESSION_LOG_PATH}/log.err"
    EXIT_CODE_FILE="${SESSION_LOG_PATH}/exit_code"

    # Build the Python command as shell-escaped arguments.
    printf -v EXPERIMENT_Q '%q' "$EXPERIMENT_NAME"
    printf -v TASK_Q '%q' "$TASK"

    PYTHON_CMD="python -u experiment.py $EXPERIMENT_Q $TASK_Q"

    if [[ -n "$RESUME_DIR" ]]; then
        printf -v RESUME_Q '%q' "$RESUME_DIR"
        PYTHON_CMD+=" --resume $RESUME_Q"
    fi

    # Shell-escape everything that is interpolated into the tmux command.
    printf -v UUID_Q '%q' "$UUID"
    printf -v VENV_Q '%q' "$VENV_ACTIVATE"
    printf -v LIQUID_JAX_Q '%q' "$LIQUID_JAX_DIR"
    printf -v STDOUT_Q '%q' "$STDOUT_FILE"
    printf -v STDERR_Q '%q' "$STDERR_FILE"
    printf -v EXIT_CODE_Q '%q' "$EXIT_CODE_FILE"

    TMUX_COMMAND="export CUDA_VISIBLE_DEVICES=$UUID_Q; "
    TMUX_COMMAND+="source $VENV_Q; "
    TMUX_COMMAND+="cd $LIQUID_JAX_Q; "
    TMUX_COMMAND+="$PYTHON_CMD > $STDOUT_Q 2> $STDERR_Q; "
    TMUX_COMMAND+="rc=\$?; "
    TMUX_COMMAND+="echo \$rc > $EXIT_CODE_Q; "
    TMUX_COMMAND+="exit \$rc"

    echo
    echo "Task       : $TASK"
    echo "Experiment : $EXPERIMENT_NAME"
    echo "tmux       : $SESSION_NAME"
    echo "MIG        : ${SELECTED_LABELS[$i]}"

    if [[ -n "$RESUME_DIR" ]]; then
        echo "resume     : $RESUME_DIR"
    else
        echo "resume     : none"
    fi

    echo "stdout     : $STDOUT_FILE"
    echo "stderr     : $STDERR_FILE"
    echo "exit code  : $EXIT_CODE_FILE"

    if (( DRY_RUN )); then
        echo "command    : $TMUX_COMMAND"
        continue
    fi

    mkdir -p "$SESSION_LOG_PATH"

    # Re-check immediately before creation in case something appeared
    # between initial validation and launch.
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        die "tmux session '$SESSION_NAME' already exists."
    fi

    tmux new-session -d -s "$SESSION_NAME" "$TMUX_COMMAND"

    tmux set-option -t "$SESSION_NAME" remain-on-exit on
done

echo
if (( DRY_RUN )); then
    echo "Dry run completed successfully."
    echo "No tmux sessions or log directories were created."
else
    echo "All experiments launched successfully."
fi