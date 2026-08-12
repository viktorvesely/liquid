#!/bin/bash

set -e

# Consider a MIG unused if at least this percentage of its
# nvidia-smi-reported memory is free.
MIN_FREE_PERCENT=99


# ---------------------------------------------------------------------------
# 1. Arguments
# ---------------------------------------------------------------------------

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <experiment_name> <task1> [task2 ...]"
    exit 1
fi

EXPERIMENT_NAME="$1"
shift

TASKS=("$@")
NUM_TASKS=${#TASKS[@]}


# ---------------------------------------------------------------------------
# 2. Kerberos revive session
# ---------------------------------------------------------------------------

REVIVE_SESSION="kerberos_revive"

if tmux has-session -t "$REVIVE_SESSION" 2>/dev/null; then
    echo "Killing existing '$REVIVE_SESSION' session..."
    tmux kill-session -t "$REVIVE_SESSION"
fi

echo "Starting new '$REVIVE_SESSION' session..."
tmux new-session -d -s "$REVIVE_SESSION" \
    "while true; do kinit -R; sleep 8h; done"


# ---------------------------------------------------------------------------
# 3. Discover MIG names + UUIDs from `nvidia-smi -L`
# ---------------------------------------------------------------------------

declare -A MIG_NAMES
declare -A MIG_UUIDS

CURRENT_GPU=""

while read -r line; do

    if [[ "$line" =~ ^GPU\ ([0-9]+): ]]; then
        CURRENT_GPU="${BASH_REMATCH[1]}"
    fi

    # Example:
    # MIG 1g.20gb Device 0: (UUID: MIG-f467...)
    if [[ "$line" =~ MIG\ +(.+)\ +Device\ +([0-9]+):\ +\(UUID:\ +(MIG-[a-f0-9-]+)\) ]]; then
        KEY="$CURRENT_GPU:${BASH_REMATCH[2]}"

        MIG_NAMES["$KEY"]="${BASH_REMATCH[1]}"
        MIG_UUIDS["$KEY"]="${BASH_REMATCH[3]}"
    fi

done < <(nvidia-smi -L)


# ---------------------------------------------------------------------------
# 4. Find unused MIG instances
#
# Candidate format:
#
#   TOTAL_MEMORY|GPU|MIG|NAME|FREE|TOTAL|UUID
#
# Sorting numerically by TOTAL_MEMORY means:
#
#   20 GB -> 40 GB -> 80 GB
#
# so 20 GB MIGs are consumed first.
# ---------------------------------------------------------------------------

declare -a CANDIDATES

while read -r GPU GI CI MIG USED_RAW SLASH TOTAL_RAW REST; do

    # Skip headers / BAR1 / anything that isn't a GPU row
    if ! [[ "$GPU" =~ ^[0-9]+$ ]]; then
        continue
    fi

    USED="${USED_RAW%MiB}"
    TOTAL="${TOTAL_RAW%MiB}"

    # Ensure parsed values are numeric
    if ! [[ "$USED" =~ ^[0-9]+$ && "$TOTAL" =~ ^[0-9]+$ ]]; then
        continue
    fi

    FREE=$((TOTAL - USED))
    KEY="$GPU:$MIG"

    NAME="${MIG_NAMES[$KEY]}"
    UUID="${MIG_UUIDS[$KEY]}"

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
# 5. Prioritize smaller MIGs
#
# In your setup this means approximately:
#
#   19938 MiB -> 40103 MiB -> 81153 MiB
#
# i.e. 1g.20gb first.
# ---------------------------------------------------------------------------

mapfile -t CANDIDATES < <(
    printf '%s\n' "${CANDIDATES[@]}" |
    sort -t'|' -k1,1n
)


# ---------------------------------------------------------------------------
# 6. Make sure enough unused MIGs exist
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
# 7. Select only as many MIGs as we need
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

    LABEL=$(printf \
        "GPU %s [MIG %s] %-10s : %5s / %5s MiB Free (%s)" \
        "$GPU" \
        "$MIG" \
        "$NAME" \
        "$FREE" \
        "$TOTAL" \
        "$UUID"
    )

    SELECTED_LABELS+=("$LABEL")

    echo "  $LABEL"
done


LOG_ROOT="$(pwd)/logs"
mkdir -p "$LOG_ROOT"

# ---------------------------------------------------------------------------
# 8. Launch one tmux session per task
# ---------------------------------------------------------------------------

echo
echo "Launching experiments:"
echo "-------------------------------------"

for ((i = 0; i < NUM_TASKS; i++)); do

    TASK="${TASKS[$i]}"
    UUID="${SELECTED_UUIDS[$i]}"

    SESSION_NAME="${EXPERIMENT_NAME}_${TASK}"

    SESSION_LOG_PATH="${LOG_ROOT}/${EXPERIMENT_NAME}_${TASK}"
    STDOUT_FILE="${SESSION_LOG_PATH}/log.out"
    STDERR_FILE="${SESSION_LOG_PATH}/log.err"

    mkdir -p "$SESSION_LOG_PATH"

    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Error: tmux session '$SESSION_NAME' already exists."
        exit 1
    fi

    tmux new-session -d -s "$SESSION_NAME" \
        "export CUDA_VISIBLE_DEVICES='$UUID'; \
         source .venv/bin/activate; \
         cd liquid_jax; \
         python -u experiment.py '$EXPERIMENT_NAME' '$TASK' \
             > '$STDOUT_FILE' \
             2> '$STDERR_FILE'"

    echo
    echo "Task       : $TASK"
    echo "Experiment : $EXPERIMENT_NAME"
    echo "tmux       : $SESSION_NAME"
    echo "MIG        : ${SELECTED_LABELS[$i]}"
    echo "stdout     : $STDOUT_FILE"
    echo "stderr     : $STDERR_FILE"

done