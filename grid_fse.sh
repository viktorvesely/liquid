#!/bin/bash

# 1. Manage Kerberos Revive Session
REVIVE_SESSION="kerberos_revive"
if tmux has-session -t "$REVIVE_SESSION" 2>/dev/null; then
    echo "Killing existing '$REVIVE_SESSION' session..."
    tmux kill-session -t "$REVIVE_SESSION"
fi
echo "Starting new '$REVIVE_SESSION' session..."
tmux new-session -d -s "$REVIVE_SESSION" "while true; do kinit -R; sleep 8h; done"

# 2. Enforce the mandatory global --agg flag
if [[ "$1" != "--agg" ]] || [[ -z "$2" ]]; then
    echo "Error: The --agg flag and its value are required."
    echo "Usage: $0 --agg <string> <task1> <uuid1> <task2> <uuid2> ..."
    exit 1
fi

AGG_FLAG="--agg $2"
shift 2

# 3. Ensure remaining arguments are provided in pairs (Task + UUID)
if [ "$#" -eq 0 ] || [ $(($# % 2)) -ne 0 ]; then
    echo "Error: Please provide pairs of TASK and MIG_UUID."
    echo "Usage: $0 --agg <string> <task1> <uuid1> <task2> <uuid2> ..."
    exit 1
fi

# 4. Launch task sessions
while [ "$#" -gt 0 ]; do
    TASK=$1
    UUID=$2
    shift 2

    SESSION_NAME="grid_${TASK}"

    tmux new-session -d -s "$SESSION_NAME" "export CUDA_VISIBLE_DEVICES=$UUID; source .venv/bin/activate; cd liquid_jax; python grid.py --tasks $TASK $AGG_FLAG"
    
    echo "Launched task '$TASK' on MIG '$UUID' in tmux session '$SESSION_NAME'"
done