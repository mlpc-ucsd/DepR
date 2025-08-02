#!/bin/bash

show_usage() {
    echo "Usage:"
    echo "  $0 <world-size> <mode> [additional Python args]"
    echo
    echo "Arguments:"
    echo "  <world-size>     Number of parallel processes (e.g., GPU count)."
    echo "                   Use 0 for a dry-run (prints commands without executing)."
    echo
    echo "  <mode>           One of:"
    echo "                     sample   - Run object sampling stage"
    echo "                     score    - Run object scoring stage"
    echo "                     collate  - Aggregate object results"
    echo "                     scene    - Run scene layout optimization"
    echo "                     all      - Run sample → score → collate"
    exit 1
}

# Require at least 2 arguments
if [ $# -lt 2 ]; then
    show_usage
fi

export OMP_NUM_THREADS=64
WORLD_SIZE=$1
MODE=$2
shift 2

RUNNER=("python" "-m" "scripts.run_inference")
SHARED_PARAMS=(--output-mesh)

# Determine dry-run mode
if [ "$WORLD_SIZE" -eq 0 ]; then
    DRY_RUN=true
    EFFECTIVE_WORLD_SIZE=1
else
    DRY_RUN=false
    EFFECTIVE_WORLD_SIZE=$WORLD_SIZE
fi

# Validate mode
VALID_MODES=("sample" "score" "collate" "scene" "all")
if [[ ! " ${VALID_MODES[@]} " =~ " ${MODE} " ]]; then
    echo "Error: Invalid mode '${MODE}'"
    show_usage
fi

run_stage() {
    local STAGE=$1
    shift
    echo "Running $STAGE..."
    for ((i=0; i<EFFECTIVE_WORLD_SIZE; i++)); do
        CMD=("${RUNNER[@]}" --mode "$STAGE" "${SHARED_PARAMS[@]}" --rank $i --world-size $EFFECTIVE_WORLD_SIZE "$@")
        if $DRY_RUN; then
            echo "[DRY-RUN] CUDA_VISIBLE_DEVICES=$i ${CMD[@]}"
        else
            CUDA_VISIBLE_DEVICES=$i "${CMD[@]}" &
        fi
    done
    if ! $DRY_RUN; then
        wait
    fi
}

run_collate() {
    echo "Running collate..."
    CMD=("${RUNNER[@]}" --mode collate "${SHARED_PARAMS[@]}" "$@")
    if $DRY_RUN; then
        echo "[DRY-RUN] ${CMD[@]}"
    else
        "${CMD[@]}"
    fi
}

# Execution flow
if [[ "$MODE" == "sample" || "$MODE" == "all" ]]; then
    run_stage sample "$@"
fi

if [[ "$MODE" == "score" || "$MODE" == "all" ]]; then
    run_stage score "$@"
fi

if [[ "$MODE" == "collate" || "$MODE" == "all" ]]; then
    run_collate "$@"
fi

if [[ "$MODE" == "scene" ]]; then
    run_stage scene "$@"
fi
