#!/usr/bin/env bash
# Source this file to get helper functions for navigating run directories.
#
# Usage:
#   source scripts/latest_run.sh
#   latest 32b no_idk          # latest 32B no_idk run (any dataset)
#   latest 2b concise aligned   # latest 2B concise run on aligned data
#   latest 4b no_idk original no_image  # no-image baseline
#   cd $(latest 32b no_idk)    # cd into it
#   resume 32b no_idk aligned  # prints a resume command
#   catalog                    # regenerate runs.tsv
#
# You can also call it directly:
#   bash scripts/latest_run.sh 32b no_idk

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CATALOG="$REPO_ROOT/logs/schedule/runs.tsv"

# Ensure the catalog exists
_catalog_ensure() {
    if [[ ! -f "$CATALOG" ]]; then
        echo "[info] Building run catalog..." >&2
        uv run python "$SCRIPT_DIR/catalog_runs.py" --logs-dir "$REPO_ROOT/logs/schedule" --output "$CATALOG"
    fi
}

# Query runs.tsv for the latest matching run directory.
# Arguments (all optional, positional):
#   $1 - model: 2b, 4b, 8b, 32b
#   $2 - prompt variant: concise, no_idk (maps to concise_no_idk), concise_no_idk
#   $3 - dataset: aligned, original
#   $4 - no_image: no_image (or empty for with-image)
_query_latest() {
    _catalog_ensure

    local model="${1:-}"
    local prompt="${2:-}"
    local dataset="${3:-}"
    local no_image="${4:-}"

    # Normalise prompt: "no_idk" -> "concise_no_idk"
    if [[ "$prompt" == "no_idk" ]]; then
        prompt="concise_no_idk"
    fi

    # Build awk filter
    local awk_filter=""
    [[ -n "$model" ]]    && awk_filter+='$5 == "'"$model"'" && '
    [[ -n "$prompt" ]]   && awk_filter+='$3 == "'"$prompt"'" && '
    [[ -n "$dataset" ]]  && awk_filter+='$2 == "'"$dataset"'" && '
    if [[ "$no_image" == "no_image" ]]; then
        awk_filter+='$4 == "true" && '
    else
        awk_filter+='$4 == "false" && '  # default: with-image only
    fi
    awk_filter+='1'

    # Extract run_dir, sort by timestamp (embedded in path), return latest
    tail -n +2 "$CATALOG" \
        | awk -F'\t' "$awk_filter" \
        | sort -t$'\t' -k1,1 \
        | tail -1 \
        | cut -f1
}

# Shortcut: prints the latest run dir for the given model + prompt variant
latest() {
    _query_latest "$@"
}

# Print a resume command for the latest run
resume() {
    local dir=$(_query_latest "$@")
    [[ -z "$dir" ]] && { echo "No run found." >&2; return 1; }

    local prompt="${2:-no_idk}"
    [[ "$prompt" == "no_idk" ]] && prompt="concise_no_idk"

    local dataset="${3:-aligned}"
    case "$dataset" in
        aligned)  inf="data/processed/vlm_compatible_val_aligned.jsonl" ;;
        original) inf="data/processed/vlm_compatible_val.jsonl" ;;
        *)        inf="data/processed/vlm_compatible_val_aligned.jsonl" ;;
    esac

    local model="${1:-8b}"
    local tp=1 dp=1
    case "$model" in
        32b) tp=4; dp=1 ;;
        2b)  tp=1; dp=4 ;;
        4b)  tp=1; dp=4 ;;
        8b)  tp=1; dp=4 ;;
    esac

    echo "bash scripts/schedule_sbatch.sh -A EUHPC_D33_243 -p boost_usr_prod \\"
    echo "    -g 4 -c 32 --tp $tp --dp $dp --gpu-util 0.90 --mem 256G -t 24:00:00 \\"
    echo "    --model Qwen/Qwen3-VL-${model^^}-Instruct --method naive-sampling \\"
    echo "    --samples-per-example 256 --prompt $prompt --max-tokens 16 \\"
    echo "    --max-model-len 1024 --max-num-seqs 2048 --chunk-size 256 \\"
    echo "    --input $inf \\"
    echo "    --output-dir $dir \\"
    echo "    --image-root /leonardo_work/EUHPC_D33_243/oven/ \\"
    echo "    --judge-model Qwen/Qwen3-4B --judge-gpus 4 \\"
    echo "    --judge-max-num-seqs 8192 \\"
    echo "    --judge-mode free-form --judge-n 1 --judge-temperature 0.7 \\"
    echo "    --judge-top-p 0.8 --judge-top-k 20 \\"
    echo "    --resume"
}

# Regenerate the catalog
catalog() {
    uv run python "$SCRIPT_DIR/catalog_runs.py" --logs-dir "$REPO_ROOT/logs/schedule" --output "$CATALOG"
}

# If run directly (not sourced), call latest
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    latest "$@"
fi
