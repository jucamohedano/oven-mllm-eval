#!/usr/bin/env bash
# shellcheck disable=SC2034
set -o errexit
set -o nounset
set -o pipefail

if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace
fi

if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
    echo 'usage: schedule_scoring.sh [-h] [OPTIONS]

Schedule a CPU-only scoring job on the SLURM cluster.  No GPUs required.

Example — score the latest 4B inference output on the free CPU partition:

    bash scripts/schedule_scoring.sh \
        --input logs/schedule/oven_naive-sampling_barebones/qwen_qwen3-vl-4b-instruct/20260527_180244_074760/20260527_180244_074760_samples.jsonl

Example — score with multiple measures using 8 cores on a DCGP node:

    bash scripts/schedule_scoring.sh \
        --input logs/schedule/oven_naive-sampling_barebones/qwen_qwen3-vl-8b-instruct/20260101_120000_000000/20260101_120000_000000_samples.jsonl \
        --measure exact_match contained \
        --num-workers 8 \
        --partition dcgp_usr_prod \
        --cpus 8

Slurm options:
    -p, --partition <PARTITION>   Partition (default: lrd_all_serial — free CPU tier)
    -A, --account <ACCOUNT>       Account (default: none needed for lrd_all_serial)
    -c, --cpus <CPUS>             CPUs per task (default: 4)
    -m, --mem <MEM>               Memory limit (default: 30G)
    -t, --time <TIME>             Time limit (default: 04:00:00)
    -n, --name <NAME>             Job name (default: oven-score)

Scoring options:
    --input <PATH>                Input samples JSONL (required)
    --output <PATH>               Per-example scored JSONL output
                                  (default: <input_dir>/<run_id>_scored.jsonl)
    --taxonomy-index <PATH>       Taxonomy index JSON
                                  (default: data/processed/oven_taxonomy_index.json)
    --measure <MEASURE> [...]     Measure(s): exact_match, contained, all
                                  (default: exact_match).  Space-separate for multiple.
    --num-workers <N>             CPU workers for parallel scoring (default: 4)
'
    exit 0
fi

cd "$(dirname "$0")"
while [ "$(find . -maxdepth 1 -name pyproject.toml | wc -l)" -ne 1 ]; do cd ..; done

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

# SLURM
SLURM_PARTITION="lrd_all_serial"
SLURM_ACCOUNT=""
SLURM_CPUS="4"
SLURM_MEM="30G"
SLURM_TIME="04:00:00"
SLURM_NAME="oven-score"

# Scoring
SCORING_INPUT=""
SCORING_OUTPUT=""
SCORING_TAXONOMY_INDEX="data/processed/oven_taxonomy_index.json"
SCORING_MEASURE="exact_match"
SCORING_NUM_WORKERS="4"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

main() {
    while [[ $# -gt 0 ]]; do
        if [[ $1 == "--" ]]; then
            shift
            break
        fi
        case $1 in
            -p|--partition)    SLURM_PARTITION="$2"; shift 2 ;;
            -A|--account)      SLURM_ACCOUNT="$2"; shift 2 ;;
            -c|--cpus)         SLURM_CPUS="$2"; shift 2 ;;
            -m|--mem)          SLURM_MEM="$2"; shift 2 ;;
            -t|--time)         SLURM_TIME="$2"; shift 2 ;;
            -n|--name)         SLURM_NAME="$2"; shift 2 ;;
            --input)           SCORING_INPUT="$2"; shift 2 ;;
            --output)          SCORING_OUTPUT="$2"; shift 2 ;;
            --taxonomy-index)  SCORING_TAXONOMY_INDEX="$2"; shift 2 ;;
            --measure)         SCORING_MEASURE="$2"; shift 2 ;;
            --num-workers)     SCORING_NUM_WORKERS="$2"; shift 2 ;;
            *) echo "Error: unknown option: $1" >&2; exit 1 ;;
        esac
    done

    # Validate required
    if [[ -z "$SCORING_INPUT" ]]; then
        echo "[error] --input is required" >&2
        exit 1
    fi
    if [[ ! -f "$SCORING_INPUT" ]]; then
        echo "[error] --input file not found: $SCORING_INPUT" >&2
        exit 1
    fi

    # Guard: --mem without a unit suffix
    if [[ "$SLURM_MEM" =~ ^[0-9]+$ ]]; then
        echo "[error] --mem '$SLURM_MEM' has no unit suffix. Did you mean '${SLURM_MEM}G'?" >&2
        exit 1
    fi

    # Build SLURM account directive
    SLURM_ACCOUNT_DIRECTIVE=""
    if [[ -n "$SLURM_ACCOUNT" ]]; then
        SLURM_ACCOUNT_DIRECTIVE="#SBATCH --account=$SLURM_ACCOUNT"
    fi

    # Default output: <input_dir>/<run_id>_scored.jsonl
    if [[ -z "$SCORING_OUTPUT" ]]; then
        INPUT_DIR="$(dirname "$SCORING_INPUT")"
        INPUT_BASENAME="$(basename "$SCORING_INPUT" .jsonl)"
        SCORING_OUTPUT="${INPUT_DIR}/${INPUT_BASENAME}_scored.jsonl"
    fi

    # Pre-flight: venv must exist
    if [[ ! -x ".venv/bin/python" ]]; then
        echo "Error: .venv/bin/python not found." >&2
        echo "Build the venv on a login node first:" >&2
        echo "    uv sync" >&2
        exit 1
    fi

    mkdir -p ./logs/slurm

    echo "[info] Scheduling scoring job:"
    echo "  Input:        $SCORING_INPUT"
    echo "  Output:       $SCORING_OUTPUT"
    echo "  Measure:      $SCORING_MEASURE"
    echo "  Workers:      $SCORING_NUM_WORKERS"
    echo "  Partition:    $SLURM_PARTITION"
    echo "  CPUs:         $SLURM_CPUS"
    echo "  Mem:          $SLURM_MEM"
    echo "  Time:         $SLURM_TIME"
    echo ""

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$SLURM_NAME
#SBATCH --output=./logs/slurm/%j.out
#SBATCH --error=./logs/slurm/%j.err
#SBATCH --partition=$SLURM_PARTITION
$SLURM_ACCOUNT_DIRECTIVE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$SLURM_CPUS
#SBATCH --mem=$SLURM_MEM
#SBATCH --time=$SLURM_TIME

set -euo pipefail
trap 'kill 0' EXIT

cd "\$SLURM_SUBMIT_DIR"

module load gcc/12.2.0

source .venv/bin/activate

echo "[info] Scoring \$SLURM_JOB_ID on \$(hostname)"
echo "  Input:    $SCORING_INPUT"
echo "  Output:   $SCORING_OUTPUT"
echo "  Measure:  $SCORING_MEASURE"
echo "  Workers:  $SCORING_NUM_WORKERS"
echo ""

python -m scripts.score_predictions \\
    --input "$SCORING_INPUT" \\
    --output "$SCORING_OUTPUT" \\
    --taxonomy-index "$SCORING_TAXONOMY_INDEX" \\
    --measure $SCORING_MEASURE \\
    --num-workers $SCORING_NUM_WORKERS

echo "[info] Done."
EOT
}

main "$@"
