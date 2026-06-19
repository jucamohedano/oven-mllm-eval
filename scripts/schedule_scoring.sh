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

Schedule a judge + scoring job (1 GPU) or a CPU-only scoring job on SLURM.

When --judge-model is set, the job requests 1 GPU and runs the judge on the
input samples before scoring.  Otherwise the job is CPU-only and just scores.

Examples:

    # Judge + score on partial shard outputs (merge first, then submit):
    cat results/RUN_DIR/*_samples_shard*.jsonl > results/RUN_DIR/RUN_ID_samples.jsonl
    bash scripts/schedule_scoring.sh \
        --input results/RUN_DIR/RUN_ID_samples.jsonl \
        --judge-model Qwen/Qwen3-8B

    # Score only (CPU):
    bash scripts/schedule_scoring.sh \
        --input results/RUN_DIR/RUN_ID_samples.jsonl

Slurm options:
    -p, --partition <PARTITION>   Partition (default: boost_usr_prod)
    -A, --account <ACCOUNT>       Account (default: none)
    -c, --cpus <CPUS>             CPUs per task (default: 8 for judge, 4 for score)
    -m, --mem <MEM>               Memory limit (default: 64G for judge, 30G for score)
    -t, --time <TIME>             Time limit (default: 04:00:00)
    -n, --name <NAME>             Job name (default: oven-judge-score / oven-score)

Judge options:
    --judge-model <MODEL>         Text-only judge model. When set, requests 1 GPU and
                                  runs the judge before scoring.
    --judge-gpus <N>              Number of GPUs for the judge (default: 1).
                                  When > 1, splits input and runs parallel judge instances.
    --judge-max-model-len <LEN>   Judge max context length (default: 2048)
    --judge-max-num-seqs <N>      Judge max concurrent sequences (default: 1024)
    --judge-gpu-util <UTIL>       Judge GPU memory utilization (default: 0.92)
    --judge-mode <MODE>           Judging mode: structured or free-form (default: structured)
    --judge-n <N>                 Generations per judge prompt — n>1 enables majority
                                  voting (default: 1)
    --judge-temperature <TEMP>    Judge temperature — set >0 for majority voting to
                                  get diverse completions (default: 0.0)
    --judge-top-p <P>             Judge top-p (nucleus) — free-form only (default: 1.0)
    --judge-top-k <K>             Judge top-k — free-form only, -1=disabled (default: -1)

Scoring options:
    --input <PATH>                Input samples JSONL (required)
    --output <PATH>               Per-example scored JSONL output
                                  (default: <input_dir>/<run_id>_scored.jsonl)
    --taxonomy-index <PATH>       Taxonomy index JSON
                                  (default: data/processed/oven_taxonomy_index.json)
    --measure <MEASURE> [...]     Measure(s): exact_match, contained, all
                                  (default: exact_match).  Space-separate for multiple.
    --num-workers <N>             CPU workers for parallel scoring (default: 0 = auto)
'
    exit 0
fi

cd "$(dirname "$0")"
while [ "$(find . -maxdepth 1 -name pyproject.toml | wc -l)" -ne 1 ]; do cd ..; done

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

# SLURM
SLURM_PARTITION="boost_usr_prod"
SLURM_ACCOUNT=""
SLURM_CPUS="4"
SLURM_MEM="30G"
SLURM_TIME="04:00:00"
SLURM_NAME="oven-score"

# Judge
INF_JUDGE_MODEL=""
INF_JUDGE_GPUS="1"
INF_JUDGE_MAX_MODEL_LEN="2048"
INF_JUDGE_MAX_NUM_SEQS="1024"
INF_JUDGE_GPU_UTIL="0.92"
INF_JUDGE_MODE="structured"
INF_JUDGE_N="1"
INF_JUDGE_TEMPERATURE="0.0"
INF_JUDGE_TOP_P="1.0"
INF_JUDGE_TOP_K="-1"
INF_JUDGE_OUTPUT=""     # override judge output path (default: auto-derived)

# Scoring
SCORING_INPUT=""
SCORING_OUTPUT=""
SCORING_TAXONOMY_INDEX="data/processed/oven_taxonomy_index.json"
SCORING_MEASURE="exact_match"
SCORING_NUM_WORKERS="0"

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
            --judge-model)     INF_JUDGE_MODEL="$2"; shift 2 ;;
            --judge-gpus)      INF_JUDGE_GPUS="$2"; shift 2 ;;
            --judge-max-model-len) INF_JUDGE_MAX_MODEL_LEN="$2"; shift 2 ;;
            --judge-max-num-seqs)  INF_JUDGE_MAX_NUM_SEQS="$2"; shift 2 ;;
            --judge-gpu-util)   INF_JUDGE_GPU_UTIL="$2"; shift 2 ;;
            --judge-mode)       INF_JUDGE_MODE="$2"; shift 2 ;;
            --judge-n)          INF_JUDGE_N="$2"; shift 2 ;;
            --judge-temperature) INF_JUDGE_TEMPERATURE="$2"; shift 2 ;;
            --judge-top-p)      INF_JUDGE_TOP_P="$2"; shift 2 ;;
            --judge-top-k)      INF_JUDGE_TOP_K="$2"; shift 2 ;;
            --judge-output)     INF_JUDGE_OUTPUT="$2"; shift 2 ;;
            *) echo "Error: unknown option: $1" >&2; exit 1 ;;
        esac
    done

    # Validate
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

    # Default output
    if [[ -z "$SCORING_OUTPUT" ]]; then
        INPUT_DIR="$(dirname "$SCORING_INPUT")"
        INPUT_BASENAME="$(basename "$SCORING_INPUT" .jsonl)"
        SCORING_OUTPUT="${INPUT_DIR}/${INPUT_BASENAME}_scored.jsonl"
    fi

    # Judge defaults
    if [[ -n "$INF_JUDGE_MODEL" ]]; then
        SLURM_NAME="${SLURM_NAME}-judge"
        SLURM_CPUS="8"
        # Only upgrade mem if user didn't set it explicitly (default is 30G).
        if [[ "$SLURM_MEM" == "30G" ]]; then
            SLURM_MEM="64G"
        fi
        # NB: do NOT use $( [[ ... ]] && echo s ) here — under `set -e` the
        # failing test (when GPUS=1) makes the command-sub return non-zero,
        # which aborts the assignment and kills the script before submit.
        if [[ "$INF_JUDGE_GPUS" -gt 1 ]]; then
            JOB_TYPE="$INF_JUDGE_GPUS GPUs (judge + score)"
        else
            JOB_TYPE="1 GPU (judge + score)"
        fi
    else
        JOB_TYPE="CPU-only (score)"
    fi

    # Pre-flight: venv must exist
    if [[ ! -x ".venv/bin/python" ]]; then
        echo "Error: .venv/bin/python not found." >&2
        echo "Build the venv on a login node first:" >&2
        echo "    uv sync" >&2
        exit 1
    fi

    mkdir -p ./logs/slurm

    echo "[info] Scheduling $JOB_TYPE job:"
    echo "  Input:        $SCORING_INPUT"
    echo "  Output:       $SCORING_OUTPUT"
    echo "  Measure:      $SCORING_MEASURE"
    echo "  Workers:      $SCORING_NUM_WORKERS"
    if [[ -n "$INF_JUDGE_MODEL" ]]; then
        echo "  Judge model:  $INF_JUDGE_MODEL  (GPUs=$INF_JUDGE_GPUS)"
        echo "  Judge mode:   $INF_JUDGE_MODE  (n=$INF_JUDGE_N, temp=$INF_JUDGE_TEMPERATURE, top_p=$INF_JUDGE_TOP_P, top_k=$INF_JUDGE_TOP_K)"
    fi
    echo "  Partition:    $SLURM_PARTITION"
    echo "  CPUs:         $SLURM_CPUS"
    echo "  Mem:          $SLURM_MEM"
    echo "  Time:         $SLURM_TIME"
    echo ""

    GPU_DIRECTIVE=""
    if [[ -n "$INF_JUDGE_MODEL" ]]; then
        GPU_DIRECTIVE="#SBATCH --gres=gpu:$INF_JUDGE_GPUS"
    fi

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
$GPU_DIRECTIVE
#SBATCH --mem=$SLURM_MEM
#SBATCH --time=$SLURM_TIME

set -euo pipefail
trap 'kill 0' EXIT

cd "\$SLURM_SUBMIT_DIR"

module load nvhpc/24.5 gcc/12.2.0
export CC=gcc CXX=g++ OMP_NUM_THREADS=1
if [[ -f ".env" ]]; then set -a; source .env; set +a; fi
source .venv/bin/activate

echo "[info] \$SLURM_JOB_ID on \$(hostname)"
echo "  Input:    $SCORING_INPUT"
echo "  Output:   $SCORING_OUTPUT"

SAMPLES="$SCORING_INPUT"

if [[ -n "$INF_JUDGE_MODEL" ]]; then
    echo "[info] Judging \$SAMPLES (GPUs=$INF_JUDGE_GPUS)..."
    _JUDGE_SLUG="\$(echo "$INF_JUDGE_MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')"
    if [[ -n "$INF_JUDGE_OUTPUT" ]]; then
        JUDGED="$INF_JUDGE_OUTPUT"
    else
        JUDGED="\$(dirname "\$SAMPLES")/\$(basename "\$SAMPLES" .jsonl)_judged_\${_JUDGE_SLUG}.jsonl"
    fi

    if [[ "$INF_JUDGE_GPUS" -le 1 ]]; then
        CUDA_VISIBLE_DEVICES=0 python -m scripts.run_judge \\
            --input "\$SAMPLES" \\
            --output "\$JUDGED" \\
            --judge-model "$INF_JUDGE_MODEL" \\
            --judge-mode "$INF_JUDGE_MODE" \\
            --judge-n "$INF_JUDGE_N" \\
            --judge-temperature "$INF_JUDGE_TEMPERATURE" \\
            --judge-top-p "$INF_JUDGE_TOP_P" \\
            --judge-top-k "$INF_JUDGE_TOP_K" \\
            --max-model-len "$INF_JUDGE_MAX_MODEL_LEN" \\
            --max-num-seqs "$INF_JUDGE_MAX_NUM_SEQS" \\
            --gpu-util "$INF_JUDGE_GPU_UTIL"
    else
        # ── Durable data-parallel judge (strided sharding) ────────
        # Each GPU processes examples[shard::num_shards] and writes to a
        # persistent _shard{N}.jsonl file.  Crashes preserve per-shard
        # progress.  GPU-count changes work: run_judge.py reads ALL
        # _shard*.jsonl files for a global resume set.
        echo "[info] Launching $INF_JUDGE_GPUS judge shards (strided)..."
        pids=()
        for i in \$(seq 0 $((INF_JUDGE_GPUS - 1))); do
            CUDA_VISIBLE_DEVICES=\$i python -m scripts.run_judge \\
                --input "\$SAMPLES" \\
                --output "\$JUDGED" \\
                --shard \$i --num-shards $INF_JUDGE_GPUS \\
                --judge-model "$INF_JUDGE_MODEL" \\
                --judge-mode "$INF_JUDGE_MODE" \\
                --judge-n "$INF_JUDGE_N" \\
                --judge-temperature "$INF_JUDGE_TEMPERATURE" \\
                --judge-top-p "$INF_JUDGE_TOP_P" \\
                --judge-top-k "$INF_JUDGE_TOP_K" \\
                --max-model-len "$INF_JUDGE_MAX_MODEL_LEN" \\
                --max-num-seqs "$INF_JUDGE_MAX_NUM_SEQS" \\
                --gpu-util "$INF_JUDGE_GPU_UTIL" &
            pids+=(\$!)
        done
        fail=0
        for pid in "\${pids[@]}"; do wait "\$pid" || fail=1; done
        if [[ \$fail -ne 0 ]]; then
            echo "[error] a judge shard failed" >&2; exit 1
        fi
        # Merge persistent shard outputs into the final judged file.
        shards=(\$(ls "\${JUDGED}"_shard*.jsonl 2>/dev/null | sort))
        if [[ \${#shards[@]} -gt 0 ]]; then
            cat "\${shards[@]}" > "\$JUDGED"
            echo "[judge] merged \${#shards[@]} shards → \$(wc -l < "\$JUDGED") rows"
        fi
    fi
    SAMPLES="\$JUDGED"
fi

echo "[info] Scoring \$SAMPLES..."
python -m scripts.score_predictions \\
    --input "\$SAMPLES" \\
    --output "$SCORING_OUTPUT" \\
    --taxonomy-index "$SCORING_TAXONOMY_INDEX" \\
    --measure $SCORING_MEASURE \\
    --num-workers $SCORING_NUM_WORKERS

# Clean up intermediate shard files only if scoring succeeded AND
# row counts match between shards and merged files.
if [[ -s "$SCORING_OUTPUT" ]]; then
    _shard_dir="\$(dirname "\$SAMPLES")"
    _jbase="\$(basename "\$JUDGED" .jsonl)"
    _judge_shard_total=\$(cat "\${_shard_dir}"/\${_jbase}_shard*.jsonl 2>/dev/null | wc -l)
    _judge_total=\$(wc -l < "\$JUDGED" 2>/dev/null || echo 0)

    if [[ "\${_judge_shard_total}" -eq "\${_judge_total}" ]]; then
        rm -f "\${_shard_dir}"/\${_jbase}_shard*.jsonl \
              "\${_shard_dir}"/shard*.log \
              "\${_shard_dir}"/engine_debug_shard*.log \
              "\${_shard_dir}"/mem_timeline.log
        echo "[cleanup] removed intermediate shard files"
    else
        echo "[cleanup] SKIPPED — judge shard/merge mismatch "\
             "(\${_judge_shard_total} shard vs \${_judge_total} merged)"
    fi
fi

echo "[info] Done. Output: $SCORING_OUTPUT"
EOT
}

main "$@"