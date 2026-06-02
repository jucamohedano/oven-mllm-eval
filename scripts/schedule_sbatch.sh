#!/usr/bin/env bash
# shellcheck disable=SC2034
set -o errexit
set -o nounset
set -o pipefail

if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace
fi

if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
    echo 'usage: schedule_sbatch.sh [-h] [OPTIONS]

Schedule an OVEN inference job on the SLURM cluster.

Prerequisites:
    Build the venv ONCE on a login node before submitting:
        uv sync --extra serve
    The compute node will only activate the existing venv, never modify it.

Slurm options:
    -p, --partition <PARTITION>   Partition to use (default: boost_usr_prod)
    -A, --account <ACCOUNT>       Account to use
    -c, --cpus <CPUS>             CPUs per task (default: 8)
    -g, --gpus <GPUS>             Number of GPUs (default: 4)
    -m, --mem <MEM>               Memory limit (default: 128G)
    -t, --time <TIME>             Time limit (default: 04:00:00)
    -n, --name <NAME>             Job name (default: oven-mllm-eval)

Inference options:
    --model <MODEL>               Model path or HF ID (default: Qwen/Qwen3-VL-8B-Instruct)
    --method <METHOD>             Sampling method: naive, naive-sampling, iterative (default: naive)
    --prompt <VARIANT>           Prompt variant: barebones, default, specific, vague (default: barebones)
    --temperature <TEMP>          Sampling temperature (default: 1.0)
    --top-p <P>                   Nucleus sampling threshold (default: 1.0)
    --top-k <K>                   Top-k sampling (default: -1 = disabled)
    --max-tokens <TOKENS>         Max tokens per sample (default: 300)
    --samples-per-example <N>     Samples per example for naive-sampling (default: 64)
    --attempts-per-round <N>      Attempts per round for iterative (default: 16)
    --max-rounds <T>              Max rounds for iterative (default: 1)
    --feedback <BOOL>             Enable feedback for iterative: true/false (default: false)
    --max-feedback-chars <N>      Max chars for feedback messages (default: 2000)

Scoring options:
    --scoring-measure <MEASURE>   Measure(s) for DirectMeasureMatcher: exact_match, contained, all
                                  (default: exact_match).  Space-separate for multiple.
    --scoring-workers <N>         Number of CPU workers for parallel scoring (default: 0 = auto)

Data options:
    --input <PATH>                Input JSONL path (default: data/processed/vlm_compatible_val.jsonl)
    --taxonomy-index <PATH>       Taxonomy index JSON (default: data/processed/oven_taxonomy_index.json)
    --max-examples <N>            Limit number of examples (default: all)
    --resume                      Skip already-completed examples

vLLM engine options:
    --tp <N>                      Tensor parallelism (default: 4)
    --dp <N>                      Data-parallel replicas (default: 1 — prefer for models that fit on 1 GPU)
    --gpu-util <UTIL>             GPU memory utilization (default: 0.92)
    --max-model-len <LEN>         Max model context length (default: 4096)
    --max-num-seqs <N>            Max concurrent sequences — lower reduces KV cache memory (default: 1024)
    --max-pixels <N>              Max pixels for image resizing (default: 262144 = 512x512)
    --min-pixels <N>              Min pixels for image resizing (default: 65536 = 256x256)
    --chunk-size <N>              Examples per llm.chat() call (default: 256)
    --enforce-eager               Disable CUDA graphs — slower but more uniform latency
    --base-model                  Use LLM.generate() with raw prompts (for base/pretrained models)

Output:
    Results are saved under logs/schedule/oven_<method>_<prompt>/<model>/<run_id>/
    following the lmms-ocw convention.  Each run produces:
        <run_id>_samples.jsonl     raw predictions (from inference)
        <run_id>_scored.jsonl      per-sample outputs + metrics (from scoring)
        <run_id>_results.json      aggregate metrics (hP, hR, hF, exact)
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
SLURM_CPUS="8"
SLURM_GPUS="4"
SLURM_MEM="128G"
SLURM_TIME="24:00:00"
SLURM_NAME="oven-mllm-eval"
SLURM_OUTPUT="./logs/slurm/%j.out"
SLURM_ERROR="./logs/slurm/%j.err"

# Inference
INF_MODEL="Qwen/Qwen3-VL-8B-Instruct"
INF_METHOD="naive"
INF_PROMPT="barebones"
INF_TEMPERATURE="1.0"
INF_TOP_P="1.0"
INF_TOP_K="-1"
INF_MAX_TOKENS="300"
INF_SAMPLES_PER_EXAMPLE="64"
INF_ATTEMPTS_PER_ROUND="16"
INF_MAX_ROUNDS="1"
INF_FEEDBACK="false"
INF_MAX_FEEDBACK_CHARS="2000"

# Data
INF_INPUT="data/processed/vlm_compatible_val.jsonl"
INF_TAXONOMY_INDEX="data/processed/oven_taxonomy_index.json"
INF_MAX_EXAMPLES=""
INF_RESUME=false

# Scoring
SCORING_MEASURE="exact_match"
INF_SCORING_WORKERS="0"

# vLLM engine
INF_TP="4"
INF_DP="1"
INF_GPU_UTIL="0.92"
INF_MAX_MODEL_LEN="4096"
INF_MAX_NUM_SEQS="1024"
INF_MAX_PIXELS="262144"
INF_MIN_PIXELS="65536"
INF_CHUNK_SIZE="256"
INF_ENFORCE_EAGER=false
INF_BASE_MODEL=false

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
            -g|--gpus)         SLURM_GPUS="$2"; shift 2 ;;
            -m|--mem)          SLURM_MEM="$2"; shift 2 ;;
            -t|--time)         SLURM_TIME="$2"; shift 2 ;;
            -n|--name)         SLURM_NAME="$2"; shift 2 ;;
            --model)           INF_MODEL="$2"; shift 2 ;;
            --method)          INF_METHOD="$2"; shift 2 ;;
            --prompt)          INF_PROMPT="$2"; shift 2 ;;
            --temperature)     INF_TEMPERATURE="$2"; shift 2 ;;
            --top-p)           INF_TOP_P="$2"; shift 2 ;;
            --top-k)           INF_TOP_K="$2"; shift 2 ;;
            --max-tokens)      INF_MAX_TOKENS="$2"; shift 2 ;;
            --samples-per-example) INF_SAMPLES_PER_EXAMPLE="$2"; shift 2 ;;
            --attempts-per-round) INF_ATTEMPTS_PER_ROUND="$2"; shift 2 ;;
            --max-rounds)      INF_MAX_ROUNDS="$2"; shift 2 ;;
            --feedback)        INF_FEEDBACK="$2"; shift 2 ;;
            --max-feedback-chars) INF_MAX_FEEDBACK_CHARS="$2"; shift 2 ;;
            --input)           INF_INPUT="$2"; shift 2 ;;
            --taxonomy-index)  INF_TAXONOMY_INDEX="$2"; shift 2 ;;
            --max-examples)    INF_MAX_EXAMPLES="$2"; shift 2 ;;
            --resume)          INF_RESUME=true; shift ;;
            --tp)              INF_TP="$2"; shift 2 ;;
            --dp)              INF_DP="$2"; shift 2 ;;
            --gpu-util)        INF_GPU_UTIL="$2"; shift 2 ;;
            --max-model-len)   INF_MAX_MODEL_LEN="$2"; shift 2 ;;
            --max-num-seqs)    INF_MAX_NUM_SEQS="$2"; shift 2 ;;
            --max-pixels)      INF_MAX_PIXELS="$2"; shift 2 ;;
            --min-pixels)      INF_MIN_PIXELS="$2"; shift 2 ;;
            --chunk-size)      INF_CHUNK_SIZE="$2"; shift 2 ;;
            --enforce-eager)   INF_ENFORCE_EAGER=true; shift ;;
            --base-model)      INF_BASE_MODEL=true; shift ;;
            --scoring-measure) SCORING_MEASURE="$2"; shift 2 ;;
            --scoring-workers) INF_SCORING_WORKERS="$2"; shift 2 ;;
            *) echo "Error: unknown option: $1" >&2; exit 1 ;;
        esac
    done

    # Guard: --mem without a unit suffix is interpreted as megabytes by SLURM.
    if [[ "$SLURM_MEM" =~ ^[0-9]+$ ]]; then
        echo "[error] --mem '$SLURM_MEM' has no unit suffix — SLURM interprets bare numbers as megabytes!" >&2
        echo "        Did you mean '${SLURM_MEM}G'? (e.g. --mem 128G)" >&2
        exit 1
    fi

    # Build SLURM account directive
    SLURM_ACCOUNT_DIRECTIVE=""
    if [[ -n "$SLURM_ACCOUNT" ]]; then
        SLURM_ACCOUNT_DIRECTIVE="#SBATCH --account=$SLURM_ACCOUNT"
    fi

    # Build max-examples flag
    MAX_EXAMPLES_FLAG=""
    if [[ -n "$INF_MAX_EXAMPLES" ]]; then
        MAX_EXAMPLES_FLAG="--max-examples $INF_MAX_EXAMPLES"
    fi

    # Build resume flag
    RESUME_FLAG=""
    if [[ "$INF_RESUME" == true ]]; then
        RESUME_FLAG="--resume"
    fi

    # Build enforce-eager flag
    ENFORCE_EAGER_FLAG=""
    if [[ "$INF_ENFORCE_EAGER" == true ]]; then
        ENFORCE_EAGER_FLAG="--enforce-eager"
    fi

    # Build base-model flag
    BASE_MODEL_FLAG=""
    if [[ "$INF_BASE_MODEL" == true ]]; then
        BASE_MODEL_FLAG="--base-model"
    fi

    # Pre-flight: venv must already be built
    if [[ ! -x ".venv/bin/python" ]]; then
        echo "Error: .venv/bin/python not found." >&2
        echo "Build the venv on a login node first:" >&2
        echo "    uv sync --extra serve" >&2
        exit 1
    fi

    mkdir -p ./logs/slurm

    echo "[info] Scheduling job:"
    echo "  Model:        $INF_MODEL"
    echo "  Method:       $INF_METHOD"
    echo "  Prompt:       $INF_PROMPT"
    echo "  Temperature:  $INF_TEMPERATURE"
    echo "  Top-p:        $INF_TOP_P"
    echo "  Top-k:        $INF_TOP_K"
    echo "  GPUs:         $SLURM_GPUS  (TP=$INF_TP, DP=$INF_DP)"
    echo "  Max pixels:   $INF_MAX_PIXELS"
    echo "  Base model:   $INF_BASE_MODEL"
    echo "  Score with:   $SCORING_MEASURE  (workers=${INF_SCORING_WORKERS:-0})"
    echo ""

    # Submit
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$SLURM_NAME
#SBATCH --output=$SLURM_OUTPUT
#SBATCH --error=$SLURM_ERROR
#SBATCH --partition=$SLURM_PARTITION
$SLURM_ACCOUNT_DIRECTIVE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$SLURM_CPUS
#SBATCH --gres=gpu:$SLURM_GPUS
#SBATCH --mem=$SLURM_MEM
#SBATCH --time=$SLURM_TIME

set -euo pipefail

# Kill the entire process group on exit — prevents orphaned vLLM worker
# processes from keeping the SLURM allocation alive after a crash.
trap 'kill 0' EXIT

# -----------------------------------------------------------------------
# Land in the submit directory
# -----------------------------------------------------------------------
cd "\$SLURM_SUBMIT_DIR"

# -----------------------------------------------------------------------
# Load cluster modules
# -----------------------------------------------------------------------
module load nvhpc/24.5
module load gcc/12.2.0

export CC=gcc
export CXX=g++
export OMP_NUM_THREADS=1

# -----------------------------------------------------------------------
# Load cluster-specific environment (HF_HOME, offline mode, etc.)
# -----------------------------------------------------------------------
if [[ -f ".env" ]]; then
    set -a; source .env; set +a
fi

# -----------------------------------------------------------------------
# Activate the pre-built venv
# The venv must already be in sync — never sync from a compute node.
# -----------------------------------------------------------------------
source .venv/bin/activate

# Fail fast if the venv is broken on this node
VENV_CHECK_OUT=\$(python -c "import vllm; print('ok')" 2>&1) || VENV_RC=\$?
VENV_RC=\${VENV_RC:-0}
if echo "\${VENV_CHECK_OUT}" | grep -q '^ok$'; then
    echo "[info] venv OK on \$(hostname)"
else
    if [ "\${VENV_RC}" -ge 128 ]; then
        echo "[fatal] venv health check was killed by signal \$(( VENV_RC - 128 )) on \$(hostname)." >&2
        echo "  This usually means the job ran out of host RAM during the import probe." >&2
        echo "  Check your --mem setting (current: $SLURM_MEM)." >&2
        echo "  Tip: use a unit suffix, e.g. --mem 128G (bare numbers are megabytes!)." >&2
    else
        echo "[fatal] venv broken on \$(hostname); resync on a login node:" >&2
        echo "    uv sync --extra serve" >&2
    fi
    exit 1
fi

# -----------------------------------------------------------------------
# Run inference
# -----------------------------------------------------------------------
echo "[info] Running inference: method=$INF_METHOD prompt=$INF_PROMPT temp=$INF_TEMPERATURE"

python -m scripts.run_inference \\
    --input "$INF_INPUT" \\
    --model "$INF_MODEL" \\
    --prompt-variant "$INF_PROMPT" \\
    --method "$INF_METHOD" \\
    --temperature "$INF_TEMPERATURE" \\
    --top-p "$INF_TOP_P" \\
    --top-k "$INF_TOP_K" \\
    --max-tokens "$INF_MAX_TOKENS" \\
    --tp "$INF_TP" \\
    --dp "$INF_DP" \\
    --gpu-util "$INF_GPU_UTIL" \\
    --max-model-len "$INF_MAX_MODEL_LEN" \\
    --max-num-seqs "$INF_MAX_NUM_SEQS" \\
    --max-pixels "$INF_MAX_PIXELS" \\
    --min-pixels "$INF_MIN_PIXELS" \\
    --chunk-size "$INF_CHUNK_SIZE" \\
    $ENFORCE_EAGER_FLAG \\
    $BASE_MODEL_FLAG \\
    \$([ "$INF_METHOD" = "naive-sampling" ] && echo "--samples-per-example $INF_SAMPLES_PER_EXAMPLE") \\
    \$([ "$INF_METHOD" = "iterative" ] && echo "--attempts-per-round $INF_ATTEMPTS_PER_ROUND --max-rounds $INF_MAX_ROUNDS --enable-feedback $INF_FEEDBACK --max-feedback-chars $INF_MAX_FEEDBACK_CHARS") \\
    $MAX_EXAMPLES_FLAG \\
    $RESUME_FLAG

# -----------------------------------------------------------------------
# Score results — find the latest run directory and score samples
# -----------------------------------------------------------------------
MODEL_SLUG=\$(echo "$INF_MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
EXPERIMENT_DIR="logs/schedule/oven_${INF_METHOD}_${INF_PROMPT}/\${MODEL_SLUG}"
OUTPUT_DIR=\$(ls -1d "\${EXPERIMENT_DIR}"/20* 2>/dev/null | sort | tail -1)

if [[ -z "\${OUTPUT_DIR}" ]]; then
    echo "[error] No output directory found under \${EXPERIMENT_DIR}" >&2
else
    SAMPLES=\$(ls -1 "\${OUTPUT_DIR}"/*_samples.jsonl 2>/dev/null | head -1)
    if [[ -z "\${SAMPLES}" ]]; then
        echo "[error] No *_samples.jsonl found in \${OUTPUT_DIR}" >&2
    else
        echo "[info] Scoring \${SAMPLES}..."
        RUN_ID=\$(basename "\${SAMPLES}" _samples.jsonl)
        SCORED_OUT="\${OUTPUT_DIR}/\${RUN_ID}_scored.jsonl"
        python -m scripts.score_predictions \\
            --input "\${SAMPLES}" \\
            --output "\${SCORED_OUT}" \\
            --taxonomy-index "$INF_TAXONOMY_INDEX" \\
            --measure $SCORING_MEASURE \\
            --num-workers $INF_SCORING_WORKERS
    fi
fi

echo "[info] Done."
EOT
}

main "$@"
