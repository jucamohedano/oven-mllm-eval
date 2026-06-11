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
    --judge-model <MODEL>         Text-only judge model (default: none). When set, runs a
                                  two-job pipeline with judge between inference and scoring.
    --judge-max-model-len <LEN>   Judge max context length (default: 2048)
    --judge-max-num-seqs <N>      Judge max concurrent sequences (default: 1024)
    --judge-gpu-util <UTIL>       Judge GPU memory utilization (default: 0.92)
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

# Output
INF_OUTPUT_DIR=""
INF_IMAGE_ROOT=""

# Judge (text-only LM for verdicts; when set, runs after inference)
INF_JUDGE_MODEL=""
INF_JUDGE_MAX_MODEL_LEN="2048"
INF_JUDGE_MAX_NUM_SEQS="1024"
INF_JUDGE_GPU_UTIL="0.92"

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
INF_RESTART_EVERY="0"

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
            --output-dir)      INF_OUTPUT_DIR="$2"; shift 2 ;;
            --image-root)      INF_IMAGE_ROOT="$2"; shift 2 ;;
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
            --restart-every)   INF_RESTART_EVERY="$2"; shift 2 ;;
            --scoring-measure) SCORING_MEASURE="$2"; shift 2 ;;
            --scoring-workers) INF_SCORING_WORKERS="$2"; shift 2 ;;
            --judge-model)     INF_JUDGE_MODEL="$2"; shift 2 ;;
            --judge-max-model-len) INF_JUDGE_MAX_MODEL_LEN="$2"; shift 2 ;;
            --judge-max-num-seqs)  INF_JUDGE_MAX_NUM_SEQS="$2"; shift 2 ;;
            --judge-gpu-util)   INF_JUDGE_GPU_UTIL="$2"; shift 2 ;;
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

    # Build output-dir flag
    OUTPUT_DIR_FLAG=""
    if [[ -n "$INF_OUTPUT_DIR" ]]; then
        OUTPUT_DIR_FLAG="--output-dir $INF_OUTPUT_DIR"
    fi

    # Build image-root flag
    IMAGE_ROOT_FLAG=""
    if [[ -n "$INF_IMAGE_ROOT" ]]; then
        IMAGE_ROOT_FLAG="--image-root $INF_IMAGE_ROOT"
    fi

    # Build method-specific flags
    METHOD_FLAGS=""
    case "$INF_METHOD" in
        naive-sampling)
            METHOD_FLAGS="--samples-per-example $INF_SAMPLES_PER_EXAMPLE" ;;
        iterative)
            METHOD_FLAGS="--attempts-per-round $INF_ATTEMPTS_PER_ROUND --max-rounds $INF_MAX_ROUNDS --enable-feedback $INF_FEEDBACK --max-feedback-chars $INF_MAX_FEEDBACK_CHARS" ;;
    esac

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
    echo "  Judge model:  ${INF_JUDGE_MODEL:-none}  (max_len=$INF_JUDGE_MAX_MODEL_LEN, seqs=$INF_JUDGE_MAX_NUM_SEQS, gpu=$INF_JUDGE_GPU_UTIL)"
    echo "  Score with:   $SCORING_MEASURE  (workers=${INF_SCORING_WORKERS:-0})"
    echo ""

    if [[ -n "$INF_JUDGE_MODEL" ]]; then
        # ═══════════════════════════════════════════════════════════════
        # Two-job pipeline: Job 1 (inference) → Job 2 (judge + scoring)
        #
        # Temp files avoid heredocs-inside-$(), which triggers a bash
        # parser bug that corrupts backslash line continuations.
        # ═══════════════════════════════════════════════════════════════

        MODEL_SLUG="$(echo "$INF_MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')"
        RUN_ID="$(date +%Y%m%d_%H%M%S)_$(printf '%06d' $((RANDOM * RANDOM % 1000000)))"
        OUTPUT_DIR="logs/schedule/oven_${INF_METHOD}_${INF_PROMPT}/${MODEL_SLUG}/${RUN_ID}"
        mkdir -p "$OUTPUT_DIR"

        TMPDIR=$(mktemp -d -p "$OUTPUT_DIR" .sbatch.XXXXXX)
        trap "rm -rf $TMPDIR" EXIT

        # ── Job 1: Inference ──────────────────────────────────────
        cat > "$TMPDIR/job1.sh" << JOB1EOF
#!/bin/bash
#SBATCH --job-name=${SLURM_NAME}-inf
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
trap 'kill 0' EXIT
cd "\$SLURM_SUBMIT_DIR"
module load nvhpc/24.5 gcc/12.2.0
export CC=gcc CXX=g++ OMP_NUM_THREADS=1
if [[ -f ".env" ]]; then set -a; source .env; set +a; fi
source .venv/bin/activate

VENV_CHECK_OUT=\$(python -c "import vllm; print('ok')" 2>&1) || VENV_RC=\$?
VENV_RC=\${VENV_RC:-0}
if echo "\${VENV_CHECK_OUT}" | grep -q '^ok$'; then
    echo "[info] venv OK on \$(hostname)"
else
    if [ "\${VENV_RC}" -ge 128 ]; then
        echo "[fatal] venv health check was killed by signal \$(( VENV_RC - 128 )) on \$(hostname)." >&2
        echo "  Check your --mem setting (current: $SLURM_MEM)." >&2
    else
        echo "[fatal] venv broken on \$(hostname); resync on a login node:" >&2
        echo "    uv sync --extra serve" >&2
    fi
    exit 1
fi

echo "[info] Job 1 (inference): method=$INF_METHOD prompt=$INF_PROMPT temp=$INF_TEMPERATURE"
echo "[info] Output dir: $OUTPUT_DIR"

if [[ "$INF_DP" -gt 1 ]]; then
    IFS=',' read -ra ALLOC_GPUS <<< "\${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

    run_shard() {
        local i="\$1" devs="\$2"
        CUDA_VISIBLE_DEVICES="\$devs" python -u -m scripts.run_inference \\
            --input "$INF_INPUT" \\
            --model "$INF_MODEL" \\
            --prompt-variant "$INF_PROMPT" \\
            --method "$INF_METHOD" \\
            --output-dir "$OUTPUT_DIR" \\
            --shard "\$i" --num-shards $INF_DP \\
            --temperature "$INF_TEMPERATURE" \\
            --top-p "$INF_TOP_P" \\
            --top-k "$INF_TOP_K" \\
            --max-tokens "$INF_MAX_TOKENS" \\
            --tp $INF_TP \\
            --gpu-util "$INF_GPU_UTIL" \\
            --max-model-len "$INF_MAX_MODEL_LEN" \\
            --max-num-seqs "$INF_MAX_NUM_SEQS" \\
            --max-pixels "$INF_MAX_PIXELS" \\
            --min-pixels "$INF_MIN_PIXELS" \\
            --chunk-size "$INF_CHUNK_SIZE" \\
            $ENFORCE_EAGER_FLAG \\
            $BASE_MODEL_FLAG \\
            $IMAGE_ROOT_FLAG \\
            $METHOD_FLAGS \\
            $MAX_EXAMPLES_FLAG \\
            $RESUME_FLAG \\
            --restart-every $INF_RESTART_EVERY \\
            2>&1 | stdbuf -oL sed "s/^/[shard \$i] /" | tee "${OUTPUT_DIR}/shard\${i}.log"
        return "\${PIPESTATUS[0]}"
    }

    need=\$(( $INF_DP * $INF_TP ))
    if [[ \${need} -gt \${#ALLOC_GPUS[@]} ]]; then
        echo "[error] --dp $INF_DP × --tp $INF_TP = \${need} GPUs, only \${#ALLOC_GPUS[@]} allocated" >&2
        exit 1
    fi

    pids=()
    for i in \$(seq 0 \$(($INF_DP - 1))); do
        devs=\$(IFS=,; echo "\${ALLOC_GPUS[*]:\$((i * $INF_TP)):$INF_TP}")
        echo "[info] launching shard \$i on GPUs \${devs} (TP=$INF_TP)"
        run_shard "\$i" "\${devs}" &
        pids+=(\$!)
    done

    fail=0
    for pid in "\${pids[@]}"; do
        wait "\$pid" || fail=1
    done
    if [[ \$fail -ne 0 ]]; then
        echo "[error] a shard failed — see ${OUTPUT_DIR}/shard*.log" >&2
        tail -n 30 "${OUTPUT_DIR}"/shard*.log >&2 || true
        exit 1
    fi

    cat "${OUTPUT_DIR}"/*_samples_shard*.jsonl > "${OUTPUT_DIR}/${RUN_ID}_samples.jsonl"
    echo "[info] Merged \$(wc -l < "${OUTPUT_DIR}/${RUN_ID}_samples.jsonl") samples"
else
    python -m scripts.run_inference \\
        --input "$INF_INPUT" \\
        --model "$INF_MODEL" \\
        --prompt-variant "$INF_PROMPT" \\
        --method "$INF_METHOD" \\
        --output-dir "$OUTPUT_DIR" \\
        --temperature "$INF_TEMPERATURE" \\
        --top-p "$INF_TOP_P" \\
        --top-k "$INF_TOP_K" \\
        --max-tokens "$INF_MAX_TOKENS" \\
        --tp "$INF_TP" \\
        --gpu-util "$INF_GPU_UTIL" \\
        --max-model-len "$INF_MAX_MODEL_LEN" \\
        --max-num-seqs "$INF_MAX_NUM_SEQS" \\
        --max-pixels "$INF_MAX_PIXELS" \\
        --min-pixels "$INF_MIN_PIXELS" \\
        --chunk-size "$INF_CHUNK_SIZE" \\
        $ENFORCE_EAGER_FLAG \\
        $BASE_MODEL_FLAG \\
        $IMAGE_ROOT_FLAG \\
        $METHOD_FLAGS \\
        $MAX_EXAMPLES_FLAG \\
        $RESUME_FLAG \\
        --restart-every $INF_RESTART_EVERY
fi
JOB1EOF

        # ── Job 2: Judge + Scoring (1 GPU) ──────────────────────
        cat > "$TMPDIR/job2.sh" << JOB2EOF
#!/bin/bash
#SBATCH --job-name=${SLURM_NAME}-judge
#SBATCH --output=$SLURM_OUTPUT
#SBATCH --error=$SLURM_ERROR
#SBATCH --partition=$SLURM_PARTITION
$SLURM_ACCOUNT_DIRECTIVE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$SLURM_CPUS
#SBATCH --gres=gpu:1
#SBATCH --mem=$SLURM_MEM
#SBATCH --time=$SLURM_TIME

set -euo pipefail
trap 'kill 0' EXIT
cd "\$SLURM_SUBMIT_DIR"
module load nvhpc/24.5 gcc/12.2.0
export CC=gcc CXX=g++ OMP_NUM_THREADS=1
if [[ -f ".env" ]]; then set -a; source .env; set +a; fi
source .venv/bin/activate

echo "[info] Job 2 (judge+score)"

SAMPLES="${OUTPUT_DIR}/${RUN_ID}_samples.jsonl"
if [[ ! -s "\${SAMPLES}" ]]; then
    echo "[error] Samples file not found: \${SAMPLES}" >&2
    exit 1
fi

echo "[info] Judging \${SAMPLES}..."
JUDGED="${OUTPUT_DIR}/${RUN_ID}_judged.jsonl"
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_judge \\
    --input "\${SAMPLES}" \\
    --output "\${JUDGED}" \\
    --judge-model "$INF_JUDGE_MODEL" \\
    --max-model-len "$INF_JUDGE_MAX_MODEL_LEN" \\
    --max-num-seqs "$INF_JUDGE_MAX_NUM_SEQS" \\
    --gpu-util "$INF_JUDGE_GPU_UTIL"

echo "[info] Scoring \${JUDGED}..."
SCORED="${OUTPUT_DIR}/${RUN_ID}_scored.jsonl"
python -m scripts.score_predictions \\
    --input "\${JUDGED}" \\
    --output "\${SCORED}" \\
    --taxonomy-index "$INF_TAXONOMY_INDEX" \\
    --measure $SCORING_MEASURE \\
    --num-workers $INF_SCORING_WORKERS

echo "[info] Done. Output: ${OUTPUT_DIR}"
JOB2EOF

        # ── Submit both jobs ────────────────────────────────────
        JOB1_ID=$(sbatch --parsable "$TMPDIR/job1.sh")
        if [[ -z "$JOB1_ID" ]]; then
            echo "[error] Failed to submit Job 1 (inference)" >&2
            exit 1
        fi
        echo "[info] Job 1 (inference): $JOB1_ID"

        JOB2_ID=$(sbatch --parsable --dependency=afterok:$JOB1_ID "$TMPDIR/job2.sh")
        echo "[info] Job 2 (judge+score): $JOB2_ID  (waits for $JOB1_ID)"
        echo ""
        echo "  Output: $OUTPUT_DIR"
    else
        # ── Single-job pipeline (no judge) ───────────────────────────
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
echo "[info] Running inference (DP=$INF_DP): method=$INF_METHOD prompt=$INF_PROMPT temp=$INF_TEMPERATURE"

MODEL_SLUG=\$(echo "$INF_MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')

if [[ "$INF_DP" -gt 1 ]]; then
    # ── Multi-process data-parallel ──────────────────────────────────
    # One independent single-GPU process per shard.  Each process slices
    # the dataset via strided sharding and writes its own samples file.
    # We merge them afterwards into the canonical samples file.

    RUN_ID="\$(date +%Y%m%d_%H%M%S)_\$(printf '%06d' \$((RANDOM * RANDOM % 1000000)))"
    OUTPUT_DIR="logs/schedule/oven_${INF_METHOD}_${INF_PROMPT}/\${MODEL_SLUG}/\${RUN_ID}"
    mkdir -p "\${OUTPUT_DIR}"
    echo "[info] Output dir: \${OUTPUT_DIR}"

    # Index into whatever GPUs SLURM actually gave us
    IFS=',' read -ra ALLOC_GPUS <<< "\${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

    run_shard() {
        local i="\$1" devs="\$2"
        CUDA_VISIBLE_DEVICES="\$devs" python -u -m scripts.run_inference \\
            --input "$INF_INPUT" \\
            --model "$INF_MODEL" \\
            --prompt-variant "$INF_PROMPT" \\
            --method "$INF_METHOD" \\
            --output-dir "\${OUTPUT_DIR}" \\
            --shard "\$i" --num-shards $INF_DP \\
            --temperature "$INF_TEMPERATURE" \\
            --top-p "$INF_TOP_P" \\
            --top-k "$INF_TOP_K" \\
            --max-tokens "$INF_MAX_TOKENS" \\
            --tp $INF_TP \\
            --gpu-util "$INF_GPU_UTIL" \\
            --max-model-len "$INF_MAX_MODEL_LEN" \\
            --max-num-seqs "$INF_MAX_NUM_SEQS" \\
            --max-pixels "$INF_MAX_PIXELS" \\
            --min-pixels "$INF_MIN_PIXELS" \\
            --chunk-size "$INF_CHUNK_SIZE" \\
            $ENFORCE_EAGER_FLAG \\
            $BASE_MODEL_FLAG \\
            $IMAGE_ROOT_FLAG \\
            \$([ "$INF_METHOD" = "naive-sampling" ] && echo "--samples-per-example $INF_SAMPLES_PER_EXAMPLE") \\
            \$([ "$INF_METHOD" = "iterative" ] && echo "--attempts-per-round $INF_ATTEMPTS_PER_ROUND --max-rounds $INF_MAX_ROUNDS --enable-feedback $INF_FEEDBACK --max-feedback-chars $INF_MAX_FEEDBACK_CHARS") \\
            $MAX_EXAMPLES_FLAG \\
            $RESUME_FLAG \\
            --restart-every $INF_RESTART_EVERY \\
            2>&1 | stdbuf -oL sed "s/^/[shard \$i] /" | tee "\${OUTPUT_DIR}/shard\${i}.log"
        return "\${PIPESTATUS[0]}"
    }

    need=\$(( $INF_DP * $INF_TP ))
    if [[ \${need} -gt \${#ALLOC_GPUS[@]} ]]; then
        echo "[error] --dp $INF_DP × --tp $INF_TP = \${need} GPUs, only \${#ALLOC_GPUS[@]} allocated" >&2
        exit 1
    fi

    pids=()
    for i in \$(seq 0 \$(($INF_DP - 1))); do
        devs=\$(IFS=,; echo "\${ALLOC_GPUS[*]:\$((i * $INF_TP)):$INF_TP}")
        echo "[info] launching shard \$i on GPUs \${devs} (TP=$INF_TP)"
        run_shard "\$i" "\${devs}" &
        pids+=(\$!)
    done

    # Wait for all shards; fail loudly if any shard dies
    fail=0
    for pid in "\${pids[@]}"; do
        wait "\$pid" || fail=1
    done
    if [[ \$fail -ne 0 ]]; then
        echo "[error] a shard failed — see \${OUTPUT_DIR}/shard*.log" >&2
        tail -n 30 "\${OUTPUT_DIR}"/shard*.log >&2 || true
        exit 1
    fi

    # Merge shard outputs into the canonical samples file
    SAMPLES="\${OUTPUT_DIR}/\${RUN_ID}_samples.jsonl"
    cat "\${OUTPUT_DIR}/\${RUN_ID}_samples_shard"*.jsonl > "\${SAMPLES}"
    echo "[info] Merged \$(wc -l < "\${SAMPLES}") samples into \${SAMPLES}"
else
    # ── Single-process path (honours --tp for large models) ─────────
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
        --gpu-util "$INF_GPU_UTIL" \\
        --max-model-len "$INF_MAX_MODEL_LEN" \\
        --max-num-seqs "$INF_MAX_NUM_SEQS" \\
        --max-pixels "$INF_MAX_PIXELS" \\
        --min-pixels "$INF_MIN_PIXELS" \\
        --chunk-size "$INF_CHUNK_SIZE" \\
        $ENFORCE_EAGER_FLAG \\
        $BASE_MODEL_FLAG \\
        $OUTPUT_DIR_FLAG \\
        \$([ "$INF_METHOD" = "naive-sampling" ] && echo "--samples-per-example $INF_SAMPLES_PER_EXAMPLE") \\
        \$([ "$INF_METHOD" = "iterative" ] && echo "--attempts-per-round $INF_ATTEMPTS_PER_ROUND --max-rounds $INF_MAX_ROUNDS --enable-feedback $INF_FEEDBACK --max-feedback-chars $INF_MAX_FEEDBACK_CHARS") \\
        $MAX_EXAMPLES_FLAG \\
        $RESUME_FLAG \\
        --restart-every $INF_RESTART_EVERY

    EXPERIMENT_DIR="logs/schedule/oven_${INF_METHOD}_${INF_PROMPT}/\${MODEL_SLUG}"
    OUTPUT_DIR=\$(ls -1d "\${EXPERIMENT_DIR}"/20* 2>/dev/null | sort | tail -1)
    SAMPLES=\$(ls -1 "\${OUTPUT_DIR}"/*_samples.jsonl 2>/dev/null | head -1)
fi

# -----------------------------------------------------------------------
# Score
# -----------------------------------------------------------------------
if [[ -z "\${SAMPLES:-}" || ! -s "\${SAMPLES}" ]]; then
    echo "[error] No samples to score" >&2
    exit 1
fi
echo "[info] Scoring \${SAMPLES}..."
RUN_ID=\$(basename "\${SAMPLES}" _samples.jsonl)
SCORED_OUT="\${OUTPUT_DIR}/\${RUN_ID}_scored.jsonl"
python -m scripts.score_predictions \\
    --input "\${SAMPLES}" \\
    --output "\${SCORED_OUT}" \\
    --taxonomy-index "$INF_TAXONOMY_INDEX" \\
    --measure $SCORING_MEASURE \\
    --num-workers $INF_SCORING_WORKERS

echo "[info] Done."
EOT
    fi
}

main "$@"
