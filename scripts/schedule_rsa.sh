#!/usr/bin/env bash
# shellcheck disable=SC2034
set -o errexit
set -o nounset
set -o pipefail

if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace
fi

if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
    echo 'usage: schedule_rsa.sh [-h] [OPTIONS]

Schedule a post-hoc Recursive Self-Aggregation (RSA) job on SLURM.

This wraps scripts/run_recursive_self_agg.py. It reads an existing
naive-sampling *_samples.jsonl with all_texts, runs RSA aggregation, and writes
a new judge-compatible RSA samples JSONL. Run scripts/schedule_scoring.sh
afterwards to judge + score the RSA output.

Slurm options:
    -p, --partition <PARTITION>   Partition to use (default: boost_usr_prod)
    -A, --account <ACCOUNT>       Account to use
    -c, --cpus <CPUS>             CPUs per task (default: 32)
    -g, --gpus <GPUS>             Number of GPUs (default: 1)
    -m, --mem <MEM>               Memory limit (default: 128G)
    -t, --time <TIME>             Time limit (default: 12:00:00)
    -n, --name <NAME>             Job name (default: oven-rsa)

RSA options:
    --input <PATH>                Input naive-sampling samples JSONL (required)
    --output <PATH>               Output RSA samples JSONL
                                  (default: <input>_rsa_nN_kK_tT.jsonl)
    --model <MODEL>               Aggregator VLM (default: Qwen/Qwen3-VL-4B-Instruct)
    --prompt-variant <VARIANT>    source, concise_no_idk, concise, ... (default: source)
    --population <N>              RSA population size N (default: 16)
    --k <K>                       Aggregation subset size K (default: 4)
    --steps <T>                   Total RSA population steps incl. input P1 (default: 2)
    --initial-selection <MODE>    first or random (default: first)
    --seed <SEED>                 Candidate sampling seed (default: 1234)
    --max-examples <N>            Limit examples (default: all)
    --resume                      Skip rows already present in output
    --overwrite                   Delete existing output before writing

Data options:
    --image-root <PATH>           Root for resolving relative image_path
    --no-image                    Text-only ablation

Sampling options:
    --temperature <TEMP>          Sampling temperature (default: 1.0)
    --top-p <P>                   Nucleus sampling threshold (default: 1.0)
    --top-k <K>                   Top-k sampling (default: -1 = disabled)
    --max-tokens <TOKENS>         Max tokens per RSA answer (default: 16)

vLLM engine options:
    --tp <N>                      Tensor parallelism (default: 1)
    --gpu-util <UTIL>             GPU memory utilization (default: 0.92)
    --max-model-len <LEN>         Max model context length (default: 2048)
    --max-num-seqs <N>            Max concurrent sequences (default: 1024)
    --max-pixels <N>              Max pixels for image resizing (default: 262144)
    --min-pixels <N>              Min pixels for image resizing (default: 65536)
    --chunk-size <N>              Examples per chunk (default: 32)
    --restart-every <N>           Restart vLLM every N chunks (default: 0 = never)
    --enforce-eager               Disable CUDA graphs
'
    exit 0
fi

cd "$(dirname "$0")"
while [ "$(find . -maxdepth 1 -name pyproject.toml | wc -l)" -ne 1 ]; do cd ..; done

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

SLURM_PARTITION="boost_usr_prod"
SLURM_ACCOUNT=""
SLURM_CPUS="32"
SLURM_GPUS="1"
SLURM_MEM="128G"
SLURM_TIME="12:00:00"
SLURM_NAME="oven-rsa"

RSA_INPUT=""
RSA_OUTPUT=""
RSA_MODEL="Qwen/Qwen3-VL-4B-Instruct"
RSA_PROMPT_VARIANT="source"
RSA_POPULATION="16"
RSA_K="4"
RSA_STEPS="2"
RSA_INITIAL_SELECTION="first"
RSA_SEED="1234"
RSA_MAX_EXAMPLES=""
RSA_RESUME="0"
RSA_OVERWRITE="0"

RSA_IMAGE_ROOT=""
RSA_NO_IMAGE="0"

RSA_TEMPERATURE="1.0"
RSA_TOP_P="1.0"
RSA_TOP_K="-1"
RSA_MAX_TOKENS="16"

RSA_TP="1"
RSA_GPU_UTIL="0.92"
RSA_MAX_MODEL_LEN="2048"
RSA_MAX_NUM_SEQS="1024"
RSA_MAX_PIXELS="262144"
RSA_MIN_PIXELS="65536"
RSA_CHUNK_SIZE="32"
RSA_RESTART_EVERY="0"
RSA_ENFORCE_EAGER="0"

main() {
    while [[ $# -gt 0 ]]; do
        if [[ $1 == "--" ]]; then
            shift
            break
        fi
        case $1 in
            -p|--partition)       SLURM_PARTITION="$2"; shift 2 ;;
            -A|--account)         SLURM_ACCOUNT="$2"; shift 2 ;;
            -c|--cpus)            SLURM_CPUS="$2"; shift 2 ;;
            -g|--gpus)            SLURM_GPUS="$2"; shift 2 ;;
            -m|--mem)             SLURM_MEM="$2"; shift 2 ;;
            -t|--time)            SLURM_TIME="$2"; shift 2 ;;
            -n|--name)            SLURM_NAME="$2"; shift 2 ;;

            --input)              RSA_INPUT="$2"; shift 2 ;;
            --output)             RSA_OUTPUT="$2"; shift 2 ;;
            --model)              RSA_MODEL="$2"; shift 2 ;;
            --prompt|--prompt-variant) RSA_PROMPT_VARIANT="$2"; shift 2 ;;
            --population)         RSA_POPULATION="$2"; shift 2 ;;
            --k)                  RSA_K="$2"; shift 2 ;;
            --steps)              RSA_STEPS="$2"; shift 2 ;;
            --initial-selection)  RSA_INITIAL_SELECTION="$2"; shift 2 ;;
            --seed)               RSA_SEED="$2"; shift 2 ;;
            --max-examples)       RSA_MAX_EXAMPLES="$2"; shift 2 ;;
            --resume)             RSA_RESUME="1"; shift ;;
            --overwrite)          RSA_OVERWRITE="1"; shift ;;

            --image-root)         RSA_IMAGE_ROOT="$2"; shift 2 ;;
            --no-image)           RSA_NO_IMAGE="1"; shift ;;

            --temperature)        RSA_TEMPERATURE="$2"; shift 2 ;;
            --top-p)              RSA_TOP_P="$2"; shift 2 ;;
            --top-k)              RSA_TOP_K="$2"; shift 2 ;;
            --max-tokens)         RSA_MAX_TOKENS="$2"; shift 2 ;;

            --tp)                 RSA_TP="$2"; shift 2 ;;
            --gpu-util)           RSA_GPU_UTIL="$2"; shift 2 ;;
            --max-model-len)      RSA_MAX_MODEL_LEN="$2"; shift 2 ;;
            --max-num-seqs)       RSA_MAX_NUM_SEQS="$2"; shift 2 ;;
            --max-pixels)         RSA_MAX_PIXELS="$2"; shift 2 ;;
            --min-pixels)         RSA_MIN_PIXELS="$2"; shift 2 ;;
            --chunk-size)         RSA_CHUNK_SIZE="$2"; shift 2 ;;
            --restart-every)      RSA_RESTART_EVERY="$2"; shift 2 ;;
            --enforce-eager)      RSA_ENFORCE_EAGER="1"; shift ;;
            *) echo "Error: unknown option: $1" >&2; exit 1 ;;
        esac
    done

    if [[ -z "$RSA_INPUT" ]]; then
        echo "[error] --input is required" >&2
        exit 1
    fi
    if [[ ! -f "$RSA_INPUT" ]]; then
        echo "[error] --input file not found: $RSA_INPUT" >&2
        exit 1
    fi
    if [[ "$SLURM_MEM" =~ ^[0-9]+$ ]]; then
        echo "[error] --mem '$SLURM_MEM' has no unit suffix. Did you mean '${SLURM_MEM}G'?" >&2
        exit 1
    fi
    if [[ "$RSA_TP" -gt "$SLURM_GPUS" ]]; then
        echo "[error] --tp ($RSA_TP) cannot exceed --gpus ($SLURM_GPUS)" >&2
        exit 1
    fi
    if [[ "$RSA_RESUME" == "1" && "$RSA_OVERWRITE" == "1" ]]; then
        echo "[error] use only one of --resume or --overwrite" >&2
        exit 1
    fi

    SLURM_ACCOUNT_DIRECTIVE=""
    if [[ -n "$SLURM_ACCOUNT" ]]; then
        SLURM_ACCOUNT_DIRECTIVE="#SBATCH --account=$SLURM_ACCOUNT"
    fi

    mkdir -p ./logs/slurm

    echo "[info] Scheduling RSA job:"
    echo "  Input:        $RSA_INPUT"
    echo "  Output:       ${RSA_OUTPUT:-<auto>}"
    echo "  Model:        $RSA_MODEL"
    echo "  RSA:          N=$RSA_POPULATION K=$RSA_K T=$RSA_STEPS"
    echo "  Max examples: ${RSA_MAX_EXAMPLES:-all}"
    echo "  Image root:   ${RSA_IMAGE_ROOT:-<cwd>}"
    echo "  Partition:    $SLURM_PARTITION"
    echo "  GPUs:         $SLURM_GPUS (TP=$RSA_TP)"
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
#SBATCH --gres=gpu:$SLURM_GPUS
#SBATCH --mem=$SLURM_MEM
#SBATCH --time=$SLURM_TIME

set -euo pipefail
trap 'kill 0' EXIT

cd "\$SLURM_SUBMIT_DIR"

module load nvhpc/24.5 gcc/12.2.0
export CC=gcc CXX=g++
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ -f ".env" ]]; then set -a; source .env; set +a; fi
source .venv/bin/activate

echo "[info] RSA job \$SLURM_JOB_ID on \$(hostname)"
echo "  Input: $RSA_INPUT"
echo "  Model: $RSA_MODEL"
echo "  RSA:   N=$RSA_POPULATION K=$RSA_K T=$RSA_STEPS"

RSA_OUTPUT="$RSA_OUTPUT"
RSA_MAX_EXAMPLES="$RSA_MAX_EXAMPLES"
RSA_IMAGE_ROOT="$RSA_IMAGE_ROOT"

RSA_CMD=(
    python scripts/run_recursive_self_agg.py
    --input "$RSA_INPUT"
    --model "$RSA_MODEL"
    --prompt-variant "$RSA_PROMPT_VARIANT"
    --population "$RSA_POPULATION"
    --k "$RSA_K"
    --steps "$RSA_STEPS"
    --initial-selection "$RSA_INITIAL_SELECTION"
    --seed "$RSA_SEED"
    --temperature "$RSA_TEMPERATURE"
    --top-p "$RSA_TOP_P"
    --top-k "$RSA_TOP_K"
    --max-tokens "$RSA_MAX_TOKENS"
    --tp "$RSA_TP"
    --gpu-util "$RSA_GPU_UTIL"
    --max-model-len "$RSA_MAX_MODEL_LEN"
    --max-num-seqs "$RSA_MAX_NUM_SEQS"
    --max-pixels "$RSA_MAX_PIXELS"
    --min-pixels "$RSA_MIN_PIXELS"
    --chunk-size "$RSA_CHUNK_SIZE"
    --restart-every "$RSA_RESTART_EVERY"
)
if [[ -n "\$RSA_OUTPUT" ]]; then RSA_CMD+=(--output "\$RSA_OUTPUT"); fi
if [[ -n "\$RSA_MAX_EXAMPLES" ]]; then RSA_CMD+=(--max-examples "\$RSA_MAX_EXAMPLES"); fi
if [[ -n "\$RSA_IMAGE_ROOT" ]]; then RSA_CMD+=(--image-root "\$RSA_IMAGE_ROOT"); fi
if [[ "$RSA_NO_IMAGE" == "1" ]]; then RSA_CMD+=(--no-image); fi
if [[ "$RSA_RESUME" == "1" ]]; then RSA_CMD+=(--resume); fi
if [[ "$RSA_OVERWRITE" == "1" ]]; then RSA_CMD+=(--overwrite); fi
if [[ "$RSA_ENFORCE_EAGER" == "1" ]]; then RSA_CMD+=(--enforce-eager); fi

"\${RSA_CMD[@]}"

echo "[info] Done."
EOT
}

main "$@"
