# oven-mllm-eval

Taxonomy-aware evaluation of multimodal LLMs on the OVEN dataset, using
string matching and hierarchical precision/recall/F-score.

## Setup

### 1. Install (local dev environment)

```bash
cd oven-mllm-eval
uv sync
```

For building the taxonomy index (only needed once):

```bash
uv sync --extra build-index
```

For running inference with vLLM (on the cluster):

```bash
uv sync --extra serve
```

For analysis/plotting:

```bash
uv sync --extra analysis
```

### 2. Download OVEN data

```bash
# OVEN validation annotations
mkdir -p data/raw
wget -P data/raw http://storage.googleapis.com/gresearch/open-vision-language/oven/oven_entity_val.jsonl
wget -P data/raw http://storage.googleapis.com/gresearch/open-vision-language/ovenid2impath.csv

# Prebuilt taxonomy chains (already included in the repo from vlm-eval)
# data/raw/oven_wikidata_chains_cleaned_labels.jsonl

# Alias file (from vlm-eval)
# Copy from vlm-eval/src/vlmeval/calculate_scores/data_files/wikidata/wikidb_aka_oven_sample.jsonl
# to data/raw/wikidb_aka_oven_sample.jsonl

# Images (HuggingFace snapshot, ~tens of GB)
huggingface-cli download ychenNLP/oven --repo-type dataset --local-dir data/images
```

### 3. Prepare OVEN data

Bridge the schema gap between raw OVEN downloads and what the evaluation
pipeline expects:

```bash
uv run python scripts/prepare_oven.py \
    --oven-val data/raw/oven_entity_val.jsonl \
    --id2path data/raw/ovenid2impath.csv \
    --image-root data/images \
    --output data/processed/vlm_compatible_val.jsonl \
    --exclude-inat
```

### 4. Build taxonomy index

```bash
uv run --extra build-index python scripts/build_taxonomy_index.py \
    --output data/processed/oven_taxonomy_index.json
```

This produces a precomputed JSON index that can be loaded at runtime without
networkx or datasets.

## Running inference

All inference uses **stochastic sampling** (no greedy mode). Defaults mirror
GRPO training settings: `temperature=1.0`, `top_p=1.0`, `top_k=-1`, `n=1`.

Inference uses vLLM's offline `LLM.chat()` API — no separate server process.
Images are passed as PIL objects directly (no base64 encoding). The `n`
samples per request share the prefill KV cache, so the expensive vision
encoding happens only once per example regardless of `n`.

Three methods are supported:
- **naive**: single sample per example (n=1)
- **naive-sampling**: draw N samples per example, pick the best match
- **iterative**: draw N samples per round for T rounds; if all fail,
  optionally feed back failed attempts and retry (each round is one
  batched `llm.chat()` call; prefix caching reuses the image KV across
  rounds)

The `--max-pixels` flag (default `512×512 = 262144`) controls Qwen-VL's
dynamic image resizing and is the single biggest throughput knob — lower
values mean fewer vision tokens and faster prefill with minimal accuracy
loss for image classification tasks.

Output is automatically organised under `logs/schedule/` following the
lmms-ocw convention.  Each run gets a **timestamped+randomised directory**
so that repeated runs never overwrite each other:

```
logs/schedule/oven_<method>_<prompt>/<model>/<YYYYMMDD_HHMMSS_RAND>/
├── <run_id>_samples.jsonl     raw predictions (from inference)
├── <run_id>_scored.jsonl      per-sample outputs + metrics (from scoring)
└── <run_id>_results.json      aggregate metrics (hP, hR, hF, exact)
```

```bash
# Naive (1 sample)
uv run --extra serve python scripts/run_inference.py \
    --input data/processed/vlm_compatible_val.jsonl \
    --prompt-variant barebones --method naive

# Naive stochastic sampling (n=64)
uv run --extra serve python scripts/run_inference.py \
    --input data/processed/vlm_compatible_val.jsonl \
    --prompt-variant barebones --method naive-sampling \
    --samples-per-example 64
# Override temperature / top-p / top-k (defaults: 1.0 / 1.0 / -1)
uv run --extra serve python scripts/run_inference.py \
    --input data/processed/vlm_compatible_val.jsonl \
    --method naive --temperature 0.7 --top-p 0.95

# Tune image resolution — the biggest throughput knob
uv run --extra serve python scripts/run_inference.py \
    --input data/processed/vlm_compatible_val.jsonl \
    --method naive-sampling --samples-per-example 64 \
    --max-pixels 262144 --min-pixels 65536

# Override the output directory
uv run --extra serve python scripts/run_inference.py \
    --input data/processed/vlm_compatible_val.jsonl \
    --method naive --output-dir results/my_experiment
```

### Score results

Scoring uses ``DirectMeasureMatcher`` (adapted from vlm-eval) with pluggable
measures (``exact_match``, ``contained``, or ``all``).  Pass ``--output`` to
preserve the raw predictions — otherwise the input file is overwritten in-place.

```bash
# Score with exact_match (default) — writes scored JSONL + aggregate results
uv run python scripts/score_predictions.py \
    --input logs/schedule/oven_naive_barebones/qwen_qwen3-vl-8b-instruct/20260525_111730_975668_samples.jsonl \
    --output logs/schedule/oven_naive_barebones/qwen_qwen3-vl-8b-instruct/20260525_111730_975668/20260525_111730_975668_scored.jsonl \
    --taxonomy-index data/processed/oven_taxonomy_index.json

# Multiple measures, parallel scoring across CPU cores
uv run python scripts/score_predictions.py \
    --input results/my_experiment/samples.jsonl \
    --output results/my_experiment/samples_scored.jsonl \
    --taxonomy-index data/processed/oven_taxonomy_index.json \
    --measure exact_match contained \
    --num-workers 8
```

## Running on the cluster

### 1. Configure environment

Copy the example env file and set your cluster paths:

```bash
cp .env.example .env
# Edit .env — set HF_HOME to your scratch path, etc.
```

Edit `configs/sync.conf` to set your remote path, then:

```bash
bash scripts/sync.sh
```

### 2. Submit SLURM job

**Option A:** Use the scheduler script (recommended — handles everything):

```bash
bash scripts/schedule_sbatch.sh \
    -A <YOUR_ACCOUNT> \
    -p boost_usr_prod \
    -g 2 --tp 2 \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --method naive-sampling \
    --prompt barebones \
    --temperature 1.0 \
    --samples-per-example 64 \
    --max-examples 10 \
    --max-model-len 8192
```

This runs inference via vLLM's offline `LLM.chat()` API and scores the results —
all inside one SLURM job.  Results land under
`logs/schedule/oven_<method>_<prompt>/<model>/<run_id>/` with three files:
`<run_id>_samples.jsonl` (raw predictions), `<run_id>_scored.jsonl`
(per-sample outputs + metrics), and `<run_id>_results.json` (aggregate).

Run `bash scripts/schedule_sbatch.sh --help` for all options.

**Option B:** Score an existing run on the free CPU tier (no GPU):

```bash
bash scripts/schedule_scoring.sh \
    --input logs/schedule/oven_naive-sampling_barebones/qwen_qwen3-vl-4b-instruct/20260527_180244_074760/20260527_180244_074760_samples.jsonl \
    --measure exact_match \
    --num-workers 4
```

Run `bash scripts/schedule_scoring.sh --help` for all options.

### 3. Sync results back

```bash
bash scripts/sync.sh   # sync.sh also pulls remote logs/ and results/
```

## Prompt variants

| Variant     | Description                                                         |
|-------------|---------------------------------------------------------------------|
| barebones   | Question + "Answer in the format 'A: <answer>.'"                    |
| default     | Question + no extra text / no full sentence / best guess / A:        |
| specific    | Question + "Be as specific as possible" + format instruction       |
| vague       | Question + "Aim for a simple answer as if talking to a child"      |

## Project structure

```
oven-mllm-eval/
├── pyproject.toml              # uv project config (lightweight deps)
├── configs/
│   └── sync.conf               # remote cluster paths for rsync
├── scripts/
│   ├── prepare_oven.py         # bridge OVEN schema gap
│   ├── build_taxonomy_index.py # precompute taxonomy lookup JSON
│   ├── run_inference.py        # vLLM offline inference (naive/naive-sampling/iterative)
│   ├── score_predictions.py    # score generation JSONL with hP/hR/hF
│   ├── schedule_sbatch.sh      # schedule GPU inference + scoring SLURM job
│   ├── schedule_scoring.sh     # schedule CPU-only scoring SLURM job
│   ├── sync.sh                 # rsync to/from remote cluster
│   └── visualize_taxonomy.py   # render taxonomy tree in browser
├── src/oven_mllm_eval/
│   ├── __init__.py
│   ├── taxonomy.py             # load precomputed taxonomy index
│   ├── matching.py             # multi-stage prediction → taxonomy node (TaxonomyMatcher)
│   ├── measures.py             # DirectMeasureMatcher + pluggable measures (ExactMatch, Contained)
│   ├── scoring.py              # score generation JSONL (multiprocess-capable)
│   ├── scores.py               # calc_hierarchical_metrics, normalize (pure Python, from vlm-eval)
│   ├── paths.py                # project-relative path constants (from vlm-eval)
│   ├── prompts.py              # prompt construction for Qwen3-VL
│   ├── io.py                   # JSONL I/O utilities
│   └── data/
│       ├── __init__.py
│       └── load_data.py        # load_oven() (networkx only for index building)
├── data/
│   ├── raw/                    # downloaded OVEN data
│   ├── processed/              # prepared JSONL + taxonomy index
│   └── images/                 # OVEN images (not synced — download on cluster)
├── splits/                     # OVEN train/val split files
├── lib/                        # JS/CSS deps for visualization
├── viz/                        # HTML taxonomy visualizations
├── logs/
│   ├── schedule/               # inference outputs (lmms-ocw convention)
│   │   └── oven_<method>_<prompt>/<model>/<run_id>/
│   │       ├── <run_id>_samples.jsonl
│   │       ├── <run_id>_scored.jsonl
│   │       └── <run_id>_results.json
│   └── slurm/                  # SLURM job logs
└── README.md
```
