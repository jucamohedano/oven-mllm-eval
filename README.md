<div align="center">

<a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/Python_3.10--3.12-blue?logo=python&logoColor=white"></a>
<a href="https://github.com/vllm-project/vllm"><img alt="vLLM" src="https://img.shields.io/badge/vLLM-offline_inference-ee4c2c"></a>
<a href="https://www.unitn.it"><img alt="University of Trento" src="https://img.shields.io/badge/MSc_Thesis-University_of_Trento-4b44ce"></a>

<h1>Taxonomy-Aware Evaluation of Multimodal LLMs</h1>

Open-domain visual entity recognition on **OVEN**, measured with a specificity-preserving taxonomy.

Juan Camacho Mohedano · University of Trento

</div>

______________________________________________________________________

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Setup](#setup)
  - [Install](#install)
  - [Data](#data)
- [Usage](#usage)
  - [Entrypoints](#entrypoints)
  - [Running the pipeline](#running-the-pipeline)
  - [Running on the cluster](#running-on-the-cluster)
  - [Analysis and figures](#analysis-and-figures)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

______________________________________________________________________

## Overview

Open-world classification is ill-posed: a Golden Retriever is correctly "animal", "dog",
"retriever", or "Golden Retriever", yet flat accuracy picks one label and marks the rest
wrong. This repository **measures** that ambiguity rather than optimising against it,
evaluating Qwen3-VL (2B/4B/8B/32B) on OVEN with hierarchical metrics derived from the
Wikidata **P279** ancestor chain.

Three findings drive the design:

1. **Models know more than they reliably say.** The correct entity is often present
   somewhere in a large sample but seldom the first answer, i.e. a gap between *coverage*
   (pass@k at large k) and *reliability* (pass@1).
2. **The instruments are not neutral.** Under a permissive LM judge the 2B model appears
   to cover more of the visual world than larger ones. Under a specificity-preserving
   audit that advantage disappears and a clean bigger-is-better ordering returns.
3. **Taxonomy-aware reasoning can be prompted for, not cheaply trained in.** Recursive
   self-aggregation recombines existing knowledge without creating new correct answers,
   and outcome-only GRPO mainly sharpens the format the prompt already produces.

## Pipeline

Three sequential phases, each a CLI reading and writing JSONL:

1. **Inference** — `scripts/run_inference.py` draws N stochastic rollouts per example via
   vLLM's offline `LLM.chat()` (no server). Writes `*_samples.jsonl`.
2. **Judge** — `scripts/run_judge.py` has a text-only LM verify each rollout against the
   ground truth, giving a binary verdict per rollout. Writes `*_judged.jsonl`.
3. **Scoring** — `scripts/score_predictions.py` maps predictions onto the taxonomy and
   computes hierarchical hP/hR/hF plus pass@k. Writes `*_scored.jsonl` + `*_results.json`.

> **Two notions of "correct" coexist and the distinction carries weight.**
> The **judge verdict** is permissive, free-form, and feeds pass@k. **Deterministic
> taxonomy/string matching** feeds hF and the support audit. Never silently swap one for
> the other.

Runs are written to timestamped directories that never overwrite:

```
logs/schedule/oven_<method>_<prompt>/<model>/<YYYYMMDD_HHMMSS_RAND>/
├── <run_id>_samples.jsonl    raw rollouts
├── <run_id>_judged.jsonl     judge verdicts
├── <run_id>_scored.jsonl     per-example metrics
├── <run_id>_results.json     aggregate metrics
└── *_metadata.json           per-phase config (model, sampling, judge, data)
```

______________________________________________________________________

## Setup

### Install

```bash
git clone https://github.com/jucamohedano/oven-mllm-eval
cd oven-mllm-eval

curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

Optional extras, installed as needed:

| Extra | Purpose |
| ----- | ------- |
| `serve` | vLLM inference (cluster) |
| `build-index` | networkx, only to rebuild the taxonomy index |
| `analysis` | pandas/matplotlib/seaborn for plots |
| `embed` | sentence-transformers for the `cascade` measure |
| `dev` | pytest + ruff |

### Data

```bash
# OVEN validation annotations
mkdir -p data/raw
wget -P data/raw http://storage.googleapis.com/gresearch/open-vision-language/oven/oven_entity_val.jsonl
wget -P data/raw http://storage.googleapis.com/gresearch/open-vision-language/ovenid2impath.csv

# Images (HuggingFace snapshot, tens of GB)
hf download ychenNLP/oven --type dataset --local-dir data/images
```

Then bridge the schema gap and build the taxonomy index:

```bash
uv run python scripts/prepare_oven.py \
    --oven-val data/raw/oven_entity_val.jsonl \
    --id2path data/raw/ovenid2impath.csv \
    --image-root data/images \
    --output data/processed/vlm_compatible_val.jsonl

uv run --extra build-index python scripts/build_taxonomy_index.py \
    --output data/processed/oven_taxonomy_index.json
```

`prepare_oven.py` accepts `--exclude-inat` to drop the iNaturalist2017 subset. The
resulting `oven_taxonomy_index.json` holds the P279 chain per entity and is loaded at
runtime by nearly everything, so scoring needs no networkx.

______________________________________________________________________

## Usage

> **TL;DR**
>
> - Use `--help` on any script to explore its options.
> - `scripts/` runs the pipeline. `analysis/` turns its outputs into figures and tables.
> - Use `scripts/schedule_sbatch.sh` for the full GPU chain on Slurm.

### Entrypoints

| Script | Purpose |
| ------ | ------- |
| `scripts/run_inference.py` | vLLM rollouts (`naive`, `naive-sampling`, `iterative`) |
| `scripts/run_judge.py` | LM-as-judge verdict per rollout |
| `scripts/score_predictions.py` | Taxonomy mapping, hP/hR/hF, pass@k |
| `scripts/run_recursive_self_agg.py` | Recursive self-aggregation (N=16, K=4, T=5) |
| `scripts/schedule_sbatch.sh` | Submit the inference → judge → scoring chain to Slurm |
| `scripts/sync.sh` | rsync workspace to the cluster and pull results back |

### Running the pipeline

All inference is **stochastic** (`temperature=1.0`, `top_p=1.0`, `top_k=-1`) to mirror GRPO
sampling. `--max-pixels` controls Qwen-VL's dynamic resize and is the single biggest
throughput knob.

```bash
# N rollouts per example
uv run --extra serve python scripts/run_inference.py \
    --input data/processed/vlm_compatible_val.jsonl \
    --prompt-variant concise_no_idk --method naive-sampling \
    --samples-per-example 64 --max-pixels 262144

# score an existing run
uv run python scripts/score_predictions.py \
    --input logs/schedule/.../<run_id>_samples.jsonl \
    --taxonomy-index data/processed/oven_taxonomy_index.json \
    --measure exact_match cascade --num-workers 8
```

Prompt variants: `barebones`, `base_pretrained`, `concise`, `concise_no_idk`, `default`,
`specific`, `vague`.

### Running on the cluster

```bash
cp .env.example .env          # set HF_HOME to your scratch path
vim configs/sync.conf         # set the remote path
bash scripts/sync.sh          # push workspace, pull results

bash scripts/schedule_sbatch.sh \
    -A <ACCOUNT> -p boost_usr_prod -g 2 --tp 2 \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --method naive-sampling --prompt concise_no_idk \
    --samples-per-example 64
```

`scripts/schedule_scoring.sh` reruns judge + scoring for an existing run on the CPU tier.

### Analysis and figures

Everything in `analysis/` consumes run artifacts and writes figures/tables into `viz/`:

```bash
uv run --extra analysis python analysis/plot_pass_at_k.py --help
uv run --extra analysis python analysis/plot_hierarchical_metrics.py --help
uv run streamlit run analysis/explore_judgments.py -- --scored <path>/<run_id>_scored.jsonl
```

______________________________________________________________________

## Repository layout

| Path | Contents |
| ---- | -------- |
| `scripts/` | Pipeline: inference, judge, scoring, data building, Slurm and sync helpers |
| `analysis/` | Plotting, example rendering, and audits that consume run outputs |
| `src/oven_mllm_eval/` | Library: taxonomy matching, measures, scoring, judge, pass@k |
| `data/` | `raw/` downloads, `processed/` prepared JSONL + taxonomy index, `images/` |
| `logs/schedule/` | Run outputs, one timestamped directory per run |
| `viz/` | Generated figures and tables |
| `docs/` | Methods, runbooks, findings, and research notes |

The `scripts/` vs `analysis/` split is deliberate: `scripts/` produces the canonical run
artifacts, `analysis/` only consumes them. The dependency runs one way, so analysis code
never affects evaluation results.

## Documentation

`docs/README.md` is the index.

| Directory | Content |
| --------- | ------- |
| `docs/methods/` | Pipeline and metric architecture (judge, cascade, hierarchical metrics) |
| `docs/operations/` | Executable runbooks (RSA, scoring, training data, key fixes) |
| `docs/findings/` | Empirical findings (prompt collapse, model diversity) |
| `docs/thesis_notes/` | Numbered research log (`NNN-slug.md`), with its own index |
| `docs/commands.md` | CLI and operational reference |

______________________________________________________________________

## Citation

```bibtex
@mastersthesis{camachomohedano2026taxonomy,
  title  = {Taxonomy-Aware Evaluation of Multimodal LLMs on Open-Domain
            Visual Entity Recognition},
  author = {Camacho Mohedano, Juan},
  school = {University of Trento},
  year   = {2026}
}
```

## Acknowledgements

Built on [OVEN](https://open-vision-language.github.io/oven/) for the benchmark and on
Snæbjarnarson et al. for the taxonomy-aware hierarchical measures (hP/hR/hF) that this
evaluation adapts. The scoring and matching code started from the `vlm-eval` codebase.
