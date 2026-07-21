# Training Data Runbook

Operational recipes for building taxonomy-aware OVEN training data. The
conceptual design lives in `docs/thesis_notes/004-taxonomy-aware-parametric-traversal-grpo.md`
and the experiment log in `docs/training/taxonomy-reasoning-training-approaches.md` §14.

## Build train-compatible OVEN rows

```bash
cd oven-mllm-eval

python scripts/prepare_oven.py \
  --oven-val ../oven_eval/oven/oven_entity_train.jsonl \
  --id2path data/raw/ovenid2impath.csv \
  --image-root /leonardo_work/EUHPC_D33_243/oven \
  --output data/processed/vlm_compatible_train.jsonl \
  --exclude-inat
```

## Align train questions to taxonomy parents

```bash
python scripts/build_aligned_questions.py \
  --input data/processed/vlm_compatible_train.jsonl \
  --chains data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
  --output data/processed/vlm_compatible_train_aligned.jsonl
```

## Regenerate train+val description chains

```bash
cat ../oven_eval/oven/oven_entity_train.jsonl \
    data/raw/oven_entity_val.jsonl \
  > /tmp/oven_entity_train_val.jsonl

python scripts/generate_oven_desc_chains.py \
  --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
  --eval-jsonl /tmp/oven_entity_train_val.jsonl \
  --output data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
  --cache data/raw/oven_wikidata_desc_cache_train_val.json \
  --chunk-size 40 \
  --sleep 0.5
```

## Reproduce coverage stats

```bash
python - <<'PY'
import json
from pathlib import Path

paths = {
    "train": Path("../oven_eval/oven/oven_entity_train.jsonl"),
    "raw_val": Path("data/raw/oven_entity_val.jsonl"),
    "val_aligned": Path("data/processed/vlm_compatible_val_aligned.jsonl"),
    "labels": Path("data/raw/oven_wikidata_chains_cleaned_labels.jsonl"),
    "descs": Path("data/raw/oven_wikidata_chains_cleaned_descs.jsonl"),
}

ids = {}
for name, path in paths.items():
    qids = set()
    rows = 0
    splits = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows += 1
            qid = row.get("id") or row.get("entity_id") or row.get("wikidata_id")
            if qid:
                qids.add(qid)
            if row.get("data_split"):
                splits[row["data_split"]] = splits.get(row["data_split"], 0) + 1
    ids[name] = qids
    print(name, rows, len(qids), splits)

for artifact in ["labels", "descs"]:
    for split in ["train", "raw_val", "val_aligned"]:
        covered = len(ids[artifact] & ids[split])
        total = len(ids[split])
        print(f"{artifact} covers {split}: {covered}/{total} ({covered / total:.1%})")
PY
```

## Balance train rows across entities

Avoids oversampling high-count entities (e.g., "shelf" with 200 images vs.
"dolmen" with 3).  Uses a deterministic hash per entity for reproducibility.

```bash
python scripts/sample_jsonl_balanced_by_key.py \
  --input data/processed/vlm_compatible_train_aligned.jsonl \
  --output data/processed/vlm_compatible_train_aligned_balanced_250k.jsonl \
  --max-rows 250000 \
  --key entity_id \
  --seed 42 \
  --shuffle-output \
  --manifest data/processed/balanced_manifest_250k.json
```

## Generate RSA solution-trace candidates (Phase 1 – greedy / T=1)

Runs a single‑pass RSA (no recursive aggregation, `--steps 1`) with greedy
decoding (`--temperature 0`) on the balanced aligned training set.  Each entity
gets one concise solution trace ending in `\boxed{answer}`.  The output JSONL
carries the traces in `rsa_initial_solutions`.

```bash
python scripts/run_recursive_self_agg.py \
  --input data/processed/vlm_compatible_train_aligned_balanced_250k.jsonl \
  --output data/processed/rsa_solution_candidates_train_250k.jsonl \
  --candidate-format solution \
  --steps 1 \
  --temperature 0 \
  --population 1 \
  --k 1 \
  --max-tokens 128 \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --image-root /leonardo_work/EUHPC_D33_243/oven \
  --enforce-eager \
  --max-model-len 2048 \
  --max-num-seqs 1024
```

## Build the VERL GRPO parquet (Phase 2)

Consumes the balanced aligned JSONL and the RSA candidate solutions.  The
`rsa_trace` mode produces **two prompt types** for the training split:

| prompt type     | probability        | what the model sees |
|-----------------|-------------------|----------------------|
| `standard`      | ~50 %             | single‑turn RSA trace prompt with `\boxed{}` format – **no candidates** |
| `aggregation`   | ~50 %             | same task, prefixed with **4 candidate solution traces** (from Phase 1) as in‑context examples |

Validation rows always get `standard` prompts.

```bash
python scripts/build_verl_oven_parquet.py \
  --input data/processed/vlm_compatible_train_aligned_balanced_250k.jsonl \
  --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
  --descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
  --output-dir data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42 \
  --dataset-mode rsa_trace \
  --candidate-solutions data/processed/rsa_solution_candidates_train_250k.jsonl \
  --aggregation-fraction 0.5 \
  --aggregation-k 4 \
  --question-policy aligned \
  --val-qid-fraction 0.02 \
  --seed 42 \
  --max-train-rows 250000 \
  --image-root /leonardo_work/EUHPC_D33_243/oven
```

Outputs: `train.parquet`, `val.parquet`, `manifest.json`.

### Taxonomy lineage

The parquet rows carry `extra_info.taxonomy_labels` (the full Wikidata chain
for the entity, leaf‑to‑root, from `oven_wikidata_chains_cleaned_labels.jsonl`).
The reward function `oven_boxed.py` uses `oven_taxonomy_index.json` (built from
the same chain file by `scripts/build_taxonomy_index.py`) for entity‑to‑path
lookup.  Both artifacts share identical pruning rules (`REMOVE_NODES_AFTER_AND_INCLUDING`
/ `REMOVE_AFTER` from `load_data.py`), so training and evaluation taxonomy
paths are consistent.

## Filter validation to unseen entities only

The raw val split contains both `entity_val_seen` (different images of training
entities) and `entity_val_unseen` (truly held‑out entities).  For GRPO validation,
keep only the unseen subset to measure genuine open‑world generalization.

```bash
cd /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval
python3 - <<'PY'
import pyarrow.parquet as pq
import pyarrow as pa

VAL_PQ = "data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/val.parquet"
VAL_UNSEEN_PQ = VAL_PQ.replace("val.parquet", "val_unseen.parquet")

tbl = pq.read_table(VAL_PQ)
extras = tbl["extra_info"].to_pylist()
mask = [e.get("data_split") == "entity_val_unseen" for e in extras]
filtered = tbl.filter(pa.array(mask))
pq.write_table(filtered, VAL_UNSEEN_PQ)

unseen_qids = len(set(e.get("entity_id","") for e,m in zip(extras, mask) if m))
print(f"filtered: {len(filtered)} rows, {unseen_qids} unseen QIDs (was {len(tbl)} rows)")
PY
```

Then point `VAL_FILE` at `val_unseen.parquet` when launching GRPO.

## Verify no taxonomy leakage between train and val

```bash
python3 - <<'PY'
import json, pyarrow.parquet as pq

DIR = "data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42"

# load taxonomy chains
with open("data/raw/oven_wikidata_chains_cleaned_labels.jsonl") as f:
    chains = {row["id"]: row["taxonomy"] for row in (json.loads(l) for l in f)}

# extract QIDs from parquet
train_qids = set()
val_qids   = set()
for split, path in [("train", f"{DIR}/train.parquet"), ("val", f"{DIR}/val_unseen.parquet")]:
    for info in pq.read_table(path, columns=["extra_info"]).to_pylist():
        if qid := info.get("entity_id", ""):
            (train_qids if split == "train" else val_qids).add(qid)
    print(f"{split}: {len(train_qids if split == 'train' else val_qids)} unique QIDs")

overlap = train_qids & val_qids
print(f"direct overlap: {len(overlap)}")
if overlap:
    print(f"  OVERLAPPING: {sorted(overlap)[:20]}")

ancestor_leak = set()
for qid in val_qids:
    for label in chains.get(qid, []):
        if label in chains:
            for tq in train_qids & set(chains[qid]):
                ancestor_leak.add((tq, qid))
print(f"train QID as val ancestor: {len(ancestor_leak)}")

print("✓ clean" if not overlap and not ancestor_leak else "✗ leakage detected")
PY
```

## v2 dataset (1-shot examples + filtered aggregation)

Adds 1-shot crossover SUV example to every prompt and filters aggregation to 1-2 correct candidates.

```bash
# GRPO-exact v2 (reasoning standard prompt)
python scripts/build_verl_oven_parquet.py \
  --input data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl \
  --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
  --descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
  --output-dir data/processed/verl_oven_rsa_v2_exact_aligned_balanced_qid_2k_seed42 \
  --dataset-mode rsa_trace \
  --candidate-solutions data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
  --aggregation-fraction 0.5 --aggregation-k 4 \
  --question-policy aligned \
  --max-train-rows 2000 --seed 42 --overwrite \
  --image-root /leonardo_work/EUHPC_D33_243/oven

# GRPO-traversal v2 (reasoning + 33% traversal prompts)
python scripts/build_verl_oven_parquet.py \
  --input data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl \
  --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
  --descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
  --output-dir data/processed/verl_oven_rsa_v2_traversal_aligned_balanced_qid_2k_seed42 \
  --dataset-mode rsa_trace \
  --candidate-solutions data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
  --aggregation-fraction 0.5 --aggregation-k 4 \
  --traversal-fraction 0.33 \
  --question-policy aligned \
  --max-train-rows 2000 --seed 42 --overwrite \
  --image-root /leonardo_work/EUHPC_D33_243/oven
```

## v3 dataset (compute-buffer standard prompt)

Uses "Think step by step" instead of "Reason carefully" for standard rows.
No traversal prompts — only standard + aggregation.

```bash
python scripts/build_verl_oven_parquet.py \
  --input data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl \
  --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
  --descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
  --output-dir data/processed/verl_oven_rsa_v3_cb_exact_aligned_balanced_qid_2k_seed42 \
  --dataset-mode rsa_trace \
  --candidate-solutions data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
  --aggregation-fraction 0.5 --aggregation-k 4 \
  --question-policy aligned \
  --max-train-rows 2000 --seed 42 --overwrite \
  --image-root /leonardo_work/EUHPC_D33_243/oven \
  --standard-prompt-variant compute_buffer

