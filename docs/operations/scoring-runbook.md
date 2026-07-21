# Scoring Runbook

Operational recipes for taxonomy-aware scoring. Conceptual details live in
`docs/methods/taxonomy-mapping-cascade.md`,
`docs/methods/rollout-hierarchical-metrics.md`, and thesis note `002`.

## Local smoke test

CPU is enough for a small `--max-examples` test:

```bash
uv sync --extra embed
hf download sentence-transformers/all-mpnet-base-v2

uv run python scripts/score_predictions.py \
    --input  <run>/<id>_judged.jsonl \
    --measure exact_match cascade \
    --output  <run>/<id>_scored_cascade.jsonl \
    --summary <run>/<id>_results_cascade.json \
    --embed-device cpu --num-workers 0
```

## Cluster scoring

The `cascade` measure embeds taxonomy labels and all unique predictions, so large
runs should use `scripts/schedule_scoring.sh --gpus 1`. The launcher auto-selects
the embedding device and exports `OVEN_NODE_EMB_DIR` so the node-embedding cache
persists outside FAST scratch.

```bash
bash scripts/schedule_scoring.sh -A <ACCOUNT> -p boost_usr_prod \
    -c 32 -m 128G -t 12:00:00 --gpus 1 \
    --input  <run>/<id>_judged.jsonl \
    --measure "exact_match cascade" \
    --output  <run>/<id>_scored_cascade.jsonl \
    --summary <run>/<id>_results_cascade.json \
    --embed-backend open_clip
```
