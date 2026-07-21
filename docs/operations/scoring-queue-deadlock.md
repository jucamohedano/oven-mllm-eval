# Multiprocessing queue deadlock in scoring

## Symptom

`python -m scripts.score_predictions` hangs with all worker processes at 100% but
`pool.map().get()` never returns. CPU is idle, no disk activity, no error — just a
silent hang that can last hours.

The progress output shows all workers finished:

```
[scoring] 7888/7888 (100.0%)
[scoring] 7888/7888 (100.0%)
[scoring] 7888/7888 (100.0%)
...
```

But the main process never proceeds past `pool.map()`.

## Environment

- Python 3.12
- Linux (Ubuntu), local machine with 16-32 cores
- `multiprocessing.Pool.map()` with `num_workers=0` (auto = all CPUs)
- Input: 47,347 rows × ~15KB each (judge output from 256 stochastic rollouts)
- Each worker scores a contiguous chunk of ~7,888 rows

## Root cause

Each scored row contains a `*_scores` field: a dict mapping every taxonomy node
label to its match score. The OVEN taxonomy has **~12,000 nodes**. So each row
carries a `{"exact_match_scores": {node_label: score, ...}}` dict with ~12K
float entries, roughly **260 KB per row**.

For one worker with 7,888 rows:

```
7,888 rows × 260 KB = ~2 GB of score data per worker
```

This is returned through `pool.map()`'s internal result queue. Python's
`multiprocessing.Pool` uses a bounded `os.pipe()` for passing results back to the
main process. When the pipe buffer fills up (~64 KB on Linux), the worker blocks
on `write()`. The main process is waiting for all workers to signal done, creating
a **circular deadlock**: workers can't finish because the pipe is full, the main
process can't drain the pipe because it's waiting for workers to finish.

## Attempted fixes

### 1. Temp files (rejected)

Workers wrote scored rows to temp JSONL files and returned only file paths.
This avoided the queue but introduced filesystem issues:
- Temp files on `/tmp` were slow or unavailable
- Concurrent writes by 16 workers contended for disk
- Cleanup complexity (`shutil.rmtree` in `finally`)

### 2. Serial fallback (rejected)

`--num-workers 1` avoids the queue entirely but takes ~20 minutes for 47K rows.
Not acceptable for routine use.

## Current solution (implemented)

Strip the `*_scores` field from each row before the worker returns it.
This is a **3-line change** in `src/oven_mllm_eval/scoring.py`:

```python
# In _score_rows(), just before return:
for row in scored_rows:
    for key in list(row):
        if key.endswith("_scores"):
            del row[key]
```

### Why this is correct

The `_scores` field is diagnostic — it records the matcher's score against every
taxonomy node for debugging. None of the downstream logic reads it:

| Consumer | Uses `_scores`? |
|----------|-----------------|
| Aggregate metrics (hP/hR/hF) | No — reads `accum` dicts, not rows |
| pass@k computation | No — reads `judge_verdicts` |
| Output JSONL | No — `_scores` field simply absent |
| `_results.json` | No — summary metrics only |

### Effect on queue size

Without `_scores`: ~15 KB per row → ~120 MB per worker → pipe handles it easily.

## Configuration

No changes needed. The strip happens transparently inside `_score_rows()`.

Workers are auto-detected via `os.sched_getaffinity(0)` when `--num-workers 0` (default).
To force serial: `--num-workers 1`.

## Files involved

| File | Role |
|------|------|
| `src/oven_mllm_eval/scoring.py` | `_score_rows()` (worker), `score_generation_file()` (orchestrator) |
| `src/oven_mllm_eval/measures.py` | `DirectMeasureMatcher.evaluate()` — produces the `_scores` dict |
| `scripts/score_predictions.py` | CLI entry point, passes `--num-workers` through |
