# Key Infrastructure Fixes

> Last updated: 2026-06-18

## 1. EXIT trap corrupted exit codes

**What we found.** `trap 'kill 0' EXIT` in Job 1's bash heredoc sends SIGTERM to the shell's own process group, corrupting the exit code to 143 (128+SIGTERM). The Slurm `--dependency=afterok:JOB1_ID` on Job 2 becomes permanently unsatisfiable — Job 2 silently never runs, and inference completes but judge+scoring never executes.

**How we discovered it.** Judge+scoring jobs that were submitted alongside inference jobs never appeared in `squeue`. Checking `sacct -j <JOB1_ID>` showed exit code 143. Tracing the heredoc found `trap 'kill 0' EXIT` without signal shielding.

**Fix.** Added `trap '' TERM` before `kill 0` to make the shell process deaf to its own SIGTERM signal, preserving the original exit code. Applied to all four trap locations (Job 1, Job 2, single-job pipeline, launcher script).

**Code modified:** `scripts/schedule_sbatch.sh`.

## 2. Shard file data loss on cleanup

**What we found.** The post-scoring cleanup step deleted intermediate `_samples_shard*.jsonl` and `_judged_shard*.jsonl` files before verifying the merged output was complete. When the 4B inference job crashed before merging, the cleanup ran on a partial 14,296-example scored file and deleted the shard files containing 101,256 additional examples — the data was permanently lost (only recovered because we had a local `sync.sh` snapshot from before the cleanup).

**How we discovered it.** The 4B `no_idk` run's `_scored.jsonl` and `_samples.jsonl` both had 14,296 lines, but the shard files had 101,256 lines total. The merge never happened because all DP=2/DP=4 attempts crashed before reaching the `cat` merge step, and the DP=1 resume wrote directly to `_samples.jsonl` without merging shards.

**Fix.**
1. Job 2 now merges any remaining inference shard files before running the judge (`cat _samples_shard* _samples.jsonl > merged`).
2. Cleanup only deletes shards if the merged `_samples.jsonl` is non-empty AND the judge shard count matches the judge merged count.
3. Both the two-job and single-job pipelines have identical cleanup guards.
4. Cleanup patterns use explicit prefixes (`_samples_shard*`, `_judged_*_shard*`) instead of the aggressive `*_shard*`.

**Code modified:** `scripts/schedule_sbatch.sh`.

## 3. Multi-GPU deadlock on 32B model

**What we found.** Running two TP=2 replicas (DP=2) on a single 4-GPU node causes NCCL deadlock during CUDA graph capture. Both processes compete for the same NCCL resources, and one or both hang during `Capturing CUDA graphs`. Logs show `enforce_eager: False` and `cudagraph_mode: FULL_AND_PIECEWISE`. A single TP=4 replica (DP=1) works correctly. Eager mode (`--enforce-eager`) bypasses the issue but reduces throughput by ~20%.

**How we discovered it.** The 32B model consistently froze on chunk 1 of inference with `--tp 2 --dp 2`. `nvidia-smi` showed two GPUs with models loaded but zero utilization. Engine debug logs showed successful model loading and graph capture, but shard logs stopped at `[chunk 1/108]` with no completion.

**Fix.** For 32B, use `--tp 4 --dp 1` (single replica, tensor parallelism across all 4 GPUs). For smaller models where TP=1 DP=4 works, no fix needed.

**Code modified:** `--enforce-eager` flag forwarding added to `scripts/schedule_sbatch.sh`; `scripts/latest_run.sh` auto-selects TP=4 for 32B.

## 4. Judge output auto-naming

**What we found.** Running two different judge models (e.g., Qwen3-4B and Qwen3-8B) on the same inference samples would overwrite each other's output because both wrote to `_judged.jsonl`. Similarly, `generations_results.json` was overwritten.

**How we discovered it.** We ran `schedule_scoring.sh` with `--judge-model Qwen/Qwen3-8B` on samples already judged by Qwen3-4B. The existing `_judged.jsonl` was overwritten without warning.

**Fix.**
1. Judge output auto-derived from model name: `_judged_{model-slug}.jsonl` (e.g., `_judged_qwen_qwen3-4b.jsonl`).
2. Results similarly auto-named: `_results_{model-slug}.json`.
3. `plot_pass_at_k.py` updated to glob `*_results*.json` patterns.
4. `scoring.py` includes `judge_model` and `judge_mode` in the results JSON.

**Code modified:** `scripts/schedule_sbatch.sh`, `scripts/schedule_scoring.sh`, `src/oven_mllm_eval/scoring.py`, `analysis/plot_pass_at_k.py`.

## 5. Global inference resume

**What we found.** When resuming a crashed inference run, each shard only read its own shard file for already-completed data IDs. If shard 2 crashed at chunk 8/113 while shards 0, 1, and 3 completed, the resume couldn't redistribute the remaining work across GPUs — shard 2 had to process 105 remaining chunks alone while the other 3 GPUs sat idle.

**How we discovered it.** After a DP=4 4B run crashed, resuming with `--resume` showed shards 0,1,3 with "0 remaining, exiting" and shard 2 with "27,096 remaining." The other 3 GPUs initialized vLLM (~30s) only to exit immediately.

**Fix.**
1. On resume, each shard reads ALL `_samples_shard*.jsonl` files (not just its own) to build a global done set.
2. Shards with zero remaining examples exit early before LLM initialization.
3. The global done set also reads the merged `_samples.jsonl` for compatibility with DP changes between resume attempts.

**Code modified:** `scripts/run_inference.py`.

## 6. Judge resume from merged output

**What we found.** The judge resume only read per-shard files (`_judged_shard*.jsonl`) for done IDs. When shard files were deleted by cleanup (see fix #2), the merged `_judged.jsonl` was never consulted, forcing re-judging of all examples on resume.

**How we discovered it.** After recovering shard files for the 4B run via `sync.sh --push`, we tried to re-judge only the remaining examples. The judge started from scratch because it couldn't find the already-completed 14,296 examples in the deleted shard files.

**Fix.** Judge resume now also reads the non-sharded merged output (`_judged.jsonl` and `_judged_{slug}.jsonl`) for done IDs.

**Code modified:** `scripts/run_judge.py`.

## 7. Job 2 `OUTPUT_DIR` unbound variable

**What we found.** The Job 2 heredoc uses escaped `\${OUTPUT_DIR}` references in merge and cleanup blocks, which run on the compute node. But `OUTPUT_DIR` was never assigned within the Job 2 script — it was only expanded inline by the launcher in earlier lines. When reached, `set -o nounset` kills the script with "unbound variable."

**How we discovered it.** The judge job failed immediately with `line 29: OUTPUT_DIR: unbound variable` in `logs/slurm/46900297.err`. The merge-before-judge and cleanup verification blocks added `\$OUTPUT_DIR` references without adding the corresponding variable assignment.

**Fix.** Added `OUTPUT_DIR="${OUTPUT_DIR}"` and `RUN_ID="${RUN_ID}"` assignments at the top of the Job 2 heredoc (expanded by the launcher at submit time).

**Code modified:** `scripts/schedule_sbatch.sh`.

## 8. `_JUDGE_SLUG` unbound variable

**What we found.** The Job 2 heredoc referenced `${_JUDGE_SLUG}` in the `JUDGED=...` line, but `_JUDGE_SLUG` was defined inside the same heredoc at a later line. Since the heredoc is unquoted (`<< JOB2EOF`), bash expands all variables at write time, but `_JUDGE_SLUG` wasn't defined yet in the launcher scope.

**How we discovered it.** The launcher itself failed with `line 516: _JUDGE_SLUG: unbound variable` before even submitting the job.

**Fix.** Moved `_JUDGE_SLUG` computation to the launcher scope, immediately before the heredoc starts, so all `\${_JUDGE_SLUG}` expansions resolve correctly.

**Code modified:** `scripts/schedule_sbatch.sh`.
