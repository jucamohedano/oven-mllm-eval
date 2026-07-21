# Judge and Scoring Pipeline

This document describes the current OVEN post-processing pipeline. It supersedes
the earlier judge architecture notes that described image-based judging,
`_matches_label` gates, and JSON-only judge output.

## Current pipeline

The pipeline has three separable phases:

1. **Inference** (`scripts/run_inference.py`) generates multiple free-form
   rollouts per example and writes them to `all_texts` in a samples JSONL file.
2. **Judging** (`scripts/run_judge.py`) evaluates each rollout against the ground
   truth answer and writes `judge_verdicts`, `judge_hit`, `judge_hit_count`,
   `judge_selected`, and `judge_selected_text`.
3. **Scoring** (`scripts/score_predictions.py`, `src/oven_mllm_eval/scoring.py`)
   computes lexical, pass@k, and taxonomy-aware metrics from judged rows.

`scripts/schedule_sbatch.sh` can schedule inference followed by a dependent
judge+score job. `scripts/schedule_scoring.sh` can run judge+score on an
existing samples file, or score-only on an already judged file.

## Judge model

The current judge is text-only: it sees the visual question, the ground-truth
answer, and one candidate rollout. The image is not passed to the judge because
semantic equivalence to a known entity label is a text judgment.

The current main path is `--judge-mode free-form`. With `--judge-with-desc`, the
prompt additionally includes available taxonomy evidence for the ground-truth
leaf, parent, and grandparent classes:

- the ground-truth label, if available;
- broader parent/grandparent labels, if available;
- descriptions for any of those levels, when present.

The prompt explicitly warns that matching only a parent or grandparent is not
enough: those classes are broader than the ground truth. This evidence exists to
reduce under-specific false positives, not to relax correctness.

## Judge outputs

Each judged row keeps the original samples fields and adds:

- `judge_verdicts`: boolean verdict per rollout;
- `judge_hit`: whether any rollout was accepted;
- `judge_hit_count`: number of accepted rollouts;
- `judge_selected`: first accepted rollout index, or fallback index;
- `judge_selected_text`: first accepted rollout text, or fallback text.

`judge_selected_text` is useful for inspection, but it is no longer the whole
taxonomy scoring story.

## Scoring semantics

Scoring starts from the unique set of rollout texts in `all_texts`, preserving
first-seen order and storing duplicate counts plus original indices. For each
taxonomy measure, the scorer evaluates those unique rollout strings and reports the
best mapped candidate under that measure. If a legacy row has no `all_texts`, it falls
back to `judge_selected_text`, then legacy prediction fields.

This has two important consequences:

- `pass@k` is independent of graph mapping. It is computed only from
  `judge_verdicts`.
- `judge_verdicts` do not filter taxonomy measures. They remain available for pass@k
  and later judge-conditioned slices over persisted rollout metrics.
- Taxonomy metrics depend on mapping candidate text to a graph node. Always read
  `num_mapped`, `num_unmapped`, and `mapping_coverage` next to hP/hR/hF.

## Supported measures

- Direct measures: `exact_match`, `contained`, `bleu`, `meteor`, `rouge`, and
  `sentence_bert` map predictions to taxonomy nodes by scoring against all node
  labels, then compute graph metrics against the ground-truth path.
- `cascade`: maps predictions through the paper-aligned cascade: CLIP text top-k
  retrieval, exact top-k matching, n-gram matching, ancestor voting, and top-score
  fallback. Defaults are `k=10`, `thr_topk=0.0015`, `thr_top2=0.001`, and vote
  threshold `>=4`.

All taxonomy measures can report mapped-only metrics and all-example metrics. The
all-example variants zero-fill unmapped examples and are therefore better for comparing
systems with different mapping coverage.

The scored JSONL also stores `<measure>_rollout_metrics` and
`cascade_rollout_metrics`. Each record is deduped by rollout text and includes
`text`, `count`, `indices`, predicted node/path, hP/hR/hF, exact flag, mapping method,
specificity metrics, and depth diagnostics. These records are the source for later
frequency-weighted means, variances, quantiles, near-miss credit, and judge-conditioned
statistics.

## Scheduling behavior

The current scheduling code separates resource concerns:

- inference jobs use the model GPU configuration requested by the run;
- judge jobs can use separate GPU allocations;
- scoring can run after judging in the same post-processing job;
- aggregate-only scoring can recompute summaries from existing scored files
  without rerunning mapping.

Operational launch examples belong in `docs/commands.md` or
`docs/operations/`, not in thesis notes.
