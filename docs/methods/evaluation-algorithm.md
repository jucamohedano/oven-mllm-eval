# OVEN Evaluation Algorithm

This document summarizes the current evaluation algorithm. It supersedes the
older `_matches_label` pipeline: inference no longer chooses correctness through
a substring gate, and taxonomy scoring no longer depends only on a single
`prediction` string.

## Phase 1 — inference

For each OVEN example, `scripts/run_inference.py` samples `k` rollouts from the
VLM and stores them in `all_texts`.

The samples row keeps the ground-truth answer and metadata needed downstream:

- `answer` / `entity_id`;
- `question`;
- `image_path` / image identifiers;
- `all_texts`;
- the run metadata and prompt variant.

Inference is responsible for producing candidate answers, not for deciding
whether they are correct.

## Phase 2 — judge

`scripts/run_judge.py` evaluates each rollout against the ground truth. In the
current main path, the judge is a text-only Qwen3 model using free-form 0/1
judgment. With `--judge-with-desc`, the prompt also includes available taxonomy
evidence for the ground-truth leaf, parent, and grandparent.

The judged row adds:

- `judge_verdicts`: one boolean per rollout;
- `judge_hit`: whether any rollout was accepted;
- `judge_hit_count`: number of accepted rollouts;
- `judge_selected` and `judge_selected_text`: inspection/fallback fields.

The judge is used to compute binary pass@k and judge-conditioned diagnostics. It is
not the taxonomy mapper, and taxonomy measures do not filter rollouts by
`judge_verdicts`.

## Phase 3 — candidate selection for scoring

`src/oven_mllm_eval/scoring.py` builds the unique rollout set for each row from
`all_texts`, preserving first-seen order and storing `count` plus original rollout
`indices` for duplicates. If a legacy row has no `all_texts`, scoring falls back to
`judge_selected_text`, then `prediction`, `iter_final_prediction`, and `output`.

## Phase 4 — taxonomy mapping and metrics

Each unique rollout text is mapped to a taxonomy node by the requested measure.

### `exact_match`

`DirectMeasureMatcher` cleans model-output wrappers, normalizes/stems the candidate
text, scores it against all taxonomy node labels, and returns the highest-scoring node.
For `exact_match`, all-zero scores still produce a deterministic top-ranked node, matching
the paper/reference direct-measure behavior.

Supported direct measures are `exact_match`, `contained`, `bleu`, `meteor`, `rouge`,
and `sentence_bert`. Row-level fields report the highest-hF rollout for that measure;
`<measure>_rollout_metrics` stores the deduped per-rollout records for later statistics.

### `cascade`

`TaxonomyMatcher` maps rollouts through the paper-aligned cascade: CLIP text top-k
retrieval, exact top-k matching, n-gram matching, ancestor voting, and top-score
fallback. Defaults are `k=10`, `thr_topk=0.0015`, `thr_top2=0.001`, and vote threshold
`>=4`. The cascade always maps to a taxonomy node when the taxonomy is non-empty.

For 256-rollout samples, cascade scoring reports a Best-of-N oracle score over
the unique rollout set. This is useful as a graded coverage ceiling, not as a
single-shot reliability metric. `cascade_rollout_metrics` stores the full deduped
per-rollout metric records.

## Phase 5 — aggregation

The summary JSON reports:

- binary pass@k from `judge_verdicts`;
- mapped-only hP/hR/hF/exact;
- all-example hP/hR/hF/exact variants with unmapped rows zero-filled;
- `num_mapped`, `num_unmapped`, and `mapping_coverage`;
- specificity-aware variants such as `specific_hF` and under-specific rates
  when available;
- `mapping_methods` for cascade runs;
- `<measure>_rollout_metrics` and `cascade_rollout_metrics` in the scored JSONL,
  each deduped by rollout text with `count` and `indices`.

Interpretation rule: pass@k is a judge/verdict metric, while hP/hR/hF are graph
mapping metrics. They answer different questions and must not be collapsed into
one "accuracy" claim.
