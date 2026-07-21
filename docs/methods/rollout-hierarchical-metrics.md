# Measuring hP/hR/hF Over Rollouts

> Date: 2026-07-01. Status: **Best-of-N implemented**; per-rollout metrics are
> persisted in deduped form (`count` + `indices`) so distribution statistics can be
> computed later without remapping.
>
> How hierarchical precision/recall/F1 are computed when each example has *n = 256*
> rollouts, not a single prediction.

## Decision (2026-06-24)

The measure is named **`cascade`** — it is the full mapping algorithm. CLIP text cosine
retrieval is only its top-k retrieval step, not a scorer. The cascade (exact → n-gram →
voting → top-score) maps a prediction to a taxonomy node; hP/hR/hF are computed on that
node. Cosine only selects the top-k candidates and orders the voting/top-score fallback,
as in the reference; it never produces hF directly.

**Best-of-N (BoN) is a baseline** for turning a 256-rollout sample into one score: map
each rollout via the cascade and report the **best (highest-hF)** one, computed over the
**unique (deduped) set** of rollout answers. Other aggregations are alternatives — the
distribution **mean** is computed downstream from persisted rollout metrics, and the BoN number should be read with
its caveat (an oracle ceiling) in mind.

## 1. The issue (what we replaced)

Hierarchical metrics used to come from **one** representative prediction per example:

- `scripts/run_judge.py` (≈ L462–471): `judge_selected = verdicts.index(True)` — the
  **first rollout the judge marked correct**; if none, `judge_selected_text =
  all_texts[0]` (rollout 0).
- `scripts/run_inference.py`: `prediction = all_texts[0]`.

So hF was scored on the first judge-correct rollout if one existed, otherwise on
rollout 0. That aggregate ≈ `1.0·(judge-hit rate) + hF(rollout 0)·(miss rate)` — it
cherry-picks the correct rollout on a hit, falls back to an *arbitrary* rollout on a
miss, is coupled to the (noisy) judge, and ignores refusals. We evaluate a
distribution of 256 rollouts but reported a single, judge- and index-dependent draw.

## 2. Best-of-N (a baseline)

hF is a **graded** score in [0, 1] (unlike binary pass@k). Per example, map each rollout
in the **unique (deduped) set** of answers to a taxonomy node (the `cascade` measure,
with CLIP text cosine retrieval by default) and score it against the ground-truth path → a set of
per-rollout values. **BoN reports the best of them:**

```
i* = argmaxᵢ hFᵢ   over the unique rollouts   →  report (hP_{i*}, hR_{i*}, hF_{i*})
```

BoN is the graded analog of **pass@256**: a baseline that measures the **latent**
knowledge — how close the model's *best* attempt gets. It is the coherent version of what
we already did (select a good rollout), but selected by graded hF over the unique
rollouts, with no judge coupling and no index-0 fallback. On a judge-hit the best rollout
is the exact one (hF≈1.0); on a miss it is the best *partial-credit* rollout (right genus,
wrong species) rather than an arbitrary one.

> BoN is an **oracle ceiling** — picking the best rollout needs the ground truth, so it
> is evaluation-only and optimistic, and it cannot distinguish "lucky once in 256" from
> "consistently right." The honest counterpart that fixes this is the distribution mean
> (§9). Read BoN with that in mind.

## 3. Conventions

1. **BoN is one rollout.** Select `i* = argmaxᵢ hFᵢ` and report **that rollout's** triple
   `(hP, hR, hF)`. Do **not** take `max hP`, `max hR`, `max hF` independently — those
   would come from different rollouts and would not describe a single prediction.
2. **Per-rollout unmapped → hF = 0.** A rollout that cannot be scored because it lacks a
   mapped path or reference path scores 0. Since any *mapped* prediction shares the root and thus has hF > 0, the
   argmax always prefers a mapped rollout over an unmapped one.
3. **Example excluded only if *all* rollouts are unmapped.** Then the BoN rollout is
   unmapped and the example drops out of `num_mapped` — matching the existing
   `exact_match` convention. With 256 rollouts this is rare (almost every example has at
   least one mappable rollout).
4. **Per example, then across examples.** BoN per example, then average across examples.

## 4. Implementation (done)

Both `cascade` and the direct text measures score the deduped rollout set. `pass@k`
remains unchanged because it is computed only from `judge_verdicts`.

- **`scoring.py` → `_score_rollouts(scored_rows, index, mapping)`** — for each example,
  scores each unique rollout against the reference path and keeps the highest-hF one;
  writes `cascade_*` fields, stores `cascade_rollout_metrics`, and accumulates the BoN triple.
- **`score_generation_file` (`do_embed` branch)** — maps **all** rollouts
  (deduped inside `build_prediction_mapping`) instead of one representative, then calls
  `_score_rollouts`.
- The `cascade` summary carries `"selection": "best_of_n"` plus the
  `mapping_methods` breakdown (of the *selected* rollout). The `hP/hR/hF` of the
  `cascade` measure are therefore the **BoN** values.
- Each scored row stores one rollout metric record per unique answer:
  `text`, `count`, `indices`, predicted node/path, hP/hR/hF, exact flag, mapping method,
  specificity metrics, and depth diagnostics. Direct measures also include their
  measure-specific score payload under `scores`. Repeated rollout strings reuse the same
  score via `count`.

```python
# _score_rollouts core
best = None  # (hF, hP, hR, node, path, method)
for record in rollout_records:
    pp = (mapping.get(record["text"]) or {}).get("predicted_path")
    if pp is not None and ref_path is not None:
        mt = calc_hierarchical_metrics([(pp, ref_path)])
        hP, hR, hF = mt["hP"][0], mt["hR"][0], mt["hF"][0]
    else:
        hP = hR = hF = 0.0          # unmapped → 0
    if best is None or hF > best[0]:
        best = (hF, hP, hR, node, pp, method)
```

*Within an example, identical rollout strings share a node/hF, so each unique string is
scored once and persisted with `count` plus original `indices`.*

## 5. Cost

Mapping/scoring now operates over unique rollout strings per row, not over one
representative answer. Node embeddings are cached by backend/model/node-set; prediction
embeddings are recomputed per scoring run over the unique rollout strings requested by
that run. Direct measures score each unique rollout against all taxonomy nodes, so
`exact_match`/`contained` can be optimized later with normalized-label lookup tables if
runtime becomes an issue.

## 6. Direct-measure rollout baselines

Direct measures (`exact_match`, `contained`, `bleu`, `meteor`, `rouge`,
`sentence_bert`) are scored over the unique rollout answers, independent of
`judge_verdicts`, and persist `<measure>_rollout_metrics` using the same deduped
`count` + `indices` shape. The row-level `<measure>_hF` remains the best-hF selected
rollout for compatibility with the existing aggregate schema. pass@k and the judge
pipeline are unchanged.

*Cost:* direct measures now score each unique rollout answer. If exact/contained ever
drag, make them dictionary lookups (`{normalized_label → node}`) instead of per-node
scans.

## 7. Diagnostics (optional)

- **Near-miss credit:** mean `hFᵢ` over rollouts where `judge_verdictᵢ = False` — the
  partial credit the binary judge misses.
- **hP / hR split** of the BoN rollout — over-general (low hR) vs wrong-branch (low hP).

## 8. Plots / migration

- Report BoN hF alongside pass@256 (both are best-of-256 ceilings — one graded, one
  binary).
- Re-score existing `_judged.jsonl` (they carry `all_texts`). Expect BoN hF to be
  **close to the old judge-selected number on hits** (the best rollout is the exact one)
  but now **coherent on misses** (best partial-credit rollout, not index-0), and
  `num_mapped` to rise toward the example count (some rollout almost always maps).

## 9. Distribution statistics from persisted rollout metrics

BoN alone is an oracle ceiling (§2). The honest, deployable counterpart is the **mean**
of the per-rollout hF: the **expected hF of a single rollout drawn at random**, which is
what you get when you sample the model once.

**Worked example.** Ground truth `house finch`. Suppose the 256 rollouts are:

| rollout answer | count | hF |
|---|---|---|
| `house finch` (exact) | 8 | 1.00 |
| `finch` (genus) | 40 | 0.89 |
| `bird` | 80 | 0.75 |
| `animal` | 28 | 0.57 |
| `"I don't know"` / garbage (unmapped) | 100 | 0.00 |

- **BoN** = `max` = **1.00** — some rollout nailed it, so the knowledge is in there.
- **mean** = `(8·1.0 + 40·0.89 + 80·0.75 + 28·0.57 + 100·0)/256` ≈ **0.47** — a typical
  single answer only gets about halfway up the correct branch (most rollouts are vague
  or refusals).

The **BoN - mean gap** is the real story: it quantifies how much the model *knows* but
does not reliably *say* (the graded version of the pass@1 <-> pass@256 spread). These
statistics can now be computed from `<measure>_rollout_metrics` and
`cascade_rollout_metrics`:

- use each record's metric values with its `count` as the frequency weight;
- average each metric over **all** rollouts, counting **unmapped rollouts as 0** (so
  refusals/garbage are reflected — this differs from BoN's "exclude only if *all*
  unmapped");
- report downstream values such as `hF_mean`, `hF_std`, quantiles, near-miss credit,
  and judge-false partial-credit statistics.

## 10. Specificity-aware hF

The original `hP/hR/hF` treats every shared ancestor equally. This is too forgiving for
fine-grained OVEN labels because a broad parent prediction can share much of the
reference path and receive a high score. We now report a parallel specificity-aware
metric:

- `specific_hP`, `specific_hR`, `specific_hF`
- `under_specific_rate`
- `over_specific_rate`
- `mean_depth_delta`

The specificity score keeps graded hierarchy credit but changes two things:

1. **Depth weighting:** deeper, more specific path suffixes count more than broad/root
   suffixes. The default decay is `0.5`, so each step toward the leaf is worth twice as
   much as the broader ancestor before it.
2. **Under-specific penalty:** if the predicted node is a strict ancestor of the
   reference node, the specificity score is multiplied by `0.5`.

The old `hP/hR/hF` fields are unchanged for compatibility. Use `specific_hF` when the
goal is discriminating fine-grained capacity; use vanilla `hF` only as lenient taxonomy
proximity.

## 11. Files

- `src/oven_mllm_eval/scoring.py` — `_score_rollouts` + `do_embed` wiring (done).
- `src/oven_mllm_eval/scores.py` — vanilla and specificity-aware hierarchical metrics.
- `src/oven_mllm_eval/embedding_matcher.py` — optional prediction-embedding cache for the
  larger unique set.
- `scripts/score_predictions.py` — (optional) flag to restore single-prediction scoring.
- `scripts/plot_pass_at_k.py` — overlay BoN hF on the pass@256 view.
