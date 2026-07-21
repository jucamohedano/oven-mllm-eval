# 002 — Taxonomy-aware metric construction: the mapping cascade and rollout aggregation

- **Date:** 2026-07-01
- **Status:** Active. Documents two metric-construction decisions for the hierarchical evaluation: (i) how a free-form prediction is **mapped to a taxonomy node** (a CLIP text-retrieval cascade aligned to the CVPR 2025 / `vlm-eval` reference, including the paper thresholds), and (ii) how the **256 rollouts per example** are represented and aggregated — binary `pass@k` from judge verdicts, graded row-level Best-of-N hF, and persisted deduped per-rollout metrics for downstream means, variances, quantiles, and judge-conditioned statistics.
- **Models / task / dataset:** Qwen3-VL-2B/4B/8B-Instruct on OVEN (Hu et al., arXiv:2302.11154), validation, **aligned** taxonomy-question variant (`data/processed/vlm_compatible_val_aligned.jsonl`, **115,552** examples), prompt `concise_no_idk`, 256 rollouts/example, Qwen3-4B free-form judge — same artifacts as [[001-model-scale-coverage-vs-reliability]].
- **Reference method:** *Taxonomy-Aware Evaluation of Vision-Language Models*, **CVPR 2025** (Snæbjarnarson et al., arXiv:2504.05457); their `ComplexMatcher` in the `vlm-eval` repo.
- **Source artifacts:** `src/oven_mllm_eval/matching.py` (`TaxonomyMatcher`, the cascade), `src/oven_mllm_eval/embedding_matcher.py` (CLIP/SentenceTransformer retrieval + cached node embeddings), `src/oven_mllm_eval/scoring.py` (`_score_rows` direct measures; `_score_rollouts` cascade BoN; persisted rollout metrics), `src/oven_mllm_eval/measures.py` (`DirectMeasureMatcher`, direct measures), `src/oven_mllm_eval/scores.py` (`normalize`, `calc_hierarchical_metrics`), `scripts/score_predictions.py`.
- **Implementation notes:** current scoring keeps paper/reference semantics while optimizing execution: `exact_match` uses a normalized-label lookup/cache that is parity-tested against the slow direct matcher and original `vlm-eval`; direct `sentence_bert` caches taxonomy/reference embeddings; cascade top-k extraction uses partial top-k selection rather than full sorting; scoring logs progress in direct-worker and embedding/cascade phases.
- **Design docs:** `docs/methods/taxonomy-mapping-cascade.md`, `docs/methods/rollout-hierarchical-metrics.md`, `docs/findings/prompt-collapse-and-question-misalignment.md`.
- **Related notes:** [[001-model-scale-coverage-vs-reliability]].

---

## 1. Why the mapping stage is a metric, not a detail

Hierarchical precision/recall/F1 (hP/hR/hF) require each prediction to be placed on a
node of the evaluation taxonomy so its ancestor path can be compared to the ground
truth. But the models emit **unconstrained text** ("a baseball stadium", "Nationals
Park", "oriole"). The text→node **mapping is therefore part of the metric**, and a
confound:

- **Too conservative** (exact string match only) → most predictions go *unmapped*, and
  hP/hR/hF measure the mapper, not the model. On the real 2B run (115,552 examples), the
  judge-selected prediction **exact-matches a node only 32.9%** of the time (+2.2% via
  aliases); **64.9% are unmapped** — and the unmapped set is a mix of surface variants
  ("washington nationals stadium"), common-name/abstraction answers ("a rice dish with
  seafood", "oriole"), and noise. So an exact-only metric discards two-thirds of the
  signal.
- **Too permissive** (loose fuzzy match) → predictions get *hijacked* onto wrong nodes,
  inflating scores.

The CVPR 2025 reference resolves this with a **cosine-retrieval cascade** validated
against human labels. Our codebase had ported the cascade but **diverged from it in
three ways, all toward over-permissiveness** (§3). This note records the corrected
construction.

### 1.1 Terminology: `ExactMatch`, `exact_match`, `exact_match_exact_match`

The code has three similarly named objects at different layers. They are related,
but they are not interchangeable:

| name | role | graph-aware? | where used |
|---|---|---:|---|
| `ExactMatch` | low-level binary string scorer: one taxonomy label vs one prediction | no | inside `DirectMeasureMatcher.match()` |
| `exact_match` | full evaluation measure: map text to a node using `DirectMeasureMatcher`, then compute graph metrics | yes | `--measure exact_match` |
| `exact_match_exact_match` | row-level boolean: mapped node exactly equals the ground-truth label | partly | output field for the `exact_match` measure |
| `cascade` | full evaluation measure: map text to a node using `TaxonomyMatcher`, then compute graph metrics | yes | `--measure cascade` |
| `pass@k` | binary rollout coverage from `judge_verdicts` | no | summary JSON |

In other words, **yes: `DirectMeasureMatcher` is the mapper used by the
`exact_match` measure**. It does two jobs through `evaluate()`:

1. **Map text to the graph.** `DirectMeasureMatcher.match()` scores the cleaned,
   normalized, stemmed prediction against every taxonomy node label using the selected
   low-level scorer. For `exact_match`, a node label gets score 1 only under exact
   equality after preprocessing. If every node scores 0, the paper-style direct matcher
   still returns the top-ranked node, matching `vlm-eval` semantics.
2. **Score the mapped node.** Once a predicted node/path exists,
   `DirectMeasureMatcher.evaluate()` computes hP/hR/hF against the ground-truth
   path and also computes the boolean `success` value stored as
   `exact_match_exact_match`.

```
prediction
  → clean / normalize / stem
  → `ExactMatch` against every taxonomy node label
  → highest-scoring predicted node/path
  → hP / hR / hF from predicted path vs ground-truth path
  → `exact_match_exact_match` from predicted node vs ground-truth label
```

The confusing output name `exact_match_exact_match` is just namespacing:
`<measure>_<metric>`. The measure is `exact_match`; the scalar metric is
`exact_match`. It means "after the `exact_match` mapper picked a node, was that node
exactly the ground-truth node?" It is not the operation that computes hP/hR/hF.

The `cascade` measure does **not** call `DirectMeasureMatcher.match()`. It has its own
exact stage (`_exact_in_topk`) inside `TaxonomyMatcher.match_prediction()`, followed by
n-gram, voting, and fallback stages. The shared idea is exact label equality; the
procedure and search space are different:

- direct measures: `DirectMeasureMatcher` + selected text metric over **all taxonomy
  nodes**. Supported direct measures are `exact_match`, `contained`, `bleu`, `meteor`,
  `rouge`, and `sentence_bert`.
- `cascade`: `TaxonomyMatcher` + CLIP text top-k + normalized exact/ngram/voting/fallback.

The production `exact_match` code uses a fast path, but it is only an implementation
optimization. It precomputes normalized/stemmed taxonomy labels and lazily caches the
top-ranked exact-match result for each normalized prediction key. The output preserves
the paper/reference behavior, including deterministic all-zero/tie ordering; parity
tests compare the fast path against the generic direct matcher and the original
`vlm-eval` `DirectMeasureMatcher`.

## 2. The cascade (text → node)

One shared algorithm maps a prediction to a taxonomy node. The production `cascade`
measure uses CLIP text-to-text retrieval for Step 1; SentenceTransformer retrieval is
kept only as an ablation backend.

```
Step 1  top-k candidates via CLIP text cosine over cached node text embeddings
Step 2  exact equality within top-k                         → exact_match_in_top_k
Step 3  n-gram overlap, N = 4,3,2:
          top-k    : node n-grams ∩ pred n-grams (partial)  → ngram_topk_match_N
          all-nodes: a pred n-gram == a FULL node label      → ngram_match_N
Step 4  ancestor voting if top-k scores ambiguous           → voting
Step 5  top-score fallback                                  → top_score
```

Defaults match the paper/reference: `k=10`, `thr_topk=0.0015`, `thr_top2=0.001`, and
ancestor votes `>=4`. Ambiguity is computed as `(softmax_top1 - softmax_topk) / k`,
with the top-2 gap checked separately. Every prediction's `mapping_method` is recorded;
the results JSON reports the breakdown for the selected Best-of-N cascade rollout.

The cascade always maps when the taxonomy is non-empty. There is no `none` mapping
method and no low-confidence NONE-floor in the paper-aligned implementation; if exact,
n-gram, and voting do not select a node, `top_score` returns the highest-scoring node.
CLIP retrieval is **only** the top-k retrieval feeding the cascade — it never scores hF
directly. The measure is therefore named **`cascade`** (the algorithm), not after its
retrieval step.

Implementation detail: taxonomy node embeddings are cached by backend/model/node set
and reused across runs. Prediction texts are deduplicated globally within a scoring run
before embedding, but they are not yet persisted as a cross-run cache. The CPU top-k
step uses partial top-k selection (`argpartition`-style) and then sorts only those
candidate indices, preserving the final top-k order while avoiding a full sort over all
taxonomy nodes. `[embed]` log lines report cache loading/building, text-encoding
batches, similarity search, top-k extraction, and cascade mapping progress.

## 3. Findings — three divergences from the reference (all fixed)

Verified by reading the reference (`vlm-eval/.../map_predictions.py`, `scores.py`).
Each divergence made our port more permissive and produced hijacks.

| # | Reference does | Our port did | Symptom |
|---|---|---|---|
| 1 | **exact equality** in top-k; no contains-over-all stage | substring **containment**, + an extra contains-over-all-nodes stage | `"food"` ⊂ `"seafood"` → maps to *Food*; `"park"` ⊂ `"parkway"` |
| 2 | all-nodes n-gram = pred n-gram **== a full label** | **partial** node-ngram ∩ pred-ngram (deepest) | stop-word bigram `"of the"` maps a stadium to "69 Stations of the Nakasendō" |
| 3 | `normalize` strips **ASCII punctuation only** (keeps Unicode) | `re.sub(r"[^a-z0-9 ]+", " ", …)` deletes Unicode letters | manufactured degenerate labels (next table) |

**Divergence 3 is the root cause** of most hijacks. Our `_normalise` deleted every
non-ASCII letter, so:

| label | reference `normalize` | our old `_normalise` |
|---|---|---|
| `España` | `españa` | `espa a` → stray `"a"` token |
| `tūī` | `tūī` | `t` → matches almost anything |
| `Усилитель мощности` (and 7 more) | `усилитель мощности` | `""` → **substring of every prediction** |

On the real taxonomy (12,805 nodes): **8 labels normalized to the empty string and 1 to
a single char** under the old rule; under the fix, **0 and 0**.

**Fixes** (all realign to the reference): exact-equality in top-k (`_exact_in_topk`);
full-label all-nodes n-gram (`_fulllabel_ngram_check`) for N = 4, 3, 2;
Unicode-preserving `_normalise` reusing the reference's `scores.normalize` (no index
rebuild needed — it only keys a near-dead `label_to_paths` fallback, and is applied
symmetrically to nodes and predictions); the old NONE-floor was removed. Net effect:
`"a beautiful oriole bird"` → **Bird** (was "Edificio España"), `"red chicken"` →
**chicken**, noise → the top-scoring taxonomy node via `top_score`; `exact_match`
numbers remain direct-measure scores. Remaining possible enhancement: embed aliases as
separate retrieval rows if we want stronger synonym recall.

## 4. Rollout aggregation — the core decision

Each example has **256 rollouts**, not one prediction. How they collapse to one number
is itself a metric decision. There are **two families**: a **binary** aggregation over
the judge's per-rollout verdicts (**pass@k**) and **graded** aggregations over the
per-rollout hF (`exact_match`, `cascade`, `mean`). The original hF aggregation was
incoherent (§4.1); pass@k was already coherent and is unchanged.

### 4.1 The incoherence we replaced
Scoring used `judge_selected_text` = the **first** rollout the judge tagged correct
(`run_judge.py`: `verdicts.index(True)`), else `all_texts[0]`. The aggregate is
≈ `1.0·(judge-hit rate) + hF(rollout 0)·(miss rate)`: it cherry-picks the correct
rollout on a hit, falls back to an **arbitrary** rollout on a miss, is coupled to the
(noisy) judge, and ignores refusals. It is neither expected quality nor best-of-N.

### 4.2 The metrics

**`pass@k` — the binary coverage metric (from the judge; unchanged).** Computed from the
256 binary `judge_verdicts`, *not* from the taxonomy mapping: `pass@k` is the unbiased
probability that ≥1 of *k* sampled rollouts is judged correct (Codex estimator, product
form, `pass_at_k.py`; reported for k = 1, 2, 4, …, 256). At `k = n = 256` it reduces
exactly to **coverage** — the fraction of examples solved at least once (decomposition
identity, [[001-model-scale-coverage-vs-reliability]] §1). It is the **binary** rollout
aggregation; the graded hF metrics below are its taxonomic counterparts. **Caveat:**
pass@k inherits the judge's leniency about answer granularity — 001 shows the raw judge
pass@k reports a spurious "smaller-is-broader" trend that a specificity-preserving check
*reverses*. Read it as a judge-coverage **upper bound**, not verified coverage.
In multi-measure summary JSON, the same pass@k curve is repeated under each measure
block for convenience. It is not recomputed differently for `exact_match` vs `cascade`;
semantically there is one measure-independent pass@k curve per judged input file.

**Direct text measures — paper-style rollout scoring.** The direct measures
(`exact_match`, `contained`, `bleu`, `meteor`, `rouge`, `sentence_bert`) ignore
`judge_verdicts` and score the **unique rollout texts** directly against all taxonomy
node labels. The row-level measure fields report the rollout with the highest hF for
compatibility with the existing aggregate schema, and each row also stores
`<measure>_rollout_metrics` so downstream analysis can compute frequency-weighted
statistics over all rollouts.

The `exact_match` measure uses `DirectMeasureMatcher`, so the string-to-node
mapping is not raw string equality. The prediction is cleaned
(`A:`, answer tags, chat markers, and trailing punctuation removed), then both the
prediction and node labels are normalized (lowercase, hyphen→space, ASCII punctuation
removed) and stemmed with the NLTK English Snowball stemmer when available. Only then
does the low-level `ExactMatch` scorer perform strict equality. Example:

```
"Dandie Dinmont Terriers." → "dandi dinmont terrier"
"Dandie Dinmont Terrier"  → "dandi dinmont terrier"
```

The paper-style direct matcher always returns the highest-scoring taxonomy node when
the taxonomy is non-empty, even when all scores are zero. For `exact_match`, that means
an unrelated string still receives a deterministic top-ranked node, matching
`vlm-eval` semantics. This behavior is intentionally different from the previous local
guard that treated all-zero exact-match scores as unmapped.
For scale, direct `exact_match` is accelerated by the cached normalized-label fast path
described in §1.1, and direct `sentence_bert` caches taxonomy/reference embeddings by
model and label set. These caches change runtime only, not the measure definition.

**`cascade` — Best-of-N (a baseline).** Map every rollout in the example's **unique
(deduped) set** via the cascade, score each, and report the **best (highest-hF)** one
(`scoring.py:_score_rollouts`, `selection="best_of_n"`). This is the **graded analog of
pass@256**: it measures the *latent* knowledge — how close the model's best attempt gets.
It is the coherent version of the old "select a good rollout" (graded hF over all unique
rollouts, no judge coupling, no index-0 fallback). It is an **oracle ceiling**: picking
the best rollout needs the ground truth, so it is evaluation-only and optimistic, and it
cannot distinguish "lucky once in 256" from "consistently right."

**Distribution statistics from persisted rollout metrics.** The expected hF of a
*single* rollout drawn at random — what you actually get at deployment — can now be
computed downstream from the persisted deduped rollout metrics. Worked example (GT
`house finch`):
8×`house finch`(1.0) + 40×`finch`(0.89) + 80×`bird`(0.75) + 28×`animal`(0.57) +
100×refusal(0.0) ⇒ **BoN 1.00, mean ≈ 0.47**. The **BoN-mean gap** is the real story:
how much the model *knows* but does not reliably *say* — the graded analog of the
pass@1 <-> pass@256 spread in [[001-model-scale-coverage-vs-reliability]]. The scored
JSONL now stores enough data to compute this without remapping: use each
`*_rollout_metrics` record's metric values with `count` as the frequency weight.

### 4.3 Binary ↔ graded correspondence

The binary judge metric and the graded hF metrics are parallel aggregations of the same
256 rollouts — pass@k along the binary (judge-verdict) axis, hF along the graded
(taxonomic) axis:

| aggregation | binary (judge verdicts) | graded (taxonomy hF) |
|---|---|---|
| single random sample | pass@1 | frequency-weighted mean hF from persisted rollout metrics |
| best-of-k | pass@k | downstream from persisted rollout metrics |
| oracle best-of-256 | pass@256 (= coverage) | `cascade` BoN hF |

### 4.4 Reading the metrics
They sit on **different footings** — do not compare hF (or hF vs pass@k) as if only the
mapping differed:

| metric | source | rollout aggregation | role |
|---|---|---|---|
| `pass@k` | judge `verdicts` (binary) | best-of-k, all k | judge-coverage curve; pass@256 = coverage (upper bound) |
| direct measures | direct text metric (hF) | best over unique rollouts + persisted rollout distribution | paper-style direct baseline |
| `cascade` | cosine-retrieval cascade (hF) | **Best-of-N** over unique rollouts | graded latent-coverage ceiling |

pass@k uses the judge only (mapping-independent). Direct measures and `cascade` now map
deduped rollout texts directly; `judge_verdicts` are used only for pass@k and
judge-conditioned downstream diagnostics. **Always report `hF` with `num_mapped`, and
pass@k as judge-coverage (not verified) — see 001.**

The most important accounting caveat is that summary `hP/hR/hF` are **conditional on
successful graph mapping**. Unmapped examples are included in `num_examples`, but they
are not included in the mean hP/hR/hF denominator. Therefore:

```
hF = 0.90, num_examples = 1000, num_mapped = 100
```

means "average hF is 0.90 among the 100 mapped examples", **not** "average hF is 0.90
over all 1000 examples." A strict mapper can look strong if it only maps easy cases.
The summary JSON now reports this explicitly:

| field | meaning |
|---|---|
| `num_examples` | total evaluated rows |
| `num_mapped` / `num_unmapped` | rows included / excluded from conditional graph means |
| `mapping_coverage` | `num_mapped / num_examples` |
| `hP`, `hR`, `hF`, `exact` | conditional means over mapped rows only |
| `hP_all`, `hR_all`, `hF_all`, `exact_all` | all-example means, treating unmapped rows as zero |
| `specific_num_mapped`, `specific_num_unmapped` | same accounting for specificity-selected candidates |
| `specific_mapping_coverage` | `specific_num_mapped / num_examples` |
| `specific_hP_all`, `specific_hR_all`, `specific_hF_all`, `specific_exact_all` | all-example specificity-aware means |

Per-row rollout-metric records provide the raw material for later statistics:

| field | meaning |
|---|---|
| `text` | unique rollout answer string |
| `count` | number of duplicate rollouts with this exact text |
| `indices` | original rollout positions in `all_texts` |
| `predicted_node` / `predicted_path` | mapped taxonomy node and path |
| `hP`, `hR`, `hF`, `exact_match` | vanilla hierarchical metrics for this rollout text |
| `specific_hP`, `specific_hR`, `specific_hF`, `under_specific`, `over_specific`, `depth_delta` | specificity-aware metrics and depth diagnostics |
| `mapping_method` | cascade/direct mapping method label |
| `scores` | direct-measure score payload, present for direct measures |

For model comparisons, report at least:

```
hF
num_mapped / num_examples
hF_all
exact
pass@1 and pass@256
```

where:

```
hF_all = hF × (num_mapped / num_examples)
```

This caveat does **not** apply to `pass@k`: pass@k is computed directly from
`judge_verdicts` and is independent of graph mapping.

## 5. Relationship to the [[001-model-scale-coverage-vs-reliability]] audit

> **Legacy full-dataset run (2026-06-25; old cascade settings).** The previous
> full-dataset `cascade` run in [[001-model-scale-coverage-vs-reliability]] §4.5 used
> the pre-paper-parity mapper. It is useful as historical context but should not be
> quoted as the current paper-aligned result. The qualitative caveat still stands:
> cascade BoN is an oracle ceiling that can reward under-specific hypernym answers with
> graded partial credit, so it must be interpreted separately from specificity-preserving
> binary support checks. Re-run the full aligned runs with the current CLIP/k=10/no-NONE
> implementation before reporting final numbers.

Note 001 verifies coverage with a **deterministic, specificity-preserving support set**
(`judge_audit.py`: exact/alias/answer⊆pred, *excluding* pred⊆answer). That is a
**separate, binary** verification from this note's **graded** taxonomy mapping. They
embody two different philosophies, and the thesis must keep them distinct:

- **hP/hR/hF via the cascade (this note)** *deliberately gives graded partial credit for
  taxonomic closeness*, including to **under-specific** answers — a hypernym maps to a
  parent node and earns hF ∈ (0,1). This is the point of a *hierarchical* metric.
- **001's "supported" set** *forbids* crediting under-specificity (it excludes
  pred⊆answer), because 001 shows the judge's leniency about granularity is exactly what
  manufactures the spurious "smaller-is-broader" coverage trend.

So the two answer different questions: cascade hF = "how taxonomically close is the
answer," supported coverage = "did it name the target entity or something more specific."
Direct `exact_match` sits closer to a strict node-identity view, while `cascade` BoN is
the graded taxonomy-proximity view. The cascade pushes back on under-specificity
gradedly (lower hF for parents) rather than by exclusion; it no longer uses a NONE-floor
because paper/reference parity is prioritized.

## 6. Reproducibility

The implementation entry points are `scripts/score_predictions.py` and
`src/oven_mllm_eval/scoring.py`. Operational command recipes live in
`../commands.md` and `../operations/`; this note records the conceptual contract.

- Input for full taxonomy-aware scoring should carry `all_texts`; judged files also
  carry `judge_verdicts`, which are needed for pass@k and judge-conditioned diagnostics.
- Node embeddings are cached and reused across runs; the cache is keyed by backend,
  model, and node set.
- Direct `sentence_bert` taxonomy/reference embeddings are cached separately by model
  and label set. Direct `exact_match` uses a normalized-label fast path that preserves
  `vlm-eval` semantics and is parity-tested against the slow path and original code.
- Cascade top-k retrieval uses partial top-k extraction instead of sorting every node
  score. This is a runtime optimization only; the final top-k ordering is unchanged.
- Progress logs are emitted during input loading, direct-worker scoring, chunk merge,
  embedding cache use/build, text encoding, similarity search, cascade mapping, summary
  aggregation, and JSON/summary writes. Logs are operational observability, not metric
  data.
- Scored fields are namespaced by measure: direct measures write
  `<measure>_*` and `<measure>_rollout_metrics`; cascade writes `cascade_*` and
  `cascade_rollout_metrics`.
- Summary fields distinguish conditional and all-example views:
  `hP/hR/hF/exact` are mapped-only, while
  `hP_all/hR_all/hF_all/exact_all` zero-fill unmapped rows.
- `num_mapped`, `num_unmapped`, and `mapping_coverage` make the denominator
  explicit and must be reported with taxonomy metrics.
- Each rollout-metric record is deduped per row and stores `text`, `count`, `indices`,
  predicted node/path, hP/hR/hF, exact flag, mapping method, specificity metrics, and
  depth diagnostics. Direct measures also store their measure score payload as `scores`.
- Summary JSON repeats the same pass@k values inside each measure block in a
  multi-measure run. That duplication is for schema convenience; pass@k is computed
  once from `judge_verdicts`.

## 7. Open questions / TODO

- [~] **Re-run the current paper-aligned taxonomy scoring on the full aligned `concise_no_idk` runs** with `exact_match cascade`: CLIP text retrieval, `k=10`, `thr_topk=0.0015`, `thr_top2=0.001`, vote threshold `>=4`, and no NONE-floor. 4B has been recomputed/synced; finish the remaining model sizes before quoting final cross-scale taxonomy numbers.
- [ ] **Compute distribution statistics** from persisted rollout metrics: frequency-weighted means, variances, quantiles, BoN-mean gaps, and judge-conditioned slices.
- [x] **Speed up `exact_match` without changing semantics:** normalized/stemmed label cache + lazy top-k cache; parity-tested against the slow direct path and original `vlm-eval`.
- [ ] If using `rouge` at scale, add a global direct-measure mapping cache. ROUGE over all nodes and millions of rollout texts remains expensive.
- [ ] Consider a cross-run prediction/mapping cache for cascade. Taxonomy node embeddings are cached, but millions of unique rollout texts are still re-encoded per run.
- [ ] Decide whether to add alias rows to CLIP retrieval for broader synonym recall.
- [~] Reconcile with 001: does the cascade's graded crediting of under-specific answers reintroduce the leniency 001 removed? **Qualitatively yes**, because hierarchical hF gives partial credit to parent nodes. Still open under the paper-aligned implementation: quantify cascade hF on the under-specific (`Ans⊃Pred`) rollouts specifically.

## 8. References
- *Taxonomy-Aware Evaluation of Vision-Language Models* — Snæbjarnarson et al., **CVPR 2025**, arXiv:2504.05457 (the mapping cascade + human-validated measure comparison).
- OVEN: *Open-domain Visual Entity Recognition* — Hu et al., arXiv:2302.11154.
- Entity-linking pattern (retrieve→rerank, generative): BLINK (Wu et al., EMNLP 2020); GENRE (De Cao et al., ICLR 2021); GER (Caron et al., CVPR 2024, on OVEN).
- pass@k unbiased estimator — Chen et al. (Codex), 2021.

## 9. Changelog
- **2026-07-01 (rev 5):** Added implementation/reproducibility details for the restored scorer: exact-match fast path with `vlm-eval` parity tests, direct `sentence_bert` reference-embedding cache, cascade partial top-k extraction, and progress logging for direct and embedding/cascade phases. Clarified that pass@k is repeated under each measure block in multi-measure summaries but is one judge-verdict curve, and that 4B has been recomputed while final cross-scale taxonomy tables remain pending.
- **2026-07-01 (rev 4):** Updated to the paper-aligned implementation: CLIP text retrieval defaults, `k=10`, `thr_topk=0.0015`, `thr_top2=0.001`, vote threshold `>=4`, no NONE-floor, and direct paper measures over rollout predictions. Documented persisted deduped `<measure>_rollout_metrics` / `cascade_rollout_metrics` for downstream statistics.
- **2026-06-26 (rev 3):** Clarified `ExactMatch` vs `DirectMeasureMatcher.match()` vs `cascade`, documented judge-positive dedup + best-hF selection, no-match behavior, and the mapped-only denominator. Added summary-field accounting for `num_unmapped`, `mapping_coverage`, all-example zero-filled metrics (`hP_all/hR_all/hF_all/exact_all`), and specificity-aware equivalents.
- **2026-06-25 (rev 2):** Legacy full-dataset `cascade` BoN run completed on the 2B/4B/8B aligned `concise_no_idk` runs (with-desc judge); these numbers predate the paper-aligned CLIP/k=10/no-NONE implementation and should be treated as historical context only.
- **2026-06-24 (rev 1):** Initial note. Documented the first cascade realignment and rollout-aggregation framing; later revisions supersede its NONE-floor and judge-correct direct-measure assumptions.
