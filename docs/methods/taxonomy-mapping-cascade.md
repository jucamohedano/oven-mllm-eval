# Taxonomy Mapping: Paper-Aligned Cascade

> Date: 2026-07-01. Status: implementation aligned back to the CVPR 2025
> `vlm-eval` `ComplexMatcher` semantics.
>
> How free-form VLM predictions are mapped to taxonomy nodes before hierarchical
> precision/recall/F1 (hP/hR/hF) are computed.

## 1. The problem

OVEN models emit unconstrained text ("a baseball stadium", "Nationals Park",
"oriole"). Taxonomy-aware metrics need each prediction placed on a node so we can
compare its ancestor path to the ground-truth path. The mapping stage is therefore
a confound: if it's too conservative, predictions go unmapped (metrics measure the
mapper, not the model); if it's too permissive, it hijacks predictions onto wrong
nodes.

Reference: **Taxonomy-Aware Evaluation of Vision-Language Models, CVPR 2025**
(`vlm-eval/.../map_predictions.py`, `ComplexMatcher`). Our implementation:
`src/oven_mllm_eval/matching.py` (`TaxonomyMatcher`).

## 2. The Cascading Algorithm

The production `cascade` measure follows the paper/reference setup: CLIP text-to-text
retrieval produces the top-k taxonomy nodes, then a deterministic cascade maps the
prediction to one taxonomy node. SentenceTransformer retrieval remains selectable only
for ablations.

```
Step 1  Retrieve top-k nodes
          default: CLIP text cosine vs cached node text embeddings
Step 2  exact-equality in top-k          → exact_match_in_top_k
Step 3  n-gram overlap, N = 4,3,2:
          top-k    : node n-grams ∩ pred n-grams (partial)   → ngram_topk_match_N
          all-nodes: a pred n-gram == a FULL node label       → ngram_match_N
Step 4  ancestor voting if top-k scores are ambiguous          → voting
Step 5  top-score fallback (highest-scoring candidate)         → top_score
```

Paper/reference hyperparameters used by default:

| parameter | value |
|---|---:|
| top-k (`k`) | `10` |
| top-k ambiguity threshold (`thr_topk`) | `0.0015` |
| top-2 ambiguity threshold (`thr_top2`) | `0.001` |
| ancestor vote threshold | `>= 4` |

The ambiguity test matches `vlm-eval`: with top-k scores sorted ascending and
softmaxed, voting is considered when `(softmax_top1 - softmax_topk) / k < thr_topk`
and `softmax_top1 - softmax_top2 < thr_top2`.

The cascade always returns a taxonomy node when the taxonomy is non-empty. There is no
NONE-floor and no low-confidence unmapped fallback in the paper-aligned path. If exact,
n-gram, and voting do not select a node, the mapper returns the top-scoring candidate.

Every selected node records a `mapping_method`, and the `cascade` summary includes a
breakdown for the selected Best-of-N rollout.

## 3. Findings — how our port diverged from the reference

The port had reimplemented three pieces *more permissively* than the reference,
which made the mapper hijack predictions onto wrong nodes.

### 3.1 Containment instead of exact-equality
The reference's in-top-k check is **exact equality** (`pos_answer == name`). Our
port used **substring containment** (`label in prediction`) and labelled it
"exact_match". It also added a **contains-over-all-nodes** stage absent from the
reference. Substring containment hijacks: `"home"` ⊂ `"home stadium"`, `"food"` ⊂
`"seafood"`, `"park"` ⊂ `"parkway"`.

### 3.2 Partial n-gram overlap instead of full-label match (all-nodes)
The reference's all-nodes n-gram stage (`n_gram_variants=False`) requires a
prediction n-gram to **equal a whole node label** (`pred_ngrams ∩ {full labels}`).
Our `_ngram_check` instead tested **partial** overlap of *node* n-grams against
*pred* n-grams, so a multi-word node was hijacked by one shared word — e.g. the
stop-word bigram `"of the"` mapped a stadium prediction to "69 Stations of the
Nakasendō".

### 3.3 Unicode-stripping normalize (the root cause)
The reference `normalize()` (`scores.py`) strips only **ASCII punctuation** and
**keeps Unicode letters**: `"España"→"españa"`, `"tūī"→"tūī"`, Cyrillic preserved.
Our `_normalise` did `re.sub(r"[^a-z0-9 ]+", " ", …)`, deleting every Unicode
letter and **manufacturing degenerate forms** the reference never has:

| label | reference | our old port |
|---|---|---|
| `España` | `españa` | `espa a` → stray `"a"` token |
| `tūī` | `tūī` | `t` → matches almost anything |
| `Усилитель мощности` (×8 such) | `усилитель мощности` | `""` → **substring of every prediction** |

These degenerate tokens were the fuel for the n-gram/containment hijacks (e.g. the
article `"a"` from `"espa a"` made `"a beautiful oriole bird"` map to "Edificio
España").

## 4. Fixes And Current Semantics

1. **Exact-equality in top-k** — `_exact_in_topk` replaces `_contains_check`;
   removed the contains-over-all stage; deleted the dead `_contains_check`.
2. **Full-label all-nodes n-gram** — new `_fulllabel_ngram_check` (a pred n-gram
   must equal a whole label), matching the reference's `n_gram_variants=False`.
3. **Paper n-gram range** — the cascade checks N = 4, 3, 2.
4. **Unicode-preserving normalize** — both `matching._normalise` and
   `measures._normalise` now reuse the reference's `scores.normalize`. The 8 empty
   labels and `tūī→t` are eliminated at the root; Unicode entities match on their
   true form. **No index rebuild needed**: `_normalise` only keys the near-dead
   `label_to_paths` fallback (the real pipeline resolves reference paths from raw
   `node_to_path`/`entity_id_to_path`), and it is applied symmetrically to nodes
   and predictions.
5. **Removed NONE-floor** — the paper/reference always maps. Our old `min_score`
   low-confidence unmapped path was removed from `cascade` scoring.
6. **Paper retrieval defaults** — `cascade` now defaults to CLIP text retrieval
   (`open_clip`, `hf-hub:apple/DFN5B-CLIP-ViT-H-14`) with `k=10` and the reported
   thresholds above. SentenceTransformer retrieval is an ablation backend.

### Behaviour before → after (both paths)

```
a beautiful oriole bird     Edificio España   →  Bird
the eiffel tower at night   (hijack)          →  Eiffel Tower
red chicken                 (hijack)          →  chicken
tūī                         t (→ spurious)    →  tūī
zzqq nonsense               some node         →  top-scoring taxonomy node
exact_match measure         hF 0.9294         →  hF 0.9294 (unchanged)
```

## 5. Faithfulness status

| Stage | Reference | Ours |
|---|---|---|
| normalize keeps Unicode | yes | ✅ (reuses `scores.normalize`) |
| in-top-k = exact equality | yes | ✅ |
| n-gram top-k = partial overlap | yes | ✅ |
| n-gram all-nodes = full-label match | yes | ✅ |
| n-gram range N | 4,3,2 | yes |
| voting ambiguity | `(softmax_top1 - softmax_topk) / k`, top-2 gap | yes |
| voting thresholds / votes | `0.0015` / `0.001` / `>=4` | yes |
| top-score fallback | always maps | yes |
| retrieval measure | CLIP text-to-text | yes |

**Notes / Open Enhancements**

- **Alias rows** — the current taxonomy index maps aliases to canonical paths, but the
  `cascade` top-k retrieval indexes node labels. Embedding aliases as separate rows is a
  future enhancement if we want broader synonym recall.

## 6. The `cascade` measure in practice

- **Node embeddings** are computed once for the taxonomy node labels and cached by
  backend, model, and node set, then reused across runs.
  **Cache-location precedence** (commit `3bda589`): explicit `cache_dir` arg →
  `$OVEN_NODE_EMB_DIR` → repo-local `data/processed/node_emb/`. On the cluster,
  point `$OVEN_NODE_EMB_DIR` at `$WORK` so the cache lives off the (full) FAST scratch.
- **The cache is robust** (`embedding_matcher.py`): writes are **atomic**
  (`<name>.tmp` + `os.replace`), so an interrupted write never leaves a corrupt
  cache; a **write failure is non-fatal** (full/over-quota filesystem → warn and
  continue *without* caching); an **unreadable/partial cache is recomputed**, not
  crashed on.
- **Prediction embeddings** are computed per run over unique rollout strings and reused
  through the prediction→node mapping dictionary.
- **Aggregation over rollouts:** per example, every unique rollout is mapped and scored;
  the row-level `cascade_hF` is the Best-of-N selected rollout. The full deduped
  per-rollout records are persisted as `cascade_rollout_metrics` so later analyses can
  compute means, variances, quantiles, near-miss credit, and judge-conditioned slices.
  Full treatment: `docs/methods/rollout-hierarchical-metrics.md` and thesis note `002`.
- The results JSON includes `mapping_methods` (counts of exact / ngram / voting /
  top_score for selected cascade rollouts) so the mapper's contribution is auditable.
  Since the paper-aligned cascade always maps, `none` should not appear as a cascade
  mapping method.

## 7. Operational entry points

Cascade flags live in `scripts/score_predictions.py`: `--map-top-k`,
`--embed-backend`, `--embed-model`, and `--embed-device`. Defaults are
`--embed-backend open_clip`, `--embed-model hf-hub:apple/DFN5B-CLIP-ViT-H-14`,
and `--map-top-k 10`.

Operational launch recipes live in `docs/operations/scoring-runbook.md`.

## 8. Files

- `src/oven_mllm_eval/matching.py` — `TaxonomyMatcher` (the cascade), `_normalise`.
- `src/oven_mllm_eval/embedding_matcher.py` — node-embedding cache + cosine retrieval.
- `src/oven_mllm_eval/scoring.py` — measure routing, embedding pass, method counts,
  Best-of-N rollout scoring, and persisted `<measure>_rollout_metrics`.
- `src/oven_mllm_eval/measures.py` — `DirectMeasureMatcher` for `exact_match`,
  `contained`, `bleu`, `meteor`, `rouge`, and `sentence_bert`.
- `scripts/score_predictions.py` — CLI (`--measure cascade`, `--map-*`).
