# 001 — Model scale vs. pass@k: the "smaller-is-broader" coverage effect is a judge artifact

- **Date:** 2026-06-23 (revised same day after correcting the support set — see §10 changelog)
- **Status:** Active finding. **Headline reversed after a methodology fix:** under a specificity-preserving support set, there is *no* small-model coverage advantage — bigger is monotonically better at every k. The earlier "2B>8B inversion survives" claim was an artifact (judge leniency + crediting under-specific predictions). The three runs are confirmed apples-to-apples. **(rev 4, 2026-06-25) corrected-metrics re-scoring corroborates and *generalizes* the finding (§4.5):** with both metric paths fixed (evidence-aware judge + graded `cascade` Best-of-N hF), the effect is an **oracle/lenient-aggregation artifact** — it appears in pass@256 *and* cascade BoN hF, and vanishes (reverses) in pass@1 *and* the strict exact-match rate. A grounded judge lowers inflation but does not flip the ordering.
- **Models:** Qwen3-VL-2B-Instruct, -4B-Instruct, -8B-Instruct (Qwen3-VL technical report, arXiv:2511.21631v2).
- **Task / dataset:** OVEN (Open-domain Visual ENtity recognition; Hu et al., arXiv:2302.11154), validation split, **aligned** taxonomy-question variant (`data/processed/vlm_compatible_val_aligned.jsonl`, **115,552** examples).
- **Prompt:** `concise_no_idk`. **Inference:** naive-sampling, 256 rollouts/example. **Judge:** Qwen3-4B free-form, **no-evidence prompt** (`build_judge_prompt_free_form` — no descriptions, no taxonomy evidence; the coexisting `*_with_desc` pass is NOT used here — see §5). **Scoring:** taxonomy-aware hP/hR/hF + exact, pass@k via unbiased Codex estimator.
- **Source artifacts:** `analysis/audit_judge_false_positives.py`, `analysis/plot_ci_distribution.py`, `src/oven_mllm_eval/judge_audit.py`, `viz/ci_distribution/ci_distribution_supported_aligned_concise_no_idk.png`.
- **Related docs:** `docs/findings/prompt-collapse-and-question-misalignment.md`, `docs/findings/model-diversity-and-dedup.md`.
- **Related notes:** [[002-taxonomy-mapping-and-rollout-metrics]] — the hP/hR/hF mapping cascade and rollout aggregation; its graded crediting of under-specific answers is the complement to this note's specificity-preserving (binary) support set (see 002 §5).

---

## 1. Headline finding

On OVEN (aligned, `concise_no_idk`, 256 rollouts/example, identical decoding across sizes), **the apparent "smaller model is broader at high k" effect is a measurement artifact, not a capability.** Two layers of leniency create the illusion; removing them restores monotonic *bigger-is-better* coverage at every sampling budget.

- **Judge metric (most lenient) — small model looks broader:** at pass@256 the judge-coverage ordering is 2B (0.857) > 4B (0.783) > 8B (0.748). This is the headline an uncritical pass@k plot would report.
- **Specificity-preserving deterministic support (conservative) — bigger is better:** restricting "correct" to rollouts verifiable as *at least as specific as the ground truth* (exact / alias / answer⊆prediction / alias⊆prediction, whole-token — see §5), the pass@256 ordering becomes **8B (0.342) > 4B (0.328) > 2B (0.317)** — monotonic in size. The same monotonic order already held at pass@1 (S p@1: 8B 0.149 > 4B 0.136 > 2B 0.094).

**There is no inversion at any k under the corrected metric.** The judge-coverage ordering runs *opposite* to verified coverage, which identifies it cleanly as judge inflation. The mechanism is quantified in §4: the 2B model wins judge-coverage largely by emitting **under-specific (hypernym) answers** that the judge accepts — 2B's judge-positives are under-specific 19.6% of the time vs ~15% for 4B/8B.

> ⚠️ **Superseded claim.** A prior version of this note claimed "the 2B>8B inversion holds at both ends (judge and supported), so it is real." That was wrong: the "supported" set then included `answer_contains_prediction` (prediction ⊆ answer), which credits under-specific predictions. Once excluded, the inversion vanishes. See §10.

### Decomposition identity (why pass@256 ≈ coverage)

For `k = n` (here n = 256, the full rollout count), the unbiased pass@k estimator reduces exactly to hit coverage:

```
pass@256(example) = 1 if c_i ≥ 1 else 0
pass@256(dataset) = (# examples with ≥1 correct rollout) / (# examples)
```

So "pass@256" is just "fraction of examples solved at least once in 256 tries." This is verified numerically below (`hit/rows` reproduces `p@256` to 3 dp).

---

## 2. The audit table (the primary evidence)

Produced by `analysis/audit_judge_false_positives.py --details` over the three `*_scored.jsonl` files (aligned, `concise_no_idk`, with-image, Qwen3-4B judge). **Support = exact ∨ alias ∨ answer⊆prediction ∨ alias⊆prediction**, all **whole-token** and specificity-preserving (see §5):

| Run | Rows | k | JudgeHit | SuppHit | J p@1 | S p@1 | J p@k | S p@k | J Pos | S Pos | S/J | UndSpec/J | JPos/Hit μ (med) |
|-----|------|---|----------|---------|-------|-------|-------|-------|-------|-------|-----|-----------|------|
| 2B | 115,552 | 256 | 99,060 | 36,609 | 0.265 | 0.094 | 0.857 | 0.317 | 7,851,659 | 2,766,679 | 35.2% | **19.6%** | 79.3 (29) |
| 4B | 115,552 | 256 | 90,519 | 37,932 | 0.360 | 0.136 | 0.783 | 0.328 | 10,651,461 | 4,020,067 | 37.7% | 15.1% | 117.7 (82) |
| 8B | 115,552 | 256 | 86,410 | 39,563 | 0.373 | 0.149 | 0.748 | 0.342 | 11,027,479 | 4,408,778 | 40.0% | 15.8% | 127.6 (110) |

Support breakdown (rollout counts, `--details`):

| Run | Exact | Alias (=) | Pred⊃Ans (answer⊆pred) | Pred⊃Alias (alias⊆pred) | Ans⊃Pred (pred⊆answer, **excluded**) |
|-----|-------|-------|------|------|------|
| 2B | 2,466,457 | 83,905 | 206,839 | 9,478 | 1,538,694 |
| 4B | 3,348,036 | 140,413 | 513,432 | 18,186 | 1,610,328 |
| 8B | 3,703,092 | 137,153 | 546,436 | 22,097 | 1,746,750 |

*(First four columns are summed into S Pos. Alias-containment `Pred⊃Alias` adds little and **increases** with size — 9.5k/18.2k/22.1k — so it does not favor the small model. Whole-token matching shrank `Pred⊃Ans` ~30% vs raw substring by dropping sub-word false matches.)*

**Column definitions:**
- **JudgeHit** — # examples with ≥1 judge-positive rollout. **SuppHit** — same, counting only *supported* (specificity-preserving) judge-positives.
- **J p@1 / S p@1** — mean pass@1 over examples, judge vs supported verdicts.
- **J p@k / S p@k** — mean pass@256; equals `JudgeHit/Rows` and `SuppHit/Rows` respectively (coverage identity).
- **J Pos** — total judge-positive rollouts (examples × 256 draws). **S Pos** — subset that are supported.
- **S/J** — supported share of judge positives. **UndSpec/J** — share of judge positives that are *under-specific* (`Ans⊃Pred`, excluded from support).
- **JPos/Hit μ (med)** — mean (median) judge-positives per hit example: concentration vs. spread.

**Four monotonic-in-size readings (all "bigger is better"):**
1. **Verified coverage** S p@k: 8B 0.342 > 4B 0.328 > 2B 0.317 (and S p@1 same order: 0.149 > 0.136 > 0.094). 
2. **Supported share** S/J: 40.0% > 37.7% > 35.2% — larger models' positives are more often verifiable.
3. **Concentration** median JPos/Hit: 110 > 82 > 29 — when 8B hits it commits (~110/256 rollouts); 2B spreads thin (median 29).
4. **Under-specificity** UndSpec/J: 2B **19.6%** ≫ 4B 15.1% ≈ 8B 15.8% — the smallest model most often emits vaguer-than-GT answers.

**Coverage identity check** (confirms p@256 = coverage): `SuppHit/Rows` = 36,609/115,552 = 0.317 (2B), 37,932/115,552 = 0.328 (4B), 39,563/115,552 = 0.342 (8B) — match S p@k exactly. Same for judge: 99,060/115,552 = 0.857, etc.

**Contrast with the judge metric:** J p@k is *decreasing* in size (0.857 → 0.783 → 0.748), i.e. the judge alone produces the spurious "smaller-is-broader" trend. The reversal between J p@k and S p@k is the core diagnostic of this note.

---

## 3. The cᵢ distribution (mechanism)

Plot: `viz/ci_distribution/ci_distribution_supported_aligned_concise_no_idk.png` (**supported-cᵢ, specificity-preserving**; `analysis/plot_ci_distribution.py --supported`). cᵢ = number of the 256 rollouts that are *supported* positives for example *i*. The histogram y-axis is **% of that model's own supported-solved examples** (cᵢ ≥ 1) — a per-model *conditional* shape, each summing to 100%.

The distribution is **bimodal**: a low-cᵢ tail (fragile single-ish hits) plus a large concentrated bin at 129–256 (model nails it across most rollouts). Both ends now order cleanly by size:

| supported-cᵢ | 2B | 4B | 8B |
|---|---|---|---|
| = 1 (fragile) | **15.3%** | 12.1% | 11.6% |
| 129–256 (committed) | **27.4%** | 40.1% | **42.4%** |
| mean supported-cᵢ | **23.9** | 34.8 | **38.2** |
| supported-solved examples | 36,609 | 37,932 | **39,563** |

CDF annotations (fraction of supported-solved examples with cᵢ ≤ t):
- 2B: cᵢ=1 **15.3%**, cᵢ≤2 22.6%, cᵢ≤4 30.3%
- 4B: cᵢ=1 12.1%, cᵢ≤2 17.9%, cᵢ≤4 24.4%
- 8B: cᵢ=1 11.6%, cᵢ≤2 17.1%, cᵢ≤4 23.4%
- Δ (2B solves / 8B misses, **n=8,365**): mean cᵢ 15.6; cᵢ=1 36.5%, cᵢ≤2 50.6%, cᵢ≤4 63.5%
- Δ (2B solves / 4B misses, **n=7,950**): mean cᵢ 13.8; cᵢ=1 39.4%, cᵢ≤2 54.0%, cᵢ≤4 66.9%

**Interpretation (corrected support set):** The size ordering is now consistent everywhere. 2B has the *fattest* fragile tail (15.3% of its solves are a single supported rollout) and the *smallest* committed bin (27.4%); 8B is the reverse (11.6% / 42.4%), with the highest mean cᵢ (38.2 vs 23.9). So when the larger model solves an example it solves it *hard*; the smaller model's solves are both fewer (lowest supported-solved count, 36,609) and more fragile.

**The 2B-uniquely-solves set is small and thin.** The examples 2B solves but a larger model misses (8B-misses: **n=8,365**; 4B-misses: **n=7,950**) are dominated by single-rollout hits (cᵢ=1 in 36.5%/39.4% of them). I.e. the smaller model's residual unique coverage is marginal and fragile — not a robust capability. Combined with §4, this closes the door on a genuine small-model coverage advantage. *(An earlier draft mis-read this same panel as "the mechanism of a real inversion"; with `answer_contains_prediction` removed there is no inversion to explain.)*

---

## 4. Judge inflation is the whole story (and there is no 4B anomaly)

Comparing the two coverage views — they run in **opposite directions**:

| metric | 2B | 4B | 8B | trend in size |
|---|---|---|---|---|
| **judge** coverage (J p@k) | 0.857 | 0.783 | 0.748 | **decreasing** (smaller looks broader) |
| **supported** coverage (S p@k) | 0.317 | 0.328 | 0.342 | **increasing** (bigger is better) |

Consequences for the thesis:

1. **The "smaller-is-broader" trend exists only under the judge.** It is not a weak-but-real effect that the judge merely exaggerates (as the superseded version claimed) — it *reverses* under specificity-preserving verification. The smaller model does not have broader genuine coverage; it has broader *judge-accepted* coverage.

2. **The mechanism is under-specification.** 2B's judge-positives are under-specific (prediction ⊆ ground truth — a hypernym/parent answer like "cat" for "domestic cat") **19.6%** of the time, vs 15.1% (4B) and 15.8% (8B). The judge — which is told to accept answers "at the level of specificity asked by the question" — is lenient about granularity, so a small model that hedges toward general categories racks up judge-positives that do not survive a specificity check. This is *the* quantified mechanism behind the illusory inversion.

3. **The "4B anomaly" was an artifact and is now resolved.** Under the old (polluted) support set 4B looked anomalously lowest; under the corrected set it sits exactly where size predicts (between 2B and 8B) on every metric. There is nothing left to explain — it was `answer_contains_prediction` inflating 2B above 4B/8B. (Open question removed from §8.)

### Metric-as-bounds framing (corrected)

- **judge pass@k = upper bound** — inflated; accepts unverifiable rollouts, including under-specific (hypernym) answers and some genuine paraphrases.
- **supported pass@k = lower bound** — conservative and specificity-preserving; still *misses* legitimate semantic equivalents the judge correctly accepts (e.g. "jumbo jet" for "Boeing 747"), so it under-counts true positives.
- The truth lies between — but **both bounds are now monotonic in size (bigger ≥ smaller)**. The only place the smaller model leads is the raw judge metric, which is exactly the artifact this note isolates.

---

## 4.5 Corroboration on the corrected metrics (with-desc judge + graded cascade hF) — 2026-06-25

> **2026-07-01 update:** the taxonomy scorer was subsequently restored to the
> paper/`vlm-eval` semantics documented in [[002-taxonomy-mapping-and-rollout-metrics]]:
> CLIP-t2t retrieval, `k=10`, `thr_topk=0.0015`, `thr_top2=0.001`, vote threshold
> `>=4`, no NONE-floor, direct measures over deduped rollout texts, and optimized
> exact-match/cascade execution. Treat the numeric taxonomy table below as historical
> until all `*_with_desc_rich_recomputed.json` summaries are complete. The pass@k
> values are unchanged by taxonomy rescoring because they come only from
> `judge_verdicts`; taxonomy hF/exact/cascade values should be refreshed from the
> recomputed summaries before being quoted as final.

After fixing **both** metric paths — (a) the judge now receives taxonomy evidence (parent/grandparent class names always shown, descriptions where available; `build_judge_prompt_free_form_with_desc`), and (b) the realigned mapping cascade + Best-of-N rollout aggregation of [[002-taxonomy-mapping-and-rollout-metrics]] — the three runs were re-scored (`--measure "exact_match cascade"`) on the **with-desc** judged files. The numbers **do not overturn the headline; they generalize it.**

**Results** (aligned, `concise_no_idk`, 256 rollouts; evidence-aware Qwen3-4B judge; `*_results_qwen_qwen3-4b_with_desc.json`):

| model | pass@1 | pass@256 | cascade BoN hF (hP/hR) | exact-match rate¹ | exact_match hF (num_mapped) |
|---|---|---|---|---|---|
| 2B | 0.179 | **0.797** | **0.880** (0.911/0.857) | 0.598 | 0.808 (55,675) |
| 4B | 0.267 | 0.728 | 0.819 (0.843/0.803) | 0.629 | 0.785 (50,800) |
| 8B | **0.274** | 0.687 | 0.811 (0.832/0.797) | **0.640** | 0.792 (52,745) |

¹ strict `exact` field of the `exact_match` measure. `cascade` is **fully mapped** (num_mapped ≈ 115,552 for all three → no denominator confound); `exact_match` num_mapped differs by model (55.7k/50.8k/52.7k — itself a coverage effect, so read its hF *with* num_mapped, per 002).

**The generalized conclusion — an *oracle/lenient-aggregation* artifact, not merely a judge artifact.** Every metric splits along one axis — *single-shot/strict* vs *oracle-best-of-256/lenient* — now visible on **both** the binary judge axis and the graded taxonomy axis:

| | single-shot / strict → **bigger-is-better** | oracle best-of-256 / lenient → **smaller-looks-broader** |
|---|---|---|
| **judge** (binary) | pass@1: 8B 0.274 > 4B 0.267 > 2B 0.179 | pass@256: 2B 0.797 > 4B 0.728 > 8B 0.687 |
| **taxonomy** (graded hF) | exact-match rate: 8B 0.640 > 4B 0.629 > 2B 0.598 | cascade BoN hF: 2B 0.880 > 4B 0.819 > 8B 0.811 |

Three refinements:

1. **The graded taxonomy metric independently reproduces the effect.** cascade BoN hF (2B **0.880** > 4B 0.819 > 8B 0.811, fully mapped) is the graded twin of judge pass@256 — both are oracle best-of-256 ceilings, both smaller-broader. This is the very leniency §4 isolates, now in an independent hP/hR-graded metric, and it directly answers [[002-taxonomy-mapping-and-rollout-metrics]] §7: **the cascade's graded crediting of under-specific answers, combined with the BoN oracle, *does* reintroduce the small-model "advantage."** cascade BoN is **not** a specificity-preserving metric.

2. **Grounding the judge reduces inflation but does not flip the ordering.** Versus the no-evidence judge (§2), the evidence-aware judge shifts pass@k *uniformly down* (pass@1 ≈ −0.09–0.10, pass@256 ≈ −0.06 per model) yet preserves both orderings (pass@1 bigger-better; pass@256 smaller-broader). So the smaller-broader effect is **robust to judge strictness** — a coverage/diversity property any oracle aggregation surfaces, not something a better judge prompt removes. (Refines the §8 grounded-judge TODO.)

3. **The strict signals stay monotonic bigger-is-better** on the corrected pipeline too: pass@1 and the strict exact-match rate both order 8B > 4B > 2B. 001's single-shot headline holds.

**Caveats — do not over-read:**
- These are the **with-desc judge** condition — a *different* judge from §2's no-evidence baseline. §2's no-evidence J/S audit table is unchanged and remains the reference for the deterministic supported-coverage reversal.
- **Not yet recomputed on these verdicts:** the deterministic `judge_audit` **supported coverage** (the specificity-preserving reversal) and the **mean hF** (002's honest single-shot counterpart to BoN). Until the mean exists, cascade BoN is an oracle ceiling, not deployable accuracy.
- **Prompt-version attribution:** synced shard metadata records `judge_with_desc=True` but not the prompt *version*. That these verdicts use the *always-show-labels* fix (not the older description-gated prompt) is taken from run intent — confirm the judge run postdates the `judge.py` prompt change before resting a thesis claim on it.
- exact_match **hF** (2B 0.808 > 8B 0.792 > 4B 0.785) is **denominator-confounded** (differing num_mapped) and selection-biased; the clean strict signal is the exact-match **rate** (8B>4B>2B), not its hF. This is **not** a resurrected 4B anomaly.

---

## 5. Full experimental configuration (reproducibility)

### Models (Qwen3-VL technical report, arXiv:2511.21631v2)
| Model | Vision encoder | LLM backbone |
|-------|----------------|--------------|
| Qwen3-VL-2B-Instruct | SigLIP2-Large (300M) | Qwen3 small backbone |
| Qwen3-VL-4B-Instruct | SigLIP2-Large (300M) | Qwen3 backbone |
| Qwen3-VL-8B-Instruct | **SigLIP2-SO-400M** (same as 32B/MoE flagship) | Qwen3 backbone |

Shared across the three: DeepStack multi-level ViT→LLM feature injection, Interleaved-MRoPE, 256K native context, Strong-to-Weak Distillation from the 235B teacher (off-policy + on-policy KL). **2B and 4B share the smaller encoder; only 8B gets the larger one.**

### Datasets
- **aligned** (used here): `data/processed/vlm_compatible_val_aligned.jsonl`, **115,552** examples. Questions regenerated from each entity's Wikidata `P279` taxonomy parent so the question granularity matches the hierarchical-eval taxonomy (`scripts/build_aligned_questions.py`). See `docs/findings/prompt-collapse-and-question-misalignment.md`.
- **original**: `data/processed/vlm_compatible_val.jsonl`, **126,199** examples (OVEN's source-dataset super-category questions).

### Prompt variants (verbatim, `src/oven_mllm_eval/prompts.py`; `{}` = OVEN question)
- `concise_no_idk` **(used here):** `"{} Answer questions directly and concisely. If you don't know, give your best guess."`
- `concise` (forces refusals, caused answer collapse — see `docs/findings/prompt-collapse-and-question-misalignment.md`): `"{} Answer questions directly and concisely. If you don't know, say 'I don't know'."`

### Inference (from run `*_metadata.json`, identical across 2B/4B/8B)
- Engine: vLLM offline `LLM.chat()`; `tensor_parallel_size=1`, `gpu_memory_utilization=0.95`, `max_model_len=1024`, `max_num_seqs=4096`, `max_pixels=262144` (512×512), `min_pixels=65536` (256×256), `enforce_eager=false`.
- Sampling: **`temperature=1.0, top_p=1.0, top_k=-1, max_tokens=16, n=256`** — i.e. unrestricted, identical decoding for all sizes. *(Note: this differs from the Qwen report's own eval config — see §6.1.)*
- Method: `naive-sampling`, `samples_per_example=256`.

### Judge (Qwen3-4B, text-only, **NO-EVIDENCE prompt**; from `*_judged*_shard*_metadata.json` + launch command)
- Model: **Qwen/Qwen3-4B**; mode: **free-form** (outputs `<answer>0</answer>`/`<answer>1</answer>`, last-occurrence parse; `src/oven_mllm_eval/judge.py:parse_free_form_output`).
- Sampling: `temperature=0.7, top_p=0.8, top_k=20, n=1, max_tokens=16`; `structured_outputs=false`, `chat=true`, `thinking_disabled=true`.
- Engine: `max_model_len=1024`, `max_num_seqs=8192`, `gpu_memory_utilization=0.92`, 4 GPUs (strided shards).
- **Prompt: `build_judge_prompt_free_form` — the *no-evidence* variant.** The judge sees only the question, ground-truth answer, and rollout text — **no Wikidata descriptions and no taxonomy/label-chain evidence.** It judges semantic equivalence at the question's level of specificity, NOT correctness.
- **⚠️ Which judge pass these numbers use (verified, not assumed):** each run dir *also* contains a separate `*_with_desc` judge pass (`judge_with_desc: True`, `desc_chains=data/raw/oven_wikidata_chains_cleaned_descs.jsonl`, `taxonomy_index=…`, prompt `build_judge_prompt_free_form_with_desc`). **Every number in this note is from the NO-EVIDENCE pass.** Provenance check: the `*_scored.jsonl` consumed by the audit/plot is byte-for-byte derived from the base `*_judged.jsonl` (matching `data_id`, `judge_verdicts`, and `judge_selected_text` — e.g. `oven_entity_val_00000000`: cᵢ=18, selected="miami marlins stadium" in both). The with-desc pass is **not** scored here; comparing the two judges is the §8 TODO.
- **Authoritative prediction:** `judge_selected_text` was used by the old taxonomy scoring path. **Superseded:** current scoring maps deduped `all_texts` for direct measures and `cascade`; the judge is used for pass@k and judge-conditioned diagnostics only. See [[002-taxonomy-mapping-and-rollout-metrics]].

### Scoring & metrics

> **REVISED — hF metric construction changed (2026-06-24; restored to paper semantics on 2026-07-01).** The hP/hR/hF scoring below no longer maps the single `judge_selected_text`: `exact_match` and the other direct measures score deduped rollout texts directly, independent of `judge_verdicts`; `cascade` maps deduped rollout texts with the paper-aligned CLIP retrieval cascade and Best-of-N aggregation. **The headline is pass@k / supported-coverage based (judge verdicts + the deterministic `judge_audit` set), so it is unaffected by taxonomy rescoring.** The §4.5 taxonomy numbers predate the final 2026-07-01 scorer and should be refreshed from recomputed summaries before final use. Still pending: the `judge_audit` supported coverage and mean hF on the with-desc verdicts. The two bullets below describe the *pre-revision* construction.

- Taxonomy-aware hP/hR/hF + exact via `DirectMeasureMatcher` (`src/oven_mllm_eval/measures.py`), taxonomy index `data/processed/oven_taxonomy_index.json`, default measure `exact_match`. *(Pre-revision description — see PENDING note above.)*
- pass@k: unbiased Codex estimator (Chen et al., 2021), product form, `src/oven_mllm_eval/pass_at_k.py`: `pass@k = 1 − ∏_{i=0}^{k-1}(n−c−i)/(n−i)`. *(Unchanged.)*

### "Supported" classification (deterministic, specificity-preserving, whole-token; `src/oven_mllm_eval/judge_audit.py`)
After an aggressive `normalize` (lowercase, strip `<…>` HTML/markup, collapse non-alphanumerics to single spaces), `classify_positive` labels a judge-positive rollout (first match wins). **The first four count as supported** (`SUPPORTED_CATEGORIES` / `is_supported`):
- `exact` — `normalize(pred) == normalize(answer)`. **[supported]**
- `alias` — `normalize(pred)` equals one of the answer's Wikidata aliases (exhaustive over **all** aliases; taxonomy index `aliases` map). **[supported]**
- `contains_answer` — `answer ⊆ pred` (whole-token): prediction is **at least as specific** as the GT. **[supported]**
- `contains_alias` — some alias `⊆ pred` (whole-token): prediction contains an equivalent name of the GT. **[supported]**
- `answer_contains_prediction` — `pred ⊆ answer` (whole-token): prediction is a **fragment of / less specific than** the GT (hypernym/parent). **[detected but EXCLUDED]**

**Whole-token containment (`_phrase_in`):** since `normalize` yields single-space-separated alphanumeric tokens, containment is tested by space-padding (`f" {needle} " in f" {haystack} "`), so only *contiguous whole-token runs* match — `" cat "` ∉ `" caterpillar "`, `" 50 "` ∉ `" 1950 "`, while `" golden retriever "` ∈ `" a golden retriever dog "`. This is word-boundary matching with no regex/tokenizer dependency, and it removed ~30% of the old raw-substring `contains_answer` matches.

**Alias-containment seeds are floored (`_is_alias_seed`):** for `contains_alias` only, aliases that are empty, purely numeric, or <4 chars (`""`, `"50"`, `"au"`, `"707"`) are barred from seeding containment — they substring-match unrelated text. They remain usable for *exact* `alias` match. (Of 3,633 aliases, 42 are ≤3 chars including an empty string, and 805 are single-token.)

**Why `answer_contains_prediction` is excluded:** on a fine-grained entity task a prediction that is a strict sub-span of the GT ("cat" for "domestic cat", "Boeing 747" for "Boeing 747-400") is *under-specific* — it names a parent class, not the target entity. Counting it credits vagueness and was the dominant inflation source (it flipped 2B above 8B at rev 2). It is still *reported* as the `UndSpec` diagnostic (§2, §4) because the under-specific *rate* is itself evidence.

Caveats: (i) whole-token matching does not stop a short alias that is a legitimate standalone token in an unrelated prediction ("right token, wrong entity") — the seed floor mitigates this. (ii) `normalize` here is intentionally stricter/different from `oven_mllm_eval.scores.normalize` (which only lowercases + strips punctuation).

---

## 6. Qwen3-VL report context — what supports the finding and what doesn't

From the Qwen3-VL technical report (arXiv:2511.21631v2). **Read critically.** After the §10 correction the headline is a *metric-validity* result and needs none of these hints; they're relevant only to the secondary observations (8B's reliability/concentration, 2B's under-specificity). One "hint" is a confound we already removed.

### 6.1 Sampling hyperparameters — a CONTROLLED confound, not evidence
The report's eval table (small models `top-k=40, presence_penalty=2.0`; large `top-k=20, presence_penalty=1.5`) is **Qwen's own evaluation config for the flagship comparison — NOT what we ran.** We held decoding **identical** across all sizes (`top_k=-1, T=1.0, top_p=1.0`). The *broader judge coverage* of 2B is therefore not a decoding artifact — but note it is **broader judge-accepted coverage, not broader verified coverage** (§4); much of it is the under-specific tail. A flatter, less-peaked 2B distribution plausibly produces more diverse tail tokens under fixed decoding, but that diversity skews toward vaguer answers the judge tolerates. → If cited at all, frame as "flatter distribution → more under-specific diversity," not "broader capability." Drop the hyperparameter table.

### 6.2 Strong-to-Weak Distillation → softer, higher-entropy student — PLAUSIBLE (for under-specificity, hypothesis)
2B/4B/8B are distilled from the 235B teacher (off-policy + on-policy KL). A capacity-limited student cannot reproduce the teacher's sharp peaks → softer posterior → more diffuse, often vaguer outputs. Consistent with 2B's higher under-specific rate (19.6% vs ~15%) and fatter cᵢ=1 tail. **Caveat:** all three are distilled, so the differentiator is *student capacity to absorb the peaks*, not the recipe. Unverifiable from our data — hypothesis.

### 6.3 Vision encoder size — CONCRETE for the (now monotonic) reliability/coverage ordering
8B's larger SigLIP2-SO-400M encoder → better visual grounding on a *visual entity recognition* task → higher pass@1, higher verified coverage, and concentrated correct mass. With the corrected support set the whole 2B<4B<8B ordering is monotonic, so it is at least *consistent* with capacity (LLM + encoder) increasing with size. **Caveat:** 2B and 4B share the *same* encoder yet 4B clearly out-covers 2B, so the encoder is not the sole driver — the LLM-backbone scale and distillation outcome matter too.

### 6.4 Square-root pretraining reweighting — DROP / heavy hedge
Family-wide loss-balancing choice (text vs multimodal). No path to *differential* size effects on OVEN. Leaning on it would be over-claiming.

### 6.5 Report empirics — partial fit, with a caution
Report notes even the 2B has strong reasoning (e.g. MathVista-mini 73.6) while 8B leads on absolute Pass@1. The "small model knows but is poorly calibrated to rank #1" narrative is **only half-applicable here**: 8B leads at *both* pass@1 and verified pass@256, so on OVEN the smaller model does not even have superior latent coverage to mis-rank — its apparent high-k edge was a judge artifact (§4), not a calibration gap. Cite this narrative cautiously and do not let it imply a real 2B coverage advantage.

---

## 7. Suggested thesis framing (draft paragraph)

> Under identical decoding across model sizes (T=1.0, top-p=1.0, top-k=−1, 256 samples per example), a naive LM-judge pass@k metric reports that the smaller Qwen3-VL model achieves *broader* high-k coverage on OVEN than the larger ones (judge pass@256: 2B 0.857 > 4B 0.783 > 8B 0.748). This trend is an artifact of judge leniency about answer granularity. When "correct" is restricted to rollouts that are deterministically verifiable *and at least as specific as the ground truth* (whole-token exact, alias, answer-contained-in-prediction, or alias-contained-in-prediction), the ordering reverses to monotonic bigger-is-better at every sampling budget (supported pass@256: 8B 0.342 > 4B 0.328 > 2B 0.317; supported pass@1 in the same order). The reversal is driven by under-specification: 19.6% of the 2B model's judge-accepted rollouts name a hypernym/parent of the target entity (e.g. "cat" for "domestic cat"), versus ≈15% for the 4B and 8B models, and the LM judge — instructed to accept answers at the question's level of specificity — admits them. Larger models also commit harder when correct (median 110/256 verified rollouts per solved example for 8B vs 29 for 2B). We therefore find no evidence that smaller VLMs possess broader latent coverage on open-domain visual entity recognition; the appearance of such coverage is a judge-evaluation artifact that a specificity-preserving verification removes.

*(Mechanistic hypotheses from the Qwen3-VL report in §6 — softer distilled-student distributions, larger 8B vision encoder — are consistent with the reliability/specificity ordering but are not needed to state the headline result, which is purely a metric-validity finding.)*

---

## 8. Open questions / TODO

- [x] **Verify the three audited runs are apples-to-apples** — DONE. All three are aligned, `concise_no_idk`, with-image (`no_image=None`), Qwen3-4B judge. 2B = `…2b-instruct/20260614_121741_936810`, 4B = `…4b-instruct/20260614_123428_725972`, 8B = `…8b-instruct/20260614_123530_550630`.
- [x] **Tabulate per-hit concentration** — DONE (§2): median judge-pos/hit 29 (2B) / 82 (4B) / 110 (8B), monotonic.
- [x] **4B "anomaly"** — RESOLVED: it was an artifact of `answer_contains_prediction`; under corrected support 4B is monotonic between 2B and 8B. (§4)
- [ ] Inspect the **unsupported judge positives** in the dashboard (`analysis/explore_judgments.py`, now support-aware): of the ~59–64% judge positives that are unsupported, what fraction are legitimate paraphrases (judge correct) vs. judge errors vs. under-specific? Bounds how much the supported lower bound undercounts.
- [x] **Word-boundary containment + alias-containment** — DONE (rev 3, §10). Whole-token matching (`_phrase_in`) replaced raw substring; added `contains_alias` (alias ⊆ pred) over all aliases with a short/numeric seed floor. S p@k essentially unchanged (0.317/0.328/0.342) — headline robust.
- [ ] Qualitatively inspect the residual **2B-solves / 8B-misses supported** set (n=8,319) — what entity types, and are they genuine or judge noise?
- [~] Re-run with the description-augmented judge (`--judge-with-desc`) — **PARTIAL (§4.5):** the evidence-aware judge lowered pass@k uniformly (pass@256 −0.06/model) but did **not** flip the smaller-broader ordering. Still open: recompute `judge_audit` `UndSpec/J` + supported coverage on the with-desc verdicts (to quantify the J-vs-S gap under the grounded judge), and try a Qwen3-8B judge.
- [ ] Compute the **mean hF** ([[002-taxonomy-mapping-and-rollout-metrics]] §4.2) so the cascade BoN ceiling in §4.5 is paired with its honest single-shot counterpart; report the BoN−mean gap per model (the graded analog of the pass@1↔pass@256 spread).
- [ ] Replicate on the **`concise`** prompt and the **original** (un-aligned) dataset to confirm the monotonic result is prompt/dataset-robust.

## 9. References
- Qwen3-VL Technical Report — arXiv:2511.21631v2 (encoders p.3; distillation p.11; sampling p.20; backbone p.2).
- OVEN: *Open-domain Visual Entity Recognition* — Hu et al., arXiv:2302.11154.
- pass@k unbiased estimator — Chen et al., *Evaluating Large Language Models Trained on Code* (Codex), 2021.

## 10. Changelog
- **2026-07-01 (rev 5):** Added a caveat that §4.5 taxonomy numbers predate the final paper-aligned scorer from [[002-taxonomy-mapping-and-rollout-metrics]] (CLIP-t2t, `k=10`, no NONE-floor, direct measures over deduped rollout texts). Clarified that pass@k is unchanged by taxonomy rescoring because it comes from `judge_verdicts`, while taxonomy hF/exact/cascade should be refreshed from recomputed summaries before final cross-scale claims.
- **2026-06-25 (rev 4):** Added **§4.5** with the corrected-metrics re-scoring (evidence-aware `--judge-with-desc` judge + the [[002-taxonomy-mapping-and-rollout-metrics]] `cascade` Best-of-N graded hF) on all three aligned `concise_no_idk` runs. Result: the finding **generalizes** — smaller-looks-broader appears in *both* oracle metrics (judge pass@256 2B 0.797>4B 0.728>8B 0.687; cascade BoN hF 2B 0.880>4B 0.819>8B 0.811) and reverses in *both* strict ones (pass@1 and the exact-match rate, 8B>4B>2B). So it is an **oracle/lenient-aggregation artifact**, not merely a judge artifact; the evidence-aware judge lowers inflation uniformly without flipping the ordering. Answers [[002-taxonomy-mapping-and-rollout-metrics]] §7 (cascade BoN does reintroduce the leniency). §2's no-evidence audit table is **unchanged** (still the supported-coverage reference); `judge_audit` supported coverage and `mean` hF on the with-desc verdicts remain pending. Updated Status, §5 PENDING note, §8 TODOs.
- **2026-06-23 (rev 3):** Hardened the support set with **whole-token containment** (`_phrase_in`; replaces raw substring, e.g. "cat" no longer ⊆ "caterpillar") and added **`contains_alias`** (alias ⊆ pred, exhaustive over all aliases, with a short/numeric seed floor `_is_alias_seed`). Net effect on the headline: negligible — S p@256 = 0.317/0.328/0.342 (2B/4B/8B), still monotonic bigger-is-better; whole-token matching trimmed ~30% of raw-substring `contains_answer` matches, alias-containment added little and scaled *with* size. UndSpec/J 19.6/15.1/15.8%. Audit table, breakdown (now incl. `Pred⊃Alias`), §3 histogram, and the supported-cᵢ plot regenerated. **Conclusion unchanged and more robust.**
- **2026-06-23 (rev 2):** Corrected the support set to be **specificity-preserving** — dropped `answer_contains_prediction` (prediction ⊆ ground truth) from the verified-support definition (`src/oven_mllm_eval/judge_audit.py:SUPPORTED_CATEGORIES`; reflected in `audit_judge_false_positives.py` and `plot_ci_distribution.py --supported`). Impact: supported pass@256 fell to 0.318/0.330/0.344 (2B/4B/8B), **reversing the previous "2B>8B" supported inversion into a monotonic bigger-is-better trend**. Rewrote §1–§5, §7; resolved the "4B anomaly." All `S*` numbers, the supported breakdown, and `viz/ci_distribution/ci_distribution_supported_aligned_concise_no_idk.png` regenerated. Judge-metric (`J*`) numbers unchanged.
- **2026-06-23 (rev 1):** Initial note (support set included `answer_contains_prediction`; claimed the inversion survived under support — **superseded**).
