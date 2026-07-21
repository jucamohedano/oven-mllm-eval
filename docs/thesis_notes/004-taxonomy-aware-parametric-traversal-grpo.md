# 004 — Taxonomy-Aware Parametric Traversal: GRPO Post-Training for Open-World Visual Entity Recognition

- **Date:** 2026-07-08 (updated)
- **Status:** Complete arc. GRPO does not internalise taxonomy traversal (≈30 flat runs, Sections 9–11; trav10 finished all 250 steps flat, entropy collapse to 0.03), but explicit traversal prompting **expands access to parametric knowledge at test time** (Section 11.3). Conclusion in Section 12. Pending: judge re-eval, reruns of two failed val_only jobs.
- **Primary sources:**
  - Venkatraman et al. — *Recursive Self-Aggregation Unlocks Deep Thinking in Large Language Models*, arXiv:2509.26626 (2026). Local: `../resources/2509.26626v2.pdf`.
  - Zhang, Kaniselvan, Schaeffer, Mireshghallah — *Reinforcement Learning Improves Traversal of Parametric Knowledge in LLMs*, arXiv:2511.05933v2 (2026).
  - Gekhman, Aharoni, Ofek, Geva, Reichart, Herzig — *Thinking to Recall: How Reasoning Unlocks Parametric Knowledge in LLMs*, arXiv:2603.09906 (2026).
  - Snæbjarnarson, Mahammad, Stemann, Labatie, Schlangen — *Taxonomy-Aware Evaluation of Vision-Language Models*, CVPR 2024 / arXiv:2504.05457.
  - Zang, Li, Zhao, Zhu, Liu — *On Large Multimodal Models as Open-World Image Classifiers*, arXiv:2503.21851 (2025).
- **Our artifacts:**
  - Reward function: `verl/verl/utils/reward_score/oven_boxed.py` (exact-match baseline + shaped auxiliary rewards)
  - Parquet builder: `oven-mllm-eval/scripts/build_verl_oven_parquet.py` (standard / aggregation / traversal prompt types)
  - Subset generation: `oven-mllm-eval/scripts/generate_train_subset_balanced.py`
  - Unlockable mining: `oven-mllm-eval/scripts/mine_unlockable_examples.py`
  - GRPO launch: `verl/examples/grpo_trainer/run_qwen3_vl_oven_rsa_trace_grpo.sh` + `schedule_qwen3_vl_oven_rsa_trace_grpo.sh`
  - Operations: `oven-mllm-eval/docs/operations/training-data-runbook.md`
- **Related notes:** [[003-recursive-self-aggregation]] (RSA method + OVEN adaptation), [[002-taxonomy-mapping-and-rollout-metrics]] (hP/hR/hF and cascade linker), [[001-model-scale-coverage-vs-reliability]] (under-specificity problem).

---

## 1. Thesis positioning

A naïve combination of Recursive Self-Aggregation (RSA) and GRPO would yield a thesis contribution that is merely "RSA + OVEN": train a VLM on RSA-generated traces with a boxed exact-match reward. The contribution we aim for is stronger:

> **Taxonomy-aware parametric traversal for open-world visual entity recognition.**
>
> RSA-style test-time compute reveals that the VLM often has the correct entity somewhere in its sampling distribution. GRPO then trains the model to access that answer more reliably through structured visual-semantic traversal.

The core scientific question becomes:

> Can RL teach a VLM to better access and traverse its latent visual-semantic knowledge for open-world entity recognition?

This reframing is motivated by three convergent findings from the recent literature:

1. **RL improves knowledge traversal, not knowledge injection** (Zhang et al., 2511.05933). Layerwise activation analysis shows that factual representations maintain high cosine similarity between instruct and reasoning models, while query representations diverge noticeably. RL primarily reshapes *how models navigate* existing knowledge rather than the knowledge representation itself. Structured prompting—which explicitly guides models through hierarchical traversal—recovers most of the instruct–reasoning gap across five model families.

2. **Reasoning unlocks parametric knowledge through two mechanisms** (Gekhman et al., 2603.09906): a computational buffer effect (the model uses generated reasoning tokens to perform latent computation independent of their semantic content) and factual priming (generating topically related facts acts as a semantic bridge to the correct answer). Critically, hallucinated intermediate facts increase the risk of wrong final answers—so verification and gating are essential.

3. **RSA exposes latent candidate entities through test-time population search** (Venkatraman et al., 2509.26626). When the base model samples a population of candidate answers, the correct entity often appears among them even when greedy decoding fails. This gap—between greedy accuracy and best-of-N accuracy—defines the "unlockable" knowledge we aim to internalize.

Together, these papers motivate a framework where GRPO post-training teaches a VLM to perform a structured search procedure: **visual evidence → coarse category → candidate entities → taxonomy traversal → specific entity**.

---

## 2. From "RSA + OVEN" to parametric traversal

### 2.1 The desired behaviour

Instead of training the model only to answer or aggregate RSA traces, we train it to perform this procedure:

```
image → visual evidence → broad category → candidate entities → taxonomy traversal → final OVEN/Wikidata entity
```

The model should learn to *search visually, retrieve relevant latent semantic/taxonomic knowledge, compare candidates, traverse toward the most specific supported OVEN node, and output the entity.*

### 2.2 Unlockable knowledge

A key diagnostic concept from the Thinking to Recall paper (Gekhman et al.) is that knowledge can be *present but inaccessible*. We operationalize this via the RSA candidate solution traces generated at temperature 1 with population 16 (`run_recursive_self_agg.py --candidate-format solution --steps 1 --temperature 1 --population 16`).

For each training entity, we classify it into one of three buckets:

| Bucket | Definition (out of 16 stochastic rollouts) | Meaning |
|---|---|---|
| **Easy** | All parsed rollouts correct | Knowledge is reliably accessible; training on these may not help |
| **Unlockable** | 1–15 rollouts correct | Knowledge is present in the sampling distribution but not reliably selected—**primary RL target** |
| **Inaccessible** | 0 rollouts correct | Knowledge is not reachable under sampling; exact-match GRPO cannot help |

The unlockable subset directly tests the traversal paper's claim: if the answer appears somewhere in base samples, can GRPO make the model output it directly more often? This is the "parametric knowledge surfacing" hypothesis.

From our 6,087 training entities with RSA solutions (T=1, N=16): 442 (7.3%) are easy, 1,603 (26.3%) are unlockable, and 4,041 (66.4%) are inaccessible. The unlockable distribution is skewed toward sparse correct-rollout counts (320 entities with only 1/16 correct, 173 with 2/16), suggesting the knowledge is genuinely difficult to access rather than trivially recoverable.

---

## 3. Prompt design: three training modes

We extend the VERL parquet builder (`build_verl_oven_parquet.py`) with three prompt types, all in `rsa_trace` dataset mode:

### 3.1 Standard prompt

Single-turn RSA trace prompt. The model sees the image and question and is asked to produce a concise solution ending in `\boxed{answer}`. This is the baseline training format—no candidates, no structured schema.

### 3.2 Aggregation prompt

Shows the image, question, and *K=4 candidate solution traces* from the RSA Phase 1 population (T=1 greedy solutions). The model is asked to synthesize the most reliable answer by checking candidates against the image. This trains aggregation behaviour: compare, verify, and select or improve.

Roughly 50% of training rows that have RSA candidates get aggregation prompts (controlled by `--aggregation-fraction`).

### 3.3 Traversal prompt (the contribution)

A structured schema prompt that guides the model through taxonomy-aware search:

```
<visual_evidence>
- 2-4 concrete, directly visible properties
</visual_evidence>

<coarse_category>
- broad category (animal, vehicle, building, food, tool, plant, ...)
</coarse_category>

<candidates>
- 3-5 candidate entities at increasing specificity
</candidates>

<traversal>
- taxonomy path: broader → finer → most specific entity
</traversal>

<decision>
- which visual evidence distinguishes the chosen entity from alternatives
</decision>

\boxed{answer}
```

This is directly motivated by the traversal paper's finding that structured prompting recovers most of the instruct–reasoning gap. The schema forces the model to externalize its search procedure in a parseable format, which enables path-matching rewards and provides interpretable outputs for analysis. Controlled by `--traversal-fraction` (default 0.0, set to 0.33 for balanced three-way split).

---

## 4. Reward function design

The reward function (`oven_boxed.py`) implements a decomposed reward with exact-match dominance and auxiliary shaping signals. All shaping terms are gated to prevent reward hacking.

### 4.1 Exact-match baseline (GRPO-exact)

```
no \boxed{}                → 0.00
\boxed{wrong}              → 0.05
\boxed{exact/alias match}  → 1.00
```

This is the strict boxed exact/alias reward. It tests: can exact-answer GRPO plus RSA traces improve one-shot OVEN entity prediction? It is a necessary baseline but not the contribution.

### 4.2 Shaped reward (GRPO-traversal)

```
R = 0.00

1. Format (5%):
   if has valid \boxed{}:  R += 0.05
   else:                   return 0.00

2. Exact / alias match (70%, dominant):
   if boxed answer matches GT or known aliases:  R += 0.70

3. Conservative taxonomy shaping — specific_hF (15%):
   Map boxed answer → taxonomy node via conservative text-only linker.
   Compare predicted path vs GT path using specificity-weighted hierarchical F1.
   specific_hF penalizes under-specific answers ("dog" for "beagle" → low).

4. Path match (5%, double-gated):
   Only if specific_hF ≥ 0.3 (answer is taxonomically plausible).
   Parse <traversal> nodes from response.
   Jaccard overlap between mentioned nodes and GT taxonomy path.

5. Aggregation improvement (5%):
   Only for aggregation prompt rows.
   Compare boxed answer against candidate answers in extra_info.
   Reward selecting the best candidate (+0.5) or improving beyond all (+1.0).

return min(R, 1.0)
```

**Key design principles:**

- **Exact/alias remains dominant (70%).** The shaping signals cannot be gamed into a high score without getting the answer right.
- **The linker is conservative and text-only.** It uses n-gram Jaccard similarity against all ~12K taxonomy node labels, with a confidence threshold of 0.5. No CLIP, no embeddings, no cascade that always returns a node. If the prediction cannot be conservatively linked, no taxonomy credit is given.
- **The path match is double-gated.** specific_hF must be ≥ 0.3 before any path credit is awarded. This prevents the model from writing beautiful traversal paths while outputting the wrong answer.
- **specific_hF uses depth-weighted suffixes** (decay=0.5): deeper, more specific matches contribute exponentially more than shallow root matches. Under-specific predictions receive a 0.5 penalty multiplier. This directly targets the under-specificity problem documented in [[001-model-scale-coverage-vs-reliability]].

### 4.3 Why specific_hF rather than raw hF

The base Qwen3-VL-4B results on OVEN validation (naive-sampling, exact_match linker) reveal the gap:

| Metric | Value | Interpretation |
|---|---|---|
| hF | 0.590 | Looks reasonable—model is "in the ballpark" |
| specific_hF | 0.412 | Reality: specific answers are much rarer |
| exact | 0.274 | Only 27% hit the exact entity node |
| under_specific_rate | 3.2% | Model predicts broader ancestors |
| mean_depth_delta | −0.41 | Predictions are ~0.4 levels too broad on average |

Standard hF overrates the model because it gives equal weight to shallow ancestor overlap. Predicting "dog" for "beagle" yields hF ≈ 0.67 because every predicted ancestor exists in the GT path—the model sounds correct but isn't specific. We use specific_hF for reward because it penalizes this behaviour through depth weighting and the under-specificity penalty. The 0.41 mean depth delta means the model needs roughly half a level more specificity—exactly what the traversal reward should incentivize.

The cascade linker (CLIP retrieval + taxonomy cascade) is explicitly NOT used for reward because its mapping coverage of 99.98% comes with a 10.4% under-specific rate—three times the exact_match linker's rate. Cascade hF of 0.847 would reward the model for vague answers that happen to overlap taxonomically.

---

## 5. Training data

### 5.1 Source

OVEN training data is built from the aligned and balanced JSONL (`vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl`, 250k rows, 6,087 unique entity QIDs). The balanced sampling caps per-entity row counts via entity-ID hashing to prevent oversampling of high-frequency entities.

### 5.2 VERL parquet construction

`build_verl_oven_parquet.py` in `rsa_trace` mode consumes the balanced JSONL plus RSA candidate solution traces and produces `train.parquet` and `val.parquet`. Training rows are split across prompt types (standard / aggregation / traversal). Validation rows always get standard prompts. The taxonomy lineage is stored in `extra_info.taxonomy_labels` (leaf-to-root Wikidata chain).

### 5.3 val_unseen filtering

The raw val split (`val.parquet`) contains both `entity_val_seen` (different images of training entities, 59,315 rows) and `entity_val_unseen` (truly held-out entities, 56,237 rows). For GRPO validation, we filter to unseen-only to measure genuine open-world generalization. The filtered `val_unseen.parquet` has 56,237 rows across 1,460 entity QIDs, with zero direct QID overlap with the training set (verified via intersection check). This is critical: any improvement on val_unseen measures generalization, not memorization.

### 5.4 Training subsets

| Subset | Rows | QIDs | Purpose |
|---|---|---|---|
| `train_2k_balanced` | 2,000 | 2,000 | GRPO-exact baseline: diverse entities, all 341 taxonomy roots covered |
| `train_2k_balanced_unlockable` | 525 | 525 | GRPO-unlockable: only entities where RSA sampling found the answer but greedy didn't |
| `train_2k_balanced_easy` | 133 | 133 | Control: entities consistently answered correctly |
| `train_2k_balanced_inaccessible` | 1,342 | 1,342 | Diagnostic: entities unreachable under sampling |
| `verl_oven_rsa_traversal_...2k` | 2,000 | 1,710 | GRPO-traversal: 640 traversal / 700 aggregation / 660 standard |

All subsets are entity-deduplicated (one row per entity) and stratified across taxonomy roots (guaranteed ≥1 entity per root in Phase 1, proportional filling in Phase 2).

---

## 6. Experiment matrix

All experiments use **Qwen3-VL-4B-Instruct** with LoRA (rank 64, alpha 32), FSDP2 across 2 GPUs, vLLM rollout (TP=1, 2 agent workers), TRAIN_BATCH_SIZE=16, 300 training steps, and val_unseen for evaluation. Config varies by version (see §6.2).

### 6.1 v1 experiments (completed — no learning observed)

All three v1 runs finished mechanically stable but **reward was flat** (mean ~0.05–0.22, no upward trend). KL exploded late in GRPO-exact. Diagnosis: (a) prompts asked for reasoning from a 4B model without showing correct examples, (b) aggregation candidates were uncontrolled difficulty (all-wrong = impossible task), (c) ROLLOUT_N=4 gave poor advantage estimation, (d) vLLM KV cache OOM caused hangs, (e) TOTAL_EPOCHS wasn't exported causing early termination.

| # | Experiment | Reward | Mean reward | KL | Verdict |
|---|---|---|---|---|---|
| 1 | GRPO-exact | Exact/alias | 0.05–0.15, late KL explosion to 0.08 | Worst |
| 2 | GRPO-traversal (exact) | Exact/alias | 0.15–0.22, KL ~0.02 | Stable but flat |
| 3 | GRPO-traversal (shaped) | Shaped | 0.15–0.22, KL ~0.018 | Best of three, still flat |

### 6.2 v2 experiments (launched 2026-07-03 — results pending)

**Fixes applied (paper-backed):**
- **1-shot examples** (Physics of LLMs 3.2): every prompt shows a correct traversal example
- **Aggregation filtering**: only emit agg prompts when 1-2 of 4 candidates are correct
- **LCS path match** (Traversal paper §4.2): replaces Jaccard with `(F1 + CSS)/2`
- **max_num_seqs=16**: reduces vLLM KV cache from ~18GB to ~4.6GB, prevents hang
- **TOTAL_EPOCHS export fix**: training now runs full 300 steps (was stopping at 125)
- **ROLLOUT_N default increased to 8**: 88% chance of ≥1 correct per group (was 65% with n=4)
- **Separate reward files**: `oven_boxed_exact.py` vs `oven_boxed.py` (shaped)

**Config:** max_prompt_length=5120, max_response_length=128 (256 for traversal), max_num_seqs=16, val_batch_size=8192, save_freq=100, test_freq=150.

| # | Experiment | Train Data | Prompt Types | Reward File | Hypothesis |
|---|---|---|---|---|---|
| 1 | **GRPO-exact v2** | 2k balanced | standard + aggregation (1-shot, filtered) | `oven_boxed_exact.py` | Baseline with fixes: does GRPO learn now? |
| 2 | **GRPO-unlockable v2** | 511 unlockable | standard + aggregation (1-shot, filtered) | `oven_boxed_exact.py` | Data efficiency: does RL work by surfacing latent knowledge? |
| 3 | **GRPO-traversal v2 (exact)** | 2k traversal | standard + aggregation + traversal (1-shot) | `oven_boxed_exact.py` | Does traversal prompt structure alone improve behaviour? |
| 4 | **GRPO-traversal v2 (shaped)** | 2k traversal | standard + aggregation + traversal (1-shot) | `oven_boxed.py` (shaped) | Does shaped reward (specific_hF + LCS path + agg) help? |

### 6.3 v3 experiment (launched 2026-07-03 — results pending)

**Compute-buffer standard prompt** (Thinking to Recall paper): standard rows use "Think step by step, then `\boxed{answer}`" instead of "Reason carefully...". The paper showed dummy filler tokens improve performance — the model needs compute tokens, not necessarily structured reasoning. Hypothesis: simpler instruction → less hallucination → stronger reward signal.

| # | Experiment | Train Data | Standard Prompt | Reward | Hypothesis |
|---|---|---|---|---|---|
| 5 | **GRPO-exact v3 (cb)** | 2k balanced | compute_buffer ("Think step by step…") | `oven_boxed_exact.py` | Compare vs v2: does simpler reasoning improve learning? |

### 6.4 A/B comparisons

- **v1 vs v2**: all fixes combined — did we fix the learning problem?
- **v2 #1 vs #2**: data composition (all vs unlockable-only)
- **v2 #1 vs #3**: prompt structure (standard/agg vs + traversal)
- **v2 #3 vs #4**: reward function (exact vs shaped)
- **v2 #1 vs v3 #5**: standard prompt variant (reasoning vs compute_buffer)

### 6.5 Expected analysis metrics

For each trained checkpoint:
- **Greedy accuracy**: exact match, hF, specific_hF on val_unseen (vs Base)
- **Oracle gap recovered**: `(trained_greedy − base_greedy) / (base_best_of_N − base_greedy)`
- **Traversal quality**: parse `<traversal>` and `<decision>`; compute path_match; measure under-specificity
- **Per-bucket breakdown**: accuracy stratified by unlockability class

---

## 7. Implementation details

### 7.1 Conservative taxonomy linker

The linker in `oven_boxed.py` is text-only and deterministic:

1. Load the OVEN taxonomy index JSON (`oven_taxonomy_index.json`) at module level (cached via `@lru_cache`).
2. Normalize the boxed answer text.
3. Exact match against all node labels and aliases (pre-built `norm_to_original` map).
4. If not found: compute n-gram Jaccard similarity (bigram, unigram) against all ~12K normalized node labels.
5. Accept the match only if Jaccard ≥ 0.5 (conservative threshold—no top-score fallback that always returns a node).
6. Return `(node_label, taxonomy_path)` or `None`.

Unlike the full cascade matcher used for evaluation scoring, this linker does NOT use CLIP/SentenceBERT embeddings, does NOT do voting fallback, and does NOT guarantee a mapping for every input. It maps only when there is reasonably strong text overlap between the answer string and a taxonomy label. This prevents the reward from crediting vague answers that the cascade would generously map to a plausible node.

### 7.2 specific_hF computation

Inlined from `oven_mllm_eval/scores.py` (no dependency on the evaluation package):

- Build ancestor suffix sets from leaf-to-root paths (each suffix anchored at root).
- Weight each suffix by `decay^(total_depth − depth − 1)` with `decay=0.5`. Deepest (most specific) suffixes get weight 1.0; root-only gets weight 0.5^(depth-1).
- Compute weighted precision (fraction of predicted suffix weight explained by GT overlap) and weighted recall (fraction of GT suffix weight explained by prediction overlap).
- Compute weighted F1.
- If the prediction path is a strict ancestor of the GT path (under-specific), multiply by `under_specific_penalty=0.5`.

The depth weighting ensures that getting the leaf node right is worth exponentially more than getting the root right. An exact leaf match yields specific_hF = 1.0; predicting the parent yields specific_hF ≈ 0.33; predicting the root yields specific_hF ≈ 0.02.

### 7.3 Path matching

Parses `<traversal>...</traversal>` tags from the model response. Splits on arrows (`→`, `->`, `>`), commas, or bullet separators. Normalizes each node name. Computes the Traversal paper's (Zhang et al., 2511.05933 §4.2) path matching score:

`PathMatch = (F1 + CSS) / 2`

where CSS (Common Subsequence Score) is the longest common subsequence between predicted and GT path nodes, normalized by GT path length. LCS rewards getting the right nodes in the right taxonomic order, even if intermediate nodes are missing or generic terms are used instead of OVEN-specific labels. This is more forgiving than pure Jaccard (v1), which penalized every missing node equally and ignored order entirely.

The score remains naturally low when the model uses generic English terms instead of OVEN taxonomy terms, creating a gradient toward taxonomy-aligned traversal.

### 7.4 Aggregation improvement

For aggregation-prompt rows only. Compares the boxed answer against the `candidate_final_answers` stored in `extra_info`:

- 1.0: final answer is correct AND all candidates were wrong (model improved beyond the population).
- 0.5: final answer is correct AND at least one candidate was also correct (model correctly identified the right candidate).
- 0.0: final answer is wrong, or no improvement over candidates.

---

## 8. Relationship to prior thesis notes

- [[003-recursive-self-aggregation]] defines the RSA test-time method and the planned aggregation-aware RL. This note (004) describes the implemented RL framework that goes beyond aggregation-aware RL into *taxonomy-aware parametric traversal*, with structured prompts, shaped rewards, and the unlockable-knowledge diagnostic.
- [[002-taxonomy-mapping-and-rollout-metrics]] defines the hP/hR/hF metrics and the cascade linker. This note explains why cascade hF is too permissive for online RL reward and why we use a conservative text-only linker + specific_hF instead.
- [[001-model-scale-coverage-vs-reliability]] documents the under-specificity problem (smaller VLMs hedge toward hypernym answers). The specific_hF reward with under-specificity penalty directly targets this failure mode.

---

## 9. Changelog

- **2026-07-02:** Initial note. Captures the full design rationale, implementation, and launched experiment matrix. Results pending.
- **2026-07-03:** v1 results: flat reward across all three runs (mean 0.05–0.22, no upward trend). KL explosion in GRPO-exact. Root cause analysis: missing 1-shot examples, uncontrolled aggregation difficulty, ROLLOUT_N=4, vLLM KV cache OOM, TOTAL_EPOCHS not exported. v2 launched with six paper-backed fixes (1-shot from Physics of LLMs 3.2, filtered aggregation, LCS path match from Traversal paper, max_num_seqs=16, total_epochs export, separate reward files). v3 compute-buffer variant launched (simpler "Think step by step" standard prompt, from Thinking to Recall's computational buffer finding). Results pending.
- **2026-07-07:** Reward-function overhaul (fuzzy matching, char-bigram linker fallback, path-match format gate, prompt_type reward gating) and Zhang-style prompt redesign. Three runs (agg05, agg08, trav08) trained to 500 steps; results flat across all metrics. Full analysis in Section 10.
- **2026-07-08:** Diagnosed the traversal-run failures as response truncation (clip_ratio 0.66–0.78 at 256 tokens). Ran the fix: trav10 (traversal-only, unlockable subset, 768 tokens, lr 1e-5, ungated format reward) validated on a dedicated traversal-prompt set, plus a base-model elicitation battery. Results: traversal scaffolding **expands access at test time** (base pass@8 exact∪fuzzy 0.31 vs standard greedy 0.175; specific hF 0.48 vs 0.27; the positive claim), but GRPO on top is flat on accuracy while the format saturates and entropy collapses — RL sharpens the prompt-elicited behaviour without expanding access. Output inspection (per supervisor advice) rules out a degenerate reward hack but shows the shaped reward is partly form credit and deeper traversals hallucinate. Full analysis in Sections 11–12. Pending: judge re-eval; reruns of the failed elicit-std and trav08-checkpoint jobs.
- **2026-07-08 (later):** trav10 completed all 250 steps — validation accuracy flat throughout (exact∪fuzzy ends 0.211, specific hF 0.338), while train entropy collapsed 0.59→0.03 and KL rose to ~8.5 with no train-reward gain. Textbook GRPO variance-collapse; confirms Section 12 conclusion. Figure/table/examples refreshed to step 250.

---

## 10. 2026-07-07 — Fuzzy reward, prompt redesign, and the aggregation/traversal runs

### 10.1 Reward-function changes (goal: surface latent parametric knowledge)

The v2 exact-match reward was too brittle to give a useful gradient: 97.8% of
validation entities have no aliases, so any near-miss earned zero reward. Five
changes were made to `verl/verl/utils/reward_score/oven_boxed.py` and
`scripts/build_verl_oven_parquet.py`, all aimed at rewarding *approach* to the
correct specific entity rather than only exact hits.

1. **Fuzzy character-bigram matching.** An answer earns the full `EXACT_REWARD`
   (0.70) on exact/alias match *or* on character-bigram Jaccard ≥ 0.50 (gated:
   ≥2 content tokens on both sides, length ratio ≥ 0.5). Character bigrams (vs
   word bigrams) resist typos and spelling variants because a single character
   error only perturbs the two overlapping windows around it. Threshold
   calibrated on OVEN: 17 near-miss positives had min similarity 0.545 versus 18
   sibling negatives with max 0.462. Example: `grey herron` vs `Grey Heron`
   scores 0.889.

2. **Character-bigram linker fallback.** `_link_prediction` gained a third stage
   (after exact and word-bigram matching): a char-bigram Jaccard pass at the same
   0.50 threshold. This lets a misspelled or near-miss answer still resolve onto
   the taxonomy so it earns `specific_hF` credit, instead of the linker returning
   `None` (and `specific_hF = 0`) for an answer that is essentially correct.

3. **Path-match reward: content-match → format gate.** The v2 `path_match`
   compared the model's `<traversal>` nodes against the canonical Wikidata P279
   chain, which the model never sees — so it was always 0. It is now a *format
   gate*: `PATH_MATCH_WEIGHT` (0.05) is awarded only when the traversal is
   well-formed (≥2 nodes), on a traversal prompt, and the answer is
   taxonomically plausible (`shF ≥ 0.3`). We reward the traversal *structure*,
   not an unmatchable target chain.

4. **Reward gating by `prompt_type`.** Traversal credit applies only to traversal
   prompts and aggregation credit only to aggregation prompts; all diagnostics
   are still computed for every prompt type. This stops the standard/aggregation
   prompts from being scored on traversal-tag structure they were never asked to
   produce.

5. **Zhang et al.-style prompt redesign** (`build_verl_oven_parquet.py`).
   `RSA_SYSTEM_PROMPT` reframed as a "visual entity recognition expert."
   `_ONESHOT_AGGREGATION` rewritten with explicit per-candidate evaluation
   reasoning and **wired into** `build_rsa_aggregation_prompt` (it had been
   defined but unused — a real bug). `TRAVERSAL_SYSTEM_PROMPT` and
   `_ONESHOT_TRAVERSAL` rewritten into four numbered steps: (1) describe visual
   evidence, (2) recall the taxonomy chain broad→specific inside `<traversal>`
   tags, (3) **evaluate the traversal against sibling/narrower alternatives and
   explain the reasoning**, (4) output the answer in `\boxed{}`. Step 3 is the
   deliberate addition intended to make the model reason about its traversal and
   consider alternatives before committing.

Resulting shaped reward on the standard prompt:
`reward = 0.05·boxed_parse + 0.70·exact_or_fuzzy + 0.15·specific_hF`
(traversal and aggregation terms gate off on the standard prompt), capped at 1.0.

### 10.2 Experiments

Three GRPO runs, Qwen3-VL-4B + LoRA on OVEN, shaped reward, rollout n=16,
seed 42, one-shot prompts, 2 GPUs, 500 steps. The suffix `05`/`08` is the
dedicated-prompt fraction (`--aggregation-fraction` / `--traversal-fraction`);
the remaining rows use the standard prompt.

- **agg05** — 50% aggregation prompts (RSA aggregation-aware RL). Finished, 4
  validation points.
- **agg08** — 80% aggregation prompts. Finished, 4 validation points.
- **trav08** — 80% Zhang-style traversal prompts. Finished, 4 validation points.

Validation uses the **standard** prompt for all three (greedy, n=1), so the
comparison is apples-to-apples; it ran every 150 steps. Metrics were extracted
from the offline wandb `.wandb` datastore (no server/upload) with
`verl/scripts/parse_wandb_datastore.py`. Values at each run's final step:

| Metric | Agg 0.5 | Agg 0.8 | Trav 0.8 |
|---|---|---|---|
| Exact match | 0.133 | 0.138 | 0.136 |
| Exact ∪ fuzzy | 0.171 | 0.175 | 0.176 |
| Specific hF | 0.266 | 0.274 | 0.272 |
| Raw hF | 0.380 | 0.387 | 0.385 |
| Linked to taxonomy | 0.582 | 0.590 | 0.589 |
| Under-specific | 0.046 | 0.047 | 0.045 |
| Over-specific | 0.001 | 0.001 | 0.002 |
| Depth Δ (pred−GT) | −0.079 | −0.077 | −0.068 |
| Traversal emitted | 0.000 | 0.000 | 0.000 |
| Path match | 0.000 | 0.000 | 0.000 |
| Shaped reward | 0.206 | 0.210 | 0.210 |
| Final step | 500 | 500 | 500 |

### 10.3 Did the model adapt its behaviour to surface parametric knowledge? — No

**Quantitatively**, every metric is flat to the third decimal across 500 steps in
all three runs (the trajectories in `viz/grpo/grpo_behavioral.png` are visually
horizontal). The shaped reward gives no gradient. The 80%-aggregation dose beats
the 50% dose on every metric, but the margin (≈0.004 exact∪fuzzy, ≈0.008
specific-hF) is within noise. `traversal_parse` and `path_match` are 0 across all
runs, including trav08 — the model never spontaneously emits `<traversal>`
structure on the standard prompt, so the trained traversal behaviour does not
transfer. Errors are almost entirely under-specific (`under_specific ≈ 0.046` vs
`over_specific ≈ 0.001`, `depth_delta ≈ −0.08`): when wrong, the model stops
short of the leaf rather than overshooting.

**Qualitatively** (`viz/grpo/grpo_generation_examples.md`), all three runs —
including trav08 — produce a flat descriptive paragraph followed by a boxed
answer, never the four-step traversal they were trained on. The persistent
failure mode is under-specific hedging and non-convergent self-correction. The
clearest instance is a trav08 plant example that oscillates
"Chinese Money Plant … Plectranthus or Lysimachia … however, the most precise …
Chinese Money Plant … but in common usage … Money Plant" without ever committing.
The candidate entities are present in the model's
distribution, but it cannot reliably surface the most specific correct one — the
exact access bottleneck (Zhang et al., 2511.05933) that GRPO was meant to close,
and did not.

**Precision / scope of this conclusion.** Because validation is greedy n=1, we
can only conclude that (i) single-sample greedy correctness and the shaped reward
did not improve, and (ii) no traversal behaviour emerged on the standard prompt.
We *cannot* conclude anything about the coverage of the full sampling
distribution — establishing whether the trained model surfaces more latent
knowledge in the RSA/best-of-N sense would require pass@k on the trained
checkpoint, which was not run. Also note `exact_or_fuzzy` (strict string match,
≈0.17) is not comparable to the judge-based base Pass@1 (0.289); the judge is far
more lenient. `specific_hF` conflates "did not link" with "linked but wrong" and
should be read together with `linked` (≈0.59).

### 10.4 Artifacts and reproducibility

- Figures/tables/examples: `viz/grpo/` (`grpo_behavioral.png`,
  `grpo_results_table.tex`/`.md`, `grpo_generation_examples.md`,
  `grpo_val_metrics.csv`, `plot_commands.md`).
- Scripts: `verl/scripts/parse_wandb_datastore.py` (datastore → CSV),
  `oven-mllm-eval/scripts/plot_grpo_training.py` (figure + tables),
  `oven-mllm-eval/scripts/extract_grpo_generation_examples.py` (examples).
- Full reproduction recipe, including how to fold in trav08's final validation
  point when it completes, is in `viz/grpo/plot_commands.md`.

---

## 11. 2026-07-08 — Traversal elicitation, the truncation fix, and behaviour emergence

Two mechanisms were suspected to explain why every traversal run in Sections 9–10
stayed flat with `traversal_parse = 0`: (i) validation always used the **standard**
prompt, so traversal behaviour was invisible at eval time even if present; and
(ii) the traversal format did not fit the response budget. Inspecting the trav08
training rollouts confirmed the second: `response_length/clip_ratio` was
**0.66–0.78** at `--max-response-length 256`, so two-thirds of traversal rollouts
were **truncated before the boxed answer** and scored as parse failures. The
intended reward landscape was never sampled.

### 11.1 Experimental design

Four jobs, Qwen3-VL-4B + LoRA, seed 42:

- **trav10 (training)** — pure traversal prompts (`--traversal-fraction 1.0`),
  trained on the **unlockable** subset (`mine_unlockable_examples.py`: 1540/6087 =
  25.3% of entities are unlockable, i.e. pass@16-hit but pass@1-miss under RSA
  n=16 T=1 sampling; 2082 traversal rows). Fixes vs the earlier runs:
  `--max-response-length 768` (truncation), `lr 1e-5` (was 1e-6), and the
  **ungated** traversal format reward (the `shF >= 0.3` gate was removed so a
  well-formed traversal earns `+0.05` even when the final answer is wrong,
  creating within-group advantage during emergence). Validation on a dedicated
  **traversal-prompt** val set (`--val-prompt-type traversal`, 2048 rows, same 122
  heldout QIDs as `val_unseen`), greedy n=1.
- **elicit-trav / elicit-std** — base model, `val_only`, n=8 sampled
  (temp 1.0), traversal vs standard prompt: the ON/OFF elicitation pair.
- **(b) trav08 checkpoint** — `val_only` under the traversal prompt.

### 11.2 Result 1 — the truncation fix works, and the behaviour appears

At 768 tokens the trav10 training rollouts have `clip_ratio` **0.008–0.035**
(vs 0.66–0.78 before), mean response ~300 tokens, `boxed_parse` 0.99.
`traversal_parse` is **~0.99 at every validation step** — the first time it is
nonzero in any run. The format now fits and is scored.

### 11.3 Result 2 — traversal scaffolding expands access at test time (the positive claim)

On the same 122 heldout QIDs, the traversal prompt beats the standard prompt at
every operating point (`viz/grpo/grpo_traversal_table.md`,
`grpo_traversal_behavioral.png`):

| Metric | Standard (greedy) | Traversal base (pass@1) | Traversal base (pass@8) | Traversal trained (step 250) |
|---|---|---|---|---|
| Exact match | 0.138 | 0.173 | 0.278 | 0.190 |
| Exact ∪ fuzzy | 0.175 | 0.197 | 0.310 | 0.211 |
| Specific hF | 0.274 | 0.316 | 0.475 | 0.338 |
| Linked to taxonomy | 0.590 | 0.645 | 0.887 | 0.682 |
| Pred. path depth | 2.56 | 2.88 | 4.16 | 3.02 |
| Traversal emitted | 0.000 | 0.979 | 1.000 | 0.986 |

The base model, prompted to traverse, produces more accurate, more
taxonomy-linked, and **deeper** (more specific) predictions than under the
standard prompt, and the gap widens with samples (pass@8 exact∪fuzzy 0.310 vs
standard greedy 0.175; specific hF 0.475 vs 0.274). This is the parametric-search
behaviour the project set out to show — Zhang et al.'s structured-prompting
result and Gekhman et al.'s reasoning-boundary expansion, reproduced on
multimodal open-world entity recognition. Note the base model *already* emits
traversals 97.9% of the time when asked: the behaviour is **prompt-elicited, not
learned**.

### 11.4 Result 3 — GRPO on top is flat, and the reward is partly form credit

The run completed all 250 steps and validation accuracy is flat throughout
(exact∪fuzzy 0.213 → 0.209 → 0.217 → 0.216 → 0.211 at steps 50/100/150/200/250;
specific hF 0.336 → 0.338; exact match 0.188 → 0.190; reward 0.299 → 0.297).
`traversal_parse` stays saturated (0.99 → 0.986). Meanwhile the policy moves
*enormously*: train-side entropy collapses **0.59 → 0.03** and KL rises to
**~8.5**, yet the **train reward does not climb** (0.43 → 0.36, if anything down).
This is the textbook GRPO variance-collapse failure: the policy sharpens toward
near-determinism on its prompt-elicited traversal mode, so the n=16 rollouts
become near-identical, within-group advantage vanishes, and the learning signal
dies — with no accuracy gain on either the training or validation set. GRPO
sharpens the already-present behaviour without expanding access.

Following the advice of the supervisor (T. De Min: *"it learns to answer in a
specific format but cannot generalise to unseen knowledge … it might have hacked
the reward … so inspect outputs"*), the generations were inspected
(`viz/grpo/grpo_traversal_examples.md`). Findings:

1. **Not a degenerate hack.** Trained (step 150) and base outputs are
   structurally identical, genuine four-step traversals with real alternative
   comparison, e.g. `animal → mammal → primate → monkey → patas monkey` with
   *"macaques or colobus differ in coloration"*. The model did not collapse into a
   trivial format that games the matcher.
2. **The shaped reward is partly decoupled from correctness.** Trained validation
   shows `score 0.30` but `exact_match 0.19` and `wrong_final 0.78`: a
   well-formed but wrong traversal still earns format + `shF` credit (~0.15), so a
   large part of the reward rise is *form*, not accuracy. The shaped `reward`
   must therefore **not** be read as accuracy — this is precisely the
   reward-composition hazard the supervisor flagged.
3. **Deeper traversals carry a hallucination cost** (Gekhman et al.'s factual-
   priming risk). Wrong answers invent intermediate facts: "Cocal" read off a
   misparsed specimen label; "Palácio da Pena" justified against *fabricated*
   sibling names ("Palácio da Redacción or Palácio de Bonal", which do not exist).
   "More specific" is not "more correct".

### 11.5 Caveats and pending

- **Training complete** (all 250 steps): validation accuracy flat throughout while
  entropy collapsed to 0.03 and KL rose to ~8.5 — confirming the flat reading.
- **No judge re-evaluation.** All numbers here are strict string match
  (`exact_or_fuzzy`), not the LM judge. The supervisor's specific prediction — that
  a judged post-trained model would show higher pass@1 — remains **untested**; it
  needs the judge pipeline run on trav10's checkpoint.
- **The paired standard-prompt n=8 arm failed** (`elicit-std`, 0 records logged),
  as did the **trav08-checkpoint probe** (`trav08ckpt`, 0 records). The standard
  baseline in the table above is therefore the greedy value from the flat July-7
  runs (≈ base policy), not a paired base-model n=8 measurement. Both jobs are
  cheap `val_only` reruns.
- Metric caveats from §10.3 still hold (n=1 greedy for the trav10 trajectory;
  `specific_hF` conflates "did not link" with "linked but wrong").

### 11.6 Artifacts

- `viz/grpo/grpo_traversal_behavioral.png` — pass@k elicitation boundary +
  training trajectory.
- `viz/grpo/grpo_traversal_table.{md,tex}` — standard vs traversal comparison.
- `viz/grpo/grpo_traversal_examples.md` — inspected generations (trained + base).
- `viz/grpo/grpo_traversal_train_metrics.csv`, `grpo_elicit_trav_n8_metrics.csv` —
  tidy source data.
- `oven-mllm-eval/scripts/plot_grpo_traversal.py` — figure + table generator.

---

## 12. Conclusion

The project asked whether GRPO can teach a 4B VLM to access and traverse its
latent visual-semantic knowledge for open-world entity recognition. Across
roughly thirty training runs the empirical arc is now complete and coherent:

1. **The knowledge is in the base distribution.** pass@16 ≈ 0.69 on val_unseen
   (Section 5); here, under a traversal prompt, the base model reaches pass@8
   exact∪fuzzy 0.31 and specific hF 0.48 — well above single-sample accuracy.

2. **Explicit taxonomy-traversal scaffolding surfaces that knowledge at test
   time.** Prompting the base model to describe evidence, recall a broad→specific
   chain, and evaluate alternatives lifts exact∪fuzzy from 0.175 to 0.197 (pass@1)
   and 0.310 (pass@8), specific hF from 0.274 to 0.475 (pass@8), and pushes
   predictions from depth 2.56 to 4.16. This is a positive, reportable result: in
   open-world visual recognition, as in the text-only settings of Zhang et al.
   (2511.05933) and Gekhman et al. (2603.09906), the navigation policy can be
   supplied externally and it improves access to parametric knowledge.

3. **GRPO does not internalise that gain.** Every reward design (exact, shaped,
   fuzzy), prompt design (standard, aggregation, traversal-structured,
   traversal-wikidata), and data curation (random, unlockable) produced flat
   validation accuracy. The final experiment removed the confounds that could have
   masked learning — truncation (256→768 tokens), sparse signal (unlockable-only
   training), weak format pressure (ungated reward), and low learning rate (10×) —
   and still, over the full 250 steps: the traversal format saturates near 0.99,
   validation accuracy sits at the elicitation floor, and the policy collapses to
   near-determinism (entropy 0.59→0.03, KL→8.5) with no gain on even the training
   reward. RL **sharpens the behaviour the prompt already elicits; it does not
   expand access to new entities.**

This is consistent with three convergent readings: the alignment/RLVR view that
verifiable-reward RL mostly re-weights the base distribution rather than expanding
its support (Yue et al.); Allen-Zhu & Li's Physics 3.2 result that knowledge
*manipulation* requires the skill to be present in training, not merely promptable
(here the traversal is prompt-supplied, and outcome-only RL can only reinforce
what the policy already samples); and the supervisor's diagnosis that the model
learns the format without generalising to unseen knowledge. Output inspection
rules out a degenerate reward hack but confirms the softer version: a substantial
part of the shaped reward is form credit, and deeper traversals fabricate
intermediate facts.

The remaining bottleneck is therefore **selection and visual discrimination, not
format**: the model can enumerate the right candidate among many (pass@k) and can
be prompted to traverse, but neither the prompt nor GRPO makes it reliably commit
to the most specific correct entity from a single greedy sample. The
well-motivated next steps — a surface-then-select reward that scores candidate
recall directly, a taxonomy-anchored process reward over the GT ancestor chain,
and SFT warm-starting on self-generated correct traversals before GRPO — follow
from this and are documented for future work. For the thesis, the pairing stands
on its own: **access to parametric knowledge in open-world visual recognition is
promptable but not, at 4B with outcome-only LoRA GRPO, cheaply trainable.**
