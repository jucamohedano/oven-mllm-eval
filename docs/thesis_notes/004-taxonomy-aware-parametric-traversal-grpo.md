# 004 — Taxonomy-Aware Parametric Traversal: GRPO Post-Training for Open-World Visual Entity Recognition

- **Date:** 2026-07-03 (updated)
- **Status:** v1 results flat (no learning). v2 launched with paper-backed fixes (1-shot, filtered agg, LCS path match, max_num_seqs=16, TOTAL_EPOCHS fix). v3 compute-buffer variant launched. v2 results pending.
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
