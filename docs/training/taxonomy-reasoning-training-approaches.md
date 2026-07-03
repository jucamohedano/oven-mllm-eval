# Training a VLM for Taxonomy-Oriented Reasoning on Open-World Entity Classification

> Research notes and recommended approach. Date: 2026-06-24.
>
> **Goal.** Take a Qwen3-VL model (2B/4B/8B) and train it to *reason through the
> taxonomy* — descend from a generic category to the specific entity — when
> answering OVEN-style open-world classification questions ("what is this baseball
> venue?" → "Nationals Park"). This document surveys the available training
> approaches (RL **and** non-RL), grounds each in Nathan Lambert's *RLHF Book*,
> and designs the reward.
>
> Primary source: **Lambert, *Reinforcement Learning from Human Feedback*** (rlhfbook.com, 23 Jun 2026).
> Chapter/page references below point into that book. External multimodal-RL
> papers are cited where the book (which is text-only) does not cover the VLM specifics.

---

## 0. TL;DR — the recommendation

1. **The task is a textbook RLVR problem** (Book Ch. 7): OVEN answers are
   verifiable against the taxonomy, so the reward can be a verification function,
   not a learned reward model.
2. **But don't start with RL.** Follow the DeepSeek-R1 recipe order (Book §3.2.3):
   generate a **cold-start reasoning dataset first** (via rejection sampling or
   distillation), SFT on it, *then* RLVR. The book and multimodal evidence both
   show pure RL from a base checkpoint is brittle.
3. **The reward must resist the generic-collapse trap.** Our hierarchical metric
   `hF` gives **0.75 reward for answering "bird" and 0.50 for a wrong "mammal"**
   on a house-finch example (quantified in §6). Rewarding raw `hF` would *train in*
   the exact failure mode we already observed in the 8B model. Use a binary
   correctness reward (judge- or taxonomy-verified) + format reward, optionally a
   **specificity-weighted** hierarchical bonus.
4. **We already built the hard parts.** The 256-rollout sampler is the dataset
   generator; the LLM judge is the verifier; the `cᵢ` distributions are the
   difficulty signal for filtering. The RL/distillation step sits on top of
   existing infrastructure.

---

## 1. Problem framing

OVEN entity-split example (from `data/processed/vlm_compatible_val_aligned.jsonl`):

```json
{
  "question": "what is this baseball venue?",
  "entity_id": "Q517545",
  "answer": "Nationals Park",
  "image_path": "data/images/oven_04944519.jpg"
}
```

We want the model to emit a **taxonomy-oriented reasoning trace** before answering, e.g.:

```
<think>This is an open-air stadium with a baseball diamond. The architecture and
the cherry-blossom-themed scoreboard suggest a US ballpark in Washington DC.
Home of the Washington Nationals → Nationals Park.</think>
<answer>Nationals Park</answer>
```

**Why OVEN is harder than the classification tasks RL has been proven on.**
Visual-RFT (the canonical "RL for visual perception" paper, [2503.01785](https://hf.co/papers/2503.01785))
does fine-grained classification on **closed** label sets (Flowers-102, Pets-37,
FGVC-Aircraft, Stanford-Cars): the model picks from a known list, so the reward is
a trivial string match. OVEN is **open-vocabulary** — millions of Wikipedia
entities, free-form generation. Consequences:

- A naive exact-string reward is **very sparse** (hard to hit the exact surface form).
- We need a **verifier** that maps free-form output → correct/incorrect: either the
  LLM judge (generative reward model, Book §5.8) or the taxonomy matcher.
- This is precisely where our existing judge + taxonomy-scoring stack becomes the
  reward function.

---

## 2. The spectrum of approaches (from the book)

The book frames a continuum from cheapest/simplest (SFT-only) to most powerful/most
infrastructure (RL). All are viable ways to install reasoning behavior; they differ
in **where the training signal comes from** and **whether the model learns from its
own rollouts**.

| # | Approach | Book ref | Signal source | Learns on-policy? | Cost | Best for us |
|---|----------|----------|---------------|-------------------|------|-------------|
| A | **Rejection Sampling / STaR (SFT)** | Ch. 9, §3.2.3 | Own correct rollouts, filtered by verifier | No (offline) | Low | Cold-start data — **start here** |
| B | **Offline distillation (Seq-KD)** | §12.3.1 | A teacher's offline traces | No | Low | If a strong teacher exists |
| C | **On-policy distillation (OPD)** | §12.3.2 | Teacher scores student's *own* rollouts (dense, per-token) | Yes | Med | Dense signal; mitigates sparse reward |
| D | **On-policy self-distillation (OPSD)** | §12.3.3 | Model + privileged hint (the answer) teaches itself | Yes | Med | Bootstrapping reasoning *without* a teacher |
| E | **RLVR (GRPO/PPO)** | Ch. 6–7 | Verifiable reward (correct/incorrect) | Yes | High | The capability-boosting main course |

The **DeepSeek-R1 recipe (§3.2.3)** combines them: A → E → A → E. We adapt this below (§8).

---

## 3. Approach A — Rejection Sampling / STaR (the cold-start generator)

**What the book says (Ch. 9).** Rejection sampling = generate N completions per
prompt, score them, fine-tune on the top ones (same loss as SFT). The "score" can
be a reward model *or a verifiable check*. The chapter explicitly notes RS "can be
applied after instruction-tuning, after RL, or even after RLVR." STaR (Book §7.3.1)
is the reasoning-specific version: sample chain-of-thought, keep traces whose final
answer is correct, SFT on them.

**How it maps to us — and why it's nearly free.** Our 256-rollout naive sampler
*is* the generation step (Book eq. 110, the `Y` matrix). Our judge + taxonomy scorer
*is* the verifier `R(yᵢ|xᵢ)` (Book eq. 112). So to build a cold-start set:

1. Sample K reasoning rollouts per OVEN example **with a CoT prompt** (the current
   runs use a terse answer-only prompt — we'd switch to a `<think>…</think><answer>…</answer>` prompt).
2. Keep rollouts whose `<answer>` is judged correct (or exact/taxonomy-matched).
3. Deduplicate (Book §9.1.2) and SFT the model on the kept (image, question, trace) triples.

This yields the "~100K filtered on-policy reasoning samples" the R1 recipe calls a
**cold start** (§3.2.3). The book's caveat (§9.2): use 10–30+ completions per prompt,
temperature 0.7–1.0.

**Multimodal precedent.** *ImageNet-Think-250K* ([2510.01582](https://hf.co/papers/2510.01582))
built exactly such a dataset by generating reasoning traces over **ImageNet-21k**
images (one of OVEN's own source datasets) with thinking VLMs (GLM-4.1V-9B-Thinking,
Kimi-VL-A3B-Thinking). We can either reuse that method or distill from a stronger
teacher (Approach B).

**Limitation.** STaR can only capture reasoning the model *already* produces
sometimes. For entities the base model never gets right, there are no correct
rollouts to learn from → no cold-start coverage there. Approaches B/D fix this.

---

## 4. Approach B — Offline distillation (Sequence-KD)

**What the book says (§12.3.1).** Classic teacher→student knowledge distillation
adapted to LMs: a stronger **teacher** generates output sequences, the student is
trained with cross-entropy to match them (Book eq. 126–127). Minimizing
cross-entropy = minimizing forward KL from teacher to student (eq. 129). This is the
"start with tailored instruction datasets with CoT sequences heavily filtered and
polished from existing models" step the book mentions as "a fast step to strong
behaviors with SFT alone before moving onto RL" (§3.2.3).

**How it maps to us.** Use a strong VLM (e.g. Qwen3-VL-32B, or a dedicated thinking
VLM) as the teacher: prompt it to produce taxonomy-descent reasoning + answer on
OVEN images, **filter to correct answers** (verifier again), and SFT the small
student on those traces. Distillation is more reliable than humans for generating
these traces (§12.1) and gives coverage on entities the student can't yet solve.

**Limitation: exposure bias (§12.3.2).** Offline traces are off the student's own
distribution, so the forward-KL objective makes the student "overestimate
low-probability regions of the teacher" and accumulate error along long traces
(the book's `O(εL²)` bound). For long taxonomy-reasoning chains this matters →
motivates the on-policy variants.

---

## 5. Approach C — On-policy distillation (OPD)

**What the book says (§12.3.2–12.3.3).** Instead of matching offline teacher text,
the **student generates the rollout on-policy** and the teacher scores each visited
token with a reverse-KL objective (Book eq. 134):

$$\mathcal{L}_{\text{OPD}}(\theta) = \mathbb{E}_{s,\,a\sim\pi_\theta(\cdot\mid s)}\sum_t D_{\text{KL}}\!\big(\pi_\theta(\cdot\mid s_t)\,\|\,\pi_T(\cdot\mid s_t)\big)$$

Because the student confronts its own mistakes and the teacher corrects them at the
visited states, this reduces the compounding error from `O(εL²)` to `O(εL)` (the
DAgger argument, §12.3.2).

**The key idea for us (eq. 135).** Modern OPD folds the KD distance **into RL as a
dense per-token reward/advantage**:

$$A_t^{\text{OPD}} = \log \pi_T(a_t\mid s_t) - \log \pi_\theta(a_t\mid s_t)$$

The book's punchline: this "acts like dense token-level feedback, providing
potentially **more useful learning feedback than the sparse verifiable rewards** or
reward-model outputs." That is directly relevant to our sparse-reward problem
(§6): a teacher's per-token signal gives gradient *everywhere*, not just on the
rare exactly-correct rollout. **Multi-Teacher OPD (MOPD, eq. 136)** can even mix
specialist teachers (e.g. a "geography entity" expert and an "organism" expert)
with prompt-dependent weights.

**Requirement.** Teacher and student must share a tokenizer (Book §12.3.3) — fine
within the Qwen3-VL family (use 32B as teacher for the 2B/4B/8B student).

---

## 6. Approach D — On-policy self-distillation (OPSD) — *strong fit, novel for us*

**What the book says (§12.3.3).** One model plays both roles. **Privileged
information** (a hint) is injected into the context to create a high-quality
"teacher" trajectory; the *no-hint* generation is then distilled toward it with a KL
loss — **no separate teacher model**. The book's example is Cursor's Composer model:
during RL, when the model hits a bug, a judge inserts a hint token, and the model
distills its no-hint generation toward the hinted one.

**Why this fits OVEN unusually well.** Our "privileged information" is sitting right
in the dataset: **the ground-truth entity and its taxonomy path.** Procedure:

1. Show the model the image + question **+ the answer as a hint** ("This is
   *Nationals Park*; explain how to identify it from the image"). With the answer in
   context, even the 2B can produce a coherent taxonomy-descent trace.
2. Strip the hint; have the model generate normally.
3. Distill the no-hint generation toward the hinted trajectory (reverse-KL).

This **bootstraps taxonomy reasoning without needing the model to already do it and
without a stronger teacher**, sidestepping STaR's coverage gap (§3) and B/C's
teacher requirement. It is the cleanest answer to "how do I get cold-start traces
for entities the model can't yet solve."

---

## 7. Approach E — RLVR (the main RL method)

**What the book says (Ch. 6–7).** Replace the reward model with a **verification
function**: `r = γ if correct else 0` (Book Fig. 25). The training loop (Ch. 7) is:

> 1. Sample multiple answers to multiple questions.
> 2. Take gradient steps toward the correct ones.
> 3. Repeat, revisiting the same data.

Policy-gradient algorithm: **GRPO** (Book §6.2.8) is the standard for reasoning —
no value network, advantages computed by group-normalizing rewards across the K
rollouts of a prompt. This is exactly what every multimodal reasoning-RL paper below
uses.

**Multimodal precedent (the book is text-only here).**

- **Visual-RFT** ([2503.01785](https://hf.co/papers/2503.01785)) — GRPO on Qwen2-VL
  with verifiable rewards; **+24.3% one-shot fine-grained classification** from
  ~100 samples. Reward = `R_acc + R_format` (see §6). The proof that RLVR transfers
  to visual classification.
- **GLM-4.1V-Thinking** ([2507.01006](https://hf.co/papers/2507.01006)) —
  "RL with Curriculum Sampling," SOTA multimodal reasoning incl. content recognition.
- **"SFT or RL?"** ([2504.11468](https://hf.co/papers/2504.11468)) — warns that
  cold-start **SFT can hinder later RL** by inducing *imitative* ("pseudo") reasoning;
  argues for keeping SFT light. A caution on Stage-1 dose (see §8).
- **Text-only reasoning boosts multimodal** (Book §7.3.3; Magistral, MiMo-VL):
  doing text-only reasoning RL *after* multimodal training *improves* multimodal
  performance — a cheap extra lever.

**The §7.3.3 "common practices" we should adopt:**

- **Offline difficulty filtering** *(most important)* — RL only learns where there
  is a gradient; drop prompts the model solves ~0% or ~100% of the time, keep the
  20–80% band. **We already have this signal** in the `cᵢ` distributions (correct
  rollouts out of 256). Build the RL prompt set directly from them.
- **Format rewards** — small reward for well-formed `<think>…</think><answer>…</answer>`.
- **Remove the KL penalty** for longer RL runs (less over-optimization pressure once
  reasoning is established); **relaxed clipping** (DAPO) for exploration.
- **Length penalties** to curb overthinking.

---

## 8. Reward design (the core question)

OVEN's open vocabulary makes the reward the crux. Three decisions: **verifier**,
**shape**, and **format**.

### 8.1 The verifier: how "correct" is decided

| Option | What it is | Pros | Cons |
|--------|-----------|------|------|
| **Exact / normalized string match** | `answer == entity_text` after normalization | Free, deterministic | Very sparse (surface-form brittle) — bad for open vocab |
| **LLM judge** (generative RM, Book §5.8) | Our Qwen3 judge decides correct/incorrect | Robust to paraphrase; reuses our stack | A judge forward-pass per rollout (×256 cost); adds noise; hackable if judge is weak |
| **Taxonomy match** | Map answer → taxonomy node, compare to reference path | Deterministic, cheap, gives hierarchy | Depends on mapping quality |

**Recommendation:** use the **taxonomy match for the dense shaped term** (it's free
and already implemented) and the **judge for the binary correctness term** during
data generation / periodic validation. Keep a temperature-0 judge to cut variance
(Book §5.8).

### 8.2 The shape: binary vs hierarchical — and the reward-hacking trap

The book's RLVR is **binary** (1/0). Visual-RFT's classification reward is **binary**
(`R_acc ∈ {0,1}`) + format. We have a richer signal (`hF`), but it is **dangerous as
a raw reward.**

**Quantified on our actual metric** (`calc_hierarchical_metrics`, leaf-first paths;
reference = `house finch → finch → bird → animal → root`):

| Model answer | hP | hR | **hF (raw)** | specificity | **hF × specificity** |
|--------------|----|----|--------------|-------------|----------------------|
| `house finch` (exact) | 1.00 | 1.00 | **1.00** | 1.00 | **1.00** |
| `purple finch` (sibling species) | 0.80 | 0.80 | **0.80** | 1.00 | **0.80** |
| `finch` (genus) | 1.00 | 0.80 | **0.89** | 0.80 | **0.71** |
| `bird` (generic) | 1.00 | 0.60 | **0.75** | 0.60 | **0.45** |
| `animal` (very generic) | 1.00 | 0.40 | **0.57** | 0.40 | **0.23** |
| `mammal` (wrong, high-level) | 0.67 | 0.40 | **0.50** | 0.60 | **0.30** |

**Read the raw-hF column.** Answering **"bird" scores 0.75** and even a **wrong
"mammal" scores 0.50.** A model optimizing raw `hF` learns that retreating to a
generic category is a safe, high-reward move versus risking a specific guess that
might score 0. **This is exactly the confident-generic-collapse we already documented
in the 8B model ("moth", "tree", "altimeter").** Rewarding `hF` would *train the
pathology in.* This is the book's over-optimization warning (Ch. 14) made concrete.

**The fix — specificity weighting.** Multiply `hF` by the relative depth of the
prediction, `spec = depth(pred)/depth(ref)`. The last column shows the ordering is
restored: exact (1.00) > sibling species (0.80) > genus (0.71) > "bird" (0.45) >
"mammal" (0.30) > "animal" (0.23). Now specific, near-correct guesses dominate and
generic retreat is penalized — while still giving **partial-credit gradient** on
fine-grained near-misses (the legitimately hard cases), which is what makes a dense
reward worth having.

### 8.3 Recommended composite reward

$$R = \underbrace{R_{\text{correct}}}_{\text{binary, judge/taxonomy}} \;+\; \underbrace{\lambda_f \, R_{\text{format}}}_{\text{well-formed think/answer}} \;+\; \underbrace{\lambda_s \, (hF \times \text{spec})}_{\text{optional dense bonus}}$$

- **`R_correct ∈ {0,1}`** — exact taxonomy/leaf match or judge-verified. The primary
  signal (matches Book + Visual-RFT). Keep `λ_s` small so the bonus never outweighs
  actually being right.
- **`R_format`** — Visual-RFT's `R_format`: 1 if the output matches
  `<think>…</think><answer>…</answer>`, else 0. Installs the reasoning structure.
- **`λ_s·(hF×spec)`** — *optional* dense term to provide gradient on hard entities
  where every rollout misses the exact answer (the sparse-reward problem). Start with
  `λ_s = 0` (pure binary, safest) and only turn it on if difficulty-filtered binary
  RL stalls for lack of gradient. **Never use raw `hF`.**

**Alternative to the dense term:** instead of shaping the outcome reward, get dense
signal from **OPD (§5, eq. 135)** — a teacher's per-token logprob gap. This is the
book's preferred way to densify sparse verifiable rewards and avoids outcome-reward
hacking entirely.

### 8.4 What about Process Reward Models?

Taxonomy reasoning is hierarchical, so a **PRM** (Book §5.6) that scores each step
(genus correct? family correct?) is conceptually appealing. **Skip it for now:** the
book notes PRMs need per-step annotations, are "less supported in open-source RLHF
tools," and blur into ORMs. The specificity-weighted hF already captures "how far
down the correct branch did you get" without step labels.

---

## 9. Recommended recipe (DeepSeek-R1 adapted, §3.2.3)

A staged plan that reuses our infrastructure and respects the cautions above:

1. **Cold-start data (Approach A or D).**
   - Switch the sampler to a `<think>/<answer>` CoT prompt; generate K≈16–32 rollouts
     per example on the **difficulty-filtered** prompt set (§7, from `cᵢ`).
   - Keep judge-correct traces (STaR). For entities with **no** correct rollout, use
     **OPSD (§6)**: feed the answer as a hint, generate a trace, keep it.
   - Deduplicate → cold-start SFT set. *Keep this SFT light* — "SFT or RL?"
     ([2504.11468](https://hf.co/papers/2504.11468)) warns heavy SFT induces imitative
     reasoning that hurts later RL.
2. **SFT** the student (2B/4B/8B) on the cold-start set. Format adherence is the goal,
   not memorization.
3. **RLVR with GRPO (Approach E).**
   - Prompt set = difficulty-filtered (20–80% solve rate from `cᵢ`).
   - Reward = §8.3 composite (start binary + format; add dense term or OPD only if needed).
   - Adopt §7.3.3 practices: format reward, drop KL after warmup, length penalty,
     monitor for generic collapse (track answer specificity, not just `hF`).
4. *(Optional)* **Rejection-sample from the RL checkpoint → second SFT** (R1 stage 3)
   to consolidate, then a short **final RL** polish.
5. *(Optional, cheap win)* **Text-only reasoning RL** after the multimodal stage —
   the Magistral/MiMo-VL effect (Book §7.3.3).

---

## 10. Using our existing `cᵢ` data for difficulty filtering

The single most-emphasized RL practice (§7.3.3) is difficulty filtering, and we are
already sitting on the exact signal. For each example we have `cᵢ` = #correct out of
256 rollouts. Map to a curriculum:

- `cᵢ = 0` (never solved): **exclude from RL** — no gradient. Candidates for OPSD
  cold-start instead.
- `1 ≤ cᵢ ≲ 205` (solved 0.5–80%): **the RL training set** — maximal gradient.
- `cᵢ ≳ 205` (solved >80%): **exclude or down-weight** — little to learn.

This also reframes our headline finding: the 2B's high pass@256 / high answer
diversity means it has **more examples in the productive `cᵢ` band** than the
stubborn 8B — i.e. the 2B may be the *better RL starting point* because it explores
(more non-zero-gradient prompts). Worth testing directly.

---

## 11. Risks & mitigations

| Risk | Source | Mitigation |
|------|--------|-----------|
| **Generic-answer collapse** | Raw `hF` rewards vagueness (§8.2); already seen in 8B | Binary or specificity-weighted reward; track answer **specificity/depth** as a guardrail metric |
| **Reward hacking generally** | Book Ch. 14 (over-optimization) | KL anchor during warmup; small `λ_s`; difficulty filtering keeps binary signal honest |
| **Judge gaming** | Weak judge as verifier (§8.1) | Temp-0 judge; periodic human/exact-match spot-checks; don't let the policy see the judge |
| **Imitative pseudo-reasoning** | Heavy cold-start SFT ([2504.11468]) | Keep Stage-2 SFT light; let RL do the capability work |
| **Sparse reward / no gradient** | Open-vocab exactness | Difficulty filtering; dense OPD signal (§5) or specificity-weighted bonus |
| **Exposure bias on long chains** | Offline distillation (§12.3.2) | Prefer on-policy (OPD/OPSD/RLVR) over offline Seq-KD |

---

## 12. Open questions / experiments to run

1. **Binary vs specificity-weighted reward** — does the dense term speed convergence,
   or does it (even weighted) drift toward generic answers? Track mean answer depth.
2. **Best RL starting model** — 2B (more exploration, more productive `cᵢ` band) vs
   8B (higher pass@1)? §10 predicts 2B; test it.
3. **OPSD vs STaR vs teacher distillation** for cold-start coverage on `cᵢ=0` entities.
4. **Does taxonomy reasoning generalize** to unseen entities (the OVEN
   `entity_val_unseen` split), or only memorize seen ones? This is the real question
   the thesis cares about.
5. **Verifier choice** — judge-as-reward vs taxonomy-match-as-reward: agreement rate,
   cost, and which yields better-calibrated RL.
6. **Text-only reasoning RL afterburner** — does the Magistral/MiMo-VL effect hold for
   OVEN entity recognition?

---

## 13. References

**Book (primary):** Lambert, *Reinforcement Learning from Human Feedback*, rlhfbook.com.
Ch. 3.2.3 (R1 recipe), Ch. 5.5–5.8 (reward model types), Ch. 6 (policy gradients / GRPO),
Ch. 7 (RLVR + common practices §7.3.3), Ch. 9 (rejection sampling), Ch. 12.3 (distillation:
Seq-KD / OPD / OPSD / MOPD), Ch. 12.6 (rubrics), Ch. 14 (over-optimization).

**Multimodal RL (external):**
- Visual-RFT — [hf.co/papers/2503.01785](https://hf.co/papers/2503.01785) (GRPO + verifiable rewards; fine-grained classification; `R_acc + R_format`)
- GLM-4.1V-Thinking — [hf.co/papers/2507.01006](https://hf.co/papers/2507.01006) (RL with Curriculum Sampling)
- SFT or RL? (VLAA-Thinking) — [hf.co/papers/2504.11468](https://hf.co/papers/2504.11468) (SFT can hinder RL)
- ImageNet-Think-250K — [hf.co/papers/2510.01582](https://hf.co/papers/2510.01582) (reasoning traces over ImageNet-21k)
- Rewards as Labels: RLVR from a Classification Perspective — [hf.co/papers/2602.05630](https://hf.co/papers/2602.05630)

**Our infrastructure referenced:** `src/oven_mllm_eval/measures.py` and `scores.py`
(hP/hR/hF), the 256-rollout sampler, the Qwen3 LLM judge, the `cᵢ` distributions,
`data/processed/oven_taxonomy_index.json` (leaf-first paths), `scripts/build_aligned_questions.py`.

---

## 14. GRPO experiment log

*Quick-lookup record of what was tried, what worked, and what failed. Updated as experiments complete.*

### v1 (July 2, 2026) — Flat reward, no learning

**Setup:** Qwen3-VL-4B, GRPO, 2 GPUs, ROLLOUT_N=4, TRAIN_BATCH_SIZE=16. Standard/aggregation prompts (no 1-shot), exact-only reward, max_prompt_length=4096.

**Result:** All three runs (exact, traversal-exact, traversal-shaped) finished mechanically stable but **mean reward was flat** (0.05–0.22), no upward trend. KL exploded in GRPO-exact (0.08 by step 90). Max reward hit 0.9 occasionally but GRPO could not increase its probability.

**Root causes identified:**
1. No examples of correct traversal in prompts — model didn't know what good reasoning looked like (Physics of LLMs 3.2: CoT examples needed in training data)
2. Aggregation candidates uncontrolled — all-wrong sets = impossible task
3. ROLLOUT_N=4 → 65% chance of ≥1 correct per group, weak advantage signal
4. max_num_seqs=64 → vLLM KV cache ~18GB, OOM hang after 30-125 steps
5. TOTAL_EPOCHS not exported → training stopped at 1 epoch (125 steps) regardless of setting
6. Only one reward file — all experiments used shaped reward, exact-only experiments were contaminated

**Lessons for v2:** 1-shot examples, filter aggregation, increase ROLLOUT_N, reduce max_num_seqs, export TOTAL_EPOCHS, split reward files.

### v2 (July 3, 2026) — Pending

**Fixes:** 1-shot examples (crossover SUV), filtered aggregation (1-2 correct), LCS path match, max_num_seqs=16, TOTAL_EPOCHS export, ROLLOUT_N=8, separate `oven_boxed_exact.py` / `oven_boxed.py`, max_prompt_length=5120, val_batch_size=8192.

**Experiments:** GRPO-exact v2, GRPO-unlockable v2, GRPO-traversal v2 (exact), GRPO-traversal v2 (shaped).

### v3 (July 3, 2026) — Pending

**Fixes:** Compute-buffer standard prompt variant ("Think step by step, then `\boxed{answer}`") replacing "Reason carefully…". Based on Thinking to Recall's finding that dummy filler tokens improve performance — model needs compute buffer, not structured reasoning.

**Experiment:** GRPO-exact v3 (cb). Compare vs v2 #1.

### Bugs encountered and fixes

| Bug | Symptom | Fix | Source |
|---|---|---|---|
| `_pad_token_ids` list/dim | `'list' object has no attribute 'dim'` at step 124 | Guard empty tokens before `tokenizer.pad()` | PR #6675 |
| `compute_timing_metrics` ZeroDivisionError | Crash when zero tokens in batch | Guard `num_tokens > 0` before division | PR #6675 |
| vLLM KV cache OOM | GPU 0% util, hang after 30-125 steps | max_num_seqs 64→16 (18GB→4.6GB) | Config |
| TOTAL_EPOCHS not exported | Training stopped at 125 steps regardless of setting | Export TOTAL_EPOCHS in schedule script heredoc | Our bug |
| Prompt > 4096 tokens | Traversal prompt overflow | max_prompt_length→5120 | Config |
| Validation OOM | 56k val rows flooded vLLM | val_batch_size=8192 | Config |
