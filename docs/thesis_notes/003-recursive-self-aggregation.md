# 003 — Recursive Self-Aggregation (RSA): method, our OVEN adaptation, and the planned aggregation-aware RL for taxonomy-aware image classification

- **Date:** 2026-06-25
- **Status:** Foundation note with methodological correction. The current OVEN script (`scripts/run_recursive_self_agg.py`) supports two modes: **answer aggregation** over existing short entity-label rollouts and **solution aggregation** over freshly generated concise solution traces ending in `\boxed{...}`. The answer-mode runs are the current baseline for comparing RSA against naive sampling. The solution mode is the more faithful RSA adaptation and also feeds the aggregation-aware RL plan.
- **Primary source:** Venkatraman, Jain, Mittal, Shah, Obando-Ceron, Bengio, Bartoldson, Kailkhura, Lajoie, Berseth, Malkin, Jain — *Recursive Self-Aggregation Unlocks Deep Thinking in Large Language Models*, arXiv:2509.26626v2 (24 Feb 2026). Code: `rsa-llm/RSA`, `rsa-llm/RSA-ARC`. Local PDF: `../resources/Venkatraman et al. - 2026 - Recursive Self-Aggregation….pdf`; reference impl `../RSA/eval_loop.py`.
- **Our artifacts:** `scripts/run_recursive_self_agg.py` (answer-mode and solution-mode RSA over OVEN samples), `scripts/schedule_rsa.sh` (SLURM launcher), `docs/operations/rsa-runbook.md` (operational runbook), `docs/training/rl-taxonomy-reasoning-recipe-plan.md` (RL recipe planning).
- **Related notes:** [[001-model-scale-coverage-vs-reliability]] (judge inflation + under-specificity — directly relevant to RSA's reward/aggregation), [[002-taxonomy-mapping-and-rollout-metrics]] (the hF metric an RSA-RL reward would optimize).

---

## 1. Thesis positioning

The thesis takes RSA — a test-time scaling method that recombines a *population* of candidate solutions through repeated self-aggregation — and adapts it to **open-domain visual entity recognition (OVEN)**. A key methodological distinction emerged from reading the paper appendix and repository:

- In the RSA paper, a "candidate solution" is a **reasoning trace plus final answer**. Aggregation can reuse partial reasoning steps, discard incorrect steps, and add new structure.
- In OVEN **answer mode**, a "candidate" is only a **short entity/class label** from `all_texts`. The script can aggregate/vote/refine labels, but it cannot aggregate candidate reasoning because the evidence is not present in the candidate population.
- In OVEN **solution mode**, the script first generates concise candidate solutions ending in `\boxed{...}` and recursively aggregates those solutions. This is closer to the paper's mechanism because the population contains reasoning text, not only labels.

This distinction matters for thesis claims. Answer-mode runs should be described
as an **answer-aggregation baseline**. Solution-mode runs are the faithful RSA
adaptation to evaluate next, because the candidate population includes reasoning
and a parseable final answer.

Three strands:

1. **Post-hoc answer aggregation on OVEN** (built): does recursively aggregating diverse candidate labels with the image in context improve taxonomy-aware accuracy over plain naive-sampling?
2. **Solution-trace RSA for OVEN** (built as an evaluation mode): generate candidate visual-recognition solutions with reasoning + final answer, then recursively aggregate those solutions.
3. **Aggregation-aware RL for taxonomy-aware image classification** (planned): adapt the paper's §4 RL recipe so the VLM is trained to aggregate candidate visual-recognition solutions into the correct, appropriately-specific entity — optimizing a taxonomy-aware reward.

The motivating link to our own findings: [[001-model-scale-coverage-vs-reliability]] shows smaller VLMs hedge toward **under-specific (hypernym) answers**. RSA aggregation is a natural lever on exactly this failure mode — aggregation can either *sharpen* toward the specific leaf (good) or *converge on the safe parent class* (bad). Whether it helps is an empirical question this work answers.

---

## 2. RSA — the test-time method (paper §3)

RSA is a **hybrid** test-time scaling method (combines parallel breadth + sequential depth) framed as an evolutionary process: a population of candidates is iteratively recombined via aggregation. It uses a **single LLM**, **no external verifier**, and relies on the model's **implicit verification** (the generation–verification gap: models judge correctness better than they produce it).

**Notation / algorithm** (one reference model `p_θref`, query `x`):
- **Population** `P_t = {τ_1^{(t)}, …, τ_N^{(t)}}` of `N` candidates.
- **Init** (Eq 1): `P_1` = `N` independent samples `τ_i^{(1)} ~ p_θref(·|x)`.
- **Subsample** (Eq 2): form `N` aggregation sets `S_i^{(t)} ⊆ P_t`, each of size `|S_i|=K`, sampled **uniformly without replacement**.
- **Aggregate** (Eq 3): `τ_i^{(t+1)} ~ p_θref(·| S_i^{(t)}, x)` — the model is given the query + `K` candidates and an aggregation prompt, and emits one improved candidate → `P_{t+1}`.
- Repeat for `t = 1 … T−1` (so `T` total population states = `T−1` aggregation updates).
- **Terminate**: uniform-random sample from `P_T` (no special selection; majority vote also possible).

**Three knobs and what they control (paper §5.4):**
- **`K` (aggregation set size):** `K=1` ⟺ pure sequential self-refinement; `K=2` gives the **largest single jump** (aggregating diverse chains beats refining one); **diminishing returns beyond `K=3`** (limited long-context attention).
- **`T` (sequential depth):** performance improves **monotonically** with `T` on nearly all tasks.
- **`N` (population size):** controls **asymptotic** performance — `pass@N` is the upper bound `pass@1` converges to. Larger `N` → higher ceiling but needs more `T` (or larger `K`) to "mix". The **`pass@N − pass@1` gap** is a useful predictor of a candidate set's *aggregability*.
- Tuning under budget: jointly increasing `N,K,T` helps; with limited `T`, reduce `N`; a large population that fails to mix is worse than a small one that evolves fast.

**Headline results:** RSA + Gemini 3 Flash → near top of ARC-AGI-2 public leaderboard (≈10% of Gemini 3 Deep Think's cost). RSA lets **Qwen3-4B-Instruct-2507** match DeepSeek-R1 / o3-mini(high) across AIME-25, HMMT-25, LiveCodeBench-v6, Reasoning Gym, SuperGPQA. Consistent gains across model families/sizes (incl. MoE, hybrid SSM, "thinking" models).

**Important qualitative mechanism (paper Appendix F):** RSA is not just answer voting. The appendix example gives Qwen3-4B-Instruct-2507 four candidate solutions for a divisor problem. The aggregated solution reuses useful intermediate observations from different candidates — e.g. candidates identify that multiples of 5 cannot end in 1; one candidate additionally observes that even divisors cannot end in 1; the aggregate combines those constraints and reorganizes the search into a table. This is the core mechanism our current OVEN answer-only adaptation cannot exploit.

---

## 3. Aggregation-aware RL — the recipe we adapt (paper §4, §5.5)

**The problem the paper identifies:** standard RL post-training optimizes the model to *directly* produce correct solutions; it does **not** teach the model to **aggregate** multiple candidates. This train/test mismatch means **standard RL can *degrade* RSA performance vs. the base reference model** (distribution shift). Confirmed empirically: standard-RL + RSA underperforms reference + RSA in 4/5 tasks (Fig 9).

**The fix — an "aggregation-aware" training dataset with two prompt types:**
1. **Standard prompts** (problem only) → train the model to propose good *initial* candidates (`P_1`). Objective (Eq 4): `max_θ E_{(x,y)~D} [ E_{τ~π_θ(·|x)} r(τ,y) − β·KL(π_θ(·|x) ‖ π_θref(·|x)) ]`.
2. **Aggregation prompts** (problem + `K` candidate solutions sampled from the **reference** model `p_θref`, formatted with the **same** RSA aggregation prompt) → train the model to aggregate. Objective (Eq 5): additionally sample `S_0 ~ p_θref(·|x)^K`; `max_θ E_{(x,y)~D, S_0} [ E_{τ~π_θ(·|x,S_0)} r(τ,y) − β·KL(π_θ(·|x,S_0) ‖ π_θref(·|x,S_0)) ]`.

Jointly optimize (1)+(2). Any policy-gradient method works (PPO/GRPO/RLOO); the paper uses **RLOO**, initializing `θ` from `θref`.

**Training setup (their §5.5):** reference = Qwen3-4B-Instruct-2507; data = 16,000 DeepScaleR math problems + 2,048 each from six Reasoning-Gym tasks the reference is weak on (`tower_of_hanoi, sokoban, knight_swap, rush_hour, arc_1d, sentence_reordering`); **4 candidate solutions** per query from the reference build the aggregation prompts (`K=4`); both the aggregation-aware model and a standard-RL baseline trained **300 steps with RLOO**; evaluated with RSA `T=10, K=4, N=16`.

**Results (Fig 9):** aggregation-aware RL + RSA **always beats** standard-RL + RSA and **significantly beats the reference** in 4/5 tasks (AIME-25 the outlier). Notably **large gains on LiveCodeBench despite no code in the training data → aggregation skill transfers out-of-domain.** Takeaway: *standard RL hurts RSA; aggregation-aware RL helps and is simple to add* — they "strongly encourage its adoption for post-training."

**Paper's own future work (relevant to us):** (i) compose RSA with explicit self-verification as a fitness function; (ii) **multi-step RL to train the policy for the end-to-end RSA procedure**, beyond the greedy single-step aggregation objective above.

**Repository check:** the RSA repo does **not** reward "thinking" directly. `RSA/rewards/math.py` rewards only final answer correctness via the last `\boxed{...}` answer; `RSA/rewards/code.py` rewards whether extracted code passes tests. There is no reward for `<think>` tags, chain length, or reasoning format. The "deep thinking" behavior is induced indirectly: prompts request reasoning/aggregation, candidate traces contain reasoning, and RLOO rewards only final correctness.

**Model-family check:** the paper's RL setup uses `Qwen/Qwen3-4B-Instruct-2507`, not a dedicated Thinking checkpoint. This supports our thesis direction: post-train **Qwen3-VL-Instruct** models and evaluate against our existing Instruct baselines. For Qwen3-VL, Hugging Face model cards expose separate Instruct and Thinking repositories; Thinking behavior is controlled by the checkpoint/tokenizer template, not by manually inserting `<think>` tags. Our OVEN prompts should not use `<think>` tags.

---

## 4. Our current OVEN adaptation: recursive aggregation over answers or solutions

`scripts/run_recursive_self_agg.py` has two candidate formats.

### Answer mode

Answer mode runs RSA **post-hoc** over an existing naive-sampling `*_samples.jsonl` — it does **not** create `P_1` itself. Design choices:

- **`P_1` from existing rollouts:** each source row's `all_texts` (the 256 naive-sampling rollouts) seeds the population (`--initial-selection first|random`, padded/truncated to `N`). So RSA reuses inference already paid for — no fresh `P_1` sampling.
- **Candidates are short labels, not reasoning chains** → a bespoke aggregation prompt (`build_oven_rsa_prompt`): image + question + `K` candidate answers → "aggregate the useful clues, choose the answer best supported by the image and question, produce one improved answer; if all candidates seem wrong, answer with a better label; return only the final answer." Singular-candidate (`K=1`) and multi-candidate variants. The question is formatted with the run's own `prompt_variant` (`--prompt-variant source` reuses each row's variant).
- **Image is in context for every aggregation** (unlike text-only RSA) — `--no-image` gives a text-only ablation.
- **Output is a normal samples JSONL**: final `all_texts` = `P_T`, `prediction` = `P_T[0]`, `method = recursive-self-aggregation`, plus an `rsa` provenance block (`population, k, steps, updates, initial_selection, initial_population, seed`). `schedule_rsa.sh` generates RSA samples only; judging and taxonomy scoring are separate downstream steps. The existing metrics still apply: pass@k comes from judge verdicts, while taxonomy hF uses the restored direct/cascade mapping over deduped `all_texts` ([[001]]/[[002]]).
- **Knobs:** `--population N` (16), `--k K` (4), `--steps T` (2 ⇒ one aggregation update; `T` *includes* `P_1`, so updates `= T−1`), sampling `--temperature 1.0 --top-p 1.0 --top-k -1 --max-tokens 16`, `--chunk-size` (prompts per update `= chunk_size × N`), durable `--resume`/`--overwrite`, optional `--restart-every` engine reinit.
- Downstream fields are stripped from source rows (`_strip_downstream_fields`) so a judged/scored input is re-usable as a clean samples file; per-chunk RNG `seed+start` for reproducible subset sampling.

**Methodological limitation:** answer mode cannot show the Appendix-F RSA mechanism because candidate labels contain no visual evidence or reasoning. The aggregator can use the image directly and can refine labels, but it cannot compare candidate visual cues, inherit a useful observation from one candidate, reject an unsupported observation from another, or make that process auditable. Therefore, answer-mode results should be reported as **answer aggregation / label refinement**, not full recursive self-aggregation over reasoning chains.

### Solution mode

Solution mode first generates an initial population of concise solution traces,
each ending with a parseable `\boxed{...}` answer, and then recursively
aggregates those traces. This makes the candidate population closer to the RSA
paper's "solution" object: the aggregator can reuse useful reasoning, discard
unsupported steps, and produce a new final answer.

The output is still a normal samples JSONL compatible with the existing judge
and scoring pipeline. The final answer extracted from `\boxed{...}` becomes the
row-level prediction/all_text candidate used downstream. Operational commands
belong in `../operations/rsa-runbook.md` and `../commands.md`.

---

## 5. Planned contribution: trace-based RSA + aggregation-aware RL for taxonomy-aware image classification

Adapt §3–§4's recipe to OVEN, training a VLM to aggregate candidate **visual-recognition solutions** into the correct, appropriately-specific entity. The unit of aggregation should become:

```text
Visual evidence:
- ...

Reasoning:
...

Final answer: <entity name>
```

No `<think>` tags. No taxonomy chain/descriptions in the prompt. The final answer remains parseable for existing judging/scoring, while the trace gives us evidence to study whether aggregation actually reuses visual cues.

**Trace-based test-time RSA pipeline:**
1. **Initial candidate-solution generation:** image + question → `N` candidate solutions with visual evidence, brief reasoning, and `Final answer: ...`.
2. **Aggregation:** image + question + `K` candidate solutions → one improved solution in the same format.
3. **Recursion:** repeat aggregation for `T−1` updates.
4. **Evaluation:** extract only `Final answer:` into `prediction`/`all_texts`; store full traces separately (`rsa_initial_solutions`, `rsa_final_solutions`, parse flags) for qualitative analysis.

**Proposed initial-solution prompt:**

```text
You are answering an open-world visual recognition question.

Use the image and question to identify the most specific entity name you can justify.
Do not use outside knowledge unless it is visually supported by the image.
If uncertain, state the strongest visual evidence and give your best specific guess.

Write your response in exactly this format:

Visual evidence:
- <visible cue 1>
- <visible cue 2>
- <visible cue 3>

Reasoning:
<brief explanation connecting the visual evidence to the answer>

Final answer: <entity name>

Question:
{question}
```

**Proposed aggregation prompt:**

```text
You are given an image, a question, and several candidate solutions.
Each candidate may contain useful visual evidence, wrong assumptions, or an incorrect final answer.

Your task is to aggregate the useful evidence, discard unsupported or contradictory claims, and produce one improved solution.
Base your answer on the image and the question, not on candidate frequency.
If all candidates are weak, use the image directly and give your best specific answer.

Write your response in exactly this format:

Visual evidence:
- <visible cue 1>
- <visible cue 2>
- <visible cue 3>

Reasoning:
<brief explanation of which candidate evidence you used or rejected>

Final answer: <entity name>

Question:
{question}

Candidate solutions:
---- Solution 1 ----
{candidate_1}
...
```

**Aggregation-aware RL adaptation:** train on two prompt types, matching the RSA paper:

1. **Standard trace prompts** (image + question only) → train the model to generate useful initial candidate solutions.
2. **Aggregation trace prompts** (image + question + `K` candidate solutions from the reference model) → train the model to aggregate candidate evidence into one improved solution.

Open design points (to resolve before building):

- **Reward `r(τ,y)`.** Candidates, in priority order to evaluate:
  - **Taxonomy-aware graded `hF`** (the [[002]] metric) — rewards getting *near* the right entity in the P279 tree, and intrinsically penalizes under-specific (parent) answers via lower hP. Likely the most principled reward for this task.
  - **Binary judge verdict** (the [[001]] Qwen3-4B free-form judge) — simplest, but [[001]] shows it **inflates** and accepts hypernyms; a judge reward could *teach the model to hedge under-specific*, the opposite of what we want. Avoid as the sole reward, or use the **specificity-preserving** support definition from [[001]] §5.
  - **Exact match** — cleanest signal but sparse on 9k+ fine-grained entities.
- **Reward target:** reward only the parsed final answer, not the reasoning trace. RSA's repo similarly rewards final correctness, not thinking length or tags.
- **Aggregation set construction.** Standard prompt = image + question; aggregation prompt = image + question + `K` candidate solution traces sampled from the reference VLM. Train/test aggregation prompts must match, per the paper's key requirement.
- **Optimizer / init.** RLOO (paper's choice) or GRPO (matches our inference defaults, "GRPO-like" sampling temp=1.0). Init from the instruct checkpoint (`θref`).
- **Data.** OVEN train split (+ the aligned-question variant, see [[001]] §5). For each query, sample `K` reference candidates to form aggregation prompts; jointly optimize Eq 4 (initial proposals) + Eq 5 (aggregation).

**Hypotheses / risks specific to our setting:**
- RSA's "implicit verification" assumes the model judges its own candidates well. For *visual* entity recognition this hinges on **visual grounding**, not just linguistic plausibility — a text-only aggregation step could discard a visually-correct rare candidate in favor of a linguistically-common wrong one. (Mitigation: image is always in context; consider a visual-evidence emphasis in the prompt.)
- **Under-specificity feedback loop:** if the reward tolerates hypernyms, aggregation may converge on the safe parent class. Choosing a specificity-preserving reward (hF, or [[001]]'s strict support) is the guard.
- **Short-answer regime:** answer-only OVEN candidates are ≤16 tokens and contain little recombinable information. This is why trace-based candidate solutions are necessary for a faithful RSA experiment.
- Paper's own next step — **multi-step RL for the end-to-end RSA procedure** — is a natural stretch goal beyond the greedy single-step objective.

---

## 6. Open questions / TODO

- [ ] Judge and score the existing answer-aggregation RSA run with `N=16, K=4, T=10`; compare judge pass@k, restored `exact_match`/`cascade` taxonomy hF, and under-specific-rate ([[001]]) vs the naive-sampling baseline at matched and unmatched compute.
- [ ] Report answer-mode results honestly as **answer aggregation**, not full RSA over reasoning traces.
- [ ] Implement trace-based initial candidate generation and trace-based aggregation prompts before making strong claims about RSA's visual-evidence recombination.
- [ ] Measure the **`pass@N − pass@1` gap** on our naive-sampling populations as an aggregability predictor before investing in RL.
- [ ] Decide the RL reward (lead: taxonomy-aware hF; fallback: exact + strict support) and confirm train/test aggregation prompts are identical.
- [ ] Verify whether aggregation reduces or amplifies under-specific (hypernym) answers — the central tie-in to [[001]].
- [ ] Sanity-check that RSA's text aggregation does not discard visually-correct rare candidates (visual-grounding ablation: `--no-image`).

## 7. References
- Venkatraman et al., *Recursive Self-Aggregation Unlocks Deep Thinking in LLMs*, arXiv:2509.26626v2 (2026). §3 algorithm (Eq 1–3); §4 aggregation-aware RL (Eq 4–5); §5.4 hyperparameters; §5.5 RL results (Fig 9).
- Ahmadian et al., *Back to Basics: …RLOO…*, ACL 2024 (the optimizer used).
- OVEN: Hu et al., arXiv:2302.11154.
