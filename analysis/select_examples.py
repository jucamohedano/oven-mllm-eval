#!/usr/bin/env python3
"""Stage 1 (cluster): select showcase examples for the thesis.

Joins the *standard* naive-sampling judged/scored files with the *RSA* judged
files per ``data_id`` across model sizes, applies a selection criterion, and
writes a small self-contained JSON (``selected_examples.json``) that Stage 2
(``render_examples.py``, run locally) turns into a LaTeX figure+table.

The heavy per-example JSONLs stay on the cluster; only the tiny selection JSON
(a handful of examples) and the referenced images need to be synced back.

Criteria (``--criterion``):
  rsa-beats-standard-all   RSA judged correct AND standard judged wrong, for
                           EVERY model present (strongest "RSA helps" story).
  rsa-beats-standard-any   RSA correct & standard wrong for >=1 model.
  all-correct              every model's standard sampling judged correct.
  all-wrong                every model's standard sampling judged wrong.
  random                   any example solved by >=1 model (no cherry-picking).

Usage (cluster)::

    python analysis/select_examples.py \
      --standard 2B=<...>_samples_scored_qwen_qwen3-4b_with_desc_rich.jsonl \
                 4B=<...> 8B=<...> 32B=<...> \
      --rsa 2B=<...>_samples_judged_rsa_solution_n16_k4_t5_qwen_qwen3-4b_with_desc_rich.jsonl \
            4B=<...> 8B=<...> \
      --criterion rsa-beats-standard-all \
      --num 3 --seed 0 \
      --output viz/examples/selected_examples_qwen.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Boxed-answer parsing (inlined from run_recursive_self_agg to stay dep-free)
# ---------------------------------------------------------------------------

def _strip_latex_answer(text: str) -> str:
    text = text.strip()
    # Unwrap \text{...} (possibly repeated / with trailing brace loss).
    text = re.sub(r"^\\text\{(.*)\}$", r"\1", text.strip())
    text = text.replace(r"\text{", "").replace("{", "").replace("}", "")
    text = text.replace("\\ ", " ")  # escaped spaces from latex majorities
    return text.strip()


def extract_boxed_answer_strict(text: str) -> str | None:
    """Return the last ``\\boxed{...}`` answer, or ``None`` if there is no box.

    Unlike the lenient variant, this NEVER falls back to the full trace — a
    step whose output has no clean box parses to ``None`` (treated as
    "no answer" for majority/correctness), so a verbose reasoning essay that
    merely *mentions* the gold entity is not miscounted as a correct answer.
    """
    matches = []
    start = 0
    needle = r"\boxed{"
    while True:
        box_start = text.find(needle, start)
        if box_start < 0:
            break
        content_start = box_start + len(needle)
        depth = 1
        idx = content_start
        while idx < len(text) and depth > 0:
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
            idx += 1
        if depth == 0:
            answer = _strip_latex_answer(text[content_start:idx - 1])
            if answer:
                matches.append(answer)
            start = idx
        else:
            break
    return matches[-1] if matches else None


def extract_boxed_answer(text: str) -> str:
    """Lenient: last ``\\boxed{...}`` answer, or the stripped text if none.

    Kept for the standard-sampling ``all_texts`` (label-only answers that have
    no box by design).  For RSA solution traces use the strict variant.
    """
    ans = extract_boxed_answer_strict(text)
    return ans if ans is not None else text.strip()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _parse_specs(specs: list[str]) -> dict[str, str]:
    """Parse LABEL=PATH pairs into an ordered dict."""
    out: dict[str, str] = {}
    for spec in specs:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"Bad spec (need LABEL=PATH): {spec!r}")
        out[label] = path
    return out


def _top_answers(texts: list[str], verdicts: list[int], top: int) -> list[dict]:
    """Distinct answers with counts and whether each was judged correct."""
    counts: Counter[str] = Counter()
    correct: dict[str, bool] = {}
    for text, verdict in zip(texts, verdicts):
        norm = str(text).strip()
        if not norm:
            continue
        counts[norm] += 1
        # An answer string counts as "correct" if any of its occurrences passed.
        correct[norm] = correct.get(norm, False) or bool(verdict)
    ranked = counts.most_common(top)
    return [
        {"answer": ans, "count": n, "correct": correct.get(ans, False)}
        for ans, n in ranked
    ]


def _load_standard(path: str, top: int) -> dict[str, dict]:
    """data_id -> standard-sampling summary."""
    out: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            did = r.get("data_id") or r.get("image_id")
            if not did:
                continue
            texts = r.get("all_texts", [])
            verdicts = r.get("judge_verdicts", [])
            out[did] = {
                "hit": bool(r.get("judge_hit")),
                "hit_count": int(r.get("judge_hit_count", sum(verdicts))),
                "best": r.get("judge_selected_text", ""),
                "top": _top_answers(texts, verdicts, top),
                "cascade_hF": r.get("cascade_hF"),
                "exact_hF": r.get("exact_match_hF"),
                # carry example metadata (same across judges); Stage 2 needs it
                "question": r.get("question", ""),
                "answer": r.get("answer", ""),
                "image_id": r.get("image_id", ""),
                "image_path": r.get("image_path", ""),
                "entity_id": r.get("entity_id", ""),
                "entity_text": r.get("entity_text", ""),
            }
    return out


def _load_rsa(path: str, top: int, aliases_by_canonical: dict) -> dict[str, dict]:
    """data_id -> RSA summary incl. per-iteration answer distribution.

    For each step we also record whether the *majority* boxed answer is
    deterministically correct (specificity-preserving supported match against
    the ground truth, via judge_audit).  This gives a per-iteration correctness
    signal even though the LM judge only verdicted the final population.
    ``solve_step`` is the first step whose majority answer is correct (or None).
    """
    from oven_mllm_eval.judge_audit import classify_positive, is_supported

    def _is_correct(answer: str, gold: str) -> bool:
        if not answer:
            return False
        return is_supported(classify_positive(
            prediction=answer, answer=gold, aliases_by_canonical=aliases_by_canonical))

    def _majority_correct(answers: list[str | None], gold: str) -> dict:
        """Two majority rules over the 16 traces:

        parsed-only : vote among traces with a clean \\boxed{}; no-box excluded.
        all-16      : no-box traces count as a vote for "no answer" (∅); a step
                      is correct only if a gold-matching answer is the strict
                      majority over ALL 16 (rewards keeping clean format).
        Returns majority string, both correctness flags, and parse count.
        """
        parsed = [a for a in answers if a]
        pop = len(answers)
        # parsed-only
        c_parsed = Counter(parsed)
        maj_p = c_parsed.most_common(1)[0][0] if c_parsed else ""
        ok_parsed = _is_correct(maj_p, gold)
        # all-16 (None -> sentinel that never matches gold)
        c_all = Counter(a if a else "∅" for a in answers)
        maj_a, _ = c_all.most_common(1)[0]
        ok_all = maj_a != "∅" and _is_correct(maj_a, gold)
        return {
            "majority": maj_p, "majority_correct": ok_parsed,
            "majority_correct_all": ok_all,
            "n_parsed": len(parsed), "pop_size": pop,
        }

    out: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            did = r.get("data_id") or r.get("image_id")
            if not did:
                continue
            gold = r.get("answer", "")
            verdicts = r.get("judge_verdicts", [])
            final_answers = r.get("all_texts", [])  # parsed boxed answers, final pop
            # Per-iteration: parse boxed answers from each step's population.
            steps = r.get("rsa_populations_by_step") or []
            iterations = []
            solve_step = None
            solve_step_all = None
            for step_idx, population in enumerate(steps):
                # Strict: unparsed traces → None (excluded from parsed vote, not
                # matched against gold as a whole essay).
                answers = [extract_boxed_answer_strict(str(t)) for t in population]
                counts = Counter(a for a in answers if a)
                mc = _majority_correct(answers, gold)
                if mc["majority_correct"] and solve_step is None:
                    solve_step = step_idx
                if mc["majority_correct_all"] and solve_step_all is None:
                    solve_step_all = step_idx
                iterations.append({
                    "step": step_idx,  # 0 = initial population
                    "majority": mc["majority"],
                    "majority_correct": mc["majority_correct"],
                    "majority_correct_all": mc["majority_correct_all"],
                    "n_parsed": mc["n_parsed"],      # # traces with a clean \boxed{}
                    "pop_size": mc["pop_size"],
                    "top": [{"answer": a, "count": n} for a, n in counts.most_common(top)],
                    "n": mc["pop_size"],
                })
            out[did] = {
                "hit": bool(r.get("judge_hit")),
                "hit_count": int(r.get("judge_hit_count", sum(verdicts))),
                "best": r.get("judge_selected_text", final_answers[0] if final_answers else ""),
                "final_top": _top_answers(final_answers, verdicts, top),
                "iterations": iterations,
                "num_steps": len(iterations),
                "solve_step": solve_step,          # first correct (parsed-only rule)
                "solve_step_all": solve_step_all,  # first correct (all-16 rule)
            }
    return out


def _trajectory_stats(rsa_model: dict[str, dict]) -> dict:
    """Aggregate per-step RSA trajectory statistics for one model.

    All quantities are over ALL examples with a recorded trajectory.  ``T`` is
    the number of steps (typically 5: step 0 = initial population).
    """
    rows = [r for r in rsa_model.values() if r.get("iterations")]
    n = len(rows)
    if n == 0:
        return {}
    T = max(len(r["iterations"]) for r in rows)

    def frac(pred) -> float:
        return round(sum(1 for r in rows if pred(r)) / n, 4)

    # Per-step parse rate (mean fraction of 16 traces with a clean box) and
    # per-step correct-majority rate under each rule.
    per_step = []
    for s in range(T):
        have = [r["iterations"][s] for r in rows if len(r["iterations"]) > s]
        m = len(have)
        per_step.append({
            "step": s,
            "parse_rate": round(sum(it["n_parsed"] / max(it["pop_size"], 1) for it in have) / m, 4),
            "correct_parsed": round(sum(1 for it in have if it["majority_correct"]) / m, 4),
            "correct_all16": round(sum(1 for it in have if it["majority_correct_all"]) / m, 4),
        })

    def solve_steps(key):
        return [r[key] for r in rows if r.get(key) is not None]

    stats = {"n_examples": n, "num_steps": T, "per_step": per_step}
    for rule, key in [("parsed", "solve_step"), ("all16", "solve_step_all")]:
        ss = solve_steps(key)
        ns = len(ss)
        final_correct = sum(
            1 for r in rows
            if r["iterations"][-1][
                "majority_correct" if rule == "parsed" else "majority_correct_all"]
        )
        # flipped correct->wrong: correct at step 0 but not at final step
        flip_cw = sum(
            1 for r in rows
            if r["iterations"][0]["majority_correct" if rule == "parsed" else "majority_correct_all"]
            and not r["iterations"][-1]["majority_correct" if rule == "parsed" else "majority_correct_all"]
        )
        stats[rule] = {
            "final_solve_rate": round(final_correct / n, 4),
            "ever_solved": round(ns / n, 4),
            "mean_solve_step": round(sum(ss) / ns, 3) if ns else None,
            "median_solve_step": sorted(ss)[ns // 2] if ns else None,
            "solved_at_step0_frac": round(sum(1 for s in ss if s == 0) / ns, 4) if ns else None,
            "late_solve_frac": round(sum(1 for s in ss if s >= 3) / ns, 4) if ns else None,
            "flip_correct_to_wrong": round(flip_cw / n, 4),
        }
    return stats


def _write_stats_table(solve_stats: dict, out_path: Path, judge_label: str,
                       table_label: str) -> None:
    """Write a booktabs LaTeX table of the per-model RSA trajectory summary."""
    models = [m for m in solve_stats if solve_stats[m]]
    if not models:
        return
    rows = []
    for m in models:
        p = solve_stats[m]["parsed"]
        ps = solve_stats[m]["per_step"]
        rows.append((
            m,
            p["final_solve_rate"], p["mean_solve_step"], p["median_solve_step"],
            p["solved_at_step0_frac"], p["late_solve_frac"],
            p["flip_correct_to_wrong"],
            ps[0]["parse_rate"], ps[-1]["parse_rate"],
        ))
    cap = (f"RSA aggregation dynamics per model size"
           + (f" ({judge_label} correctness)" if judge_label else "")
           + r". Solve-rate is the fraction of examples whose majority boxed "
             r"answer is correct at the final step; $T$ is the first step at "
             r"which the majority becomes correct (step 0 = initial population). "
             r"Parse-rate is the mean fraction of the 16 traces per step that "
             r"emit a clean \texttt{\textbackslash boxed\{\}} answer. Flip C$\to$W "
             r"is the fraction that are correct initially but wrong after "
             r"aggregation. Correctness is deterministic supported match.")
    lines = [
        r"% Requires \usepackage{booktabs}",
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\caption{{{cap}}}", rf"\label{{{table_label}}}",
        r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
        (r"Model & Solve & Mean $T$ & Med $T$ & Solved@0 & Late($T{\geq}3$) & "
         r"Flip C$\to$W & Parse@0 & Parse@final \\"),
        r"\midrule",
    ]
    for (m, solve, mean_t, med_t, s0, late, flip, p0, pf) in rows:
        mean_s = f"{mean_t:.2f}" if mean_t is not None else "--"
        med_s = f"{med_t}" if med_t is not None else "--"
        s0_s = f"{s0:.1%}" if s0 is not None else "--"
        late_s = f"{late:.1%}" if late is not None else "--"
        lines.append(
            f"{m} & {solve:.3f} & {mean_s} & {med_s} & {s0_s} & {late_s} & "
            f"{flip:.3f} & {p0:.2f} & {pf:.2f} \\\\".replace("%", r"\%")
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[saved] stats table: {out_path}")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _select(
    data_ids: list[str],
    standard: dict[str, dict[str, dict]],
    rsa: dict[str, dict[str, dict]],
    criterion: str,
    models: list[str],
    late_solve_min_step: int = 3,
    min_other_std_wrong: int = 2,
) -> list[str]:
    rsa_models = [m for m in models if m in rsa]

    def std_hit(did, m):
        return standard.get(m, {}).get(did, {}).get("hit", False)

    def rsa_hit(did, m):
        return rsa.get(m, {}).get(did, {}).get("hit", False)

    def late_solve(did, m) -> bool:
        """RSA majority wrong at every step < min_step, correct at some step >= min_step."""
        rr = rsa.get(m, {}).get(did)
        if not rr:
            return False
        its = rr["iterations"]
        if len(its) <= late_solve_min_step:
            return False
        early_all_wrong = all(not it["majority_correct"] for it in its[:late_solve_min_step])
        late_solved = any(it["majority_correct"] for it in its[late_solve_min_step:])
        return early_all_wrong and late_solved

    picked = []
    for did in data_ids:
        if criterion == "rsa-beats-standard-all":
            if rsa_models and all(
                rsa_hit(did, m) and not std_hit(did, m) for m in rsa_models
            ):
                picked.append(did)
        elif criterion == "rsa-beats-standard-any":
            if any(rsa_hit(did, m) and not std_hit(did, m) for m in rsa_models):
                picked.append(did)
        elif criterion == "rsa-rescues":
            # Showcase: >=1 model where RSA (judge) solves it but that model's
            # standard sampling missed, AND >=`min_other_std_wrong` OTHER models
            # also miss under standard sampling (so "RSA rescues one, others
            # fail").  "solved"/"missed" = LM judge verdict on the final pop.
            rescued = [m for m in rsa_models if rsa_hit(did, m) and not std_hit(did, m)]
            if rescued:
                others_wrong = sum(1 for m in models if not std_hit(did, m))
                # subtract the rescued model itself so the count is "other" models
                if (others_wrong - 1) >= min_other_std_wrong:
                    picked.append(did)
        elif criterion == "rsa-late-solve":
            # RSA solves late (majority wrong at every step < min_step, correct
            # at/after it) for AT LEAST ONE model, and standard sampling misses
            # for that same model.  The renderer shows the first such model.
            if any(late_solve(did, m) and not std_hit(did, m) for m in rsa_models):
                picked.append(did)
        elif criterion == "all-correct":
            if all(std_hit(did, m) for m in models):
                picked.append(did)
        elif criterion == "all-wrong":
            if all(not std_hit(did, m) for m in models):
                picked.append(did)
        elif criterion == "random":
            if any(std_hit(did, m) for m in models):
                picked.append(did)
        else:
            raise SystemExit(f"Unknown criterion: {criterion}")
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description="Select showcase examples (Stage 1, cluster)")
    ap.add_argument("--standard", nargs="+", required=True,
                    help="LABEL=PATH standard scored JSONLs (e.g. 2B=... 4B=... 8B=... 32B=...)")
    ap.add_argument("--rsa", nargs="+", default=[],
                    help="LABEL=PATH RSA judged JSONLs (e.g. 2B=... 4B=... 8B=...)")
    ap.add_argument("--criterion", default="rsa-beats-standard-all",
                    choices=["rsa-beats-standard-all", "rsa-beats-standard-any",
                             "rsa-rescues", "all-correct", "all-wrong", "random",
                             "rsa-late-solve"])
    ap.add_argument("--min-other-std-wrong", type=int, default=2,
                    help="For --criterion rsa-rescues: how many OTHER models "
                         "(besides the RSA-rescued one) must also miss under "
                         "standard sampling. Default 2.")
    ap.add_argument("--late-solve-min-step", type=int, default=3,
                    help="For --criterion rsa-late-solve: the RSA majority answer "
                         "must be WRONG at every step < this and CORRECT at some "
                         "step >= this. Default 3 (wrong at steps 0,1,2; right at "
                         ">=3). Requires all RSA models to show the pattern.")
    ap.add_argument("--taxonomy-index", default="data/processed/oven_taxonomy_index.json",
                    help="Taxonomy index (for per-step deterministic correctness).")
    ap.add_argument("--num", type=int, default=3, help="How many examples to select.")
    ap.add_argument("--top", type=int, default=6,
                    help="Top-N distinct answers to record per model (for appendix).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge-label", default="",
                    help="Human label for the judge used (recorded in the JSON).")
    ap.add_argument("--available-images", default=None,
                    help="Optional text file of image_ids present on the cluster "
                         "(one per line, with or without .jpg). When set, only "
                         "examples whose image is available are selected — "
                         "avoids picking examples whose image cannot be pulled. "
                         "Build it on the cluster with: "
                         "ls <image_dir> | sed 's/\\.jpg$//' > available_images.txt")
    ap.add_argument("--output", required=True, help="Output selection JSON path.")
    ap.add_argument("--stats-table", default=None,
                    help="Optional path for a booktabs LaTeX table of the "
                         "per-model RSA trajectory statistics.")
    ap.add_argument("--stats-table-label", default="tab:rsa-trajectory",
                    help="LaTeX \\label for --stats-table.")
    args = ap.parse_args()

    available_images: set[str] | None = None
    if args.available_images:
        available_images = set()
        for line in Path(args.available_images).read_text().splitlines():
            name = line.strip()
            if not name:
                continue
            if name.endswith(".jpg") or name.endswith(".jpeg"):
                name = name.rsplit(".", 1)[0]
            available_images.add(name)
        print(f"[images] {len(available_images)} available image_ids loaded from manifest")

    std_specs = _parse_specs(args.standard)
    rsa_specs = _parse_specs(args.rsa) if args.rsa else {}
    models = list(std_specs.keys())

    # Alias map for per-step deterministic correctness (RSA iteration trace).
    aliases_by_canonical: dict = {}
    if rsa_specs:
        from oven_mllm_eval.judge_audit import build_alias_map
        index = json.loads(Path(args.taxonomy_index).read_text())
        aliases_by_canonical = build_alias_map(index)

    print(f"[load] standard: {list(std_specs)}  rsa: {list(rsa_specs)}")
    standard = {m: _load_standard(p, args.top) for m, p in std_specs.items()}
    rsa = {m: _load_rsa(p, args.top, aliases_by_canonical) for m, p in rsa_specs.items()}

    # Candidate data_ids = intersection across all standard files (same eval set).
    common = None
    for m, d in standard.items():
        ids = set(d.keys())
        common = ids if common is None else (common & ids)
    common = sorted(common or [])
    print(f"[join] {len(common)} common data_ids across standard files")

    # ── RSA trajectory statistics (independent of example selection) ────
    solve_stats = {m: _trajectory_stats(rsa[m]) for m in rsa_specs}
    if any(solve_stats.values()):
        print("\n[trajectory] per-model RSA dynamics (parsed-only rule):")
        for m, st in solve_stats.items():
            if not st:
                continue
            p = st["parsed"]
            ps = st["per_step"]
            s0 = f"{p['solved_at_step0_frac']:.1%}" if p['solved_at_step0_frac'] is not None else "n/a"
            late = f"{p['late_solve_frac']:.1%}" if p['late_solve_frac'] is not None else "n/a"
            print(f"  {m}: final solve-rate={p['final_solve_rate']:.3f}  "
                  f"ever-solved={p['ever_solved']:.3f}  "
                  f"mean T={p['mean_solve_step']}  median={p['median_solve_step']}  "
                  f"solved@0={s0}  late(T≥3)={late}  "
                  f"flip C→W={p['flip_correct_to_wrong']:.3f}")
            print("       parse-rate by step: "
                  + " ".join(f"t{s['step']}={s['parse_rate']:.2f}" for s in ps))
            print("       correct(parsed)   : "
                  + " ".join(f"t{s['step']}={s['correct_parsed']:.3f}" for s in ps))
            print("       correct(all-16)   : "
                  + " ".join(f"t{s['step']}={s['correct_all16']:.3f}" for s in ps))
    if args.stats_table:
        _write_stats_table(solve_stats, Path(args.stats_table),
                           args.judge_label, args.stats_table_label)

    matches = _select(common, standard, rsa, args.criterion, models,
                      late_solve_min_step=args.late_solve_min_step,
                      min_other_std_wrong=args.min_other_std_wrong)
    print(f"[select] criterion={args.criterion}: {len(matches)} matches")
    if not matches:
        raise SystemExit("No examples matched the criterion.")

    # Keep only examples whose image is actually available (if a manifest was
    # given) — avoids selecting examples that cannot be rendered.
    if available_images is not None:
        def _img_id(did: str) -> str:
            for m in models:
                s = standard[m].get(did)
                if s:
                    return s["image_id"]
            return ""
        before = len(matches)
        matches = [d for d in matches if _img_id(d) in available_images]
        print(f"[images] {len(matches)}/{before} matches have an available image")
        if not matches:
            raise SystemExit("No matched examples have an available image.")

    rng = random.Random(args.seed)
    rng.shuffle(matches)
    chosen = matches[:args.num]

    examples = []
    for did in chosen:
        # Metadata from any standard file that has it (they agree).
        meta = next(standard[m][did] for m in models if did in standard[m])
        ex = {
            "data_id": did,
            "image_id": meta["image_id"],
            "image_path": meta["image_path"],
            "entity_id": meta["entity_id"],
            "entity_text": meta["entity_text"],
            "question": meta["question"],
            "answer": meta["answer"],
            "standard": {},
            "rsa": {},
        }
        for m in models:
            s = standard[m].get(did)
            if s:
                ex["standard"][m] = {
                    "hit": s["hit"], "hit_count": s["hit_count"],
                    "best": s["best"], "top": s["top"],
                    "cascade_hF": s["cascade_hF"], "exact_hF": s["exact_hF"],
                }
        for m in rsa_specs:
            rr = rsa[m].get(did)
            if rr:
                ex["rsa"][m] = {
                    "hit": rr["hit"], "hit_count": rr["hit_count"],
                    "best": rr["best"], "final_top": rr["final_top"],
                    "iterations": rr["iterations"], "num_steps": rr["num_steps"],
                    "solve_step": rr["solve_step"],
                }
        examples.append(ex)

    out = {
        "criterion": args.criterion,
        "judge_label": args.judge_label,
        "models_standard": models,
        "models_rsa": list(rsa_specs.keys()),
        "num_selected": len(examples),
        "seed": args.seed,
        "late_solve_min_step": args.late_solve_min_step,
        "solve_iteration_stats": solve_stats,
        "examples": examples,
        # Images to pull back (Stage 2 needs them): QIDs + image_ids.
        "images_needed": [
            {"entity_id": ex["entity_id"], "image_id": ex["image_id"]}
            for ex in examples
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[saved] {output_path}  ({len(examples)} examples)")
    print("[images] pull these QIDs back with sync.sh --pull-images:")
    print("   " + ",".join(ex["entity_id"] for ex in examples if ex["entity_id"]))


if __name__ == "__main__":
    main()
