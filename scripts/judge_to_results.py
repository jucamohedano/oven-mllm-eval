#!/usr/bin/env python3
"""Compute pass@k and judge stats from a judged JSONL; write a results JSON.

Does NOT touch the taxonomy — just the binary judge verdicts.  Useful to get
pass@k numbers quickly before running the full taxonomy-aware scoring pipeline
(e.g. while the cascade measure is being reworked).

Usage::

    python scripts/judge_to_results.py \\
        --judged <run_dir>/<run_id>_samples_judged_<suffix>.jsonl

Output is written alongside the judged file::

    <run_dir>/<run_id>_results_<suffix>.json

The results JSON follows the ``{"measures": [{"measure", "metrics": {...}}]}``
format, so ``plot_metrics_from_results.py`` can consume it (pass@k-only;
hP/hR/hF panels will be empty unless taxonomy measures are added later).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oven_mllm_eval.pass_at_k import pass_at_k


def _load_judge_metadata(judged_path: Path) -> dict:
    candidates = [
        judged_path.with_suffix("").with_name(f"{judged_path.stem}_metadata.json"),
    ]
    candidates.extend(sorted(judged_path.parent.glob(f"{judged_path.name}_shard*_metadata.json")))
    candidates.extend(sorted(judged_path.parent.glob(f"{judged_path.stem}_shard*_metadata.json")))
    for path in candidates:
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute pass@k + judge stats from a judged JSONL")
    ap.add_argument("--judged", required=True,
                    help="Path to *_judged_*.jsonl (from run_judge.py)")
    ap.add_argument("--output", default=None,
                    help="Results JSON output path. Default: auto-derive from "
                         "--judged (replace _samples_judged_ → _results_, "
                         ".jsonl → .json).")
    ap.add_argument("--base-summary", default=None,
                    help="Optional existing taxonomy results JSON. When set, "
                         "copy it and inject judge/pass@k metrics instead of "
                         "writing a judge-only summary.")
    ap.add_argument("--judged2", default=None,
                    help="Second judged JSONL for dual-judge agreement mode. "
                         "When set, a rollout is counted as correct only when "
                         "*both* judges agree (AND of verdicts). The two files "
                         "must share the same examples (inner join on data_id).")
    ap.add_argument("--n", type=int, default=None,
                    help="Rollouts per example. Default: infer from each row's "
                         "judge_verdicts length.")
    args = ap.parse_args()

    judged_path = Path(args.judged)
    if not judged_path.exists():
        ap.error(f"--judged file not found: {judged_path}")

    dual = args.judged2 is not None
    if dual:
        judged2_path = Path(args.judged2)
        if not judged2_path.exists():
            ap.error(f"--judged2 file not found: {judged2_path}")

    # Auto-derive output path:  <dir>/<run_id>_results_<suffix>.json
    if args.output:
        output_path = Path(args.output)
    else:
        stem = judged_path.name
        if stem.endswith(".jsonl"):
            stem = stem[:-6]
        stem = re.sub(r"_samples_judged_|_judged_", "_results_", stem, count=1)
        if dual:
            stem += "_dual"
        output_path = judged_path.parent / f"{stem}.json"

    # ── Load judged files ────────────────────────────────────────────
    def _load_judged(path: Path) -> dict[str, list[int]]:
        """Return {data_id: [verdict0, verdict1, …]} for valid rows."""
        rows: dict[str, list[int]] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                verdicts = row.get("judge_verdicts")
                if not verdicts:
                    continue
                did = row.get("data_id", row.get("image_id", ""))
                if did:
                    rows[did] = [int(v) for v in verdicts]
        return rows

    j1 = _load_judged(judged_path)
    if not j1:
        ap.error("No rows with judge_verdicts found in --judged — is this a judged file?")

    if dual:
        j2 = _load_judged(judged2_path)
        if not j2:
            ap.error("No rows with judge_verdicts found in --judged2")
        common = sorted(set(j1) & set(j2))
        if not common:
            ap.error("No common data_id between --judged and --judged2")
        skipped1 = len(j1) - len(common)
        skipped2 = len(j2) - len(common)
        if skipped1 or skipped2:
            print(f"  [info] inner-join: {len(common)} matched, "
                  f"skipped {skipped1} (j1-only) + {skipped2} (j2-only)")
        # Dual-judge: rollout hit only if BOTH judges say 1
        counts: list[tuple[int, int]] = []
        judge_hits = 0
        total = 0
        agree_rollouts = 0
        disagree_rollouts = 0
        for did in common:
            v1 = j1[did]
            v2 = j2[did]
            n = args.n if args.n is not None else min(len(v1), len(v2))
            v1 = v1[:n]
            v2 = v2[:n]
            c = sum(a and b for a, b in zip(v1, v2))
            # agreement stats
            a_agree = sum(a == b for a, b in zip(v1, v2))
            agree_rollouts += a_agree
            disagree_rollouts += (n - a_agree)
            counts.append((n, c))
            if c > 0:
                judge_hits += 1
            total += 1
        measure_name = "judge_agreement"
        extra_meta = {
            "judge_model_1": args.judged,
            "judge_model_2": args.judged2,
            "examples_matched": total,
            "examples_skipped_j1_only": skipped1,
            "examples_skipped_j2_only": skipped2,
            "rollout_agreements": agree_rollouts,
            "rollout_disagreements": disagree_rollouts,
            "rollout_agreement_rate": round(
                agree_rollouts / (agree_rollouts + disagree_rollouts), 6
            ) if (agree_rollouts + disagree_rollouts) > 0 else 0.0,
        }
        unparseable = 0  # dual mode: no per-judge parse stats
    else:
        counts = []
        judge_hits = 0
        total = 0
        unparseable = 0
        for did, verdicts in j1.items():
            n = args.n if args.n is not None else len(verdicts)
            c = int(sum(verdicts[:n]))
            counts.append((n, c))
            if c > 0:
                judge_hits += 1
            total += 1
        measure_name = "judge"
        extra_meta = {}

    if total == 0:
        ap.error("No rows with judge_verdicts found — is this a judged file?")

    # ── Compute pass@k (mean over examples) ─────────────────────────
    n_values = [n for n, _ in counts]
    c_values = [c for _, c in counts]
    n_max = max(n_values)
    k_vals = [k for k in [1, 2, 4, 8, 16, 32, 64, 128, n_max] if k <= n_max]
    k_vals = sorted(set(k_vals))
    pk_metrics: dict[str, float] = {}
    for k in k_vals:
        pk_metrics[f"pass@{k}"] = float(
            np.mean([pass_at_k(n, c, k) for n, c in counts if k <= n]))

    rollout_fields = (
        {"num_rollouts_per_example": n_values[0]}
        if len(set(n_values)) == 1
        else {
            "num_rollouts_per_example_min": min(n_values),
            "num_rollouts_per_example_max": max(n_values),
        }
    )

    judge_metrics = {
        **pk_metrics,
        "num_examples": total,
        "judge_hit": judge_hits,
        "judge_hit_frac": round(judge_hits / total, 6) if total else 0.0,
        "judge_hit_count_mean": round(float(np.mean(c_values)), 1),
        "judge_hit_count_median": float(np.median(c_values)),
        **rollout_fields,
        "num_judge_unparseable": unparseable,
    }

    # ── Write results JSON ──────────────────────────────────────────
    if args.base_summary:
        base_summary = Path(args.base_summary)
        if not base_summary.exists():
            ap.error(f"--base-summary file not found: {base_summary}")
        results = json.loads(base_summary.read_text(encoding="utf-8"))
        if "measures" in results:
            for entry in results["measures"]:
                entry.setdefault("metrics", {}).update(judge_metrics)
        else:
            results.update(judge_metrics)
    else:
        results = {
        "measures": [
            {
                "measure": measure_name,
                "metrics": judge_metrics,
            }
        ]
        }
        if extra_meta:
            results.update(extra_meta)  # dual-judge: agreement stats at top level

    metadata = _load_judge_metadata(judged_path)
    if metadata:
        results["judge_model"] = metadata.get("judge_model", "unknown")
        results["judge_mode"] = metadata.get("judge_mode", "unknown")
        results["judge_with_desc"] = metadata.get("judge_with_desc", False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fout:
        json.dump(results, fout, indent=2)

    print(f"[saved] {output_path}")
    if dual and extra_meta:
        agree = extra_meta["rollout_agreements"]
        disagree = extra_meta["rollout_disagreements"]
        print(f"  judge agreement: {agree} agree + {disagree} disagree "
              f"({extra_meta['rollout_agreement_rate']:.4f} agree rate)")
    print(f"  examples={total}  judge_hit={judge_hits} "
          f"({judge_hits/total:.4f})  cᵢ μ={np.mean(c_values):.1f}  "
          f"m={np.median(c_values):.0f}")
    for k in k_vals:
        print(f"  pass@{k:>3d} = {pk_metrics[f'pass@{k}']:.6f}")


if __name__ == "__main__":
    main()
