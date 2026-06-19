#!/usr/bin/env python3
"""Plot the distribution of cᵢ (correct rollouts per example) across models.

cᵢ = number of the 256 rollouts the judge marked correct.  This is the
quantity the unbiased pass@k estimator is built on.

Two panels:
  Left  — histogram of cᵢ | cᵢ ≥ 1, binned log-ish.
  Right — CDF: fraction of solved examples with cᵢ ≤ t.

Also shows the *differential* subset: examples where the 2B model has
cᵢ ≥ 1 but the 8B has cᵢ = 0 — these are the wins driving the inversion.

Usage::

    uv run python scripts/plot_ci_distribution.py \
        --scored-2b logs/schedule/.../2b_run \
        --scored-4b logs/schedule/.../4b_run \
        --scored-8b logs/schedule/.../8b_run \
        --output viz/ci_distribution.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def _model_label_from_path(path: str) -> str:
    m = re.search(r"qwen_qwen3-vl-(\d+b)", path)
    if m:
        return f"Qwen3-VL {m.group(1).upper()}"
    return Path(path).parent.parent.name


def _find_scored(run_dir: Path) -> Path | None:
    for pattern in ["*_scored.jsonl", "*_samples_scored.jsonl"]:
        files = sorted(run_dir.glob(pattern))
        if files:
            return files[0]
    return None


def _load_ci(path: str) -> tuple[list[int], int, int]:
    """Return (all_ci, num_solved, total_examples)."""
    ci_vals: list[int] = []
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            v = r.get("judge_verdicts", [])
            if v:
                ci_vals.append(sum(v))
    solved = sum(1 for c in ci_vals if c >= 1)
    return ci_vals, solved, len(ci_vals)


def _load_data_id_map(path: str) -> dict[str, int]:
    result: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            v = r.get("judge_verdicts", [])
            if v:
                result[r["data_id"]] = sum(v)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Plot cᵢ distribution across models"
    )
    parser.add_argument("--scored-2b", required=True,
                        help="Path to 2B _scored.jsonl (or directory)")
    parser.add_argument("--scored-4b", required=True,
                        help="Path to 4B _scored.jsonl (or directory)")
    parser.add_argument("--scored-8b", required=True,
                        help="Path to 8B _scored.jsonl (or directory)")
    parser.add_argument("--output", default="viz/ci_distribution.png")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    # Resolve paths
    models: dict[str, dict] = {}
    for size, arg in [("2B", args.scored_2b), ("4B", args.scored_4b), ("8B", args.scored_8b)]:
        p = Path(arg)
        if p.is_dir():
            f = _find_scored(p)
            if not f:
                print(f"Error: no scored file in {p}"); return
            p = f
        ci_all, solved, total = _load_ci(str(p))
        models[size] = {
            "ci_all": ci_all,
            "solved": solved,
            "total": total,
            "label": _model_label_from_path(str(p)),
            "path": str(p),
        }
        print(f"  {models[size]['label']}: {total} examples, "
              f"solved={solved} ({solved/total*100:.1f}%), "
              f"mean_cᵢ={np.mean(ci_all):.1f}")

    # Differentials
    ci_2b_map = _load_data_id_map(models["2B"]["path"])
    ci_4b_map = _load_data_id_map(models["4B"]["path"])
    ci_8b_map = _load_data_id_map(models["8B"]["path"])
    common_28 = set(ci_2b_map) & set(ci_8b_map)
    common_24 = set(ci_2b_map) & set(ci_4b_map)

    def _compute_diff(ci_a_map, ci_b_map, common_set):
        diff = []
        for did in common_set:
            if ci_a_map[did] >= 1 and ci_b_map[did] == 0:
                diff.append(ci_a_map[did])
        return diff

    diff_8b = _compute_diff(ci_2b_map, ci_8b_map, common_28)
    diff_4b = _compute_diff(ci_2b_map, ci_4b_map, common_24)
    diffs = {"8B": diff_8b, "4B": diff_4b}

    for name, d in diffs.items():
        n = len(d)
        print(f"  Differential (2B solves, {name} doesn't): {n} examples")
        if n:
            print(f"    mean cᵢ = {np.mean(d):.1f}, "
                  f"cᵢ=1: {sum(1 for c in d if c==1)} ({sum(1 for c in d if c==1)/n*100:.1f}%), "
                  f"cᵢ≤2: {sum(1 for c in d if c<=2)} ({sum(1 for c in d if c<=2)/n*100:.1f}%), "
                  f"cᵢ≤4: {sum(1 for c in d if c<=4)} ({sum(1 for c in d if c<=4)/n*100:.1f}%)")

    # ── Plot ────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("colorblind", 5)
    colors = {"2B": palette[0], "4B": palette[3], "8B": palette[2], "diff": palette[1]}

    bin_edges = [1, 2, 3, 5, 9, 17, 33, 65, 129, 257]
    bin_labels = ["1", "2", "3–4", "5–8", "9–16", "17–32", "33–64", "65–128", "129–256"]

    fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(bin_labels))
    width = 0.22
    order = ["2B", "4B", "8B"]

    # ── Left: Histogram of cᵢ | cᵢ ≥ 1 (as % of solved examples) ──
    all_pct_max = 0
    for idx, size in enumerate(order):
        ci_pos = [c for c in models[size]["ci_all"] if c >= 1]
        n_solved = len(ci_pos)
        pcts = [sum(1 for c in ci_pos if bin_edges[i] <= c < bin_edges[i+1]) / n_solved * 100
                for i in range(len(bin_edges) - 1)]
        all_pct_max = max(all_pct_max, max(pcts))
        offset = (idx - 1) * width
        bar = ax_hist.bar(x + offset, pcts, width, label=models[size]["label"],
                          color=colors[size], alpha=0.85)
        for b, pct in zip(bar, pcts):
            if pct > 0.5:
                ax_hist.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                             f"{pct:.1f}%", ha="center", va="bottom", fontsize=6,
                             fontweight="bold", color=colors[size])

    ax_hist.set_xlabel("cᵢ (correct rollouts out of 256)", fontsize=11)
    ax_hist.set_ylabel("% of solved examples", fontsize=11)
    ax_hist.set_title("Histogram of cᵢ | cᵢ ≥ 1", fontsize=12)
    ax_hist.set_xticks(x)
    ax_hist.set_xticklabels(bin_labels, fontsize=9)
    ax_hist.legend(fontsize=9)
    ax_hist.set_ylim(0, all_pct_max * 1.25)

    # ── Right: CDF ──
    for size in order:
        ci_pos = sorted([c for c in models[size]["ci_all"] if c >= 1])
        if not ci_pos:
            continue
        t = np.arange(1, max(ci_pos) + 1)
        cdf = [sum(1 for c in ci_pos if c <= ti) / len(ci_pos) for ti in t]
        ax_cdf.plot(t, cdf, linewidth=1.8, label=models[size]["label"],
                    color=colors[size], alpha=0.9)

    diff_colors = {"8B": palette[2], "4B": palette[3]}
    diff_linestyles = {"8B": "--", "4B": ":"}
    for name, d in diffs.items():
        if not d:
            continue
        ds = sorted(d)
        t_d = np.arange(1, max(ds) + 1)
        cdf_d = [sum(1 for c in ds if c <= ti) / len(ds) for ti in t_d]
        ax_cdf.plot(t_d, cdf_d, linewidth=1.8, linestyle=diff_linestyles[name],
                    label=f"2B wins, {name} misses (n={len(d)})",
                    color=diff_colors[name], alpha=0.85)

    ax_cdf.set_xlabel("t (cᵢ ≤ t)", fontsize=11)
    ax_cdf.set_ylabel("Fraction of solved examples", fontsize=11)
    ax_cdf.set_title("CDF of cᵢ | cᵢ ≥ 1", fontsize=12)
    ax_cdf.legend(fontsize=8.5)
    ax_cdf.set_xscale("log", base=2)
    ax_cdf.set_xlim(1, 256)
    ax_cdf.set_ylim(0, 1.02)

    # Stats annotation
    stats_lines = []
    for size in order:
        ci_pos = [c for c in models[size]["ci_all"] if c >= 1]
        s1 = sum(1 for c in ci_pos if c == 1) / len(ci_pos) * 100
        s2 = sum(1 for c in ci_pos if c <= 2) / len(ci_pos) * 100
        s4 = sum(1 for c in ci_pos if c <= 4) / len(ci_pos) * 100
        stats_lines.append(
            f"{models[size]['label']}: cᵢ=1 {s1:.0f}%,  cᵢ≤2 {s2:.0f}%,  cᵢ≤4 {s4:.0f}%"
        )
    for name, d in diffs.items():
        if not d:
            continue
        n = len(d)
        s1d = sum(1 for c in d if c == 1) / n * 100
        s2d = sum(1 for c in d if c <= 2) / n * 100
        s4d = sum(1 for c in d if c <= 4) / n * 100
        stats_lines.append(
            f"Δ 2B wins/{name} misses: cᵢ=1 {s1d:.0f}%,  cᵢ≤2 {s2d:.0f}%,  cᵢ≤4 {s4d:.0f}%"
        )
    ax_cdf.text(0.98, 0.05, "\n".join(stats_lines), transform=ax_cdf.transAxes,
                fontsize=7.5, verticalalignment="bottom", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(
        args.title or "cᵢ distribution — correct rollouts per example",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
