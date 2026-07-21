#!/usr/bin/env python3
"""Plot modal count gap: when a smaller model succeeds and a larger one fails.

Shows a dual histogram of the most-frequent-answer count (modal) on the
intersection where model A hits and model B misses.  The modal count
measures how "stubborn" a model is: a high modal count (e.g., 250+/256)
means the model repeats the same wrong answer, while a low modal count
means it explores different answers.

Usage::

    uv run python scripts/plot_modal_gap.py \
        --scored-a logs/schedule/.../2b_run/*_scored.jsonl \
        --scored-b logs/schedule/.../8b_run/*_scored.jsonl \
        --label-a "Qwen3-VL 2B" --label-b "Qwen3-VL 8B" \
        --output viz/modal_gap_2b_vs_8b.png
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def _model_label_from_path(path: str) -> str:
    m = re.search(r"qwen_qwen3-vl-(\d+b)", path)
    if m:
        return f"Qwen3-VL {m.group(1).upper()}"
    m = re.search(r"qwen3-vl-(\d+b)", path)
    if m:
        return f"Qwen3-VL {m.group(1).upper()}"
    return Path(path).parent.parent.name


def _find_scored(run_dir: Path) -> Path | None:
    """Find scored JSONL in *run_dir*, trying multiple naming conventions."""
    for pattern in ["*_scored.jsonl", "*_samples_scored.jsonl"]:
        files = sorted(run_dir.glob(pattern))
        if files:
            return files[0]
    return None


def _load(scored_path: str) -> dict[str, dict]:
    data: dict[str, dict] = {}
    with open(scored_path) as f:
        for line in f:
            r = json.loads(line.strip())
            data[r["data_id"]] = r
    print(f"  {scored_path}: {len(data)} examples")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Plot answer diversity gap between two models"
    )
    parser.add_argument("--scored-a", required=True,
                        help="Path to model A _scored.jsonl (the one that succeeds)")
    parser.add_argument("--scored-b", required=True,
                        help="Path to model B _scored.jsonl (the one that fails)")
    parser.add_argument("--label-a", default=None,
                        help="Label for model A (default: auto-detect)")
    parser.add_argument("--label-b", default=None,
                        help="Label for model B (default: auto-detect)")
    parser.add_argument("--output", default="viz/diversity_gap.png",
                        help="Output image path")
    parser.add_argument("--title", default=None,
                        help="Plot title")
    args = parser.parse_args()

    label_a = args.label_a or _model_label_from_path(args.scored_a)
    label_b = args.label_b or _model_label_from_path(args.scored_b)

    # ── Load ────────────────────────────────────────────────────────
    path_a = Path(args.scored_a)
    if path_a.is_dir():
        f = _find_scored(path_a)
        if not f:
            print(f"Error: no scored file in {path_a}"); return
        path_a = f
    path_b = Path(args.scored_b)
    if path_b.is_dir():
        f = _find_scored(path_b)
        if not f:
            print(f"Error: no scored file in {path_b}"); return
        path_b = f

    data_a = _load(str(path_a))
    data_b = _load(str(path_b))

    # ── Intersection + filter ───────────────────────────────────────
    common = set(data_a) & set(data_b)
    print(f"  Shared: {len(common)}")

    modal_a: list[int] = []
    modal_b: list[int] = []

    for did in common:
        va = any(data_a[did].get("judge_verdicts", []))
        vb = any(data_b[did].get("judge_verdicts", []))
        if not (va and not vb):
            continue

        ta = data_a[did].get("all_texts", [])
        tb = data_b[did].get("all_texts", [])

        ca = Counter(ta)
        cb = Counter(tb)
        modal_a.append(ca.most_common(1)[0][1])  # count of most common answer
        modal_b.append(cb.most_common(1)[0][1])

    n = len(modal_a)
    pct = n / len(common) * 100
    print(f"  {label_a} hits, {label_b} misses: {n} ({pct:.1f}%)")

    # ── Stats ───────────────────────────────────────────────────────
    # % where modal ≥ 128 (more than half the rollouts are identical)
    hi_a = sum(1 for m in modal_a if m >= 128)
    hi_b = sum(1 for m in modal_b if m >= 128)
    print(f"  {label_a} with modal ≥ 128: {hi_a} ({hi_a/n*100:.1f}%)")
    print(f"  {label_b} with modal ≥ 128: {hi_b} ({hi_b/n*100:.1f}%)")
    print(f"  {label_a} with modal = 256: {sum(1 for m in modal_a if m == 256)} ({sum(1 for m in modal_a if m == 256)/n*100:.1f}%)")
    print(f"  {label_b} with modal = 256: {sum(1 for m in modal_b if m == 256)} ({sum(1 for m in modal_b if m == 256)/n*100:.1f}%)")

    # ── Plot ────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import seaborn as sns
    import numpy as np

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("colorblind", 4)
    color_a = palette[0]  # blue   (2B)
    color_b = palette[2]  # green  (8B)

    # Bin edges: powers of 2 over the full 1–256 range
    bin_edges = [1, 2, 4, 8, 16, 32, 64, 128, 257]
    bin_labels = ["1", "2–3", "4–7", "8–15", "16–31", "32–63", "64–127", "128–256"]

    counts_a = [sum(1 for m in modal_a if bin_edges[i] <= m < bin_edges[i+1])
                for i in range(len(bin_edges) - 1)]
    counts_b = [sum(1 for m in modal_b if bin_edges[i] <= m < bin_edges[i+1])
                for i in range(len(bin_edges) - 1)]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(bin_labels))
    width = 0.35

    bars_a = ax.bar(x - width/2, counts_a, width, label=f"{label_a} (correct)",
                    color=color_a, alpha=0.85)
    bars_b = ax.bar(x + width/2, counts_b, width, label=f"{label_b} (wrong)",
                    color=color_b, alpha=0.85)

    for bar, count in zip(bars_a, counts_a):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.02,
                    str(count), ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color=color_a)
    for bar, count in zip(bars_b, counts_b):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.02,
                    str(count), ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color=color_b)

    ax.set_xlabel("Most frequent answer count (out of 256 rollouts)", fontsize=12)
    ax.set_ylabel("Number of examples", fontsize=12)
    ax.set_title(
        args.title
        or f"Modal count gap: when {label_a} is correct and {label_b} is wrong\n"
           f"({n:,} examples, {pct:.1f}% of dataset)",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_yscale("log")
    ax.set_ylim(0.5, max(max(counts_a), max(counts_b)) * 1.3)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
