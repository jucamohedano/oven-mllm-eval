#!/usr/bin/env python3
"""Plot hierarchical metrics (hP, hR, hF) with optional no-image baseline.

Usage::

    uv run python scripts/plot_hierarchical_metrics.py \
        --run-dirs logs/schedule/.../2b_with_image ... \
        --run-dirs2 logs/schedule/.../2b_no_image ... \
        --output viz/hierarchical_metrics.png
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
    m = re.search(r"qwen3-vl-(\d+b)", path)
    if m:
        return f"Qwen3-VL {m.group(1).upper()}"
    return Path(path).parent.parent.name


def _find_results(run_dir: Path) -> Path | None:
    for pattern in ["*_results*.json", "generations_results.json"]:
        files = sorted(run_dir.glob(pattern))
        if files:
            return files[0]
    return None


def _load_hierarchical(run_dirs: list[str]) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for run_dir in run_dirs:
        p = Path(run_dir)
        if not p.is_dir():
            print(f"[skip] {run_dir}: not a directory")
            continue
        rf = _find_results(p)
        if not rf:
            print(f"[skip] {run_dir}: no results found")
            continue
        with open(rf) as f:
            data = json.load(f)
        label = _model_label_from_path(str(p))
        results[label] = {
            "hP": data.get("hP", 0),
            "hR": data.get("hR", 0),
            "hF": data.get("hF", 0),
            "exact": data.get("exact", 0),
        }
        print(f"[ok] {label}: hP={results[label]['hP']:.4f} "
              f"hR={results[label]['hR']:.4f} hF={results[label]['hF']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Plot hierarchical metrics across models"
    )
    parser.add_argument("--run-dirs", required=True, nargs="+",
                        help="Run dirs for primary variant (solid bars).")
    parser.add_argument("--run-dirs2", default=None, nargs="+",
                        help="Run dirs for second variant (hatched, e.g. no-image).")
    parser.add_argument("--label2", default="no image",
                        help="Label for second variant (default: 'no image')")
    parser.add_argument("--output", default="viz/hierarchical_metrics.png")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    results = _load_hierarchical(args.run_dirs)
    results2 = _load_hierarchical(args.run_dirs2) if args.run_dirs2 else {}

    if not results:
        print("No results found.")
        return

    # ── Plot ────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("colorblind", 4)
    # 2B=palette[0] (blue), 4B=palette[3] (red), 8B=palette[2] (green)
    model_colors = {
        "Qwen3-VL 2B": palette[0],
        "Qwen3-VL 4B": palette[3],
        "Qwen3-VL 8B": palette[2],
    }

    order = ["Qwen3-VL 2B", "Qwen3-VL 4B", "Qwen3-VL 8B"]
    labels = [m for m in order if m in results]

    metrics = ["hF", "exact"]
    n_metrics = len(metrics)
    n_models = len(labels)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 5))

    x = np.arange(n_models)
    width = 0.35 if not results2 else 0.3

    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        vals1 = [results[m][metric] for m in labels]
        bars1 = ax.bar(x - width/2, vals1, width, label="with image",
                       color=[model_colors[m] for m in labels], alpha=0.85)

        if results2:
            vals2 = [results2.get(m, {}).get(metric, 0) for m in labels]
            bars2 = ax.bar(x + width/2, vals2, width, label=args.label2,
                           color=[model_colors[m] for m in labels], alpha=0.35,
                           hatch="//", edgecolor=[model_colors[m] for m in labels])

        for bar, val in zip(bars1, vals1):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                        fontweight="bold")
        if results2:
            for bar, val in zip(bars2, vals2):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ymax = max(max(vals1), max(vals2) if results2 else 0)
        ax.set_ylim(0, ymax * 1.2 if ymax > 0.5 else 1.0)
        ax.set_ylabel("Score", fontsize=11)
        if mi == n_metrics - 1 and results2:
            ax.legend(fontsize=9)

    fig.suptitle(
        args.title or "Hierarchical metrics — with image vs no-image baseline",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
