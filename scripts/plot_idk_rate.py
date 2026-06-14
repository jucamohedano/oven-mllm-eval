#!/usr/bin/env python3
"""Plot "I don't know" rate across models from scored judge outputs.

Counts exact IDK matches per model and produces a bar chart showing the
fraction of rollouts where the model explicitly refuses to answer.

Usage::

    uv run python scripts/plot_idk_rate.py \
        --scored-dir logs/schedule/oven_naive-sampling_concise \
        --output viz/idk_rate.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Exact IDK strings to count (case-insensitive after stripping)
IDK_EXACT = {
    "i don't know",
    "i don't know.",
    "i don't know,",
    "i dont know",
    "i dont know.",
}

IDK_FUZZY_PATTERNS = [
    re.compile(p) for p in [
        r"\bi don'?t know\b",
        r"\bnot sure\b",
        r"\bcannot (?:determine|identify|tell|answer|say)\b",
        r"\bcan'?t (?:determine|identify|tell|answer|say)\b",
        r"\bno idea\b",
        r"\bunsure\b",
        r"\bunknown\b",
        r"\bunable to (?:determine|identify|tell)\b",
        r"\bimpossible to (?:determine|identify|tell|say)\b",
        r"\bnot possible to (?:determine|identify|tell)\b",
    ]
]

def _model_label(slug: str) -> str:
    m = re.search(r"-(\d+b)-", slug)
    size = m.group(1).upper() if m else slug
    if "qwen3-vl" in slug:
        return f"Qwen3-VL {size}"
    if "qwen2-vl" in slug:
        return f"Qwen2-VL {size}"
    return slug


def _compute_idk(scored_file: Path) -> tuple[int, int, int]:
    """Return (total, idk_exact, idk_fuzzy) for a scored JSONL file."""
    total = 0
    idk_exact = 0
    idk_fuzzy = 0
    with open(scored_file) as f:
        for line in f:
            row = json.loads(line.strip())
            for text in row.get("all_texts", []):
                total += 1
                t = text.strip().lower()
                if t in IDK_EXACT:
                    idk_exact += 1
                    idk_fuzzy += 1
                elif any(p.search(t) for p in IDK_FUZZY_PATTERNS):
                    idk_fuzzy += 1
    return total, idk_exact, idk_fuzzy


def _model_label_from_path(run_dir: Path) -> str:
    """Derive a model label from the run directory path.

    ``.../qwen_qwen3-vl-2b-instruct/20260614_014326_932940`` → ``Qwen3-VL 2B``
    """
    model_dir = run_dir.parent.name
    return _model_label(model_dir)


def plot_idk_rate(
    results: dict[str, dict[str, float]],
    output_path: str,
    title: str | None = None,
):
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(results.keys())
    exact_vals = [results[l]["exact"] for l in labels]

    x = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, exact_vals, width, color="#d62728", alpha=0.85)

    ax.set_ylabel("Rollouts (%)", fontsize=11)
    ax.set_title(title or "\"I don't know\" rate by model — concise prompt", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, max(exact_vals) * 1.12)

    for bar, val in zip(bars, exact_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot IDK rate across models from scored judge outputs"
    )
    parser.add_argument(
        "--run-dirs", required=True, nargs="+",
        help="Run directories to plot (space-separated). "
             "Each dir should contain a *_scored.jsonl file.",
    )
    parser.add_argument(
        "--output", default="viz/idk_rate.png",
        help="Output image path (default: viz/idk_rate.png)",
    )
    parser.add_argument("--title", default=None, help="Plot title")
    args = parser.parse_args()

    results: dict[str, dict[str, float]] = {}
    for run_dir in args.run_dirs:
        run_path = Path(run_dir)
        if not run_path.is_dir():
            print(f"[skip] {run_dir}: not a directory")
            continue
        scored_files = sorted(run_path.glob("*_scored.jsonl"))
        if not scored_files:
            print(f"[skip] {run_dir}: no _scored.jsonl found")
            continue

        scored_file = scored_files[0]
        total, exact, fuzzy = _compute_idk(scored_file)
        label = _model_label_from_path(run_path)
        results[label] = {
            "exact": exact / total * 100 if total else 0,
            "fuzzy": fuzzy / total * 100 if total else 0,
            "total": total,
        }
        print(
            f"[ok] {label}: {exact / total * 100:.1f}% exact, "
            f"{fuzzy / total * 100:.1f}% fuzzy "
            f"({exact:,} / {total:,} rollouts)"
        )

    if not results:
        print("No scored files found.")
        return

    plot_idk_rate(results, args.output, args.title)


if __name__ == "__main__":
    main()
