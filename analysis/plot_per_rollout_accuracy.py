#!/usr/bin/env python3
"""Plot per-rollout accuracy: fraction correct at each rollout position.

Usage::

    uv run --extra analysis python scripts/plot_per_rollout_accuracy.py \
        --run-dirs <dir1> <dir2> ... \
        --common-only \
        --output viz/per_rollout_accuracy.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _model_label(slug: str) -> str:
    m = re.search(r"-(\d+b)-", slug)
    size = m.group(1).upper() if m else slug
    if "qwen3-vl" in slug:
        family = "Qwen3-VL"
    elif "qwen2-vl" in slug:
        family = "Qwen2-VL"
    else:
        family = slug.split("_")[0]
    return f"{family} {size}"


def _shard_files(run_path: Path) -> list[Path]:
    shards = sorted(
        f for f in run_path.iterdir()
        if f.name.endswith("_judged.jsonl_shard0.jsonl")
        or f.name.endswith("_judged.jsonl_shard1.jsonl")
        or f.name.endswith("_judged.jsonl_shard2.jsonl")
        or f.name.endswith("_judged.jsonl_shard3.jsonl")
    )
    if not shards:
        shards = sorted(
            f for f in run_path.iterdir()
            if f.suffix == ".jsonl" and "judged" in f.stem and "shard" not in f.stem
        )
    return shards


def _collect_ids(run_dir: str) -> set[str]:
    ids: set[str] = set()
    for sf in _shard_files(Path(run_dir)):
        with open(sf) as fh:
            for line in fh:
                row = json.loads(line.strip())
                if row.get("judge_verdicts"):
                    ids.add(row["data_id"])
    return ids


def _load_verdicts(run_dir: str, filter_ids: set[str] | None) -> tuple[str, list[list[bool]]]:
    run_path = Path(run_dir)
    label = _model_label(run_path.parent.name)
    verdicts: list[list[bool]] = []
    for sf in _shard_files(run_path):
        with open(sf) as fh:
            for line in fh:
                row = json.loads(line.strip())
                if filter_ids is not None and row.get("data_id") not in filter_ids:
                    continue
                v = row.get("judge_verdicts")
                if v:
                    verdicts.append(v)
    return label, verdicts


def main():
    parser = argparse.ArgumentParser(description="Plot per-rollout accuracy across models")
    parser.add_argument("--run-dirs", nargs="+", required=True,
                        help="Run directories to compare")
    parser.add_argument("--common-only", action="store_true",
                        help="Only use examples present in ALL runs")
    parser.add_argument("--output", default="viz/per_rollout_accuracy.png")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    # ── Compute common subset ──────────────────────────────────────
    filter_ids: set[str] | None = None
    if args.common_only:
        all_ids = [_collect_ids(d) for d in args.run_dirs]
        filter_ids = all_ids[0]
        for ids in all_ids[1:]:
            filter_ids &= ids
        print(f"Common examples: {len(filter_ids)}")

    # ── Load ───────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("colorblind", len(args.run_dirs))

    # k values: powers of 2 up to the rollout count (same grid as pass@k).
    first_verdicts = _load_verdicts(args.run_dirs[0], filter_ids)[1]
    n_max = len(first_verdicts[0]) if first_verdicts else 256
    ks = [2**i for i in range(0, 12) if 2**i <= n_max]
    ks.append(n_max)

    n_models = len(args.run_dirs)
    bar_width = 0.8 / n_models
    x = range(len(ks))

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for idx, (run_dir, color) in enumerate(zip(args.run_dirs, palette)):
        label, all_verdicts = _load_verdicts(run_dir, filter_ids)
        n_examples = len(all_verdicts)
        print(f"[ok] {label}: {n_examples} examples × {n_max} rollouts")

        acc = []
        for k in ks:
            correct = sum(v[k - 1] for v in all_verdicts)
            acc.append(correct / n_examples)

        offset = (idx - n_models / 2 + 0.5) * bar_width
        ax.bar([pos + offset for pos in x], acc, bar_width,
               color=color, label=label, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Rollout position (k)", fontsize=11)
    ax.set_ylabel("Accuracy at k-th rollout", fontsize=11)
    ax.set_ylim(0, None)
    ax.set_title(args.title or "Per-rollout accuracy at sampled positions", fontsize=13)
    ax.legend(title="Model", fontsize=9, title_fontsize=10)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
