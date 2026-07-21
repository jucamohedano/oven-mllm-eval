#!/usr/bin/env python3
"""Plot GRPO taxonomy-reasoning validation metrics and emit the results table.

Reads the tidy CSV produced by ``verl/scripts/parse_wandb_datastore.py`` (columns:
run, step, metric, value) and produces:

  * a behavioural figure (left: accuracy/specific-hF trajectories showing flatness;
    right: under- vs over-specific error asymmetry at the final logged step),
  * a LaTeX results table and a Markdown mirror.

All inputs come from the offline wandb ``.wandb`` datastore, so the pipeline is
fully reproducible for additional runs (e.g. trav08 once it finishes) by
re-parsing and re-running this script.

Usage::

    .venv/bin/python analysis/plot_grpo_training.py \
        --csv viz/grpo/grpo_val_metrics.csv \
        --fig viz/grpo/grpo_behavioral.png \
        --table viz/grpo/grpo_results_table.tex \
        --md viz/grpo/grpo_results_table.md
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def apply_thesis_style():
    """Match the thesis plot aesthetic (see analysis/plot_pass_at_k.py)."""
    import seaborn as sns

    sns.set_theme(
        context="talk",
        style="whitegrid",
        rc={
            "axes.facecolor": "#FBFAF7",
            "figure.facecolor": "white",
            "axes.edgecolor": "#3A3A3A",
            "axes.labelcolor": "#202020",
            "text.color": "#202020",
            "grid.color": "#D8D3CA",
            "grid.linewidth": 0.9,
            "font.family": "DejaVu Serif",
        },
    )
    return sns


# Okabe-Ito palette, consistent with the other viz scripts in this repo.
RUN_ORDER = ["agg05", "agg08", "trav08"]
RUN_LABELS = {
    "agg05": "Agg 0.5",
    "agg08": "Agg 0.8",
    "trav08": "Trav 0.8",
}
RUN_COLORS = {
    "agg05": "#0072B2",   # blue
    "agg08": "#E69F00",   # orange
    "trav08": "#009E73",  # green
}
RUN_MARKERS = {"agg05": "o", "agg08": "s", "trav08": "D"}

METRIC = "val-aux/oven_taxonomy_reasoning/{}/mean@1"
REWARD = "val-core/oven_taxonomy_reasoning/reward/mean@1"


def short_run(name: str) -> str:
    for tag in RUN_ORDER:
        if tag in name:
            return tag
    return name


def load(csv_path: Path):
    # data[run][metric][step] = value
    data: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            run = short_run(row["run"])
            step = int(row["step"])
            data[run][row["metric"]][step] = float(row["value"])
    return data


def series(data, run, key):
    d = data[run].get(key, {})
    steps = sorted(d)
    return steps, [d[s] for s in steps]


def final_step(data, run):
    steps = set()
    for m in data[run].values():
        steps.update(m)
    return max(steps) if steps else None


def at_final(data, run, metric_key):
    d = data[run].get(metric_key, {})
    if not d:
        return None
    return d[max(d)]


def make_figure(data, out_path: Path, no_title: bool):
    sns = apply_thesis_style()
    runs = [r for r in RUN_ORDER if r in data]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # --- Left: trajectories (flatness) ---
    for run in runs:
        c = RUN_COLORS[run]
        mk = RUN_MARKERS[run]
        s_ef, v_ef = series(data, run, METRIC.format("exact_or_fuzzy"))
        s_hf, v_hf = series(data, run, METRIC.format("specific_hF"))
        axL.plot(s_ef, v_ef, marker=mk, color=c, lw=2.4, markersize=8.0,
                 markeredgecolor="white", markeredgewidth=1.1, solid_capstyle="round",
                 label=f"{RUN_LABELS[run]} — exact/fuzzy")
        axL.plot(s_hf, v_hf, marker=mk, color=c, lw=1.6, ls="--", alpha=0.75,
                 markersize=6.5, markeredgecolor="white", markeredgewidth=0.9,
                 label=f"{RUN_LABELS[run]} — specific hF")
    axL.set_xlabel("Training step", fontsize=13, labelpad=8)
    axL.set_ylabel("Validation score (mean@1, greedy)", fontsize=13, labelpad=8)
    axL.set_ylim(0, 0.40)
    axL.tick_params(labelsize=11)
    axL.grid(axis="both", alpha=0.7)
    leg = axL.legend(fontsize=9, ncol=1, loc="upper right", framealpha=0.92, fancybox=True, borderpad=0.7)
    leg.get_frame().set_edgecolor("#D8D3CA")
    if not no_title:
        axL.set_title("Validation accuracy and hierarchical F1 are flat")

    # --- Right: under- vs over-specific error asymmetry at final step ---
    metrics = [
        ("under_specific", "Under-specific"),
        ("over_specific", "Over-specific"),
        ("traversal_parse", "Traversal emitted"),
    ]
    x = range(len(metrics))
    width = 0.25
    for i, run in enumerate(runs):
        vals = [at_final(data, run, METRIC.format(k)) or 0.0 for k, _ in metrics]
        offset = (i - (len(runs) - 1) / 2) * width
        axR.bar([xi + offset for xi in x], vals, width, color=RUN_COLORS[run],
                edgecolor="white", linewidth=0.8, label=RUN_LABELS[run])
    axR.set_xticks(list(x))
    axR.set_xticklabels([lbl for _, lbl in metrics], fontsize=11)
    axR.set_ylabel("Fraction of validation set (final step)", fontsize=13, labelpad=8)
    axR.tick_params(labelsize=11)
    axR.grid(axis="y", alpha=0.7)
    leg = axR.legend(fontsize=9.5, loc="upper right", framealpha=0.92, fancybox=True, borderpad=0.7)
    leg.get_frame().set_edgecolor("#D8D3CA")
    if not no_title:
        axR.set_title("Errors are under-specific; no traversal on standard prompt")

    sns.despine(ax=axL)
    sns.despine(ax=axR, left=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[fig] wrote {out_path}")


# Rows of the results table: (metric_key, display, is_reward)
TABLE_ROWS = [
    ("exact_match", "Exact match", False),
    ("exact_or_fuzzy", "Exact $\\cup$ fuzzy", False),
    ("specific_hF", "Specific hF", False),
    ("raw_hF", "Raw hF", False),
    ("linked", "Linked to taxonomy", False),
    ("under_specific", "Under-specific", False),
    ("over_specific", "Over-specific", False),
    ("depth_delta", "Depth $\\Delta$ (pred$-$GT)", False),
    ("traversal_parse", "Traversal emitted", False),
    ("path_match", "Path match", False),
]


def build_tables(data, tex_path: Path, md_path: Path):
    runs = [r for r in RUN_ORDER if r in data]
    fsteps = {r: final_step(data, r) for r in runs}

    def cell(run, key):
        v = at_final(data, run, METRIC.format(key))
        return "--" if v is None else f"{v:.3f}"

    # reward row
    def reward_cell(run):
        d = data[run].get(REWARD, {})
        return "--" if not d else f"{d[max(d)]:.3f}"

    # ---- LaTeX ----
    col = "l" + "r" * len(runs)
    lines = [
        "% Auto-generated by analysis/plot_grpo_training.py — do not edit by hand.",
        "\\begin{tabular}{" + col + "}",
        "\\toprule",
        "Metric & " + " & ".join(RUN_LABELS[r] for r in runs) + " \\\\",
        "\\midrule",
    ]
    for key, disp, _ in TABLE_ROWS:
        lines.append(disp + " & " + " & ".join(cell(r, key) for r in runs) + " \\\\")
    lines.append("\\midrule")
    lines.append("Shaped reward & " + " & ".join(reward_cell(r) for r in runs) + " \\\\")
    lines.append("\\midrule")
    lines.append("Final step & " + " & ".join(str(fsteps[r]) for r in runs) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"[tex] wrote {tex_path}")

    # ---- Markdown ----
    md = ["| Metric | " + " | ".join(RUN_LABELS[r] for r in runs) + " |",
          "|" + "---|" * (len(runs) + 1)]
    for key, disp, _ in TABLE_ROWS:
        clean = disp.replace("$\\cup$", "∪").replace("$-$", "−").replace("$\\Delta$", "Δ")
        md.append(f"| {clean} | " + " | ".join(cell(r, key) for r in runs) + " |")
    md.append("| **Shaped reward** | " + " | ".join(reward_cell(r) for r in runs) + " |")
    md.append("| _Final step_ | " + " | ".join(str(fsteps[r]) for r in runs) + " |")
    md_path.write_text("\n".join(md) + "\n")
    print(f"[md] wrote {md_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--fig", type=Path, required=True)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--md", type=Path, required=True)
    ap.add_argument("--no-title", action="store_true")
    args = ap.parse_args()

    data = load(args.csv)
    make_figure(data, args.fig, args.no_title)
    build_tables(data, args.table, args.md)


if __name__ == "__main__":
    main()
