#!/usr/bin/env python3
"""Plot the traversal elicitation + training experiment (2026-07-08).

Two panels:
  * left: pass@k-style boundary for the BASE model under the traversal prompt
    (best@k from the n=8 elicitation run), with the standard-prompt greedy
    baseline as a reference line - the Gekhman/Zhang "scaffolding expands access"
    figure on OVEN;
  * right: the trav10 GRPO training trajectory (greedy@1 on the traversal val),
    showing traversal_parse saturated near 1 while accuracy stays flat.

Also emits a Markdown + LaTeX comparison table (standard vs traversal).

Inputs are the tidy CSVs from parse_wandb_datastore.py:
  --elicit  grpo_elicit_trav_n8_metrics.csv   (base, n=8, traversal prompt)
  --train   grpo_traversal_train_metrics.csv   (trav10 GRPO, greedy@1)

Usage::
    .venv/bin/python analysis/plot_grpo_traversal.py \
        --elicit viz/grpo/grpo_elicit_trav_n8_metrics.csv \
        --train  viz/grpo/grpo_traversal_train_metrics.csv \
        --fig viz/grpo/grpo_traversal_behavioral.png \
        --table viz/grpo/grpo_traversal_table.tex \
        --md viz/grpo/grpo_traversal_table.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import apply_thesis_style

PREFIX = "val-aux/oven_taxonomy_reasoning/"
CORE = "val-core/oven_taxonomy_reasoning/"


# Standard-prompt greedy baseline (July-7 runs, uniformly flat ~= base policy).
# Taken from viz/grpo/grpo_results_table.md (agg08 final step).
STD_BASELINE = {
    "exact_match": 0.138,
    "exact_or_fuzzy": 0.175,
    "specific_hF": 0.274,
    "linked": 0.590,
    "pred_path_depth": 2.56,
    "traversal_parse": 0.000,
}


def load_csv(path: Path):
    rows = list(csv.DictReader(path.open()))
    # metric -> step -> value
    out: dict[str, dict[int, float]] = {}
    for r in rows:
        out.setdefault(r["metric"], {})[int(r["step"])] = float(r["value"])
    return out


def elicit_val(data, metric, agg):
    """agg like 'mean@8' or 'best@8' -> the single logged value (step 1).

    best@k/worst@k are logged with a '/mean' suffix (over the val set);
    mean@k is logged bare.
    """
    for key in (f"{PREFIX}{metric}/{agg}", f"{PREFIX}{metric}/{agg}/mean"):
        d = data.get(key, {})
        if d:
            return next(iter(d.values()))
    return None


def train_series(data, metric):
    key = f"{PREFIX}{metric}/mean@1"
    d = data.get(key, {})
    steps = sorted(d)
    return steps, [d[s] for s in steps]


def make_figure(elicit, train, out_path, no_title):
    sns = apply_thesis_style()
    plot_kw = dict(lw=2.4, markersize=8.0, markeredgecolor="white",
                   markeredgewidth=1.1, solid_capstyle="round")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # --- Left: base-model pass@k boundary under traversal prompt ---
    ks = [1, 2, 4, 8]
    def boundary(metric):
        vals = []
        for k in ks:
            agg = "mean@8" if k == 1 else f"best@{k}"
            vals.append(elicit_val(elicit, metric, agg))
        return vals
    ef = boundary("exact_or_fuzzy")
    hf = boundary("specific_hF")
    axL.plot(ks, ef, marker="o", color="#0072B2", label="exact/fuzzy (traversal prompt)", **plot_kw)
    axL.plot(ks, hf, marker="s", color="#009E73", label="specific hF (traversal prompt)", **plot_kw)
    axL.axhline(STD_BASELINE["exact_or_fuzzy"], ls="--", color="#0072B2", alpha=0.7, lw=2.0,
                label="exact/fuzzy (standard prompt, greedy)")
    axL.axhline(STD_BASELINE["specific_hF"], ls="--", color="#009E73", alpha=0.7, lw=2.0,
                label="specific hF (standard prompt, greedy)")
    axL.set_xscale("log", base=2)
    axL.set_xticks(ks)
    axL.set_xticklabels([str(k) for k in ks])
    axL.set_xlabel("k (samples)", fontsize=13, labelpad=8)
    axL.set_ylabel("Score (base model, n=8)", fontsize=13, labelpad=8)
    axL.set_ylim(0, 0.55)
    axL.tick_params(labelsize=11)
    axL.grid(axis="both", alpha=0.7)
    leg = axL.legend(fontsize=9.5, loc="upper left", framealpha=0.92, fancybox=True, borderpad=0.7)
    leg.get_frame().set_edgecolor("#D8D3CA")
    if not no_title:
        axL.set_title("Traversal scaffolding expands access (base model)")

    # --- Right: trav10 GRPO training trajectory ---
    for metric, c, mk, lbl in [
        ("traversal_parse", "#D55E00", "^", "traversal emitted"),
        ("specific_hF", "#009E73", "s", "specific hF"),
        ("exact_or_fuzzy", "#0072B2", "o", "exact/fuzzy"),
    ]:
        s, v = train_series(train, metric)
        axR.plot(s, v, marker=mk, color=c, label=lbl, **plot_kw)
    axR.axhline(STD_BASELINE["exact_or_fuzzy"], ls="--", color="#0072B2", alpha=0.6, lw=2.0,
                label="exact/fuzzy (standard prompt)")
    axR.set_xlabel("Training step", fontsize=13, labelpad=8)
    axR.set_ylabel("Validation score (greedy@1, traversal prompt)", fontsize=13, labelpad=8)
    axR.set_ylim(0, 1.05)
    axR.tick_params(labelsize=11)
    axR.grid(axis="both", alpha=0.7)
    leg = axR.legend(fontsize=9.5, loc="center right", framealpha=0.92, fancybox=True, borderpad=0.7)
    leg.get_frame().set_edgecolor("#D8D3CA")
    if not no_title:
        axR.set_title("GRPO: format saturated, accuracy flat")

    sns.despine(ax=axL)
    sns.despine(ax=axR)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[fig] wrote {out_path}")


TABLE_METRICS = [
    ("exact_match", "Exact match"),
    ("exact_or_fuzzy", "Exact $\\cup$ fuzzy"),
    ("specific_hF", "Specific hF"),
    ("linked", "Linked to taxonomy"),
    ("pred_path_depth", "Pred. path depth"),
    ("traversal_parse", "Traversal emitted"),
]


def build_tables(elicit, train, tex_path, md_path):
    # Columns: Standard(greedy) | Traversal base pass@1 | pass@8 | Traversal trained(step150)
    def std(m):
        v = STD_BASELINE.get(m)
        return "--" if v is None else f"{v:.3f}"
    def base1(m):
        v = elicit_val(elicit, m, "mean@8")
        return "--" if v is None else f"{v:.3f}"
    def base8(m):
        v = elicit_val(elicit, m, "best@8")
        return "--" if v is None else f"{v:.3f}"
    def trained(m):
        s, v = train_series(train, m)
        return f"{v[-1]:.3f}" if v else "--"

    # Final training step, for the trained-column header.
    _s, _ = train_series(train, "exact_or_fuzzy")
    final = _s[-1] if _s else "?"
    cols = ["Standard (greedy)", "Traversal base (pass@1)", "Traversal base (pass@8)", f"Traversal trained (step {final})"]
    funcs = [std, base1, base8, trained]

    # LaTeX
    lines = [
        "% Auto-generated by analysis/plot_grpo_traversal.py",
        "\\begin{tabular}{l" + "r" * len(cols) + "}",
        "\\toprule",
        "Metric & " + " & ".join(cols) + " \\\\",
        "\\midrule",
    ]
    for m, disp in TABLE_METRICS:
        lines.append(disp + " & " + " & ".join(f(m) for f in funcs) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"[tex] wrote {tex_path}")

    # Markdown
    md = ["| Metric | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for m, disp in TABLE_METRICS:
        clean = disp.replace("$\\cup$", "∪")
        md.append(f"| {clean} | " + " | ".join(f(m) for f in funcs) + " |")
    md_path.write_text("\n".join(md) + "\n")
    print(f"[md] wrote {md_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--elicit", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--fig", type=Path, required=True)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--md", type=Path, required=True)
    ap.add_argument("--no-title", action="store_true")
    args = ap.parse_args()

    elicit = load_csv(args.elicit)
    train = load_csv(args.train)
    make_figure(elicit, train, args.fig, args.no_title)
    build_tables(elicit, train, args.table, args.md)


if __name__ == "__main__":
    main()
