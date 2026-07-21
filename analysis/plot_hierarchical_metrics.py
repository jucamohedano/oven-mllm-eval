#!/usr/bin/env python3
"""Plot taxonomy-aware hierarchical metrics from results JSON files.

This script reads the nested ``{"measures": [{"measure", "metrics": {...}}]}``
summaries produced by ``scripts/score_predictions.py`` / ``schedule_scoring.sh``.
It plots only taxonomy-aware metrics:

  - standard hP/hR/hF
  - specificity-penalized hP/hR/hF
  - strict leaf correctness (exact / exact_all)
  - mapped-only views
  - all-example views, where unmapped examples are zero-filled

Use ``analysis/plot_pass_at_k.py`` for pass@k plots.

Example:

    python analysis/plot_hierarchical_metrics.py \
        --results \
          2B=logs/.../2b_results_recomputed.json \
          4B=logs/.../4b_results_recomputed.json \
          8B=logs/.../8b_results_recomputed.json \
        --out-prefix viz/taxonomy/with_desc_rich_recomputed_aligned_concise_no_idk \
        --title "with-desc-rich judge · aligned · concise_no_idk"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _common import tex_escape


STANDARD_METRICS = ["hP", "hR", "hF"]
SPECIFIC_METRICS = ["specific_hP", "specific_hR", "specific_hF"]
STANDARD_ALL_METRICS = ["hP_all", "hR_all", "hF_all"]
SPECIFIC_ALL_METRICS = ["specific_hP_all", "specific_hR_all", "specific_hF_all"]
STRICT_LEAF_METRICS = ["exact"]
STRICT_LEAF_ALL_METRICS = ["exact_all"]

MODEL_COLORS = {
    "2B": "#0072B2",
    "4B": "#E69F00",
    "8B": "#009E73",
    "32B": "#D55E00",
    "Qwen3-VL 2B": "#0072B2",
    "Qwen3-VL 4B": "#E69F00",
    "Qwen3-VL 8B": "#009E73",
    "Qwen3-VL 32B": "#D55E00",
}
MEASURE_TITLES = {
    "exact_match": "ExactMatch",
    "cascade": "ComplexMatcher",
    "contained": "Contained",
    "summary": "Summary",
}


def _load_measures(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "measures" in data:
        return {entry["measure"]: entry["metrics"] for entry in data["measures"]}
    if isinstance(data, list):
        return {entry["measure"]: entry["metrics"] for entry in data}
    if isinstance(data, dict):
        return {"summary": data}
    raise ValueError(f"Unsupported results format: {path}")


def _parse_result_specs(items: list[str]) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    labels: list[str] = []
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    for item in items:
        label, sep, path = item.partition("=")
        if not sep or not label or not path:
            raise SystemExit(f"Invalid --results item: {item!r}. Expected LABEL=PATH.")
        result_path = Path(path)
        if not result_path.exists():
            raise SystemExit(f"Results file not found: {result_path}")
        labels.append(label)
        runs[label] = _load_measures(result_path)
    return labels, runs


def _available_measures(
    labels: list[str],
    runs: dict[str, dict[str, dict[str, Any]]],
    requested: list[str],
) -> list[str]:
    measures = [measure for measure in requested if all(measure in runs[label] for label in labels)]
    missing = [measure for measure in requested if measure not in measures]
    for measure in missing:
        missing_labels = [label for label in labels if measure not in runs[label]]
        print(f"[skip] measure={measure}: missing for {', '.join(missing_labels)}")
    if not measures:
        raise SystemExit("No requested measures are present in every results file.")
    return measures


def _metric_group(variant: str, view: str) -> tuple[list[str], list[str], str]:
    if variant == "standard" and view == "mapped":
        return STANDARD_METRICS, ["hP", "hR", "hF"], "standard_mapped"
    if variant == "specific" and view == "mapped":
        return SPECIFIC_METRICS, ["specific hP", "specific hR", "specific hF"], "specific_mapped"
    if variant == "standard" and view == "all":
        return STANDARD_ALL_METRICS, ["hP all", "hR all", "hF all"], "standard_all"
    if variant == "specific" and view == "all":
        return SPECIFIC_ALL_METRICS, ["specific hP all", "specific hR all", "specific hF all"], "specific_all"
    if variant == "leaf" and view == "mapped":
        return STRICT_LEAF_METRICS, ["strict leaf exact"], "leaf_mapped"
    if variant == "leaf" and view == "all":
        return STRICT_LEAF_ALL_METRICS, ["strict leaf exact all"], "leaf_all"
    raise ValueError(f"Unsupported variant/view: {variant}/{view}")


def _print_table(
    labels: list[str],
    runs: dict[str, dict[str, dict[str, Any]]],
    measures: list[str],
) -> None:
    fields = [
        "hF",
        "specific_hF",
        "hF_all",
        "specific_hF_all",
        "num_mapped",
        "num_unmapped",
        "mapping_coverage",
        "under_specific_rate",
        "mean_depth_delta",
    ]
    print("\nmodel\tmeasure\t" + "\t".join(fields))
    for label in labels:
        for measure in measures:
            metrics = runs[label][measure]
            values = []
            for field in fields:
                value = metrics.get(field)
                if isinstance(value, float):
                    values.append(f"{value:.6f}")
                else:
                    values.append("" if value is None else str(value))
            print(f"{label}\t{measure}\t" + "\t".join(values))


def _metric_value(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return "--"


def _latex_table(
    labels: list[str],
    runs: dict[str, dict[str, dict[str, Any]]],
    measures: list[str],
) -> str:
    table_measures = [measure for measure in ["exact_match", "cascade"] if measure in measures]
    table_measures += [measure for measure in measures if measure not in table_measures]
    metric_keys = STANDARD_METRICS

    colspec = "l" + "c" * (len(table_measures) * len(metric_keys))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Hierarchical taxonomy metrics for ExactMatch and ComplexMatcher.}",
        r"\label{tab:taxonomy-hierarchical-metrics}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
    ]

    group_header = [""]
    cmidrules = []
    start_col = 2
    for measure in table_measures:
        group_header.append(rf"\multicolumn{{3}}{{c}}{{{tex_escape(MEASURE_TITLES.get(measure, measure))}}}")
        end_col = start_col + len(metric_keys) - 1
        cmidrules.append(rf"\cmidrule(lr){{{start_col}-{end_col}}}")
        start_col = end_col + 1
    lines.append(" & ".join(group_header) + r" \\")
    lines.append(" ".join(cmidrules))
    lines.append("Model & " + " & ".join(metric_keys * len(table_measures)) + r" \\")
    lines.append(r"\midrule")

    for label in labels:
        values = []
        for measure in table_measures:
            metrics = runs[label][measure]
            values.extend(_metric_value(metrics, key) for key in metric_keys)
        lines.append(f"{tex_escape(label)} & " + " & ".join(values) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def _write_latex_table(
    *,
    labels: list[str],
    runs: dict[str, dict[str, dict[str, Any]]],
    measures: list[str],
    out_path: Path,
) -> None:
    latex = _latex_table(labels, runs, measures)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex)
    print("\n" + latex)
    print(f"[saved] {out_path}")


def _plot_group(
    *,
    labels: list[str],
    runs: dict[str, dict[str, dict[str, Any]]],
    measures: list[str],
    metric_keys: list[str],
    metric_titles: list[str],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
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

    fallback_palette = sns.color_palette("colorblind", len(labels))
    fig, axes = plt.subplots(1, len(measures), figsize=(5.2 * len(measures), 5.2), squeeze=False, sharey=True)
    x = np.arange(len(metric_keys))
    width = min(0.8 / max(len(labels), 1), 0.18)

    for measure_index, (ax, measure) in enumerate(zip(axes[0], measures)):
        for label_index, label in enumerate(labels):
            values = [float(runs[label][measure].get(metric_key, 0.0) or 0.0) for metric_key in metric_keys]
            offset = (label_index - (len(labels) - 1) / 2) * width
            color = MODEL_COLORS.get(label, fallback_palette[label_index])
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=label,
                color=color,
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(1.0, bar.get_height() + 0.015),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                )

        ax.set_title(MEASURE_TITLES.get(measure, measure), fontsize=15, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_titles)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("")
        ax.grid(axis="x", alpha=0.35)
        ax.grid(axis="y", alpha=0.9)
        if measure_index == 0:
            ax.set_ylabel("score", fontsize=13, labelpad=8)
        else:
            ax.set_ylabel("")
        sns.despine(ax=ax, top=True, right=True)

    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        legend_labels,
        title="Model",
        fontsize=10.5,
        title_fontsize=11.5,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        borderpad=0.7,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(labels),
    )
    legend.get_frame().set_edgecolor("#D8D3CA")

    if title:
        fig.suptitle(title, fontsize=17, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.88))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.9))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[saved] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot taxonomy-aware hP/hR/hF metrics from nested results JSON files."
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="LABEL=PATH pairs, e.g. 2B=..._results_recomputed.json",
    )
    parser.add_argument(
        "--measures",
        nargs="+",
        default=["exact_match", "cascade"],
        help="Measures to plot. Default: exact_match cascade.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["standard", "specific", "leaf"],
        default=["standard", "specific"],
        help="Metric variants to plot. Use leaf for strict leaf correctness. Default: standard specific.",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        choices=["mapped", "all"],
        default=["mapped", "all"],
        help="Denominator views to plot. Default: mapped all.",
    )
    parser.add_argument(
        "--out-prefix",
        default="viz/hierarchical_plots/hierarchical",
        help="Output prefix; writes <prefix>_taxonomy_<variant>_<view>.png.",
    )
    parser.add_argument(
        "--latex-table",
        default=None,
        help="Output path for the LaTeX table. Default: <out-prefix>_taxonomy_table.tex.",
    )
    parser.add_argument("--no-latex-table", action="store_true", help="Do not write the LaTeX table.")
    parser.add_argument("--title", default=None)
    parser.add_argument("--no-table", action="store_true", help="Do not print TSV diagnostics.")
    args = parser.parse_args()

    labels, runs = _parse_result_specs(args.results)
    measures = _available_measures(labels, runs, args.measures)

    if not args.no_table:
        _print_table(labels, runs, measures)

    out_prefix = Path(args.out_prefix)
    if not args.no_latex_table:
        latex_table_path = Path(args.latex_table) if args.latex_table else out_prefix.with_name(
            f"{out_prefix.name}_taxonomy_table.tex"
        )
        _write_latex_table(labels=labels, runs=runs, measures=measures, out_path=latex_table_path)

    title_prefix = args.title
    for variant in args.variants:
        for view in args.views:
            metric_keys, metric_titles, suffix = _metric_group(variant, view)
            out_path = out_prefix.with_name(f"{out_prefix.name}_taxonomy_{suffix}.png")
            _plot_group(
                labels=labels,
                runs=runs,
                measures=measures,
                metric_keys=metric_keys,
                metric_titles=metric_titles,
                out_path=out_path,
                title=f"{title_prefix} — taxonomy {suffix.replace('_', ' ')}" if title_prefix else "",
            )


if __name__ == "__main__":
    main()
