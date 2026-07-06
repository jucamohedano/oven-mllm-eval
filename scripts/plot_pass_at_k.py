#!/usr/bin/env python3
"""Plot pass@k curves comparing models from evaluation results.

Walks a logs directory, picks the most recent run per model, extracts
pass@k metrics, and produces a comparison plot.

Usage::

    uv run python scripts/plot_pass_at_k.py \
        --logs-root logs/schedule/oven_naive-sampling_concise \
        --output viz/pass_at_k_comparison.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MODEL_ORDER = ["Qwen3-VL 2B", "Qwen3-VL 4B", "Qwen3-VL 8B", "Qwen3-VL 32B"]
MODEL_COLORS = {
    "Qwen3-VL 2B": "#0072B2",   # Okabe-Ito blue
    "Qwen3-VL 4B": "#E69F00",   # orange
    "Qwen3-VL 8B": "#009E73",   # green
    "Qwen3-VL 32B": "#D55E00",  # vermillion
}
MODEL_MARKERS = {
    "Qwen3-VL 2B": "o",
    "Qwen3-VL 4B": "s",
    "Qwen3-VL 8B": "D",
    "Qwen3-VL 32B": "^",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_label(slug: str) -> str:
    """Derive a display label from a model directory slug.

    ``qwen_qwen3-vl-4b-instruct`` → ``Qwen3-VL 4B``
    ``qwen_qwen3-vl-8b-instruct`` → ``Qwen3-VL 8B``
    """
    # Extract size: look for pattern like "-2b-" or "-4b-"
    m = re.search(r"-(\d+b)-", slug)
    size = m.group(1).upper() if m else slug
    # Extract family
    if "qwen3-vl" in slug:
        family = "Qwen3-VL"
    elif "qwen2-vl" in slug:
        family = "Qwen2-VL"
    elif "internvl3" in slug:
        family = "InternVL3"
    else:
        family = slug.split("_")[0]
    return f"{family} {size}"


def _extract_metrics(results_json: dict | list) -> dict[str, float]:
    """Extract a flat metrics dict from a results.json file.

    Handles both single-measure (flat dict) and multi-measure
    (list of {measure, metrics}) formats.
    """
    if isinstance(results_json, dict) and "measures" in results_json:
        results_json = results_json["measures"]
    if isinstance(results_json, list):
        for entry in results_json:
            if entry.get("measure") == "exact_match":
                return entry.get("metrics", {})
        # Fallback: first measure
        return results_json[0].get("metrics", {}) if results_json else {}
    return results_json


def _find_latest_run(model_dir: Path) -> Path | None:
    """Return the most recent timestamped run directory under *model_dir*."""
    run_dirs = sorted(
        [d for d in model_dir.iterdir() if d.is_dir() and re.match(r"^\d{8}_\d{6}_\d{6}$", d.name)],
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


def _select_results_file(run_path: Path, results_pattern: str | None = None) -> Path | None:
    """Select the summary JSON to plot from a run directory."""
    if results_pattern:
        matches = sorted(run_path.glob(results_pattern))
        return matches[0] if matches else None

    results_file = run_path / "common_results.json"
    if not results_file.exists():
        results_files = sorted(run_path.glob(f"{run_path.name}_results*.json"))
        results_file = results_files[0] if results_files else run_path / f"{run_path.name}_results.json"
    if not results_file.exists():
        results_file = run_path / "generations_results.json"
    return results_file if results_file.exists() else None


def collect_results(logs_root: str | Path, results_pattern: str | None = None) -> dict[str, dict[str, float]]:
    """Walk *logs_root* and collect pass@k metrics per model.

    Returns
    -------
    dict mapping model label → {k: pass@k_value, ...} sorted by k.
    Only includes runs that have pass@k data (judge pipeline completed).
    """
    root = Path(logs_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Logs root not found: {root}")

    results: dict[str, dict[str, float]] = {}

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue

        run_dir = _find_latest_run(model_dir)
        if run_dir is None:
            print(f"[skip] {model_dir.name}: no run directories found")
            continue

        results_file = _select_results_file(run_dir, results_pattern)
        if results_file is None:
            print(f"[skip] {model_dir.name}: no _results.json in {run_dir.name}")
            continue

        with open(results_file) as f:
            data = json.load(f)

        metrics = _extract_metrics(data)
        pass_k = {k: v for k, v in metrics.items()
                   if k.startswith("pass@") and "_majority" not in k}
        if not pass_k:
            print(f"[skip] {model_dir.name}: no pass@k in results (judge not run yet?)")
            continue

        # Sort by k value: pass@1, pass@2, pass@4, ...
        pass_k = dict(sorted(pass_k.items(), key=lambda kv: int(kv[0].split("@")[1].split("_")[0])))
        label = _model_label(model_dir.name)
        results[label] = pass_k
        print(f"[ok] {label}: {len(pass_k)} pass@k values from {run_dir.name}")

    return results


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def _ordered_model_labels(
    results: dict[str, dict[str, float]],
    results2: dict[str, dict[str, float]] | None = None,
) -> list[str]:
    ordered_labels = [label for label in MODEL_ORDER if label in results or (results2 and label in results2)]
    ordered_labels += [label for label in results if label not in ordered_labels]
    if results2:
        ordered_labels += [label for label in results2 if label not in ordered_labels]
    return ordered_labels


def _pass_k_index(key: str) -> int:
    return int(key.split("@")[1].split("_")[0])


def _pass_k_keys(
    results: dict[str, dict[str, float]],
    results2: dict[str, dict[str, float]] | None = None,
) -> list[str]:
    keys = {key for pass_k in results.values() for key in pass_k if key.startswith("pass@")}
    if results2:
        keys.update(key for pass_k in results2.values() for key in pass_k if key.startswith("pass@"))
    return sorted(keys, key=_pass_k_index)


def _tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _format_pass_value(pass_k: dict[str, float] | None, key: str) -> str:
    if not pass_k or key not in pass_k:
        return "--"
    return f"{float(pass_k[key]):.3f}"


def _default_table_label(output_path: Path) -> str:
    slug = output_path.stem.replace("_", "-")
    return f"tab:{slug}"


def _sanitize_table_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9:.-]+", "-", label.strip())


def _latex_pass_at_k_table(
    *,
    results: dict[str, dict[str, float]],
    results2: dict[str, dict[str, float]] | None,
    label1: str,
    label2: str,
    caption: str | None,
    table_label: str,
    include_comment: bool = True,
) -> str:
    ordered_labels = _ordered_model_labels(results, results2)
    keys = _pass_k_keys(results, results2)
    method_label2 = label2 or "secondary"
    has_secondary = bool(results2)

    if caption is None:
        if has_secondary:
            caption = f"pass@k for {label1} and {method_label2}."
        else:
            caption = f"pass@k for {label1}."

    colspec = "ll" + "c" * len(keys) if has_secondary else "l" + "c" * len(keys)
    header = ["Model"]
    if has_secondary:
        header.append("Method")
    header.extend(f"@{_pass_k_index(key)}" for key in keys)

    lines = []
    if include_comment:
        lines.append(r"% Requires \usepackage{booktabs}")
    lines.extend(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{3.5pt}",
            rf"\caption{{{_tex_escape(caption)}}}",
            rf"\label{{{table_label}}}",
            rf"\begin{{tabular}}{{{colspec}}}",
            r"\toprule",
            " & ".join(header) + r" \\",
            r"\midrule",
        ]
    )

    for label in ordered_labels:
        if has_secondary:
            primary_values = [_format_pass_value(results.get(label), key) for key in keys]
            lines.append(
                f"{_tex_escape(label)} & {_tex_escape(label1)} & " + " & ".join(primary_values) + r" \\"
            )
            secondary_values = [_format_pass_value(results2.get(label) if results2 else None, key) for key in keys]
            lines.append(
                f" & {_tex_escape(method_label2)} & " + " & ".join(secondary_values) + r" \\"
            )
        else:
            values = [_format_pass_value(results.get(label), key) for key in keys]
            lines.append(f"{_tex_escape(label)} & " + " & ".join(values) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def write_latex_table(
    *,
    results: dict[str, dict[str, float]],
    output_path: str | Path,
    results2: dict[str, dict[str, float]] | None = None,
    label1: str = "standard",
    label2: str = "",
    caption: str | None = None,
    table_label: str | None = None,
    append: bool = False,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table_label = _sanitize_table_label(table_label or _default_table_label(output))
    include_comment = not append or not output.exists()
    latex = _latex_pass_at_k_table(
        results=results,
        results2=results2,
        label1=label1,
        label2=label2,
        caption=caption,
        table_label=table_label,
        include_comment=include_comment,
    )
    mode = "a" if append else "w"
    with output.open(mode) as f:
        if append and output.exists() and output.stat().st_size > 0:
            f.write("\n")
        f.write(latex)
    print(f"Saved LaTeX table: {output}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pass_at_k(
    results: dict[str, dict[str, float]],
    output_path: str,
    title: str | None = None,
    results2: dict[str, dict[str, float]] | None = None,
    label1: str = "standard",
    label2: str = "",
    annotate_endpoints: bool = True,
    show_title: bool = True,
    x_label: str = "Number of sampled rollouts (k)",
):
    """Create a pass@k comparison plot and save to *output_path*."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D

    ordered_labels = _ordered_model_labels(results, results2)

    n_colors = len(ordered_labels)
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
    fallback_palette = sns.color_palette("colorblind", n_colors)

    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    endpoint_offsets: dict[tuple[str, str], float] = {}
    if annotate_endpoints:
        endpoints = []
        for label in ordered_labels:
            pass_k = results.get(label)
            if pass_k:
                endpoints.append((("primary", label), int(list(pass_k.keys())[-1].split("@")[1]), list(pass_k.values())[-1]))
            pass_k2 = results2.get(label) if results2 else None
            if pass_k2:
                endpoints.append((("secondary", label), int(list(pass_k2.keys())[-1].split("@")[1]), list(pass_k2.values())[-1]))
        endpoints = sorted(endpoints, key=lambda item: (item[1], item[2]))
        clusters: list[list[tuple[tuple[str, str], int, float]]] = []
        for item in endpoints:
            if clusters and item[1] == clusters[-1][-1][1] and item[2] - clusters[-1][-1][2] <= 0.035:
                clusters[-1].append(item)
            else:
                clusters.append([item])
        for cluster in clusters:
            if len(cluster) == 1:
                endpoint_offsets[cluster[0][0]] = 0.0
                continue
            step = 11.0
            start = -step * (len(cluster) - 1) / 2.0
            for idx, (key, _k, _value) in enumerate(cluster):
                endpoint_offsets[key] = start + idx * step

    for i, label in enumerate(ordered_labels):
        if label not in results:
            continue
        pass_k = results[label]
        ks = [int(k.split("@")[1]) for k in pass_k]
        values = list(pass_k.values())
        color = MODEL_COLORS.get(label, fallback_palette[i])
        marker = MODEL_MARKERS.get(label, "o")
        ax.plot(
            ks,
            values,
            marker=marker,
            linestyle="-",
            label=label,
            color=color,
            linewidth=2.6,
            markersize=7.0,
            markeredgecolor="white",
            markeredgewidth=1.2,
            solid_capstyle="round",
        )
        if annotate_endpoints and ks:
            ax.annotate(
                f"{values[-1]:.2f}",
                xy=(ks[-1], values[-1]),
                xytext=(8, endpoint_offsets.get(("primary", label), 0.0)),
                textcoords="offset points",
                va="center",
                fontsize=10,
                color=color,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "#FBFAF7", "edgecolor": "none", "alpha": 0.82},
            )

    if results2:
        for i, label in enumerate(ordered_labels):
            if label not in results2:
                continue
            pass_k = results2[label]
            ks = [int(k.split("@")[1]) for k in pass_k]
            values = list(pass_k.values())
            color = MODEL_COLORS.get(label, fallback_palette[i])
            marker = MODEL_MARKERS.get(label, "o")
            ax.plot(
                ks,
                values,
                marker=marker,
                linestyle="--",
                color=color,
                linewidth=2.1,
                markersize=6.0,
                markeredgecolor="white",
                markeredgewidth=1.0,
                markerfacecolor=color,
                alpha=1.0,
                dash_capstyle="round",
            )
            if annotate_endpoints and ks:
                ax.annotate(
                    f"{values[-1]:.2f}",
                    xy=(ks[-1], values[-1]),
                    xytext=(8, endpoint_offsets.get(("secondary", label), 0.0)),
                    textcoords="offset points",
                    va="center",
                    fontsize=10,
                    color=color,
                    fontweight="bold",
                    alpha=0.9,
                    bbox={"boxstyle": "round,pad=0.12", "facecolor": "#FBFAF7", "edgecolor": "none", "alpha": 0.82},
                )

    ax.set_xscale("log", base=2)
    ax.set_xlabel(x_label, fontsize=13, labelpad=8)
    ax.set_ylabel("pass@k", fontsize=13, labelpad=8)
    ax.set_ylim(0, 1.02)
    if show_title:
        ax.set_title(title or "pass@k by model size", fontsize=17, pad=14, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.grid(axis="x", alpha=0.55)
    ax.grid(axis="y", alpha=0.9)
    sns.despine(ax=ax, top=True, right=True)

    from matplotlib.ticker import FixedLocator
    all_ks = {_pass_k_index(k) for pass_k in results.values() for k in pass_k}
    if results2:
        all_ks.update(_pass_k_index(k) for pass_k in results2.values() for k in pass_k)
    all_ks = sorted(all_ks)
    ax.xaxis.set_major_locator(FixedLocator(all_ks))
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.set_xticks(all_ks)
    ax.set_xticklabels([str(k) for k in all_ks])

    model_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS.get(label, fallback_palette[i]),
            marker=MODEL_MARKERS.get(label, "o"),
            linestyle="-",
            linewidth=2.6,
            markersize=7.0,
            markeredgecolor="white",
            markeredgewidth=1.2,
            label=label,
        )
        for i, label in enumerate(ordered_labels)
    ]
    model_legend = ax.legend(
        handles=model_handles,
        title="Model",
        fontsize=10.5,
        title_fontsize=11.5,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        borderpad=0.7,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    model_legend.get_frame().set_edgecolor("#D8D3CA")
    ax.add_artist(model_legend)

    if results2:
        method_handles = [
            Line2D([0], [0], color="#3A3A3A", linestyle="-", linewidth=2.6, label=label1),
            Line2D([0], [0], color="#3A3A3A", linestyle="--", linewidth=2.1, label=label2 or "secondary"),
        ]
        method_legend = ax.legend(
            handles=method_handles,
            title="Method",
            fontsize=10.5,
            title_fontsize=11.5,
            frameon=True,
            fancybox=True,
            framealpha=0.92,
            borderpad=0.7,
            loc="upper left",
            bbox_to_anchor=(1.01, 0.58),
        )
        method_legend.get_frame().set_edgecolor("#D8D3CA")

    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot pass@k comparison across models")
    parser.add_argument("--logs-root", default=None,
                        help="Path to experiment directory (e.g. logs/schedule/oven_naive-sampling_concise). "
                             "Auto-selects the latest run per model.")
    parser.add_argument("--run-dirs", default=None, nargs="+",
                        help="Specific run directories to plot (space-separated). "
                             "Overrides --logs-root for precise control.")
    parser.add_argument("--results-file", default=None,
                        help="Path to a single _results.json (for testing one run)")
    parser.add_argument("--results-pattern", default=None,
                        help="Glob pattern used inside each run dir to select the summary JSON "
                             "(e.g. '*google_gemma-4-e4b-it_with_desc_rich.json').")
    parser.add_argument("--results-pattern2", default=None,
                        help="Glob pattern used inside each --run-dirs2 directory. "
                             "Defaults to --results-pattern.")
    parser.add_argument("--output", default="viz/pass_at_k_comparison.png",
                        help="Output image path (default: viz/pass_at_k_comparison.png)")
    parser.add_argument("--title", default=None,
                        help="Plot title (default: auto-generated)")
    parser.add_argument("--run-dirs2", default=None, nargs="+",
                        help="Second set of run dirs (dashed lines, e.g. no-image baseline).")
    parser.add_argument("--label1", default="standard",
                        help="Method label for the first set (default: 'standard')")
    parser.add_argument("--label2", default="no image",
                        help="Method label for the second set (default: 'no image')")
    parser.add_argument("--x-label", default=None,
                        help="Custom x-axis label.")
    parser.add_argument("--latex-table", default=None,
                        help="Optional output path for a LaTeX pass@k table using the plotted results.")
    parser.add_argument("--latex-caption", default=None,
                        help="Caption for --latex-table. Defaults to a generic caption.")
    parser.add_argument("--latex-label", default=None,
                        help="LaTeX label for --latex-table. Defaults to tab:<output-stem>.")
    parser.add_argument("--append-latex-table", action="store_true",
                        help="Append the table to --latex-table instead of overwriting it.")
    parser.add_argument("--no-annotate-endpoints", action="store_true",
                        help="Do not label the final pass@k value at the end of each curve.")
    parser.add_argument("--no-title", action="store_true",
                        help="Do not draw a title inside the plot.")
    args = parser.parse_args()

    if args.results_file:
        # Single-file mode
        with open(args.results_file) as f:
            data = json.load(f)
        metrics = _extract_metrics(data)
        pass_k = {k: v for k, v in metrics.items()
                   if k.startswith("pass@") and "_majority" not in k}
        if not pass_k:
            print(f"No pass@k found in {args.results_file}")
            return
        pass_k = dict(sorted(pass_k.items(), key=lambda kv: int(kv[0].split("@")[1].split("_")[0])))
        label = Path(args.results_file).parent.parent.name
        results = {_model_label(label): pass_k}
        print(f"[ok] {_model_label(label)}: {len(pass_k)} pass@k values")
    elif args.run_dirs:
        # Explicit run directories — verify all use the same input data
        results = {}
        inputs_seen: list[tuple[str, str]] = []  # (label, input_file)
        for run_dir in args.run_dirs:
            run_path = Path(run_dir)
            if not run_path.is_dir():
                print(f"[skip] {run_dir}: not a directory")
                continue
            # Read metadata to check input file
            meta_files = sorted(run_path.glob("*_metadata.json"))
            meta_files = [m for m in meta_files if not m.name.endswith("_shard0_metadata.json")
                          and not m.name.endswith("_shard1_metadata.json")
                          and not m.name.endswith("_shard2_metadata.json")
                          and not m.name.endswith("_shard3_metadata.json")]
            input_file = None
            if meta_files:
                with open(meta_files[0]) as f:
                    meta = json.load(f)
                input_file = meta.get("data", {}).get("input", "unknown")
            results_file = _select_results_file(run_path, args.results_pattern)
            if results_file is None:
                print(f"[skip] {run_dir}: no matching results JSON found")
                continue
            with open(results_file) as f:
                data = json.load(f)
            metrics = _extract_metrics(data)
            pass_k = {k: v for k, v in metrics.items()
                       if k.startswith("pass@") and "_majority" not in k}
            if not pass_k:
                print(f"[skip] {run_dir}: no pass@k in results")
                continue
            pass_k = dict(sorted(pass_k.items(), key=lambda kv: int(kv[0].split("@")[1].split("_")[0])))
            label = _model_label(run_path.parent.name)
            results[label] = pass_k
            inputs_seen.append((label, input_file or "unknown"))
            print(f"[ok] {label}: {len(pass_k)} pass@k values from {run_path.name}"
                  f"  (input: {input_file or '?'})")

        # Assert same input file
        unique_inputs = set(inp for _, inp in inputs_seen)
        if len(unique_inputs) > 1:
            print(f"[ERROR] Runs use different input files:")
            for label, inp in inputs_seen:
                print(f"  {label}: {inp}")
            return
        if unique_inputs:
            input_tag = next(iter(unique_inputs))
            # Derive a short label from the input path
            if "aligned" in input_tag:
                input_note = "aligned questions"
            elif "vlm_compatible_val.jsonl" in input_tag:
                input_note = "original OVEN questions"
            else:
                input_note = Path(input_tag).stem
            if not args.title:
                args.title = f"pass@k — {input_note} (naive-sampling, 256 rollouts)"
    elif args.logs_root:
        results = collect_results(args.logs_root, args.results_pattern)
    else:
        parser.error("One of --logs-root, --run-dirs, or --results-file is required")

    if not results:
        print("No results with pass@k found — has the judge pipeline run?")
        return

    results2: dict[str, dict[str, float]] = {}
    if args.run_dirs2:
        results_pattern2 = args.results_pattern2 or args.results_pattern
        for run_dir in args.run_dirs2:
            run_path = Path(run_dir)
            if not run_path.is_dir():
                print(f"[skip] {run_dir}: not a directory")
                continue
            results_file = _select_results_file(run_path, results_pattern2)
            if results_file is None:
                print(f"[skip] {run_dir}: no matching results JSON found")
                continue
            with open(results_file) as f:
                data = json.load(f)
            metrics = _extract_metrics(data)
            pass_k = {k: v for k, v in metrics.items()
                       if k.startswith("pass@") and "_majority" not in k}
            if not pass_k:
                continue
            pass_k = dict(sorted(pass_k.items(), key=lambda kv: int(kv[0].split("@")[1].split("_")[0])))
            label = _model_label(run_path.parent.name)
            results2[label] = pass_k
            print(f"[ok] {label} ({args.label2}): {len(pass_k)} pass@k values from {run_path.name}")

    if args.latex_table:
        write_latex_table(
            results=results,
            output_path=args.latex_table,
            results2=results2 if results2 else None,
            label1=args.label1,
            label2=args.label2,
            caption=args.latex_caption,
            table_label=args.latex_label,
            append=args.append_latex_table,
        )

    plot_pass_at_k(
        results,
        args.output,
        args.title,
        results2 if results2 else None,
        args.label1,
        args.label2,
        annotate_endpoints=not args.no_annotate_endpoints,
        show_title=not args.no_title,
        x_label=args.x_label or ("Number of evaluated candidate answers (k)" if results2 else "Number of sampled rollouts (k)"),
    )


if __name__ == "__main__":
    main()
