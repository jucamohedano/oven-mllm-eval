#!/usr/bin/env python3
"""Plot the distribution of cᵢ (correct rollouts per example) across models.

cᵢ = number of the 256 rollouts the judge marked correct.  This is the
quantity the unbiased pass@k estimator is built on.

Outputs:
  Histogram — cᵢ | cᵢ ≥ 1, binned log-ish, as a fraction of solved examples.
  CDF       — fraction of solved examples with cᵢ ≤ t.

The script can save the legacy combined two-panel figure or split the
histogram and CDF into separate files. It can also optionally overlay the
*differential* subset: examples where the 2B model has cᵢ ≥ 1 but the 4B/8B
comparison model has cᵢ = 0.

Usage::

    uv run python scripts/plot_ci_distribution.py \
        --scored-2b logs/schedule/.../2b_run \
        --scored-4b logs/schedule/.../4b_run \
        --scored-8b logs/schedule/.../8b_scored.jsonl \
        --scored-32b logs/schedule/.../32b_scored.jsonl \
        --output-hist viz/ci_distribution_hist.png \
        --output-cdf viz/ci_distribution_cdf.png \
        --no-combined
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from oven_mllm_eval.judge_audit import build_alias_map, classify_positive, is_supported


MODEL_ORDER = ["2B", "4B", "8B", "32B"]
MODEL_COLORS = {
    "2B": "#0072B2",
    "4B": "#E69F00",
    "8B": "#009E73",
    "32B": "#D55E00",
}
MODEL_MARKERS = {
    "2B": "o",
    "4B": "s",
    "8B": "D",
    "32B": "^",
}


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


def _supported_ci(row: dict, aliases_by_canonical: dict[str, set[str]]) -> int:
    answer = row.get("answer", "")
    texts = row.get("all_texts", [])
    verdicts = row.get("judge_verdicts", [])
    return sum(
        1
        for text, verdict in zip(texts, verdicts)
        if verdict
        and is_supported(
            classify_positive(
                prediction=text,
                answer=answer,
                aliases_by_canonical=aliases_by_canonical,
            )
        )
    )


def _load_ci(
    path: str,
    aliases_by_canonical: dict[str, set[str]] | None = None,
) -> tuple[list[int], list[int], int, int, int]:
    """Return (judge_ci, supported_ci, judge_solved, supported_solved, total_examples)."""
    aliases_by_canonical = aliases_by_canonical or {}
    compute_supported = bool(aliases_by_canonical)
    ci_vals: list[int] = []
    supported_vals: list[int] = []
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            v = r.get("judge_verdicts", [])
            if v:
                ci_vals.append(sum(v))
                supported_vals.append(
                    _supported_ci(r, aliases_by_canonical) if compute_supported else 0
                )
    solved = sum(1 for c in ci_vals if c >= 1)
    supported_solved = sum(1 for c in supported_vals if c >= 1)
    return ci_vals, supported_vals, solved, supported_solved, len(ci_vals)


def _load_data_id_map(
    path: str,
    *,
    supported: bool,
    aliases_by_canonical: dict[str, set[str]],
) -> dict[str, int]:
    result: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            v = r.get("judge_verdicts", [])
            if v:
                result[r["data_id"]] = (
                    _supported_ci(r, aliases_by_canonical) if supported else sum(v)
                )
    return result


def _set_thesis_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "#fffdfa",
            "axes.facecolor": "#fffdfa",
            "axes.edgecolor": "#3b3b3b",
            "axes.linewidth": 1.6,
            "axes.labelcolor": "#2f2f2f",
            "xtick.color": "#2f2f2f",
            "ytick.color": "#2f2f2f",
            "grid.color": "#ded9ce",
            "grid.linewidth": 0.9,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "savefig.facecolor": "#fffdfa",
        }
    )


def _style_axes(ax, *, show_grid: bool = True) -> None:
    if show_grid:
        ax.grid(True, color="#ded9ce", linewidth=0.9)
    else:
        ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#3b3b3b")
    ax.spines["bottom"].set_color("#3b3b3b")
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)


def _positive_ci(models: dict[str, dict], size: str, ci_key: str) -> list[int]:
    return [c for c in models[size][ci_key] if c >= 1]


def _save_figure(fig, output: str | Path) -> None:
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path}")


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
    parser.add_argument("--scored-32b", default=None,
                        help="Path to 32B _scored.jsonl (or directory)")
    parser.add_argument(
        "--taxonomy-index",
        default="data/processed/oven_taxonomy_index.json",
        help="Taxonomy index with aliases for supported-cᵢ checks",
    )
    parser.add_argument(
        "--supported",
        action="store_true",
        help="Plot supported cᵢ instead of judge cᵢ.",
    )
    parser.add_argument("--output", default="viz/ci_distribution.png",
                        help="Combined two-panel output path.")
    parser.add_argument("--output-hist", default=None,
                        help="Optional standalone histogram output path.")
    parser.add_argument("--output-cdf", default=None,
                        help="Optional standalone CDF output path.")
    parser.add_argument("--no-combined", action="store_true",
                        help="Do not write the combined two-panel figure.")
    parser.add_argument("--include-differentials", action="store_true",
                        help="Overlay 2B-solves/4B-or-8B-misses differential CDFs.")
    parser.add_argument("--annotate-hist", action="store_true",
                        help="Annotate histogram bars with percentages.")
    parser.add_argument("--annotate-cdf-stats", action="store_true",
                        help="Add the low-c_i percentage stats box to the CDF.")
    parser.add_argument("--no-title", action="store_true",
                        help="Do not draw figure titles.")
    parser.add_argument("--no-grid", action="store_true",
                        help="Do not draw the background grid.")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    aliases_by_canonical = {}
    if args.supported:
        index = json.loads(Path(args.taxonomy_index).read_text())
        aliases_by_canonical = build_alias_map(index)
    ci_key = "supported_ci_all" if args.supported else "ci_all"
    ci_label = "supported-cᵢ" if args.supported else "judge-cᵢ"

    # Resolve paths
    models: dict[str, dict] = {}
    model_args = [
        ("2B", args.scored_2b),
        ("4B", args.scored_4b),
        ("8B", args.scored_8b),
    ]
    if args.scored_32b:
        model_args.append(("32B", args.scored_32b))

    for size, arg in model_args:
        p = Path(arg)
        if p.is_dir():
            f = _find_scored(p)
            if not f:
                print(f"Error: no scored file in {p}"); return
            p = f
        ci_all, supported_ci_all, solved, supported_solved, total = _load_ci(
            str(p),
            aliases_by_canonical,
        )
        models[size] = {
            "ci_all": ci_all,
            "supported_ci_all": supported_ci_all,
            "solved": solved,
            "supported_solved": supported_solved,
            "total": total,
            "label": _model_label_from_path(str(p)),
            "path": str(p),
        }
        active_ci = models[size][ci_key]
        active_solved = sum(1 for c in active_ci if c >= 1)
        print(f"  {models[size]['label']}: {total} examples, "
              f"solved={solved} ({solved/total*100:.1f}%), "
              f"supported_solved={supported_solved} ({supported_solved/total*100:.1f}%), "
              f"plotted_solved={active_solved} ({active_solved/total*100:.1f}%), "
              f"mean_{ci_label}={np.mean(active_ci):.1f}")

    # Differentials
    diffs: dict[str, list[int]] = {}
    if args.include_differentials:
        ci_maps = {
            size: _load_data_id_map(
                models[size]["path"],
                supported=args.supported,
                aliases_by_canonical=aliases_by_canonical,
            )
            for size in ["2B", "4B", "8B"]
            if size in models
        }

        def _compute_diff(ci_a_map, ci_b_map) -> list[int]:
            diff = []
            for did in set(ci_a_map) & set(ci_b_map):
                if ci_a_map[did] >= 1 and ci_b_map[did] == 0:
                    diff.append(ci_a_map[did])
            return diff

        if "2B" in ci_maps:
            for size in ["4B", "8B"]:
                if size in ci_maps:
                    diffs[size] = _compute_diff(ci_maps["2B"], ci_maps[size])

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

    _set_thesis_style()

    bin_edges = [1, 2, 3, 5, 9, 17, 33, 65, 129, 257]
    bin_labels = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", "129-256"]
    order = [size for size in MODEL_ORDER if size in models]

    def draw_hist(ax) -> None:
        x = np.arange(len(bin_labels))
        width = min(0.8 / max(len(order), 1), 0.22)
        all_pct_max = 0.0
        for idx, size in enumerate(order):
            ci_pos = _positive_ci(models, size, ci_key)
            n_solved = len(ci_pos)
            if not n_solved:
                continue
            pcts = [
                sum(1 for c in ci_pos if bin_edges[i] <= c < bin_edges[i + 1]) / n_solved * 100
                for i in range(len(bin_edges) - 1)
            ]
            all_pct_max = max(all_pct_max, max(pcts))
            offset = (idx - (len(order) - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                pcts,
                width,
                label=models[size]["label"],
                color=MODEL_COLORS[size],
                alpha=0.92,
                edgecolor="#fffdfa",
                linewidth=0.8,
            )
            if args.annotate_hist:
                for bar, pct in zip(bars, pcts):
                    if pct > 1.0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.35,
                            f"{pct:.0f}%",
                            ha="center",
                            va="bottom",
                            fontsize=6.5,
                            color=MODEL_COLORS[size],
                            fontweight="bold",
                        )

        ax.set_xlabel(f"{ci_label} out of 256", fontsize=13)
        ax.set_ylabel("Fraction of solved examples (%)", fontsize=13)
        if not args.no_title:
            ax.set_title(f"Histogram of {ci_label} among solved examples", fontsize=15, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=10)
        ax.set_ylim(0, max(all_pct_max * 1.18, 1.0))
        ax.legend(title="Model", fontsize=10, title_fontsize=11, frameon=True)
        _style_axes(ax, show_grid=not args.no_grid)

    def draw_cdf(ax) -> None:
        for size in order:
            ci_pos = sorted(_positive_ci(models, size, ci_key))
            if not ci_pos:
                continue
            t = np.arange(1, max(ci_pos) + 1)
            cdf = [sum(1 for c in ci_pos if c <= ti) / len(ci_pos) for ti in t]
            ax.plot(
                t,
                cdf,
                linewidth=2.4,
                marker=MODEL_MARKERS[size],
                markevery=max(len(t) // 9, 1),
                markersize=5.5,
                label=models[size]["label"],
                color=MODEL_COLORS[size],
                alpha=0.96,
            )

        diff_linestyles = {"4B": ":", "8B": "--"}
        for size, d in diffs.items():
            if not d:
                continue
            ds = sorted(d)
            t_d = np.arange(1, max(ds) + 1)
            cdf_d = [sum(1 for c in ds if c <= ti) / len(ds) for ti in t_d]
            ax.plot(
                t_d,
                cdf_d,
                linewidth=2.0,
                linestyle=diff_linestyles.get(size, "--"),
                label=f"2B wins, {size} misses (n={len(d):,})",
                color=MODEL_COLORS[size],
                alpha=0.9,
            )

        ax.set_xlabel(f"Threshold t ({ci_label} <= t)", fontsize=13)
        ax.set_ylabel("Fraction of solved examples", fontsize=13)
        if not args.no_title:
            ax.set_title(f"CDF of {ci_label} among solved examples", fontsize=15, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xlim(1, 256)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128", "256"])
        ax.legend(title="Model", fontsize=10, title_fontsize=11, frameon=True)
        _style_axes(ax, show_grid=not args.no_grid)

        if args.annotate_cdf_stats:
            stats_lines = []
            for size in order:
                ci_pos = _positive_ci(models, size, ci_key)
                if not ci_pos:
                    continue
                s1 = sum(1 for c in ci_pos if c == 1) / len(ci_pos) * 100
                s2 = sum(1 for c in ci_pos if c <= 2) / len(ci_pos) * 100
                s4 = sum(1 for c in ci_pos if c <= 4) / len(ci_pos) * 100
                stats_lines.append(
                    f"{models[size]['label']}: =1 {s1:.0f}%, <=2 {s2:.0f}%, <=4 {s4:.0f}%"
                )
            for size, d in diffs.items():
                if not d:
                    continue
                n = len(d)
                s1d = sum(1 for c in d if c == 1) / n * 100
                s2d = sum(1 for c in d if c <= 2) / n * 100
                s4d = sum(1 for c in d if c <= 4) / n * 100
                stats_lines.append(f"2B wins/{size} misses: =1 {s1d:.0f}%, <=2 {s2d:.0f}%, <=4 {s4d:.0f}%")
            ax.text(
                0.98,
                0.05,
                "\n".join(stats_lines),
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="#f3eadb", edgecolor="#d6cdbf", alpha=0.92),
            )

    if not args.no_combined:
        fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(14.5, 5.6))
        draw_hist(ax_hist)
        draw_cdf(ax_cdf)
        if not args.no_title:
            fig.suptitle(
                args.title or "c_i distribution: correct rollouts per example",
                fontsize=16,
                fontweight="bold",
            )
        fig.tight_layout()
        _save_figure(fig, args.output)
        plt.close(fig)

    if args.output_hist:
        fig, ax_hist = plt.subplots(1, 1, figsize=(10.2, 5.8))
        draw_hist(ax_hist)
        fig.tight_layout()
        _save_figure(fig, args.output_hist)
        plt.close(fig)

    if args.output_cdf:
        fig, ax_cdf = plt.subplots(1, 1, figsize=(8.8, 5.8))
        draw_cdf(ax_cdf)
        fig.tight_layout()
        _save_figure(fig, args.output_cdf)
        plt.close(fig)


if __name__ == "__main__":
    main()
