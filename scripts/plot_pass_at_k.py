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


def collect_results(logs_root: str | Path) -> dict[str, dict[str, float]]:
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

        # Try common_results.json, then <run_id>_results*.json, then generations_results.json
        results_file = run_dir / "common_results.json"
        if not results_file.exists():
            results_files = sorted(run_dir.glob(f"{run_dir.name}_results*.json"))
            results_file = results_files[0] if results_files else run_dir / f"{run_dir.name}_results.json"
        if not results_file.exists():
            results_file = run_dir / "generations_results.json"
        if not results_file.exists():
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
# Plotting
# ---------------------------------------------------------------------------

def plot_pass_at_k(
    results: dict[str, dict[str, float]],
    output_path: str,
    title: str | None = None,
    results2: dict[str, dict[str, float]] | None = None,
    label2: str = "",
):
    """Create a pass@k comparison plot and save to *output_path*."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_colors = len(results) + (len(results2) if results2 else 0)
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("colorblind", n_colors)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, (label, pass_k) in enumerate(results.items()):
        ks = [int(k.split("@")[1]) for k in pass_k]
        values = list(pass_k.values())
        ax.plot(ks, values, "o-", label=label, color=palette[i], linewidth=1.8, markersize=4.5)

    if results2:
        offset = len(results)
        for i, (label, pass_k) in enumerate(results2.items()):
            ks = [int(k.split("@")[1]) for k in pass_k]
            values = list(pass_k.values())
            lbl = f"{label} ({label2})" if label2 else label
            ax.plot(ks, values, "s--", label=lbl, color=palette[offset + i],
                    linewidth=1.8, markersize=4.5, alpha=0.85)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("k (number of rollouts)", fontsize=11)
    ax.set_ylabel("pass@k", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(title or "pass@k by model size", fontsize=13)
    ax.legend(title="Model", fontsize=9, title_fontsize=10)
    ax.tick_params(labelsize=9)

    from matplotlib.ticker import FixedLocator
    all_ks = sorted({int(k.split("@")[1]) for pass_k in results.values() for k in pass_k})
    ax.xaxis.set_major_locator(FixedLocator(all_ks))
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())

    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
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
    parser.add_argument("--output", default="viz/pass_at_k_comparison.png",
                        help="Output image path (default: viz/pass_at_k_comparison.png)")
    parser.add_argument("--title", default=None,
                        help="Plot title (default: auto-generated)")
    parser.add_argument("--run-dirs2", default=None, nargs="+",
                        help="Second set of run dirs (dashed lines, e.g. no-image baseline).")
    parser.add_argument("--label2", default="no image",
                        help="Label suffix for second set (default: 'no image')")
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
            results_file = run_path / "common_results.json"
            if not results_file.exists():
                results_files = sorted(run_path.glob(f"{run_path.name}_results*.json"))
                results_file = results_files[0] if results_files else run_path / f"{run_path.name}_results.json"
            if not results_file.exists():
                results_file = run_path / "generations_results.json"
            if not results_file.exists():
                print(f"[skip] {run_dir}: no _results.json found")
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
        results = collect_results(args.logs_root)
    else:
        parser.error("One of --logs-root, --run-dirs, or --results-file is required")

    if not results:
        print("No results with pass@k found — has the judge pipeline run?")
        return

    results2: dict[str, dict[str, float]] = {}
    if args.run_dirs2:
        for run_dir in args.run_dirs2:
            run_path = Path(run_dir)
            if not run_path.is_dir():
                print(f"[skip] {run_dir}: not a directory")
                continue
            results_file = run_path / "common_results.json"
            if not results_file.exists():
                results_files = sorted(run_path.glob(f"{run_path.name}_results*.json"))
                results_file = results_files[0] if results_files else run_path / f"{run_path.name}_results.json"
            if not results_file.exists():
                results_file = run_path / "generations_results.json"
            if not results_file.exists():
                print(f"[skip] {run_dir}: no _results.json found")
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

    plot_pass_at_k(results, args.output, args.title, results2 if results2 else None, args.label2)


if __name__ == "__main__":
    main()
