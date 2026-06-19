#!/usr/bin/env python3
"""Scan logs/schedule/ for completed runs and write a flat TSV catalog.

Usage::

    uv run python scripts/catalog_runs.py                     # write logs/schedule/runs.tsv
    uv run python scripts/catalog_runs.py --print             # print to stdout
    uv run python scripts/catalog_runs.py --dir logs/schedule/oven_naive-sampling_concise  # filter
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _model_slug(model: str) -> str:
    m = re.search(r"(\d+)b", model, re.IGNORECASE)
    return f"{m.group(1)}b" if m else model


def _detect_dataset(input_path: str) -> str:
    if "aligned" in input_path:
        return "aligned"
    return "original"


def _has_files(run_dir: Path, pattern: str) -> int:
    return len(list(run_dir.glob(pattern)))


def _find_judge_model(run_dir: Path) -> str:
    """Extract judge model from judge shard metadata files."""
    for mf in sorted(run_dir.glob("*_judged*_shard*_metadata.json")):
        try:
            d = json.loads(mf.read_text())
            jm = d.get("judge_model", "")
            if jm:
                return jm
        except (json.JSONDecodeError, KeyError):
            continue
    return ""


def _scan_run(run_dir: Path, logs_dir: Path) -> dict | None:
    """Read inference metadata and detect output files for one run directory."""
    # Find inference metadata (has 'model' and 'prompt_variant', but NOT 'judge_model')
    meta_files = sorted(run_dir.glob("*_metadata.json"))
    inference_meta = None
    for mf in meta_files:
        try:
            d = json.loads(mf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        # Inference metadata has 'model' + 'prompt_variant' + 'data'
        if "model" in d and "prompt_variant" in d and "data" in d:
            inference_meta = d
            break

    if not inference_meta:
        return None

    model = inference_meta.get("model", "")
    prompt = inference_meta.get("prompt_variant", "")
    data = inference_meta.get("data", {})
    dataset = _detect_dataset(data.get("input", ""))
    no_image = data.get("no_image", False)
    sharding = inference_meta.get("sharding", {})
    num_shards = sharding.get("num_shards", 1)
    n_per_shard = data.get("num_examples", 0)
    total_examples = num_shards * n_per_shard

    # Detect output files
    scored = _has_files(run_dir, "*_scored.jsonl")
    # Count judged outputs: files matching *_judged*.jsonl that are NOT shard files
    all_judged = list(run_dir.glob("*_judged*.jsonl"))
    judged = sum(1 for f in all_judged if "_shard" not in f.name)
    results = _has_files(run_dir, "*_results*.json")
    judge_model = _find_judge_model(run_dir)

    # Status
    if results or scored:
        status = "complete"
    elif judged:
        status = "judged"
    elif any("_shard" not in f.name for f in run_dir.glob("*_samples.jsonl")) \
            or _has_files(run_dir, "*_samples_shard*.jsonl"):
        status = "samples"
    else:
        status = "empty"

    # Make run_dir relative to repo root (logs_dir.parent)
    logs_abs = logs_dir.resolve()
    repo_root = logs_abs.parent
    try:
        run_dir_rel = str(run_dir.relative_to(repo_root))
    except ValueError:
        run_dir_rel = str(run_dir)  # fallback to absolute
    return {
        "run_dir": run_dir_rel,
        "dataset": dataset,
        "prompt": prompt,
        "no_image": str(no_image).lower(),
        "model": _model_slug(model),
        "model_full": model,
        "total_examples": total_examples,
        "status": status,
        "scored": scored,
        "judged": judged,
        "results": results,
        "judge_model": _model_slug(judge_model) if judge_model else "",
    }


def catalog(logs_dir: Path, filter_dir: str | None = None) -> list[dict]:
    rows = []
    pattern = f"{filter_dir}/**/*_metadata.json" if filter_dir else "**/*_metadata.json"
    seen_dirs: set[str] = set()

    for mf in sorted(logs_dir.glob(pattern)):
        run_dir = str(mf.parent)
        if run_dir in seen_dirs:
            continue
        seen_dirs.add(run_dir)
        row = _scan_run(mf.parent, logs_dir)
        if row:
            rows.append(row)

    # Second pass: timestamped dirs without metadata (crashed before writing anything)
    for run_dir in sorted(logs_dir.glob("*/*/20*")):
        rd = str(run_dir)
        if rd not in seen_dirs and run_dir.is_dir():
            seen_dirs.add(rd)
            # Try to infer model from parent dir name
            model = _model_slug(run_dir.parent.name)
            logs_abs = logs_dir.resolve()
            repo_root = logs_abs.parent
            try:
                run_dir_rel = str(run_dir.relative_to(repo_root))
            except ValueError:
                run_dir_rel = str(run_dir)
            rows.append({
                "run_dir": run_dir_rel,
                "dataset": "",
                "prompt": "",
                "no_image": "false",
                "model": model,
                "model_full": run_dir.parent.name,
                "total_examples": 0,
                "status": "crashed",
                "scored": 0,
                "judged": 0,
                "results": 0,
                "judge_model": "",
            })

    return rows


COLUMNS = ["run_dir", "dataset", "prompt", "no_image", "model", "total_examples",
           "status", "scored", "judged", "results", "judge_model", "model_full"]


def write_tsv(rows: list[dict], output: Path) -> None:
    with open(output, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(c, "")) for c in COLUMNS) + "\n")
    print(f"Wrote {len(rows)} runs to {output}")


def print_table(rows: list[dict]) -> None:
    # Pretty-print with aligned columns
    cols = ["model", "dataset", "prompt", "no_image", "total_examples", "status",
            "judge_model", "run_dir"]
    widths = {c: len(c) for c in cols}
    for r in rows:
        for c in cols:
            widths[c] = max(widths[c], len(str(r.get(c, ""))))
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main():
    parser = argparse.ArgumentParser(description="Catalog experiment runs")
    parser.add_argument("--logs-dir", default="logs/schedule",
                        help="Root logs directory (default: logs/schedule)")
    parser.add_argument("--dir", default=None,
                        help="Filter to a specific subdirectory")
    parser.add_argument("--output", default=None,
                        help="Output TSV path (default: LOGS_DIR/runs.tsv)")
    parser.add_argument("--print", action="store_true",
                        help="Print table to stdout instead of writing TSV")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.is_dir():
        print(f"Error: {logs_dir} not found", file=sys.stderr)
        sys.exit(1)

    rows = catalog(logs_dir, args.dir)

    if not rows:
        print("No runs found.")
        return

    if args.print:
        print_table(rows)
    else:
        output = Path(args.output) if args.output else logs_dir / "runs.tsv"
        write_tsv(rows, output)


if __name__ == "__main__":
    main()
