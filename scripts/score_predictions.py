#!/usr/bin/env python3
"""Score a generation JSONL file with taxonomy-aware metrics.

Computes hP/hR/hF and exact-match rate using DirectMeasureMatcher
(pluggable measure, mirrors vlm-eval's process_predictions_with_strategy).

By default, scoring merges the metrics into the samples file in-place
and writes ``<run_id>_results.json`` alongside it (following lmms-ocw
convention).  Pass ``--output`` and/or ``--summary`` to override.

Usage::

    # Exact match
    uv run python scripts/score_predictions.py \\
        --input results/my_experiment/samples.jsonl \\
        --taxonomy-index data/processed/oven_taxonomy_index.json \\
        --measure exact_match
        --output results/my_experiment/samples_scored.jsonl \\

    # Contained
    uv run python scripts/score_predictions.py \\
        --input results/my_experiment/samples.jsonl \\
        --taxonomy-index data/processed/oven_taxonomy_index.json \\
        --measure contained
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oven_mllm_eval.scoring import score_generation_file


def main():
    parser = argparse.ArgumentParser(description="Score generation JSONL")
    parser.add_argument("--input", required=True, help="Input samples JSONL")
    parser.add_argument("--taxonomy-index", default=None,
                        help="Path to taxonomy index JSON")
    parser.add_argument("--output", default=None,
                        help="Per-example scored JSONL output (default: overwrite input)")
    parser.add_argument("--summary", default=None,
                        help="Aggregate results JSON output (default: <run_id>_results.json)")
    parser.add_argument("--measure", default="exact_match", nargs="+",
                        help="Measure(s) from ALL_MEASURES, space-separated. "
                             "Use 'all' for every registered measure. "
                             "(default: exact_match)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of worker processes for parallel scoring. "
                             "0 = auto (all available CPUs via sched_getaffinity). "
                             "Each worker loads its own copy of the taxonomy index.")
    args = parser.parse_args()

    summary = score_generation_file(
        input_path=args.input,
        taxonomy_index_path=args.taxonomy_index,
        output_path=args.output,
        summary_path=args.summary,
        measure=args.measure,
        num_workers=args.num_workers,
    )

    print("Summary:")
    if isinstance(summary, list):
        for entry in summary:
            print(f"  [{entry['measure']}]")
            for k, v in entry["metrics"].items():
                print(f"    {k}: {v}")
    else:
        for k, v in summary.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
