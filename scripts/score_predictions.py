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

from oven_mllm_eval.scoring import aggregate_scored_file, score_generation_file


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
    parser.add_argument("--embed-model",
                        default="sentence-transformers/all-mpnet-base-v2",
                        help="Sentence-embedding model for the 'cascade' measure's top-k retrieval.")
    parser.add_argument("--map-top-k", type=int, default=3,
                        help="Top-k taxonomy nodes retrieved by cosine (default: 3).")
    parser.add_argument("--map-min-score", type=float, default=0.35,
                        help="Cosine NONE-floor: below this with no lexical hit → "
                             "unmapped (default: 0.35).")
    parser.add_argument("--embed-device", default="cpu",
                        help="Device for embedding ('cpu' or 'cuda'). Default: cpu.")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Score only the first N examples (quick test).")
    parser.add_argument("--aggregate", "--agregate", action="store_true",
                        help="Only aggregate metrics from an already-scored JSONL. "
                             "Does not load the taxonomy index, recompute matches, "
                             "or write per-example scored rows.")
    args = parser.parse_args()

    if args.aggregate:
        summary = aggregate_scored_file(
            input_path=args.input,
            summary_path=args.summary,
            measure=args.measure,
        )
    else:
        summary = score_generation_file(
            input_path=args.input,
            taxonomy_index_path=args.taxonomy_index,
            output_path=args.output,
            summary_path=args.summary,
            measure=args.measure,
            num_workers=args.num_workers,
            embed_model=args.embed_model,
            map_top_k=args.map_top_k,
            map_min_score=args.map_min_score,
            embed_device=args.embed_device,
            max_examples=args.max_examples,
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
