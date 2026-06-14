#!/usr/bin/env python3
"""Build a version of the OVEN validation set with aligned questions.

OVEN's original questions were generated from source-dataset super-categories
(WordNet synsets for ImageNet21k).  The vlm-eval taxonomy we use for
hierarchical evaluation is built from Wikidata ``P279`` chains.  These are
different hierarchies, producing misaligned question–answer pairs.

This script replaces generic OVEN questions with questions generated from
the entity's Wikidata taxonomy parent.  For entities whose taxonomy parents
are all too generic, it keeps the original OVEN question if it is already
category-specific.

Algorithm
---------
1. Load all 9,459 taxonomy chains and count how many chains each node
   appears in (IDF-like).
2. Find the natural frequency gap — nodes appearing in a huge fraction of
   chains are "universal ancestors" (entity, object, taxon, …).  The gap
   is detected automatically; no hardcoded threshold.
3. For each example:
   a. If the entity has a specific parent in its chain → generate
      ``"what is this {parent}?"``.
   b. If all parents are generic but the original OVEN question is
      category-specific → keep the OVEN question.
   c. Otherwise → mark as excluded (no usable question).

Usage::

    uv run python scripts/build_aligned_questions.py \
        --input data/processed/vlm_compatible_val.jsonl \
        --chains data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
        --output data/processed/vlm_compatible_val_aligned.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


# OVEN generic templates — these ask about "object" / "content" without
# specifying a semantic category, so they misalign with the entity.
GENERIC_OVEN_TEMPLATES = {
    "what is the main object?",
    "what is shown in the photo?",
    "what is the main content of this image?",
}


def _find_generic_threshold(chains: dict[str, list[str]]) -> int:
    """Find the frequency threshold separating generic from specific nodes.

    Computes how many chains each node appears in, then finds the largest
    gap between adjacent frequencies.  The threshold is the upper bound
    of the generic region: nodes appearing *above* this threshold are
    considered universal ancestors.
    """
    node_freq: Counter[str] = Counter()
    for tax in chains.values():
        seen: set[str] = set()
        for node in tax:
            if node not in seen:
                node_freq[node] += 1
                seen.add(node)

    sorted_freqs = sorted(set(node_freq.values()), reverse=True)
    gaps = [
        (sorted_freqs[i] - sorted_freqs[i + 1], sorted_freqs[i + 1])
        for i in range(len(sorted_freqs) - 1)
    ]
    _, threshold = max(gaps)  # largest gap → natural cutoff
    return threshold


def _first_specific_parent(
    chain: list[str],
    node_freq: Counter[str],
    threshold: int,
) -> str | None:
    """Return the first chain node (closest to leaf) below the threshold."""
    for node in chain[1:]:  # skip leaf (index 0)
        if node_freq.get(node, 0) <= threshold:
            return node
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Build aligned OVEN questions from taxonomy parents"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to vlm_compatible_val.jsonl",
    )
    parser.add_argument(
        "--chains", required=True,
        help="Path to oven_wikidata_chains_cleaned_labels.jsonl",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write the aligned JSONL",
    )
    args = parser.parse_args()

    # ── Load taxonomy chains ───────────────────────────────────────
    chains: dict[str, list[str]] = {}
    with open(args.chains) as f:
        for line in f:
            r = json.loads(line.strip())
            chains[r["id"]] = r["taxonomy"]

    # ── Compute node frequencies and threshold ─────────────────────
    node_freq: Counter[str] = Counter()
    for tax in chains.values():
        seen: set[str] = set()
        for node in tax:
            if node not in seen:
                node_freq[node] += 1
                seen.add(node)

    threshold = _find_generic_threshold(chains)

    generic_nodes = sorted(
        n for n, c in node_freq.items() if c > threshold
    )
    print(
        f"Loaded {len(chains)} chains, {len(node_freq)} unique nodes.\n"
        f"Generic threshold: >{threshold} chains.\n"
        f"Generic nodes ({len(generic_nodes)}): {generic_nodes}"
    )

    # ── Process examples ───────────────────────────────────────────
    stats: dict[str, int] = Counter()

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            row = json.loads(line.strip())
            eid = row.get("entity_id", "")

            if eid not in chains:
                stats["excluded_no_chain"] += 1
                continue

            chain = chains[eid]
            parent = _first_specific_parent(chain, node_freq, threshold)

            if parent is not None:
                # Group 1: generate question from chain parent
                row["oven_question"] = row["question"]
                row["question"] = f"what is this {parent}?"
                row["question_source"] = "chain_parent"
                stats["generated_from_chain"] += 1
            elif row.get("question", "") not in GENERIC_OVEN_TEMPLATES:
                # Group 2a: keep aligned OVEN question
                row["question_source"] = "oven_original"
                stats["kept_oven_aligned"] += 1
            else:
                # Group 2b: neither source gives a usable question — skip
                stats["excluded_generic"] += 1
                continue

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── Report ─────────────────────────────────────────────────────
    total = sum(stats.values())
    usable = stats["generated_from_chain"] + stats["kept_oven_aligned"]
    print()
    print(f"Total examples:                {total:6d}")
    print(f"  Generated from chain parent: {stats['generated_from_chain']:6d} "
          f"({stats['generated_from_chain'] / total * 100:.1f}%)")
    print(f"  Kept OVEN (already aligned): {stats['kept_oven_aligned']:6d} "
          f"({stats['kept_oven_aligned'] / total * 100:.1f}%)")
    print(f"  ─────────────────────────────────────")
    print(f"  Usable:                       {usable:6d} "
          f"({usable / total * 100:.1f}%)")
    print(f"  Excluded (generic OVEN Q):    {stats['excluded_generic']:6d} "
          f"({stats['excluded_generic'] / total * 100:.1f}%)")
    print(f"  Excluded (no chain):          {stats['excluded_no_chain']:6d} "
          f"({stats['excluded_no_chain'] / total * 100:.1f}%)")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
