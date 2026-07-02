#!/usr/bin/env python3
"""Generate a balanced subset of the OVEN training parquet for GRPO training.

Samples one row per entity, stratified by taxonomy root, to produce a compact
dataset that covers the full taxonomy diversity with minimal row count.

Usage:
    python scripts/generate_train_subset_balanced.py \\
        --input data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/train.parquet \\
        --output data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/train_2k_balanced.parquet \\
        --target 2000 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a balanced training subset parquet.")
    parser.add_argument("--input", required=True, help="Path to train.parquet")
    parser.add_argument("--output", required=True, help="Path for output subset parquet")
    parser.add_argument(
        "--target", type=int, default=2000,
        help="Target number of rows (default: 2000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--rows-per-entity", type=int, default=1,
        help="Rows to sample per selected entity (default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    tbl = pq.read_table(args.input)
    extras = tbl["extra_info"].to_pylist()

    # Group rows by entity_id
    by_entity: dict[str, list[int]] = defaultdict(list)
    for i, ei in enumerate(extras):
        qid = ei.get("entity_id", "")
        if qid:
            by_entity[qid].append(i)

    if not by_entity:
        raise SystemExit("No entity_id rows found in input parquet.")

    # Group entities by taxonomy root (last element of taxonomy_labels)
    root_entities: dict[str, list[str]] = defaultdict(list)
    for qid, indices in by_entity.items():
        labels = extras[indices[0]].get("taxonomy_labels", [])
        root = labels[-1] if labels else "__no_taxonomy__"
        root_entities[root].append(qid)

    print(f"Input: {len(tbl)} rows, {len(by_entity)} unique QIDs, {len(root_entities)} taxonomy roots")

    # Stratified sampling
    selected_qids: set[str] = set()

    # Phase 1: take up to 1 entity per root (ensures full taxonomy coverage)
    for root, qids in sorted(root_entities.items()):
        if qids:
            selected_qids.add(random.choice(qids))

    print(f"Phase 1 (1 per root): {len(selected_qids)} entities")

    # Phase 2: fill remaining slots, weighted by root frequency
    remaining = args.target - len(selected_qids)
    if remaining > 0:
        pool = [(qid, root) for root, qids in root_entities.items()
                for qid in qids if qid not in selected_qids]
        if len(pool) <= remaining:
            selected_qids.update(qid for qid, _ in pool)
        else:
            # Weighted sampling: entities from larger roots get proportionally more slots
            root_weights = {root: len(qids) for root, qids in root_entities.items()}
            weighted_pool = []
            for qid, root in pool:
                weighted_pool.extend([qid] * root_weights.get(root, 1))
            while len(selected_qids) < args.target:
                candidate = random.choice(weighted_pool)
                if candidate not in selected_qids:
                    selected_qids.add(candidate)
                    # Remove all copies of this qid from weighted_pool to avoid bias
                    weighted_pool = [q for q in weighted_pool if q != candidate]

    print(f"Phase 2 (filled): {len(selected_qids)} entities")

    # Sample rows per selected entity
    row_indices: list[int] = []
    for qid in sorted(selected_qids):
        indices = by_entity[qid]
        k = min(args.rows_per_entity, len(indices))
        row_indices.extend(random.sample(indices, k))

    row_indices.sort()

    # Build and write subset
    subset = tbl.take(row_indices)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(subset, args.output)

    # Coverage report
    roots_in_subset: set[str] = set()
    prompt_types = defaultdict(int)
    for i in row_indices:
        labels = extras[i].get("taxonomy_labels", [])
        if labels:
            roots_in_subset.add(labels[-1])
        prompt_types[extras[i].get("prompt_type", "?")] += 1

    print(f"\nOutput: {len(subset)} rows, {len(selected_qids)} unique QIDs")
    print(f"Taxonomy roots covered: {len(roots_in_subset)} / {len(root_entities)}")
    print(f"Prompt types: {dict(prompt_types)}")
    print(f"Written to: {args.output}")


if __name__ == "__main__":
    main()
