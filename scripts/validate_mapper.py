#!/usr/bin/env python3
"""Validate the taxonomy mapper against a hand-labeled set.

Reads a TSV of ``gold_node <TAB> prediction`` pairs (the CVPR reference's
``human_expert.tsv`` format), maps each prediction with the ``sentence_bert``
cosine matcher, and reports mapping accuracy (predicted_node == gold_node),
the mapping-method breakdown, and the NONE (unmapped) rate.

Use this to pick ``--map-min-score`` and confirm the mapper tracks model
quality, not mapping quality, before trusting hP/hR/hF.

Usage::

    uv run python scripts/validate_mapper.py \\
        --labeled-tsv data/processed/mapper_validation.tsv \\
        --taxonomy-index data/processed/oven_taxonomy_index.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oven_mllm_eval.taxonomy import load_taxonomy_index
from oven_mllm_eval.matching import _normalise
from oven_mllm_eval.embedding_matcher import build_prediction_mapping


def read_pairs(path: str) -> list[tuple[str, str]]:
    """Read ``gold_node <TAB> prediction`` pairs; strips wrapping <...>."""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            gold, pred = line.split("\t", 1)
            pred = pred.strip()
            if pred.startswith("<") and pred.endswith(">"):
                pred = pred[1:-1].strip()
            pairs.append((gold.strip(), pred))
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Validate taxonomy mapper on a labeled TSV")
    ap.add_argument("--labeled-tsv", required=True)
    ap.add_argument("--taxonomy-index", default=None)
    ap.add_argument("--embed-model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--map-top-k", type=int, default=3)
    ap.add_argument("--map-min-score", type=float, default=0.35)
    ap.add_argument("--embed-device", default="cpu")
    args = ap.parse_args()

    index = load_taxonomy_index(args.taxonomy_index)
    pairs = read_pairs(args.labeled_tsv)
    if not pairs:
        print("No pairs read.")
        return

    mapping = build_prediction_mapping(
        [p for _, p in pairs], index, model_name=args.embed_model,
        k=args.map_top_k, min_score=args.map_min_score, device=args.embed_device,
    )

    correct = 0
    none = 0
    methods: dict = {}
    for gold, pred in pairs:
        m = mapping.get(pred) or {}
        node = m.get("predicted_node")
        method = m.get("mapping_method")
        methods[method] = methods.get(method, 0) + 1
        if node is None:
            none += 1
        elif _normalise(node) == _normalise(gold):
            correct += 1

    n = len(pairs)
    print(f"pairs:            {n}")
    print(f"mapping accuracy: {correct}/{n} = {correct / n:.1%}")
    print(f"NONE rate:        {none}/{n} = {none / n:.1%}")
    print("method breakdown:")
    for meth, c in sorted(methods.items(), key=lambda x: -x[1]):
        print(f"   {str(meth):24} {c:5}  ({c / n:.1%})")


if __name__ == "__main__":
    main()
