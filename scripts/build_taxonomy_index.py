#!/usr/bin/env python3
"""Build a precomputed OVEN taxonomy index JSON.

This script requires the ``build-index`` extra (``networkx``) because it uses
vlm-eval's load_oven() to build the NetworkX tree and then serialises the
lookup tables to a lightweight JSON file.

At runtime, the inference and scoring code loads this JSON instead of building
the tree — no networkx dependency needed.

Usage::

    uv run --extra build-index python scripts/build_taxonomy_index.py \
        --output data/processed/oven_taxonomy_index.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Ensure the project is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Build OVEN taxonomy index")
    parser.add_argument("--output", default="data/processed/oven_taxonomy_index.json")
    args = parser.parse_args()

    from oven_mllm_eval.data.load_data import load_oven

    print("Loading OVEN taxonomy (this may take a minute)...")
    tree, split, aka_map, name_to_file = load_oven()

    print(f"  Tree: {tree.number_of_nodes()} nodes, {tree.number_of_edges()} edges")
    print(f"  Paths: {len(split)}")

    # Build node → path (root-to-leaf)
    import networkx as nx

    node_to_path = {}
    for node in tree.nodes():
        if node == "root":
            continue
        try:
            path = nx.shortest_path(tree, "root", node)
            # Skip the synthetic "root" node in the path
            path = path[1:]
        except nx.NetworkXNoPath:
            continue
        node_to_path[node] = path

    # Build label_to_paths: normalised label → list of root-to-leaf paths
    label_to_paths = defaultdict(list)
    for node, path in node_to_path.items():
        norm = _normalise(node)
        label_to_paths[norm].append(path)

    # Build entity_id_to_path (from chain data)
    # We re-read the chains to preserve entity_id mapping
    from oven_mllm_eval.data.load_data import out_labels, _load_jsonl
    from oven_mllm_eval.data.load_data import REMOVE_NODES_AFTER_AND_INCLUDING, REMOVE_AFTER

    chains = _load_jsonl(out_labels)
    entity_id_to_path = {}
    for entry in chains:
        eid = entry.get("id", "")
        ancestry = list(entry.get("taxonomy", []))
        ancestry.reverse()

        # Apply the same pruning as load_oven
        if set(ancestry).intersection(REMOVE_NODES_AFTER_AND_INCLUDING):
            for i, n in enumerate(ancestry):
                if n in REMOVE_NODES_AFTER_AND_INCLUDING:
                    ancestry = ancestry[i + 1 :]
                    break
        if set(ancestry).intersection(REMOVE_AFTER):
            for i in range(len(ancestry) - 1, 0, -1):
                n = ancestry[i]
                if n in REMOVE_AFTER:
                    ancestry = ancestry[i:]
                    break

        if ancestry:
            entity_id_to_path[eid] = ancestry

    # Collect all node labels (original casing)
    all_nodes = sorted(node_to_path.keys())

    # Build alias map: normalised alias → canonical node name
    aliases = {}
    for node_label, akas in aka_map.items():
        for alias in akas:
            norm_alias = _normalise(alias)
            if norm_alias not in aliases:
                aliases[norm_alias] = node_label

    index = {
        "label_to_paths": {k: v for k, v in label_to_paths.items()},
        "entity_id_to_path": entity_id_to_path,
        "node_to_path": node_to_path,
        "all_nodes": all_nodes,
        "aliases": aliases,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"Taxonomy index written to {output_path}")
    print(f"  {len(label_to_paths)} normalised labels")
    print(f"  {len(entity_id_to_path)} entity IDs")
    print(f"  {len(all_nodes)} tree nodes")
    print(f"  {len(aliases)} aliases")


def _normalise(text: str) -> str:
    import re
    text = text.lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    main()
