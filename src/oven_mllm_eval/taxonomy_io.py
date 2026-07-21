"""Loaders for entity taxonomy chains used by the judge and analysis scripts."""

from __future__ import annotations

import json
from pathlib import Path


def load_taxonomy_chains(path: str | Path) -> dict[str, list[str]]:
    """Load taxonomy chains keyed by entity QID."""
    chains: dict[str, list[str]] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("id")
            taxonomy = row.get("taxonomy", [])
            if qid and isinstance(taxonomy, list):
                chains[qid] = [str(item) for item in taxonomy]
    return chains


def load_label_chains_from_index(path: str | Path) -> dict[str, list[str]]:
    """Load entity leaf-to-root label chains from the precomputed taxonomy index."""
    index = json.loads(Path(path).read_text())
    chains = {}
    for qid, path_labels in index.get("entity_id_to_path", {}).items():
        if not isinstance(path_labels, list):
            continue
        labels = [str(label) for label in path_labels]
        if labels and labels[-1] == "root":
            labels = labels[:-1]
        chains[qid] = labels
    return chains
