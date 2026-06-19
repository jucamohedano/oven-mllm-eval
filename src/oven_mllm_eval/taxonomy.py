"""OVEN taxonomy index loader.

Loads the precomputed taxonomy index (built by scripts/build_taxonomy_index.py).
No networkx or datasets dependency at runtime — just JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def load_taxonomy_index(path: Optional[str | Path] = None) -> dict:
    """Load the precomputed OVEN taxonomy index.

    Parameters
    ----------
    path : str or Path, optional
        Path to the taxonomy index JSON.  Defaults to the processed data dir.

    Returns
    -------
    dict
        With keys: label_to_paths, entity_id_to_path, node_to_path,
        all_nodes, aliases.
    """
    if path is None:
        from oven_mllm_eval.paths import OVEN_TAXONOMY_INDEX
        path = Path(OVEN_TAXONOMY_INDEX)
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Taxonomy index not found at {path}. "
            "Run `uv run --extra build-index python scripts/build_taxonomy_index.py` first."
        )

    with open(path, "r") as f:
        return json.load(f)
