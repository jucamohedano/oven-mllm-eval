"""Semantic (cosine) retrieval of taxonomy nodes for prediction mapping.

Builds and caches sentence-embedding vectors for all taxonomy node labels,
then retrieves the top-k nearest nodes for a batch of predictions.  The
retrieved top-k are fed into ``TaxonomyMatcher``'s cascade (exact / n-gram /
voting) — this module only replaces the cascade's lexical Step 1 with cosine
retrieval, matching the CVPR 2025 reference (the ``cascade`` measure).

Node embeddings are computed once and cached to disk (keyed by model + the
node set), so the expensive encode runs a single time across all runs.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
# Cache location precedence: explicit arg → $OVEN_NODE_EMB_DIR → repo-local default.
# On the cluster point $OVEN_NODE_EMB_DIR at $WORK so the cache is off the (full) FAST scratch.
DEFAULT_CACHE_DIR = "data/processed/node_emb"


class EmbeddingNodeIndex:
    """Cached sentence-embedding index over taxonomy node labels."""

    def __init__(
        self,
        all_nodes: list[str],
        model_name: str = DEFAULT_MODEL,
        cache_dir: str | None = None,
        device: str = "cpu",
    ):
        self.all_nodes = list(all_nodes)
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir or os.environ.get("OVEN_NODE_EMB_DIR")
                              or DEFAULT_CACHE_DIR)
        self._model = None
        self.node_emb = self._build_or_load()

    def _cache_path(self) -> Path:
        key = hashlib.md5("\n".join(self.all_nodes).encode("utf-8")).hexdigest()[:12]
        slug = self.model_name.split("/")[-1]
        return self.cache_dir / f"{slug}_{key}.npy"

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "sentence-transformers is required for the 'cascade' measure "
                    "measure. Install it with `uv sync` (it is declared in "
                    "pyproject.toml) and download the model with "
                    f"`hf download {self.model_name}`."
                ) from e
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _build_or_load(self) -> np.ndarray:
        cp = self._cache_path()
        if cp.exists():
            try:
                return np.load(cp)
            except (OSError, ValueError) as e:  # corrupt/partial cache → recompute
                print(f"[embed] WARNING: cached {cp} unreadable ({e}); recomputing.",
                      flush=True)
        model = self._load_model()
        print(f"[embed] encoding {len(self.all_nodes):,} taxonomy nodes "
              f"with {self.model_name} (one-time, caching to {cp})", flush=True)
        emb = model.encode(
            self.all_nodes,
            batch_size=256,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        # Best-effort cache: a write failure (e.g. full/over-quota filesystem)
        # must not kill scoring.  Write atomically so a failed write never
        # leaves a corrupt cache for the next run.
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            tmp = cp.parent / (cp.name + ".tmp")
            with open(tmp, "wb") as f:
                np.save(f, emb)
            tmp.replace(cp)
        except OSError as e:
            print(f"[embed] WARNING: could not cache node embeddings to {cp} ({e}); "
                  f"continuing without the cache (set $OVEN_NODE_EMB_DIR to a writable "
                  f"location to enable caching).", flush=True)
        return emb

    def search(self, predictions: list[str], k: int = 3) -> list[tuple[list[int], list[float]]]:
        """Cosine top-k node indices + scores for each prediction."""
        from sentence_transformers.util import semantic_search

        model = self._load_model()
        pred_emb = model.encode(
            list(predictions),
            batch_size=256,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        hits = semantic_search(pred_emb, self.node_emb, top_k=k)
        out: list[tuple[list[int], list[float]]] = []
        for row in hits:
            out.append(([h["corpus_id"] for h in row], [float(h["score"]) for h in row]))
        return out


def build_prediction_mapping(
    predictions: list[str],
    index: dict,
    *,
    model_name: str = DEFAULT_MODEL,
    k: int = 3,
    min_score: float = 0.35,
    device: str = "cpu",
    cache_dir: str | None = None,
) -> dict[str, dict]:
    """Map each unique prediction → taxonomy node via cosine retrieval + cascade.

    Returns ``{prediction: match_dict}`` where ``match_dict`` has
    ``predicted_node`` / ``predicted_path`` / ``mapping_method`` (the latter is
    ``"none"`` when the cosine score is below ``min_score`` and no lexical hit
    is found).  The expensive embed + cascade runs once per *unique* prediction.
    """
    from oven_mllm_eval.matching import TaxonomyMatcher

    uniq = sorted({p for p in predictions if p})
    node_index = EmbeddingNodeIndex(index["all_nodes"], model_name, cache_dir, device)
    hits = node_index.search(uniq, k=k)
    matcher = TaxonomyMatcher(index, k=k, min_score=min_score)

    mapping: dict[str, dict] = {}
    for pred, (idxs, scores) in zip(uniq, hits):
        mapping[pred] = matcher.match_prediction(pred, top_idxs=idxs, top_scores=scores)
    return mapping
