"""Semantic retrieval of taxonomy nodes for cascade mapping."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

DEFAULT_BACKEND = "open_clip"
DEFAULT_MODEL = "hf-hub:apple/DFN5B-CLIP-ViT-H-14"
DEFAULT_CACHE_DIR = "data/processed/node_emb"


class EmbeddingNodeIndex:
    def __init__(
        self,
        all_nodes: list[str],
        *,
        backend: str = DEFAULT_BACKEND,
        model_name: str = DEFAULT_MODEL,
        cache_dir: str | None = None,
        device: str = "cpu",
    ):
        self.all_nodes = list(all_nodes)
        self.backend = backend
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir or os.environ.get("OVEN_NODE_EMB_DIR") or DEFAULT_CACHE_DIR)
        self._model = None
        self._tokenizer = None
        self.node_emb = self._build_or_load()

    def _cache_path(self) -> Path:
        key = hashlib.md5("\n".join(self.all_nodes).encode("utf-8")).hexdigest()[:12]
        backend_slug = self.backend.replace("-", "_")
        model_slug = hashlib.md5(self.model_name.encode("utf-8")).hexdigest()[:12]
        return self.cache_dir / f"{backend_slug}_{model_slug}_{key}.npy"

    def _load_model(self):
        if self._model is not None:
            return self._model, self._tokenizer

        if self.backend == "sentence_transformer":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "sentence-transformers is required for backend='sentence_transformer'."
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._tokenizer = None
            return self._model, self._tokenizer

        if self.backend == "open_clip":
            try:
                import open_clip
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "open_clip_torch is required for backend='open_clip'."
                ) from exc
            self._model, _ = open_clip.create_model_from_pretrained(self.model_name, device=self.device)
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            return self._model, self._tokenizer

        raise ValueError(f"Unknown embedding backend '{self.backend}'.")

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        model, tokenizer = self._load_model()
        if self.backend == "sentence_transformer":
            return model.encode(
                texts,
                batch_size=256,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            )

        import torch

        outputs = []
        batch_size = 256
        total = len(texts)
        num_batches = (total + batch_size - 1) // batch_size if total else 0
        report_every = max(1, num_batches // 10) if num_batches else 1
        print(
            f"[embed] encoding {total:,} texts with {self.backend}:{self.model_name} "
            f"on {self.device} (batch_size={batch_size})",
            flush=True,
        )
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            with torch.no_grad():
                tokens = tokenizer(batch).to(self.device)
                feats = model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                outputs.append(feats.cpu().numpy())
            batch_idx = start // batch_size + 1
            if batch_idx % report_every == 0 or batch_idx == num_batches:
                print(
                    f"[embed] encoded batch {batch_idx}/{num_batches} "
                    f"({min(start + batch_size, total):,}/{total:,} texts)",
                    flush=True,
                )
        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 1), dtype=np.float32)

    def _build_or_load(self) -> np.ndarray:
        cp = self._cache_path()
        if cp.exists():
            try:
                emb = np.load(cp)
                print(
                    f"[embed] loaded cached taxonomy node embeddings from {cp} "
                    f"(shape={emb.shape})",
                    flush=True,
                )
                return emb
            except (OSError, ValueError) as exc:
                print(f"[embed] WARNING: cached {cp} unreadable ({exc}); recomputing.", flush=True)

        print(
            f"[embed] encoding {len(self.all_nodes):,} taxonomy nodes with {self.backend}:{self.model_name} "
            f"(one-time, caching to {cp})",
            flush=True,
        )
        emb = self._encode_texts(self.all_nodes)
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            tmp = cp.parent / (cp.name + ".tmp")
            with open(tmp, "wb") as f:
                np.save(f, emb)
            tmp.replace(cp)
        except OSError as exc:
            print(
                f"[embed] WARNING: could not cache node embeddings to {cp} ({exc}); continuing without the cache.",
                flush=True,
            )
        return emb

    def search(self, predictions: list[str], k: int = 10) -> list[tuple[list[int], list[float]]]:
        print(
            f"[embed] searching {len(predictions):,} unique predictions against "
            f"{len(self.all_nodes):,} taxonomy nodes (top_k={k})",
            flush=True,
        )
        pred_emb = self._encode_texts(list(predictions))
        print("[embed] computing prediction-node similarity matrix", flush=True)
        scores = pred_emb @ self.node_emb.T
        print("[embed] extracting top-k taxonomy candidates", flush=True)
        out: list[tuple[list[int], list[float]]] = []
        total = len(scores)
        report_every = max(1, min(10000, total // 10)) if total else 1
        for row in scores:
            top_k = min(k, len(row))
            candidate_idxs = np.argpartition(row, -top_k)[-top_k:]
            idxs = candidate_idxs[np.argsort(row[candidate_idxs])[::-1]]
            out.append((idxs.tolist(), [float(row[idx]) for idx in idxs]))
            if len(out) % report_every == 0 or len(out) == total:
                print(f"[embed] top-k extracted for {len(out):,}/{total:,} predictions", flush=True)
        return out


def build_prediction_mapping(
    predictions: list[str],
    index: dict,
    *,
    backend: str = DEFAULT_BACKEND,
    model_name: str = DEFAULT_MODEL,
    k: int = 10,
    device: str = "cpu",
    cache_dir: str | None = None,
) -> dict[str, dict]:
    from oven_mllm_eval.matching import TaxonomyMatcher

    uniq = sorted({prediction for prediction in predictions if prediction})
    node_index = EmbeddingNodeIndex(
        index["all_nodes"],
        backend=backend,
        model_name=model_name,
        cache_dir=cache_dir,
        device=device,
    )
    hits = node_index.search(uniq, k=k)
    matcher = TaxonomyMatcher(index, k=k)

    mapping: dict[str, dict] = {}
    print(f"[cascade] mapping {len(uniq):,} unique predictions with taxonomy cascade", flush=True)
    report_every = max(1, min(10000, len(uniq) // 10)) if uniq else 1
    for pred, (idxs, row_scores) in zip(uniq, hits):
        mapping[pred] = matcher.match_prediction(pred, top_idxs=idxs, top_scores=row_scores)
        if len(mapping) % report_every == 0 or len(mapping) == len(uniq):
            print(f"[cascade] mapped {len(mapping):,}/{len(uniq):,} unique predictions", flush=True)
    return mapping
