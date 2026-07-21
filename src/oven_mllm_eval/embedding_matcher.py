"""Semantic retrieval of taxonomy nodes for cascade mapping."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterator

import numpy as np

DEFAULT_BACKEND = "open_clip"
DEFAULT_MODEL = "hf-hub:apple/DFN5B-CLIP-ViT-H-14"
DEFAULT_CACHE_DIR = "data/processed/node_emb"
DEFAULT_SEARCH_CHUNK_SIZE = 4096


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

    def _encode_texts(
        self,
        texts: list[str],
        *,
        label: str = "texts",
        quiet: bool = False,
    ) -> np.ndarray:
        model, tokenizer = self._load_model()
        if self.backend == "sentence_transformer":
            if not quiet:
                print(
                    f"[embed] encoding {len(texts):,} {label} with "
                    f"{self.backend}:{self.model_name} on {self.device}",
                    flush=True,
                )
            return model.encode(
                texts,
                batch_size=256,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=not quiet,
            )

        import torch

        outputs = []
        batch_size = 256
        total = len(texts)
        num_batches = (total + batch_size - 1) // batch_size if total else 0
        report_every = max(1, num_batches // 10) if num_batches else 1
        if not quiet:
            print(
                f"[embed] encoding {total:,} {label} with {self.backend}:{self.model_name} "
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
            if not quiet and (batch_idx % report_every == 0 or batch_idx == num_batches):
                print(
                    f"[embed] encoded batch {batch_idx}/{num_batches} "
                    f"({min(start + batch_size, total):,}/{total:,} {label})",
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
        emb = self._encode_texts(self.all_nodes, label="taxonomy nodes")
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

    def _search_chunk_size(self) -> int:
        raw = os.environ.get("OVEN_EMBED_SEARCH_CHUNK_SIZE")
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                print(
                    f"[embed] WARNING: invalid OVEN_EMBED_SEARCH_CHUNK_SIZE={raw!r}; "
                    f"using {DEFAULT_SEARCH_CHUNK_SIZE}",
                    flush=True,
                )
        return DEFAULT_SEARCH_CHUNK_SIZE

    @staticmethod
    def _topk_from_scores(scores: np.ndarray, k: int) -> Iterator[tuple[list[int], list[float]]]:
        for row in scores:
            top_k = min(k, len(row))
            if top_k <= 0:
                yield [], []
                continue
            candidate_idxs = np.argpartition(row, -top_k)[-top_k:]
            idxs = candidate_idxs[np.argsort(row[candidate_idxs])[::-1]]
            yield idxs.tolist(), [float(row[idx]) for idx in idxs]

    def iter_search(
        self,
        predictions: list[str],
        k: int = 10,
        *,
        chunk_size: int | None = None,
    ) -> Iterator[tuple[str, list[int], list[float]]]:
        """Yield top-k node hits without materializing the full score matrix."""
        total = len(predictions)
        chunk_size = chunk_size or self._search_chunk_size()
        print(
            f"[embed] searching {total:,} unique predictions against "
            f"{len(self.all_nodes):,} taxonomy nodes (top_k={k}, search_chunk_size={chunk_size:,})",
            flush=True,
        )
        if not predictions:
            return

        node_emb_t = self.node_emb.T
        num_chunks = (total + chunk_size - 1) // chunk_size
        report_every = max(1, total // 10)
        processed = 0
        next_report = report_every
        for chunk_idx, chunk_start in enumerate(range(0, total, chunk_size), start=1):
            chunk = predictions[chunk_start:chunk_start + chunk_size]
            chunk_end = chunk_start + len(chunk)
            score_block_mib = (
                len(chunk) * len(self.all_nodes) * np.dtype(np.float32).itemsize / (1024 ** 2)
            )
            print(
                f"[embed] chunk {chunk_idx:,}/{num_chunks:,}: predictions "
                f"{chunk_start + 1:,}-{chunk_end:,} "
                f"(score_block~{score_block_mib:.1f} MiB)",
                flush=True,
            )
            pred_emb = self._encode_texts(chunk, label="prediction chunk", quiet=True)
            scores = pred_emb @ node_emb_t
            for pred, (idxs, row_scores) in zip(chunk, self._topk_from_scores(scores, k)):
                yield pred, idxs, row_scores

            processed += len(chunk)
            print(
                f"[embed] chunk {chunk_idx:,}/{num_chunks:,} done; "
                f"searched top-k for {processed:,}/{total:,} predictions",
                flush=True,
            )
            if processed >= next_report or processed == total:
                print(
                    f"[embed] searched top-k for {processed:,}/{total:,} predictions",
                    flush=True,
                )
                while next_report <= processed:
                    next_report += report_every

    def search(self, predictions: list[str], k: int = 10) -> list[tuple[list[int], list[float]]]:
        out: list[tuple[list[int], list[float]]] = []
        for _pred, idxs, row_scores in self.iter_search(predictions, k=k):
            out.append((idxs, row_scores))
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
    matcher = TaxonomyMatcher(index, k=k)

    mapping: dict[str, dict] = {}
    print(f"[cascade] mapping {len(uniq):,} unique predictions with taxonomy cascade", flush=True)
    report_every = max(1, min(10000, len(uniq) // 10)) if uniq else 1
    for pred, idxs, row_scores in node_index.iter_search(uniq, k=k):
        mapping[pred] = matcher.match_prediction(pred, top_idxs=idxs, top_scores=row_scores)
        if len(mapping) % report_every == 0 or len(mapping) == len(uniq):
            print(f"[cascade] mapped {len(mapping):,}/{len(uniq):,} unique predictions", flush=True)
    return mapping
