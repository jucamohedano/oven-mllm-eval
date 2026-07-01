# ==============================================================================
# Adapted from vlm-eval/src/vlmeval/calculate_scores/measures.py
# and vlm-eval/src/vlmeval/calculate_scores/map_predictions.py (DirectMeasureMatcher)
# ==============================================================================

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from oven_mllm_eval.scores import normalize


def _clean_prediction(text: str) -> str:
    text = text.replace("A: ", "").replace("A:", "")
    text = text.replace("<answer>", "").replace("</answer>", "")
    text = text.replace("<s>", "").replace("</s>", "")
    text = text.split("<|end_header_id|>")[-1]
    text = text.split("<|eot_id|>")[0]
    text = text.strip()
    if text and text[-1] in ".!?,;":
        text = text[:-1].strip()
    return text


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", normalize(text)).strip()


def _stem_texts(texts: list[str]) -> list[str]:
    try:
        from nltk.stem.snowball import SnowballStemmer
    except ImportError:
        return [_normalise(text) for text in texts]

    stemmer = SnowballStemmer("english")
    out = []
    for text in texts:
        text = _normalise(text)
        out.append(" ".join(stemmer.stem(token) for token in text.split()))
    return out


class ExactMatch:
    def compute(self, references: List[str] | None = None, predictions: List[str] | None = None, **kwargs) -> dict:
        references = references or []
        predictions = predictions or []
        scores = [1.0 if ref == pred else 0.0 for ref, pred in zip(references, predictions)]
        return {"score": sum(scores) / len(scores) if scores else 0.0, "scores": scores}


class Contained:
    def compute(self, references: List[str] | None = None, predictions: List[str] | None = None, **kwargs) -> dict:
        references = references or []
        predictions = predictions or []
        scores = [1.0 if ref in pred else 0.0 for ref, pred in zip(references, predictions)]
        return {"score": sum(scores) / len(scores) if scores else 0.0, "scores": scores}


class RougeScore:
    def __init__(self) -> None:
        self._metric = None

    def compute(self, references: List[str] | None = None, predictions: List[str] | None = None, **kwargs) -> dict:
        references = references or []
        predictions = predictions or []
        if self._metric is None:
            try:
                import evaluate
            except ImportError as exc:  # pragma: no cover
                raise ImportError("evaluate is required for the 'rouge' measure.") from exc
            self._metric = evaluate.load("rouge", keep_in_memory=True)
        result = self._metric.compute(
            references=references,
            predictions=predictions,
            rouge_types=["rouge1"],
            use_aggregator=False,
        )
        return {"rouge1": result["rouge1"]}


class BleuScore:
    def __init__(self) -> None:
        self._metric = None

    def compute(self, references: List[str] | None = None, predictions: List[str] | None = None, **kwargs) -> dict:
        references = references or []
        predictions = predictions or []
        if self._metric is None:
            try:
                import evaluate
            except ImportError as exc:  # pragma: no cover
                raise ImportError("evaluate is required for the 'bleu' measure.") from exc
            self._metric = evaluate.load("bleu", keep_in_memory=True)

        scores = []
        for ref, pred in zip(references, predictions):
            score = self._metric.compute(
                references=[[ref]],
                predictions=[pred],
                smooth=True,
                max_order=2,
            )["bleu"]
            scores.append(score)
        return {"bleu": scores}


class MeteorScore:
    def compute(self, references: List[str] | None = None, predictions: List[str] | None = None, **kwargs) -> dict:
        references = references or []
        predictions = predictions or []
        try:
            from nltk import word_tokenize
            from nltk.translate import meteor_score
        except ImportError as exc:  # pragma: no cover
            raise ImportError("nltk is required for the 'meteor' measure.") from exc

        scores = []
        for ref, pred in zip(references, predictions):
            scores.append(
                meteor_score.single_meteor_score(
                    word_tokenize(ref),
                    word_tokenize(pred),
                )
            )
        return {"meteor": scores}


class SentenceBertScore:
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

    def __init__(self) -> None:
        self._model = None
        self._ref_cache_key = None
        self._ref_cache_emb = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "sentence-transformers is required for the 'sentence_bert' measure."
                ) from exc
            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model

    def _cache_path(self, references: list[str]) -> Path:
        cache_dir = Path(os.environ.get("OVEN_NODE_EMB_DIR") or "data/processed/node_emb")
        refs_key = hashlib.md5("\n".join(references).encode("utf-8")).hexdigest()[:12]
        model_key = hashlib.md5(self.MODEL_NAME.encode("utf-8")).hexdigest()[:12]
        return cache_dir / f"sentence_bert_direct_{model_key}_{refs_key}.npy"

    def _encode_references(self, references: list[str]) -> np.ndarray:
        refs_key = hashlib.md5("\n".join(references).encode("utf-8")).hexdigest()
        if self._ref_cache_key == refs_key and self._ref_cache_emb is not None:
            return self._ref_cache_emb

        cp = self._cache_path(references)
        if cp.exists():
            try:
                emb = np.load(cp)
                print(
                    f"[sentence_bert] loaded cached reference embeddings from {cp} "
                    f"(shape={emb.shape})",
                    flush=True,
                )
                self._ref_cache_key = refs_key
                self._ref_cache_emb = emb
                return emb
            except (OSError, ValueError) as exc:
                print(
                    f"[sentence_bert] WARNING: cached {cp} unreadable ({exc}); recomputing.",
                    flush=True,
                )

        model = self._load_model()
        print(
            f"[sentence_bert] encoding {len(references):,} reference labels with {self.MODEL_NAME} "
            f"(one-time, caching to {cp})",
            flush=True,
        )
        emb = model.encode(references, normalize_embeddings=True, convert_to_numpy=True)
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            tmp = cp.parent / (cp.name + ".tmp")
            with open(tmp, "wb") as f:
                np.save(f, emb)
            tmp.replace(cp)
        except OSError as exc:
            print(
                f"[sentence_bert] WARNING: could not cache reference embeddings to {cp} ({exc}); "
                "continuing without the cache.",
                flush=True,
            )
        self._ref_cache_key = refs_key
        self._ref_cache_emb = emb
        return emb

    def compute(self, references: List[str] | None = None, predictions: List[str] | None = None, **kwargs) -> dict:
        references = references or []
        predictions = predictions or []
        if not references or not predictions:
            return {"scores": []}

        model = self._load_model()
        ref_emb = self._encode_references(references)
        unique_predictions = list(dict.fromkeys(predictions))
        unique_pred_emb = model.encode(
            unique_predictions,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        pred_lookup = dict(zip(unique_predictions, unique_pred_emb))
        pred_emb = np.vstack([pred_lookup[pred] for pred in predictions])
        scores = (ref_emb * pred_emb).sum(axis=1).tolist()
        return {"scores": scores}


ALL_MEASURES: Dict[str, dict] = {
    "exact_match": {
        "measure": ExactMatch(),
        "params": {"stem": True},
        "extra_params": {"specificity_keys": ["scores"]},
    },
    "contained": {
        "measure": Contained(),
        "params": {"stem": True},
        "extra_params": {"specificity_keys": ["scores"]},
    },
    "bleu": {
        "measure": BleuScore(),
        "params": {"stem": True},
        "extra_params": {"specificity_keys": ["bleu"]},
    },
    "meteor": {
        "measure": MeteorScore(),
        "params": {"stem": True},
        "extra_params": {"specificity_keys": ["meteor"]},
    },
    "rouge": {
        "measure": RougeScore(),
        "params": {"stem": True},
        "extra_params": {"specificity_keys": ["rouge1"]},
    },
    "sentence_bert": {
        "measure": SentenceBertScore(),
        "extra_params": {"specificity_keys": ["scores"]},
    },
}

for _name, _measure in ALL_MEASURES.items():
    _measure["name"] = _name

class DirectMeasureMatcher:
    def __init__(self, index: dict, measure: dict, top_k: int = 5):
        self.index = index
        self.all_nodes: list[str] = index["all_nodes"]
        self.node_to_path: dict = index.get("node_to_path", {})
        self.measure = measure
        self.top_k = top_k

    def match(self, prediction: str) -> dict | None:
        if not self.all_nodes:
            return None

        prediction = _clean_prediction(prediction)
        references = list(self.all_nodes)
        measure_params = self.measure.get("params", {}).copy()

        if measure_params.pop("stem", False):
            references = _stem_texts(references)
            prediction = _stem_texts([prediction])[0]

        measure_results = self.measure["measure"].compute(
            references=references,
            predictions=[prediction] * len(references),
            **measure_params,
        )
        score_key = self.measure["extra_params"]["specificity_keys"][0]
        all_scores = np.asarray(measure_results[score_key], dtype=float)

        actual_top_k = min(self.top_k, len(all_scores))
        top_k_idxs = np.argsort(all_scores)[-actual_top_k:][::-1]
        best_idx = int(top_k_idxs[0])
        best_node = self.all_nodes[best_idx]
        best_path = self.node_to_path.get(best_node)
        best_score = float(all_scores[best_idx])

        top_k_candidates = [
            {
                "name": self.all_nodes[int(idx)],
                "path": self.node_to_path.get(self.all_nodes[int(idx)]),
                "score": float(all_scores[int(idx)]),
            }
            for idx in top_k_idxs
        ]

        return {
            "predicted_node": best_node,
            "predicted_path": best_path,
            "mapping_method": self.measure.get("name", "direct_measure"),
            "scores": {
                "score": best_score,
                "top_k_candidates": top_k_candidates,
            },
        }

    def evaluate(self, prediction: str, reference: str, reference_path: list[str] | None = None) -> dict:
        if reference_path is None:
            reference_path = self.node_to_path.get(reference)
            if reference_path is None:
                paths = self.index.get("label_to_paths", {}).get(_normalise(reference), [])
                if paths:
                    reference_path = paths[0]

        match = self.match(prediction)
        if match is None or reference_path is None:
            return {
                "success": False,
                "predicted_node": None,
                "predicted_path": None,
                "reference_path": reference_path,
                "hP": 0.0,
                "hR": 0.0,
                "hF": 0.0,
                "specific_hP": 0.0,
                "specific_hR": 0.0,
                "specific_hF": 0.0,
                "under_specific": False,
                "over_specific": False,
                "depth_delta": None,
                "mapping_method": None,
                "scores": None,
            }

        from oven_mllm_eval.scores import calc_hierarchical_metrics, calc_specificity_hierarchical_metrics

        pred_path = match["predicted_path"]
        metrics = calc_hierarchical_metrics([(pred_path, reference_path)])
        specific_metrics = calc_specificity_hierarchical_metrics([(pred_path, reference_path)])
        exact = _normalise(match["predicted_node"] or "") == _normalise(reference)

        return {
            "success": exact,
            "predicted_node": match["predicted_node"],
            "predicted_path": pred_path,
            "reference_path": reference_path,
            "hP": metrics["hP"][0],
            "hR": metrics["hR"][0],
            "hF": metrics["hF"][0],
            "specific_hP": specific_metrics["specific_hP"][0],
            "specific_hR": specific_metrics["specific_hR"][0],
            "specific_hF": specific_metrics["specific_hF"][0],
            "under_specific": specific_metrics["under_specific"][0],
            "over_specific": specific_metrics["over_specific"][0],
            "depth_delta": specific_metrics["depth_delta"][0],
            "mapping_method": match["mapping_method"],
            "scores": match.get("scores"),
        }
