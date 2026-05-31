# ==============================================================================
# Adapted from vlm-eval/src/vlmeval/calculate_scores/measures.py
# and vlm-eval/src/vlmeval/calculate_scores/map_predictions.py (DirectMeasureMatcher)
#
# Pluggable measures for scoring predictions against taxonomy node labels.
# The DirectMeasureMatcher replaces networkx lookups with precomputed index
# lookups (same pattern as TaxonomyMatcher in matching.py).
# ==============================================================================

from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List, Optional

from oven_mllm_eval.scores import normalize


# ---------------------------------------------------------------------------
# Measure classes — pluggable scoring functions
# ---------------------------------------------------------------------------

class ExactMatch:
    """Score = 1 if prediction string-equals reference, else 0."""

    def compute(self, references: List[str] = None, predictions: List[str] = None,
                **kwargs) -> dict:
        if references is None:
            references = []
        if predictions is None:
            predictions = []
        scores = [1 if ref == pred else 0 for ref, pred in zip(references, predictions)]
        return {"score": sum(scores) / len(scores) if scores else 0.0,
                "scores": scores}


class Contained:
    """Score = 1 if reference label is a substring of prediction, else 0."""

    def compute(self, references: List[str] = None, predictions: List[str] = None,
                **kwargs) -> dict:
        if references is None:
            references = []
        if predictions is None:
            predictions = []
        scores = [1 if ref.lower() in pred.lower() else 0
                  for ref, pred in zip(references, predictions)]
        return {"score": sum(scores) / len(scores) if scores else 0.0,
                "scores": scores}


# ---------------------------------------------------------------------------
# Measure registry — mirrors vlm-eval's ALL_MEASURES structure
# ---------------------------------------------------------------------------

ALL_MEASURES: Dict[str, dict] = {
    "exact_match": {
        "measure": ExactMatch(),
        "params": {"stem": True},
        "extra_params": {
            "specificity_keys": ["scores"],
        },
    },
    "contained": {
        "measure": Contained(),
        "params": {"stem": True},
        "extra_params": {
            "specificity_keys": ["scores"],
        },
    },
}


# ---------------------------------------------------------------------------
# DirectMeasureMatcher — scores all taxonomy nodes, returns best match
# ---------------------------------------------------------------------------

class DirectMeasureMatcher:
    """Map a prediction to a taxonomy node by scoring against all node labels.

    Unlike ``TaxonomyMatcher`` (which implements the multi-stage algorithm from
    the paper), this matcher uses a single pluggable measure (e.g. exact_match,
    contained) to score the prediction against every taxonomy node and returns
    the highest-scoring node.

    Adapted from ``vlm-eval/src/vlmeval/calculate_scores/map_predictions.py``
    with networkx replaced by precomputed index lookups.
    """

    def __init__(
        self,
        index: dict,
        measure: dict,
        top_k: int = 5,
    ):
        self.index = index
        self.all_nodes: list[str] = index["all_nodes"]
        self.node_to_path: dict = index.get("node_to_path", {})
        self.measure = measure
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(self, prediction: str) -> dict | None:
        """Score prediction against all taxonomy nodes, return best match.

        Returns None only if the taxonomy has no nodes.
        """
        if not self.all_nodes:
            return None

        # Mirror vlm-eval's clean_pred() in data_for_positioning.py:
        # strip "A:", <answer>/<s> tags, and trailing punctuation
        # before scoring, so free-form model output has a chance to
        # match taxonomy labels.
        prediction = self._clean_prediction(prediction)

        measure_params = self.measure.get("params", {}).copy()
        references = list(self.all_nodes)

        # Optional stemming
        if measure_params.get("stem"):
            references, prediction = self._stem(references, prediction)
            del measure_params["stem"]

        # Score prediction against every taxonomy node
        measure_results = self.measure["measure"].compute(
            references=references,
            predictions=[prediction] * len(references),
            **measure_params,
        )

        score_key = self.measure["extra_params"]["specificity_keys"][0]
        all_scores = np.array(measure_results[score_key])

        # Top-k → best (mirrors vlm-eval DirectMeasureMatcher.match)
        actual_top_k = min(self.top_k, len(all_scores))
        top_k_idxs = np.argsort(all_scores)[-actual_top_k:][::-1]
        best_idx = top_k_idxs[0]
        best_node = self.all_nodes[best_idx]
        best_path = self.node_to_path.get(best_node)
        best_score = float(all_scores[best_idx])

        top_k_candidates = []
        for idx in top_k_idxs:
            node = self.all_nodes[idx]
            path = self.node_to_path.get(node)
            top_k_candidates.append({
                "name": node,
                "path": path,
                "score": float(all_scores[idx]),
            })

        return {
            "predicted_node": best_node,
            "predicted_path": best_path,
            "mapping_method": self.measure.get("name", "direct_measure"),
            "scores": {
                "score": best_score,
                "top_k_candidates": top_k_candidates,
            },
        }

    def evaluate(
        self,
        prediction: str,
        reference: str,
        reference_path: list[str] | None = None,
    ) -> dict:
        """Evaluate a single prediction against a reference.

        Mirrors ``TaxonomyMatcher.evaluate()`` so the two matchers are
        interchangeable in ``scoring.py``.
        """
        if reference_path is None:
            reference_path = self.node_to_path.get(reference)
            if reference_path is None:
                paths = self.index.get("label_to_paths", {}).get(
                    self._normalise(reference), []
                )
                if paths:
                    reference_path = paths[0]

        match = self.match(prediction)

        if match is None or reference_path is None:
            return {
                "success": False,
                "predicted_node": None,
                "predicted_path": None,
                "reference_path": reference_path,
                "hP": 0.0, "hR": 0.0, "hF": 0.0,
                "mapping_method": None,
                "scores": None,
            }

        from oven_mllm_eval.scores import calc_hierarchical_metrics

        pred_path = match["predicted_path"]
        metrics = calc_hierarchical_metrics([(pred_path, reference_path)])
        # Mirror vlm-eval's check_topks / data.py: normalise both sides
        # before comparing so casing and punctuation differences don't
        # cause false negatives.
        exact = self._normalise(match["predicted_node"] or "") == self._normalise(reference)

        return {
            "success": exact,
            "predicted_node": match["predicted_node"],
            "predicted_path": pred_path,
            "reference_path": reference_path,
            "hP": metrics["hP"][0],
            "hR": metrics["hR"][0],
            "hF": metrics["hF"][0],
            "mapping_method": match["mapping_method"],
            "scores": match.get("scores"),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_prediction(text: str) -> str:
        """Strip model-output formatting before scoring.

        Mirrors vlm-eval's ``clean_pred()`` in
        ``data_for_positioning.py``.
        """
        text = text.replace("A: ", "").replace("A:", "")
        text = text.replace("<answer>", "").replace("</answer>", "")
        text = text.replace("<s>", "").replace("</s>", "")
        # Llama-3 chat markers
        text = text.split("<|end_header_id|>")[-1]
        text = text.split("<|eot_id|>")[0]
        text = text.strip()
        # Strip trailing punctuation (model often appends "." or ",")
        if text and text[-1] in ".!?,;":
            text = text[:-1].strip()
        return text

    @staticmethod
    def _normalise(text: str) -> str:
        import re
        text = text.lower().replace("_", " ").replace("-", " ")
        text = re.sub(r"[^a-z0-9 ]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _stem(references: list[str], prediction: str) -> tuple[list[str], str]:
        """Normalize and stem references and prediction.

        Uses nltk SnowballStemmer if available; falls back to normalize() only.
        """
        try:
            from nltk.stem.snowball import SnowballStemmer
            stemmer = SnowballStemmer("english")

            def stem_one(seqs):
                out = []
                for seq in seqs:
                    seq = normalize(seq)
                    seq = " ".join(stemmer.stem(w) for w in seq.split())
                    out.append(seq)
                return out

            return stem_one(references), stem_one([prediction])[0]
        except ImportError:
            # nltk not installed — normalize only
            return [normalize(r) for r in references], normalize(prediction)
