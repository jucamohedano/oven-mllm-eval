"""Prediction → taxonomy node mapping.

Implements the multi-stage matching algorithm from:
    Snæbjarnarson et al., "Hierarchy-Aware Evaluation of Free-Form
    Predictions From Vision-And-Language Models", CVPR 2024.

The mapping proceeds in stages (Algorithm 1 in the paper):
    1. Score all nodes against the prediction via a pluggable similarity function
    2. Contains check in top-k — most specific node whose label appears verbatim
    3. Contains check in all nodes
    4. N-gram overlap check — for n in (4, 3, 2), top-k first, then all nodes
    5. Voting fallback — common ancestor if top-k scores are ambiguous
    6. Top-score fallback — always returns a node

All graph operations are computed from the precomputed taxonomy index:
    tree.nodes()        -> index["all_nodes"]
    nx.shortest_path()  -> index["node_to_path"][v]          (root->leaf)
    anc(v)              -> index["node_to_path"][v][::-1]     (leaf->root)
    specificity         -> len(index["node_to_path"][v])      (path length)
    v.parent            -> index["node_to_path"][v][-2]       (penultimate)
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Callable, Optional

from oven_mllm_eval.scores import calc_hierarchical_metrics


# ---------------------------------------------------------------------------
# Similarity functions — pluggable, no CLIP / no heavy deps
# ---------------------------------------------------------------------------

def ngram_overlap_similarity(pred: str, label: str, n: int = 2) -> float:
    """Jaccard similarity of n-gram sets between prediction and label.

    Provides a continuous score (unlike binary contained/exact) so the
    top-k ordering is meaningful even without CLIP.
    """
    pred_ngrams = _get_n_grams(pred, n)
    label_ngrams = _get_n_grams(label, n)
    if not label_ngrams:
        return 0.0
    return len(pred_ngrams & label_ngrams) / len(label_ngrams)


def contained_similarity(pred: str, label: str) -> float:
    """Return 1.0 if the label appears as a substring of the prediction."""
    return 1.0 if _normalise(label) in _normalise(pred) else 0.0


# ---------------------------------------------------------------------------
# Text helpers
#   remove_stuff  – mirrors map_predictions.py:remove_stuff
#   get_n_grams   – mirrors map_predictions.py:get_n_grams
# ---------------------------------------------------------------------------

def _remove_stuff(text: str) -> str:
    """Strip model-output formatting before matching."""
    text = text.replace("A: ", "").replace("A:", "")
    text = text.replace("<answer>", "").replace("</answer>", "")
    text = text.replace("<s>", "").replace("</s>", "")
    return text.strip()


def _normalise(text: str) -> str:
    """Lower-case, replace dashes/underscores with spaces, strip punctuation."""
    text = text.lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _get_n_grams(text: str, n: int) -> set[str]:
    """Extract n-grams from a normalised string."""
    words = text.split()
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _softmax(scores: list[float]) -> list[float]:
    """Compute softmax over a list of scores."""
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


# ---------------------------------------------------------------------------
# TaxonomyMatcher
# ---------------------------------------------------------------------------

class TaxonomyMatcher:
    """Map a free-text prediction to a node in the OVEN taxonomy tree.

    Implements the cascading mapping algorithm from the CVPR 2024 paper,
    using a pluggable similarity function.  The precomputed taxonomy index
    replaces all networkx graph operations — see module docstring for the
    translation table.

    Parameters
    ----------
    index : dict
        Precomputed taxonomy index from ``taxonomy.load_taxonomy_index()``.
    similarity_fn : (str, str) -> float, optional
        Similarity function ``m(pred, label) -> score``.  Defaults to
        ``ngram_overlap_similarity`` (bigram Jaccard), which provides
        meaningful ordering without requiring CLIP or other models.
    k : int
        Number of top-scoring nodes to consider (default 3, from the
        paper's OVEN hyper-parameters).
    thr_topk : float
        Max softmax difference between top-1 and top-k for voting (default 0.005).
    thr_top2 : float
        Max softmax difference between top-1 and top-2 for voting (default 0.001).
    thr_vote : int
        Min votes for a common ancestor to be selected (default 2).
    min_score : float, optional
        Retrieval-score floor for the embedding path.  When set, a prediction
        with no lexical hit whose best top-k score is below this maps to None
        (``mapping_method="none"``).  Default None (no floor; lexical path).
    """

    def __init__(
        self,
        index: dict,
        similarity_fn: Callable[[str, str], float] | None = None,
        k: int = 3,
        thr_topk: float = 0.005,
        thr_top2: float = 0.001,
        thr_vote: int = 2,
        min_score: float | None = None,
    ):
        self.index = index
        self.node_to_path = index.get("node_to_path", {})
        self.label_to_paths = index.get("label_to_paths", {})
        self.aliases = index.get("aliases", {})
        self.all_nodes: list[str] = index.get("all_nodes", [])

        self._similarity_fn = similarity_fn or ngram_overlap_similarity
        self.k = k
        self.thr_topk = thr_topk
        self.thr_top2 = thr_top2
        self.thr_vote = thr_vote
        # When set (embedding path), a prediction whose best retrieval score is
        # below this floor and has no lexical hit maps to None ("none" method).
        self.min_score = min_score

        # Pre-normalise all node labels for faster scoring
        self._norm_labels: list[str] = [_normalise(n) for n in self.all_nodes]

        # normalised_label -> original_label (aliases map to their canonical node)
        self._norm_to_original: dict[str, str] = {}
        for node in self.all_nodes:
            self._norm_to_original[_normalise(node)] = node
        for alias, canonical in self.aliases.items():
            n = _normalise(alias)
            if n not in self._norm_to_original:
                self._norm_to_original[n] = canonical

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match_prediction(
        self,
        prediction: str,
        top_idxs: list[int] | None = None,
        top_scores: list[float] | None = None,
    ) -> dict | None:
        """Map a free-text prediction to a taxonomy node.

        Returns None only if the taxonomy has no nodes (should never happen
        with OVEN).  The algorithm always produces a node via the fallback,
        unless ``min_score`` gates it to the ``"none"`` method.

        Parameters
        ----------
        top_idxs, top_scores : optional
            Precomputed top-k node indices and retrieval scores (e.g. from
            cosine similarity).  When provided, the lexical Step 1 is skipped
            and these drive the cascade — this is the ``sentence_bert`` path.
        """
        if not self.all_nodes:
            return None

        # Embedding path = caller supplied cosine top-k.  It follows the CVPR
        # reference cascade (exact-equality in top-k, no contains-over-all),
        # whereas the lexical path keeps the original containment behaviour.
        embedding_mode = top_idxs is not None

        cleaned = _remove_stuff(prediction)
        norm_pred = _normalise(cleaned)

        # --- Step 1: obtain top-k candidates + the all-node iteration order --
        if top_idxs is None:
            #   Lexical: score all nodes, sort, take top-k.
            #   Equivalent: S = [(m(pred, v.label), v) for v in T]
            scores = [self._similarity_fn(norm_pred, nl) for nl in self._norm_labels]
            # Argsort descending to preserve the original paper's ordering of
            # equal-scored nodes (stable sort → later equal nodes don't swap).
            ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
            k = min(self.k, len(ranked))
            top_idxs = ranked[:k]
            top_scores = [scores[i] for i in top_idxs]
            all_order = ranked
        else:
            #   Embedding: top-k already retrieved.  The all-node stages
            #   (3, 4) select by depth, not score, so iteration order is
            #   irrelevant — use the original node order.
            all_order = range(len(self.all_nodes))

        top_nodes = [self.all_nodes[i] for i in top_idxs]
        s_k = _softmax(top_scores)

        # Normalised labels for the contains / n-gram stages
        top_norm = [self._norm_labels[i] for i in top_idxs]
        all_norm_sorted = [self._norm_labels[i] for i in all_order]
        all_nodes_sorted = [self.all_nodes[i] for i in all_order]

        # --- Step 2: match in top-k ---------------------------------------
        #   Embedding path: exact equality (reference check_topks).
        #   Lexical path: containment (original behaviour).
        if embedding_mode:
            cand = self._exact_in_topk(norm_pred, top_norm, top_nodes)
        else:
            cand = self._contains_check(norm_pred, top_norm, top_nodes)
        if cand is not None:
            return self._make_result(cand, "exact_match_in_top_k")

        # --- Step 3: contains check in all nodes (lexical path only) -------
        #   The reference has no contains-over-all stage; on the embedding
        #   path it would hijack cosine retrieval via spurious substrings
        #   (e.g. "food" in "seafood"), so it is skipped there.
        if not embedding_mode:
            cand = self._contains_check(norm_pred, all_norm_sorted, all_nodes_sorted)
            if cand is not None:
                return self._make_result(cand, "exact_match")

        # --- Step 4: n-gram overlap ----------------------------------------
        #   For n in (4, 3, 2), try top-k first, then all nodes
        for n in (4, 3, 2):
            pred_ngrams = _get_n_grams(norm_pred, n)
            if not pred_ngrams:
                continue

            # Top-k first
            cand = self._ngram_check(pred_ngrams, top_norm, top_nodes, n)
            if cand is not None:
                return self._make_result(cand, f"ngram_topk_match_{n}")

            # Then all nodes
            cand = self._ngram_check(pred_ngrams, all_norm_sorted, all_nodes_sorted, n)
            if cand is not None:
                return self._make_result(cand, f"ngram_match_{n}")

        # --- NONE-floor: weak retrieval and no lexical hit ----------------
        if self.min_score is not None and (not top_scores or max(top_scores) < self.min_score):
            return {"predicted_node": None, "predicted_path": None, "mapping_method": "none"}

        # --- Step 5: voting — if top-k scores are ambiguous ---------------
        if len(s_k) >= 2 and (s_k[0] - s_k[1] < self.thr_top2) and (s_k[0] - s_k[-1] < self.thr_topk):
            cand = self._voting_fallback(top_nodes)
            if cand is not None:
                return self._make_result(cand, "voting")

        # --- Step 6: top-score fallback -----------------------------------
        return self._make_result(top_nodes[0], "top_score")

    def evaluate(
        self,
        prediction: str,
        reference: str,
        reference_path: list[str] | None = None,
        top_idxs: list[int] | None = None,
        top_scores: list[float] | None = None,
    ) -> dict:
        """Evaluate a single prediction against a reference.

        Parameters
        ----------
        prediction : str
            The model's free-text output.
        reference : str
            The ground-truth entity label.
        reference_path : list[str], optional
            The ground-truth taxonomy path (root->leaf).  Looked up from the
            index if not provided.

        Returns
        -------
        dict
            success, predicted_node, predicted_path, reference_path, hP, hR,
            hF, mapping_method.
        """
        if reference_path is None:
            reference_path = self.node_to_path.get(reference)
            if reference_path is None:
                paths = self.label_to_paths.get(_normalise(reference), [])
                if paths:
                    reference_path = paths[0]

        match = self.match_prediction(prediction, top_idxs=top_idxs, top_scores=top_scores)
        pred_path = match["predicted_path"] if match else None

        if match is None or reference_path is None or pred_path is None:
            return {
                "success": False,
                "predicted_node": match["predicted_node"] if match else None,
                "predicted_path": pred_path,
                "reference_path": reference_path,
                "hP": 0.0, "hR": 0.0, "hF": 0.0,
                "mapping_method": match["mapping_method"] if match else None,
            }
        metrics = calc_hierarchical_metrics([(pred_path, reference_path)])
        # Mirror vlm-eval's check_topks / data.py: normalise both sides
        # before comparing so casing and punctuation differences don't
        # cause false negatives.
        exact = _normalise(match["predicted_node"] or "") == _normalise(reference)

        return {
            "success": exact,
            "predicted_node": match["predicted_node"],
            "predicted_path": pred_path,
            "reference_path": reference_path,
            "hP": metrics["hP"][0],
            "hR": metrics["hR"][0],
            "hF": metrics["hF"][0],
            "mapping_method": match["mapping_method"],
        }

    @classmethod
    def from_json(cls, path: str | None = None, *, similarity: str = "ngram_overlap") -> "TaxonomyMatcher":
        """Load taxonomy index and return a matcher.

        Parameters
        ----------
        path : str or None
            Path to the taxonomy index JSON.  None uses the default location.
        similarity : str
            One of ``"ngram_overlap"`` (default) or ``"contained"``.
        """
        from oven_mllm_eval.taxonomy import load_taxonomy_index
        index = load_taxonomy_index(path)
        fn = {
            "ngram_overlap": ngram_overlap_similarity,
            "contained": contained_similarity,
        }[similarity]
        return cls(index, similarity_fn=fn)

    # ------------------------------------------------------------------
    # Matching stages — each returns a node label or None
    # ------------------------------------------------------------------

    def _exact_in_topk(self, norm_pred: str, norm_labels: list[str], original_labels: list[str]) -> str | None:
        """Return the most specific top-k node whose label exactly equals norm_pred.

        Reference-faithful in-top-k check (``check_topks`` in the CVPR code).
        """
        best_node = None
        best_depth = -1
        for norm_label, original in zip(norm_labels, original_labels):
            if norm_label and norm_label == norm_pred:
                depth = len(self.node_to_path.get(original, []))
                if depth > best_depth:
                    best_depth = depth
                    best_node = original
        return best_node

    def _contains_check(self, norm_pred: str, norm_labels: list[str], original_labels: list[str]) -> str | None:
        """Return the most specific node whose normalised label is a substring of norm_pred.

        Most specific = deepest in taxonomy = longest path.
        networkx equivalent: len(anc(v)) → len(node_to_path[v]).
        """
        best_node = None
        best_depth = -1
        for norm_label, original in zip(norm_labels, original_labels):
            # Skip degenerate labels: an empty normalised label (e.g. an
            # all-non-ASCII node stripped by _normalise) is a substring of
            # every prediction, and a 1-char label matches spuriously.
            if len(norm_label) < 2:
                continue
            if norm_label in norm_pred:
                depth = len(self.node_to_path.get(original, []))
                if depth > best_depth:
                    best_depth = depth
                    best_node = original
        return best_node

    def _ngram_check(self, pred_ngrams: set[str], norm_labels: list[str], original_labels: list[str], n: int) -> str | None:
        """Return the most specific node whose n-grams overlap with pred_ngrams."""
        best_node = None
        best_depth = -1
        for norm_label, original in zip(norm_labels, original_labels):
            node_ngrams = _get_n_grams(norm_label, n)
            if pred_ngrams & node_ngrams:
                depth = len(self.node_to_path.get(original, []))
                if depth > best_depth:
                    best_depth = depth
                    best_node = original
        return best_node

    def _voting_fallback(self, top_nodes: list[str]) -> str | None:
        """Most specific common ancestor among top-k nodes with >= thr_vote votes.

        For each node, anc(v) = node_to_path[v][::-1] (leaf->root).
        Votes are counted at each depth from the leaf.
        networkx equivalent: voting logic in paper's Algorithm 1.
        """
        votes: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for node in top_nodes:
            path = self.node_to_path.get(node, [])
            # leaf->root: path[::-1]
            for i, anc in enumerate(path[::-1]):
                votes[i][anc] += 1

        # Deepest (highest i) first
        for i in sorted(votes.keys(), reverse=True):
            node, count = max(votes[i].items(), key=lambda x: x[1])
            if count > self.thr_vote:
                return node
        return None

    def _make_result(self, node: str, method: str) -> dict:
        """Build the result dict for a matched node."""
        path = self.node_to_path.get(node)
        if path is None:
            paths = self.label_to_paths.get(_normalise(node), [])
            path = paths[0] if paths else None
        return {
            "predicted_node": node,
            "predicted_path": path,
            "mapping_method": method,
        }
