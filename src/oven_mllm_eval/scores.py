# ==============================================================================
# Adapted from vlm-eval/src/vlmeval/calculate_scores/scores.py
#
# calc_hierarchical_metrics is pure Python with no external dependencies.
# ==============================================================================

import statistics
from collections import defaultdict
from string import punctuation
from typing import Dict, List, Optional, Tuple, Union


def remove_punctuation(s):
    return s.translate(str.maketrans("", "", punctuation))


def normalize(s):
    s = s.lower()
    s = s.replace("-", " ")
    s = remove_punctuation(s)
    return s


def calc_hierarchical_metrics(path_pairs):
    """
    Calculate hierarchical Precision (hP), Recall (hR), and F-score (hF)
    for pairs of taxonomy paths.
    """
    results = {
        'hR': [],
        'hP': [],
        'hF': []
    }
    for pred_path, ref_path in path_pairs:
        pred_ancestors = set()
        ref_ancestors = set()

        for i in range(len(pred_path)):
            subpath = tuple(pred_path[-(i+1):])
            pred_ancestors.add(subpath)

        for i in range(len(ref_path)):
            subpath = tuple(ref_path[-(i+1):])
            ref_ancestors.add(subpath)

        common = pred_ancestors.intersection(ref_ancestors)

        if len(ref_ancestors) > 0:
            hR = len(common) / len(ref_ancestors)
        else:
            hR = 0.0

        if len(pred_ancestors) > 0:
            hP = len(common) / len(pred_ancestors)
        else:
            hP = 0.0

        if hP + hR > 0:
            hF = 2 * (hP * hR) / (hP + hR)
        else:
            hF = 0.0

        results['hR'].append(hR)
        results['hP'].append(hP)
        results['hF'].append(hF)

    return results


def _suffix_weights(path, decay=0.5):
    """Weight deeper/specific suffixes more than broad/root suffixes.

    Paths in the OVEN taxonomy index are leaf→root.  The broadest suffix is
    therefore ``("root",)`` and the most specific suffix is the full
    leaf→root path.  With the default decay=0.5, each step toward the leaf is
    worth twice as much as the broader ancestor before it.
    """
    total_depth = len(path)
    return {
        tuple(path[-(i + 1):]): decay ** (total_depth - (i + 1))
        for i in range(total_depth)
    }


def _is_strict_ancestor_path(pred_path, ref_path):
    """Whether pred_path names a broader ancestor of ref_path."""
    return (
        len(pred_path) < len(ref_path)
        and len(pred_path) > 0
        and tuple(ref_path[-len(pred_path):]) == tuple(pred_path)
    )


def _is_strict_descendant_path(pred_path, ref_path):
    """Whether pred_path is more specific than ref_path."""
    return (
        len(pred_path) > len(ref_path)
        and len(ref_path) > 0
        and tuple(pred_path[-len(ref_path):]) == tuple(ref_path)
    )


def calc_specificity_hierarchical_metrics(
    path_pairs,
    *,
    decay=0.5,
    under_specific_penalty=0.5,
):
    """
    Calculate specificity-aware hierarchical Precision/Recall/F-score.

    This keeps the original hP/hR/hF semantics but makes the score more
    discriminative for fine-grained entity recognition:

    - deeper, more specific taxonomy matches receive larger weights;
    - broad/root-only overlap contributes little;
    - predictions that are strict ancestors of the reference are penalized.

    The output is intended to sit next to the original hP/hR/hF, not replace it.
    """
    results = {
        "specific_hR": [],
        "specific_hP": [],
        "specific_hF": [],
        "under_specific": [],
        "over_specific": [],
        "depth_delta": [],
    }

    for pred_path, ref_path in path_pairs:
        if not pred_path or not ref_path:
            results["specific_hR"].append(0.0)
            results["specific_hP"].append(0.0)
            results["specific_hF"].append(0.0)
            results["under_specific"].append(False)
            results["over_specific"].append(False)
            results["depth_delta"].append(None)
            continue

        pred_weights = _suffix_weights(pred_path, decay=decay)
        ref_weights = _suffix_weights(ref_path, decay=decay)
        pred_ancestors = set(pred_weights)
        ref_ancestors = set(ref_weights)
        common = pred_ancestors.intersection(ref_ancestors)

        pred_total = sum(pred_weights.values())
        ref_total = sum(ref_weights.values())
        hP = (
            sum(pred_weights[suffix] for suffix in common) / pred_total
            if pred_total > 0
            else 0.0
        )
        hR = (
            sum(ref_weights[suffix] for suffix in common) / ref_total
            if ref_total > 0
            else 0.0
        )

        if hP + hR > 0:
            hF = 2 * (hP * hR) / (hP + hR)
        else:
            hF = 0.0

        under_specific = _is_strict_ancestor_path(pred_path, ref_path)
        over_specific = _is_strict_descendant_path(pred_path, ref_path)
        if under_specific:
            hP *= under_specific_penalty
            hR *= under_specific_penalty
            hF *= under_specific_penalty

        results["specific_hR"].append(hR)
        results["specific_hP"].append(hP)
        results["specific_hF"].append(hF)
        results["under_specific"].append(under_specific)
        results["over_specific"].append(over_specific)
        results["depth_delta"].append(len(pred_path) - len(ref_path))

    return results
