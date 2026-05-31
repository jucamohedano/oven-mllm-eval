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
