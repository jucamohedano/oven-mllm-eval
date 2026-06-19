"""Unbiased pass@k estimator from the Codex paper (Chen et al., 2021).

Uses the numerically stable product form::

    pass@k = 1 − ∏_{i=0}^{k-1} (n − c − i) / (n − i)

which is equivalent to 1 − C(n−c, k) / C(n, k) but avoids computing
large binomial coefficients.

Source: adapted from the limit-of-RLVR project.
"""

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k.

    Parameters
    ----------
    n : int
        Total number of samples per problem.
    c : int
        Number of correct samples.
    k : int
        Number of samples to evaluate (k ≤ n).

    Returns
    -------
    float
        Estimated probability that at least one of *k* samples is correct.
    """
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))
