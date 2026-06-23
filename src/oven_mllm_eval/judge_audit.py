"""Shared judge-verdict auditing logic.

A judge positive (the LM marked a rollout correct) is "supported" when it can
be mechanically verified against the ground truth by exact match, a known
alias, or containment.  This module is the single source of truth for that
classification, used by both the batch auditor
(``scripts/audit_judge_false_positives.py``) and the interactive dashboard
(``scripts/explore_judgments.py``).

Note: ``normalize`` here is intentionally more aggressive than
``oven_mllm_eval.scores.normalize`` — it strips HTML/markup and collapses
whitespace, because rollout text is free-form model output.
"""

from __future__ import annotations

import re
from typing import Any

# Exact-string variants treated as an explicit refusal to answer.
IDK_VARIANTS = {
    "i don't know",
    "i don't know.",
    "i don't know,",
    "i dont know",
    "i dont know.",
}


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_idk(text: str) -> bool:
    return text.strip().lower() in IDK_VARIANTS


def build_alias_map(index: dict[str, Any]) -> dict[str, set[str]]:
    aliases_by_canonical: dict[str, set[str]] = {}
    for alias_norm, canonical in index.get("aliases", {}).items():
        aliases_by_canonical.setdefault(normalize(canonical), set()).add(alias_norm)
    return aliases_by_canonical


def classify_positive(
    *,
    prediction: str,
    answer: str,
    aliases_by_canonical: dict[str, set[str]] | None = None,
) -> str | None:
    """Return how a judge-positive prediction is supported, or None.

    Categories: ``exact``, ``alias``, ``contains_answer``,
    ``answer_contains_prediction``.  ``aliases_by_canonical`` may be omitted
    (e.g. when no taxonomy index is loaded), disabling only alias matching.
    """
    aliases_by_canonical = aliases_by_canonical or {}
    pred_norm = normalize(prediction)
    answer_norm = normalize(answer)

    if pred_norm == answer_norm:
        return "exact"

    if pred_norm in aliases_by_canonical.get(answer_norm, set()):
        return "alias"

    if answer_norm and answer_norm in pred_norm:
        return "contains_answer"

    if pred_norm and pred_norm in answer_norm:
        return "answer_contains_prediction"

    return None
