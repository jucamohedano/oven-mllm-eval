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

# Categories that count as a *verified* positive: the prediction is at least as
# specific as the ground truth (or one of its aliases).  `answer_contains_prediction`
# (prediction ⊆ ground truth) is detected but deliberately EXCLUDED: it credits
# predictions that are LESS specific than the ground truth (e.g. "cat" for
# "domestic cat", "Boeing 747" for "Boeing 747-400"). On a fine-grained entity
# task those are under-specific, so counting them inflates the support set.
SUPPORTED_CATEGORIES = ("exact", "alias", "contains_answer", "contains_alias")


def is_supported(category: str | None) -> bool:
    return category in SUPPORTED_CATEGORIES


def _phrase_in(needle: str, haystack: str) -> bool:
    """Whole-token containment over normalized text.

    Both strings are single-space-separated alphanumeric tokens (see
    ``normalize``), so space-padding makes ``in`` match only contiguous token
    runs — e.g. " cat " is NOT in " caterpillar ", and " 50 " is NOT in
    " 1950 ", while " golden retriever " IS in " a golden retriever dog ".
    This is word-boundary matching without a regex or tokenizer dependency.
    """
    return bool(needle) and f" {needle} " in f" {haystack} "


def _is_alias_seed(alias: str) -> bool:
    """Whether an alias may seed a *containment* check (not exact match).

    Short or purely-numeric aliases ("", "50", "au", "707") substring-match
    unrelated predictions, so they are barred from containment seeding — they
    can still match via exact ``alias`` equality.
    """
    return len(alias) >= 4 and not alias.isdigit()


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
    """Classify how a judge-positive prediction relates to the ground truth.

    Categories (first match wins):
    - ``exact``                      — prediction == answer
    - ``alias``                      — prediction == one of the answer's aliases
    - ``contains_answer``            — answer ⊆ prediction (whole-token)
    - ``contains_alias``             — some alias ⊆ prediction (whole-token)
    - ``answer_contains_prediction`` — prediction ⊆ answer (whole-token)
    - ``None``                       — no relation

    Only the first four count as *verified* support (``SUPPORTED_CATEGORIES`` /
    ``is_supported``): each requires the prediction to be at least as specific
    as the answer (or an equivalent alias).  ``answer_contains_prediction`` is
    returned for diagnostics only — it is under-specific (a hypernym/fragment).

    All containment uses whole-token matching (``_phrase_in``).  Alias matching
    is exhaustive over every alias; containment additionally skips short/numeric
    alias seeds (``_is_alias_seed``).  ``aliases_by_canonical`` may be omitted,
    disabling only alias-based matching.
    """
    aliases_by_canonical = aliases_by_canonical or {}
    pred_norm = normalize(prediction)
    answer_norm = normalize(answer)
    aliases = aliases_by_canonical.get(answer_norm, set())

    if pred_norm == answer_norm:
        return "exact"

    if pred_norm in aliases:
        return "alias"

    if _phrase_in(answer_norm, pred_norm):
        return "contains_answer"

    if any(_phrase_in(a, pred_norm) for a in aliases if _is_alias_seed(a)):
        return "contains_alias"

    if _phrase_in(pred_norm, answer_norm):
        return "answer_contains_prediction"

    return None
