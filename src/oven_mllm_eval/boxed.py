r"""Parsing of ``\boxed{...}`` answers and answer-string normalisation.

Consolidated from ``run_recursive_self_agg.py``, ``build_verl_oven_parquet.py``
and ``mine_unlockable_examples.py``, which each carried a byte-identical copy.
Keeping one implementation matters: these functions decide what counts as the
model's answer, so independent copies drifting apart changes results silently.

Note the fallback contract of :func:`extract_boxed_answer`. When no ``\boxed{}``
is present it returns ``(text.strip(), False)``, i.e. the *whole* text. Callers
that need "no box means no answer" must check the returned flag, or use a strict
variant, otherwise a long reasoning trace can be matched as if it were a label.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["extract_boxed_answer", "normalize_answer"]


def _strip_latex_answer(answer: str) -> str:
    answer = answer.strip().strip("$").strip().strip(".").strip()
    wrappers = (r"\text", r"\mathrm", r"\operatorname", r"\mathbf")
    changed = True
    while changed:
        changed = False
        for wrapper in wrappers:
            prefix = wrapper + "{"
            if answer.startswith(prefix) and answer.endswith("}"):
                answer = answer[len(prefix):-1].strip()
                changed = True
    return answer.strip().strip("$").strip().strip(".").strip()


def extract_boxed_answer(text: str) -> tuple[str, bool]:
    """Extract the last ``\\boxed{...}`` answer, supporting nested braces.

    Returns ``(answer, True)`` when a box was found, else ``(text.strip(), False)``.
    """
    matches: list[str] = []
    start = 0
    needle = r"\boxed{"
    while True:
        box_start = text.find(needle, start)
        if box_start < 0:
            break
        content_start = box_start + len(needle)
        depth = 1
        idx = content_start
        while idx < len(text) and depth > 0:
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            idx += 1
        if depth == 0:
            answer = _strip_latex_answer(text[content_start:idx - 1])
            if answer:
                matches.append(answer)
            start = idx
        else:
            break
    if matches:
        return matches[-1], True
    return text.strip(), False


def normalize_answer(text: str) -> str:
    """Lowercase, strip LaTeX wrappers and punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
