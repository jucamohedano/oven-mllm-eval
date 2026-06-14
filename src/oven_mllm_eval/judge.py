"""Shared helpers for the judge model (Phase 2 of the evaluation pipeline)."""

from __future__ import annotations

import json
import re

# ---------------------------------------------------------------------------
# JSON schema for guided decoding
# ---------------------------------------------------------------------------

JUDGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["yes", "no"]},
        "reason": {"type": "string", "maxLength": 200},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are verifying whether a model's answer to a visual question "
    "matches the ground truth."
)


def build_judge_prompt(question: str, ground_truth: str, rollout_text: str) -> str:
    """Build a text-only judge prompt for a single rollout.

    The prompt asks the judge to decide whether *rollout_text* is
    semantically equivalent to *ground_truth*, given *question* for
    disambiguation context.
    """
    return (
        f"{_JUDGE_SYSTEM}\n\n"
        f"Question: {question}\n"
        f"Ground truth: {ground_truth}\n"
        f"Model's answer: {rollout_text}\n\n"
        "Is the model's answer semantically equivalent to the ground truth?\n\n"
        "Respond with a JSON object containing exactly two fields:\n"
        '- "verdict": either "yes" or "no"\n'
        '- "reason":  one short sentence explaining your decision\n\n'
        "Always include both fields, for both yes and no verdicts."
    )


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


class JudgeParseError(ValueError):
    """Raised when the judge output cannot be parsed."""


def parse_judge_output(output_text: str) -> tuple[bool, str]:
    """Parse a guided-decoded JSON judge output.

    Returns
    -------
    (verdict, reason)
        ``verdict`` is ``True`` for "yes", ``False`` for "no".
        ``reason`` is the one-sentence explanation string.

    Raises
    ------
    JudgeParseError
        If the output is not valid JSON or missing required fields.
    """
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"invalid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise JudgeParseError(f"expected JSON object, got {type(parsed).__name__}")

    verdict_raw = parsed.get("verdict")
    reason_raw = parsed.get("reason")

    if verdict_raw not in ("yes", "no"):
        raise JudgeParseError(f"verdict must be 'yes' or 'no', got {verdict_raw!r}")

    if not isinstance(reason_raw, str) or not reason_raw.strip():
        raise JudgeParseError(f"reason missing or empty, got {reason_raw!r}")

    return verdict_raw == "yes", reason_raw.strip()


# ---------------------------------------------------------------------------
# Free-form prompt (no CoT, no reason — adapted from answer-matching)
# ---------------------------------------------------------------------------


def build_judge_prompt_free_form(
    question: str, ground_truth: str, rollout_text: str
) -> str:
    """Build a free-form judge prompt for OVEN entity matching.

    No chain-of-thought, no post-hoc reason.  The model outputs ``0`` or ``1``
    inside ``<answer>…</answer>`` tags.

    The prompt is intentionally neutral about specificity: the question
    already encodes the expected granularity (e.g. "what breed is this?"
    vs "what type of animal is this?"), so the judge only needs to check
    whether the response and ground truth refer to the same entity.
    """
    return (
        "Your task is to judge whether the given response to a question "
        "matches a given ground truth answer or not. You are provided with "
        "a question, a ground truth answer, and the response you need to "
        "judge.\n\n"
        "The response matches the ground truth if both are semantically "
        "equivalent — they refer to the same entity at the level of "
        "specificity asked by the question.\n\n"
        "Possible judgments:\n\n"
        '"0": The response does not match the ground-truth answer.\n'
        '"1": The response matches the ground-truth.\n\n'
        f'Question: "{question}"\n'
        f'Ground truth: "{ground_truth}"\n'
        f'Response: "{rollout_text}"\n\n'
        "Your job is to ONLY check whether the given response matches "
        "the ground truth answer or not in the context of the question. "
        "You DO NOT NEED to assess the correctness of the response. "
        "This is part of an automated evaluation process, therefore you "
        "MUST OUTPUT your final answer as \"0\" or \"1\" in "
        "<answer> </answer> tags.\n"
        "YOU SHOULD ALWAYS END YOUR RESPONSE WITH <answer>0</answer> OR "
        "<answer>1</answer> TAGS."
    )


def parse_free_form_output(output_text: str) -> tuple[bool, str]:
    """Parse a free-form judge output.

    Extracts the verdict from ``<answer>0</answer>`` or
    ``<answer>1</answer>`` tags.  Takes the **last** occurrence (in case the
    model emits text after a preliminary answer).

    Returns
    -------
    (verdict, matched_digit)
        ``verdict`` is ``True`` for ``"1"``, ``False`` for ``"0"``.
        ``matched_digit`` is ``"0"`` or ``"1"`` (or ``""`` if no tag found).
    """
    matches = list(re.finditer(r"<answer>\s*(\d)\s*</answer>", output_text))
    if matches:
        digit = matches[-1].group(1)  # last occurrence
        return digit == "1", digit
    return False, ""
