"""Shared helpers for the judge model (Phase 2 of the evaluation pipeline)."""

from __future__ import annotations

import json

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
