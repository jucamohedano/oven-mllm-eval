from __future__ import annotations

from typing import Optional

"""Prompt construction for OVEN evaluation.

Adapted from vlm-eval/src/vlmeval/prompt_templates.py.

We use a simplified structure: instead of per-model keys, we have a single
"generic" prompt template per variant since we target Qwen3-VL via the
OpenAI-compatible vLLM API. The model's chat template is applied server-side,
so we only need the user-facing text.

The ``{}`` placeholder is replaced with the OVEN question text.
"""

# ---------------------------------------------------------------------------
# Prompt variants — the {] placeholder receives the question string
# ---------------------------------------------------------------------------

PROMPT_VARIANTS = {
    "base_pretrained": "Q: {}\nA:",
    "barebones": "{} Answer in the format 'A: <answer>.'",
    "concise": "{} Answer questions directly and concisely. If you don't know, say 'I don't know'.",
    "concise_no_idk": "{} Answer questions directly and concisely. If you don't know, give your best guess.",
    "default": (
        "{} Do not give any extra text. Do not answer in a full sentence. "
        "Do not specify your certainty about the answer. Give your best guess "
        "if you are not sure. Answer in the format 'A: <answer>.'"
    ),
    "specific": (
        "{} Do not give any extra text. Do not answer in a full sentence. "
        "Do not specify your certainty about the answer. Give your best guess "
        "if you are not sure. Be as specific as possible. "
        "Answer in the format 'A: <answer>.'"
    ),
    "vague": (
        "{} Do not try to be overly specific. Aim for a simple answer as if "
        "you were talking to a child. Answer in the format 'A: <answer>.'"
    ),
}


def get_prompt(question: str, variant: str = "barebones") -> str:
    """Format a question with the given prompt variant.

    Parameters
    ----------
    question : str
        The OVEN question text (e.g. "what is the model of this aircraft?").
    variant : str
        One of 'barebones', 'default', 'specific', 'vague'.

    Returns
    -------
    str
        The formatted prompt string.
    """
    template = PROMPT_VARIANTS.get(variant)
    if template is None:
        raise ValueError(
            f"Unknown prompt variant '{variant}'. "
            f"Available: {list(PROMPT_VARIANTS.keys())}"
        )
    return template.format(question)
