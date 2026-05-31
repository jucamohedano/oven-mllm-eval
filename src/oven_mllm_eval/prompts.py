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
    "barebones": "{} Answer in the format 'A: <answer>.'",
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


def build_messages(
    question: str,
    variant: str = "barebones",
    feedback: Optional[str] = None,
) -> list[dict]:
    """Build a chat-style message list for the vLLM API.

    Parameters
    ----------
    question : str
        The OVEN question.
    variant : str
        Prompt variant name.
    feedback : str, optional
        If given, appended as a follow-up user message (for iterative resampling).

    Returns
    -------
    list[dict]
        Messages in OpenAI chat format:
        [{"role": "user", "content": [{"type": "text", ...}, {"type": "image_url", ...}]}]
        The image placeholder is added by the caller (run_inference.py) since
        we need the actual file path or base64 data.
    """
    prompt_text = get_prompt(question, variant)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                # image_url entry added by the caller
            ],
        }
    ]

    if feedback:
        messages.append({"role": "assistant", "content": ""})  # placeholder
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": feedback}],
        })

    return messages
