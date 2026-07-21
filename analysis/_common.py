"""Helpers shared by the analysis scripts.

Only for things used by more than one script in ``analysis/``. Anything the
evaluation pipeline also needs belongs in ``src/oven_mllm_eval`` instead, so the
one-way dependency (analysis consumes the library, never the reverse) holds.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["tex_escape", "find_image", "IMAGE_EXTENSIONS"]

# OVEN images are not consistently lowercase .jpg on disk.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".JPEG", ".JPG", ".png")


def tex_escape(text) -> str:
    """Escape a string for literal inclusion in LaTeX.

    Covers ``~`` and ``^`` as well as the usual specials. ``None`` becomes an
    empty string so callers can pass optional fields directly.
    """
    if text is None:
        return ""
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
        .replace("#", r"\#").replace("_", r"\_")
        .replace("{", r"\{").replace("}", r"\}")
        .replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    )


def find_image(image_id: str, image_dir: Path) -> Path | None:
    """Locate an image by id, trying the known extensions.

    Returns ``None`` when nothing matches, so callers can skip missing images.
    """
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None
