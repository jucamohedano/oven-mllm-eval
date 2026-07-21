"""Helpers shared by the analysis scripts.

Only for things used by more than one script in ``analysis/``. Anything the
evaluation pipeline also needs belongs in ``src/oven_mllm_eval`` instead, so the
one-way dependency (analysis consumes the library, never the reverse) holds.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "tex_escape",
    "find_image",
    "IMAGE_EXTENSIONS",
    "BG", "INK", "MUTED", "GRID", "BLUE", "GREEN", "ORANGE",
    "set_style",
    "apply_thesis_style",
    "center_crop_square",
]

# OVEN images are not consistently lowercase .jpg on disk.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".JPEG", ".JPG", ".png")

# Shared palette for the hand-drawn diagram figures. Colour-blind safe (Okabe-Ito).
BG = "#fffdfa"
INK = "#2f2f2f"
MUTED = "#6b675f"
GRID = "#ded9ce"
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"


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


def set_style() -> None:
    """Matplotlib style for the hand-drawn diagram figures."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "text.color": INK,
            "axes.labelcolor": INK,
        }
    )


def apply_thesis_style():
    """Seaborn theme matching the thesis result plots. Returns the seaborn module."""
    import seaborn as sns

    sns.set_theme(
        context="talk",
        style="whitegrid",
        rc={
            "axes.facecolor": "#FBFAF7",
            "figure.facecolor": "white",
            "axes.edgecolor": "#3A3A3A",
            "axes.labelcolor": "#202020",
            "text.color": "#202020",
            "grid.color": "#D8D3CA",
            "grid.linewidth": 0.9,
            "font.family": "DejaVu Serif",
        },
    )
    return sns


def center_crop_square(image):
    """Crop an image array (H, W, ...) to a centred square."""
    height, width = image.shape[:2]
    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    return image[y0 : y0 + side, x0 : x0 + side]
