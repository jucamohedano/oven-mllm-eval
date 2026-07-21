#!/usr/bin/env python3
"""Draw the introduction figure for specificity ambiguity.

The figure shows one Golden Retriever photograph next to a stack of answers
that are all semantically valid at different specificity levels, while flat
exact-match evaluation designates only one target string.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from _common import BG, BLUE, GREEN, GRID, INK, MUTED, ORANGE, center_crop_square, set_style


PANEL = "#f8f6f0"
VERMILLION = "#D55E00"


DEFAULT_IMAGE = (
    "docs/thesis_latex/template-latex-lm-disi-en/lm_master_disi_en/"
    "figures/golden-retriever.jpg"
)
DEFAULT_OUTPUT = (
    "viz/specificity_granularity/ch1_specificity_ambiguity.png"
)
DEFAULT_PDF = "viz/specificity_granularity/ch1_specificity_ambiguity.pdf"
DEFAULT_SVG = "viz/specificity_granularity/ch1_specificity_ambiguity.svg"


@dataclass(frozen=True)
class AnswerLevel:
    answer: str
    detail: str
    specificity: float
    flat_target: bool = False
    italic_answer: bool = False


LEVELS = [
    AnswerLevel("animal", "broad ancestor class", 0.10),
    AnswerLevel("dog", "common-level category", 0.30),
    AnswerLevel("Canis lupus familiaris", "scientific species label", 0.50, italic_answer=True),
    AnswerLevel("retriever", "breed-group ancestor", 0.70),
    AnswerLevel("Golden Retriever", "designated exact-match target", 1.00, flat_target=True),
]


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255 for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(max(0, min(1, value)) * 255):02x}" for value in rgb)


def blend(color_a: str, color_b: str, weight_b: float) -> str:
    rgb_a = _hex_to_rgb(color_a)
    rgb_b = _hex_to_rgb(color_b)
    rgb = tuple((1 - weight_b) * a + weight_b * b for a, b in zip(rgb_a, rgb_b))
    return _rgb_to_hex(rgb)


def draw_photo_card(ax, image_path: Path) -> None:
    import matplotlib.image as mpimg
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyBboxPatch, Rectangle

    left, bottom = 0.55, 0.78
    width, height = 4.05, 4.55
    card = FancyBboxPatch(
        (left, bottom),
        width,
        height,
        boxstyle="round,pad=0.045,rounding_size=0.08",
        facecolor="#ffffff",
        edgecolor=GRID,
        linewidth=1.2,
        zorder=2,
    )
    card.set_path_effects(
        [
            pe.withSimplePatchShadow(
                offset=(2.0, -2.0),
                shadow_rgbFace=(0.74, 0.70, 0.62),
                alpha=0.18,
            )
        ]
    )
    ax.add_patch(card)

    image_left = left + 0.22
    image_bottom = bottom + 0.72
    image_size = width - 0.44
    if image_path.exists():
        image = center_crop_square(mpimg.imread(image_path))
        ax.imshow(
            image,
            extent=(image_left, image_left + image_size, image_bottom, image_bottom + image_size),
            interpolation="lanczos",
            zorder=3,
        )
    else:
        ax.add_patch(
            Rectangle(
                (image_left, image_bottom),
                image_size,
                image_size,
                facecolor="#f3eee5",
                edgecolor="none",
                zorder=3,
            )
        )
        ax.text(
            image_left + image_size / 2,
            image_bottom + image_size / 2,
            "image missing",
            ha="center",
            va="center",
            fontsize=12,
            color=MUTED,
            zorder=4,
        )
    ax.add_patch(
        Rectangle(
            (image_left, image_bottom),
            image_size,
            image_size,
            facecolor="none",
            edgecolor=INK,
            linewidth=1.0,
            alpha=0.22,
            zorder=4,
        )
    )

def draw_answer_card(ax, *, level: AnswerLevel, y: float) -> None:
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyBboxPatch

    left = 5.92
    width = 6.10
    height = 0.68
    edge = blend(ORANGE, BLUE, level.specificity)
    face = blend("#fbf2df", "#e4f2f8", level.specificity)

    if level.flat_target:
        edge = BLUE
        face = "#eaf5fb"
        linewidth = 2.4
    else:
        linewidth = 1.45

    card = FancyBboxPatch(
        (left, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.075",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=4,
    )
    card.set_path_effects(
        [
            pe.withSimplePatchShadow(
                offset=(1.5, -1.5),
                shadow_rgbFace=(0.74, 0.70, 0.62),
                alpha=0.16,
            )
        ]
    )
    ax.add_patch(card)

    ax.text(
        left + 0.26,
        y + 0.12,
        level.answer,
        ha="left",
        va="center",
        fontsize=13.2 if level.flat_target else 12.5,
        fontweight="bold" if level.flat_target else "normal",
        fontstyle="italic" if level.italic_answer else "normal",
        color=INK,
        zorder=6,
    )
    ax.text(
        left + 0.26,
        y - 0.15,
        level.detail,
        ha="left",
        va="center",
        fontsize=8.7,
        color=MUTED,
        zorder=6,
    )


def draw_specificity_axis(ax, *, top_y: float, bottom_y: float) -> None:
    from matplotlib.patches import FancyArrowPatch

    x = 5.23
    arrow = FancyArrowPatch(
        (x, top_y + 0.32),
        (x, bottom_y - 0.32),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.6,
        color=MUTED,
        alpha=0.55,
        zorder=3,
    )
    ax.add_patch(arrow)
    ax.text(
        x,
        top_y + 0.44,
        "general",
        ha="center",
        va="center",
        fontsize=8.6,
        color=MUTED,
        zorder=5,
    )
    ax.text(
        x,
        bottom_y - 0.44,
        "specific",
        ha="center",
        va="center",
        fontsize=8.6,
        color=MUTED,
        zorder=5,
    )
    ax.text(
        x - 0.25,
        (top_y + bottom_y) / 2,
        "increasing specificity",
        ha="center",
        va="center",
        rotation=90,
        fontsize=9.0,
        color=MUTED,
        zorder=5,
    )


def build_figure(image_path: Path, output: Path, *, pdf: Path | None, svg: Path | None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    set_style()
    fig, ax = plt.subplots(figsize=(13.2, 6.2))
    ax.set_xlim(0, 12.35)
    ax.set_ylim(0.25, 5.75)
    ax.axis("off")

    ax.add_patch(
        Rectangle(
            (0.25, 0.45),
            11.65,
            5.08,
            facecolor=PANEL,
            edgecolor="none",
            alpha=0.52,
            zorder=0,
        )
    )
    draw_photo_card(ax, image_path)

    ax.text(
        5.92,
        5.28,
        "Answers at different specificity levels",
        ha="left",
        va="center",
        fontsize=14.2,
        fontweight="bold",
        color=INK,
        zorder=4,
    )
    y_positions = [4.44, 3.68, 2.92, 2.16, 1.22]
    draw_specificity_axis(ax, top_y=y_positions[0], bottom_y=y_positions[-1])
    for level, y in zip(LEVELS, y_positions):
        draw_answer_card(ax, level=level, y=y)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved: {output}")
    if pdf:
        pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf, bbox_inches="tight")
        print(f"Saved: {pdf}")
    if svg:
        svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(svg, bbox_inches="tight")
        print(f"Saved: {svg}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the specificity ambiguity thesis figure.")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Path to the Golden Retriever image.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="PNG output path.")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="PDF output path. Pass an empty string to skip.")
    parser.add_argument("--svg", default=DEFAULT_SVG, help="SVG output path. Pass an empty string to skip.")
    args = parser.parse_args()

    build_figure(
        Path(args.image),
        Path(args.output),
        pdf=Path(args.pdf) if args.pdf else None,
        svg=Path(args.svg) if args.svg else None,
    )


if __name__ == "__main__":
    main()
