#!/usr/bin/env python3
"""Draw a simplified Qwen3-VL architecture diagram for the thesis."""

from __future__ import annotations

import argparse
from pathlib import Path


BG = "#fffdfa"
PANEL = "#f8f6f0"
INK = "#2f2f2f"
MUTED = "#6b675f"
GRID = "#ded9ce"
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#6A5ACD"

DEFAULT_OUTPUT = "viz/architectures/qwen3_vl_simplified_architecture.png"
DEFAULT_PDF = "viz/architectures/qwen3_vl_simplified_architecture.pdf"
DEFAULT_SVG = "viz/architectures/qwen3_vl_simplified_architecture.svg"


def set_style() -> None:
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


def draw_round_box(
    ax,
    *,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    subtitle: str | None = None,
    edge: str,
    face: str,
    fontsize: float = 13.0,
    label_weight: str = "bold",
    zorder: int = 4,
) -> None:
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyBboxPatch

    left, bottom = xy
    patch = FancyBboxPatch(
        (left, bottom),
        width,
        height,
        boxstyle="round,pad=0.055,rounding_size=0.12",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.8,
        zorder=zorder,
    )
    patch.set_path_effects(
        [
            pe.withSimplePatchShadow(
                offset=(1.7, -1.7),
                shadow_rgbFace=(0.74, 0.70, 0.62),
                alpha=0.18,
            )
        ]
    )
    ax.add_patch(patch)
    ax.text(
        left + width / 2,
        bottom + height * (0.58 if subtitle else 0.50),
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=label_weight,
        color=INK,
        zorder=zorder + 1,
    )
    if subtitle:
        ax.text(
            left + width / 2,
            bottom + height * 0.30,
            subtitle,
            ha="center",
            va="center",
            fontsize=8.8,
            color=MUTED,
            zorder=zorder + 1,
        )


def draw_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    rad: float = 0.0,
    linewidth: float = 2.0,
    alpha: float = 0.9,
) -> None:
    from matplotlib.patches import FancyArrowPatch

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        connectionstyle=f"arc3,rad={rad}",
        mutation_scale=13,
        linewidth=linewidth,
        color=color,
        alpha=alpha,
        zorder=3,
    )
    ax.add_patch(arrow)


def draw_image_input(ax) -> None:
    from matplotlib.patches import Circle, Polygon, Rectangle

    draw_round_box(
        ax,
        xy=(0.55, 3.38),
        width=2.05,
        height=1.25,
        label="Image input",
        subtitle=None,
        edge=BLUE,
        face="#eef7fc",
        fontsize=12.0,
    )
    ax.add_patch(Rectangle((0.92, 3.50), 0.54, 0.38, facecolor="#cfe6f3", edgecolor=BLUE, linewidth=0.9, zorder=6))
    ax.add_patch(Circle((1.27, 3.82), 0.06, facecolor=ORANGE, edgecolor="none", zorder=7))
    ax.add_patch(
        Polygon(
            [(0.95, 3.52), (1.10, 3.72), (1.23, 3.58), (1.42, 3.86), (1.44, 3.52)],
            closed=True,
            facecolor=GREEN,
            edgecolor="none",
            alpha=0.72,
            zorder=7,
        )
    )


def draw_text_input(ax) -> None:
    draw_round_box(
        ax,
        xy=(0.55, 1.45),
        width=2.05,
        height=1.25,
        label="Text input",
        subtitle=None,
        edge=ORANGE,
        face="#fff4dd",
        fontsize=12.0,
    )
    for idx, line in enumerate(["What is", "shown here?"]):
        ax.text(1.02, 1.93 - idx * 0.23, line, ha="left", va="center", fontsize=9.0, color=INK, zorder=7)


def draw_token_strip(
    ax,
    *,
    x: float,
    y: float,
    colors: list[str],
    label: str | None = None,
    size: float = 0.15,
    gap: float = 0.05,
) -> None:
    from matplotlib.patches import Rectangle

    for idx, color in enumerate(colors):
        ax.add_patch(
            Rectangle(
                (x + idx * (size + gap), y),
                size,
                size,
                facecolor=color,
                edgecolor=INK,
                linewidth=0.6,
                alpha=0.86,
                zorder=7,
            )
        )
    if label:
        ax.text(
            x + (len(colors) * (size + gap) - gap) / 2,
            y - 0.16,
            label,
            ha="center",
            va="top",
            fontsize=7.8,
            color=MUTED,
            zorder=7,
        )


def draw_shared_token_space(ax) -> None:
    draw_round_box(
        ax,
        xy=(5.42, 2.02),
        width=2.70,
        height=1.45,
        label="Shared token space",
        subtitle=None,
        edge=GREEN,
        face="#edf8f3",
        fontsize=12.2,
    )
    draw_token_strip(
        ax,
        x=5.93,
        y=2.36,
        colors=[BLUE, BLUE, BLUE, GREEN, ORANGE, ORANGE, ORANGE, ORANGE],
        label=None,
    )
    ax.text(
        6.77,
        2.17,
        "vision + text as one sequence",
        ha="center",
        va="center",
        fontsize=8.6,
        color=MUTED,
        zorder=7,
    )


def build_figure(output: Path, *, pdf: Path | None, svg: Path | None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    set_style()
    fig, ax = plt.subplots(figsize=(13.4, 5.8))
    ax.set_xlim(0, 12.55)
    ax.set_ylim(0.55, 5.65)
    ax.axis("off")

    ax.add_patch(Rectangle((0.25, 0.78), 12.05, 4.55, facecolor=PANEL, edgecolor="none", alpha=0.55, zorder=0))

    ax.text(
        0.55,
        5.18,
        "Qwen3-VL simplified multimodal flow",
        ha="left",
        va="center",
        fontsize=18.0,
        fontweight="bold",
        color=INK,
        zorder=4,
    )

    draw_image_input(ax)
    draw_text_input(ax)

    draw_round_box(
        ax,
        xy=(3.15, 3.40),
        width=1.80,
        height=1.05,
        label="Vision encoder",
        subtitle="visual features",
        edge=BLUE,
        face="#eaf5fb",
        fontsize=11.2,
    )
    draw_round_box(
        ax,
        xy=(3.15, 1.55),
        width=1.80,
        height=1.05,
        label="Text encoder",
        subtitle="token embeddings",
        edge=ORANGE,
        face="#fff4dd",
        fontsize=11.2,
    )

    draw_shared_token_space(ax)
    draw_round_box(
        ax,
        xy=(8.70, 2.02),
        width=2.05,
        height=1.45,
        label="Qwen3 LM\nDecoder",
        subtitle=None,
        edge=PURPLE,
        face="#f1effc",
        fontsize=12.0,
    )
    ax.text(
        9.73,
        2.19,
        "autoregressive generation",
        ha="center",
        va="center",
        fontsize=8.5,
        color=MUTED,
        zorder=7,
    )
    draw_round_box(
        ax,
        xy=(11.25, 2.15),
        width=0.85,
        height=1.18,
        label="Text\noutput",
        subtitle=None,
        edge=GREEN,
        face="#edf8f3",
        fontsize=10.4,
    )

    draw_token_strip(ax, x=3.45, y=3.17, colors=[BLUE, BLUE, BLUE], label="vision tokens")
    draw_token_strip(ax, x=3.50, y=1.31, colors=[ORANGE, ORANGE, ORANGE], label="text tokens")

    draw_arrow(ax, (2.60, 4.00), (3.15, 3.98), color=BLUE)
    draw_arrow(ax, (2.60, 2.08), (3.15, 2.08), color=ORANGE)
    draw_arrow(ax, (4.95, 3.92), (5.42, 3.08), color=BLUE, rad=-0.12)
    draw_arrow(ax, (4.95, 2.08), (5.42, 2.42), color=ORANGE, rad=0.12)
    draw_arrow(ax, (8.12, 2.75), (8.70, 2.75), color=GREEN)
    draw_arrow(ax, (10.75, 2.75), (11.25, 2.75), color=GREEN)

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
    parser = argparse.ArgumentParser(description="Plot a simplified Qwen3-VL architecture diagram.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="PNG output path.")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="PDF output path. Pass an empty string to skip.")
    parser.add_argument("--svg", default=DEFAULT_SVG, help="SVG output path. Pass an empty string to skip.")
    args = parser.parse_args()

    build_figure(
        Path(args.output),
        pdf=Path(args.pdf) if args.pdf else None,
        svg=Path(args.svg) if args.svg else None,
    )


if __name__ == "__main__":
    main()
