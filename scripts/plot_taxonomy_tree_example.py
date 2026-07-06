#!/usr/bin/env python3
"""Draw a small taxonomy-tree example for the thesis.

The figure shows two root-to-leaf chains that share the same domain and
immediate parent, plus one chain from a different domain. Entity IDs and image
IDs are fixed from the local OVEN validation metadata so the plot is stable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


BG = "#fffdfa"
INK = "#2f2f2f"
MUTED = "#6b675f"
GRID = "#ded9ce"
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"


@dataclass(frozen=True)
class LeafInfo:
    label: str
    entity_id: str
    image_id: str
    color: str
    y: float


NODE_HALF_WIDTH = {
    "root": 0.24,
    "work": 0.30,
    "architectural": 0.54,
    "building": 0.39,
    "stadium": 0.37,
    "baseball": 0.45,
    "entity": 0.29,
    "object": 0.30,
    "physical": 0.70,
    "organisms": 0.64,
    "taxon": 0.31,
}


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


def draw_box(
    ax,
    *,
    x: float,
    y: float,
    text: str,
    color: str,
    width: float = 0.92,
    height: float = 0.22,
    fontsize: float = 9.5,
    face_alpha: float = 0.09,
    weight: str = "normal",
    zorder: int = 3,
) -> None:
    import matplotlib.patheffects as pe

    patch = ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=INK,
        zorder=zorder,
        bbox={
            "boxstyle": "round,pad=0.34,rounding_size=0.08",
            "facecolor": color,
            "edgecolor": color,
            "alpha": face_alpha,
            "linewidth": 1.4,
        },
    )
    patch.set_path_effects(
        [
            pe.withSimplePatchShadow(
                offset=(1.3, -1.3),
                shadow_rgbFace=(0.82, 0.78, 0.70),
                alpha=0.18,
            )
        ]
    )


def center_crop_square(image):
    height, width = image.shape[:2]
    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    return image[y0 : y0 + side, x0 : x0 + side]


def draw_leaf_card_with_image(
    ax,
    *,
    left: float,
    leaf: LeafInfo,
    image_dir: Path,
    width: float = 3.05,
    height: float = 0.68,
) -> None:
    import matplotlib.image as mpimg
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyBboxPatch, Rectangle

    bottom = leaf.y - height / 2
    card = FancyBboxPatch(
        (left, bottom),
        width,
        height,
        boxstyle="round,pad=0.035,rounding_size=0.065",
        facecolor="#ffffff",
        edgecolor=leaf.color,
        linewidth=1.8,
        alpha=0.98,
        zorder=5,
    )
    card.set_path_effects(
        [
            pe.withSimplePatchShadow(
                offset=(1.7, -1.7),
                shadow_rgbFace=(0.74, 0.70, 0.62),
                alpha=0.22,
            )
        ]
    )
    ax.add_patch(card)

    pad = 0.08
    thumb = height - 2 * pad
    thumb_left = left + pad
    thumb_bottom = leaf.y - thumb / 2
    image_path = image_dir / f"{leaf.image_id}.jpg"
    if image_path.exists():
        image = center_crop_square(mpimg.imread(image_path))
        ax.imshow(
            image,
            extent=(thumb_left, thumb_left + thumb, thumb_bottom, thumb_bottom + thumb),
            zorder=6,
            interpolation="lanczos",
        )
    else:
        ax.add_patch(
            Rectangle(
                (thumb_left, thumb_bottom),
                thumb,
                thumb,
                facecolor="#f7f1e8",
                edgecolor="none",
                zorder=6,
            )
        )
        ax.text(
            thumb_left + thumb / 2,
            leaf.y,
            "image\nmissing",
            ha="center",
            va="center",
            fontsize=6.8,
            color=MUTED,
            zorder=7,
        )
    ax.add_patch(
        Rectangle(
            (thumb_left, thumb_bottom),
            thumb,
            thumb,
            facecolor="none",
            edgecolor=leaf.color,
            linewidth=1.0,
            zorder=7,
        )
    )

    text_left = thumb_left + thumb + 0.14
    ax.text(
        text_left,
        leaf.y + 0.13,
        leaf.label,
        ha="left",
        va="center",
        fontsize=10.1,
        color=INK,
        zorder=7,
    )
    ax.text(
        text_left,
        leaf.y - 0.13,
        f"{leaf.entity_id}  |  {leaf.image_id}",
        ha="left",
        va="center",
        fontsize=9.3,
        color=MUTED,
        zorder=7,
    )


def draw_edge(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    rad: float = 0.0,
    linewidth: float = 2.2,
    alpha: float = 0.9,
) -> None:
    from matplotlib.patches import FancyArrowPatch

    edge = FancyArrowPatch(
        start,
        end,
        arrowstyle="-",
        connectionstyle=f"arc3,rad={rad}",
        linewidth=linewidth,
        color=color,
        alpha=alpha,
        zorder=1,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_patch(edge)


def draw_badge(ax, *, x: float, y: float, text: str, color: str) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=8.6,
        fontweight="bold",
        color=color,
        bbox={
            "boxstyle": "round,pad=0.28,rounding_size=0.07",
            "facecolor": BG,
            "edgecolor": color,
            "linewidth": 1.2,
        },
        zorder=4,
    )


def build_figure(
    output: Path,
    *,
    pdf: Path | None,
    svg: Path | None,
    with_title: bool,
    image_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    set_style()

    fig, ax = plt.subplots(figsize=(14.0, 6.4))
    ax.set_xlim(-0.35, 11.0)
    ax.set_ylim(-0.25, 4.15 if with_title else 3.76)
    ax.axis("off")

    # Subtle branch panels.
    ax.add_patch(
        Rectangle(
            (0.75, 1.68),
            9.85,
            2.05,
            facecolor=BLUE,
            alpha=0.035,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (0.75, 0.08),
            9.85,
            1.15,
            facecolor=GREEN,
            alpha=0.035,
            edgecolor="none",
            zorder=0,
        )
    )

    if with_title:
        ax.text(
            0.0,
            3.95,
            "Example OVEN Taxonomy Chains",
            ha="left",
            va="top",
            fontsize=18,
            fontweight="bold",
            color=INK,
        )
        ax.text(
            0.0,
            3.68,
            "Two leaves share the same domain and parent; the third follows a different domain.",
            ha="left",
            va="top",
            fontsize=11,
            color=MUTED,
        )

    # Node coordinates.
    nodes: dict[str, tuple[float, float, str, str]] = {
        "root": (0.45, 2.12, "root", INK),
        "work": (1.55, 2.70, "work", BLUE),
        "architectural": (2.85, 2.70, "architectural\nstructure", BLUE),
        "building": (4.05, 2.70, "building", BLUE),
        "stadium": (5.05, 2.70, "stadium", BLUE),
        "baseball": (6.20, 2.70, "baseball\nvenue", BLUE),
        "entity": (1.55, 0.68, "entity", GREEN),
        "object": (2.55, 0.68, "object", GREEN),
        "physical": (3.85, 0.68, "group/class of\nphysical objects", GREEN),
        "organisms": (5.35, 0.68, "group/class of\norganisms", GREEN),
        "taxon": (6.55, 0.68, "taxon", GREEN),
    }

    def left_edge(key: str) -> tuple[float, float]:
        x, y = nodes[key][:2]
        return (x - NODE_HALF_WIDTH[key], y)

    def right_edge(key: str) -> tuple[float, float]:
        x, y = nodes[key][:2]
        return (x + NODE_HALF_WIDTH[key], y)

    # Edges: shared branch and different-domain branch. Coordinates terminate at
    # approximate box/card boundaries so the lines do not run through labels.
    shared = ["root", "work", "architectural", "building", "stadium", "baseball"]
    other = ["root", "entity", "object", "physical", "organisms", "taxon"]
    for a, b in zip(shared, shared[1:]):
        start = (nodes["root"][0] + NODE_HALF_WIDTH["root"], nodes["root"][1] + 0.05) if a == "root" else right_edge(a)
        draw_edge(ax, start, left_edge(b), color=BLUE, rad=0.08 if a == "root" else 0.0)
    for a, b in zip(other, other[1:]):
        start = (nodes["root"][0] + 0.08, nodes["root"][1] - 0.14) if a == "root" else right_edge(a)
        draw_edge(ax, start, left_edge(b), color=GREEN, rad=-0.08 if a == "root" else 0.0)

    leaves = [
        LeafInfo("Nationals Park", "Q517545", "oven_04944518", BLUE, 3.10),
        LeafInfo("Fenway Park", "Q49136", "oven_04951065", BLUE, 2.28),
        LeafInfo("Greater Antillean Grackle", "Q577270", "oven_04967883", GREEN, 0.68),
    ]
    leaf_card_left = 7.56
    for leaf in leaves[:2]:
        offset = 0.08 if leaf.y > nodes["baseball"][1] else -0.08
        draw_edge(ax, (right_edge("baseball")[0], nodes["baseball"][1] + offset), (leaf_card_left, leaf.y), color=BLUE, rad=0.08)
    draw_edge(ax, right_edge("taxon"), (leaf_card_left, leaves[2].y), color=GREEN, rad=0.0)

    for key, (x, y, label, color) in nodes.items():
        draw_box(
            ax,
            x=x,
            y=y,
            text=label,
            color=color,
            width=1.0,
            fontsize=10 if key != "root" else 11,
            face_alpha=0.12 if key != "root" else 0.06,
            weight="bold" if key in {"root", "baseball", "taxon"} else "normal",
        )

    for leaf in leaves:
        draw_leaf_card_with_image(ax, left=leaf_card_left, leaf=leaf, image_dir=image_dir)

    draw_badge(ax, x=5.85, y=3.36, text="shared parent", color=BLUE)
    # Thin separators to make the branch split obvious without adding a grid.
    ax.plot([0.75, 10.6], [1.49, 1.49], color=GRID, linewidth=1.0, alpha=0.75)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
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
    parser = argparse.ArgumentParser(description="Plot a thesis taxonomy-tree example.")
    parser.add_argument(
        "--output",
        default="viz/taxonomy_tree/taxonomy_tree_example.png",
        help="PNG output path.",
    )
    parser.add_argument(
        "--pdf",
        default="viz/taxonomy_tree/taxonomy_tree_example.pdf",
        help="PDF output path. Pass an empty string to skip.",
    )
    parser.add_argument(
        "--svg",
        default="viz/taxonomy_tree/taxonomy_tree_example.svg",
        help="SVG output path. Pass an empty string to skip.",
    )
    parser.add_argument(
        "--with-title",
        action="store_true",
        help="Draw an embedded title and subtitle.",
    )
    parser.add_argument(
        "--image-dir",
        default="data/images",
        help="Directory containing OVEN image files named <image_id>.jpg.",
    )
    args = parser.parse_args()
    build_figure(
        Path(args.output),
        pdf=Path(args.pdf) if args.pdf else None,
        svg=Path(args.svg) if args.svg else None,
        with_title=args.with_title,
        image_dir=Path(args.image_dir),
    )


if __name__ == "__main__":
    main()
