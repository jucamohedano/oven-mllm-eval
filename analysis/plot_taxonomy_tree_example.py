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
    x: float
    y: float


NODE_HALF_HEIGHT = 0.16


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
    center_x: float,
    leaf: LeafInfo,
    image_dir: Path,
    width: float = 3.60,
    height: float = 1.12,
) -> None:
    import matplotlib.image as mpimg
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyBboxPatch, Rectangle

    left = center_x - width / 2
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

    pad = 0.11
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
            fontsize=7.2,
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
        leaf.y + 0.16,
        leaf.label,
        ha="left",
        va="center",
        fontsize=9.7,
        color=INK,
        zorder=7,
    )
    ax.text(
        text_left,
        leaf.y - 0.14,
        f"{leaf.entity_id}  |  {leaf.image_id}",
        ha="left",
        va="center",
        fontsize=8.2,
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

    fig, ax = plt.subplots(figsize=(12.4, 12.0 if with_title else 11.2))
    ax.set_xlim(0.0, 12.4)
    ax.set_ylim(-0.05, 11.6 if with_title else 10.95)
    ax.axis("off")

    # Subtle branch panels.
    ax.add_patch(
        Rectangle(
            (0.15, 0.35),
            7.95,
            8.10,
            facecolor=BLUE,
            alpha=0.035,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (8.20, 0.35),
            3.95,
            8.10,
            facecolor=GREEN,
            alpha=0.035,
            edgecolor="none",
            zorder=0,
        )
    )

    if with_title:
        ax.text(
            0.35,
            11.35,
            "Example OVEN Taxonomy Chains",
            ha="left",
            va="top",
            fontsize=18,
            fontweight="bold",
            color=INK,
        )
        ax.text(
            0.35,
            11.02,
            "Two leaves share the same domain and parent; the third follows a different domain.",
            ha="left",
            va="top",
            fontsize=11,
            color=MUTED,
        )

    # Node coordinates.
    nodes: dict[str, tuple[float, float, str, str]] = {
        "root": (6.25, 10.12, "root", INK),
        "work": (4.08, 8.98, "work", BLUE),
        "architectural": (4.08, 7.95, "architectural\nstructure", BLUE),
        "building": (4.08, 6.90, "building", BLUE),
        "stadium": (4.08, 5.85, "stadium", BLUE),
        "baseball": (4.08, 4.78, "baseball\nvenue", BLUE),
        "entity": (10.10, 8.98, "entity", GREEN),
        "object": (10.10, 7.95, "object", GREEN),
        "physical": (10.10, 6.90, "group/class of\nphysical objects", GREEN),
        "organisms": (10.10, 5.85, "group/class of\norganisms", GREEN),
        "taxon": (10.10, 4.78, "taxon", GREEN),
    }

    def top_edge(key: str) -> tuple[float, float]:
        x, y = nodes[key][:2]
        return (x, y + NODE_HALF_HEIGHT)

    def bottom_edge(key: str) -> tuple[float, float]:
        x, y = nodes[key][:2]
        return (x, y - NODE_HALF_HEIGHT)

    # Edges: shared branch and different-domain branch. Coordinates terminate at
    # approximate box/card boundaries so the lines do not run through labels.
    shared = ["root", "work", "architectural", "building", "stadium", "baseball"]
    other = ["root", "entity", "object", "physical", "organisms", "taxon"]
    for a, b in zip(shared, shared[1:]):
        start = bottom_edge(a)
        draw_edge(ax, start, top_edge(b), color=BLUE, rad=0.08 if a == "root" else 0.0)
    for a, b in zip(other, other[1:]):
        start = bottom_edge(a)
        draw_edge(ax, start, top_edge(b), color=GREEN, rad=-0.08 if a == "root" else 0.0)

    leaves = [
        LeafInfo("Nationals Park", "Q517545", "oven_04944518", BLUE, 2.15, 2.42),
        LeafInfo("Fenway Park", "Q49136", "oven_04951065", BLUE, 6.00, 2.42),
        LeafInfo("Greater Antillean Grackle", "Q577270", "oven_04967883", GREEN, 10.10, 2.42),
    ]
    for leaf in leaves[:2]:
        draw_edge(ax, bottom_edge("baseball"), (leaf.x, leaf.y + 0.56), color=BLUE, rad=0.10 if leaf.x < nodes["baseball"][0] else -0.10)
    draw_edge(ax, bottom_edge("taxon"), (leaves[2].x, leaves[2].y + 0.56), color=GREEN, rad=0.0)

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
        draw_leaf_card_with_image(ax, center_x=leaf.x, leaf=leaf, image_dir=image_dir)

    draw_badge(ax, x=4.08, y=3.62, text="shared parent", color=BLUE)
    # Thin separators to make the branch split obvious without adding a grid.
    ax.plot([8.15, 8.15], [0.35, 8.45], color=GRID, linewidth=1.0, alpha=0.75)

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
