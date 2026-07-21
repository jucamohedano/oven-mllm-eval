#!/usr/bin/env python3
"""Render a 3x3 grid of one representative image per OVEN source dataset.

For a presentation slide showing what the OVEN validation split looks like.
Each cell is one image, captioned with the dataset name and the entity label.

Images are resolved by *dataset membership* (from the aligned val JSONL), not by
hardcoded image_id, so the grid adapts to whatever is present in --image-dir. A
small curated preference list picks a recognizable example per dataset when it is
available locally; otherwise any local image of that dataset is used.

Usage::

    python analysis/plot_dataset_grid.py \
      --image-dir data/images \
      --out viz/dataset_grid/oven_dataset_grid.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# The 9 OVEN source datasets, in the grid order (row-major), with a display name.
DATASETS = [
    ("imagenet21k", "ImageNet21k-P"),
    ("gldv2", "Google Landmarks v2"),
    ("inaturalist", "iNaturalist2017"),
    ("car196", "Cars196"),
    ("aircraft", "FGVC-Aircraft"),
    ("sun397", "SUN397"),
    ("food101", "Food101"),
    ("sports100", "Sports100"),
    ("oxford_flower", "Oxford Flowers"),
]

# Preferred recognizable example per dataset (used if present locally).
PREFERRED = {
    "gldv2": "oven_05021957",       # 30 St Mary Axe (the Gherkin)
    "inaturalist": "oven_04970386",  # Eastern box turtle
    "car196": "oven_04950780",       # McLaren 12C
    "aircraft": "oven_00002959",     # Boeing
}


def resolve_image(image_id: str, image_dir: Path) -> Path | None:
    for ext in (".jpg", ".jpeg", ".JPEG", ".JPG", ".png"):
        p = image_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def build_index(aligned: Path, image_dir: Path) -> dict[str, list[tuple[str, str]]]:
    """dataset -> [(image_id, label), ...] restricted to locally-present images."""
    by_ds: dict[str, list[tuple[str, str]]] = {ds: [] for ds, _ in DATASETS}
    seen: set[str] = set()
    with aligned.open() as fh:
        for line in fh:
            r = json.loads(line)
            ds = r.get("dataset")
            iid = r.get("image_id", "")
            if ds not in by_ds or iid in seen:
                continue
            if resolve_image(iid, image_dir) is None:
                continue
            seen.add(iid)
            label = r.get("entity_text") or r.get("answer") or ""
            by_ds[ds].append((iid, label))
    return by_ds


def pick(ds: str, candidates: list[tuple[str, str]]) -> tuple[str, str] | None:
    if not candidates:
        return None
    pref = PREFERRED.get(ds)
    if pref:
        for iid, label in candidates:
            if iid == pref:
                return iid, label
    return candidates[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="3x3 grid of one image per OVEN dataset")
    ap.add_argument("--aligned", default="data/processed/vlm_compatible_val_aligned.jsonl")
    ap.add_argument("--image-dir", default="data/images")
    ap.add_argument("--out", default="viz/dataset_grid/oven_dataset_grid.png")
    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    by_ds = build_index(Path(args.aligned), image_dir)

    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    missing = []
    for ax, (ds, display) in zip(axes.flat, DATASETS):
        chosen = pick(ds, by_ds[ds])
        ax.axis("off")
        if chosen is None:
            missing.append(ds)
            ax.set_title(f"{display}\n(no local image)", fontsize=12)
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="0.92"))
            continue
        iid, label = chosen
        img = plt.imread(resolve_image(iid, image_dir))
        ax.imshow(img)
        ax.set_title(display, fontsize=13, fontweight="bold", pad=4)
        ax.text(0.5, -0.06, label, transform=ax.transAxes, ha="center", va="top",
                fontsize=11, style="italic")

    fig.suptitle("OVEN validation split: nine source datasets", fontsize=16, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=4.0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[saved] {out}")
    if missing:
        print(f"[warn] no local image for: {', '.join(missing)} "
              f"(pull with scripts/sync.sh --pull-images)")


if __name__ == "__main__":
    main()
