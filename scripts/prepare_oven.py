#!/usr/bin/env python3
"""Prepare OVEN validation data for vlm-eval compatibility.

This bridges the schema gap between OVEN's raw download format and what
vlm-eval's load_oven() expects. Specifically, it adds three fields that
are missing from the raw OVEN JSONL:

  - ``answer``: a copy of ``entity_text`` (load_oven reads this key)
  - ``dataset``: the source dataset name, looked up in ovenid2impath.csv
  - ``image_path``: the on-disk location of the image

Usage::

    uv run python scripts/prepare_oven.py \
        --oven-val data/raw/oven_entity_val.jsonl \
        --id2path data/raw/ovenid2impath.csv \
        --image-root data/images \
        --output data/processed/vlm_compatible_val.jsonl
"""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare OVEN validation JSONL")
    parser.add_argument("--oven-val", required=True, help="Path to oven_entity_val.jsonl")
    parser.add_argument("--id2path", required=True, help="Path to ovenid2impath.csv")
    parser.add_argument("--image-root", default="data/images", help="Root directory for images")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--exclude-inat", action="store_true", help="Drop iNaturalist rows")
    args = parser.parse_args()

    # Load image_id → source dataset name
    id2dataset = {}
    id2filename = {}
    with open(args.id2path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header if present
        for row in reader:
            if len(row) < 2:
                continue
            image_id = row[0]
            rel_path = row[1]
            # Dataset is the first path component, e.g. "oven/images_stratified/oven_00000.jpg"
            parts = rel_path.split("/")
            dataset_name = parts[0] if parts else "unknown"
            id2dataset[image_id] = dataset_name
            id2filename[image_id] = rel_path

    # Process rows
    kept = 0
    skipped = 0
    with open(args.oven_val, "r") as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)

            # Add the three missing fields
            r["answer"] = r["entity_text"]

            image_id = r.get("image_id", "")
            r["dataset"] = id2dataset.get(image_id, "unknown")

            # Set image_path to the actual on-disk location
            rel_file = id2filename.get(image_id, f"{image_id}.jpg")
            # Try to find the actual file; fallback to image_id.jpg
            possible_path = Path(args.image_root) / rel_file
            if possible_path.exists():
                r["image_path"] = str(possible_path)
            else:
                # Fallback: assume images are named by image_id under image-root
                fallback = Path(args.image_root) / f"{image_id}.jpg"
                r["image_path"] = str(fallback)

            # Optionally filter out iNaturalist
            if args.exclude_inat and r["dataset"] == "inaturalist":
                skipped += 1
                continue

            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Done: {kept} rows written, {skipped} rows skipped.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
