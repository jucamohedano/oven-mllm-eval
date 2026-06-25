#!/usr/bin/env python3
"""Deduplicate a samples JSONL file by data_id, keeping the first occurrence.

Usage:
    uv run python scripts/dedup_samples.py path/to/run_samples.jsonl
    uv run python scripts/dedup_samples.py path/to/run_samples.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def dedup(path: Path, dry_run: bool = False, drop_malformed: bool = False) -> None:
    lines: list[str] = []
    seen: set[str] = set()
    duplicates = 0
    malformed = 0

    with open(path) as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                malformed += 1
                if not drop_malformed:
                    lines.append(raw)  # default: keep unparseable lines as-is
                continue
            key = row.get("data_id") or row.get("image_id") or ""
            if key and key in seen:
                duplicates += 1
            else:
                if key:
                    seen.add(key)
                lines.append(raw)

    dropped = malformed if drop_malformed else 0
    total_in = len(lines) + duplicates + dropped
    print(f"{path}: {total_in} rows → {len(lines)} kept "
          f"({duplicates} duplicates removed, {dropped} malformed dropped)")

    if duplicates == 0 and dropped == 0:
        print("Nothing to remove — file unchanged.")
        return

    if dry_run:
        print("Dry run — no changes written.")
        return

    tmp = path.with_suffix(".jsonl.dedup_tmp")
    with open(tmp, "w") as f:
        for line in lines:
            f.write(line + "\n")
    tmp.replace(path)
    print(f"Written: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate a samples JSONL by data_id")
    parser.add_argument("path", type=Path, help="Path to samples JSONL file")
    parser.add_argument("--dry-run", action="store_true", help="Report duplicates without writing")
    parser.add_argument("--drop-malformed", action="store_true",
                        help="Drop unparseable JSON lines (e.g. a truncated tail) instead of keeping them")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: {args.path} not found", file=sys.stderr)
        sys.exit(1)

    dedup(args.path, dry_run=args.dry_run, drop_malformed=args.drop_malformed)


if __name__ == "__main__":
    main()
