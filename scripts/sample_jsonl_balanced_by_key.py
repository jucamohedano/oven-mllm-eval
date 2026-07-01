#!/usr/bin/env python3
"""Sample a JSONL file as evenly as possible across a key such as entity_id."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-rows", required=True, type=int)
    parser.add_argument("--key", default="entity_id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-output", action="store_true")
    parser.add_argument("--manifest", type=Path, help="Optional manifest JSON path.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_rows <= 0:
        raise SystemExit("--max-rows must be positive")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    if args.manifest and args.manifest.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing manifest: {args.manifest}")
    return args


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def count_keys(path: Path, key_name: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    skipped = 0
    for row in iter_jsonl(path):
        key = str(row.get(key_name) or "")
        if key:
            counts[key] += 1
        else:
            skipped += 1
    if skipped:
        print(f"skipped rows without {key_name}: {skipped}")
    return counts


def allocate_targets(counts: Counter[str], max_rows: int, seed: int) -> dict[str, int]:
    keys = sorted(counts)
    if not keys:
        raise SystemExit("no rows with requested key")

    total_available = sum(counts.values())
    target_total = min(max_rows, total_available)
    base = target_total // len(keys)
    targets = {key: min(counts[key], base) for key in keys}
    remaining = target_total - sum(targets.values())

    rng = random.Random(seed)
    while remaining > 0:
        expandable = [key for key in keys if targets[key] < counts[key]]
        if not expandable:
            break
        rng.shuffle(expandable)
        for key in expandable:
            if remaining <= 0:
                break
            targets[key] += 1
            remaining -= 1
    return {key: value for key, value in targets.items() if value > 0}


def reservoir_sample(
    path: Path,
    key_name: str,
    targets: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: Counter[str] = Counter()
    rngs = {key: random.Random(stable_seed(seed, key)) for key in targets}

    for row in iter_jsonl(path):
        key = str(row.get(key_name) or "")
        target = targets.get(key, 0)
        if target <= 0:
            continue
        seen[key] += 1
        bucket = samples[key]
        if len(bucket) < target:
            bucket.append(row)
            continue
        replace_at = rngs[key].randrange(seen[key])
        if replace_at < target:
            bucket[replace_at] = row

    selected: list[dict[str, Any]] = []
    for key in sorted(samples):
        selected.extend(samples[key])
    return selected


def summarize(selected: list[dict[str, Any]], key_name: str) -> dict[str, Any]:
    counts = Counter(str(row.get(key_name) or "") for row in selected)
    values = list(counts.values())
    return {
        "rows": len(selected),
        "keys": len(counts),
        "min_rows_per_key": min(values) if values else 0,
        "max_rows_per_key": max(values) if values else 0,
        "mean_rows_per_key": (sum(values) / len(values)) if values else 0,
    }


def main() -> None:
    args = parse_args()
    counts = count_keys(args.input, args.key)
    targets = allocate_targets(counts, args.max_rows, args.seed)
    selected = reservoir_sample(args.input, args.key, targets, args.seed)

    if args.shuffle_output:
        random.Random(args.seed).shuffle(selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "key": args.key,
        "seed": args.seed,
        "requested_rows": args.max_rows,
        "available_rows": sum(counts.values()),
        "available_keys": len(counts),
        "targeted_keys": len(targets),
        **summarize(selected, args.key),
    }
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
