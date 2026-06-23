#!/usr/bin/env python3
"""Repair strided inference shard files without losing completed samples.

This script is for interrupted runs whose ``*_samples.jsonl`` and
``*_samples_shard*.jsonl`` files are inconsistent, usually after changing the
data-parallel shard count across resume attempts.

It collects completed sample rows from the existing files, deduplicates by
``data_id``, then rewrites canonical shard files according to the target
``--dp`` and the dataset order in ``--input``.  Dry-run is the default; pass
``--apply`` to move old files into a backup directory and write repaired shards.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                yield line_no, None


def get_data_id(row: dict[str, Any]) -> str | None:
    data_id = row.get("data_id")
    return str(data_id) if data_id else None


def rollout_count(row: dict[str, Any]) -> int:
    all_texts = row.get("all_texts")
    if isinstance(all_texts, list):
        return len(all_texts)
    return 0


def is_complete(row: dict[str, Any], expected_rollouts: int) -> bool:
    if not get_data_id(row):
        return False
    if expected_rollouts <= 0:
        return True
    return rollout_count(row) >= expected_rollouts


def discover_sample_files(run_dir: Path, run_id: str) -> list[Path]:
    files: list[Path] = []
    merged = run_dir / f"{run_id}_samples.jsonl"
    if merged.exists():
        files.append(merged)
    files.extend(sorted(run_dir.glob(f"{run_id}_samples_shard*.jsonl")))
    return files


def load_input_ids(input_path: Path) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for line_no, row in read_jsonl(input_path):
        if row is None:
            raise ValueError(f"Malformed JSON in input at {input_path}:{line_no}")
        data_id = get_data_id(row)
        if not data_id:
            raise ValueError(f"Missing data_id in input at {input_path}:{line_no}")
        if data_id in seen:
            raise ValueError(f"Duplicate data_id in input: {data_id}")
        seen.add(data_id)
        ids.append(data_id)
    return ids


def collect_completed_rows(
    files: list[Path],
    valid_ids: set[str],
    expected_rollouts: int,
) -> tuple[dict[str, dict[str, Any]], Counter]:
    rows: dict[str, dict[str, Any]] = {}
    stats: Counter = Counter()

    for path in files:
        for _, row in read_jsonl(path):
            stats["rows_seen"] += 1
            if row is None:
                stats["malformed"] += 1
                continue
            if not isinstance(row, dict):
                stats["non_object"] += 1
                continue

            data_id = get_data_id(row)
            if not data_id:
                stats["missing_data_id"] += 1
                continue
            if data_id not in valid_ids:
                stats["outside_input"] += 1
                continue
            if not is_complete(row, expected_rollouts):
                stats["incomplete"] += 1
                continue

            previous = rows.get(data_id)
            if previous is not None:
                stats["duplicates"] += 1
                if rollout_count(row) <= rollout_count(previous):
                    continue
            rows[data_id] = row

    stats["completed_unique"] = len(rows)
    return rows, stats


def shard_counts(input_ids: list[str], completed: dict[str, dict[str, Any]], dp: int):
    counts = []
    for shard in range(dp):
        expected = len(input_ids[shard::dp])
        done = sum(1 for data_id in input_ids[shard::dp] if data_id in completed)
        counts.append((shard, expected, done, expected - done))
    return counts


def backup_existing(files: list[Path], backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=False)
    for path in files:
        shutil.move(str(path), str(backup_dir / path.name))


def write_repaired_shards(
    run_dir: Path,
    run_id: str,
    input_ids: list[str],
    completed: dict[str, dict[str, Any]],
    dp: int,
) -> list[Path]:
    outputs = [run_dir / f"{run_id}_samples_shard{shard}.jsonl" for shard in range(dp)]
    handles = [path.open("w", encoding="utf-8") for path in outputs]
    try:
        for index, data_id in enumerate(input_ids):
            row = completed.get(data_id)
            if row is None:
                continue
            shard = index % dp
            handles[shard].write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        for handle in handles:
            handle.close()
    return outputs


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    source_files: list[Path],
    stats: Counter,
    counts,
    backup_dir: Path | None,
) -> None:
    manifest = {
        "run_dir": str(args.run_dir),
        "run_id": args.run_id,
        "input": str(args.input),
        "dp": args.dp,
        "expected_rollouts": args.expected_rollouts,
        "source_files": [str(path) for path in source_files],
        "stats": dict(stats),
        "shards": [
            {"shard": shard, "expected": expected, "completed": done, "remaining": remaining}
            for shard, expected, done, remaining in counts
        ],
        "backup_dir": str(backup_dir) if backup_dir else None,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dp", type=int, required=True, help="Target data-parallel shard count")
    parser.add_argument("--run-id", default=None, help="Defaults to the run directory name")
    parser.add_argument(
        "--expected-rollouts",
        type=int,
        default=256,
        help="Require at least this many all_texts entries per row; use 0 to disable",
    )
    parser.add_argument("--apply", action="store_true", help="Move old files and write repaired shards")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.input = args.input.resolve()
    args.run_id = args.run_id or args.run_dir.name

    if args.dp < 1:
        raise SystemExit("--dp must be >= 1")
    if not args.run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {args.run_dir}")
    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    input_ids = load_input_ids(args.input)
    source_files = discover_sample_files(args.run_dir, args.run_id)
    if not source_files:
        raise SystemExit(f"No sample files found for run_id={args.run_id} in {args.run_dir}")

    completed, stats = collect_completed_rows(source_files, set(input_ids), args.expected_rollouts)
    counts = shard_counts(input_ids, completed, args.dp)

    print(f"run_dir: {args.run_dir}")
    print(f"run_id:  {args.run_id}")
    print(f"input examples: {len(input_ids)}")
    print("source files:")
    for path in source_files:
        print(f"  - {path.name}")
    print("stats:")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print("target shards:")
    for shard, expected, done, remaining in counts:
        print(f"  shard{shard}: completed={done} expected={expected} remaining={remaining}")

    if not args.apply:
        print("dry-run only; pass --apply to rewrite shard files")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = args.run_dir / f"repair_backup_{timestamp}"
    backup_existing(source_files, backup_dir)
    outputs = write_repaired_shards(args.run_dir, args.run_id, input_ids, completed, args.dp)
    manifest_path = args.run_dir / f"{args.run_id}_repair_manifest_{timestamp}.json"
    write_manifest(manifest_path, args, source_files, stats, counts, backup_dir)

    print(f"moved old sample files to: {backup_dir}")
    print("wrote repaired shards:")
    for path in outputs:
        print(f"  - {path.name}")
    print(f"wrote manifest: {manifest_path.name}")


if __name__ == "__main__":
    main()
