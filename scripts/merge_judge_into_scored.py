#!/usr/bin/env python3
"""Merge judge fields into an existing scored JSONL and update pass@k summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oven_mllm_eval.pass_at_k import pass_at_k


def _row_id(row: dict) -> str:
    return str(row.get("data_id") or row.get("image_id") or "")


def _judge_payload(row: dict) -> dict:
    return {key: value for key, value in row.items() if key.startswith("judge_")}


def _pass_metrics(judge_rows: list[dict]) -> dict:
    verdict_rows = [row for row in judge_rows if row.get("judge_verdicts")]
    if not verdict_rows:
        return {}

    n_values = [len(row["judge_verdicts"]) for row in verdict_rows]
    n_max = max(n_values)
    ks = sorted({k for k in [1, 2, 4, 8, 16, 32, 64, 128, n_max] if 0 < k <= n_max})

    metrics: dict[str, float | int] = {}
    for k in ks:
        values = []
        for row in verdict_rows:
            verdicts = row["judge_verdicts"]
            n = len(verdicts)
            if k <= n:
                values.append(pass_at_k(n, int(sum(verdicts)), k))
        if values:
            metrics[f"pass@{k}"] = sum(values) / len(values)

    if any(row.get("judge_verdicts_majority") for row in verdict_rows):
        for k in ks:
            values = []
            for row in verdict_rows:
                verdicts = row.get("judge_verdicts_majority")
                if not verdicts:
                    continue
                n = len(verdicts)
                if k <= n:
                    values.append(pass_at_k(n, int(sum(verdicts)), k))
            if values:
                metrics[f"pass@{k}_majority"] = sum(values) / len(values)

    parse_rows = [row for row in judge_rows if row.get("judge_parse_ok") is not None]
    if parse_rows:
        metrics["num_judge_rollouts"] = sum(len(row.get("judge_parse_ok") or []) for row in parse_rows)
        metrics["num_judge_unparseable"] = sum(
            sum(1 for ok in (row.get("judge_parse_ok") or []) if not ok)
            for row in parse_rows
        )

    return metrics


def _update_summary(summary_path: Path, metrics: dict) -> None:
    if not summary_path.exists():
        return
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if "measures" in data:
        for entry in data["measures"]:
            entry.setdefault("metrics", {}).update(metrics)
    else:
        data.update(metrics)

    tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge judge_* fields into an existing scored JSONL."
    )
    parser.add_argument("--judged", required=True, help="Judged JSONL from run_judge.py")
    parser.add_argument("--scored", required=True, help="Existing scored JSONL to update")
    parser.add_argument("--output", default=None, help="Output JSONL. Default: overwrite --scored")
    parser.add_argument("--summary", default=None, help="Existing results JSON to update with pass@k")
    args = parser.parse_args()

    judged_path = Path(args.judged)
    scored_path = Path(args.scored)
    output_path = Path(args.output) if args.output else scored_path

    if not judged_path.exists():
        parser.error(f"--judged file not found: {judged_path}")
    if not scored_path.exists():
        parser.error(f"--scored file not found: {scored_path}")

    judge_by_id: dict[str, dict] = {}
    judge_rows_for_summary: list[dict] = []
    with judged_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = _row_id(row)
            if not rid:
                continue
            payload = _judge_payload(row)
            judge_by_id[rid] = payload
            judge_rows_for_summary.append(payload)

    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    total = 0
    merged = 0
    with scored_path.open("r", encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            payload = judge_by_id.get(_row_id(row))
            if payload is not None:
                row.update(payload)
                merged += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(output_path)

    metrics = _pass_metrics(judge_rows_for_summary)
    if args.summary and metrics:
        _update_summary(Path(args.summary), metrics)

    print(f"[merge] merged judge fields into {merged:,}/{total:,} scored rows")
    if args.summary and metrics:
        print(f"[merge] updated pass@k/judge stats in {args.summary}")


if __name__ == "__main__":
    main()
