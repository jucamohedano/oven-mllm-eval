#!/usr/bin/env python3
"""Audit judge-positive rollouts against ground-truth labels.

The judge is allowed to accept semantic equivalents, so this script does not
claim every non-exact positive is wrong. It reports judge positives supported
by exact match, alias match, or containment only.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from oven_mllm_eval.judge_audit import build_alias_map, classify_positive
from oven_mllm_eval.pass_at_k import pass_at_k


def load_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_scored_inputs(paths: list[str]) -> list[Path]:
    scored: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            scored.append(path)
        elif path.is_dir():
            scored.extend(sorted(path.glob("*_scored.jsonl")))
            scored.extend(sorted(path.glob("*_samples_scored.jsonl")))
        else:
            raise FileNotFoundError(raw)
    return sorted(set(scored))


def audit_file(
    scored_path: Path,
    *,
    index: dict[str, Any],
) -> dict[str, Any]:
    aliases_by_canonical = build_alias_map(index)

    counts: Counter[str] = Counter()
    total_rows = 0
    judge_pass1_values: list[float] = []
    supported_pass1_values: list[float] = []
    judge_pass_full_values: list[float] = []
    supported_pass_full_values: list[float] = []

    for row in load_jsonl(scored_path):
        total_rows += 1
        answer = row.get("answer", "")
        texts = row.get("all_texts", [])
        verdicts = row.get("judge_verdicts", [])
        judge_positive_count = 0
        supported_positive_count = 0

        for text, verdict in zip(texts, verdicts):
            if not verdict:
                continue
            judge_positive_count += 1

            category = classify_positive(
                prediction=text,
                answer=answer,
                aliases_by_canonical=aliases_by_canonical,
            )
            counts["judge_positive_rollouts"] += 1
            if category is not None:
                counts[category] += 1
                supported_positive_count += 1

        if verdicts:
            n = len(verdicts)
            judge_pass1_values.append(pass_at_k(n, judge_positive_count, 1))
            supported_pass1_values.append(pass_at_k(n, supported_positive_count, 1))
            judge_pass_full_values.append(pass_at_k(n, judge_positive_count, n))
            supported_pass_full_values.append(pass_at_k(n, supported_positive_count, n))
            if judge_positive_count:
                counts["judge_hit_examples"] += 1
                if supported_positive_count:
                    counts["supported_hit_examples"] += 1

    positives = counts["judge_positive_rollouts"]
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    summary = {
        "scored_path": str(scored_path),
        "rows": total_rows,
        "judge_hit_examples": counts["judge_hit_examples"],
        "supported_hit_examples": counts["supported_hit_examples"],
        "judge_pass@1": mean(judge_pass1_values),
        "supported_pass@1": mean(supported_pass1_values),
        "judge_pass@full": mean(judge_pass_full_values),
        "supported_pass@full": mean(supported_pass_full_values),
        "judge_positive_rollouts": positives,
        "supported_positive_rollouts": (
            counts["exact"]
            + counts["alias"]
            + counts["contains_answer"]
            + counts["answer_contains_prediction"]
        ),
        "exact": counts["exact"],
        "alias": counts["alias"],
        "contains_answer": counts["contains_answer"],
        "answer_contains_prediction": counts["answer_contains_prediction"],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure judge-positive rollouts supported by exact/alias/containment checks."
    )
    parser.add_argument("inputs", nargs="+", help="Run dirs or *_scored.jsonl files")
    parser.add_argument(
        "--taxonomy-index",
        default="data/processed/oven_taxonomy_index.json",
        help="Taxonomy index with labels, aliases, and entity paths",
    )
    args = parser.parse_args()

    index = json.loads(Path(args.taxonomy_index).read_text())
    scored_files = find_scored_inputs(args.inputs)
    if not scored_files:
        raise SystemExit("No *_scored.jsonl files found")

    import sys

    fieldnames = [
        "scored_path",
        "rows",
        "judge_hit_examples",
        "supported_hit_examples",
        "judge_pass@1",
        "supported_pass@1",
        "judge_pass@full",
        "supported_pass@full",
        "judge_positive_rollouts",
        "supported_positive_rollouts",
        "exact",
        "alias",
        "contains_answer",
        "answer_contains_prediction",
    ]
    out_writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, delimiter="\t")
    out_writer.writeheader()
    for scored_path in scored_files:
        summary = audit_file(
            scored_path,
            index=index,
        )
        out_writer.writerow(summary)


if __name__ == "__main__":
    main()
