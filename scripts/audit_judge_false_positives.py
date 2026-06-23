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
import re
from collections import Counter
from pathlib import Path
from typing import Any


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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


def pass_at_k(n: int, c: int, k: int) -> float:
    if n <= 0 or c <= 0:
        return 0.0
    if k >= n:
        return 1.0
    if n - c < k:
        return 1.0

    miss_prob = 1.0
    for i in range(k):
        miss_prob *= (n - c - i) / (n - i)
    return 1.0 - miss_prob


def build_alias_map(index: dict[str, Any]) -> dict[str, set[str]]:
    aliases_by_canonical: dict[str, set[str]] = {}
    for alias_norm, canonical in index.get("aliases", {}).items():
        aliases_by_canonical.setdefault(normalize(canonical), set()).add(alias_norm)
    return aliases_by_canonical


def classify_positive(
    *,
    prediction: str,
    answer: str,
    aliases_by_canonical: dict[str, set[str]],
) -> str | None:
    pred_norm = normalize(prediction)
    answer_norm = normalize(answer)

    if pred_norm == answer_norm:
        return "exact"

    if pred_norm in aliases_by_canonical.get(answer_norm, set()):
        return "alias"

    if answer_norm and answer_norm in pred_norm:
        return "contains_answer"

    if pred_norm and pred_norm in answer_norm:
        return "answer_contains_prediction"

    return None


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
