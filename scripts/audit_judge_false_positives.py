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
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from oven_mllm_eval.judge_audit import (
    SUPPORTED_CATEGORIES,
    build_alias_map,
    classify_positive,
    is_supported,
)
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
    rollout_sizes: Counter[int] = Counter()
    judge_pos_per_hit: list[int] = []
    supported_pos_per_hit: list[int] = []

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
                if is_supported(category):
                    supported_positive_count += 1

        if verdicts:
            n = len(verdicts)
            rollout_sizes[n] += 1
            judge_pass1_values.append(pass_at_k(n, judge_positive_count, 1))
            supported_pass1_values.append(pass_at_k(n, supported_positive_count, 1))
            judge_pass_full_values.append(pass_at_k(n, judge_positive_count, n))
            supported_pass_full_values.append(pass_at_k(n, supported_positive_count, n))
            if judge_positive_count:
                counts["judge_hit_examples"] += 1
                judge_pos_per_hit.append(judge_positive_count)
                if supported_positive_count:
                    counts["supported_hit_examples"] += 1
                    supported_pos_per_hit.append(supported_positive_count)

    positives = counts["judge_positive_rollouts"]
    supported_positives = sum(counts[c] for c in SUPPORTED_CATEGORIES)
    under_specific = counts["answer_contains_prediction"]
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    summary = {
        "scored_path": str(scored_path),
        "run": run_label(scored_path),
        "rows": total_rows,
        "k": format_rollout_sizes(rollout_sizes),
        "judge_hit_examples": counts["judge_hit_examples"],
        "supported_hit_examples": counts["supported_hit_examples"],
        "judge_pass@1": mean(judge_pass1_values),
        "supported_pass@1": mean(supported_pass1_values),
        "judge_pass@k": mean(judge_pass_full_values),
        "supported_pass@k": mean(supported_pass_full_values),
        "judge_positive_rollouts": positives,
        "supported_positive_rollouts": supported_positives,
        "supported_positive_%": (100.0 * supported_positives / positives if positives else 0.0),
        "under_specific_rollouts": under_specific,
        "under_specific_%": (100.0 * under_specific / positives if positives else 0.0),
        "mean_judge_pos_per_hit": mean(judge_pos_per_hit),
        "median_judge_pos_per_hit": statistics.median(judge_pos_per_hit) if judge_pos_per_hit else 0.0,
        "mean_supported_pos_per_hit": mean(supported_pos_per_hit),
        "median_supported_pos_per_hit": (
            statistics.median(supported_pos_per_hit) if supported_pos_per_hit else 0.0
        ),
        "exact": counts["exact"],
        "alias": counts["alias"],
        "contains_answer": counts["contains_answer"],
        "contains_alias": counts["contains_alias"],
        "answer_contains_prediction": counts["answer_contains_prediction"],
    }
    return summary


def run_label(scored_path: Path) -> str:
    for part in scored_path.parts:
        if part.startswith("qwen_qwen3-vl-") and part.endswith("-instruct"):
            return part.removeprefix("qwen_qwen3-vl-").removesuffix("-instruct").upper()
    return scored_path.stem


def format_rollout_sizes(rollout_sizes: Counter[int]) -> str:
    if not rollout_sizes:
        return "0"
    if len(rollout_sizes) == 1:
        return str(next(iter(rollout_sizes)))
    values = sorted(rollout_sizes)
    return f"var:{values[0]}-{values[-1]}"


def format_int(value: Any) -> str:
    return f"{int(value):,}"


def format_float(value: Any) -> str:
    return f"{float(value):.3f}"


def format_percent(value: Any) -> str:
    return f"{float(value):.1f}%"


def print_compact_table(summaries: list[dict[str, Any]], *, details: bool) -> None:
    columns = [
        ("run", "Run", str),
        ("rows", "Rows", format_int),
        ("k", "k", str),
        ("judge_hit_examples", "JudgeHit", format_int),
        ("supported_hit_examples", "SuppHit", format_int),
        ("judge_pass@1", "J p@1", format_float),
        ("supported_pass@1", "S p@1", format_float),
        ("judge_pass@k", "J p@k", format_float),
        ("supported_pass@k", "S p@k", format_float),
        ("judge_positive_rollouts", "J Pos", format_int),
        ("supported_positive_rollouts", "S Pos", format_int),
        ("supported_positive_%", "S/J", format_percent),
        ("under_specific_%", "UndSpec/J", format_percent),
        ("mean_judge_pos_per_hit", "J Pos/Hit μ", format_float),
        ("median_judge_pos_per_hit", "J Pos/Hit med", format_float),
        ("mean_supported_pos_per_hit", "S Pos/Hit μ", format_float),
        ("median_supported_pos_per_hit", "S Pos/Hit med", format_float),
    ]
    if details:
        columns.extend(
            [
                ("exact", "Exact", format_int),
                ("alias", "Alias", format_int),
                ("contains_answer", "Pred⊃Ans", format_int),
                ("contains_alias", "Pred⊃Alias", format_int),
                ("answer_contains_prediction", "Ans⊃Pred", format_int),
            ]
        )

    rendered_rows = [
        [formatter(summary[key]) for key, _, formatter in columns]
        for summary in summaries
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rendered_rows))
        for index, (_, header, _) in enumerate(columns)
    ]

    def render(values: list[str]) -> str:
        return "  ".join(
            value.ljust(widths[index]) if index == 0 else value.rjust(widths[index])
            for index, value in enumerate(values)
        )

    headers = [header for _, header, _ in columns]
    print(render(headers))
    print(render(["-" * width for width in widths]))
    for row in rendered_rows:
        print(render(row))
    print("\nS = supported (whole-token): exact, alias, answer⊆pred, or alias⊆pred. All ≥ as specific as GT.")
    print("UndSpec = answer-contains-prediction (pred ⊆ answer): under-specific, NOT counted as supported.")
    print("p@k uses each row's full rollout count shown in the k column.")


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
    parser.add_argument(
        "--tsv",
        action="store_true",
        help="Print the full machine-readable TSV instead of the compact terminal table.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include exact/alias/containment support breakdown columns in the compact table.",
    )
    args = parser.parse_args()

    index = json.loads(Path(args.taxonomy_index).read_text())
    scored_files = find_scored_inputs(args.inputs)
    if not scored_files:
        raise SystemExit("No *_scored.jsonl files found")

    summaries = [
        audit_file(
            scored_path,
            index=index,
        )
        for scored_path in scored_files
    ]

    if not args.tsv:
        print_compact_table(summaries, details=args.details)
        return

    fieldnames = [
        "run",
        "scored_path",
        "rows",
        "k",
        "judge_hit_examples",
        "supported_hit_examples",
        "judge_pass@1",
        "supported_pass@1",
        "judge_pass@k",
        "supported_pass@k",
        "judge_positive_rollouts",
        "supported_positive_rollouts",
        "supported_positive_%",
        "under_specific_rollouts",
        "under_specific_%",
        "mean_judge_pos_per_hit",
        "median_judge_pos_per_hit",
        "mean_supported_pos_per_hit",
        "median_supported_pos_per_hit",
        "exact",
        "alias",
        "contains_answer",
        "contains_alias",
        "answer_contains_prediction",
    ]
    import sys

    out_writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, delimiter="\t")
    out_writer.writeheader()
    for summary in summaries:
        out_writer.writerow(summary)


if __name__ == "__main__":
    main()
