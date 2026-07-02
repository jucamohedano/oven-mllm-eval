#!/usr/bin/env python3
"""Mine the RSA candidate solutions to classify training entities as easy,
unlockable, or inaccessible.

Definition:
- easy: all sampled rollouts (or a very high fraction) produce the correct answer
- unlockable: at least one rollout is correct, but not all (knowledge is reachable
  through sampling but not reliably selected)
- inaccessible: no rollout produces the correct answer

Uses the existing RSA candidate solution file generated with T=1, population=16.
Outputs entity-level classifications and a filtered parquet with only unlockable rows.

Usage:
    python scripts/mine_unlockable_examples.py \
        --rsa-file data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
        --train-parquet data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/train_2k_balanced.parquet \
        --output-dir data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Boxed answer extraction — mirrors oven_boxed.py
# ---------------------------------------------------------------------------

def _strip_latex_answer(answer: str) -> str:
    answer = answer.strip().strip("$").strip().strip(".").strip()
    wrappers = (r"\text", r"\mathrm", r"\operatorname", r"\mathbf")
    changed = True
    while changed:
        changed = False
        for wrapper in wrappers:
            prefix = wrapper + "{"
            if answer.startswith(prefix) and answer.endswith("}"):
                answer = answer[len(prefix):-1].strip()
                changed = True
    return answer.strip().strip("$").strip().strip(".").strip()


def extract_boxed_answer(text: str) -> tuple[str, bool]:
    matches: list[str] = []
    start = 0
    needle = r"\boxed{"
    while True:
        box_start = text.find(needle, start)
        if box_start < 0:
            break
        content_start = box_start + len(needle)
        depth = 1
        idx = content_start
        while idx < len(text) and depth > 0:
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            idx += 1
        if depth == 0:
            answer = _strip_latex_answer(text[content_start:idx - 1])
            if answer:
                matches.append(answer)
            start = idx
        else:
            break
    if matches:
        return matches[-1], True
    return text.strip(), False


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Alias support — mirrors oven_boxed.py
# ---------------------------------------------------------------------------

def load_alias_index(taxonomy_index_path: str | None) -> tuple[dict[str, set[str]], dict[str, str]]:
    if not taxonomy_index_path:
        return {}, {}
    index_path = Path(taxonomy_index_path)
    if not index_path.exists():
        print(f"[warn] Taxonomy index not found: {index_path} — aliases disabled")
        return {}, {}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    aliases_by_canonical: dict[str, set[str]] = {}
    canonical_by_alias: dict[str, str] = {}
    for alias, canonical in index.get("aliases", {}).items():
        alias_norm = normalize_answer(str(alias))
        canonical_norm = normalize_answer(str(canonical))
        if not alias_norm or not canonical_norm:
            continue
        aliases_by_canonical.setdefault(canonical_norm, set()).add(alias_norm)
        canonical_by_alias.setdefault(alias_norm, canonical_norm)
    return aliases_by_canonical, canonical_by_alias


def build_valid_answers(answer: str, entity_text: str, taxonomy_labels: list[str],
                         aliases_by_canonical: dict[str, set[str]],
                         canonical_by_alias: dict[str, str]) -> set[str]:
    answers = {
        normalize_answer(v) for v in (answer, entity_text, (taxonomy_labels or [None])[0] if taxonomy_labels else "")
        if v and str(v).strip()
    }
    answers.discard("")

    expanded = set(answers)
    for ans in list(answers):
        canonical = canonical_by_alias.get(ans)
        if canonical:
            expanded.add(canonical)
    for ans in list(expanded):
        expanded.update(aliases_by_canonical.get(ans, set()))
    expanded.discard("")
    return expanded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine unlockable training examples from RSA candidate solutions.")
    parser.add_argument("--rsa-file", required=True, type=Path,
                        help="RSA candidate solutions JSONL (merged, n16_k4_t1)")
    parser.add_argument("--train-parquet", required=True, type=Path,
                        help="Training parquet to filter (e.g. train_2k_balanced.parquet)")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory for output files")
    parser.add_argument("--taxonomy-index", default=None, type=str,
                        help="Path to oven_taxonomy_index.json for alias expansion")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for tie-breaking (default: 42)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load alias index
    aliases_by_canonical, canonical_by_alias = load_alias_index(args.taxonomy_index)

    # --- Phase 1: classify entities from RSA file ---
    print(f"Reading RSA file: {args.rsa_file}")
    entity_class: dict[str, str] = {}          # entity_id → easy | unlockable | inaccessible
    entity_correct_count: dict[str, int] = {}  # entity_id → number of correct rollouts (out of n)
    entity_total_rollouts: dict[str, int] = {} # entity_id → total parsed rollouts
    entity_answers: dict[str, str] = {}         # entity_id → ground truth answer
    entity_examples: dict[str, list[dict]] = {} # entity_id → raw RSA rows (for analysis)

    with open(args.rsa_file, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = row.get("entity_id") or row.get("id") or ""
            if not qid:
                continue

            answer = row.get("answer", "")
            entity_text = row.get("entity_text", "")
            taxonomy_labels = row.get("taxonomy_labels", [])
            valid_answers = build_valid_answers(answer, entity_text, taxonomy_labels,
                                                 aliases_by_canonical, canonical_by_alias)

            if not valid_answers:
                continue

            # Check all initial solution traces (T=1, n=16 per entity)
            initial_solutions = row.get("rsa_initial_solutions", [])
            if not initial_solutions:
                continue

            correct = 0
            total_parsed = 0
            for sol in initial_solutions:
                boxed, parsed = extract_boxed_answer(str(sol))
                if parsed:
                    total_parsed += 1
                    if normalize_answer(boxed) in valid_answers:
                        correct += 1

            n = len(initial_solutions)
            entity_total_rollouts[qid] = total_parsed
            entity_correct_count[qid] = correct
            entity_answers[qid] = answer or entity_text or ""

            if total_parsed == 0:
                entity_class[qid] = "unparseable"
            elif correct == 0:
                entity_class[qid] = "inaccessible"
            elif correct == total_parsed:
                entity_class[qid] = "easy"
            else:
                entity_class[qid] = "unlockable"

    # --- Phase 2: report ---
    counts = Counter(entity_class.values())
    total = len(entity_class)
    print(f"\nEntity classification (n={total}):")
    for cls in ["easy", "unlockable", "inaccessible", "unparseable"]:
        n_cls = counts.get(cls, 0)
        print(f"  {cls:15s}: {n_cls:5d} ({n_cls/total*100:5.1f}%)")

    # Deeper stats on unlockable
    unlockable_correct = [entity_correct_count[q] for q, c in entity_class.items() if c == "unlockable"]
    if unlockable_correct:
        print(f"\nUnlockable correct-rollout distribution (out of max {max(entity_total_rollouts.values(), default=16)}):")
        cnt = Counter(unlockable_correct)
        for k in sorted(cnt):
            print(f"  {k:2d} correct: {cnt[k]:4d} entities")

    # --- Phase 3: filter training parquet ---
    print(f"\nReading training parquet: {args.train_parquet}")
    tbl = pq.read_table(args.train_parquet)
    extras = tbl["extra_info"].to_pylist()

    # Map parquet row indices to entity classification
    unlockable_indices = []
    all_classified_indices = defaultdict(list)
    missing = 0

    for i, ei in enumerate(extras):
        qid = ei.get("entity_id", "")
        cls = entity_class.get(qid)
        if cls is None:
            missing += 1
            continue
        all_classified_indices[cls].append(i)
        if cls == "unlockable":
            unlockable_indices.append(i)

    if missing:
        print(f"[warn] {missing} parquet rows have no RSA classification")

    # Write classification report
    report_path = args.output_dir / "entity_classification_report.json"
    report = {
        "rsa_file": str(args.rsa_file),
        "train_parquet": str(args.train_parquet),
        "total_entities_in_rsa": total,
        "classification": dict(counts),
        "train_parquet_rows": len(tbl),
        "train_parquet_rows_by_class": {cls: len(idxs) for cls, idxs in all_classified_indices.items()},
        "train_parquet_unclassified": missing,
        "unlockable_entity_examples": [
            {
                "entity_id": qid,
                "answer": entity_answers[qid],
                "correct_rollouts": entity_correct_count[qid],
                "total_parsed_rollouts": entity_total_rollouts[qid],
            }
            for qid, cls in entity_class.items()
            if cls == "unlockable"
        ][:20],  # first 20 for quick inspection
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nClassification report: {report_path}")

    # Write unlockable-only parquet
    if unlockable_indices:
        unlockable_indices.sort()
        subset = tbl.take(unlockable_indices)
        parquet_path = args.output_dir / f"{args.train_parquet.stem}_unlockable.parquet"
        pq.write_table(subset, parquet_path)

        # Coverage check
        unlockable_qids = set(extras[i].get("entity_id", "") for i in unlockable_indices)
        roots = set()
        prompt_types = Counter()
        for i in unlockable_indices:
            labels = extras[i].get("taxonomy_labels", [])
            if labels:
                roots.add(labels[-1])
            prompt_types[extras[i].get("prompt_type", "?")] += 1

        print(f"\nUnlockable parquet: {len(subset)} rows, {len(unlockable_qids)} unique QIDs")
        print(f"Taxonomy roots: {len(roots)}")
        print(f"Prompt types: {dict(prompt_types)}")
        print(f"Written to: {parquet_path}")
    else:
        print("\nNo unlockable examples found — nothing to write.")

    # Also write easy-only and inaccessible-only for completeness
    for cls in ["easy", "inaccessible"]:
        indices = all_classified_indices.get(cls, [])
        if indices:
            subset = tbl.take(sorted(indices))
            pq_path = args.output_dir / f"{args.train_parquet.stem}_{cls}.parquet"
            pq.write_table(subset, pq_path)
            qids = set(extras[i].get("entity_id", "") for i in indices)
            print(f"{cls.capitalize()} parquet: {len(subset)} rows, {len(qids)} QIDs → {pq_path}")


if __name__ == "__main__":
    main()
