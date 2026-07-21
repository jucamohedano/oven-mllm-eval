#!/usr/bin/env python3
"""Build lightweight Verl GRPO parquet files for OVEN taxonomy training.

The parquet stores image paths, not image bytes. Verl/Qwen-VL will load images
from the ``images=[{"image": "file:///..."}]`` field at training time.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any


DIRECT_SYSTEM_PROMPT = """You are a vision-language model for open-world image classification.
Answer the image question with the most specific entity name. Return only the final answer."""

RSA_SYSTEM_PROMPT = (
    "You are a visual entity recognition expert. "
    "Your task is to identify the most specific Wikidata entity in an image. "
    "For each problem, you will be given an image, a question, and several "
    "candidate solutions. Some candidates may be correct, others may contain "
    "errors. Evaluate each candidate against the visual evidence, then produce "
    "a single, well-reasoned solution. "
    "End with the most specific entity name in \\boxed{}."
)

TRAVERSAL_SYSTEM_PROMPT = (
    "You are a visual entity recognition expert. "
    "Your task is to identify the most specific entity in an image by first "
    "retrieving the relevant taxonomy hierarchy from memory. "
    "For each problem, follow these steps:\n"
    "1. Describe the visual evidence you observe in the image.\n"
    "2. Recall the taxonomy chain from broad category to the most specific "
    "matching entity. Write this inside <traversal> tags, one entity per line.\n"
    "3. Evaluate your traversal: is the entity at the end truly the most "
    "specific match? Could a narrower or sibling entity fit better? Consider "
    "alternatives and explain your reasoning.\n"
    "4. Output the most specific entity in \\boxed{}."
)

TRAVERSAL_WIKIDATA_SYSTEM_PROMPT = (
    "You identify entities in images by traversing the Wikidata subclass (P279) hierarchy. "
    "Entities are organized from broad categories to specific items through chains of "
    "subclass edges. Your task is to follow these chains from broad to specific."
)

# ---------------------------------------------------------------------------
# 1-shot examples — use Eiffel Tower, a well-known entity with a real OVEN
# taxonomy path (root → architectural element → perforated block → latticework →
# lattice tower → Eiffel Tower).  The example teaches the format; the entity is
# unlikely to appear in training data.
# ---------------------------------------------------------------------------

_ONESHOT_STANDARD = """\
Here is an example of the expected format:
Question: what type of vehicle is this?
The image shows a four-wheeled motor vehicle with a enclosed cabin, four doors, and a rear liftgate. It has higher ground clearance than a sedan with roof rails and plastic body cladding around the wheel arches, which are characteristic of a crossover SUV designed for both urban and light off-road use.
\\boxed{crossover SUV}

Now answer the following problem with the same format:"""

_ONESHOT_AGGREGATION = """\
Here is an example showing the expected format and reasoning:

Question: what type of vehicle is this?

Candidate solutions (may contain mistakes):
---- Solution 1 ----
The four-door vehicle with a rear hatch is a family car. \\boxed{station wagon}
---- Solution 2 ----
The raised vehicle with roof rails and plastic cladding is a crossover SUV. \\boxed{crossover SUV}
---- Solution 3 ----
A four-wheel drive vehicle with high ground clearance for off-road use. \\boxed{off-road SUV}
---- Solution 4 ----
The vehicle with an enclosed cabin and liftgate, combining car-like handling with SUV styling and raised suspension. \\boxed{crossover SUV}

Evaluation of each candidate:
- Solution 1 says "station wagon" — this misses the raised suspension, roof rails, and plastic cladding visible in the image. Incorrect.
- Solution 2 says "crossover SUV" — the raised ground clearance, roof rails, plastic wheel-arch cladding, and car-based platform are all features of a crossover SUV. Correct.
- Solution 3 says "off-road SUV" — off-road SUVs are body-on-frame with higher ground clearance and no car-like unibody construction. The image shows a unibody design. Incorrect.
- Solution 4 says "crossover SUV" — same correct reasoning as Solution 2, also correctly identifies the car-based platform with SUV styling. Correct.

Final answer: \\boxed{crossover SUV}

Now answer the following problem with the same format. Evaluate each candidate against what you see in the image, then produce the most specific entity name in \\boxed{}:"""

_ONESHOT_TRAVERSAL = """\
Here is an example of the expected format:

Question: what type of plant is shown?

Step 1 — Visual evidence:
- Small green leaves arranged in a rosette, each leaf tipped with red
- Hinged trap structures with tooth-like spines along the margins
- Three sensitive trigger hairs on the inner surface of each trap
- Grows low to the ground from a bulb-like rhizome

Step 2 — Taxonomy traversal:
<traversal>
plant
flowering plant
carnivorous plant
Venus flytrap
</traversal>

Step 3 — Evaluate:
The hinged snap-trap with marginal teeth and trigger hairs is unique to the Venus flytrap. Other carnivorous plants have different trapping mechanisms: sundews use sticky glandular tentacles and lack rapid movement; pitcher plants are passive pitfall traps formed from modified leaves. No narrower entity within the carnivorous plant group matches the visual evidence better than Venus flytrap.

Step 4 — Final answer:
\\boxed{Venus flytrap}

Now answer the following problem with the same format:"""

_ONESHOT_TRAVERSAL_WIKIDATA = """\
Example: what type of plant is shown?
<traversal>
plant
flowering plant
carnivorous plant
Venus flytrap
</traversal>
\\boxed{Venus flytrap}

Now answer the following question with the same format:"""

GENERIC_OVEN_QUESTIONS = {
    "what is the main object?",
    "what is shown in the photo?",
    "what is the main content of this image?",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Aligned OVEN train JSONL.")
    parser.add_argument(
        "--val-input",
        help="Optional aligned OVEN validation JSONL. If omitted, a QID-heldout split is made from --input.",
    )
    parser.add_argument("--labels", required=True, help="Taxonomy labels JSONL.")
    parser.add_argument("--descs", help="Taxonomy descriptions JSONL.")
    parser.add_argument("--output-dir", required=True, help="Directory for train.parquet, val.parquet, manifest.json.")
    parser.add_argument("--image-root", help="Fallback image root if rows do not contain image_path.")
    parser.add_argument("--data-source", default="oven_taxonomy_reasoning")
    parser.add_argument("--ability", default="open_world_image_classification")
    parser.add_argument(
        "--standard-prompt-variant",
        choices=["reasoning", "compute_buffer"],
        default="reasoning",
        help="Standard prompt style: 'reasoning' (default, 'Reason carefully...') or "
        "'compute_buffer' ('Think step by step...'). Only affects rsa_trace standard rows.",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=["direct", "rsa_trace"],
        default="direct",
        help="direct keeps the old final-answer prompt; rsa_trace builds standard/aggregation trace prompts.",
    )
    parser.add_argument(
        "--candidate-solutions",
        help="JSONL generated by run_recursive_self_agg.py --candidate-format solution --steps 1 for train rows.",
    )
    parser.add_argument(
        "--aggregation-fraction",
        type=float,
        default=0.5,
        help="Train-row probability of using an aggregation prompt in rsa_trace mode.",
    )
    parser.add_argument(
        "--aggregation-k",
        type=int,
        default=4,
        help="Number of cached candidate solution traces to include in aggregation prompts.",
    )
    parser.add_argument(
        "--traversal-fraction",
        type=float,
        default=0.0,
        help="Train-row probability of using a structured traversal prompt in rsa_trace mode. "
        "When >0, rows not selected for traversal fall through to the standard/aggregation split. "
        "Default 0.0 preserves existing behaviour.",
    )
    parser.add_argument(
        "--traversal-variant",
        choices=["structured", "wikidata"],
        default="structured",
        help="Traversal prompt style: 'structured' (5-section visual-semantic, current default) "
        "or 'wikidata' (P279 subclass chain, line-delimited).",
    )
    parser.add_argument(
        "--val-prompt-type",
        choices=["standard", "traversal"],
        default="standard",
        help="Prompt type for val-split rows in rsa_trace mode. 'traversal' builds a "
        "traversal-prompt validation set so traversal behaviour is measurable at eval time. "
        "Default 'standard' preserves existing behaviour.",
    )
    parser.add_argument(
        "--question-policy",
        choices=["aligned", "oven_original", "mixed", "dual"],
        default="aligned",
        help="Which question text to expose. dual emits both aligned and usable original-OVEN variants.",
    )
    parser.add_argument(
        "--oven-question-fraction",
        type=float,
        default=0.25,
        help="Probability of using the original OVEN question when --question-policy mixed.",
    )
    parser.add_argument(
        "--allow-generic-oven-questions",
        action="store_true",
        help="Allow generic original OVEN questions such as 'what is the main object?'.",
    )
    parser.add_argument("--val-qid-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument(
        "--ancestor-depth",
        type=int,
        default=0,
        help="Number of ancestor levels to expose, excluding the leaf answer. Default 0 avoids taxonomy leakage.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing train.parquet/val.parquet/manifest.json.")
    parser.add_argument(
        "--include-label-only-evidence",
        action="store_true",
        help="Include ancestor labels even when a description is unavailable.",
    )
    parser.add_argument("--max-train-rows", type=int, default=0, help="Optional smoke-test cap.")
    parser.add_argument("--max-val-rows", type=int, default=0, help="Optional smoke-test cap.")
    args = parser.parse_args()

    if not 0 <= args.aggregation_fraction <= 1:
        raise SystemExit("--aggregation-fraction must be in [0, 1]")
    if args.aggregation_k <= 0:
        raise SystemExit("--aggregation-k must be positive")
    if not 0 <= args.oven_question_fraction <= 1:
        raise SystemExit("--oven-question-fraction must be in [0, 1]")
    if not 0 <= args.traversal_fraction <= 1:
        raise SystemExit("--traversal-fraction must be in [0, 1]")
    if args.dataset_mode == "rsa_trace" and args.aggregation_fraction > 0 and not args.candidate_solutions:
        raise SystemExit("--dataset-mode rsa_trace with aggregation requires --candidate-solutions")
    return args


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_chain_file(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    chains: dict[str, list[str]] = {}
    for row in iter_jsonl(path):
        qid = row.get("id") or row.get("entity_id")
        taxonomy = row.get("taxonomy") or []
        if qid:
            chains[str(qid)] = [str(item or "") for item in taxonomy]
    return chains


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
    matches = []
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


def _candidate_list(row: dict[str, Any]) -> list[str]:
    """Return cached solution traces from an RSA solution-mode row."""
    for key in ("rsa_initial_solutions", "candidate_solutions"):
        values = row.get(key)
        if isinstance(values, list):
            return [str(value) for value in values if str(value).strip()]

    populations = row.get("rsa_populations_by_step")
    if isinstance(populations, list):
        for population in populations:
            if isinstance(population, list):
                values = [str(value) for value in population if str(value).strip()]
                if values:
                    return values

    values = row.get("rsa_final_solutions")
    if isinstance(values, list):
        return [str(value) for value in values if str(value).strip()]
    return []


def load_candidate_solutions(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    candidates_by_id: dict[str, list[str]] = {}
    rows = 0
    usable = 0
    for row in iter_jsonl(path):
        rows += 1
        row_id = str(row.get("data_id") or row.get("image_id") or "")
        if not row_id:
            continue
        candidates = _candidate_list(row)
        if candidates:
            candidates_by_id[row_id] = candidates
            usable += 1
    print(f"loaded candidate solution traces: rows={rows} usable={usable} path={path}")
    return candidates_by_id


def scan_qids(path: Path) -> set[str]:
    qids: set[str] = set()
    for row in iter_jsonl(path):
        qid = row.get("entity_id")
        if qid:
            qids.add(str(qid))
    return qids


def select_val_qids(qids: set[str], fraction: float, seed: int) -> set[str]:
    if fraction <= 0:
        return set()
    ordered = sorted(qids)
    count = max(1, round(len(ordered) * fraction))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return set(ordered[:count])


def _path_variants(path: Path) -> list[Path]:
    variants = [path]
    for ext in (".jpg", ".jpeg", ".JPEG", ".JPG", ".png", ".PNG", ".webp", ".WEBP"):
        if path.suffix != ext:
            variants.append(path.with_suffix(ext))
    return variants


def image_uri(row: dict[str, Any], image_root: str | None) -> str:
    path = str(row.get("image_path") or "")
    if path.startswith(("file://", "http://", "https://")):
        return path

    image_id = str(row.get("image_id") or "")
    if not path and not image_id:
        raise ValueError(f"row has no image_path and no usable image_id: {row.get('data_id')}")

    root = Path(image_root) if image_root else Path.cwd()
    candidates: list[Path] = []
    names: list[str] = []
    if path:
        raw = Path(path)
        candidates.append(raw if raw.is_absolute() else root / raw)
        names.append(raw.name)
    if image_id:
        names.append(f"{image_id}.jpg")

    for search_root in (root, root / "data" / "images", root / "images"):
        for name in names:
            if name:
                candidates.append(search_root / name)

    seen: set[str] = set()
    for candidate in candidates:
        for variant in _path_variants(candidate):
            key = str(variant)
            if key in seen:
                continue
            seen.add(key)
            if variant.exists():
                return f"file://{variant}"

    raw_path = Path(path) if path else root / f"{image_id}.jpg"
    if raw_path.is_absolute():
        return f"file://{raw_path}"
    return str(raw_path)


def ancestor_evidence(
    qid: str,
    labels_by_qid: dict[str, list[str]],
    descs_by_qid: dict[str, list[str]],
    depth: int,
    include_label_only: bool,
) -> tuple[list[str], list[str], list[str]]:
    labels = labels_by_qid.get(qid, [])
    descs = descs_by_qid.get(qid, [])
    lines: list[str] = []
    evidence_labels: list[str] = []
    evidence_descs: list[str] = []

    # Skip index 0: it is the ground-truth leaf label and would leak the answer.
    for idx in range(1, min(len(labels), depth + 1)):
        label = labels[idx].strip()
        desc = descs[idx].strip() if idx < len(descs) else ""
        if not label:
            continue
        if desc:
            lines.append(f"- {label}: {desc}")
            evidence_labels.append(label)
            evidence_descs.append(desc)
        elif include_label_only:
            lines.append(f"- {label}")
            evidence_labels.append(label)
            evidence_descs.append("")
    return lines, evidence_labels, evidence_descs


def _normalize_question(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _usable_oven_question(row: dict[str, Any], allow_generic: bool) -> str:
    oven_question = str(row.get("oven_question") or "").strip()
    aligned_question = str(row.get("question") or "").strip()
    if not oven_question:
        return ""
    if not allow_generic and _normalize_question(oven_question) in GENERIC_OVEN_QUESTIONS:
        return ""
    if _normalize_question(oven_question) == _normalize_question(aligned_question):
        return ""
    return oven_question


def question_variants(row: dict[str, Any], args: argparse.Namespace, rng: random.Random) -> list[tuple[str, str]]:
    aligned_question = str(row.get("question") or "").strip()
    oven_question = _usable_oven_question(row, args.allow_generic_oven_questions)

    if args.question_policy == "aligned":
        return [(aligned_question, "aligned")]
    if args.question_policy == "oven_original":
        return [(oven_question, "oven_original")] if oven_question else [(aligned_question, "aligned")]
    if args.question_policy == "mixed":
        if oven_question and rng.random() < args.oven_question_fraction:
            return [(oven_question, "oven_original")]
        return [(aligned_question, "aligned")]
    if args.question_policy == "dual":
        variants = [(aligned_question, "aligned")]
        if oven_question:
            variants.append((oven_question, "oven_original"))
        return variants
    raise ValueError(f"unknown question policy: {args.question_policy}")


def build_direct_prompt(question: str, evidence_lines: list[str]) -> list[dict[str, str]]:
    user_parts = [f"<image>\nQuestion: {question.strip()}"]
    if evidence_lines:
        user_parts.append(
            "Taxonomy context (ancestor classes only; these are not candidate answers):\n"
            + "\n".join(evidence_lines)
        )
    user_parts.append("Answer with only the most specific entity name.")
    return [
        {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def _load_alias_index() -> tuple[dict[str, set[str]], dict[str, str]]:
    """Load alias maps from the taxonomy index (cached)."""
    import os
    path = os.environ.get("OVEN_TAXONOMY_INDEX", "data/processed/oven_taxonomy_index.json")
    index_path = Path(path)
    if not index_path.exists():
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


_ONESHOT_STANDARD_COMPUTE_BUFFER = """\
Here is an example of the expected format:
Question: what type of vehicle is this?
Let me think: the image shows a four-wheeled motor vehicle with an enclosed cabin, four doors, and a rear liftgate. It has higher ground clearance than a sedan, with roof rails and plastic body cladding around the wheel arches. These features point to a crossover SUV — built on a car platform but with SUV styling and raised suspension.
\\boxed{crossover SUV}

Now answer the following problem with the same format:"""


def build_rsa_standard_prompt(question: str, prompt_variant: str = "reasoning") -> list[dict[str, str]]:
    """Standard prompt — no 1-shot, matching Zhang et al. and RSA paper designs.

    The model learns behaviour from reward, not from imitation.  One-shot
    examples are kept for evaluation only (structured prompting at test time).
    """
    if prompt_variant == "compute_buffer":
        user_prompt = "\n".join(
            [
                "<image>",
                "Think step by step about what you see. Then give the most specific entity name in \\boxed{}.",
                "",
                "Problem:",
                question.strip(),
                "",
                "Think step by step. End with the final answer in \\boxed{}.",
            ]
        )
    else:
        user_prompt = "\n".join(
            [
                "<image>",
                "You are given an image and an open-world visual recognition problem.",
                "Identify the most specific entity shown.",
                r"End with the final answer in \boxed{}.",
                "",
                "Problem:",
                question.strip(),
            ]
        )
    return [
        {"role": "system", "content": RSA_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_rsa_aggregation_prompt(question: str, candidates: list[str]) -> list[dict[str, str]]:
    """RSA-style aggregation prompt with a 1-shot example teaching candidate evaluation.

    The 1-shot demonstrates: evaluate each candidate individually against visual
    evidence, identify which are correct/wrong and why, then select the most
    specific correct answer in ``\\boxed{}``.

    Follows the prompt structure from Venkatraman et al. (2026, ``eval_loop.py``),
    adapted for multimodal visual recognition.  Single-candidate rows get a
    refinement instruction; multi-candidate rows get an aggregation instruction.
    """
    candidate_blocks = "\n\n".join(
        f"---- Solution {idx} ----\n{candidate.strip()}"
        for idx, candidate in enumerate(candidates, start=1)
    )

    n = len(candidates)
    if n == 1:
        instruction = (
            "You are given an image, a recognition problem, and a candidate solution. "
            "The candidate may be incomplete or contain errors. "
            "Evaluate the candidate against what you see in the image. "
            "If it is correct, confirm why. If it is wrong, explain why and provide "
            "the correct answer based on the visual evidence. "
            r"End with the most specific entity name in \boxed{}."
        )
        candidate_label = "Candidate solution (may contain mistakes):"
        closing = (
            "Now evaluate the candidate solution. "
            r"Provide your reasoning and end with the final answer in \boxed{}."
        )
    else:
        instruction = (
            "You are given an image, a recognition problem, and several candidate solutions. "
            "Some candidates may be correct, others may contain errors. "
            "Evaluate each candidate individually against what you see in the image, "
            "explain why it is correct or wrong, then select the best answer. "
            "If all candidates are wrong, provide the correct answer from your own knowledge. "
            r"End with the most specific entity name in \boxed{}."
        )
        candidate_label = "Candidate solutions (may contain mistakes):"
        closing = (
            "Now evaluate each candidate against the image. "
            "Explain your reasoning for each, then give the final answer in \\boxed{}."
        )

    user_prompt = "\n".join(
        [
            _ONESHOT_AGGREGATION,
            "",
            "---",
            "",
            "<image>",
            instruction,
            "",
            "Problem:",
            question.strip(),
            "",
            candidate_label,
            candidate_blocks,
            "",
            closing,
        ]
    )
    return [
        {"role": "system", "content": RSA_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_rsa_traversal_prompt(question: str) -> list[dict[str, str]]:
    """Zhang et al.-style structured recall prompt with 1-shot example.

    The model follows four numbered steps:
      1. Visual evidence — describe what you see
      2. Taxonomy traversal — recall the hierarchy broad→specific in <traversal> tags
      3. Evaluate — compare against alternatives, verify specificity
      4. Final answer — the most specific entity in \\boxed{}

    The <traversal> tag enables the reward function's path_match signal;
    the numbered-step structure matches Zhang et al.'s Structured Recall
    template, and step 3 (evaluate alternatives) encourages the model to
    actively compare candidates before committing to an answer.
    """
    user_prompt = "\n".join(
        [
            _ONESHOT_TRAVERSAL,
            "",
            "---",
            "",
            "<image>",
            "Follow the same four-step procedure shown above.",
            "",
            "Problem:",
            question.strip(),
            "",
            "Step 1 — Visual evidence:",
        ]
    )
    return [
        {"role": "system", "content": TRAVERSAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_rsa_traversal_wikidata_prompt(question: str) -> list[dict[str, str]]:
    """Build a Wikidata-aware traversal prompt.

    Teaches the model to output a P279 subclass chain inside <traversal> tags,
    one entity per line, from broad category to most specific entity.
    """
    user_prompt = "\n".join(
        [
            "<image>",
            question.strip(),
            "",
            "Output the Wikidata subclass (P279) chain from broad category to",
            "the most specific entity matching the image.",
            "Write one entity per line inside <traversal> tags:",
            "",
            "<traversal>",
            "broad category",
            "finer subclass",
            "...",
            "most specific entity",
            "</traversal>",
            "",
            r"End with the final answer in \boxed{}.",
            "",
            "Problem:",
            question.strip(),
            "",
            "<traversal>",
        ]
    )
    return [
        {"role": "system", "content": TRAVERSAL_WIKIDATA_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _candidate_correct(candidate_text: str, valid_answer_norms: set[str]) -> bool:
    """Check whether a candidate solution's boxed answer matches the ground truth."""
    boxed, parsed = extract_boxed_answer(str(candidate_text))
    if not parsed:
        return False
    return normalize_answer(boxed) in valid_answer_norms


def _build_valid_answer_norms(row: dict[str, Any]) -> set[str]:
    """Build a set of valid normalized answers for a row (mirrors oven_boxed.py)."""
    aliases_by_canonical, canonical_by_alias = _load_alias_index()
    labels = row.get("taxonomy_labels") or []
    leaf = str(labels[0]) if labels else ""

    answers = {
        normalize_answer(str(v))
        for v in (row.get("answer", ""), row.get("entity_text", ""), leaf)
        if str(v).strip()
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


def choose_prompt_type(
    row: dict[str, Any],
    *,
    parquet_split: str,
    args: argparse.Namespace,
    candidates_by_id: dict[str, list[str]],
    rng: random.Random,
) -> tuple[str, list[str], list[int], str]:
    if args.dataset_mode != "rsa_trace":
        return "direct", [], [], ""
    if parquet_split != "train":
        return args.val_prompt_type, [], [], ""

    # Traversal roll first (opt-in via --traversal-fraction)
    if args.traversal_fraction > 0 and rng.random() < args.traversal_fraction:
        return "traversal", [], [], ""

    # Remaining rows: standard vs aggregation with controlled difficulty
    row_id = str(row.get("data_id") or row.get("image_id") or "")
    candidates = candidates_by_id.get(row_id, [])
    if (
        candidates
        and len(candidates) >= args.aggregation_k
        and args.aggregation_fraction > 0
        and rng.random() < args.aggregation_fraction
    ):
        # Classify each candidate as correct/wrong
        valid = _build_valid_answer_norms(row)
        correct_idxs = [i for i, c in enumerate(candidates)
                        if _candidate_correct(c, valid)]
        wrong_idxs = [i for i, c in enumerate(candidates)
                      if not _candidate_correct(c, valid)]

        # Only emit aggregation if we can construct a set with 1-2 correct
        target_correct = rng.choice([1, 2])
        n_available_correct = min(target_correct, len(correct_idxs))
        n_needed_wrong = args.aggregation_k - n_available_correct

        if n_available_correct >= 1 and len(wrong_idxs) >= n_needed_wrong:
            chosen = (
                rng.sample(correct_idxs, n_available_correct)
                + rng.sample(wrong_idxs, n_needed_wrong)
            )
            rng.shuffle(chosen)
            selected = [candidates[i] for i in chosen]
            return "aggregation", selected, chosen, "candidate_solutions"
        # else: fall through to standard (not enough correct or wrong to build a balanced set)

    return "standard", [], [], ""


def make_record(
    row: dict[str, Any],
    *,
    parquet_split: str,
    index: int,
    args: argparse.Namespace,
    labels_by_qid: dict[str, list[str]],
    descs_by_qid: dict[str, list[str]],
    question: str,
    question_used: str,
    prompt_type: str,
    candidate_solutions: list[str],
    candidate_indices: list[int],
    candidate_source: str,
) -> dict[str, Any]:
    qid = str(row.get("entity_id") or "")
    labels = labels_by_qid.get(qid, [])
    descs = descs_by_qid.get(qid, [])
    evidence_lines, evidence_labels, evidence_descs = ancestor_evidence(
        qid,
        labels_by_qid,
        descs_by_qid,
        args.ancestor_depth,
        args.include_label_only_evidence,
    )
    answer = str(row.get("answer") or row.get("entity_text") or "")

    if args.dataset_mode == "direct":
        prompt = build_direct_prompt(question, evidence_lines)
        answer_format = "plain"
    elif prompt_type == "aggregation":
        prompt = build_rsa_aggregation_prompt(question, candidate_solutions)
        answer_format = "boxed"
    elif prompt_type == "traversal":
        if args.traversal_variant == "wikidata":
            prompt = build_rsa_traversal_wikidata_prompt(question)
        else:
            prompt = build_rsa_traversal_prompt(question)
        answer_format = "boxed"
    else:
        prompt = build_rsa_standard_prompt(question, prompt_variant=args.standard_prompt_variant)
        answer_format = "boxed"

    candidate_final_answers = [extract_boxed_answer(candidate)[0] for candidate in candidate_solutions]

    return {
        "data_source": args.data_source,
        "prompt": prompt,
        "ability": args.ability,
        "reward_model": {"style": "rule", "ground_truth": answer},
        "images": [{"image": image_uri(row, args.image_root)}],
        "extra_info": {
            "index": index,
            "split": parquet_split,
            "data_split": str(row.get("data_split") or ""),
            "data_id": str(row.get("data_id") or ""),
            "image_id": str(row.get("image_id") or ""),
            "image_path": str(row.get("image_path") or ""),
            "entity_id": qid,
            "answer": answer,
            "entity_text": str(row.get("entity_text") or ""),
            "question": question,
            "aligned_question": str(row.get("question") or ""),
            "oven_question": str(row.get("oven_question") or ""),
            "question_source": str(row.get("question_source") or ""),
            "question_policy": args.question_policy,
            "question_used": question_used,
            "dataset": str(row.get("dataset") or ""),
            "taxonomy_labels": labels,
            "taxonomy_descriptions": descs,
            "evidence_labels": evidence_labels,
            "evidence_descriptions": evidence_descs,
            "dataset_mode": args.dataset_mode,
            "prompt_type": prompt_type,
            "answer_format": answer_format,
            "aggregation_k": args.aggregation_k if prompt_type == "aggregation" else 0,
            "aggregation_fraction": args.aggregation_fraction if args.dataset_mode == "rsa_trace" else 0.0,
            "candidate_source": candidate_source,
            "candidate_indices": candidate_indices,
            "candidate_final_answers": candidate_final_answers,
            "candidate_solutions": candidate_solutions,
        },
    }


def parquet_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("data_source", pa.string()),
            ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
            ("ability", pa.string()),
            ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
            ("images", pa.list_(pa.struct([("image", pa.string())]))),
            (
                "extra_info",
                pa.struct(
                    [
                        ("index", pa.int64()),
                        ("split", pa.string()),
                        ("data_split", pa.string()),
                        ("data_id", pa.string()),
                        ("image_id", pa.string()),
                        ("image_path", pa.string()),
                        ("entity_id", pa.string()),
                        ("answer", pa.string()),
                        ("entity_text", pa.string()),
                        ("question", pa.string()),
                        ("aligned_question", pa.string()),
                        ("oven_question", pa.string()),
                        ("question_source", pa.string()),
                        ("question_policy", pa.string()),
                        ("question_used", pa.string()),
                        ("dataset", pa.string()),
                        ("taxonomy_labels", pa.list_(pa.string())),
                        ("taxonomy_descriptions", pa.list_(pa.string())),
                        ("evidence_labels", pa.list_(pa.string())),
                        ("evidence_descriptions", pa.list_(pa.string())),
                        ("dataset_mode", pa.string()),
                        ("prompt_type", pa.string()),
                        ("answer_format", pa.string()),
                        ("aggregation_k", pa.int64()),
                        ("aggregation_fraction", pa.float64()),
                        ("candidate_source", pa.string()),
                        ("candidate_indices", pa.list_(pa.int64())),
                        ("candidate_final_answers", pa.list_(pa.string())),
                        ("candidate_solutions", pa.list_(pa.string())),
                    ]
                ),
            ),
        ]
    )


class ParquetSink:
    def __init__(self, path: Path, schema, chunk_size: int):
        import pyarrow.parquet as pq

        self.path = path
        self.schema = schema
        self.chunk_size = chunk_size
        self.writer = pq.ParquetWriter(path, schema=schema)
        self.buffer: list[dict[str, Any]] = []
        self.count = 0
        self.qids: set[str] = set()
        self.prompt_type_counts: dict[str, int] = {}
        self.question_used_counts: dict[str, int] = {}

    def write(self, record: dict[str, Any]) -> None:
        import pyarrow as pa

        self.buffer.append(record)
        self.count += 1
        extra_info = record["extra_info"]
        qid = extra_info["entity_id"]
        if qid:
            self.qids.add(qid)
        prompt_type = str(extra_info["prompt_type"])
        question_used = str(extra_info["question_used"])
        self.prompt_type_counts[prompt_type] = self.prompt_type_counts.get(prompt_type, 0) + 1
        self.question_used_counts[question_used] = self.question_used_counts.get(question_used, 0) + 1
        if len(self.buffer) >= self.chunk_size:
            self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema))
            self.buffer.clear()

    def close(self) -> None:
        import pyarrow as pa

        if self.buffer:
            self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema))
            self.buffer.clear()
        self.writer.close()


def emit_records_for_row(
    row: dict[str, Any],
    *,
    sink: ParquetSink,
    parquet_split: str,
    args: argparse.Namespace,
    labels_by_qid: dict[str, list[str]],
    descs_by_qid: dict[str, list[str]],
    candidates_by_id: dict[str, list[str]],
    rng: random.Random,
    max_rows: int,
) -> None:
    for question, question_used in question_variants(row, args, rng):
        if max_rows and sink.count >= max_rows:
            return
        prompt_type, selected_candidates, candidate_indices, candidate_source = choose_prompt_type(
            row,
            parquet_split=parquet_split,
            args=args,
            candidates_by_id=candidates_by_id,
            rng=rng,
        )
        record = make_record(
            row,
            parquet_split=parquet_split,
            index=sink.count,
            args=args,
            labels_by_qid=labels_by_qid,
            descs_by_qid=descs_by_qid,
            question=question,
            question_used=question_used,
            prompt_type=prompt_type,
            candidate_solutions=selected_candidates,
            candidate_indices=candidate_indices,
            candidate_source=candidate_source,
        )
        sink.write(record)


def write_from_explicit_files(
    args: argparse.Namespace,
    train_sink: ParquetSink,
    val_sink: ParquetSink,
    labels_by_qid: dict[str, list[str]],
    descs_by_qid: dict[str, list[str]],
    candidates_by_id: dict[str, list[str]],
) -> None:
    rng = random.Random(args.seed)
    for sink, split, path, max_rows in [
        (train_sink, "train", Path(args.input), args.max_train_rows),
        (val_sink, "val", Path(args.val_input), args.max_val_rows),
    ]:
        for row in iter_jsonl(path):
            if max_rows and sink.count >= max_rows:
                break
            emit_records_for_row(
                row,
                sink=sink,
                parquet_split=split,
                args=args,
                labels_by_qid=labels_by_qid,
                descs_by_qid=descs_by_qid,
                candidates_by_id=candidates_by_id,
                rng=rng,
                max_rows=max_rows,
            )


def write_from_qid_split(
    args: argparse.Namespace,
    train_sink: ParquetSink,
    val_sink: ParquetSink,
    val_qids: set[str],
    labels_by_qid: dict[str, list[str]],
    descs_by_qid: dict[str, list[str]],
    candidates_by_id: dict[str, list[str]],
) -> None:
    rng = random.Random(args.seed)
    for row in iter_jsonl(Path(args.input)):
        qid = str(row.get("entity_id") or "")
        sink = val_sink if qid in val_qids else train_sink
        split = "val" if qid in val_qids else "train"
        max_rows = args.max_val_rows if split == "val" else args.max_train_rows
        if max_rows and sink.count >= max_rows:
            continue
        emit_records_for_row(
            row,
            sink=sink,
            parquet_split=split,
            args=args,
            labels_by_qid=labels_by_qid,
            descs_by_qid=descs_by_qid,
            candidates_by_id=candidates_by_id,
            rng=rng,
            max_rows=max_rows,
        )


def main() -> None:
    args = parse_args()
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise SystemExit("pyarrow is required: pip install pyarrow") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in [output_dir / "train.parquet", output_dir / "val.parquet", output_dir / "manifest.json"] if path.exists()]
    if existing and not args.overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise SystemExit(f"refusing to overwrite existing output(s): {paths}; pass --overwrite to replace them")

    labels_by_qid = load_chain_file(Path(args.labels))
    descs_by_qid = load_chain_file(Path(args.descs)) if args.descs else {}
    candidates_by_id = load_candidate_solutions(Path(args.candidate_solutions)) if args.candidate_solutions else {}
    schema = parquet_schema()
    train_sink = ParquetSink(output_dir / "train.parquet", schema=schema, chunk_size=args.chunk_size)
    val_sink = ParquetSink(output_dir / "val.parquet", schema=schema, chunk_size=args.chunk_size)

    val_qids: set[str] = set()
    if args.val_input:
        write_from_explicit_files(args, train_sink, val_sink, labels_by_qid, descs_by_qid, candidates_by_id)
        split_mode = "explicit_val_input"
    else:
        qids = scan_qids(Path(args.input))
        val_qids = select_val_qids(qids, args.val_qid_fraction, args.seed)
        write_from_qid_split(args, train_sink, val_sink, val_qids, labels_by_qid, descs_by_qid, candidates_by_id)
        split_mode = "qid_heldout_from_train"

    train_sink.close()
    val_sink.close()

    manifest = {
        "input": args.input,
        "val_input": args.val_input,
        "labels": args.labels,
        "descs": args.descs,
        "candidate_solutions": args.candidate_solutions,
        "output_dir": str(output_dir),
        "split_mode": split_mode,
        "dataset_mode": args.dataset_mode,
        "seed": args.seed,
        "val_qid_fraction": args.val_qid_fraction if not args.val_input else None,
        "val_qids": len(val_qids) if val_qids else None,
        "train_rows": train_sink.count,
        "train_qids": len(train_sink.qids),
        "train_prompt_types": train_sink.prompt_type_counts,
        "train_question_used": train_sink.question_used_counts,
        "val_rows": val_sink.count,
        "val_qids_observed": len(val_sink.qids),
        "val_prompt_types": val_sink.prompt_type_counts,
        "val_question_used": val_sink.question_used_counts,
        "question_policy": args.question_policy,
        "oven_question_fraction": args.oven_question_fraction,
        "allow_generic_oven_questions": args.allow_generic_oven_questions,
        "ancestor_depth": args.ancestor_depth,
        "leaf_answer_evidence_in_prompt": False,
        "include_label_only_evidence": args.include_label_only_evidence,
        "aggregation_fraction": args.aggregation_fraction if args.dataset_mode == "rsa_trace" else 0.0,
        "aggregation_k": args.aggregation_k if args.dataset_mode == "rsa_trace" else 0,
        "traversal_fraction": args.traversal_fraction if args.dataset_mode == "rsa_trace" else 0.0,
        "traversal_variant": args.traversal_variant if args.dataset_mode == "rsa_trace" else "structured",
        "val_prompt_type": args.val_prompt_type if args.dataset_mode == "rsa_trace" else "standard",
        "standard_prompt_variant": args.standard_prompt_variant if args.dataset_mode == "rsa_trace" else "reasoning",
        "candidate_solution_rows": len(candidates_by_id),
        "answer_format": "boxed" if args.dataset_mode == "rsa_trace" else "plain",
        "note": (
            "direct mode keeps the original final-answer prompt. rsa_trace mode builds paper-faithful "
            "standard and single-step aggregation prompts from cached train candidate solution traces; "
            "validation rows are standard prompts unless a future command intentionally supplies val candidates."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
