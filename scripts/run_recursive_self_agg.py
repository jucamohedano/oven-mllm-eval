#!/usr/bin/env python3
"""Post-hoc Recursive Self-Aggregation (RSA) for OVEN sampled rollouts.

This adapts the population-update loop from ``../RSA/eval_loop.py`` and the
paper in ``../resources/2509.26626v2.pdf`` to our OVEN setting.

In the default ``--candidate-format answer`` mode, this script reads an existing
``*_samples.jsonl`` produced by ``--method naive-sampling`` and treats each
row's ``all_texts`` as P1.  In ``--candidate-format solution`` mode, it first
generates a fresh P1 of concise solution traces ending in ``\boxed{...}``, then
recursively aggregates solution traces.

The output remains compatible with ``scripts/run_judge.py`` and
``scripts/score_predictions.py``: each row contains a final ``all_texts`` list
and a ``prediction`` field, so the existing judge/scoring pipeline can be reused
unchanged.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

# Ensure project is importable when run as ``python scripts/...``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oven_mllm_eval.io import append_jsonl
from oven_mllm_eval.prompts import PROMPT_VARIANTS, get_prompt
from oven_mllm_eval.boxed import extract_boxed_answer
from oven_mllm_eval.paths import path_variants


RSA_METHOD = "recursive-self-aggregation"
RSA_SYSTEM_PROMPT = "You are a careful visual problem solver for open-world image recognition."

DOWNSTREAM_PREFIXES = (
    "judge_",
    "exact_match_",
    "contained_",
    "cascade_",          # current taxonomy-mapping scored fields
    "sentence_bert_",    # legacy name (kept so old scored files still strip clean)
    "scored_",
)
DOWNSTREAM_KEYS = {
    "scored_reference_path",
    "hP",
    "hR",
    "hF",
    "exact_match",
    "mapping_method",
}


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("data_id") or row.get("image_id") or "")


def _strip_downstream_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Remove judge/scoring fields if the source file was not a clean samples file."""
    return {
        key: value
        for key, value in row.items()
        if key not in DOWNSTREAM_KEYS
        and not any(key.startswith(prefix) for prefix in DOWNSTREAM_PREFIXES)
    }


def _default_output_path(input_path: Path, population: int, k: int, steps: int, candidate_format: str) -> Path:
    format_part = "" if candidate_format == "answer" else f"_{candidate_format}"
    return input_path.with_name(f"{input_path.stem}_rsa{format_part}_n{population}_k{k}_t{steps}.jsonl")


def _load_rows(path: Path, max_examples: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_examples is not None and len(rows) >= max_examples:
                break
    return rows


def _clean_output_jsonl(path: Path) -> None:
    if not path.exists():
        return
    valid_lines: list[str] = []
    bad = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            valid_lines.append(line)
    if bad:
        print(f"Resuming: dropping {bad} malformed line(s) from {path}")
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for line in valid_lines:
                handle.write(line + "\n")
        tmp.replace(path)


def _done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = _row_id(row)
            if row_id:
                done.add(row_id)
    return done


def _global_done_ids(base_output: Path) -> set[str]:
    """Union of done IDs across the merged file and every ``_shard*`` file.

    Reading all shard files gives a global resume set, so a data-parallel run
    resumes correctly even if the shard count changed between attempts (mirrors
    ``scripts/run_inference.py``).
    """
    done: set[str] = set()
    candidates = [base_output, *sorted(base_output.parent.glob(f"{base_output.stem}_shard*.jsonl"))]
    for path in candidates:
        done |= _done_ids(path)
    return done


def _resolve_image_path(row: dict[str, Any], root: Path) -> str:
    """Resolve image paths, including stale absolute paths from prepared JSONL.

    Some prepared OVEN rows contain absolute paths under the old image root
    while the cluster stores files under ``<root>/data/images``.  Prefer the
    stored path when valid, then fall back to root/basename and image_id names.
    """
    raw_path = str(row.get("image_path") or "")
    image_id = str(row.get("image_id") or "")
    if not raw_path and not image_id:
        raise ValueError(f"Empty image_path and image_id in example: {_row_id(row)}")

    candidates: list[Path] = []
    if raw_path:
        path = Path(raw_path)
        candidates.append(path if path.is_absolute() else root / path)
        candidates.append(path)
        names = [path.name]
    else:
        names = []

    if image_id:
        names.append(f"{image_id}.jpg")

    search_roots = [root, root / "data" / "images", root / "images"]
    for search_root in search_roots:
        for name in names:
            if name:
                candidates.append(search_root / name)

    tried: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        for variant in path_variants(candidate):
            key = str(variant)
            if key in seen:
                continue
            seen.add(key)
            tried.append(key)
            if variant.exists():
                return str(variant)

    preview = ", ".join(tried[:8])
    if len(tried) > 8:
        preview += ", ..."
    raise FileNotFoundError(
        f"Image not found for data_id={row.get('data_id')} image_id={image_id}; "
        f"tried: {preview} (cwd={Path.cwd()})"
    )


def _load_pil(path: str) -> Image.Image:
    if not path:
        raise ValueError("Empty image_path in example")
    image = Image.open(path)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    return image


def _load_images(paths: list[str]) -> list[Image.Image]:
    """Decode one chunk of images; mirrors the chunk-local memory pattern used by inference."""
    with ThreadPoolExecutor(max_workers=16) as pool:
        return list(pool.map(_load_pil, paths))


def _initial_population(
    all_texts: list[str],
    population: int,
    selection: str,
    rng: random.Random,
) -> list[str]:
    candidates = [str(text) for text in all_texts if str(text).strip()]
    if not candidates:
        return ["" for _ in range(population)]
    if len(candidates) >= population:
        if selection == "random":
            return rng.sample(candidates, population)
        return candidates[:population]
    repeats = (population + len(candidates) - 1) // len(candidates)
    padded = (candidates * repeats)[:population]
    if selection == "random":
        rng.shuffle(padded)
    return padded


def _sample_subsets(
    population: list[str],
    k: int,
    n_subsets: int,
    rng: random.Random,
) -> list[list[str]]:
    """Sample N subsets of size K without replacement, as in RSA Algorithm 1."""
    if not population:
        return [[] for _ in range(n_subsets)]
    k_eff = min(k, len(population))
    return [rng.sample(population, k_eff) for _ in range(n_subsets)]


def build_oven_initial_solution_prompt(question: str) -> str:
    return "\n".join(
        [
            "You are given an image and an open-world visual recognition problem.",
            "Write a concise solution that uses the image, the question, and relevant visual or world knowledge to identify the most specific entity name.",
            r"Reason carefully and end with the final answer in \boxed{}.",
            "",
            "Problem:",
            question.strip(),
            "",
            r"Now write a concise solution. End with the final answer in \boxed{}.",
        ]
    )


def build_oven_rsa_prompt(
    question: str,
    candidates: list[str],
    prompt_variant: str,
    candidate_format: str,
) -> str:
    """Build the OVEN adaptation of the RSA aggregation prompt.

    RSA's original prompt gives the problem plus K candidate reasoning chains.
    In answer mode candidates are short OVEN labels. In solution mode candidates
    are full solution traces ending in ``\boxed{...}``.
    """
    if candidate_format == "solution":
        return build_oven_solution_aggregation_prompt(question, candidates)

    formatted_question = get_prompt(question, prompt_variant)
    parts: list[str] = []

    if len(candidates) == 1:
        parts.append(
            "You are given an image, a question about the image, and one candidate answer. "
            "The candidate may be incomplete or wrong. Use the image and the question "
            "to produce one improved answer. If the candidate is wrong, answer with a better label. "
            "Return only the final answer; do not explain.\n"
        )
    else:
        parts.append(
            "You are given an image, a question about the image, and several candidate answers. "
            "Some candidates may be incorrect or under-specific. Aggregate the useful clues, "
            "choose the answer best supported by the image and question, and produce one improved answer. "
            "If all candidates seem wrong, answer with a better label. "
            "Return only the final answer; do not explain.\n"
        )

    parts.append("Question:\n")
    parts.append(formatted_question.strip() + "\n")

    if len(candidates) == 1:
        parts.append("Candidate answer (may contain mistakes):\n")
        parts.append(f"---- Candidate ----\n{candidates[0].strip()}\n")
        parts.append("Now write the improved final answer.")
    else:
        parts.append("Candidate answers (may contain mistakes):\n")
        for i, answer in enumerate(candidates, 1):
            parts.append(f"---- Answer {i} ----\n{answer.strip()}\n")
        parts.append("Now write a single improved final answer.")

    return "\n".join(parts)


def build_oven_solution_aggregation_prompt(question: str, candidate_solutions: list[str]) -> str:
    parts: list[str] = []
    if len(candidate_solutions) == 1:
        parts.append(
            "You are given an open-world visual recognition problem and a candidate solution. "
            "The candidate may be incomplete or contain errors. "
            "Refine this trajectory and produce an improved, higher-quality solution. "
            "If it is entirely wrong, attempt a new strategy. "
            r"End with the final result in \boxed{}."
        )
    else:
        parts.append(
            "You are given an open-world visual recognition problem and several candidate solutions. "
            "Some candidates may be incorrect or contain errors. "
            "Aggregate the useful ideas and produce a single, high-quality solution. "
            "Reason carefully; if candidates disagree, choose the path best supported by the image and question. "
            "If all are incorrect, then attempt a different strategy. "
            r"End with the final result in \boxed{}."
        )

    parts.append("\nProblem:\n")
    parts.append(question.strip() + "\n")

    if len(candidate_solutions) == 1:
        parts.append("Candidate solution (may contain mistakes):\n")
        parts.append(f"---- Candidate ----\n{candidate_solutions[0].strip()}\n")
        parts.append(r"Now refine the candidate into an improved solution. Provide clear reasoning and end with the final answer in \boxed{}.")
    else:
        parts.append("Candidate solutions (may contain mistakes):\n")
        for i, solution in enumerate(candidate_solutions, 1):
            parts.append(f"---- Solution {i} ----\n{solution.strip()}\n")
        parts.append(r"Now write a single improved solution. Provide clear reasoning and end with the final answer in \boxed{}.")

    return "\n".join(parts)


def _make_conversation(
    row: dict[str, Any],
    image: Image.Image | None,
    candidates: list[str],
    prompt_variant: str,
    candidate_format: str,
    initial_solution: bool = False,
) -> list[dict[str, Any]]:
    question = row.get("question")
    if question is None:
        raise KeyError(f"Example missing 'question' field. Available keys: {sorted(row.keys())}")
    if prompt_variant == "source":
        prompt_variant = row.get("prompt_variant") or "concise_no_idk"
    if initial_solution:
        prompt_text = build_oven_initial_solution_prompt(question)
    else:
        prompt_text = build_oven_rsa_prompt(question, candidates, prompt_variant, candidate_format)
    content: list[dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image_pil", "image_pil": image})
    content.append({"type": "text", "text": prompt_text})
    messages: list[dict[str, Any]] = []
    if candidate_format == "solution":
        messages.append({"role": "system", "content": RSA_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": content})
    return messages


def _write_metadata(
    output_path: Path,
    args: argparse.Namespace,
    input_rows: int,
    processed_rows: int,
    resumed_rows: int,
) -> None:
    metadata = {
        "model": args.model,
        "method": RSA_METHOD,
        "source_input": str(args.input),
        "output": str(output_path),
        "rsa": {
            "candidate_format": args.candidate_format,
            "population": args.population,
            "k": args.k,
            "steps": args.steps,
            "updates": max(0, args.steps - 1),
            "initial_selection": args.initial_selection,
            "seed": args.seed,
            "prompt_variant": args.prompt_variant,
            "stores_populations_by_step": args.candidate_format == "solution",
            "system_prompt": RSA_SYSTEM_PROMPT if args.candidate_format == "solution" else None,
            "reference": {
                "paper": "../resources/2509.26626v2.pdf",
                "code": "../RSA/eval_loop.py",
            },
        },
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "n": 1,
            "initial_n": args.population if args.candidate_format == "solution" else None,
        },
        "vllm": {
            "tensor_parallel_size": args.tp,
            "gpu_memory_utilization": args.gpu_util,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_pixels": args.max_pixels,
            "min_pixels": args.min_pixels,
            "enforce_eager": args.enforce_eager,
        },
        "data": {
            "input_rows": input_rows,
            "processed_rows": processed_rows,
            "resumed_rows": resumed_rows,
            "max_examples": args.max_examples,
            "chunk_size": args.chunk_size,
            "no_image": args.no_image,
            "shard": args.shard,
            "num_shards": args.num_shards,
        },
    }
    metadata_path = output_path.with_suffix("").with_name(f"{output_path.stem}_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_engine(args: argparse.Namespace):
    from vllm import LLM

    extra_llm_kwargs: dict[str, Any] = {}
    if "qwen" in args.model.lower():
        extra_llm_kwargs["mm_processor_kwargs"] = {
            "max_pixels": args.max_pixels,
            "min_pixels": args.min_pixels,
        }

    return LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
        enforce_eager=args.enforce_eager,
        disable_custom_all_reduce=(args.tp > 1),
        limit_mm_per_prompt={"image": 1},
        mm_processor_cache_gb=0,
        trust_remote_code=True,
        **extra_llm_kwargs,
    )


def _restart_engine(old_llm: Any, args: argparse.Namespace):
    del old_llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    time.sleep(10)
    return _build_engine(args)


def _run_chunk(
    llm: Any,
    rows: list[dict[str, Any]],
    images: list[Image.Image | None],
    populations: list[list[str]],
    args: argparse.Namespace,
    rng: random.Random,
    sampling_params: Any,
    show_tqdm: bool,
) -> tuple[list[list[str]], list[list[list[str]]] | None]:
    current = populations
    populations_by_step = [[list(population) for population in current]] if args.candidate_format == "solution" else None
    for update_idx in range(1, args.steps):
        requests: list[list[dict[str, Any]]] = []
        for row, image, population in zip(rows, images, current):
            subsets = _sample_subsets(population, args.k, args.population, rng)
            for subset in subsets:
                requests.append(_make_conversation(row, image, subset, args.prompt_variant, args.candidate_format))

        print(
            f"  RSA update {update_idx}/{args.steps - 1}: "
            f"{len(rows)} examples × N={args.population} prompts"
        )
        outputs = llm.chat(requests, sampling_params, use_tqdm=show_tqdm)
        texts = [completion.text for output in outputs for completion in output.outputs]
        expected = len(rows) * args.population
        if len(texts) != expected:
            raise RuntimeError(f"Expected {expected} RSA outputs, got {len(texts)}")
        current = [
            texts[i:i + args.population]
            for i in range(0, len(texts), args.population)
        ]
        if populations_by_step is not None:
            populations_by_step.append([list(population) for population in current])
    return current, populations_by_step


def _generate_initial_solution_populations(
    llm: Any,
    rows: list[dict[str, Any]],
    images: list[Image.Image | None],
    args: argparse.Namespace,
    sampling_params: Any,
    show_tqdm: bool,
) -> list[list[str]]:
    requests = [
        _make_conversation(row, image, [], args.prompt_variant, "solution", initial_solution=True)
        for row, image in zip(rows, images)
    ]
    print(f"  RSA initial solutions: {len(rows)} examples × N={args.population} prompts")
    outputs = llm.chat(requests, sampling_params, use_tqdm=show_tqdm)
    texts = [completion.text for output in outputs for completion in output.outputs]
    expected = len(rows) * args.population
    if len(texts) != expected:
        raise RuntimeError(f"Expected {expected} initial solution outputs, got {len(texts)}")
    return [texts[i:i + args.population] for i in range(0, len(texts), args.population)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run post-hoc Recursive Self-Aggregation over OVEN naive-sampling outputs."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input *_samples.jsonl rows to aggregate/evaluate")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output samples JSONL. Default: <input>_rsa[_solution]_nN_kK_tT.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct",
                        help="Aggregator VLM model path or HF ID")
    parser.add_argument("--candidate-format", choices=["answer", "solution"], default="answer",
                        help="answer: current label-only RSA over all_texts; "
                             "solution: generate and aggregate solution traces ending in \\boxed{}")
    parser.add_argument("--prompt-variant", default="source",
                        choices=["source", *PROMPT_VARIANTS.keys()],
                        help="Question formatting inside the RSA aggregation prompt. "
                             "'source' reuses each row's prompt_variant.")

    parser.add_argument("--population", type=int, default=16,
                        help="RSA population size N (default: 16)")
    parser.add_argument("--k", type=int, default=4,
                        help="Aggregation subset size K (default: 4)")
    parser.add_argument("--steps", type=int, default=2,
                        help="Total RSA population steps T, including input P1 (default: 2)")
    parser.add_argument("--initial-selection", choices=["first", "random"], default="first",
                        help="How to choose P1 from existing all_texts when more than N are available")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Seed for candidate subset sampling")

    # External data-parallel sharding: one process per GPU, each takes a stride.
    parser.add_argument("--shard", type=int, default=0,
                        help="This process's shard index (0-based) for data-parallel runs")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total shards. Each process handles examples[shard::num_shards]")

    parser.add_argument("--image-root", default=None,
                        help="Root for resolving relative image_path (default: cwd)")
    parser.add_argument("--no-image", action="store_true",
                        help="Text-only ablation: aggregate without image input")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit number of input examples")
    parser.add_argument("--resume", action="store_true",
                        help="Skip rows already present in the output JSONL")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete an existing output JSONL before writing")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and print one RSA prompt without loading vLLM")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=16)

    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu-util", type=float, default=0.92)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=512 * 512)
    parser.add_argument("--min-pixels", type=int, default=256 * 256)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=16,
                        help="Examples per chunk. Actual prompts per update = chunk_size × N")
    parser.add_argument("--restart-every", type=int, default=0,
                        help="Restart vLLM every N chunks (0 = never)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.population <= 0:
        raise SystemExit("--population must be > 0")
    if args.k <= 0:
        raise SystemExit("--k must be > 0")
    if args.k > args.population:
        raise SystemExit("--k cannot exceed --population")
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be > 0")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be > 0")
    if not args.input.exists():
        raise SystemExit(f"--input not found: {args.input}")

    base_output = args.output or _default_output_path(
        args.input,
        args.population,
        args.k,
        args.steps,
        args.candidate_format,
    )
    if args.num_shards > 1:
        if not (0 <= args.shard < args.num_shards):
            raise SystemExit(f"--shard must be in [0, {args.num_shards}), got {args.shard}")
        output_path = base_output.with_name(f"{base_output.stem}_shard{args.shard}.jsonl")
    else:
        output_path = base_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix("").with_name(f"{output_path.stem}_metadata.json")
    if args.overwrite and not args.resume:
        output_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    elif output_path.exists() and not args.resume and not args.dry_run:
        raise SystemExit(f"Output already exists: {output_path}. Pass --resume or --overwrite.")

    rows = _load_rows(args.input, args.max_examples)
    input_rows = len(rows)
    if not rows:
        raise SystemExit("No input rows found")

    # Strided sharding — balances load even if the file is ordered by category.
    if args.num_shards > 1:
        rows = rows[args.shard::args.num_shards]
        print(f"Shard {args.shard}/{args.num_shards}: {len(rows)} examples")

    missing_all_texts = sum(1 for row in rows if not row.get("all_texts"))
    if args.candidate_format == "answer" and missing_all_texts:
        raise SystemExit(f"{missing_all_texts} rows have no all_texts; answer-format RSA requires naive-sampling outputs")

    if args.resume:
        _clean_output_jsonl(output_path)
        done = _global_done_ids(base_output)
        rows = [row for row in rows if _row_id(row) not in done]
        print(f"Resuming: {len(done)} already done globally, {len(rows)} remaining in this shard")
    else:
        done = set()

    if not rows:
        print(f"Input:  {args.input}")
        print(f"Output: {output_path}")
        print("No examples to process — exiting.")
        return

    rng = random.Random(args.seed)
    if args.candidate_format == "answer":
        first_population = _initial_population(
            rows[0].get("all_texts", []),
            args.population,
            args.initial_selection,
            rng,
        )
        first_subset = _sample_subsets(first_population, args.k, 1, rng)[0]
        preview = build_oven_rsa_prompt(
            rows[0].get("question", ""),
            first_subset,
            (rows[0].get("prompt_variant") or "concise_no_idk")
            if args.prompt_variant == "source"
            else args.prompt_variant,
            args.candidate_format,
        )
    else:
        preview = "\n\n--- initial solution prompt ---\n"
        preview += build_oven_initial_solution_prompt(rows[0].get("question", ""))
        preview += "\n\n--- aggregation prompt preview ---\n"
        preview += build_oven_rsa_prompt(
            rows[0].get("question", ""),
            [r"Candidate visual reasoning. \boxed{candidate entity}" for _ in range(args.k)],
            "source",
            args.candidate_format,
        )

    print(f"Input:  {args.input}")
    print(f"Output: {output_path}")
    print(
        f"RSA: format={args.candidate_format}, N={args.population}, K={args.k}, T={args.steps} "
        f"({max(0, args.steps - 1)} aggregation update(s))"
    )
    print(f"Rows: input={input_rows}, to_process={len(rows)}, resumed={len(done)}")

    if args.dry_run:
        print("\n--- RSA prompt preview ---")
        print(preview)
        print("--- end preview ---")
        return

    if args.no_image:
        print("Text-only RSA — no images loaded.")
        resolved_paths = ["" for _ in rows]
    else:
        image_root = Path(args.image_root) if args.image_root else Path.cwd()
        print(f"Resolving image paths... (root: {image_root})")
        with ThreadPoolExecutor(max_workers=16) as pool:
            resolved_paths = list(pool.map(
                lambda row: _resolve_image_path(row, image_root),
                rows,
            ))

    print(f"Initializing vLLM engine: model={args.model} tp={args.tp}")
    llm = _build_engine(args)

    from vllm import SamplingParams

    sampling_kwargs: dict[str, Any] = {
        "n": 1,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if args.top_k != -1:
        sampling_kwargs["top_k"] = args.top_k
    sampling_params = SamplingParams(**sampling_kwargs)
    initial_sampling_kwargs = dict(sampling_kwargs)
    initial_sampling_kwargs["n"] = args.population
    initial_sampling_params = SamplingParams(**initial_sampling_kwargs)

    n_chunks = (len(rows) + args.chunk_size - 1) // args.chunk_size
    show_tqdm = args.num_shards <= 1  # \r redraws collide when shards share stdout
    processed = 0

    for chunk_idx in range(n_chunks):
        if args.restart_every and chunk_idx > 0 and chunk_idx % args.restart_every == 0:
            print(f"[restart] reinitializing engine after {args.restart_every} chunks", flush=True)
            llm = _restart_engine(llm, args)

        start = chunk_idx * args.chunk_size
        end = min(start + args.chunk_size, len(rows))
        chunk_rows = rows[start:end]
        print(f"[chunk {chunk_idx + 1}/{n_chunks}] examples {start}–{end - 1}")

        chunk_rng = random.Random(args.seed + start)
        chunk_images: list[Image.Image | None]
        if args.no_image:
            chunk_images = [None] * len(chunk_rows)
        else:
            chunk_images = _load_images(resolved_paths[start:end])

        if args.candidate_format == "answer":
            populations = [
                _initial_population(
                    row.get("all_texts", []),
                    args.population,
                    args.initial_selection,
                    chunk_rng,
                )
                for row in chunk_rows
            ]
        else:
            populations = _generate_initial_solution_populations(
                llm=llm,
                rows=chunk_rows,
                images=chunk_images,
                args=args,
                sampling_params=initial_sampling_params,
                show_tqdm=show_tqdm,
            )

        final_populations, populations_by_step = _run_chunk(
            llm=llm,
            rows=chunk_rows,
            images=chunk_images,
            populations=populations,
            args=args,
            rng=chunk_rng,
            sampling_params=sampling_params,
            show_tqdm=show_tqdm,
        )

        for row_idx, (row, initial_population, final_population) in enumerate(zip(chunk_rows, populations, final_populations)):
            source_n = len(row.get("all_texts", []))
            clean_row = _strip_downstream_fields(row)
            resolved_prompt_variant = (
                (row.get("prompt_variant") or "concise_no_idk")
                if args.prompt_variant == "source"
                else args.prompt_variant
            )
            parsed_population = final_population
            parse_flags: list[bool] = []
            if args.candidate_format == "solution":
                parsed_pairs = [extract_boxed_answer(text) for text in final_population]
                parsed_population = [answer for answer, _ in parsed_pairs]
                parse_flags = [ok for _, ok in parsed_pairs]

            rsa_info = {
                "source_method": row.get("method"),
                "source_prompt_variant": row.get("prompt_variant"),
                "source_n_samples": source_n,
                "candidate_format": args.candidate_format,
                "population": args.population,
                "k": args.k,
                "steps": args.steps,
                "updates": max(0, args.steps - 1),
                "initial_selection": args.initial_selection,
                "seed": args.seed,
            }
            if args.candidate_format == "answer":
                rsa_info["initial_population"] = initial_population

            out_row = {
                **clean_row,
                "prediction": parsed_population[0] if parsed_population else "",
                "method": RSA_METHOD,
                "prompt_variant": resolved_prompt_variant,
                "rsa_candidate_format": args.candidate_format,
                "sampling": (
                    f"rsa: N={args.population}, K={args.k}, T={args.steps}; "
                    f"temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, n=1"
                ),
                "n_samples": len(parsed_population),
                "all_texts": parsed_population,
                "rsa": rsa_info,
            }
            if args.candidate_format == "solution":
                row_populations_by_step = (
                    [step_populations[row_idx] for step_populations in populations_by_step]
                    if populations_by_step is not None
                    else [initial_population, final_population]
                )
                out_row.update(
                    {
                        "rsa_initial_solutions": initial_population,
                        "rsa_populations_by_step": row_populations_by_step,
                        "rsa_final_solutions": final_population,
                        "rsa_raw_prediction": final_population[0] if final_population else "",
                        "rsa_parse_ok": parse_flags[0] if parse_flags else False,
                        "rsa_parse_ok_all": parse_flags,
                    }
                )
            append_jsonl(output_path, out_row)

        processed += len(chunk_rows)
        del chunk_images, final_populations
        print(f"[chunk {chunk_idx + 1}/{n_chunks}] done — {processed}/{len(rows)} examples", flush=True)

    _write_metadata(
        output_path=output_path,
        args=args,
        input_rows=input_rows,
        processed_rows=processed,
        resumed_rows=len(done),
    )
    print(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()
