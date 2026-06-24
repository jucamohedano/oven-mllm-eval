#!/usr/bin/env python3
"""Post-hoc Recursive Self-Aggregation (RSA) for OVEN sampled rollouts.

This adapts the population-update loop from ``../RSA/eval_loop.py`` and the
paper in ``../resources/2509.26626v2.pdf`` to our OVEN setting.

Unlike ``scripts/run_inference.py``, this script does not create the initial
population itself.  It reads an existing ``*_samples.jsonl`` produced by
``--method naive-sampling`` and treats each row's ``all_texts`` as P1.  It then
recursively updates a fixed-size population by sampling K candidate answers and
asking the VLM, with the image and question, to produce one improved answer.

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


RSA_METHOD = "recursive-self-aggregation"

DOWNSTREAM_PREFIXES = (
    "judge_",
    "exact_match_",
    "contained_",
    "sentence_bert_",
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


def _default_output_path(input_path: Path, population: int, k: int, steps: int) -> Path:
    return input_path.with_name(
        f"{input_path.stem}_rsa_n{population}_k{k}_t{steps}.jsonl"
    )


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


def _resolve_image_path(path: str, root: Path) -> str:
    """Resolve image paths like ``scripts/run_inference.py`` does."""
    if not path:
        raise ValueError("Empty image_path in example")
    p = Path(path) if Path(path).is_absolute() else root / path
    if not p.exists():
        for ext in (".JPEG", ".jpeg", ".JPG"):
            alt = p.with_suffix(ext)
            if alt.exists():
                return str(alt)
        raise FileNotFoundError(f"Image not found: {p.resolve()} (cwd={Path.cwd()})")
    return str(p)


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


def build_oven_rsa_prompt(
    question: str,
    candidate_answers: list[str],
    prompt_variant: str,
) -> str:
    """Build the OVEN adaptation of the RSA aggregation prompt.

    RSA's original prompt gives the problem plus K candidate reasoning chains.
    Here the candidate objects are short OVEN entity/class answers, so the model
    is asked to aggregate visual evidence and candidate labels into one concise
    final answer instead of writing a long reasoning chain.
    """
    formatted_question = get_prompt(question, prompt_variant)
    parts: list[str] = []

    if len(candidate_answers) == 1:
        parts.append(
            "You are given an image question and one candidate answer. "
            "The candidate may be incomplete or wrong. Use the image and the question "
            "to produce one improved answer. If the candidate is wrong, answer with a better label. "
            "Return only the final answer; do not explain.\n"
        )
    else:
        parts.append(
            "You are given an image question and several candidate answers. "
            "Some candidates may be incorrect or under-specific. Aggregate the useful clues, "
            "choose the answer best supported by the image and question, and produce one improved answer. "
            "If all candidates seem wrong, answer with a better label. "
            "Return only the final answer; do not explain.\n"
        )

    parts.append("Question:\n")
    parts.append(formatted_question.strip() + "\n")

    if len(candidate_answers) == 1:
        parts.append("Candidate answer (may contain mistakes):\n")
        parts.append(f"---- Candidate ----\n{candidate_answers[0].strip()}\n")
        parts.append("Now write the improved final answer.")
    else:
        parts.append("Candidate answers (may contain mistakes):\n")
        for i, answer in enumerate(candidate_answers, 1):
            parts.append(f"---- Answer {i} ----\n{answer.strip()}\n")
        parts.append("Now write a single improved final answer.")

    return "\n".join(parts)


def _make_conversation(
    row: dict[str, Any],
    image: Image.Image | None,
    candidates: list[str],
    prompt_variant: str,
) -> list[dict[str, Any]]:
    question = row.get("question")
    if question is None:
        raise KeyError(f"Example missing 'question' field. Available keys: {sorted(row.keys())}")
    if prompt_variant == "source":
        prompt_variant = row.get("prompt_variant") or "concise_no_idk"
    prompt_text = build_oven_rsa_prompt(question, candidates, prompt_variant)
    content: list[dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image_pil", "image_pil": image})
    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


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
            "population": args.population,
            "k": args.k,
            "steps": args.steps,
            "updates": max(0, args.steps - 1),
            "initial_selection": args.initial_selection,
            "seed": args.seed,
            "prompt_variant": args.prompt_variant,
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
) -> list[list[str]]:
    current = populations
    for update_idx in range(1, args.steps):
        requests: list[list[dict[str, Any]]] = []
        for row, image, population in zip(rows, images, current):
            subsets = _sample_subsets(population, args.k, args.population, rng)
            for subset in subsets:
                requests.append(_make_conversation(row, image, subset, args.prompt_variant))

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
    return current


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run post-hoc Recursive Self-Aggregation over OVEN naive-sampling outputs."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input *_samples.jsonl with all_texts")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output samples JSONL. Default: <input>_rsa_nN_kK_tT.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct",
                        help="Aggregator VLM model path or HF ID")
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

    output_path = args.output or _default_output_path(args.input, args.population, args.k, args.steps)
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

    missing_all_texts = sum(1 for row in rows if not row.get("all_texts"))
    if missing_all_texts:
        raise SystemExit(f"{missing_all_texts} rows have no all_texts; RSA requires naive-sampling outputs")

    if args.resume:
        _clean_output_jsonl(output_path)
        done = _done_ids(output_path)
        rows = [row for row in rows if _row_id(row) not in done]
        print(f"Resuming: {len(done)} already done, {len(rows)} remaining")
    else:
        done = set()

    rng = random.Random(args.seed)
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
    )

    print(f"Input:  {args.input}")
    print(f"Output: {output_path}")
    print(
        f"RSA: N={args.population}, K={args.k}, T={args.steps} "
        f"({max(0, args.steps - 1)} aggregation update(s))"
    )
    print(f"Rows: input={input_rows}, to_process={len(rows)}, resumed={len(done)}")

    if args.dry_run:
        print("\n--- RSA prompt preview ---")
        print(preview)
        print("--- end preview ---")
        return

    if not rows:
        print("No examples to process — exiting.")
        return

    if args.no_image:
        print("Text-only RSA — no images loaded.")
        resolved_paths = ["" for _ in rows]
    else:
        image_root = Path(args.image_root) if args.image_root else Path.cwd()
        print(f"Resolving image paths... (root: {image_root})")
        with ThreadPoolExecutor(max_workers=16) as pool:
            resolved_paths = list(pool.map(
                lambda path: _resolve_image_path(path, image_root),
                [row.get("image_path", "") for row in rows],
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

    n_chunks = (len(rows) + args.chunk_size - 1) // args.chunk_size
    show_tqdm = True
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
        populations = [
            _initial_population(
                row.get("all_texts", []),
                args.population,
                args.initial_selection,
                chunk_rng,
            )
            for row in chunk_rows
        ]

        chunk_images: list[Image.Image | None]
        if args.no_image:
            chunk_images = [None] * len(chunk_rows)
        else:
            chunk_images = _load_images(resolved_paths[start:end])

        final_populations = _run_chunk(
            llm=llm,
            rows=chunk_rows,
            images=chunk_images,
            populations=populations,
            args=args,
            rng=chunk_rng,
            sampling_params=sampling_params,
            show_tqdm=show_tqdm,
        )

        for row, initial_population, final_population in zip(chunk_rows, populations, final_populations):
            source_n = len(row.get("all_texts", []))
            clean_row = _strip_downstream_fields(row)
            resolved_prompt_variant = (
                (row.get("prompt_variant") or "concise_no_idk")
                if args.prompt_variant == "source"
                else args.prompt_variant
            )
            out_row = {
                **clean_row,
                "prediction": final_population[0] if final_population else "",
                "method": RSA_METHOD,
                "prompt_variant": resolved_prompt_variant,
                "sampling": (
                    f"rsa: N={args.population}, K={args.k}, T={args.steps}; "
                    f"temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, n=1"
                ),
                "n_samples": len(final_population),
                "all_texts": final_population,
                "rsa": {
                    "source_method": row.get("method"),
                    "source_prompt_variant": row.get("prompt_variant"),
                    "source_n_samples": source_n,
                    "population": args.population,
                    "k": args.k,
                    "steps": args.steps,
                    "updates": max(0, args.steps - 1),
                    "initial_selection": args.initial_selection,
                    "initial_population": initial_population,
                    "seed": args.seed,
                },
            }
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
