#!/usr/bin/env python3
"""Run inference on OVEN using vLLM's offline ``LLM.chat()`` API.

Supports three sampling modes:

  1. **naive**: single sample per example (n=1).
  2. **naive-sampling**: draw N independent samples, pick the best match.
  3. **iterative**: draw N samples per round for T rounds; if all fail
     matching, optionally feed back failed attempts and retry.

All modes use stochastic sampling (temperature > 0).  Defaults mirror
verl's GRPO training config: temperature=1.0, top_p=1.0, top_k=-1.

Output structure follows the lmms-ocw convention::

    logs/schedule/oven_<method>_<prompt>/<model>/<run_id>/
        <run_id>_samples.jsonl      per-sample outputs + metrics
        <run_id>_results.json       aggregate metrics

Usage examples::

    # Naive (1 sample per example, stochastic)
    uv run --extra serve python scripts/run_inference.py \\
        --input data/processed/vlm_compatible_val.jsonl \\
        --method naive --prompt-variant barebones

    # Naive sampling (n=64)
    uv run --extra serve python scripts/run_inference.py \\
        --input data/processed/vlm_compatible_val.jsonl \\
        --method naive-sampling --prompt-variant barebones \\
        --temperature 1.0 --samples-per-example 64
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from PIL import Image
from vllm import LLM, SamplingParams

# Ensure project is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oven_mllm_eval.io import append_jsonl
from oven_mllm_eval.prompts import get_prompt, PROMPT_VARIANTS


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def build_output_dir(model: str, method: str, prompt_variant: str, output_root: str = "logs/schedule") -> tuple[Path, str]:
    """Build the output directory and run ID following the lmms-ocw convention.

    Structure::

        <output_root>/oven_<method>_<prompt>/<model_slug>/<run_id>/

    where ``model_slug`` normalises the model name for filesystem safety
    (e.g. ``Qwen/Qwen3-VL-8B-Instruct`` → ``qwen_qwen3-vl-8b-instruct``)
    and ``run_id`` is ``YYYYMMDD_HHMMSS_<rand6>`` so that repeated runs never
    overwrite each other.  The random suffix avoids collisions when two jobs
    start in the same second.

    Returns
    -------
    (Path, str)
        The output directory and the run_id prefix (used for file naming).
    """
    model_slug = model.replace("/", "_").lower()
    dirname = f"oven_{method}_{prompt_variant}"
    date_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand_id = secrets.randbelow(1_000_000)
    run_id = f"{date_id}_{rand_id:06d}"
    return Path(output_root) / dirname / model_slug / run_id, run_id


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _raise_missing_question(example: dict):
    """Called when 'question' key is missing — raises a clear error."""
    raise KeyError(
        f"Example missing 'question' field. Available keys: {sorted(example.keys())}. "
        f"Re-run prepare_oven.py with the correct raw data that includes 'question'."
    )


def _matches_label(text: str, ground_truth: str) -> bool:
    """Simple normalised string matching."""
    label = _normalise(ground_truth)
    if not label:
        return False
    answer_region = text.split("```")[-1]
    answer = _normalise(answer_region)
    return re.search(rf"(^| ){re.escape(label)}( |$)", answer) is not None


def _normalise(text: str) -> str:
    text = text.lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _build_feedback(round_idx: int, failed_texts: list[str], max_chars: int = 2000, final_round: bool = False) -> str:
    """Build a feedback message from failed attempts."""
    chunks = []
    for idx, text in enumerate(failed_texts, start=1):
        clipped = text if len(text) <= max_chars else text[:max_chars] + "\n...(truncated)"
        chunks.append(f"Attempt {idx}:\n{clipped}")
    failed_block = "\n\n".join(chunks)

    if final_round:
        return f"All sampled answers in final round {round_idx} were incorrect.\n\nFailed attempts:\n{failed_block}"
    return (
        f"All sampled answers in round {round_idx} were incorrect.\n\n"
        f"Failed attempts:\n{failed_block}\n\n"
        "Use these failed attempts as evidence. Reflect on what they missed, then answer again. "
        "State the final class label clearly."
    )


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_pil(path: str) -> Image.Image | None:
    """Load an image as RGB PIL, returning None if the file doesn't exist."""
    if not path or not Path(path).exists():
        return None
    img = Image.open(path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _make_conversation(example: dict, image: Image.Image | None, prompt_variant: str) -> list[dict]:
    """Build a single-turn conversation for ``LLM.chat()`` (instruct models).

    PIL images are passed directly — no base64 encoding needed.
    """
    question = example.get("question")
    if question is None:
        raise KeyError(
            f"Example missing 'question' field. Available keys: {sorted(example.keys())}. "
            f"Re-run prepare_oven.py with the correct raw data that includes 'question'."
        )
    prompt_text = get_prompt(question, prompt_variant)
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


def _make_raw_prompt(example: dict, image: Image.Image | None, prompt_variant: str) -> dict:
    """Build a raw prompt dict for ``LLM.generate()`` (base/pretrained models).

    Returns a dict with ``prompt`` and ``multi_modal_data`` keys that vLLM's
    processor handles directly — no chat template required.
    """
    question = example.get("question")
    if question is None:
        raise KeyError(
            f"Example missing 'question' field. Available keys: {sorted(example.keys())}. "
            f"Re-run prepare_oven.py with the correct raw data that includes 'question'."
        )
    result: dict = {"prompt": get_prompt(question, prompt_variant)}
    if image is not None:
        result["multi_modal_data"] = {"image": image}
    return result


# ---------------------------------------------------------------------------
# Iterative method
# ---------------------------------------------------------------------------

def run_iterative(
    llm: LLM,
    examples: list[dict],
    images: list[Image.Image | None],
    prompt_variant: str,
    sampling_kwargs: dict,
    max_tokens: int,
    attempts_per_round: int,
    max_rounds: int,
    enable_feedback: bool,
    max_feedback_chars: int,
    output_path: Path,
):
    """Iterative resampling with optional feedback.

    Each round batches all still-active examples into one ``llm.chat()`` call
    with ``n=attempts_per_round``.  vLLM shares the prefill KV cache across
    the n samples and across rounds (via prefix caching) — the expensive
    vision encoding only happens once per example.
    """
    # Per-example state
    states: list[dict] = []
    for i, (ex, img) in enumerate(zip(examples, images)):
        prompt_text = get_prompt(ex.get("question") or _raise_missing_question(ex), prompt_variant)
        content = []
        if img is not None:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt_text})
        states.append({
            "idx": i,
            "conversation": [{"role": "user", "content": content}],
            "all_attempts": [],
            "success": False,
            "success_round": 0,
            "best_response": "",
            "total_attempts": 0,
        })

    round_sampling = SamplingParams(
        n=attempts_per_round,
        temperature=sampling_kwargs["temperature"],
        top_p=sampling_kwargs["top_p"],
        max_tokens=max_tokens,
        **({"top_k": sampling_kwargs["top_k"]} if "top_k" in sampling_kwargs else {}),
    )

    for round_idx in range(1, max_rounds + 1):
        active = [s for s in states if not s["success"]]
        if not active:
            break

        conversations = [s["conversation"] for s in active]
        print(f"[iterative] round {round_idx}/{max_rounds}: "
              f"{len(active)} active × n={attempts_per_round}")

        outputs = llm.chat(conversations, round_sampling, use_tqdm=True)

        newly_succeeded = []
        for state, request_output in zip(active, outputs):
            ground_truth = examples[state["idx"]].get("answer", "")
            texts = [co.text for co in request_output.outputs]
            failed_texts: list[str] = []

            for attempt_idx, text in enumerate(texts, start=1):
                state["total_attempts"] += 1
                matched = _matches_label(text, ground_truth)
                state["all_attempts"].append({
                    "round": round_idx, "attempt": attempt_idx,
                    "text": text, "matched": matched,
                })
                if matched and not state["success"]:
                    state["success"] = True
                    state["success_round"] = round_idx
                    state["best_response"] = text
                else:
                    failed_texts.append(text)

            if state["success"]:
                newly_succeeded.append(state)
            elif enable_feedback and round_idx < max_rounds:
                feedback = _build_feedback(round_idx, failed_texts, max_chars=max_feedback_chars)
                state["conversation"].append({"role": "assistant", "content": ""})
                state["conversation"].append({"role": "user", "content": feedback})

        # Stream succeeded results to disk
        for state in newly_succeeded:
            _write_iterative_result(state, examples[state["idx"]],
                prompt_variant, sampling_kwargs, attempts_per_round,
                max_rounds, enable_feedback, output_path)

    # Write failures that exhausted max_rounds
    for state in states:
        if not state["success"]:
            _write_iterative_result(state, examples[state["idx"]],
                prompt_variant, sampling_kwargs, attempts_per_round,
                max_rounds, enable_feedback, output_path)


def _write_iterative_result(state, example, prompt_variant, sampling_kwargs,
                            attempts_per_round, max_rounds, enable_feedback, output_path):
    prediction = (
        state["best_response"] if state["success"]
        else (state["all_attempts"][-1]["text"] if state["all_attempts"] else "")
    )
    append_jsonl(output_path, {
        **example,
        "prediction": prediction,
        "method": "iterative",
        "prompt_variant": prompt_variant,
        "sampling": f"temp={sampling_kwargs['temperature']}, top_p={sampling_kwargs['top_p']}, top_k={sampling_kwargs.get('top_k', -1)}, n={attempts_per_round}",
        "attempts_per_round": attempts_per_round,
        "max_rounds": max_rounds,
        "enable_feedback": enable_feedback,
        "success": state["success"],
        "success_round": state["success_round"],
        "total_attempts": state["total_attempts"],
        "attempts": state["all_attempts"],
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run OVEN inference via vLLM offline")

    # Data I/O
    parser.add_argument("--input", required=True, help="Input JSONL (prepared OVEN data)")
    parser.add_argument("--taxonomy-index", default="data/processed/oven_taxonomy_index.json",
                        help="Path to taxonomy index JSON")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: auto-generated from model/method/prompt)")
    parser.add_argument("--output-root", default="logs/schedule",
                        help="Root directory for auto-generated output paths (default: logs/schedule)")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct", help="Model path or HF ID")
    parser.add_argument("--prompt-variant", default="barebones", choices=list(PROMPT_VARIANTS.keys()))
    parser.add_argument("--method", default="naive", choices=["naive", "naive-sampling", "iterative"])
    parser.add_argument("--max-tokens", type=int, default=300)

    # Sampling — decoupled from method, always stochastic
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Nucleus sampling threshold (default: 1.0 = disabled)")
    parser.add_argument("--top-k", type=int, default=-1,
                        help="Top-k sampling (default: -1 = disabled)")
    parser.add_argument("--n", type=int, default=1,
                        help="Number of completions per request (default: 1). "
                             "Overridden by --samples-per-example for naive-sampling "
                             "and --attempts-per-round for iterative.")

    # naive-sampling
    parser.add_argument("--samples-per-example", type=int, default=64)

    # iterative
    parser.add_argument("--attempts-per-round", type=int, default=16)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--enable-feedback", type=lambda x: x.lower() in ("true", "1", "yes"), default=False)
    parser.add_argument("--max-feedback-chars", type=int, default=2000)

    # vLLM engine
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism (default: 1)")
    parser.add_argument("--gpu-util", type=float, default=0.92, help="GPU memory utilization (default: 0.92)")
    parser.add_argument("--max-model-len", type=int, default=1024, help="Max model context length (default: 1024)")
    parser.add_argument("--max-num-seqs", type=int, default=1024, help="Max number of sequences (default: 1024)")
    parser.add_argument("--max-pixels", type=int, default=512 * 512,
                        help="Max pixels for image resizing (default: 262144 = 512x512)")
    parser.add_argument("--min-pixels", type=int, default=256 * 256,
                        help="Min pixels for image resizing (default: 65536 = 256x256)")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable CUDA graphs — slower but more uniform step latency")
    parser.add_argument("--base-model", action="store_true",
                        help="Use LLM.generate() with raw prompts (for base/pretrained models "
                             "that lack a chat template)")

    # Chunking — write results to disk after every chunk, so a crash at 99%
    # only loses one chunk.  The LLM engine is reused across chunks.
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="Examples per llm.chat() call (default: 256). "
                             "Larger = more GPU utilisation; smaller = less lost work on crash.")

    # limits
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed examples in output")
    # External data-parallel sharding: one process per GPU, each takes a stride.
    parser.add_argument("--shard", type=int, default=0, help="This process's shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total shards. Each process handles examples[shard::num_shards]")
    args = parser.parse_args()

    # Validate sampling params
    if args.temperature <= 0:
        parser.error(f"--temperature must be > 0, got {args.temperature}")

    # Determine n from method
    n = args.n
    if args.method == "naive-sampling":
        n = args.samples_per_example
    elif args.method == "iterative":
        n = args.attempts_per_round

    sampling_kwargs = {"n": n, "temperature": args.temperature, "top_p": args.top_p}
    if args.top_k != -1:
        sampling_kwargs["top_k"] = args.top_k

    # Load examples
    examples = []
    with open(args.input, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if args.max_examples:
        examples = examples[:args.max_examples]

    # Strided sharding — balances load even if the file is ordered by category/size.
    if args.num_shards > 1:
        if not (0 <= args.shard < args.num_shards):
            parser.error(f"--shard must be in [0, {args.num_shards}), got {args.shard}")
        examples = examples[args.shard::args.num_shards]
        print(f"Shard {args.shard}/{args.num_shards}: {len(examples)} examples")

    # Build output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        run_id = output_dir.name
    else:
        output_dir, run_id = build_output_dir(args.model, args.method, args.prompt_variant, args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_shard{args.shard}" if args.num_shards > 1 else ""
    output_path = output_dir / f"{run_id}_samples{suffix}.jsonl"

    # Resume support — filter before loading images
    if args.resume and output_path.exists():
        done_ids = set()
        with open(output_path, "r") as f:
            for line in f:
                row = json.loads(line)
                done_ids.add(row.get("data_id", row.get("image_id", "")))
        examples = [e for e in examples if e.get("data_id", e.get("image_id", "")) not in done_ids]
        print(f"Resuming: {len(done_ids)} already done, {len(examples)} remaining")

    # Load images in parallel
    print("Loading images...")
    with ThreadPoolExecutor(max_workers=16) as pool:
        images = list(pool.map(lambda e: _load_pil(e.get("image_path", "")), examples))

    # Build prompts (conversations for instruct, raw dicts for base)
    if args.base_model:
        raw_prompts = [
            _make_raw_prompt(ex, img, args.prompt_variant)
            for ex, img in zip(examples, images)
        ]
    else:
        conversations = [
            _make_conversation(ex, img, args.prompt_variant)
            for ex, img in zip(examples, images)
        ]

    # Build vLLM engine
    print(f"Initializing vLLM engine: model={args.model} tp={args.tp} max_model_len={args.max_model_len}")

    # Per-model-family kwargs — avoids passing processor-specific args to the
    # wrong model (e.g. Qwen's max_pixels/min_pixels would error on InternVL).
    extra_llm_kwargs: dict = {}
    if "qwen" in args.model.lower():
        extra_llm_kwargs["mm_processor_kwargs"] = {
            "max_pixels": args.max_pixels,
            "min_pixels": args.min_pixels,
        }

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
        enforce_eager=args.enforce_eager,
        limit_mm_per_prompt={"image": 1},
        mm_processor_cache_gb=0,
        async_scheduling=True,
        trust_remote_code=True,
        **extra_llm_kwargs,
    )

    print(f"Running {args.method} inference on {len(examples)} examples")
    print(f"  Model:     {args.model}")
    print(f"  Prompt:    {args.prompt_variant}")
    print(f"  Sampling:  temp={args.temperature}, top_p={args.top_p}, n={n}")
    print(f"  Max tok:   {args.max_tokens}")
    print(f"  Chunk:     {args.chunk_size} examples per {'llm.generate()' if args.base_model else 'llm.chat()'} call")
    print(f"  Output:    {output_dir}")

    # ── Write run metadata ──────────────────────────────────────────
    metadata = {
        "model": args.model,
        "method": args.method,
        "prompt_variant": args.prompt_variant,
        "base_model": args.base_model,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "n": n,
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
            "input": args.input,
            "taxonomy_index": args.taxonomy_index,
            "max_examples": args.max_examples,
            "chunk_size": args.chunk_size,
            "resume": args.resume,
            "num_examples": len(examples),
        },
    }
    if args.num_shards > 1:
        metadata["sharding"] = {"shard": args.shard, "num_shards": args.num_shards}
    if args.method == "naive-sampling":
        metadata["naive_sampling"] = {"samples_per_example": args.samples_per_example}
    elif args.method == "iterative":
        metadata["iterative"] = {
            "attempts_per_round": args.attempts_per_round,
            "max_rounds": args.max_rounds,
            "enable_feedback": args.enable_feedback,
            "max_feedback_chars": args.max_feedback_chars,
        }

    with open(output_dir / f"{run_id}_metadata.json", "w") as mf:
        json.dump(metadata, mf, indent=2, ensure_ascii=False)
    # ────────────────────────────────────────────────────────────────

    if args.method == "iterative":
        if args.base_model:
            parser.error("--base-model is incompatible with --method iterative "
                         "(iterative requires chat-template-based conversation history)")
        # Iterative: chunk *outside* the round loop so each chunk runs
        # all its rounds and writes before moving on.
        n_chunks = (len(examples) + args.chunk_size - 1) // args.chunk_size
        for ci in range(n_chunks):
            s = ci * args.chunk_size
            e = min(s + args.chunk_size, len(examples))
            print(f"[chunk {ci + 1}/{n_chunks}] examples {s}–{e - 1}")
            run_iterative(
                llm, examples[s:e], images[s:e], args.prompt_variant,
                sampling_kwargs, args.max_tokens,
                args.attempts_per_round, args.max_rounds,
                args.enable_feedback, args.max_feedback_chars,
                output_path,
            )
    else:
        sampling_params = SamplingParams(**sampling_kwargs, max_tokens=args.max_tokens)
        n_chunks = (len(examples) + args.chunk_size - 1) // args.chunk_size

        for ci in range(n_chunks):
            s = ci * args.chunk_size
            e = min(s + args.chunk_size, len(examples))
            chunk_exs = examples[s:e]

            n_ex = len(chunk_exs)
            api_label = "llm.generate()" if args.base_model else "llm.chat()"
            print(f"[chunk {ci + 1}/{n_chunks}] {n_ex} examples × n={n} [{api_label}]")

            if args.base_model:
                chunk_prompts = raw_prompts[s:e]
                outputs = llm.generate(chunk_prompts, sampling_params, use_tqdm=True)
            else:
                chunk_convs = conversations[s:e]
                outputs = llm.chat(chunk_convs, sampling_params, use_tqdm=True)

            for example, request_output in zip(chunk_exs, outputs):
                all_texts = [co.text for co in request_output.outputs]
                ground_truth = example.get("answer", "")

                if args.method == "naive-sampling":
                    success = False
                    best_text = all_texts[0] if all_texts else ""
                    for t in all_texts:
                        if _matches_label(t, ground_truth):
                            success = True
                            best_text = t
                            break
                    prediction = best_text if success else (all_texts[0] if all_texts else "")
                    result = {
                        **example,
                        "prediction": prediction,
                        "method": "naive-sampling",
                        "prompt_variant": args.prompt_variant,
                        "sampling": f"temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, n={n}",
                        "n_samples": n,
                        "success": success,
                        "all_texts": all_texts,
                    }
                else:  # naive
                    prediction = all_texts[0] if all_texts else ""
                    result = {
                        **example,
                        "prediction": prediction,
                        "method": "naive",
                        "prompt_variant": args.prompt_variant,
                        "sampling": f"temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, n={n}",
                    }

                append_jsonl(output_path, result)

    print(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
