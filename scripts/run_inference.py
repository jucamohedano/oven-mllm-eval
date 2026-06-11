#!/usr/bin/env python3
"""Run inference on OVEN using vLLM's offline ``LLM.chat()`` API.

Supports three sampling modes:

  1. **naive**: single sample per example (n=1).
  2. **naive-sampling**: draw N independent samples, write all rollouts
     (verdicts deferred to the judge model).
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
import gc
import json
import secrets
import sys
import time
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
# Helpers
# ---------------------------------------------------------------------------

def _raise_missing_question(example: dict):
    """Called when 'question' key is missing — raises a clear error."""
    raise KeyError(
        f"Example missing 'question' field. Available keys: {sorted(example.keys())}. "
        f"Re-run prepare_oven.py with the correct raw data that includes 'question'."
    )


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

def _load_pil(path: str) -> Image.Image:
    """Load an image as RGB PIL, raising on missing or unreadable files."""
    if not path:
        raise ValueError("Empty image_path in example")
    p = Path(path)
    if not p.exists():
        for ext in (".JPEG", ".jpeg", ".JPG"):
            alt = p.with_suffix(ext)
            if alt.exists():
                p = alt
                break
        else:
            raise FileNotFoundError(f"Image not found: {p.resolve()} (cwd={Path.cwd()})")
    img = Image.open(p)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def _resolve_image_path(path: str, root: Path) -> str:
    """Resolve a (possibly relative) image path and verify it exists.

    Cheap preflight (stat only, no decode) so a missing file fails the run
    immediately instead of at chunk N.  Returns the resolved path string.
    """
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


def _load_images(paths: list[str]) -> list[Image.Image]:
    """Decode a batch of images in parallel.

    Called per chunk (NOT upfront for the whole dataset): PIL pixel buffers
    for 60k+ images would otherwise accumulate in host RAM as vLLM touches
    them, growing RSS monotonically until the SLURM cgroup OOM-kills the
    engine core process.  Loading per chunk keeps the working set to one
    chunk and lets the GC reclaim it after each llm.chat() call.
    """
    with ThreadPoolExecutor(max_workers=16) as pool:
        return list(pool.map(_load_pil, paths))


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
        content.append({"type": "image_pil", "image_pil": image})
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
    use_tqdm: bool = True,
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
            content.append({"type": "image_pil", "image_pil": img})
        content.append({"type": "text", "text": prompt_text})
        states.append({
            "idx": i,
            "conversation": [{"role": "user", "content": content}],
            "all_attempts": [],
            "success": None,
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

        outputs = llm.chat(conversations, round_sampling, use_tqdm=use_tqdm)

        newly_succeeded = []
        for state, request_output in zip(active, outputs):
            texts = [co.text for co in request_output.outputs]
            failed_texts: list[str] = []

            for attempt_idx, text in enumerate(texts, start=1):
                state["total_attempts"] += 1
                # Verdict deferred to judge (no _matches_label gate).
                state["all_attempts"].append({
                    "round": round_idx, "attempt": attempt_idx,
                    "text": text, "matched": None,
                })
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
    # Verdict deferred to judge; use first rollout as placeholder.
    prediction = state["all_attempts"][0]["text"] if state["all_attempts"] else ""
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
    parser.add_argument("--image-root", default=None,
                        help="Root directory for resolving relative image_path. "
                             "Defaults to cwd.")
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
    parser.add_argument("--async-scheduling", action="store_true",
                        help="Enable vLLM async scheduling. Off by default: there are "
                             "known EngineCore crash reports for Qwen3-VL + multimodal "
                             "with async scheduling enabled on 0.11.x.")
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
    parser.add_argument("--restart-every", type=int, default=0,
                        help="Restart vLLM engine every N chunks to avoid memory leak (0 = never)")
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

    # Progress bars use \r redraws that collide when multiple shards share stdout.
    show_tqdm = args.num_shards <= 1

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
        valid_lines: list[str] = []
        n_bad = 0
        with open(output_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A SIGKILL (e.g. cgroup OOM) mid-write can truncate the
                    # final line.  Drop it so the example re-runs and
                    # downstream readers (merge/judge/score) don't choke.
                    n_bad += 1
                    continue
                valid_lines.append(line)
                done_ids.add(row.get("data_id", row.get("image_id", "")))
        if n_bad:
            print(f"Resuming: dropping {n_bad} malformed line(s) from {output_path}")
            tmp = output_path.with_name(output_path.name + ".tmp")
            with open(tmp, "w") as f:
                for line in valid_lines:
                    f.write(line + "\n")
            tmp.replace(output_path)
        examples = [e for e in examples if e.get("data_id", e.get("image_id", "")) not in done_ids]
        print(f"Resuming: {len(done_ids)} already done, {len(examples)} remaining")

    # Resolve image paths — relative paths break when cwd != project root
    # (SLURM, DP shard processes).  Defaults to cwd; use --image-root to override.
    #
    # IMPORTANT: we only *resolve and stat* paths here.  Decoding all images
    # upfront (and keeping them referenced via per-example conversations)
    # makes host RSS grow chunk after chunk as PIL lazily materialises pixel
    # buffers — which looks exactly like an engine "memory leak" and ends in
    # the cgroup OOM-killer SIGKILLing the EngineCore process.  Images are
    # now decoded per chunk inside the inference loop.
    image_root = Path(args.image_root) if args.image_root else Path.cwd()
    print(f"Resolving image paths... (root: {image_root})")
    with ThreadPoolExecutor(max_workers=16) as pool:
        resolved_paths = list(pool.map(
            lambda p: _resolve_image_path(p, image_root),
            [e.get("image_path", "") for e in examples],
        ))

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

    def _build_engine() -> LLM:
        return LLM(
            model=args.model,
            tensor_parallel_size=args.tp,
            gpu_memory_utilization=args.gpu_util,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            enable_prefix_caching=True,
            enforce_eager=args.enforce_eager,
            limit_mm_per_prompt={"image": 1},
            mm_processor_cache_gb=0,
            async_scheduling=args.async_scheduling,
            trust_remote_code=True,
            **extra_llm_kwargs,
        )

    def _restart_engine(old: LLM) -> LLM:
        """Tear down the engine as thoroughly as in-process restart allows.

        NOTE: this is best-effort.  In vLLM V1 the engine core runs in a
        child process and `del llm` does not synchronously release GPU
        memory, nor does it reclaim anything leaked in *this* (parent)
        process where multimodal preprocessing runs.  For long jobs prefer
        the process-level restart loop in schedule_sbatch.sh (--resume).
        """
        del old
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        time.sleep(10)  # give the old EngineCore process time to exit
        return _build_engine()

    llm = _build_engine()

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
            "async_scheduling": args.async_scheduling,
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
            # Periodic engine restart to work around vLLM 0.11.2 memory leak
            if args.restart_every and ci > 0 and ci % args.restart_every == 0:
                print(f"[restart] reinitializing engine after {args.restart_every} chunks", flush=True)
                llm = _restart_engine(llm)
            s = ci * args.chunk_size
            e = min(s + args.chunk_size, len(examples))
            print(f"[chunk {ci + 1}/{n_chunks}] examples {s}–{e - 1}")
            chunk_images = _load_images(resolved_paths[s:e])
            run_iterative(
                llm, examples[s:e], chunk_images, args.prompt_variant,
                sampling_kwargs, args.max_tokens,
                args.attempts_per_round, args.max_rounds,
                args.enable_feedback, args.max_feedback_chars,
                output_path,
                use_tqdm=show_tqdm,
            )
            del chunk_images
    else:
        sampling_params = SamplingParams(**sampling_kwargs, max_tokens=args.max_tokens)
        n_chunks = (len(examples) + args.chunk_size - 1) // args.chunk_size

        for ci in range(n_chunks):
            # Periodic engine restart to work around vLLM 0.11.2 memory leak
            if args.restart_every and ci > 0 and ci % args.restart_every == 0:
                print(f"[restart] reinitializing engine after {args.restart_every} chunks", flush=True)
                llm = _restart_engine(llm)

            s = ci * args.chunk_size
            e = min(s + args.chunk_size, len(examples))
            chunk_exs = examples[s:e]

            n_ex = len(chunk_exs)
            api_label = "llm.generate()" if args.base_model else "llm.chat()"
            print(f"[chunk {ci + 1}/{n_chunks}] {n_ex} examples × n={n} [{api_label}]")

            # Decode this chunk's images and build prompts now — and only
            # now — so their memory can be reclaimed after the chunk.
            chunk_images = _load_images(resolved_paths[s:e])
            if args.base_model:
                chunk_prompts = [
                    _make_raw_prompt(ex, img, args.prompt_variant)
                    for ex, img in zip(chunk_exs, chunk_images)
                ]
                outputs = llm.generate(chunk_prompts, sampling_params, use_tqdm=show_tqdm)
            else:
                chunk_convs = [
                    _make_conversation(ex, img, args.prompt_variant)
                    for ex, img in zip(chunk_exs, chunk_images)
                ]
                outputs = llm.chat(chunk_convs, sampling_params, use_tqdm=show_tqdm)

            for example, request_output in zip(chunk_exs, outputs):
                all_texts = [co.text for co in request_output.outputs]

                if args.method == "naive-sampling":
                    prediction = all_texts[0] if all_texts else ""
                    # Verdict deferred to judge.
                    result = {
                        **example,
                        "prediction": prediction,
                        "method": "naive-sampling",
                        "prompt_variant": args.prompt_variant,
                        "sampling": f"temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, n={n}",
                        "n_samples": n,
                        "success": None,
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

            # Drop chunk-local references so PIL buffers, processed mm
            # inputs and RequestOutputs are reclaimable before next chunk.
            del outputs, chunk_images
            print(f"[chunk {ci + 1}/{n_chunks}] done — {e}/{len(examples)} examples", flush=True)

    print(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    main()