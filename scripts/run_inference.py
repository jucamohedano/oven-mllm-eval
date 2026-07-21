#!/usr/bin/env python3
"""Run inference on OVEN using vLLM's offline ``LLM.chat()`` API.

Supports three sampling modes:

  1. **naive**: single sample per example (n=1).
  2. **naive-sampling**: draw N independent samples, write all rollouts
     (verdicts deferred to the judge model).

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

from oven_mllm_eval.io import append_jsonl, write_jsonl
from oven_mllm_eval.images import load_pil, resolve_image_path, load_images
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

# ---------------------------------------------------------------------------
# Image loading — see src/oven_mllm_eval/images.py (shared with the judge)
# ---------------------------------------------------------------------------

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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run OVEN inference via vLLM offline")

    # Data I/O
    parser.add_argument("--input", required=True, help="Input JSONL (prepared OVEN data)")
    parser.add_argument("--image-root", default=None,
                        help="Root directory for resolving relative image_path. "
                             "Defaults to cwd.")
    parser.add_argument("--no-image", action="store_true",
                        help="Text-only baseline: skip image loading, send only the question.")
    parser.add_argument("--taxonomy-index", default="data/processed/oven_taxonomy_index.json",
                        help="Path to taxonomy index JSON")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: auto-generated from model/method/prompt)")
    parser.add_argument("--output-root", default="logs/schedule",
                        help="Root directory for auto-generated output paths (default: logs/schedule)")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct", help="Model path or HF ID")
    parser.add_argument("--prompt-variant", default="barebones", choices=list(PROMPT_VARIANTS.keys()))
    parser.add_argument("--method", default="naive", choices=["naive", "naive-sampling"])
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
                             "Overridden by --samples-per-example for naive-sampling.")

    # naive-sampling
    parser.add_argument("--samples-per-example", type=int, default=64)


    # vLLM engine
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism (default: 1)")
    parser.add_argument("--gpu-util", type=float, default=0.92, help="GPU memory utilization (default: 0.92)")
    parser.add_argument("--max-model-len", type=int, default=1024, help="Max model context length (default: 1024)")
    parser.add_argument("--max-num-seqs", type=int, default=1024, help="Max number of sequences (default: 1024)")
    parser.add_argument("--max-pixels", type=int, default=512 * 512,
                        help="Max pixels for image resizing (default: 262144 = 512x512)")
    parser.add_argument("--min-pixels", type=int, default=256 * 256,
                        help="Min pixels for image resizing (default: 65536 = 256x256)")
    parser.add_argument("--image-workers", type=int, default=16,
                        help="Threads used for per-chunk PIL image decoding (default: 16)")
    parser.add_argument("--prefetch-images", action="store_true",
                        help="Decode the next image chunk while the current chunk is in vLLM. "
                             "Improves GPU utilization at the cost of one extra chunk of host RAM.")
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
    if args.image_workers <= 0:
        parser.error(f"--image-workers must be > 0, got {args.image_workers}")

    # Determine n from method
    n = args.n
    if args.method == "naive-sampling":
        n = args.samples_per_example

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
    input_num_examples = len(examples)

    # Strided sharding — balances load even if the file is ordered by category/size.
    if args.num_shards > 1:
        if not (0 <= args.shard < args.num_shards):
            parser.error(f"--shard must be in [0, {args.num_shards}), got {args.shard}")
        examples = examples[args.shard::args.num_shards]
        print(f"Shard {args.shard}/{args.num_shards}: {len(examples)} examples")
    shard_num_examples = len(examples)

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

    # Resume support — read ALL shard files for a global done set, then
    # filter before loading images.  This allows a crashed DP shard to
    # resume on a different GPU count (the remaining work is split across
    # whatever --num-shards is currently set to).
    resume_already_done = 0
    if args.resume:
        done_ids: set[str] = set()
        # Collect IDs from every _shard*.jsonl file (global resume set)
        for shard_file in sorted(output_dir.glob(f"{run_id}_samples_shard*.jsonl")):
            if not shard_file.exists():
                continue
            with open(shard_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    done_ids.add(row.get("data_id", row.get("image_id", "")))
        # Also read the non-sharded output (DP=1 path)
        merged = output_dir / f"{run_id}_samples.jsonl"
        if merged.exists():
            with open(merged, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    done_ids.add(row.get("data_id", row.get("image_id", "")))

        # Clean up truncated lines in THIS shard's file
        if output_path.exists():
            valid_lines: list[str] = []
            n_bad = 0
            with open(output_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json.loads(line)
                        valid_lines.append(line)
                    except json.JSONDecodeError:
                        n_bad += 1
            if n_bad:
                print(f"Resuming: dropping {n_bad} malformed line(s) from {output_path}")
                tmp = output_path.with_name(output_path.name + ".tmp")
                with open(tmp, "w") as f:
                    for line in valid_lines:
                        f.write(line + "\n")
                tmp.replace(output_path)

        examples = [e for e in examples if e.get("data_id", e.get("image_id", "")) not in done_ids]
        resume_already_done = len(done_ids)
        print(f"Resuming: {len(done_ids)} already done, {len(examples)} remaining")

    if not examples:
        print("No examples to process — exiting.")
        return

    # Resolve image paths — relative paths break when cwd != project root
    # (SLURM, DP shard processes).  Defaults to cwd; use --image-root to override.
    #
    # IMPORTANT: we only *resolve and stat* paths here.  Decoding all images
    # upfront (and keeping them referenced via per-example conversations)
    # makes host RSS grow chunk after chunk as PIL lazily materialises pixel
    # buffers — which looks exactly like an engine "memory leak" and ends in
    # the cgroup OOM-killer SIGKILLing the EngineCore process.  Images are
    # now decoded per chunk inside the inference loop.
    if args.no_image:
        print("Text-only baseline — no images loaded.")
        resolved_paths = ["" for _ in examples]
    else:
        image_root = Path(args.image_root) if args.image_root else Path.cwd()
        print(f"Resolving image paths... (root: {image_root})")
        with ThreadPoolExecutor(max_workers=16) as pool:
            resolved_paths = list(pool.map(
                lambda p: resolve_image_path(p, image_root),
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
            # TP>1 with CUDA graphs can hit custom_all_reduce.cuh crashes
            # on some A100 topologies. Falls back to NCCL all-reduce.
            disable_custom_all_reduce=(args.tp > 1),
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
            "image_workers": args.image_workers,
            "prefetch_images": args.prefetch_images,
            "enforce_eager": args.enforce_eager,
            "async_scheduling": args.async_scheduling,
        },
        "data": {
            "input": args.input,
            "taxonomy_index": args.taxonomy_index,
            "max_examples": args.max_examples,
            "chunk_size": args.chunk_size,
            "resume": args.resume,
            "input_num_examples": input_num_examples,
            "shard_num_examples": shard_num_examples,
            "resume_already_done": resume_already_done,
            "resume_remaining_examples": len(examples),
            "no_image": args.no_image,
            "num_examples": len(examples),
        },
    }
    if args.num_shards > 1:
        metadata["sharding"] = {"shard": args.shard, "num_shards": args.num_shards}
    if args.method == "naive-sampling":
        metadata["naive_sampling"] = {"samples_per_example": args.samples_per_example}

    with open(output_dir / f"{run_id}_metadata.json", "w") as mf:
        json.dump(metadata, mf, indent=2, ensure_ascii=False)
    # ────────────────────────────────────────────────────────────────

    sampling_params = SamplingParams(**sampling_kwargs, max_tokens=args.max_tokens)
    n_chunks = (len(examples) + args.chunk_size - 1) // args.chunk_size
    prefetch_enabled = args.prefetch_images and not args.no_image
    prefetch_pool: ThreadPoolExecutor | None = None
    next_images_future = None

    if prefetch_enabled:
        prefetch_pool = ThreadPoolExecutor(max_workers=1)
        first_end = min(args.chunk_size, len(examples))
        next_images_future = prefetch_pool.submit(
            load_images,
            resolved_paths[:first_end],
            args.image_workers,
        )

    try:
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
            load_start = time.perf_counter()
            if args.no_image:
                chunk_images = [None] * (e - s)
            elif prefetch_enabled:
                assert next_images_future is not None
                chunk_images = next_images_future.result()
                next_images_future = None
            else:
                chunk_images = load_images(resolved_paths[s:e], args.image_workers)
            load_seconds = time.perf_counter() - load_start

            if prefetch_enabled and prefetch_pool is not None and e < len(examples):
                next_e = min(e + args.chunk_size, len(examples))
                next_images_future = prefetch_pool.submit(
                    load_images,
                    resolved_paths[e:next_e],
                    args.image_workers,
                )

            prompt_start = time.perf_counter()
            if args.base_model:
                chunk_prompts = [
                    _make_raw_prompt(ex, img, args.prompt_variant)
                    for ex, img in zip(chunk_exs, chunk_images)
                ]
                prompt_seconds = time.perf_counter() - prompt_start
                infer_start = time.perf_counter()
                outputs = llm.generate(chunk_prompts, sampling_params, use_tqdm=show_tqdm)
            else:
                chunk_convs = [
                    _make_conversation(ex, img, args.prompt_variant)
                    for ex, img in zip(chunk_exs, chunk_images)
                ]
                prompt_seconds = time.perf_counter() - prompt_start
                infer_start = time.perf_counter()
                outputs = llm.chat(chunk_convs, sampling_params, use_tqdm=show_tqdm)
            infer_seconds = time.perf_counter() - infer_start

            format_start = time.perf_counter()
            chunk_results = []
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

                chunk_results.append(result)
            format_seconds = time.perf_counter() - format_start

            write_start = time.perf_counter()
            write_jsonl(output_path, chunk_results, append=True)
            write_seconds = time.perf_counter() - write_start

            # Drop chunk-local references so PIL buffers, processed mm
            # inputs and RequestOutputs are reclaimable before next chunk.
            del outputs, chunk_images, chunk_results
            load_label = "load_wait" if prefetch_enabled else "load"
            print(
                f"[chunk {ci + 1}/{n_chunks}] done — {e}/{len(examples)} examples "
                f"({load_label}={load_seconds:.1f}s prompt={prompt_seconds:.1f}s "
                f"infer={infer_seconds:.1f}s format={format_seconds:.1f}s "
                f"write={write_seconds:.1f}s)",
                flush=True,
            )
    finally:
        if prefetch_pool is not None:
            prefetch_pool.shutdown(wait=False, cancel_futures=True)

    print(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
