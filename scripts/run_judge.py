#!/usr/bin/env python3
"""Judge phase: text-only LM verifies rollout answers against ground truth.

Reads ``_samples.jsonl`` from Phase 1 (inference), runs every rollout
through an 8B text-only judge with guided JSON decoding, and writes
``_judged.jsonl`` with per-rollout verdicts + reasons.

Usage::

    uv run --extra serve python scripts/run_judge.py \
        --input logs/schedule/.../run_id_samples.jsonl \
        --output logs/schedule/.../run_id_judged.jsonl \
        --judge-model Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# Ensure project is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# Also make judge_lib importable from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge_lib import JUDGE_JSON_SCHEMA, build_judge_prompt, parse_judge_output, JudgeParseError
from oven_mllm_eval.io import append_jsonl


def main():
    parser = argparse.ArgumentParser(description="Judge OVEN rollouts with a text-only LM")

    # Data I/O
    parser.add_argument("--input", required=True, help="Path to _samples.jsonl from Phase 1")
    parser.add_argument("--output", required=True, help="Path to _judged.jsonl output")

    # Model
    parser.add_argument("--judge-model", default="Qwen/Qwen3-8B",
                        help="Text-only judge model (default: Qwen/Qwen3-8B)")
    parser.add_argument("--max-tokens", type=int, default=80,
                        help="Max tokens per judge call (default: 80)")

    # vLLM engine
    parser.add_argument("--max-model-len", type=int, default=2048,
                        help="Max model context length (default: 2048)")
    parser.add_argument("--max-num-seqs", type=int, default=1024,
                        help="Max concurrent sequences (default: 1024)")
    parser.add_argument("--gpu-util", type=float, default=0.92,
                        help="GPU memory utilization (default: 0.92)")

    # Batching
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="Examples per chunk — streams results to disk (default: 256)")

    # Limits
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit number of examples (for smoke tests)")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load samples
    # ------------------------------------------------------------------
    examples = []
    with open(args.input, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if args.max_examples:
        examples = examples[:args.max_examples]

    if not examples:
        print("No examples to judge — exiting.")
        return

    # Quick sanity check
    empty_all_texts = sum(1 for e in examples if not e.get("all_texts"))
    if empty_all_texts:
        print(f"Warning: {empty_all_texts}/{len(examples)} examples have empty all_texts — "
              "they will be written through with empty verdicts.")

    # ------------------------------------------------------------------
    # Initialize judge LLM
    # ------------------------------------------------------------------
    print(f"Initializing judge: model={args.judge_model} tp=1")
    llm = LLM(
        model=args.judge_model,
        tensor_parallel_size=1,
        enable_prefix_caching=True,        # batched rollouts share prefix
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_util,
        structured_outputs_config={"backend": "guidance"},  # avoids xgrammar hang
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0,                     # deterministic verdicts
        max_tokens=args.max_tokens,
        structured_outputs=StructuredOutputsParams(json=JUDGE_JSON_SCHEMA),
    )

    # ------------------------------------------------------------------
    # Judge loop — per-example batching for prefix-cache reuse
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_chunks = (len(examples) + args.chunk_size - 1) // args.chunk_size
    total_examples = len(examples)
    processed = 0
    parse_errors = 0

    print(f"Judging {total_examples} examples (chunk_size={args.chunk_size})")

    for ci in range(n_chunks):
        s = ci * args.chunk_size
        e = min(s + args.chunk_size, total_examples)
        chunk = examples[s:e]
        print(f"[chunk {ci + 1}/{n_chunks}] examples {s}–{e - 1}")

        for example in chunk:
            all_texts = example.get("all_texts", [])

            if not all_texts:
                # Empty rollouts — write through with empty verdicts
                append_jsonl(output_path, {
                    **example,
                    "judge_verdicts": [],
                    "judge_reasons": [],
                    "judge_hit": False,
                    "judge_hit_count": 0,
                    "judge_selected": -1,
                    "judge_selected_text": "",
                })
                processed += 1
                continue

            question = example.get("question", "")
            ground_truth = example.get("answer", "")

            # Batch all k rollouts into one llm.generate() call.
            # The shared prefix (instructions + question + ground_truth)
            # hits the KV cache for rollouts 2..k.
            prompts = [
                build_judge_prompt(question, ground_truth, rollout_text)
                for rollout_text in all_texts
            ]

            outputs = llm.generate(prompts, sampling_params)

            verdicts: list[bool] = []
            reasons: list[str] = []
            for output in outputs:
                try:
                    v, r = parse_judge_output(output.outputs[0].text)
                except JudgeParseError:
                    parse_errors += 1
                    v = False
                    r = "(parse error)"
                verdicts.append(v)
                reasons.append(r)

            judge_hit = any(verdicts)
            judge_hit_count = sum(verdicts)
            try:
                judge_selected = verdicts.index(True)
            except ValueError:
                judge_selected = -1
            judge_selected_text = (
                all_texts[judge_selected] if judge_selected >= 0
                else all_texts[0]
            )

            append_jsonl(output_path, {
                **example,
                "judge_verdicts": verdicts,
                "judge_reasons": reasons,
                "judge_hit": judge_hit,
                "judge_hit_count": judge_hit_count,
                "judge_selected": judge_selected,
                "judge_selected_text": judge_selected_text,
            })
            processed += 1

        print(f"[chunk {ci + 1}/{n_chunks}] done — {e}/{total_examples} examples")

    # ------------------------------------------------------------------
    # Write metadata
    # ------------------------------------------------------------------
    metadata_path = output_path.with_suffix("").with_name(
        f"{output_path.stem}_metadata.json"
    )
    with open(metadata_path, "w") as mf:
        json.dump({
            "judge_model": args.judge_model,
            "input": args.input,
            "num_examples": processed,
            "parse_errors": parse_errors,
            "vllm": {
                "max_model_len": args.max_model_len,
                "max_num_seqs": args.max_num_seqs,
                "gpu_memory_utilization": args.gpu_util,
            },
            "sampling": {
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "structured_outputs": True,
                "backend": "guidance",
            },
        }, mf, indent=2, ensure_ascii=False)

    if parse_errors:
        print(f"Warning: {parse_errors} parse errors (guided decoding should prevent these)")

    print(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()
