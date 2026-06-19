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
from datetime import datetime
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# Ensure project is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oven_mllm_eval.io import append_jsonl
from oven_mllm_eval.judge import (
    JUDGE_JSON_SCHEMA,
    build_judge_prompt,
    build_judge_prompt_free_form,
    parse_judge_output,
    parse_free_form_output,
    JudgeParseError,
)


def main():
    parser = argparse.ArgumentParser(description="Judge OVEN rollouts with a text-only LM")

    # Data I/O
    parser.add_argument("--input", required=True, help="Path to _samples.jsonl from Phase 1")
    parser.add_argument("--output", required=True, help="Path to _judged.jsonl output")

    # Model
    parser.add_argument("--judge-model", default="Qwen/Qwen3-8B",
                        help="Text-only judge model (default: Qwen/Qwen3-8B)")
    parser.add_argument("--max-tokens", type=int, default=16,
                        help="Max tokens per judge call (default: 16)")

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

    # Data-parallel sharding (mirrors run_inference.py)
    parser.add_argument("--shard", type=int, default=0,
                        help="This shard index (0-based, default: 0)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total shards. Each handles examples[shard::num_shards]")

    # Judge mode
    parser.add_argument("--judge-mode", default="structured",
                        choices=["structured", "free-form"],
                        help="Judging mode: structured (JSON, guidance backend) or "
                             "free-form (text-only, <answer>0/1</answer> tags) "
                             "(default: structured)")
    parser.add_argument("--judge-n", type=int, default=1,
                        help="Generations per judge prompt — n>1 enables majority "
                             "voting (default: 1)")
    parser.add_argument("--judge-temperature", type=float, default=0.0,
                        help="Judge temperature — set >0 for majority voting to "
                             "get diverse completions (default: 0.0)")
    parser.add_argument("--judge-top-p", type=float, default=1.0,
                        help="Judge top-p (nucleus sampling) — only used in free-form "
                             "mode (default: 1.0)")
    parser.add_argument("--judge-top-k", type=int, default=-1,
                        help="Judge top-k — only used in free-form mode. "
                             "-1 = disabled (default: -1)")

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

    # Strided sharding — balances load across GPUs (mirrors run_inference.py).
    if args.num_shards > 1:
        if not (0 <= args.shard < args.num_shards):
            parser.error(f"--shard must be in [0, {args.num_shards}), got {args.shard}")
        examples = examples[args.shard::args.num_shards]
        print(f"Shard {args.shard}/{args.num_shards}: {len(examples)} examples")

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
    mode = args.judge_mode
    n_gen = args.judge_n

    # Warn about n>1 with temperature=0 (deterministic → identical completions)
    if n_gen > 1 and args.judge_temperature == 0.0:
        print(
            "Warning: --judge-n > 1 with --judge-temperature 0.0 — "
            "completions will be identical, majority voting has no effect. "
            "Consider temperature ≥ 0.1 for diverse completions."
        )

    llm_kwargs: dict = dict(
        model=args.judge_model,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_util,
        trust_remote_code=True,
    )
    if mode == "structured":
        llm_kwargs["structured_outputs_config"] = {"backend": "guidance"}
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=args.max_tokens,
            structured_outputs=StructuredOutputsParams(json=JUDGE_JSON_SCHEMA),
            n=1,  # structured outputs: single completion, deterministic
        )
    else:  # free-form
        sampling_params = SamplingParams(
            temperature=args.judge_temperature,
            top_p=args.judge_top_p,
            top_k=args.judge_top_k,
            max_tokens=args.max_tokens,
            n=n_gen,
        )

    print(f"Initializing judge: model={args.judge_model} mode={mode} "
          f"n={sampling_params.n} temperature={sampling_params.temperature} "
          f"top_p={sampling_params.top_p} top_k={sampling_params.top_k} tp=1")
    llm = LLM(**llm_kwargs)

    # ------------------------------------------------------------------
    # Judge loop — flat batch per chunk for max GPU utilization
    #
    # Each chunk flattens ALL rollouts into one llm.generate() call so
    # vLLM's scheduler can fill max_num_seqs, keeping the GPU saturated.
    # Per-example results are reconstructed from the flat output list.
    # A crash loses at most one chunk (~1 min with chunk_size=256).
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.num_shards > 1:
        output_path = output_path.parent / f"{output_path.name}_shard{args.shard}.jsonl"

    # Resume: skip examples already written to the output file (this shard),
    # other shard files (enables GPU-count changes), and the merged output.
    done_ids: set[str] = set()
    for shard_file in sorted(output_path.parent.glob(f"{Path(args.output).stem}_shard*.jsonl")):
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
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done_ids.add(row.get("data_id", row.get("image_id", "")))
    # Also read the non-sharded merged output (e.g. _judged.jsonl)
    merged = output_path.parent / Path(args.output).name
    if merged.exists() and merged != output_path:
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

    n_chunks = (len(examples) + args.chunk_size - 1) // args.chunk_size
    total_examples = len(examples)
    processed = len(done_ids)
    parse_errors = 0
    maxed_out = 0       # completions that hit --max-tokens (truncated)

    print(f"Judging {total_examples} examples (chunk_size={args.chunk_size})"
          + (f", resuming from {processed}" if processed else ""))

    for ci in range(n_chunks):
        s = ci * args.chunk_size
        e = min(s + args.chunk_size, total_examples)
        chunk = [ex for ex in examples[s:e]
                 if ex.get("data_id", ex.get("image_id", "")) not in done_ids]
        if not chunk:
            continue
        now_str = datetime.now().strftime("%H:%M:%S")
        print(f"{now_str}  [chunk {ci + 1}/{n_chunks}] examples {s}–{e - 1}"
              + f" ({len(chunk)} remaining)")

        # ── Dedup: build unique prompts ─────────────────────────────
        # Naive-sampling produces many byte-identical rollouts for short
        # entity answers.  Judge each unique prompt once, then fan the
        # verdict back to every rollout that shares it.
        build_fn = (
            build_judge_prompt_free_form if mode == "free-form"
            else build_judge_prompt
        )
        prompt_to_pos: dict[str, int] = {}
        unique_prompts: list[str] = []
        plan: list[tuple[dict, list[int]]] = []  # (example, [pos for each rollout])

        for example in chunk:
            all_texts = example.get("all_texts", [])
            if not all_texts:
                plan.append((example, []))
                continue
            question = example.get("question", "")
            ground_truth = example.get("answer", "")
            positions: list[int] = []
            for rollout_text in all_texts:
                prompt = build_fn(question, ground_truth, rollout_text)
                pos = prompt_to_pos.get(prompt)
                if pos is None:
                    pos = len(unique_prompts)
                    prompt_to_pos[prompt] = pos
                    unique_prompts.append(prompt)
                positions.append(pos)
            plan.append((example, positions))

        total_rollouts = sum(len(p) for _, p in plan)
        n_unique = len(unique_prompts)
        if n_unique < total_rollouts:
            print(f"  dedup: {total_rollouts} rollouts → {n_unique} unique prompts "
                  f"({total_rollouts - n_unique} saved)")

        # ── Judge unique prompts ────────────────────────────────────
        if unique_prompts:
            if mode == "free-form":
                # llm.chat() applies the chat template so Qwen3 hits
                # its stop token; enable_thinking=False suppresses the
                # <think> block that would otherwise burn decode budget.
                conversations = [
                    [{"role": "user", "content": p}] for p in unique_prompts
                ]
                unique_outputs = llm.chat(
                    conversations,
                    sampling_params,
                    chat_template_kwargs={"enable_thinking": False},
                )
            else:
                unique_outputs = llm.generate(unique_prompts, sampling_params)
        else:
            unique_outputs = []

        # ── Fan results back to examples ────────────────────────────
        for example, positions in plan:
            if not positions:
                empty_row: dict = {
                    **example,
                    "judge_verdicts": [],
                    "judge_parse_ok": [],
                    "judge_hit": False,
                    "judge_hit_count": 0,
                    "judge_selected": -1,
                    "judge_selected_text": "",
                }
                if mode == "structured":
                    empty_row["judge_reasons"] = []
                append_jsonl(output_path, empty_row)
                processed += 1
                continue

            all_texts = example.get("all_texts", [])
            example_outputs = [unique_outputs[pos] for pos in positions]

            verdicts: list[bool] = []           # first-prediction
            reasons: list[str] = []
            parse_ok: list[bool] = []            # True = at least one completion parseable
            verdicts_majority: list[bool] = []  # majority-vote (n>1 only)
            votes_meta: list[dict] = []         # per-rollout voting metadata

            for output in example_outputs:
                if mode == "free-form":
                    # ── Free-form: parse <answer>0/1</answer> from each completion ──
                    # Unparseable completions (no <answer> tag) are *skipped*
                    # rather than counted as wrong, matching answer-matching
                    # behaviour (llm_judge.py:382).  Last completion is a
                    # fallback: if all n are unparseable, we keep that one
                    # (counted as False).
                    completions = output.outputs  # length == n_gen
                    votes: list[bool] = []
                    any_parseable = False
                    for j, completion in enumerate(completions):
                        if getattr(completion, "finish_reason", None) == "length":
                            maxed_out += 1
                        v, tag = parse_free_form_output(completion.text)
                        if tag == "" and j < n_gen - 1:
                            continue       # skip unparseable (not last)
                        any_parseable = True
                        votes.append(v)

                    first_v = votes[0]
                    verdicts.append(first_v)
                    parse_ok.append(any_parseable)
                    reasons.append(completions[0].text if completions else "")

                    if n_gen > 1:
                        yes_count = sum(votes)
                        no_count = len(votes) - yes_count
                        verdicts_majority.append(yes_count > no_count)
                        votes_meta.append({
                            "votes_yes": yes_count,
                            "votes_no": no_count,
                        })
                else:
                    completion = output.outputs[0]
                    if getattr(completion, "finish_reason", None) == "length":
                        maxed_out += 1
                    try:
                        v, r = parse_judge_output(completion.text)
                    except JudgeParseError:
                        parse_errors += 1
                        v = False
                        r = "(parse error)"
                    verdicts.append(v)
                    parse_ok.append(True)  # structured outputs are always parseable
                    reasons.append(r)

            judge_hit = any(verdicts)
            judge_hit_count = sum(verdicts)
            try:
                judge_selected = verdicts.index(True)
            except ValueError:
                judge_selected = -1
            judge_selected_text = (
                all_texts[judge_selected] if judge_selected >= 0
                else (all_texts[0] if all_texts else "")
            )

            row: dict = {
                **example,
                "judge_verdicts": verdicts,
                "judge_parse_ok": parse_ok,
                "judge_hit": judge_hit,
                "judge_hit_count": judge_hit_count,
                "judge_selected": judge_selected,
                "judge_selected_text": judge_selected_text,
            }
            if mode == "structured":
                row["judge_reasons"] = reasons
            else:
                row["judge_raw"] = reasons  # raw first-completion text for debugging

            if n_gen > 1:
                m_hit = any(verdicts_majority)
                m_count = sum(verdicts_majority)
                try:
                    m_selected = verdicts_majority.index(True)
                except ValueError:
                    m_selected = -1
                m_text = (
                    all_texts[m_selected] if m_selected >= 0
                    else (all_texts[0] if all_texts else "")
                )
                row["judge_verdicts_majority"] = verdicts_majority
                row["judge_hit_majority"] = m_hit
                row["judge_hit_count_majority"] = m_count
                row["judge_selected_majority"] = m_selected
                row["judge_selected_text_majority"] = m_text
                row["judge_votes"] = votes_meta

            append_jsonl(output_path, row)
            processed += 1

        now_str = datetime.now().strftime("%H:%M:%S")
        print(f"{now_str}  [chunk {ci + 1}/{n_chunks}] done — {processed}/{total_examples} examples")

    # ------------------------------------------------------------------
    # Write metadata
    # ------------------------------------------------------------------
    metadata_path = output_path.with_suffix("").with_name(
        f"{output_path.stem}_metadata.json"
    )
    with open(metadata_path, "w") as mf:
        json.dump({
            "judge_model": args.judge_model,
            "judge_mode": mode,
            "input": args.input,
            "num_examples": processed,
            "parse_errors": parse_errors,
            "maxed_out": maxed_out,
            "vllm": {
                "max_model_len": args.max_model_len,
                "max_num_seqs": args.max_num_seqs,
                "gpu_memory_utilization": args.gpu_util,
            },
            "sampling": {
                "temperature": sampling_params.temperature,
                "max_tokens": args.max_tokens,
                "n": n_gen,
                "structured_outputs": (mode == "structured"),
                "chat": (mode == "free-form"),
                "thinking_disabled": (mode == "free-form"),
                **({"backend": "guidance"} if mode == "structured" else {}),
            },
        }, mf, indent=2, ensure_ascii=False)

    if parse_errors:
        print(f"Warning: {parse_errors} parse errors")
    if maxed_out:
        print(f"Warning: {maxed_out} completions hit --max-tokens (truncated)")

    print(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()
