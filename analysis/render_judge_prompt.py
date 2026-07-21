#!/usr/bin/env python3
"""Render an example LM-judge prompt as a thesis LaTeX figure (verbatim).

Builds the ACTUAL judge prompt via the same code path run_judge.py uses
(``build_judge_prompt_free_form_with_desc`` for the evidence-augmented judge,
or ``build_judge_prompt_free_form`` for the no-evidence judge), for a chosen
example and response, and writes it inside a ``verbatim``-style LaTeX box.

Usage::

    python analysis/render_judge_prompt.py \
      --selection viz/examples/rsa_rescues_qwen.json \
      --data-id oven_entity_val_00025942 \
      --response "The image shows ... \\boxed{Eastern box turtle}" \
      --with-desc \
      --out viz/examples/judge_prompt_boxturtle.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oven_mllm_eval.judge import (
    build_judge_prompt_free_form,
    build_judge_prompt_free_form_with_desc,
)
from oven_mllm_eval.taxonomy_io import load_label_chains_from_index, load_taxonomy_chains


def _example_meta(selection: str, data_id: str) -> dict:
    sel = json.loads(Path(selection).read_text())
    ex = next((e for e in sel["examples"] if e["data_id"] == data_id), None)
    if ex is None:
        raise SystemExit(f"data_id {data_id} not in {selection}")
    return ex


def _verbatim_tex(prompt: str, caption: str, label: str) -> str:
    # fancyvrb's Verbatim wraps long lines and keeps the monospace prompt exact.
    lines = [
        r"% Requires: \usepackage{fancyvrb}  (Verbatim with line wrapping)",
        r"\begin{figure}[t]",
        r"\centering",
        r"\begin{Verbatim}[frame=single,fontsize=\scriptsize,breaklines=true,"
        r"breakanywhere=true]",
        prompt,
        r"\end{Verbatim}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{figure}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Render an example judge prompt to LaTeX")
    ap.add_argument("--selection", required=True, help="selection JSON (for question/gold/entity_id)")
    ap.add_argument("--data-id", required=True)
    ap.add_argument("--response", required=True, help="The rollout/response text being judged")
    ap.add_argument("--with-desc", action="store_true",
                    help="Use the evidence-augmented judge prompt (taxonomy "
                         "descriptions). Otherwise the no-evidence prompt.")
    ap.add_argument("--taxonomy-index", default="data/processed/oven_taxonomy_index.json")
    ap.add_argument("--desc-chains", default="data/raw/oven_wikidata_chains_cleaned_descs.jsonl")
    ap.add_argument("--label", default="fig:judge-prompt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ex = _example_meta(args.selection, args.data_id)
    question, gold, entity_id = ex["question"], ex["answer"], ex["entity_id"]

    if args.with_desc:
        label_chains = load_label_chains_from_index(args.taxonomy_index)
        desc_chains = load_taxonomy_chains(args.desc_chains)
        lc = label_chains.get(entity_id)
        dc = desc_chains.get(entity_id)
        prompt = build_judge_prompt_free_form_with_desc(question, gold, args.response, lc, dc)
        judge_desc = "evidence-augmented (taxonomy descriptions)"
    else:
        prompt = build_judge_prompt_free_form(question, gold, args.response)
        judge_desc = "no-evidence"

    caption = (f"Example prompt given to the free-form LM judge "
               f"({judge_desc} variant) for the question "
               f"\\emph{{{question}}} (ground truth: {gold}). The judge outputs a "
               r"binary verdict inside \texttt{<answer>...</answer>} tags.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(_verbatim_tex(prompt, caption, args.label))
    print(f"[saved] {args.out}")
    print("\n----- prompt preview -----\n")
    print(prompt)


if __name__ == "__main__":
    main()
