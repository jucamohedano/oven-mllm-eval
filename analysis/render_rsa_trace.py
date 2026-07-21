#!/usr/bin/env python3
"""Render a single RSA example as a per-iteration full-reasoning trace (LaTeX).

Layout produced (one example)::

    Question: <q>
    [input image]
    t=0: <one representative full reasoning trace>   [correct/incorrect]
    t=1: <one representative full reasoning trace>
    ...
    t=T-1: <final reasoning trace>
    Standard sampling (best of 256): <answer>  [incorrect]

RSA has no single chain (each step resamples K candidates and regenerates the
whole population), so for each step we show ONE representative trace: the trace
whose boxed answer equals that step's majority boxed answer (falling back to the
longest trace if no box is present at that step).

Input is a trace JSONL row that carries ``rsa_populations_by_step`` (pull from
the cluster judged file; e.g. viz/examples/traces/late_solve_traces_*.jsonl).
Standard-sampling context is read from the selection JSON (optional).

Usage::

    python scripts/render_rsa_trace.py \
      --trace viz/examples/traces/late_solve_traces_8B.jsonl \
      --data-id oven_entity_val_00078437 \
      --model 8B \
      --selection viz/examples/rsa_rescues_qwen.json \
      --image-dir data/images \
      --out-prefix viz/examples/rsa_trace_waterfall
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path


def _strip(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\\text\{(.*)\}$", r"\1", text)
    return text.replace("\\ ", " ").strip()


def boxed(text: str) -> str | None:
    """Last \\boxed{...} answer (nested-brace safe), or None."""
    matches, start = [], 0
    while True:
        i = text.find(r"\boxed{", start)
        if i < 0:
            break
        j, depth = i + 7, 1
        while j < len(text) and depth:
            depth += (text[j] == "{") - (text[j] == "}")
            j += 1
        if depth == 0:
            a = _strip(text[i + 7:j - 1])
            if a:
                matches.append(a)
            start = j
        else:
            break
    return matches[-1] if matches else None


def _representative(population: list[str]) -> tuple[str, str | None]:
    """Pick one trace representing the step: the one whose box == the majority
    box; if no boxes, the longest trace (most complete reasoning)."""
    boxes = [(t, boxed(t)) for t in population]
    parsed = [b for _, b in boxes if b]
    if parsed:
        maj = Counter(parsed).most_common(1)[0][0]
        for t, b in boxes:
            if b == maj:
                return t, b
    # no clean box this step: longest trace, no boxed answer
    longest = max(population, key=len)
    return longest, None


# Verbatim prompt templates from run_recursive_self_agg.py (candidate-format
# "solution").  The exact K candidates inserted at each aggregation step are not
# logged by the RSA run, so the aggregation prompt is shown as a template with a
# placeholder for the K sampled solution traces.

def initial_prompt(question: str) -> str:
    return "\n".join([
        "You are given an image and an open-world visual recognition problem.",
        "Write a concise solution that uses the image, the question, and relevant "
        "visual or world knowledge to identify the most specific entity name.",
        r"Reason carefully and end with the final answer in \boxed{}.",
        "",
        "Problem:",
        question.strip(),
        "",
        r"Now write a concise solution. End with the final answer in \boxed{}.",
    ])


def aggregation_prompt_template(question: str, k: int) -> str:
    parts = [
        "You are given an open-world visual recognition problem and several "
        "candidate solutions. Some candidates may be incorrect or contain errors. "
        "Aggregate the useful ideas and produce a single, high-quality solution. "
        "Reason carefully; if candidates disagree, choose the path best supported "
        "by the image and question. If all are incorrect, then attempt a different "
        "strategy. End with the final result in \\boxed{}.",
        "\nProblem:\n",
        question.strip() + "\n",
        "Candidate solutions (may contain mistakes):",
    ]
    for i in range(1, k + 1):
        parts.append(f"---- Solution {i} ----\n<full reasoning trace of candidate "
                     f"{i}, sampled from the previous population>")
    parts.append(r"Now write a single improved solution. Provide clear reasoning "
                 r"and end with the final answer in \boxed{}.")
    return "\n".join(parts)


def _tex(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
        .replace("#", r"\#").replace("_", r"\_")
        .replace("{", r"\{").replace("}", r"\}")
        .replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    )


def _tex_prompt(text: str) -> str:
    """Escape prompt text for a \\ttfamily block while showing the literal
    ``\\boxed{}`` token (prompts are quoted verbatim, so LaTeX-looking tokens
    stay readable)."""
    SENT = "\x00BOX\x00"
    text = text.replace(r"\boxed{}", SENT)
    text = _tex(text)
    # render the sentinel as a monospace literal \boxed{}
    return text.replace(SENT, r"{\textbackslash}boxed\{\}")


def _resolve_image(image_id: str, image_dir: Path) -> Path | None:
    for ext in (".jpg", ".jpeg", ".JPEG", ".JPG", ".png"):
        p = image_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Render one RSA example as a per-step reasoning trace")
    ap.add_argument("--trace", required=True, help="Trace JSONL with rsa_populations_by_step")
    ap.add_argument("--data-id", default=None, help="Which row (default: first row)")
    ap.add_argument("--model", default="", help="Model label, for the caption (e.g. 8B)")
    ap.add_argument("--selection", default=None, help="selection JSON for standard-sampling context")
    ap.add_argument("--image-dir", default="data/images")
    ap.add_argument("--max-chars", type=int, default=0,
                    help="Truncate each trace to N chars (0 = full text).")
    ap.add_argument("--show-prompts", action="store_true",
                    help="Include the initial (t=0) and aggregation (t>=1) prompt "
                         "templates. The exact K candidates are not logged, so the "
                         "aggregation prompt shows a placeholder for them.")
    ap.add_argument("--k", type=int, default=4,
                    help="K candidates per aggregation step (for the prompt "
                         "template; default 4, matching the n16_k4_t5 runs).")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    row = None
    for line in open(args.trace):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if args.data_id is None or r.get("data_id") == args.data_id:
            row = r
            break
    if row is None:
        raise SystemExit(f"data_id {args.data_id} not found in {args.trace}")

    q, gold = row["question"], row["answer"]
    image_id = row.get("image_id", "")
    steps = row["rsa_populations_by_step"]

    # Standard-sampling answer for the same example (optional).
    std_line = None
    if args.selection:
        sel = json.loads(Path(args.selection).read_text())
        ex = next((e for e in sel["examples"] if e["data_id"] == row.get("data_id")), None)
        if ex and args.model in ex.get("standard", {}):
            s = ex["standard"][args.model]
            mark = r"\cmark" if s["hit"] else r"\xmark"
            std_line = f"{_tex(s['best'])}~{mark}"

    out_prefix = Path(args.out_prefix)
    fig_dir = out_prefix.parent / f"{out_prefix.name}_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    src = _resolve_image(image_id, Path(args.image_dir))
    if src:
        shutil.copy(src, fig_dir / f"{image_id}{src.suffix}")
    else:
        print(f"[warn] image {image_id} not found in {args.image_dir}")

    lines = [
        r"% Requires: graphicx, pifont, enumitem",
        r"% \newcommand{\cmark}{\textcolor{green!60!black}{\ding{51}}}",
        r"% \newcommand{\xmark}{\textcolor{red!70!black}{\ding{55}}}",
        r"\begin{figure}[p]",
        r"\centering",
        rf"\includegraphics[width=0.5\linewidth]{{{fig_dir.name}/{image_id}}}",
        r"\par\medskip",
        rf"\textbf{{Question:}} {_tex(q)} \quad \textbf{{Ground truth:}} {_tex(gold)}",
        r"\par\medskip",
    ]

    if args.show_prompts:
        init_p = _tex_prompt(initial_prompt(q))
        agg_p = _tex_prompt(aggregation_prompt_template(q, args.k))
        lines += [
            r"{\scriptsize\ttfamily\raggedright",
            r"\noindent\textbf{\upshape\rmfamily Prompt at $t=0$ (initial "
            r"solution):}\\[-0.3em]",
            init_p.replace("\n", r"\\" + "\n"),
            r"\par\medskip",
            r"\noindent\textbf{\upshape\rmfamily Prompt at $t\geq 1$ (aggregation "
            rf"of $K={args.k}$ candidates; the exact candidates are resampled each "
            r"step and are not logged):}\\[-0.3em]",
            agg_p.replace("\n", r"\\" + "\n"),
            r"\par}",
            r"\medskip",
        ]

    lines += [
        r"{\footnotesize\raggedright",
        r"\begin{description}[leftmargin=3.2em,style=nextline]",
    ]
    for t, population in enumerate(steps):
        trace, box = _representative(population)
        truncated = args.max_chars and len(trace) > args.max_chars
        if truncated:
            trace = trace[:args.max_chars].rstrip()
        ok = (box is not None) and (
            box.lower().strip() == gold.lower().strip()
            or gold.lower() in box.lower()
        )
        mark = (r"~\cmark" if ok else r"~\xmark") if box else ""
        ans = f" \\hfill\\textit{{boxed: {_tex(box)}}}{mark}" if box else r" \hfill\textit{(no \boxed{})}"
        lines.append(rf"\item[$t={t}$]{ans}\\[-0.2em]")
        body = _tex(trace) + (r"\,\dots" if truncated else "")
        lines.append(body)
    lines.append(r"\end{description}")
    if std_line:
        lines.append(r"\medskip\noindent\textbf{Standard sampling (best of 256):} " + std_line)
    lines += [
        r"}",
        rf"\caption{{RSA per-iteration reasoning for {args.model} on "
        rf"\emph{{{_tex(q)}}} (ground truth: {_tex(gold)}). Each step shows one "
        rf"representative trace from the 16-candidate population.}}",
        rf"\label{{fig:rsa-trace-{row.get('data_id','x')}}}",
        r"\end{figure}",
    ]

    out_tex = out_prefix.with_name(f"{out_prefix.name}.tex")
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"[saved] {out_tex}")
    print(f"[saved] image  → {fig_dir}/")


if __name__ == "__main__":
    main()
