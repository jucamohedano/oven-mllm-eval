#!/usr/bin/env python3
"""Stage 2 (local): render selected examples into a thesis LaTeX figure+table.

Reads the ``selected_examples.json`` produced by ``select_examples.py`` and the
images pulled back with ``sync.sh --pull-images``, and writes, under
``viz/examples/``:

  * ``<prefix>_figures/<image_id>.jpg``  copies of the referenced images
  * ``<prefix>_main.tex``       one figure per example: the image + a compact
                                per-model "best answer" table (standard vs RSA,
                                with correct/incorrect marks and hF).
  * ``<prefix>_appendix.tex``   per example, a fuller table: top-N distinct
                                standard answers per model, and the RSA
                                per-iteration answer trace.

Usage::

    python scripts/render_examples.py \
      --selection viz/examples/selected_examples_qwen.json \
      --image-dir data/images \
      --out-prefix viz/examples/rsa_wins_qwen
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


CHECK = r"\cmark"   # define in preamble: \newcommand{\cmark}{\ding{51}}
CROSS = r"\xmark"   #                       \newcommand{\xmark}{\ding{55}}


def _tex_escape(text: str) -> str:
    if text is None:
        return ""
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
        .replace("#", r"\#").replace("_", r"\_")
        .replace("{", r"\{").replace("}", r"\}")
        .replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    )


def _mark(correct: bool) -> str:
    return CHECK if correct else CROSS


def _resolve_image(image_id: str, image_dir: Path) -> Path | None:
    for ext in (".jpg", ".jpeg", ".JPEG", ".JPG", ".png"):
        p = image_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def _fmt_hf(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "--"


def render_main(examples, models_std, models_rsa, fig_reldir) -> str:
    """One figure per example: image + compact best-answer table."""
    lines = ["% Requires: booktabs, graphicx, pifont",
             r"% \newcommand{\cmark}{\textcolor{green!60!black}{\ding{51}}}",
             r"% \newcommand{\xmark}{\textcolor{red!70!black}{\ding{55}}}",
             ""]
    for i, ex in enumerate(examples, 1):
        q = _tex_escape(ex["question"])
        ans = _tex_escape(ex["answer"])
        img_id = ex["image_id"]
        rsa_present = bool(ex.get("rsa"))
        ncols = "ll" + ("l" if rsa_present else "")
        head = ["Model", "Standard best"]
        if rsa_present:
            head.append("RSA best")
        lines += [
            r"\begin{figure}[t]",
            r"\centering",
            rf"\includegraphics[width=0.42\linewidth]{{{fig_reldir}/{img_id}}}",
            r"\\[0.5em]",
            r"{\small",
            rf"\begin{{tabular}}{{{ncols}}}",
            r"\toprule",
            " & ".join(head) + r" \\",
            r"\midrule",
        ]
        for m in models_std:
            s = ex["standard"].get(m)
            if not s:
                continue
            std_cell = f"{_tex_escape(s['best'])} {_mark(s['hit'])}"
            row = [m, std_cell]
            if rsa_present:
                r = ex["rsa"].get(m)
                row.append(f"{_tex_escape(r['best'])} {_mark(r['hit'])}" if r else "--")
            lines.append(" & ".join(row) + r" \\")
        lines += [
            r"\bottomrule",
            r"\end{tabular}}",
            rf"\caption{{\textbf{{Q:}} {q} \quad \textbf{{Ground truth:}} {ans}. "
            rf"Standard sampling (best of 256 rollouts) vs.\ recursive self-aggregation "
            rf"(RSA, final answer). \cmark/\xmark\ = judge correct/incorrect.}}",
            rf"\label{{fig:example-{i}}}",
            r"\end{figure}",
            "",
        ]
    return "\n".join(lines)


def render_appendix(examples, models_std, models_rsa) -> str:
    """Per example: top-N standard answers + RSA iteration trace."""
    lines = ["% Requires: booktabs", ""]
    for i, ex in enumerate(examples, 1):
        q = _tex_escape(ex["question"])
        ans = _tex_escape(ex["answer"])
        lines += [
            rf"\paragraph{{Example {i}: {q} (GT: {ans})}}",
            "",
            r"\noindent\textit{Standard sampling --- top distinct answers "
            r"(count, \cmark\ if judged correct):}",
            "",
            r"{\small",
            r"\begin{tabular}{ll}",
            r"\toprule",
            r"Model & Top answers \\",
            r"\midrule",
        ]
        for m in models_std:
            s = ex["standard"].get(m)
            if not s:
                continue
            parts = []
            for t in s["top"]:
                mk = CHECK if t["correct"] else ""
                parts.append(f"{_tex_escape(t['answer'])}~($\\times${t['count']}){mk}")
            lines.append(f"{m} & " + ", ".join(parts) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}}", ""]

        # RSA iteration trace (if present)
        if ex.get("rsa"):
            lines += [
                r"\noindent\textit{RSA --- majority answer per iteration "
                r"(step 0 = initial population):}",
                "",
                r"{\small",
                r"\begin{tabular}{l" + "l" * 8 + r"}",
                r"\toprule",
            ]
            # Determine max steps across models
            max_steps = max(
                (r["num_steps"] for r in ex["rsa"].values()), default=0
            )
            hdr = ["Model"] + [f"t={s}" for s in range(max_steps)]
            lines += [" & ".join(hdr) + r" \\", r"\midrule"]
            for m in models_rsa:
                r = ex["rsa"].get(m)
                if not r:
                    continue
                cells = [m]
                for it in r["iterations"]:
                    # Prefer the recorded majority + correctness; fall back to top.
                    ans = it.get("majority") or (it["top"][0]["answer"] if it["top"] else "")
                    mk = _mark(it["majority_correct"]) if "majority_correct" in it else ""
                    cell = f"{_tex_escape(ans)}~{mk}".strip() if ans else "--"
                    cells.append(cell)
                cells += ["--"] * (1 + max_steps - len(cells))
                lines.append(" & ".join(cells[:1 + max_steps]) + r" \\")
            lines += [r"\bottomrule", r"\end{tabular}}"]
            solve = {m: ex["rsa"][m].get("solve_step") for m in models_rsa if m in ex["rsa"]}
            if any(v is not None for v in solve.values()):
                notes = ", ".join(
                    f"{m}: first correct at $t{{=}}{v}$" for m, v in solve.items()
                    if v is not None
                )
                lines.append(rf"\noindent\footnotesize {notes}.")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render selected examples to LaTeX (Stage 2, local)")
    ap.add_argument("--selection", required=True, help="selected_examples.json from Stage 1")
    ap.add_argument("--image-dir", default="data/images",
                    help="Local dir with pulled images (default: data/images)")
    ap.add_argument("--out-prefix", default="viz/examples/examples",
                    help="Output path prefix under viz/examples/")
    ap.add_argument("--skip-missing-images", action="store_true",
                    help="Drop examples whose image is not present locally "
                         "(instead of emitting a figure with a broken path). "
                         "Use when you over-selected and some images could not "
                         "be pulled.")
    ap.add_argument("--max-examples", type=int, default=None,
                    help="Cap the number of rendered examples (after "
                         "--skip-missing-images filtering).")
    args = ap.parse_args()

    sel = json.loads(Path(args.selection).read_text())
    examples = sel["examples"]
    models_std = sel["models_standard"]
    models_rsa = sel.get("models_rsa", [])
    image_dir = Path(args.image_dir)

    if args.skip_missing_images:
        kept = [ex for ex in examples if _resolve_image(ex["image_id"], image_dir)]
        dropped = len(examples) - len(kept)
        if dropped:
            print(f"[skip] dropped {dropped} example(s) with no local image")
        examples = kept
    if args.max_examples is not None:
        examples = examples[:args.max_examples]
    if not examples:
        raise SystemExit("No examples left to render (all images missing?).")

    out_prefix = Path(args.out_prefix)
    fig_dir = out_prefix.parent / f"{out_prefix.name}_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Copy images; warn on any missing so the user knows to pull them.
    missing = []
    for ex in examples:
        src = _resolve_image(ex["image_id"], image_dir)
        if src is None:
            missing.append((ex["image_id"], ex["entity_id"]))
            continue
        shutil.copy(src, fig_dir / f"{ex['image_id']}{src.suffix}")
    if missing:
        print("[warn] missing images (pull with sync.sh --pull-images):")
        for img_id, qid in missing:
            print(f"   {qid}  ({img_id})")
        print("   QIDs: " + ",".join(q for _, q in missing if q))

    # LaTeX uses a path relative to the .tex location; both live under viz/examples/.
    fig_reldir = fig_dir.name

    main_tex = render_main(examples, models_std, models_rsa, fig_reldir)
    appendix_tex = render_appendix(examples, models_std, models_rsa)

    main_path = out_prefix.with_name(f"{out_prefix.name}_main.tex")
    appendix_path = out_prefix.with_name(f"{out_prefix.name}_appendix.tex")
    main_path.write_text(main_tex)
    appendix_path.write_text(appendix_tex)
    print(f"[saved] {main_path}")
    print(f"[saved] {appendix_path}")
    print(f"[saved] images → {fig_dir}/")


if __name__ == "__main__":
    main()
