#!/usr/bin/env python3
"""Interactive dashboard to explore model rollouts and judge verdicts.

Usage::

    uv run streamlit run scripts/explore_judgments.py -- \
        --scored logs/schedule/.../2b_run/*_scored.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

from oven_mllm_eval.judge_audit import (
    build_alias_map,
    classify_positive,
    is_supported,
)


# Streamlit is imported inside main() so the --help flag works without it installed.


def _load(scored_path: str, max_examples: int | None) -> list[dict]:
    rows = []
    with open(scored_path) as f:
        for line in f:
            rows.append(json.loads(line.strip()))
            if max_examples and len(rows) >= max_examples:
                break
    return rows


def main():
    parser = argparse.ArgumentParser(description="Explore judge verdicts interactively")
    parser.add_argument("--scored", default=os.environ.get("EXPLORE_SCORED", ""),
                        help="Path to _scored.jsonl")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit examples loaded (for faster startup)")
    parser.add_argument("--taxonomy-index",
                        default="data/processed/oven_taxonomy_index.json",
                        help="Taxonomy index for alias-aware support classification "
                             "(optional; alias matching is skipped if missing)")
    args, _ = parser.parse_known_args()

    if not args.scored:
        print("Usage: --scored <path> or set EXPLORE_SCORED env var")
        sys.exit(1)

    # Load alias map once for classify_positive (degrades gracefully if absent).
    index_path = Path(args.taxonomy_index)
    aliases_by_canonical = (
        build_alias_map(json.loads(index_path.read_text()))
        if index_path.is_file() else {}
    )

    import altair as alt
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="Judge Verdict Explorer", layout="wide")

    # ── Load ────────────────────────────────────────────────────────
    @st.cache_data
    def load_data(path, max_ex):
        return _load(path, max_ex)

    rows = load_data(args.scored, args.max_examples)
    st.sidebar.title("Data")
    st.sidebar.write(f"Loaded {len(rows):,} examples")

    # ── Support classification of judge-hit examples ────────────────
    # Classify each judge-positive example by its judge-selected prediction,
    # using the same rule as the audit (is_supported / under-specific). This is
    # what surfaces the note-001 mechanism: under-specific = hypernym answers the
    # judge accepted but that are NOT counted as supported.
    def _selected_pred(r):
        return r.get("judge_selected_text") or r.get("prediction") or ""

    support_by_id = {
        r["data_id"]: classify_positive(
            prediction=_selected_pred(r),
            answer=r.get("answer", ""),
            aliases_by_canonical=aliases_by_canonical,
        )
        for r in rows
        if any(r.get("judge_verdicts", []))
    }
    n_hit = len(support_by_id)
    n_supp = sum(1 for c in support_by_id.values() if is_supported(c))
    n_under = sum(1 for c in support_by_id.values() if c == "answer_contains_prediction")
    n_unsupp = n_hit - n_supp - n_under
    if n_hit:
        st.sidebar.caption(
            f"**Judge hits: {n_hit:,}** (by selected prediction)\n\n"
            f"✅ supported: {n_supp:,} ({n_supp/n_hit:.1%})\n\n"
            f"⚠️ under-specific: {n_under:,} ({n_under/n_hit:.1%})\n\n"
            f"❌ unsupported: {n_unsupp:,} ({n_unsupp/n_hit:.1%})"
        )

    # ── Filters (only the three that matter) ────────────────────────
    st.sidebar.title("Filters")
    filter_hit = st.sidebar.radio("Judge result", ["all", "hit", "miss"], horizontal=True)
    support_filter = st.sidebar.selectbox(
        "Support class (judge hits)",
        ["all", "supported", "under-specific (hypernym)", "unsupported"],
        help="How the judge-selected prediction relates to the ground truth. "
             "'under-specific' = prediction ⊆ GT (the note-001 mechanism).",
    )
    search = st.sidebar.text_input("Search answer / rollout text", "")

    def _keep(r) -> bool:
        v = r.get("judge_verdicts", [])
        hit = any(v)
        if filter_hit == "hit" and not hit:
            return False
        if filter_hit == "miss" and hit:
            return False
        if support_filter != "all":
            if not hit:
                return False
            cat = support_by_id.get(r["data_id"])
            if support_filter == "supported" and not is_supported(cat):
                return False
            if support_filter == "under-specific (hypernym)" and cat != "answer_contains_prediction":
                return False
            if support_filter == "unsupported" and (
                is_supported(cat) or cat == "answer_contains_prediction"
            ):
                return False
        if search:
            s = search.lower()
            if s not in r.get("answer", "").lower() and not any(
                s in t.lower() for t in r.get("all_texts", [])
            ):
                return False
        return True

    filtered = [r for r in rows if _keep(r)]
    st.sidebar.write(f"Showing {len(filtered):,} / {len(rows):,}")

    # ── Header + at-a-glance metrics ────────────────────────────────
    st.title("Judge Verdict Explorer")
    if not filtered:
        st.warning("No examples match the current filters.")
        return

    m = st.columns(3)
    m[0].metric("Filtered", f"{len(filtered):,}")
    hit_rate = sum(1 for r in filtered if any(r.get("judge_verdicts", []))) / len(filtered)
    m[1].metric("Judge hit rate", f"{hit_rate:.1%}")
    avg_ci = sum(sum(r.get("judge_verdicts", [])) for r in filtered) / len(filtered)
    m[2].metric("Avg cᵢ", f"{avg_ci:.1f}")

    # ── One navigator: Prev / Random / Next over the filtered set ───
    st.session_state.setdefault("cursor", 0)
    nav = st.columns([1, 1, 1, 4])
    if nav[0].button("◀ Prev", use_container_width=True):
        st.session_state.cursor -= 1
    if nav[1].button("🎲 Random", use_container_width=True):
        st.session_state.cursor = random.randrange(len(filtered))
    if nav[2].button("Next ▶", use_container_width=True):
        st.session_state.cursor += 1
    st.session_state.cursor = max(0, min(st.session_state.cursor, len(filtered) - 1))
    nav[3].markdown(f"### Example {st.session_state.cursor + 1:,} of {len(filtered):,}")

    r = filtered[st.session_state.cursor]
    v = r.get("judge_verdicts", [])
    texts = r.get("all_texts", [])
    ci = sum(v)
    pred = _selected_pred(r)
    gt = r.get("answer", "")

    # ── Detail: question, GT, support verdict, key metrics ──────────
    st.markdown(f"**Q:** {r.get('question', '?')}")
    st.markdown(
        f"**Ground truth:** `{gt}`  ·  "
        f"**Entity:** {r.get('entity_text', '?')} (`{r.get('entity_id', '?')}`)"
    )

    support = classify_positive(
        prediction=pred, answer=gt, aliases_by_canonical=aliases_by_canonical
    )
    if is_supported(support):
        st.success(f"**Selected prediction:** `{pred}`  —  supported ({support})")
    elif support == "answer_contains_prediction":
        st.warning(
            f"**Selected prediction:** `{pred}`  —  under-specific "
            "(prediction ⊆ ground truth; a hypernym/parent, not counted as supported)"
        )
    else:
        st.error(f"**Selected prediction:** `{pred}`  —  no mechanical support")

    d = st.columns(3)
    d[0].metric("cᵢ (correct / 256)", ci)
    d[1].metric("Judge hit", "✅ Yes" if any(v) else "❌ No")
    d[2].metric("Unique answers", len(set(texts)))

    # ── Answer distribution: color-coded horizontal bars ────────────
    st.subheader("Answer distribution")
    if texts:
        answer_verdict: dict[str, int] = {}
        for t, vd in zip(texts, v):
            answer_verdict.setdefault(t, vd)
        top = Counter(texts).most_common(12)
        df = pd.DataFrame(
            [
                {
                    "answer": (a[:40] + "…") if len(a) > 40 else (a or "∅ (empty)"),
                    "rollouts": n,
                    "verdict": "correct" if answer_verdict.get(a) else "incorrect",
                }
                for a, n in top
            ]
        )
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("rollouts:Q", title="rollouts (of 256)"),
                y=alt.Y("answer:N", sort="-x", title=None),
                color=alt.Color(
                    "verdict:N",
                    scale=alt.Scale(domain=["correct", "incorrect"],
                                    range=["#2ecc71", "#e74c3c"]),
                    legend=alt.Legend(title="judge verdict"),
                ),
                tooltip=["answer", "rollouts", "verdict"],
            )
            .properties(height=min(30 * len(df) + 30, 420))
        )
        st.altair_chart(chart, use_container_width=True)

    # ── Optional drill-down, collapsed to keep the page clean ───────
    with st.expander("Individual rollouts (first 100)"):
        only_correct = st.checkbox("Only correct rollouts", value=False, key="only_correct")
        shown = [
            f"{i + 1:3d}. {'✅' if vd else '❌'} {t[:200]}"
            for i, (t, vd) in enumerate(zip(texts[:100], v[:100]))
            if not (only_correct and not vd)
        ]
        st.text("\n".join(shown) or "—")

    raw = r.get("judge_raw", [])
    if raw:
        with st.expander("Judge raw outputs (first 10)"):
            for raw_text in raw[:10]:
                st.code(raw_text[:500], language="text")


if __name__ == "__main__":
    main()
