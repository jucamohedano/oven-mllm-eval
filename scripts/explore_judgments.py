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

from oven_mllm_eval.judge_audit import IDK_VARIANTS, build_alias_map, classify_positive


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

    import streamlit as st

    st.set_page_config(page_title="Judge Verdict Explorer", layout="wide")

    # ── Load ────────────────────────────────────────────────────────
    @st.cache_data
    def load_data(path, max_ex):
        return _load(path, max_ex)

    rows = load_data(args.scored, args.max_examples)
    st.sidebar.title("Data")
    st.sidebar.write(f"Loaded {len(rows):,} examples")

    # ── Filters ─────────────────────────────────────────────────────
    st.sidebar.title("Filters")

    # Build filterable fields
    entities = sorted(set(r.get("entity_text", "") for r in rows))
    questions = sorted(set(r.get("question", "") for r in rows))

    search = st.sidebar.text_input("Search answer / rollout text", "")

    filter_hit = st.sidebar.selectbox(
        "Judge result", ["all", "correct (hit)", "wrong (miss)"]
    )
    judge_hit_target = {"correct (hit)": True, "wrong (miss)": False}.get(filter_hit)

    min_ci = st.sidebar.slider("Min cᵢ (correct rollouts)", 0, 256, 0, step=1)
    max_ci = st.sidebar.slider("Max cᵢ", 0, 256, 256, step=1)

    show_idk = st.sidebar.checkbox("Only examples with IDK rollouts", value=False)
    show_false_positives = st.sidebar.checkbox(
        "Judge says correct — manual inspection needed", value=False
    )

    entity_filter = st.sidebar.selectbox("Entity", ["all"] + entities)
    question_filter = st.sidebar.selectbox("Question template", ["all"] + questions)

    # ── Apply filters ───────────────────────────────────────────────
    IDK = IDK_VARIANTS

    filtered = []
    for r in rows:
        v = r.get("judge_verdicts", [])
        ci = sum(v)
        if judge_hit_target is not None and any(v) != judge_hit_target:
            continue
        if ci < min_ci or ci > max_ci:
            continue
        if entity_filter != "all" and r.get("entity_text", "") != entity_filter:
            continue
        if question_filter != "all" and r.get("question", "") != question_filter:
            continue
        if show_idk:
            texts = r.get("all_texts", [])
            if not any(t.strip().lower() in IDK for t in texts):
                continue
        if show_false_positives:
            if ci == 0 or ci > 5:
                continue
            if not any(v):
                continue
        if search:
            texts = r.get("all_texts", [])
            answer = r.get("answer", "")
            if not any(search.lower() in t.lower() for t in texts) and search.lower() not in answer.lower():
                continue
        filtered.append(r)

    st.sidebar.write(f"Showing {len(filtered):,} / {len(rows):,}")

    # ── Summary stats ───────────────────────────────────────────────
    st.title("Judge Verdict Explorer")

    if not filtered:
        st.warning("No examples match filters.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filtered examples", len(filtered))
    with col2:
        hit_rate = sum(1 for r in filtered if any(r.get("judge_verdicts", []))) / len(filtered)
        st.metric("Judge hit rate", f"{hit_rate:.1%}")
    with col3:
        avg_ci = sum(sum(r.get("judge_verdicts", [])) for r in filtered) / len(filtered)
        st.metric("Avg cᵢ", f"{avg_ci:.1f}")

    # ── Random sample button ────────────────────────────────────────
    if st.button("🎲 Random example"):
        r = random.choice(filtered)
        st.session_state["selected"] = r["data_id"]

    # ── Search by ID ────────────────────────────────────────────────
    search_id = st.text_input("Or search by data_id / image_id", "")
    if search_id:
        for r in filtered:
            if search_id in r.get("data_id", "") or search_id in r.get("image_id", ""):
                st.session_state["selected"] = r["data_id"]
                break

    # ── Example list ────────────────────────────────────────────────
    st.subheader("Examples")
    page_size = 20
    page = st.number_input("Page", 0, len(filtered) // page_size, 0)
    start = page * page_size
    end = min(start + page_size, len(filtered))

    for r in filtered[start:end]:
        v = r.get("judge_verdicts", [])
        ci = sum(v)
        hit = any(v)
        icon = "✅" if hit else "❌"

        col1, col2, col3, col4 = st.columns([3, 1.5, 1.5, 1])
        with col1:
            st.write(f"{icon} `{r['data_id']}`")
            q = r.get("question", "")
            if len(q) > 80:
                q = q[:77] + "..."
            st.write(f"Q: *{q}*")
        with col2:
            st.write(f"GT: **{r.get('answer', '?')}**")
        with col3:
            st.write(f"cᵢ = {ci}/256")
        with col4:
            if st.button("🔍", key=f"btn_{r['data_id']}"):
                st.session_state["selected"] = r["data_id"]

    # ── Detail view ─────────────────────────────────────────────────
    if "selected" not in st.session_state:
        st.info("Click 🔍 on any example or try 🎲 random to inspect rollouts.")
        return

    selected = st.session_state["selected"]
    selected_row = next((r for r in filtered if r["data_id"] == selected), None)
    if selected_row is None:
        st.error(f"Example {selected} not in filtered set.")
        return

    # ── Full detail ─────────────────────────────────────────────────
    st.divider()
    st.subheader(f"📋 {selected}")

    r = selected_row

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.write(f"**Question:** {r.get('question', '?')}")
        st.write(f"**Ground Truth:** `{r.get('answer', '?')}`")
        st.write(f"**Entity:** {r.get('entity_text', '?')} ({r.get('entity_id', '?')})")
        st.write(f"**Original OVEN Q:** {r.get('oven_question', r.get('question', '?'))}")
        # ── Prediction vs ground truth ──
        st.divider()
        pred = r.get("judge_selected_text", r.get("prediction", ""))
        gt = r.get("answer", "")
        st.write("**🔍 Prediction vs Ground Truth:**")

        # Mechanical support check — same classifier the batch auditor uses.
        support = classify_positive(
            prediction=pred, answer=gt, aliases_by_canonical=aliases_by_canonical
        )
        if support is not None:
            st.success(f"Prediction: `{pred}`  _(supported: {support})_")
        else:
            st.error(f"Prediction: `{pred}`  _(no mechanical support)_")
        st.write(f"Ground truth: `{gt}`")
    with col_b:
        v = r.get("judge_verdicts", [])
        ci = sum(v)
        st.metric("cᵢ (correct)", ci)
        st.metric("Judge hit", "✅ Yes" if any(v) else "❌ No")
        texts = r.get("all_texts", [])
        unique = len(set(texts))
        st.metric("Unique answers", unique)
        idk_count = sum(1 for t in texts if t.strip().lower() in IDK)
        st.metric("IDK count", idk_count)
        if any(v) and ci <= 5:
            st.warning("⚠️ Low cᵢ — possible false positive")

    # ── Answer distribution ─────────────────────────────────────────
    st.subheader("Answer distribution (top 15)")
    texts = r.get("all_texts", [])
    c = Counter(texts)
    # Color-code: green for judged correct, red for wrong
    verdicts = r.get("judge_verdicts", [])

    # Build a map: answer string → first judge verdict
    answer_verdict = {}
    for t, vd in zip(texts, verdicts):
        if t not in answer_verdict:
            answer_verdict[t] = vd

    top_answers = c.most_common(15)
    for ans, count in top_answers:
        correct = answer_verdict.get(ans, False)
        symbol = "✅" if correct else "❌"
        pct = count / len(texts) * 100
        bar = "█" * int(pct)
        st.write(f"{symbol} `{ans}` ({count}/{len(texts)} = {pct:.1f}%) {bar}")

    # ── Rollout browser ─────────────────────────────────────────────
    st.subheader("Individual rollouts (first 100)")
    show_correct_only = st.checkbox("Show only correct rollouts", value=False)

    for i, (text, vd) in enumerate(zip(texts[:100], verdicts[:100])):
        if show_correct_only and not vd:
            continue
        symbol = "✅" if vd else "❌"
        display = text
        if len(display) > 200:
            display = display[:197] + "..."
        st.write(f"{i+1:3d}. {symbol} {display}")

    # ── Show also judge_raw if available ────────────────────────────
    raw = r.get("judge_raw", [])
    if raw:
        st.subheader("Judge raw outputs (first 10)")
        for i, raw_text in enumerate(raw[:10]):
            st.code(raw_text[:500], language="text")


if __name__ == "__main__":
    main()
