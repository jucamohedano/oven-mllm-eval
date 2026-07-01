"""Scoring utilities — thin wrapper around DirectMeasureMatcher.evaluate().

Mirrors vlm-eval's ``process_predictions_with_strategy()`` flow:
    1. Load predictions + taxonomy index.
    2. Create DirectMeasureMatcher with a pluggable measure from ALL_MEASURES.
    3. For each prediction, call matcher.evaluate() → collect hP/hR/hF.
    4. Write per-sample scored JSONL + aggregate results JSON.

Output follows the lmms-ocw convention.  When ``--output`` is given::

    <output>                            per-sample scored JSONL
    <run_dir>/<run_id>_results.json     aggregate metrics (unless ``--summary``)

When ``--output`` is omitted, the **input file is overwritten** with the
scored rows — the original predictions are not preserved.  Always pass
``--output`` if you want to keep the raw predictions.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _new_accum() -> dict:
    return {
        "hP": [],
        "hR": [],
        "hF": [],
        "exact": [],
        "mapped": 0,
        "specific_hP": [],
        "specific_hR": [],
        "specific_hF": [],
        "specific_exact": [],
        "specific_mapped": 0,
        "under_specific": [],
        "over_specific": [],
        "depth_delta": [],
    }


def _mean(values: list[float] | list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _dedupe_rollout_texts(texts: list[str]) -> list[dict]:
    """Return stable unique rollout records with counts and original indices."""
    deduped: OrderedDict[str, dict] = OrderedDict()
    for idx, text in enumerate(texts):
        if not text:
            continue
        record = deduped.setdefault(text, {"text": text, "count": 0, "indices": []})
        record["count"] += 1
        record["indices"].append(idx)
    return list(deduped.values())


def _row_rollout_records(row: dict) -> list[dict]:
    records = _dedupe_rollout_texts(row.get("all_texts") or [])
    if records:
        return records
    fallback = (row.get("judge_selected_text")
                or row.get("prediction")
                or row.get("iter_final_prediction")
                or row.get("output", ""))
    return [{"text": fallback, "count": 1, "indices": []}]


def _add_mapping_coverage_metrics(
    metrics: dict,
    *,
    total: int,
    mapped: int,
    hP: list[float],
    hR: list[float],
    hF: list[float],
    exact: list[int] | None = None,
    prefix: str = "",
) -> None:
    """Add explicit mapped/unmapped and all-example zero-filled metrics.

    The existing hP/hR/hF/exact fields are conditional on successful graph
    mapping.  The *_all fields treat unmapped examples as zero, making coverage
    effects visible in summary JSONs and downstream plots.
    """
    key_prefix = f"{prefix}_" if prefix else ""
    coverage = mapped / total if total else 0.0
    metrics.update({
        f"{key_prefix}num_unmapped": max(total - mapped, 0),
        f"{key_prefix}mapping_coverage": coverage,
        f"{key_prefix}hP_all": sum(hP) / total if total else 0.0,
        f"{key_prefix}hR_all": sum(hR) / total if total else 0.0,
        f"{key_prefix}hF_all": sum(hF) / total if total else 0.0,
    })
    if exact is not None:
        metrics[f"{key_prefix}exact_all"] = sum(exact) / total if total else 0.0


def _derive_results_path(samples_path: Path) -> Path:
    """Derive the results JSON path from the samples or judged JSONL path.

    ``<run_id>_samples.jsonl`` → ``<run_id>_results.json``
    ``<run_id>_judged_qwen_qwen3-4b.jsonl`` → ``<run_id>_results_qwen_qwen3-4b.json``
    """
    name = samples_path.name
    m = re.match(r"(.+?)(?:_samples)?_judged_([A-Za-z0-9_.-]+)\.jsonl$", name)
    if m:
        return samples_path.parent / f"{m.group(1)}_results_{m.group(2)}.json"
    m = re.match(r"(.+)_samples\.jsonl$", name)
    if m:
        return samples_path.parent / f"{m.group(1)}_results.json"
    return samples_path.parent / "generations_results.json"


def _score_rows(
    args: tuple[list[dict], list[str], str],
) -> tuple[list[dict], dict[str, dict]]:
    """Score a chunk of rows with DirectMeasureMatcher.

    This is a module-level function so it can be pickled for
    ``multiprocessing.Pool.map``.  Each worker loads its own copy of the
    taxonomy index and builds its own matchers — no shared state.

    Parameters
    ----------
    args : tuple
        ``(rows, measure_names, taxonomy_index_path)`` — a single-tuple
        argument so the function works with ``Pool.map``.

    Returns
    -------
    (list[dict], dict)
        Tuple of ``(scored_rows, accum)`` where ``accum`` is
        ``{measure_name: {"hP": [...], "hR": [...], ...}}``.
    """
    rows, measure_names, taxonomy_index_path = args

    from oven_mllm_eval.taxonomy import load_taxonomy_index
    from oven_mllm_eval.measures import ALL_MEASURES, DirectMeasureMatcher

    index = load_taxonomy_index(taxonomy_index_path)
    matchers = {m: DirectMeasureMatcher(index, ALL_MEASURES[m]) for m in measure_names}

    accum = {m: _new_accum() for m in measure_names}
    scored_rows = []

    total = len(rows)
    report_every = max(1, min(1000, total // 10))  # ~10 updates per chunk
    for i, row in enumerate(rows):
        answer = (row.get("answer", "")
                  .replace("A: ", "").replace("A:", "")
                  .replace("<answer>", "").replace("</answer>", "")
                  .replace("<s>", "").replace("</s>", "")
                  .strip())
        rollout_records = _row_rollout_records(row)
        entity_id = row.get("entity_id")

        # Look up reference path once (shared across measures)
        ref_path = None
        for matcher in matchers.values():
            if entity_id:
                ref_path = (matcher.node_to_path.get(answer)
                            or matcher.index.get("entity_id_to_path", {}).get(entity_id))
            else:
                ref_path = matcher.node_to_path.get(answer)
            if ref_path is not None:
                break

        scored_row = {**row}
        for matcher_name, matcher in matchers.items():
            # Best of the deduped rollout texts by original hF for this
            # measure. Also keep a separate best candidate under the
            # specificity-aware score; this preserves existing metrics while
            # exposing the stricter discriminative signal.
            result = None
            specific_result = None
            rollout_metrics = []
            for rollout_record in rollout_records:
                r = matcher.evaluate(rollout_record["text"], answer, reference_path=ref_path)
                metric_record = {
                    "text": rollout_record["text"],
                    "count": rollout_record["count"],
                    "indices": rollout_record["indices"],
                    "predicted_node": r["predicted_node"],
                    "predicted_path": r["predicted_path"],
                    "hP": r["hP"],
                    "hR": r["hR"],
                    "hF": r["hF"],
                    "exact_match": r["success"],
                    "mapping_method": r["mapping_method"],
                    "scores": r["scores"],
                    "specific_hP": r["specific_hP"],
                    "specific_hR": r["specific_hR"],
                    "specific_hF": r["specific_hF"],
                    "specific_exact_match": r["success"],
                    "under_specific": r["under_specific"],
                    "over_specific": r["over_specific"],
                    "depth_delta": r["depth_delta"],
                }
                rollout_metrics.append(metric_record)
                if result is None or r["hF"] > result["hF"]:
                    result = r
                if specific_result is None or r["specific_hF"] > specific_result["specific_hF"]:
                    specific_result = r

            prefix = matcher_name
            scored_row.update({
                f"{prefix}_predicted_node": result["predicted_node"],
                f"{prefix}_predicted_path": result["predicted_path"],
                f"{prefix}_hP": result["hP"],
                f"{prefix}_hR": result["hR"],
                f"{prefix}_hF": result["hF"],
                f"{prefix}_exact_match": result["success"],
                f"{prefix}_mapping_method": result["mapping_method"],
                f"{prefix}_scores": result["scores"],
                f"{prefix}_rollout_metrics": rollout_metrics,
                f"{prefix}_specific_predicted_node": specific_result["predicted_node"],
                f"{prefix}_specific_predicted_path": specific_result["predicted_path"],
                f"{prefix}_specific_hP": specific_result["specific_hP"],
                f"{prefix}_specific_hR": specific_result["specific_hR"],
                f"{prefix}_specific_hF": specific_result["specific_hF"],
                f"{prefix}_specific_exact_match": specific_result["success"],
                f"{prefix}_specific_mapping_method": specific_result["mapping_method"],
                f"{prefix}_under_specific": specific_result["under_specific"],
                f"{prefix}_over_specific": specific_result["over_specific"],
                f"{prefix}_depth_delta": specific_result["depth_delta"],
            })

            if result["predicted_path"] is not None and result["reference_path"] is not None:
                accum[matcher_name]["hP"].append(result["hP"])
                accum[matcher_name]["hR"].append(result["hR"])
                accum[matcher_name]["hF"].append(result["hF"])
                accum[matcher_name]["exact"].append(int(result["success"]))
                accum[matcher_name]["mapped"] += 1
            if (
                specific_result["predicted_path"] is not None
                and specific_result["reference_path"] is not None
            ):
                accum[matcher_name]["specific_hP"].append(specific_result["specific_hP"])
                accum[matcher_name]["specific_hR"].append(specific_result["specific_hR"])
                accum[matcher_name]["specific_hF"].append(specific_result["specific_hF"])
                accum[matcher_name]["specific_exact"].append(int(specific_result["success"]))
                accum[matcher_name]["under_specific"].append(int(specific_result["under_specific"]))
                accum[matcher_name]["over_specific"].append(int(specific_result["over_specific"]))
                if specific_result["depth_delta"] is not None:
                    accum[matcher_name]["depth_delta"].append(specific_result["depth_delta"])
                accum[matcher_name]["specific_mapped"] += 1

        scored_row["scored_reference_path"] = result["reference_path"]
        scored_rows.append(scored_row)

        if (i + 1) % report_every == 0 or i == total - 1:
            pct = (i + 1) / total * 100
            print(f"[scoring] {i + 1}/{total} ({pct:.1f}%)", flush=True)

    return scored_rows, accum


def _score_rollouts(scored_rows: list[dict], index: dict,
                    mapping: dict[str, dict]) -> tuple[dict, dict]:
    """Best-of-N hierarchical scoring over a sample's rollouts.

    For each example, every rollout in ``all_texts`` is mapped to a node (via
    the precomputed ``mapping``) and scored against the reference path; the
    rollout with the highest hF is selected and *its* (hP, hR, hF) reported.
    The expensive embed + cascade already ran once per unique rollout string.

    Per-rollout metrics are persisted in deduped form so downstream analysis can
    compute frequency-weighted means, variances, quantiles, and judge-conditioned
    diagnostics without remapping.
    """
    from oven_mllm_eval.scores import (
        calc_hierarchical_metrics,
        calc_specificity_hierarchical_metrics,
    )
    from oven_mllm_eval.matching import _normalise

    n2p = index.get("node_to_path", {})
    eid2p = index.get("entity_id_to_path", {})
    accum = _new_accum()
    method_counts: dict = {}
    total = len(scored_rows)
    report_every = max(1, min(1000, total // 10)) if total else 1

    print(f"[cascade] scoring per-row rollout metrics for {total:,} rows", flush=True)

    for row_idx, row in enumerate(scored_rows, start=1):
        answer = (row.get("answer", "")
                  .replace("A: ", "").replace("A:", "")
                  .replace("<answer>", "").replace("</answer>", "")
                  .replace("<s>", "").replace("</s>", "")
                  .strip())
        entity_id = row.get("entity_id")
        ref_path = n2p.get(answer) or (eid2p.get(entity_id) if entity_id else None)
        rollout_records = _row_rollout_records(row)

        # Best-of-N: score each unique rollout once, keep the highest-hF one.
        # Any mapped prediction shares the root → hF > 0, so the argmax always
        # prefers a mapped rollout over an unmapped one (hF = 0).
        best = None  # (hF, hP, hR, node, path, method)
        specific_best = None  # (specific_hF, specific_hP, specific_hR, node, path, method, under, over, depth_delta)
        rollout_metrics = []
        for rollout_record in rollout_records:
            text = rollout_record["text"]
            m = mapping.get(text) or {}
            pp = m.get("predicted_path")
            if pp is not None and ref_path is not None:
                mt = calc_hierarchical_metrics([(pp, ref_path)])
                hP, hR, hF = mt["hP"][0], mt["hR"][0], mt["hF"][0]
                smt = calc_specificity_hierarchical_metrics([(pp, ref_path)])
                specific_hP = smt["specific_hP"][0]
                specific_hR = smt["specific_hR"][0]
                specific_hF = smt["specific_hF"][0]
                under_specific = smt["under_specific"][0]
                over_specific = smt["over_specific"][0]
                depth_delta = smt["depth_delta"][0]
            else:
                hP = hR = hF = 0.0
                specific_hP = specific_hR = specific_hF = 0.0
                under_specific = over_specific = False
                depth_delta = None
            exact_for_rollout = (
                _normalise(m.get("predicted_node") or "") == _normalise(answer)
                if pp is not None and ref_path is not None
                else False
            )
            rollout_metrics.append({
                "text": text,
                "count": rollout_record["count"],
                "indices": rollout_record["indices"],
                "predicted_node": m.get("predicted_node"),
                "predicted_path": pp,
                "hP": hP,
                "hR": hR,
                "hF": hF,
                "exact_match": exact_for_rollout,
                "mapping_method": m.get("mapping_method"),
                "specific_hP": specific_hP,
                "specific_hR": specific_hR,
                "specific_hF": specific_hF,
                "specific_exact_match": exact_for_rollout,
                "under_specific": under_specific,
                "over_specific": over_specific,
                "depth_delta": depth_delta,
            })
            if best is None or hF > best[0]:
                best = (hF, hP, hR, m.get("predicted_node"), pp, m.get("mapping_method"))
            if specific_best is None or specific_hF > specific_best[0]:
                specific_best = (
                    specific_hF,
                    specific_hP,
                    specific_hR,
                    m.get("predicted_node"),
                    pp,
                    m.get("mapping_method"),
                    under_specific,
                    over_specific,
                    depth_delta,
                )

        if best is None:  # no rollouts
            best = (0.0, 0.0, 0.0, None, None, None)
        if specific_best is None:
            specific_best = (0.0, 0.0, 0.0, None, None, None, False, False, None)
        hF, hP, hR, pred_node, pred_path, method = best
        (
            specific_hF,
            specific_hP,
            specific_hR,
            specific_pred_node,
            specific_pred_path,
            specific_method,
            under_specific,
            over_specific,
            depth_delta,
        ) = specific_best
        method_counts[method] = method_counts.get(method, 0) + 1

        if pred_path is not None and ref_path is not None:
            exact = _normalise(pred_node or "") == _normalise(answer)
            accum["hP"].append(hP)
            accum["hR"].append(hR)
            accum["hF"].append(hF)
            accum["exact"].append(int(exact))
            accum["mapped"] += 1
        else:
            exact = False
        if specific_pred_path is not None and ref_path is not None:
            specific_exact = _normalise(specific_pred_node or "") == _normalise(answer)
            accum["specific_hP"].append(specific_hP)
            accum["specific_hR"].append(specific_hR)
            accum["specific_hF"].append(specific_hF)
            accum["specific_exact"].append(int(specific_exact))
            accum["under_specific"].append(int(under_specific))
            accum["over_specific"].append(int(over_specific))
            if depth_delta is not None:
                accum["depth_delta"].append(depth_delta)
            accum["specific_mapped"] += 1
        else:
            specific_exact = False

        row.update({
            "cascade_predicted_node": pred_node,
            "cascade_predicted_path": pred_path,
            "cascade_hP": hP,
            "cascade_hR": hR,
            "cascade_hF": hF,
            "cascade_exact_match": exact,
            "cascade_mapping_method": method,
            "cascade_rollout_metrics": rollout_metrics,
            "cascade_specific_predicted_node": specific_pred_node,
            "cascade_specific_predicted_path": specific_pred_path,
            "cascade_specific_hP": specific_hP,
            "cascade_specific_hR": specific_hR,
            "cascade_specific_hF": specific_hF,
            "cascade_specific_exact_match": specific_exact,
            "cascade_specific_mapping_method": specific_method,
            "cascade_under_specific": under_specific,
            "cascade_over_specific": over_specific,
            "cascade_depth_delta": depth_delta,
        })

        if row_idx % report_every == 0 or row_idx == total:
            print(f"[cascade] scored rollout metrics for {row_idx:,}/{total:,} rows", flush=True)

    return accum, method_counts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_generation_file(
    input_path: str | Path,
    taxonomy_index_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    summary_path: Optional[str | Path] = None,
    measure: str | Sequence[str] = "exact_match",
    num_workers: int = 1,
    embed_model: str = "hf-hub:apple/DFN5B-CLIP-ViT-H-14",
    embed_backend: str = "open_clip",
    map_top_k: int = 10,
    embed_device: str = "cpu",
    max_examples: int | None = None,
) -> dict | list[dict]:
    """Score a generation JSONL file with one or more pluggable measures.

    Uses ``DirectMeasureMatcher`` (adapted from vlm-eval) to score each
    prediction against all taxonomy node labels via the chosen measure(s).

    Each row must have at least ``answer`` and ``prediction`` fields.

    Parameters
    ----------
    input_path : str or Path
        Path to the generation JSONL.
    taxonomy_index_path : str or Path, optional
        Path to the precomputed taxonomy index.
    output_path : str or Path, optional
        If given, write per-example scored JSONL here.  Default: overwrite
        the input file with metrics merged in.
    summary_path : str or Path, optional
        If given, write aggregate metrics here.  Default: derive from the
        input filename (``<run_id>_results.json``).
    measure : str or sequence of str
        Measure key(s) from ``ALL_MEASURES``.  Use ``"all"`` to select all
        registered measures.  Default ``"exact_match"``.
    num_workers : int
        Number of worker processes for parallel scoring.  Default 0 (auto: 2/3 of
        ``os.cpu_count()``, minimum 1).  Set to 1 for serial execution.
        Each worker loads its own copy of the taxonomy index.

    Returns
    -------
    dict or list[dict]
        Single-measure: aggregate metrics dict.
        Multi-measure: list of ``{"measure": name, "metrics": {...}}`` dicts.
    """
    # Auto-detect workers: 0 → use all available CPUs (via sched_getaffinity)
    if num_workers == 0:
        try:
            num_workers = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            num_workers = os.cpu_count() or 1

    from oven_mllm_eval.taxonomy import load_taxonomy_index
    from oven_mllm_eval.measures import ALL_MEASURES

    # Resolve measures
    if isinstance(measure, str):
        if measure == "all":
            measure_names = list(ALL_MEASURES.keys())
        else:
            measure_names = [measure]
    else:
        measure_names = list(measure)

    # "cascade" is the cascading mapping algorithm (cosine top-k retrieval →
    # exact/n-gram/voting); not a per-row ALL_MEASURES entry, handled separately.
    EMBED_MEASURE = "cascade"
    do_embed = EMBED_MEASURE in measure_names
    lexical_names = [m for m in measure_names if m != EMBED_MEASURE]

    for m in lexical_names:
        if m not in ALL_MEASURES:
            raise ValueError(
                f"Unknown measure '{m}'. Available: "
                f"{list(ALL_MEASURES.keys()) + [EMBED_MEASURE]}"
            )

    input_path = Path(input_path)

    # Default output: overwrite the samples file with scored rows
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)

    # Default results: derive from samples filename
    if summary_path is None:
        summary_path = _derive_results_path(input_path)
    else:
        summary_path = Path(summary_path)

    # Resolve taxonomy index path once (before spawning workers)
    if taxonomy_index_path is None:
        from oven_mllm_eval.paths import OVEN_TAXONOMY_INDEX
        resolved_index_path = str(Path(OVEN_TAXONOMY_INDEX))
    else:
        resolved_index_path = str(Path(taxonomy_index_path))

    # Read all rows
    print(f"[score] reading input rows from {input_path}", flush=True)
    rows = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_examples and len(rows) >= max_examples:
                break  # quick-test cap — stop reading early
    print(
        f"[score] loaded {len(rows):,} rows; measures={measure_names}; "
        f"workers={num_workers}; output={output_path}; summary={summary_path}",
        flush=True,
    )

    # Score lexical measures — parallel or serial
    accum = {m: _new_accum() for m in lexical_names}
    if lexical_names:
        print(f"[score] scoring lexical/direct measures: {lexical_names}", flush=True)
        if num_workers > 1:
            # Contiguous chunks of roughly equal size.  All rows do the same
            # work (one prediction vs ~12K node labels), so round-robin isn't
            # needed — and contiguous chunks preserve row order in output.
            chunk_size = (len(rows) + num_workers - 1) // num_workers
            chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
            args = [(chunk, lexical_names, resolved_index_path) for chunk in chunks]
            print(
                f"[score] launched {len(chunks)} scoring chunks "
                f"(chunk_size~{chunk_size:,})",
                flush=True,
            )

            try:
                with multiprocessing.Pool(num_workers) as pool:
                    results = pool.map(_score_rows, args)
            except Exception:
                logger.warning(
                    "multiprocessing.Pool.map failed — falling back to serial. "
                    "Error details:", exc_info=True,
                )
                results = [_score_rows((rows, lexical_names, resolved_index_path))]
        else:
            results = [_score_rows((rows, lexical_names, resolved_index_path))]

        print("[score] merging scored chunks and accumulators", flush=True)
        scored_rows = []
        for chunk_rows, chunk_accum in results:
            scored_rows.extend(chunk_rows)
            for m in lexical_names:
                for key in (
                    "hP", "hR", "hF", "exact",
                    "specific_hP", "specific_hR", "specific_hF", "specific_exact",
                    "under_specific", "over_specific", "depth_delta",
                ):
                    accum[m][key].extend(chunk_accum[m][key])
                accum[m]["mapped"] += chunk_accum[m]["mapped"]
                accum[m]["specific_mapped"] += chunk_accum[m]["specific_mapped"]
        print(f"[score] merged {len(scored_rows):,} scored rows", flush=True)
    else:
        scored_rows = [dict(r) for r in rows]
        print("[score] no lexical/direct measures requested", flush=True)

    # Score the embedding measure (cosine retrieval → cascade), single pass.
    embed_method_counts: dict | None = None
    if do_embed:
        from oven_mllm_eval.taxonomy import load_taxonomy_index
        from oven_mllm_eval.embedding_matcher import build_prediction_mapping

        print(f"[cascade] loading taxonomy index from {resolved_index_path}", flush=True)
        _index = load_taxonomy_index(resolved_index_path)
        # Best-of-N: map every rollout (deduped internally by build_prediction_mapping),
        # not a single representative prediction.
        _rollouts = [
            record["text"]
            for row in scored_rows
            for record in _row_rollout_records(row)
        ]
        _unique_rollouts = len({r for r in _rollouts if r})
        print(
            f"[cascade] collected {len(_rollouts):,} deduped-per-row rollout records "
            f"({_unique_rollouts:,} globally unique texts)",
            flush=True,
        )
        _mapping = build_prediction_mapping(
            _rollouts,
            _index,
            backend=embed_backend,
            model_name=embed_model,
            k=map_top_k,
            device=embed_device,
        )
        print(f"[cascade] built mapping for {len(_mapping):,} unique texts", flush=True)
        accum[EMBED_MEASURE], embed_method_counts = _score_rollouts(
            scored_rows, _index, _mapping
        )

    # Aggregate per measure
    print("[score] aggregating per-measure summaries", flush=True)
    summaries = []
    for matcher_name in measure_names:
        a = accum[matcher_name]
        total_examples = len(scored_rows)
        s = {
            "hP": _mean(a["hP"]),
            "hR": _mean(a["hR"]),
            "hF": _mean(a["hF"]),
            "exact": _mean(a["exact"]),
            "num_examples": total_examples,
            "num_mapped": a["mapped"],
        }
        _add_mapping_coverage_metrics(
            s,
            total=total_examples,
            mapped=a["mapped"],
            hP=a["hP"],
            hR=a["hR"],
            hF=a["hF"],
            exact=a["exact"],
        )
        s.update({
            "specific_hP": _mean(a["specific_hP"]),
            "specific_hR": _mean(a["specific_hR"]),
            "specific_hF": _mean(a["specific_hF"]),
            "specific_exact": _mean(a["specific_exact"]),
            "specific_num_mapped": a["specific_mapped"],
            "under_specific_rate": _mean(a["under_specific"]),
            "over_specific_rate": _mean(a["over_specific"]),
            "mean_depth_delta": _mean(a["depth_delta"]),
        })
        _add_mapping_coverage_metrics(
            s,
            total=total_examples,
            mapped=a["specific_mapped"],
            hP=a["specific_hP"],
            hR=a["specific_hR"],
            hF=a["specific_hF"],
            exact=a["specific_exact"],
            prefix="specific",
        )
        summaries.append({"measure": matcher_name, "metrics": s})

    # Attach mapping-method breakdown (exact / ngram / voting / top_score)
    # to the embedding measure for auditability.
    if embed_method_counts is not None:
        for _s in summaries:
            if _s["measure"] == EMBED_MEASURE:
                _s["metrics"]["mapping_methods"] = embed_method_counts
                _s["metrics"]["selection"] = "best_of_n"

    # ── pass@k from judge verdicts ──────────────────────────────────
    # Uses the numerically stable product-form estimator:
    #   pass@k(n, c, k) = 1 - ∏_{i=0}^{k-1} (n - c - i) / (n - i)
    # which is equivalent to 1 - C(n-c, k) / C(n, k).
    from oven_mllm_eval.pass_at_k import pass_at_k as _pass_at_k_fn

    _judge_rows = [r for r in scored_rows if r.get("judge_verdicts")]
    if _judge_rows:
        print(f"[score] computing pass@k from {len(_judge_rows):,} judged rows", flush=True)
        _ns = [len(r["judge_verdicts"]) for r in _judge_rows]
        _n_max = max(_ns) if _ns else 0
        _candidate_ks = [2**i for i in range(0, 12)]  # 1, 2, 4, 8, ..., 2048
        _ks = sorted({k for k in _candidate_ks if 0 < k <= _n_max})
        _ks.append(_n_max)  # always include the full rollout count

        def _compute_pass_at_k(verdicts_key: str) -> dict[str, float]:
            result: dict[str, float] = {}
            for _k in _ks:
                _vals: list[float] = []
                for _n, r in zip(_ns, _judge_rows):
                    _v = r.get(verdicts_key)
                    if _v is None:
                        continue
                    _c = sum(_v)
                    if _n == 0:
                        continue
                    _vals.append(_pass_at_k_fn(_n, _c, _k))
                if _vals:
                    result[f"pass@{_k}"] = sum(_vals) / len(_vals)
            return result

        _pass_at_k = _compute_pass_at_k("judge_verdicts")

        # Majority-vote pass@k (extra, when available)
        if any(r.get("judge_verdicts_majority") for r in _judge_rows):
            _pass_at_k_majority = _compute_pass_at_k("judge_verdicts_majority")
            for _k, _v in _pass_at_k_majority.items():
                _pass_at_k[f"{_k}_majority"] = _v

        for _s in summaries:
            _s["metrics"].update(_pass_at_k)

    # ── Judge parse stats ──────────────────────────────────────────
    # Count rollouts where the judge produced no parseable output
    # (judge_parse_ok=False) so users can gauge free-form reliability.
    _judge_unparseable = 0
    _judge_rollouts = 0
    for _r in scored_rows:
        _ok = _r.get("judge_parse_ok")
        if _ok is not None:
            _judge_rollouts += len(_ok)
            _judge_unparseable += sum(1 for ok in _ok if not ok)
    if _judge_rollouts:
        for _s in summaries:
            _s["metrics"]["num_judge_unparseable"] = _judge_unparseable
            _s["metrics"]["num_judge_rollouts"] = _judge_rollouts

    # Drop stale unprefixed keys from old scoring runs so rows stay clean
    _STALE_KEYS = {"scored_predicted_node", "scored_predicted_path",
                   "hP", "hR", "hF", "exact_match", "mapping_method"}
    print("[score] removing stale unprefixed score keys", flush=True)
    for row in scored_rows:
        for k in _STALE_KEYS:
            row.pop(k, None)

    # Write per-sample scored JSONL (overwrites input by default)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[score] writing {len(scored_rows):,} scored rows to {output_path}", flush=True)
    write_report_every = max(1, min(1000, len(scored_rows) // 10)) if scored_rows else 1
    with open(output_path, "w") as f:
        for row_idx, row in enumerate(scored_rows, start=1):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if row_idx % write_report_every == 0 or row_idx == len(scored_rows):
                print(f"[score] wrote {row_idx:,}/{len(scored_rows):,} scored rows", flush=True)

    # Write aggregate results JSON
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Include judge model info from metadata if available
    _summary_data: dict = (
        summaries[0]["metrics"] if len(summaries) == 1
        else {"measures": summaries}
    )
    _judge_meta_files = sorted(input_path.parent.glob(
        f"{input_path.stem}_shard*_metadata.json"
    ))
    if not _judge_meta_files:
        # Also try with _judged prefix (old naming convention)
        _judge_meta_files = sorted(input_path.parent.glob(
            f"{input_path.stem.replace('_samples', '')}_judged*_metadata.json"
        ))
    if _judge_meta_files:
        with open(_judge_meta_files[0]) as _jmf:
            _jmeta = json.load(_jmf)
        _summary_data["judge_model"] = _jmeta.get("judge_model", "unknown")
        _summary_data["judge_mode"] = _jmeta.get("judge_mode", "unknown")

    print(f"[score] writing summary JSON to {summary_path}", flush=True)
    with open(summary_path, "w") as f:
        json.dump(_summary_data, f, indent=2)
    print("[score] scoring complete", flush=True)

    return summaries[0]["metrics"] if len(summaries) == 1 else summaries


def aggregate_scored_file(
    input_path: str | Path,
    summary_path: Optional[str | Path] = None,
    measure: str | Sequence[str] = "exact_match",
) -> dict | list[dict]:
    """Aggregate an already-scored JSONL file without recomputing matches.

    This is useful when per-row ``<measure>_hP``, ``<measure>_hR``,
    ``<measure>_hF``, and ``<measure>_exact_match`` fields already exist and
    only the aggregate results JSON needs to be regenerated.
    """
    input_path = Path(input_path)
    if summary_path is None:
        summary_path = _derive_results_path(input_path)
    else:
        summary_path = Path(summary_path)

    if isinstance(measure, str):
        measure_names = [measure]
    else:
        measure_names = list(measure)
    if measure_names == ["all"]:
        measure_names = sorted({
            key[:-3]
            for row in _iter_jsonl(input_path)
            for key in row
            if key.endswith("_hP")
        })

    rows = list(_iter_jsonl(input_path))
    summaries = []
    for matcher_name in measure_names:
        required = [
            f"{matcher_name}_hP",
            f"{matcher_name}_hR",
            f"{matcher_name}_hF",
            f"{matcher_name}_exact_match",
        ]
        missing = [field for field in required if rows and field not in rows[0]]
        if missing:
            raise ValueError(
                f"Input does not look scored for measure '{matcher_name}'. "
                f"Missing fields: {missing}"
            )

        hP: list[float] = []
        hR: list[float] = []
        hF: list[float] = []
        exact: list[int] = []
        mapped = 0
        specific_hP: list[float] = []
        specific_hR: list[float] = []
        specific_hF: list[float] = []
        specific_exact: list[int] = []
        under_specific: list[int] = []
        over_specific: list[int] = []
        depth_delta: list[float] = []
        specific_mapped = 0
        for row in rows:
            reference_path = row.get("scored_reference_path")
            predicted_path = row.get(f"{matcher_name}_predicted_path")
            if predicted_path is None or reference_path is None:
                pass
            else:
                hP.append(float(row.get(f"{matcher_name}_hP", 0.0) or 0.0))
                hR.append(float(row.get(f"{matcher_name}_hR", 0.0) or 0.0))
                hF.append(float(row.get(f"{matcher_name}_hF", 0.0) or 0.0))
                exact.append(int(bool(row.get(f"{matcher_name}_exact_match", False))))
                mapped += 1

            specific_predicted_path = row.get(f"{matcher_name}_specific_predicted_path")
            if specific_predicted_path is None or reference_path is None:
                continue
            specific_hP.append(float(row.get(f"{matcher_name}_specific_hP", 0.0) or 0.0))
            specific_hR.append(float(row.get(f"{matcher_name}_specific_hR", 0.0) or 0.0))
            specific_hF.append(float(row.get(f"{matcher_name}_specific_hF", 0.0) or 0.0))
            specific_exact.append(int(bool(row.get(f"{matcher_name}_specific_exact_match", False))))
            under_specific.append(int(bool(row.get(f"{matcher_name}_under_specific", False))))
            over_specific.append(int(bool(row.get(f"{matcher_name}_over_specific", False))))
            delta = row.get(f"{matcher_name}_depth_delta")
            if delta is not None:
                depth_delta.append(float(delta))
            specific_mapped += 1

        total_examples = len(rows)
        metrics = {
            "hP": _mean(hP),
            "hR": _mean(hR),
            "hF": _mean(hF),
            "exact": _mean(exact),
            "num_examples": total_examples,
            "num_mapped": mapped,
        }
        _add_mapping_coverage_metrics(
            metrics,
            total=total_examples,
            mapped=mapped,
            hP=hP,
            hR=hR,
            hF=hF,
            exact=exact,
        )
        metrics.update({
            "specific_hP": _mean(specific_hP),
            "specific_hR": _mean(specific_hR),
            "specific_hF": _mean(specific_hF),
            "specific_exact": _mean(specific_exact),
            "specific_num_mapped": specific_mapped,
            "under_specific_rate": _mean(under_specific),
            "over_specific_rate": _mean(over_specific),
            "mean_depth_delta": _mean(depth_delta),
        })
        _add_mapping_coverage_metrics(
            metrics,
            total=total_examples,
            mapped=specific_mapped,
            hP=specific_hP,
            hR=specific_hR,
            hF=specific_hF,
            exact=specific_exact,
            prefix="specific",
        )
        summaries.append({"measure": matcher_name, "metrics": metrics})

    _add_judge_metrics(summaries, rows)
    _write_summary(input_path, summary_path, summaries)
    return summaries[0]["metrics"] if len(summaries) == 1 else summaries


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _add_judge_metrics(summaries: list[dict], scored_rows: list[dict]) -> None:
    from oven_mllm_eval.pass_at_k import pass_at_k as _pass_at_k_fn

    _judge_rows = [r for r in scored_rows if r.get("judge_verdicts")]
    if _judge_rows:
        _ns = [len(r["judge_verdicts"]) for r in _judge_rows]
        _n_max = max(_ns) if _ns else 0
        _candidate_ks = [2**i for i in range(0, 12)]
        _ks = sorted({k for k in _candidate_ks if 0 < k <= _n_max})
        _ks.append(_n_max)

        def _compute_pass_at_k(verdicts_key: str) -> dict[str, float]:
            result: dict[str, float] = {}
            for _k in _ks:
                _vals: list[float] = []
                for _n, row in zip(_ns, _judge_rows):
                    _v = row.get(verdicts_key)
                    if _v is None:
                        continue
                    _c = sum(_v)
                    if _n == 0:
                        continue
                    _vals.append(_pass_at_k_fn(_n, _c, _k))
                if _vals:
                    result[f"pass@{_k}"] = sum(_vals) / len(_vals)
            return result

        _pass_at_k = _compute_pass_at_k("judge_verdicts")
        if any(r.get("judge_verdicts_majority") for r in _judge_rows):
            _pass_at_k_majority = _compute_pass_at_k("judge_verdicts_majority")
            for _k, _v in _pass_at_k_majority.items():
                _pass_at_k[f"{_k}_majority"] = _v

        for _s in summaries:
            _s["metrics"].update(_pass_at_k)

    _judge_unparseable = 0
    _judge_rollouts = 0
    for row in scored_rows:
        _ok = row.get("judge_parse_ok")
        if _ok is not None:
            _judge_rollouts += len(_ok)
            _judge_unparseable += sum(1 for ok in _ok if not ok)
    if _judge_rollouts:
        for _s in summaries:
            _s["metrics"]["num_judge_unparseable"] = _judge_unparseable
            _s["metrics"]["num_judge_rollouts"] = _judge_rollouts


def _write_summary(input_path: Path, summary_path: Path, summaries: list[dict]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_data: dict = (
        summaries[0]["metrics"] if len(summaries) == 1
        else {"measures": summaries}
    )
    judge_meta_files = sorted(input_path.parent.glob(
        f"{input_path.stem}_shard*_metadata.json"
    ))
    if not judge_meta_files:
        judge_meta_files = sorted(input_path.parent.glob(
            f"{input_path.stem.replace('_samples', '')}_judged*_metadata.json"
        ))
    if judge_meta_files:
        with open(judge_meta_files[0]) as handle:
            judge_meta = json.load(handle)
        summary_data["judge_model"] = judge_meta.get("judge_model", "unknown")
        summary_data["judge_mode"] = judge_meta.get("judge_mode", "unknown")

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_data, handle, indent=2)
