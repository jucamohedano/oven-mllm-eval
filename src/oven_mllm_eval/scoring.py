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
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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

    accum = {m: {"hP": [], "hR": [], "hF": [], "exact": [], "mapped": 0}
             for m in measure_names}
    scored_rows = []

    total = len(rows)
    report_every = max(1, min(1000, total // 10))  # ~10 updates per chunk
    for i, row in enumerate(rows):
        answer = (row.get("answer", "")
                  .replace("A: ", "").replace("A:", "")
                  .replace("<answer>", "").replace("</answer>", "")
                  .replace("<s>", "").replace("</s>", "")
                  .strip())
        # judge_selected_text is the authoritative prediction when the
        # judge ran (Phase 2); fall back to raw prediction for backward
        # compatibility with pre-judge runs.
        prediction = (row.get("judge_selected_text")
                      or row.get("prediction")
                      or row.get("iter_final_prediction")
                      or row.get("output", ""))
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
            result = matcher.evaluate(prediction, answer, reference_path=ref_path)

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
            })

            if result["predicted_path"] is not None and result["reference_path"] is not None:
                accum[matcher_name]["hP"].append(result["hP"])
                accum[matcher_name]["hR"].append(result["hR"])
                accum[matcher_name]["hF"].append(result["hF"])
                accum[matcher_name]["exact"].append(int(result["success"]))
                accum[matcher_name]["mapped"] += 1

        scored_row["scored_reference_path"] = result["reference_path"]
        scored_rows.append(scored_row)

        if (i + 1) % report_every == 0 or i == total - 1:
            pct = (i + 1) / total * 100
            print(f"[scoring] {i + 1}/{total} ({pct:.1f}%)", flush=True)

    return scored_rows, accum


def _extract_prediction(row: dict) -> str:
    """Authoritative prediction string (mirrors the priority in _score_rows)."""
    return (row.get("judge_selected_text")
            or row.get("prediction")
            or row.get("iter_final_prediction")
            or row.get("output", ""))


def _score_embedding(scored_rows: list[dict], index: dict,
                     mapping: dict[str, dict]) -> tuple[dict, dict]:
    """Annotate rows with ``sentence_bert_*`` fields from a precomputed mapping.

    The expensive embed + cascade already ran once per unique prediction
    (``mapping``); here we only join it back and compute hP/hR/hF per row
    (cheap — the reference path differs per example).
    """
    from oven_mllm_eval.scores import calc_hierarchical_metrics
    from oven_mllm_eval.matching import _normalise

    n2p = index.get("node_to_path", {})
    eid2p = index.get("entity_id_to_path", {})
    accum = {"hP": [], "hR": [], "hF": [], "exact": [], "mapped": 0}
    method_counts: dict = {}

    for row in scored_rows:
        answer = (row.get("answer", "")
                  .replace("A: ", "").replace("A:", "")
                  .replace("<answer>", "").replace("</answer>", "")
                  .replace("<s>", "").replace("</s>", "")
                  .strip())
        entity_id = row.get("entity_id")
        ref_path = n2p.get(answer) or (eid2p.get(entity_id) if entity_id else None)

        m = mapping.get(_extract_prediction(row)) or {}
        method = m.get("mapping_method")
        method_counts[method] = method_counts.get(method, 0) + 1
        pred_node = m.get("predicted_node")
        pred_path = m.get("predicted_path")

        if pred_path is not None and ref_path is not None:
            mt = calc_hierarchical_metrics([(pred_path, ref_path)])
            hP, hR, hF = mt["hP"][0], mt["hR"][0], mt["hF"][0]
            exact = _normalise(pred_node or "") == _normalise(answer)
            accum["hP"].append(hP)
            accum["hR"].append(hR)
            accum["hF"].append(hF)
            accum["exact"].append(int(exact))
            accum["mapped"] += 1
        else:
            hP = hR = hF = 0.0
            exact = False

        row.update({
            "sentence_bert_predicted_node": pred_node,
            "sentence_bert_predicted_path": pred_path,
            "sentence_bert_hP": hP,
            "sentence_bert_hR": hR,
            "sentence_bert_hF": hF,
            "sentence_bert_exact_match": exact,
            "sentence_bert_mapping_method": method,
        })

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
    embed_model: str = "sentence-transformers/all-mpnet-base-v2",
    map_top_k: int = 3,
    map_min_score: float = 0.35,
    embed_device: str = "cpu",
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

    # "sentence_bert" is the cosine-retrieval path (not a per-row ALL_MEASURES
    # entry); it is handled separately below.
    EMBED_MEASURE = "sentence_bert"
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
    rows = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    # Score lexical measures — parallel or serial
    accum = {m: {"hP": [], "hR": [], "hF": [], "exact": [], "mapped": 0}
             for m in lexical_names}
    if lexical_names:
        if num_workers > 1:
            # Contiguous chunks of roughly equal size.  All rows do the same
            # work (one prediction vs ~12K node labels), so round-robin isn't
            # needed — and contiguous chunks preserve row order in output.
            chunk_size = (len(rows) + num_workers - 1) // num_workers
            chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
            args = [(chunk, lexical_names, resolved_index_path) for chunk in chunks]

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

        scored_rows = []
        for chunk_rows, chunk_accum in results:
            scored_rows.extend(chunk_rows)
            for m in lexical_names:
                for key in ("hP", "hR", "hF", "exact"):
                    accum[m][key].extend(chunk_accum[m][key])
                accum[m]["mapped"] += chunk_accum[m]["mapped"]
    else:
        scored_rows = [dict(r) for r in rows]

    # Score the embedding measure (cosine retrieval → cascade), single pass.
    embed_method_counts: dict | None = None
    if do_embed:
        from oven_mllm_eval.taxonomy import load_taxonomy_index
        from oven_mllm_eval.embedding_matcher import build_prediction_mapping

        _index = load_taxonomy_index(resolved_index_path)
        _preds = [_extract_prediction(r) for r in scored_rows]
        _mapping = build_prediction_mapping(
            _preds, _index, model_name=embed_model, k=map_top_k,
            min_score=map_min_score, device=embed_device,
        )
        accum[EMBED_MEASURE], embed_method_counts = _score_embedding(
            scored_rows, _index, _mapping
        )

    # Aggregate per measure
    summaries = []
    for matcher_name in measure_names:
        a = accum[matcher_name]
        if a["mapped"] > 0:
            s = {
                "hP": sum(a["hP"]) / len(a["hP"]),
                "hR": sum(a["hR"]) / len(a["hR"]),
                "hF": sum(a["hF"]) / len(a["hF"]),
                "exact": sum(a["exact"]) / len(a["exact"]),
                "num_examples": len(scored_rows),
                "num_mapped": a["mapped"],
            }
        else:
            s = {
                "hP": 0.0, "hR": 0.0, "hF": 0.0, "exact": 0.0,
                "num_examples": len(scored_rows), "num_mapped": 0,
            }
        summaries.append({"measure": matcher_name, "metrics": s})

    # Attach mapping-method breakdown (exact / ngram / voting / top_score /
    # none) to the embedding measure for auditability.
    if embed_method_counts is not None:
        for _s in summaries:
            if _s["measure"] == EMBED_MEASURE:
                _s["metrics"]["mapping_methods"] = embed_method_counts

    # ── pass@k from judge verdicts ──────────────────────────────────
    # Uses the numerically stable product-form estimator:
    #   pass@k(n, c, k) = 1 - ∏_{i=0}^{k-1} (n - c - i) / (n - i)
    # which is equivalent to 1 - C(n-c, k) / C(n, k).
    from oven_mllm_eval.pass_at_k import pass_at_k as _pass_at_k_fn

    _judge_rows = [r for r in scored_rows if r.get("judge_verdicts")]
    if _judge_rows:
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
    for row in scored_rows:
        for k in _STALE_KEYS:
            row.pop(k, None)

    # Write per-sample scored JSONL (overwrites input by default)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in scored_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

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

    with open(summary_path, "w") as f:
        json.dump(_summary_data, f, indent=2)

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
        for row in rows:
            reference_path = row.get("scored_reference_path")
            predicted_path = row.get(f"{matcher_name}_predicted_path")
            if predicted_path is None or reference_path is None:
                continue
            hP.append(float(row.get(f"{matcher_name}_hP", 0.0) or 0.0))
            hR.append(float(row.get(f"{matcher_name}_hR", 0.0) or 0.0))
            hF.append(float(row.get(f"{matcher_name}_hF", 0.0) or 0.0))
            exact.append(int(bool(row.get(f"{matcher_name}_exact_match", False))))
            mapped += 1

        if mapped:
            metrics = {
                "hP": sum(hP) / len(hP),
                "hR": sum(hR) / len(hR),
                "hF": sum(hF) / len(hF),
                "exact": sum(exact) / len(exact),
                "num_examples": len(rows),
                "num_mapped": mapped,
            }
        else:
            metrics = {
                "hP": 0.0,
                "hR": 0.0,
                "hF": 0.0,
                "exact": 0.0,
                "num_examples": len(rows),
                "num_mapped": 0,
            }
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
