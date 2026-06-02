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
    """Derive the results JSON path from the samples JSONL path.

    ``<run_id>_samples.jsonl`` → ``<run_id>_results.json``
    """
    m = re.match(r"(.+)_samples\.jsonl$", samples_path.name)
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
        prediction = (row.get("prediction")
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

    for m in measure_names:
        if m not in ALL_MEASURES:
            raise ValueError(
                f"Unknown measure '{m}'. Available: {list(ALL_MEASURES.keys())}"
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

    # Score — parallel or serial
    if num_workers > 1:
        # Contiguous chunks of roughly equal size.  All rows do the same
        # work (one prediction vs ~12K node labels), so round-robin isn't
        # needed — and contiguous chunks preserve row order in output.
        chunk_size = (len(rows) + num_workers - 1) // num_workers
        chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
        args = [(chunk, measure_names, resolved_index_path) for chunk in chunks]

        try:
            with multiprocessing.Pool(num_workers) as pool:
                results = pool.map(_score_rows, args)
        except Exception:
            logger.warning(
                "multiprocessing.Pool.map failed — falling back to serial. "
                "Error details:", exc_info=True,
            )
            results = [_score_rows((rows, measure_names, resolved_index_path))]
    else:
        results = [_score_rows((rows, measure_names, resolved_index_path))]

    # Merge results from all chunks
    accum = {m: {"hP": [], "hR": [], "hF": [], "exact": [], "mapped": 0}
             for m in measure_names}
    scored_rows = []
    for chunk_rows, chunk_accum in results:
        scored_rows.extend(chunk_rows)
        for m in measure_names:
            for key in ("hP", "hR", "hF", "exact"):
                accum[m][key].extend(chunk_accum[m][key])
            accum[m]["mapped"] += chunk_accum[m]["mapped"]

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
    with open(summary_path, "w") as f:
        json.dump(summaries[0]["metrics"] if len(summaries) == 1 else summaries,
                  f, indent=2)

    return summaries[0]["metrics"] if len(summaries) == 1 else summaries
