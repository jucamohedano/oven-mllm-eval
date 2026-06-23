#!/usr/bin/env python3
"""Generate OVEN cleaned description chains from cleaned label chains."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from pathlib import Path


SPARQL_URL = "https://query.wikidata.org/sparql"


def load_eval_ids(path: Path) -> set[str]:
    ids = set()
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            entity_id = row.get("entity_id") or row.get("entity")
            if entity_id:
                ids.add(entity_id)
    return ids


def chunks(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def sparql_descriptions(qids: list[str], retries: int = 5) -> dict[str, dict[str, str]]:
    values = " ".join(f"wd:{qid}" for qid in qids)
    query = f"""
    SELECT ?root ?class ?classLabel ?classDescription WHERE {{
      VALUES ?root {{ {values} }}
      {{
        BIND(?root AS ?class)
      }}
      UNION
      {{
        ?root wdt:P31/wdt:P279* ?class .
      }}
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en" .
        ?class rdfs:label ?classLabel .
        ?class schema:description ?classDescription .
      }}
    }}
    """
    url = SPARQL_URL + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(
                    url,
                    headers={
                        "Accept": "application/sparql-results+json",
                        "User-Agent": "oven-mllm-eval/0.1",
                    },
                ),
                timeout=120,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            if attempt == retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            sleep_for = float(retry_after) if retry_after else 15 * (attempt + 1)
            time.sleep(sleep_for)
        except HTTPError as exc:
            if attempt == retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            sleep_for = float(retry_after) if retry_after else 15 * (attempt + 1)
            time.sleep(sleep_for)
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 * (attempt + 1))

    out: dict[str, dict[str, str]] = {qid: {} for qid in qids}
    for binding in payload["results"]["bindings"]:
        root = binding["root"]["value"].rsplit("/", 1)[-1]
        label = binding.get("classLabel", {}).get("value", "")
        desc = binding.get("classDescription", {}).get("value", "")
        if label and desc:
            out.setdefault(root, {}).setdefault(label, desc)
    return out


def root_descriptions(qids: list[str], retries: int = 5) -> dict[str, str]:
    values = " ".join(f"wd:{qid}" for qid in qids)
    query = f"""
    SELECT ?root ?rootDescription WHERE {{
      VALUES ?root {{ {values} }}
      ?root schema:description ?rootDescription .
      FILTER(LANG(?rootDescription) = "en")
    }}
    """
    url = SPARQL_URL + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(
                    url,
                    headers={
                        "Accept": "application/sparql-results+json",
                        "User-Agent": "oven-mllm-eval/0.1",
                    },
                ),
                timeout=60,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 * (attempt + 1))
    return {
        binding["root"]["value"].rsplit("/", 1)[-1]: binding["rootDescription"]["value"]
        for binding in payload["results"]["bindings"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="data/raw/oven_wikidata_chains_cleaned_labels.jsonl")
    parser.add_argument("--eval-jsonl", default="data/raw/oven_entity_val.jsonl")
    parser.add_argument("--output", default="data/raw/oven_wikidata_chains_cleaned_descs.jsonl")
    parser.add_argument("--cache", default="data/raw/oven_wikidata_desc_cache.json")
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    eval_ids = load_eval_ids(Path(args.eval_jsonl))
    label_rows = []
    with Path(args.labels).open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] in eval_ids:
                label_rows.append(row)

    label_by_id = {row["id"]: row["taxonomy"] for row in label_rows}
    qids = sorted(label_by_id)
    cache_path = Path(args.cache)
    desc_by_id: dict[str, dict[str, str]] = {}
    if cache_path.exists():
        desc_by_id = json.loads(cache_path.read_text())

    remaining = [qid for qid in qids if qid not in desc_by_id]
    for batch in chunks(remaining, args.chunk_size):
        desc_by_id.update(sparql_descriptions(batch))
        cache_path.write_text(json.dumps(desc_by_id, ensure_ascii=False))
        print(f"fetched {len(desc_by_id)}/{len(qids)}", flush=True)
        if args.sleep:
            time.sleep(args.sleep)

    root_remaining = [
        qid
        for qid in qids
        if label_by_id.get(qid) and not desc_by_id.get(qid, {}).get(label_by_id[qid][0])
    ]
    for batch in chunks(root_remaining, 25):
        roots = root_descriptions(batch)
        for qid, desc in roots.items():
            if desc and label_by_id.get(qid):
                desc_by_id.setdefault(qid, {})[label_by_id[qid][0]] = desc
        cache_path.write_text(json.dumps(desc_by_id, ensure_ascii=False))
        if args.sleep:
            time.sleep(args.sleep)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for qid in qids:
            descs = desc_by_id.get(qid, {})
            taxonomy = [descs.get(label, "") for label in label_by_id[qid]]
            handle.write(json.dumps({"id": qid, "taxonomy": taxonomy}, ensure_ascii=False) + "\n")

    missing = sum(
        1
        for qid, labels in label_by_id.items()
        for label in labels
        if not desc_by_id.get(qid, {}).get(label)
    )
    total = sum(len(labels) for labels in label_by_id.values())
    print(f"wrote={output} rows={len(qids)} missing_descriptions={missing}/{total}")


if __name__ == "__main__":
    main()
