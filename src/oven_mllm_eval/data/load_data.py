# ==============================================================================
# Adapted from vlm-eval/src/vlmeval/calculate_scores/load_data.py
#
# Used only by scripts/build_taxonomy_index.py (offline).  Requires networkx
# (install via --extra build-index).
# ==============================================================================

import json
import os
from collections import defaultdict
import random

import networkx as nx

from oven_mllm_eval.paths import (
    out_labels, PACKAGE_DIR, OVEN_SAMPLES, OVEN_IMAGES_DIR,
    INAT_IMAGES_DIR, OVEN_AKAS,
)


def _load_jsonl(path):
    """Read a JSONL file into a list of dicts (replaces datasets.load_dataset)."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def split_data(data, seed=0, ratios=(0.5, 0.1, 0.4)):
    """Split data into train, eval, and test sets based on provided ratios."""
    assert sum(ratios) == 1, "Ratios must sum to 1"

    if not isinstance(data, list):
        data = list(data)

    data.sort()

    random.seed(seed)
    random.shuffle(data)

    train_size = int(ratios[0] * len(data))
    eval_size = int(ratios[1] * len(data))
    test_size = len(data) - train_size - eval_size

    train_set = data[:train_size]
    eval_set = data[train_size:train_size + eval_size]
    test_set = data[train_size + eval_size:]
    dataset = {
        "train": train_set,
        "val": eval_set,
        "test": test_set
    }

    return dataset


def get_splits(data, name, seed=0, ratios=(0.5, 0.1, 0.4)):
    split_folder = "./splits"
    if not os.path.exists(split_folder):
        os.makedirs(split_folder)
    split_file = f"{split_folder}/split_{name}_{seed}_{ratios[0]}_{ratios[1]}_{ratios[2]}.json"
    if os.path.exists(split_file):
        try:
            with open(split_file, "r") as f:
                splits = json.load(f)
        except (json.JSONDecodeError, ValueError):
            os.remove(split_file)
    if not os.path.exists(split_file):
        splits = split_data(data, seed=seed, ratios=ratios)
        with open(split_file, "w") as f:
            json.dump(splits, f)

    for k, v in splits.items():
        tuple_set = []
        for item in v:
            tuple_set.append(tuple(item))
        splits[k] = tuple_set
    return splits


def build_networkx_tree(taxa):
    tree = nx.DiGraph()
    for taxon in taxa:
        for i in range(len(taxon)):
            parent = "root" if i == 0 else taxon[i - 1]
            child = taxon[i]
            tree.add_edge(parent, child)
    return tree


# ---------------------------------------------------------------------------
# load_inat — kept for reference, NOT used in our OVEN-only pipeline.
# ---------------------------------------------------------------------------

def load_inat(split="test"):
    inat_file = f"{PACKAGE_DIR}/calculate_scores/data_files/inat/val.json"
    inat_images = INAT_IMAGES_DIR
    inat_full_taxa = _load_jsonl(f"{PACKAGE_DIR}/calculate_scores/data_files/inat/inat_taxa.json")

    inat_data = json.load(open(inat_file))
    inat_taxa = [record["file_name"] for record in inat_data["images"]]
    print(f"Loaded {len(inat_taxa)} samples from {inat_file}")
    common_name_map = {r["name"]: [r["common_name"]] for r in inat_data["categories"]}

    for entry in inat_full_taxa:
        if (
            "preferred_common_name" in entry
            and entry["preferred_common_name"] is not None
        ):
            if entry["name"] not in common_name_map:
                common_name_map[entry["name"]] = [entry["preferred_common_name"]]

    ancestors = set()
    for entry in inat_taxa:
        parts = entry.split("/")[1].split("_")
        kingdom, phylum, class_, order, family, genus, species = parts[1:]
        species = f"{genus} {species}"
        ancestry = (kingdom, phylum, class_, order, family, genus, species)
        ancestors.add(ancestry)

    name_to_file = defaultdict(list)
    for entry in inat_data["images"]:
        name_spl = entry["file_name"].split("/")[-2].split("_")
        name = " ".join(name_spl[-2:])
        name = common_name_map.get(name, [name])[0]
        file_path = f"{inat_images}/{entry['file_name']}"
        name_to_file[name].append(file_path)

    tree = build_networkx_tree(ancestors)

    splits = get_splits(ancestors, "inat", seed=0, ratios=(0.00001, 0.00001, 0.99998))
    split = splits[split]
    assert set(split).issubset(set(ancestors)), "Split not subset of ancestors, cache issue?"
    return tree, split, common_name_map, name_to_file


# Insepction of KG show some issues, very generic terms
REMOVE_NODES_AFTER_AND_INCLUDING = set(
    [
        "class",
        "occurrence",
        "container",
        "representation",
        "intangible good",
        "church building",
        "fixed construction",
        "geographic location",
        "artificial object",
        "artificial physical object",
        "Ferienanlage",
        "social system",
        "product model",
        "intentional human activity",
    ]
)

REMOVE_AFTER = set(["reservoir"])


def load_oven(split="test", return_datasets=False):
    wikidb_chains = out_labels
    wikidb_akas = OVEN_AKAS
    oven_samples = OVEN_SAMPLES
    oven_images = OVEN_IMAGES_DIR

    oven_data = _load_jsonl(wikidb_chains)
    oven_akas = _load_jsonl(wikidb_akas)
    oven_samples = _load_jsonl(oven_samples)

    id_to_aka = {}

    for entry in oven_akas:
        if "en" not in entry["aka"]:
            continue
        if entry["aka"]["en"] is None:
            continue
        try:
            akas = [r["value"] for r in entry["aka"]["en"]]
        except:
            breakpoint()
        id_to_aka[entry["id"]] = akas

    print(f"Loaded {len(oven_data)} samples from {wikidb_chains}")
    ancestors = set()

    name_to_file = defaultdict(list)
    dataset_info = {}
    dataset_info["num_examples"] = defaultdict(int)
    id_to_name_samples = {}

    for entry in oven_samples:
        name = entry.get("original_answer", entry["answer"])
        dataset = entry["dataset"]
        dataset_info[entry["answer"]] = dataset
        dataset_info[entry["entity_text"]] = dataset
        dataset_info[name] = dataset

        id_to_name_samples[entry["entity_id"]] = name

        file_name = entry["image_path"].split("/")[-1]
        file_path = f"{oven_images}/{file_name}"

        name_to_file[name].append(file_path)

    aka_map = {}
    for entry in oven_data:
        if entry["id"] in id_to_name_samples:
            name_in_samples = id_to_name_samples[entry["id"]]
            entry["taxonomy"][0] = name_in_samples

        ancestry = entry["taxonomy"]
        ancestry.reverse()

        if set(ancestry).intersection(REMOVE_NODES_AFTER_AND_INCLUDING):
            for i, node in enumerate(ancestry):
                if node in REMOVE_NODES_AFTER_AND_INCLUDING:
                    ancestry = ancestry[i + 1:]
                    break

        if set(ancestry).intersection(REMOVE_AFTER):
            for i in range(len(ancestry) - 1, 0, -1):
                node = ancestry[i]
                if node in REMOVE_AFTER:
                    ancestry = ancestry[i:]
                    break

        ancestry = tuple(ancestry)
        ancestors.add(ancestry)
        if entry["id"] not in id_to_aka:
            continue
        new_names = [name for name in id_to_aka[entry["id"]] if name not in ancestry]
        aka_map[ancestry[-1]] = new_names

    tree = build_networkx_tree(ancestors)

    splits = get_splits(ancestors, "oven", seed=0, ratios=(0.00001, 0.00001, 0.99998))
    split = splits[split]
    assert set(split).issubset(set(ancestors)), "Split not subset of ancestors, cache issue?"
    if return_datasets:
        return tree, split, aka_map, name_to_file, dataset_info
    return tree, split, aka_map, name_to_file


if __name__ == "__main__":
    tree, split, common_name_map, name_to_file, dataset_info = load_oven(return_datasets=True)
    print(f"Tree has {tree.number_of_nodes()} nodes and {tree.number_of_edges()} edges")
    print(f"Split has {len(split)} entries")
