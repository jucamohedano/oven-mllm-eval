#!/usr/bin/env python3
"""Visualize sampled OVEN taxonomy chains as an interactive HTML graph.

Samples a few taxonomy paths from the index, builds a small tree from just
those chains, and renders it with PyVis. Leaf nodes show example images (if
they exist on disk) or the image path as a tooltip.

Usage:
    # Sample 5 random chains
    python scripts/visualize_taxonomy.py --n-chains 5

    # Sample chains containing a specific label
    python scripts/visualize_taxonomy.py --n-chains 5 --filter "aircraft"

    # More chains, only deep ones
    python scripts/visualize_taxonomy.py --n-chains 20 --min-depth 4
"""
from __future__ import annotations

import argparse
import base64
import json
import random
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import networkx as nx
from pyvis.network import Network


def sample_chains(
    label_to_paths: dict[str, list[list[str]]],
    n_chains: int = 5,
    min_depth: int = 2,
    filter_label: str | None = None,
    seed: int = 42,
) -> list[list[str]]:
    """Sample N taxonomy chains, optionally filtering by label substring."""
    candidates = []
    for label, paths in label_to_paths.items():
        for path in paths:
            if len(path) < min_depth:
                continue
            if filter_label and filter_label.lower() not in " ".join(path).lower():
                continue
            candidates.append(path)

    rng = random.Random(seed)
    if len(candidates) <= n_chains:
        return candidates
    return rng.sample(candidates, n_chains)


def chains_to_tree(chains: list[list[str]]) -> nx.DiGraph:
    """Build a DiGraph from sampled chains."""
    G = nx.DiGraph()
    for path in chains:
        for parent, child in zip(path, path[1:]):
            G.add_edge(parent, child)
    return G


def load_image_map(data_path: str | Path) -> dict[str, list[str]]:
    """Map entity labels to image file paths from the prepared JSONL."""
    label_to_images: dict[str, list[str]] = defaultdict(list)
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row.get("original_answer") or row.get("answer", "")
            img = row.get("image_path", "")
            if label and img:
                label_to_images[label].append(img)
    return dict(label_to_images)


def _encode_image(path: str, max_px: int = 128) -> str | None:
    """Base64-encode an image, resizing to max_px on the longest side."""
    try:
        from PIL import Image
    except ImportError:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        img = Image.open(p).convert("RGB")
        img.thumbnail((max_px, max_px))
        buf = BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def _tree_layout(G: nx.DiGraph, x_span: int = 1200, y_span: int = 800) -> dict[str, tuple[int, int]]:
    """Simple top-down tree layout without numpy or graphviz.

    Roots at top, leaves at bottom. Siblings spread horizontally.
    """
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    if not roots:
        roots = [list(G.nodes())[0]]

    # BFS to assign depth
    depth: dict[str, int] = {}
    children: dict[str, list[str]] = defaultdict(list)
    for parent, child in G.edges():
        children[parent].append(child)

    queue = [(r, 0) for r in roots]
    visited = set()
    while queue:
        node, d = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        depth[node] = d
        for child in children[node]:
            if child not in visited:
                queue.append((child, d + 1))

    # Any nodes not reached (disconnected) get depth 0
    for node in G.nodes():
        if node not in depth:
            depth[node] = 0

    max_depth = max(depth.values()) or 1

    # Assign x by spreading leaves evenly, parents centered over their children
    pos: dict[str, tuple[int, int]] = {}
    leaf_counter = [0]
    total_leaves = sum(1 for n in G.nodes() if G.out_degree(n) == 0) or 1

    def _assign(node: str) -> int:
        """Return the x-center for this subtree."""
        kids = children[node]
        if not kids:
            x = int(50 + leaf_counter[0] / total_leaves * x_span)
            leaf_counter[0] += 1
            y = int(depth[node] / max_depth * y_span)
            pos[node] = (x, y)
            return x
        child_xs = [_assign(c) for c in kids if c not in pos]
        # For kids already assigned (shared nodes), just use their position
        for c in kids:
            if c in pos and c not in [k for k in kids if k in pos and kids.index(c) < len(child_xs)]:
                child_xs.append(pos[c][0])
        x = sum(child_xs) // len(child_xs) if child_xs else 0
        y = int(depth[node] / max_depth * y_span)
        pos[node] = (x, y)
        return x

    for r in roots:
        _assign(r)

    # Nodes not visited by the tree walk (disconnected)
    for node in G.nodes():
        if node not in pos:
            pos[node] = (leaf_counter[0] * 80 % x_span, 0)

    return pos


def render_html(
    G: nx.DiGraph,
    image_map: dict[str, list[str]] | None = None,
    output: str = "viz/taxonomy_sample.html",
):
    """Render the tree as an interactive PyVis HTML page."""
    net = Network(height="900px", width="100%", directed=True,
                  bgcolor="#1a1a2e", font_color="#e0e0e0")

    pos = _tree_layout(G)

    for node in G.nodes():
        x, y = pos[node]

        is_leaf = G.out_degree(node) == 0
        is_root = G.in_degree(node) == 0
        has_image = image_map and node in image_map

        # Tooltip HTML
        title = f"<b>{node}</b>"
        if has_image:
            imgs = image_map[node]
            title += f"<br>{len(imgs)} image(s)"
            data_uri = _encode_image(imgs[0])
            if data_uri:
                title += f'<br><img src="{data_uri}" width="128">'

        size = 25 if is_leaf else 15
        color = "#e94560" if is_leaf else ("#0f3460" if is_root else "#16213e")

        net.add_node(node, label=node, title=title, x=x, y=y,
                     size=size, color=color, font={"size": 12})

    for parent, child in G.edges():
        net.add_edge(parent, child, color="#533483", width=1)

    # Physics off — positions set manually for tree layout
    net.set_options('''{
        "physics": {"enabled": false},
        "edges": {"smooth": {"type": "curvedCW", "roundness": 0.15}},
        "interaction": {"hover": true, "tooltipDelay": 100}
    }''')

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    net.show(output, notebook=False)


def main():
    parser = argparse.ArgumentParser(description="Visualize sampled OVEN taxonomy chains")
    parser.add_argument("--taxonomy", default="data/processed/taxonomy_index.json")
    parser.add_argument("--data", default="data/processed/vlm_compatible_val.jsonl",
                        help="Prepared JSONL for image lookup")
    parser.add_argument("--output", default="viz/taxonomy_sample.html")
    parser.add_argument("--n-chains", type=int, default=5,
                        help="Number of chains to sample")
    parser.add_argument("--min-depth", type=int, default=2,
                        help="Minimum chain depth to consider")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only sample chains containing this label substring")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.taxonomy) as f:
        index = json.load(f)

    chains = sample_chains(
        index["label_to_paths"],
        n_chains=args.n_chains,
        min_depth=args.min_depth,
        filter_label=args.filter,
        seed=args.seed,
    )
    if not chains:
        print("No chains matched. Try lowering --min-depth or changing --filter.")
        return

    G = chains_to_tree(chains)
    print(f"Sampled {len(chains)} chains → {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    image_map = None
    if Path(args.data).exists():
        image_map = load_image_map(args.data)
        n_with_img = sum(1 for n in G.nodes() if n in image_map)
        print(f"Image map: {n_with_img}/{G.number_of_nodes()} nodes have images")

    render_html(G, image_map, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
