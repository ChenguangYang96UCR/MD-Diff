#!/usr/bin/env python3
"""Profile 1D discrete-Morse construction on ogbn-arxiv.

This script is designed for scalability measurement, not model training.
It:

1. downloads/loads ogbn-arxiv with OGB's library-agnostic loader;
2. removes self-loops, symmetrizes, and deduplicates citation edges;
3. extracts connected BFS subgraphs at requested node counts;
4. constructs the current degree-based vertex score and an equivalent
   array-based 1D discrete Morse function in O(V + E) storage;
5. identifies critical vertices/edges and the gradient pairing;
6. samples the same critical-cell pairs for four path-search settings;
7. records runtime and process RSS memory in CSV files.

The four path configurations are:

    pairing + enforce_f=True
    pairing + enforce_f=False
    relaxed + enforce_f=True
    relaxed + enforce_f=False

The current repository's generic simplex implementation performs expensive
global scans and clique enumeration. This profiler intentionally specializes
the same construction to a graph (maximum dimension 1), using integer arrays
and CSR incidence lists so that the full ogbn-arxiv graph can be measured.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import psutil
from ogb.nodeproppred import NodePropPredDataset
from scipy import sparse


PATH_CONFIGS = (
    ("pairing", True),
    ("pairing", False),
    ("relaxed", True),
    ("relaxed", False),
)

CONSTRUCTION_FIELDS = (
    "size_requested",
    "nodes",
    "edges",
    "density",
    "morse_seed",
    "stage",
    "seconds",
    "rss_start_mb",
    "rss_end_mb",
    "rss_peak_mb",
    "rss_delta_mb",
    "critical_vertices",
    "critical_edges",
    "total_critical_cells",
    "gradient_pairs",
    "status",
    "error",
)

PATH_FIELDS = (
    "size_requested",
    "nodes",
    "edges",
    "mode",
    "enforce_f",
    "eps",
    "sampled_pairs",
    "successful_pairs",
    "successful_pair_ratio",
    "unique_path_edges",
    "sampled_edge_coverage",
    "mean_path_length",
    "p50_path_length",
    "p95_path_length",
    "mean_bfs_steps",
    "p50_bfs_steps",
    "p95_bfs_steps",
    "seconds",
    "seconds_per_pair",
    "rss_start_mb",
    "rss_end_mb",
    "rss_peak_mb",
    "rss_delta_mb",
    "candidate_all_pairs",
    "estimated_all_pairs_seconds",
    "status",
    "error",
)


def mb(value: int) -> float:
    return value / (1024.0 * 1024.0)


class RSSMonitor:
    """Sample process RSS in a background thread during one stage."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.start_rss = 0
        self.end_rss = 0
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            rss = self.process.memory_info().rss
            if rss > self.peak_rss:
                self.peak_rss = rss
            self._stop.wait(self.interval)

    def __enter__(self) -> "RSSMonitor":
        self.start_rss = self.process.memory_info().rss
        self.peak_rss = self.start_rss
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.end_rss = self.process.memory_info().rss
        self.peak_rss = max(self.peak_rss, self.end_rss)

    def values_mb(self) -> Dict[str, float]:
        return {
            "rss_start_mb": mb(self.start_rss),
            "rss_end_mb": mb(self.end_rss),
            "rss_peak_mb": mb(self.peak_rss),
            "rss_delta_mb": mb(self.end_rss - self.start_rss),
        }


@contextmanager
def timed_rss_stage(name: str):
    start = time.perf_counter()
    with RSSMonitor() as monitor:
        payload = {}
        yield payload
    elapsed = time.perf_counter() - start
    payload.update(
        {
            "stage": name,
            "seconds": elapsed,
            **monitor.values_mb(),
        }
    )


def append_csv(path: Path, fields: Sequence[str], rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def load_ogbn_arxiv(root: str) -> Tuple[int, np.ndarray]:
    # OGB currently loads its trusted preprocessed dataset cache through a
    # torch.load() call that does not pass weights_only. PyTorch >= 2.6 changed
    # that call's default to weights_only=True, which is incompatible with the
    # legacy OGB pickle. This environment variable changes only torch.load
    # callsites that did not explicitly provide weights_only.
    #
    # Security: use this only for the official OGB dataset/cache downloaded
    # from a trusted source. weights_only=False uses Python's general unpickler.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    dataset = NodePropPredDataset(name="ogbn-arxiv", root=root)
    graph, _ = dataset[0]
    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    return int(graph["num_nodes"]), edge_index


def undirected_unique_edges(
    num_nodes: int,
    edge_index: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    src = edge_index[0]
    dst = edge_index[1]
    keep = src != dst
    src = src[keep]
    dst = dst[keep]

    u = np.minimum(src, dst).astype(np.int64, copy=False)
    v = np.maximum(src, dst).astype(np.int64, copy=False)
    keys = u * np.int64(num_nodes) + v
    keys = np.unique(keys)
    u = keys // np.int64(num_nodes)
    v = keys % np.int64(num_nodes)
    return u.astype(np.int64), v.astype(np.int64)


def build_symmetric_csr(
    num_nodes: int,
    u: np.ndarray,
    v: np.ndarray,
) -> sparse.csr_matrix:
    rows = np.concatenate((u, v))
    cols = np.concatenate((v, u))
    data = np.ones(rows.shape[0], dtype=np.uint8)
    matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(num_nodes, num_nodes),
        dtype=np.uint8,
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def bfs_nodes(
    adjacency: sparse.csr_matrix,
    target_size: int,
    seed_node: int,
) -> np.ndarray:
    """Return up to target_size nodes, continuing across components if needed."""
    n = adjacency.shape[0]
    target_size = min(target_size, n)
    visited = np.zeros(n, dtype=np.bool_)
    selected = np.empty(target_size, dtype=np.int64)
    count = 0

    start_candidates = [seed_node]
    degree_order = np.argsort(-np.diff(adjacency.indptr))
    start_candidates.extend(degree_order.tolist())

    for start in start_candidates:
        if count >= target_size:
            break
        if visited[start]:
            continue

        queue = deque([int(start)])
        visited[start] = True

        while queue and count < target_size:
            node = queue.popleft()
            selected[count] = node
            count += 1

            begin, end = adjacency.indptr[node], adjacency.indptr[node + 1]
            for neighbor in adjacency.indices[begin:end]:
                neighbor = int(neighbor)
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

    return selected[:count]


def induced_edges(
    adjacency: sparse.csr_matrix,
    nodes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    subgraph = adjacency[nodes][:, nodes].tocoo()
    keep = subgraph.row < subgraph.col
    u = subgraph.row[keep].astype(np.int64)
    v = subgraph.col[keep].astype(np.int64)
    return u, v


def build_incidence_csr(
    num_nodes: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """CSR mapping from a vertex to incident edge indices."""
    m = edge_u.shape[0]
    vertices = np.concatenate((edge_u, edge_v))
    edge_ids = np.concatenate(
        (np.arange(m, dtype=np.int64), np.arange(m, dtype=np.int64))
    )
    order = np.argsort(vertices, kind="stable")
    vertices = vertices[order]
    edge_ids = edge_ids[order]
    counts = np.bincount(vertices, minlength=num_nodes)
    indptr = np.empty(num_nodes + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    return indptr, edge_ids


def construct_morse_1d(
    num_nodes: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    seed: int,
) -> dict:
    """Equivalent array-based specialization of the current 1D pipeline."""
    rng = np.random.default_rng(seed)
    degree = np.bincount(
        np.concatenate((edge_u, edge_v)),
        minlength=num_nodes,
    ).astype(np.int64)
    max_degree = int(degree.max()) if num_nodes else 0

    # Current initialize_vertex_weights:
    # g(v) = max_degree - degree(v) + Uniform(0, 0.5).
    vertex_f = (
        max_degree
        - degree.astype(np.float64)
        + rng.uniform(0.0, 0.5, size=num_nodes)
    )

    m = edge_u.shape[0]
    edge_f = np.empty(m, dtype=np.float64)
    flag = np.zeros(num_nodes, dtype=np.bool_)

    # Match construct_discrete_morse_function's edge iteration order.
    for edge_id in range(m):
        u = int(edge_u[edge_id])
        v = int(edge_v[edge_id])

        if vertex_f[u] >= vertex_f[v]:
            high, low = u, v
        else:
            high, low = v, u

        if (not flag[high]) and vertex_f[high] > vertex_f[low]:
            edge_f[edge_id] = (vertex_f[high] + vertex_f[low]) / 2.0
            flag[high] = True
        else:
            edge_f[edge_id] = vertex_f[high] + rng.uniform(0.0, 0.5)

    incidence_indptr, incidence_edges = build_incidence_csr(
        num_nodes,
        edge_u,
        edge_v,
    )

    # A vertex is critical iff no incident edge beta satisfies f(beta) <= f(v).
    critical_vertex = np.ones(num_nodes, dtype=np.bool_)
    for vertex in range(num_nodes):
        begin = incidence_indptr[vertex]
        end = incidence_indptr[vertex + 1]
        incident = incidence_edges[begin:end]
        if incident.size and np.any(edge_f[incident] <= vertex_f[vertex]):
            critical_vertex[vertex] = False

    # With max_dimension=1, an edge is critical iff neither endpoint has
    # f(vertex) >= f(edge).
    critical_edge = (
        (vertex_f[edge_u] < edge_f)
        & (vertex_f[edge_v] < edge_f)
    )

    # Match construct_gradient_vector_field: iterate edges, pair an edge with
    # the eligible unpaired endpoint having minimum f.
    paired_vertex_edge = np.full(num_nodes, -1, dtype=np.int64)
    paired_edge_vertex = np.full(m, -1, dtype=np.int64)

    for edge_id in range(m):
        u = int(edge_u[edge_id])
        v = int(edge_v[edge_id])
        candidates = []

        if paired_vertex_edge[u] < 0 and edge_f[edge_id] <= vertex_f[u]:
            candidates.append(u)
        if paired_vertex_edge[v] < 0 and edge_f[edge_id] <= vertex_f[v]:
            candidates.append(v)

        if candidates:
            vertex = min(candidates, key=lambda x: vertex_f[x])
            paired_vertex_edge[vertex] = edge_id
            paired_edge_vertex[edge_id] = vertex

    return {
        "vertex_f": vertex_f,
        "edge_f": edge_f,
        "critical_vertex": critical_vertex,
        "critical_edge": critical_edge,
        "paired_vertex_edge": paired_vertex_edge,
        "paired_edge_vertex": paired_edge_vertex,
        "incidence_indptr": incidence_indptr,
        "incidence_edges": incidence_edges,
    }


def critical_cell_codes(morse: dict, num_nodes: int) -> np.ndarray:
    vertices = np.flatnonzero(morse["critical_vertex"]).astype(np.int64)
    edges = np.flatnonzero(morse["critical_edge"]).astype(np.int64)
    return np.concatenate((vertices, num_nodes + edges))


def sample_cell_pairs(
    cells: np.ndarray,
    sample_count: int,
    seed: int,
) -> List[Tuple[int, int]]:
    candidate_count = len(cells) * (len(cells) - 1) // 2
    sample_count = min(sample_count, candidate_count)
    if sample_count <= 0:
        return []

    rng = random.Random(seed)
    pairs = set()

    while len(pairs) < sample_count:
        i = rng.randrange(len(cells))
        j = rng.randrange(len(cells) - 1)
        if j >= i:
            j += 1
        a, b = int(cells[i]), int(cells[j])
        pairs.add((a, b) if a < b else (b, a))

    return sorted(pairs)


def cell_dimension(code: int, num_nodes: int) -> int:
    return 0 if code < num_nodes else 1


def gradient_neighbors_codes(
    code: int,
    *,
    num_nodes: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    morse: dict,
    mode: str,
    enforce_f: bool,
    eps: float,
) -> Iterable[int]:
    vertex_f = morse["vertex_f"]
    edge_f = morse["edge_f"]

    if code < num_nodes:
        vertex = code

        if mode == "pairing":
            edge_id = int(morse["paired_vertex_edge"][vertex])
            if edge_id >= 0:
                if (not enforce_f) or (
                    edge_f[edge_id] <= vertex_f[vertex] + eps
                ):
                    yield num_nodes + edge_id
        elif mode == "relaxed":
            begin = morse["incidence_indptr"][vertex]
            end = morse["incidence_indptr"][vertex + 1]
            for edge_id in morse["incidence_edges"][begin:end]:
                edge_id = int(edge_id)
                if (not enforce_f) or (
                    edge_f[edge_id] <= vertex_f[vertex] + eps
                ):
                    yield num_nodes + edge_id
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    else:
        edge_id = code - num_nodes
        paired_face = int(morse["paired_edge_vertex"][edge_id])
        for vertex in (int(edge_u[edge_id]), int(edge_v[edge_id])):
            if vertex == paired_face:
                continue
            if (not enforce_f) or (
                vertex_f[vertex] < edge_f[edge_id] - eps
            ):
                yield vertex


def find_path_codes(
    start: int,
    end: int,
    *,
    num_nodes: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    morse: dict,
    mode: str,
    enforce_f: bool,
    eps: float,
    max_steps: int,
) -> Tuple[Optional[List[int]], int]:
    """BFS equivalent of find_between_criticals for encoded 0/1-cells."""
    if cell_dimension(start, num_nodes) < cell_dimension(end, num_nodes):
        start, end = end, start
        reverse_result = True
    else:
        reverse_result = False

    if start == end:
        return [start], 0

    queue = deque([start])
    parent = {start: -1}
    steps = 0

    while queue and steps < max_steps:
        current = queue.popleft()
        steps += 1

        for neighbor in gradient_neighbors_codes(
            current,
            num_nodes=num_nodes,
            edge_u=edge_u,
            edge_v=edge_v,
            morse=morse,
            mode=mode,
            enforce_f=enforce_f,
            eps=eps,
        ):
            if neighbor in parent:
                continue
            parent[neighbor] = current

            if neighbor == end:
                path = [end]
                cursor = end
                while parent[cursor] != -1:
                    cursor = parent[cursor]
                    path.append(cursor)
                path.reverse()
                if reverse_result:
                    path.reverse()
                return path, steps

            queue.append(neighbor)

    return None, steps


def profile_paths(
    pairs: Sequence[Tuple[int, int]],
    *,
    num_nodes: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    morse: dict,
    mode: str,
    enforce_f: bool,
    eps: float,
    max_steps: int,
) -> dict:
    successful = 0
    path_lengths = []
    bfs_steps = []
    covered_edges = set()

    for start, end in pairs:
        path, steps = find_path_codes(
            start,
            end,
            num_nodes=num_nodes,
            edge_u=edge_u,
            edge_v=edge_v,
            morse=morse,
            mode=mode,
            enforce_f=enforce_f,
            eps=eps,
            max_steps=max_steps,
        )
        bfs_steps.append(steps)

        if path is None:
            continue

        successful += 1
        path_lengths.append(len(path) - 1)
        for code in path:
            if code >= num_nodes:
                covered_edges.add(code - num_nodes)

    return {
        "successful": successful,
        "path_lengths": np.asarray(path_lengths, dtype=np.float64),
        "bfs_steps": np.asarray(bfs_steps, dtype=np.float64),
        "covered_edges": covered_edges,
    }


def percentile_or_zero(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0


def construction_row(
    requested_size: int,
    n: int,
    m: int,
    seed: int,
    payload: dict,
    critical_vertices: int = 0,
    critical_edges: int = 0,
    gradient_pairs: int = 0,
) -> dict:
    return {
        "size_requested": requested_size,
        "nodes": n,
        "edges": m,
        "density": (2.0 * m / (n * (n - 1))) if n > 1 else 0.0,
        "morse_seed": seed,
        **payload,
        "critical_vertices": critical_vertices,
        "critical_edges": critical_edges,
        "total_critical_cells": critical_vertices + critical_edges,
        "gradient_pairs": gradient_pairs,
        "status": "ok",
        "error": "",
    }


def parse_sizes(text: str, full_size: int) -> List[int]:
    result = []
    for item in text.split(","):
        item = item.strip().lower()
        value = full_size if item == "full" else int(item)
        value = min(value, full_size)
        if value > 1 and value not in result:
            result.append(value)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="dataset/ogb",
        help="OGB download/cache directory.",
    )
    parser.add_argument(
        "--sizes",
        default="1000,5000,10000,25000,50000,full",
        help="Comma-separated BFS subgraph sizes; use 'full' for all nodes.",
    )
    parser.add_argument("--morse_seed", type=int, default=42)
    parser.add_argument("--path_seed", type=int, default=2026)
    parser.add_argument(
        "--sample_pairs",
        type=int,
        default=1000,
        help="Critical-cell pairs sampled per path configuration.",
    )
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument(
        "--skip_paths_above",
        type=int,
        default=50000,
        help="Skip sampled path search when a subgraph exceeds this node count.",
    )
    parser.add_argument(
        "--output_dir",
        default="morse_ogbn_arxiv_profile",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    construction_csv = output_dir / "construction_profile.csv"
    paths_csv = output_dir / "path_profile.csv"
    metadata_json = output_dir / "metadata.json"

    for path in (construction_csv, paths_csv):
        if path.exists():
            path.unlink()

    metadata = {
        "dataset": "ogbn-arxiv",
        "root": args.root,
        "morse_seed": args.morse_seed,
        "path_seed": args.path_seed,
        "sample_pairs": args.sample_pairs,
        "eps": args.eps,
        "max_steps": args.max_steps,
        "path_configs": PATH_CONFIGS,
    }

    print("Loading ogbn-arxiv...")
    with timed_rss_stage("load_ogb") as load_payload:
        num_nodes, directed_edge_index = load_ogbn_arxiv(args.root)
    print(load_payload)

    with timed_rss_stage("symmetrize_deduplicate") as sym_payload:
        full_u, full_v = undirected_unique_edges(
            num_nodes,
            directed_edge_index,
        )
        del directed_edge_index
        gc.collect()
    print(sym_payload)

    with timed_rss_stage("build_full_csr") as csr_payload:
        full_adjacency = build_symmetric_csr(num_nodes, full_u, full_v)
    print(csr_payload)

    full_edges = int(full_u.shape[0])
    degrees = np.diff(full_adjacency.indptr)
    seed_node = int(np.argmax(degrees))
    sizes = parse_sizes(args.sizes, num_nodes)

    metadata.update(
        {
            "official_directed_nodes": num_nodes,
            "undirected_unique_edges": full_edges,
            "bfs_seed_node": seed_node,
            "sizes": sizes,
            "load_stage": load_payload,
            "symmetrize_stage": sym_payload,
            "csr_stage": csr_payload,
        }
    )
    metadata_json.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(
        f"Full graph: {num_nodes:,} nodes, "
        f"{full_edges:,} unique undirected edges"
    )

    for requested_size in sizes:
        print(f"\n===== Requested size: {requested_size:,} =====")

        try:
            with timed_rss_stage("extract_subgraph") as sub_payload:
                if requested_size == num_nodes:
                    sub_nodes = np.arange(num_nodes, dtype=np.int64)
                    edge_u = full_u
                    edge_v = full_v
                else:
                    sub_nodes = bfs_nodes(
                        full_adjacency,
                        requested_size,
                        seed_node,
                    )
                    edge_u, edge_v = induced_edges(
                        full_adjacency,
                        sub_nodes,
                    )

            n = int(sub_nodes.shape[0])
            m = int(edge_u.shape[0])
            append_csv(
                construction_csv,
                CONSTRUCTION_FIELDS,
                [
                    construction_row(
                        requested_size,
                        n,
                        m,
                        args.morse_seed,
                        sub_payload,
                    )
                ],
            )
            print(
                f"Subgraph: {n:,} nodes, {m:,} edges; "
                f"{sub_payload['seconds']:.3f}s; "
                f"peak RSS {sub_payload['rss_peak_mb']:.1f} MB"
            )

            with timed_rss_stage("construct_morse_1d") as morse_payload:
                morse = construct_morse_1d(
                    n,
                    edge_u,
                    edge_v,
                    args.morse_seed,
                )

            critical_vertices = int(morse["critical_vertex"].sum())
            critical_edges = int(morse["critical_edge"].sum())
            gradient_pairs = int(
                np.count_nonzero(morse["paired_vertex_edge"] >= 0)
            )
            row = construction_row(
                requested_size,
                n,
                m,
                args.morse_seed,
                morse_payload,
                critical_vertices,
                critical_edges,
                gradient_pairs,
            )
            append_csv(
                construction_csv,
                CONSTRUCTION_FIELDS,
                [row],
            )
            print(
                f"Morse: {morse_payload['seconds']:.3f}s; "
                f"peak RSS {morse_payload['rss_peak_mb']:.1f} MB; "
                f"critical V/E={critical_vertices:,}/{critical_edges:,}; "
                f"gradient pairs={gradient_pairs:,}"
            )

            cells = critical_cell_codes(morse, n)
            candidate_all_pairs = len(cells) * (len(cells) - 1) // 2

            if n > args.skip_paths_above or args.sample_pairs <= 0:
                print(
                    "Skipping sampled path search at this size. "
                    f"Critical cells={len(cells):,}; "
                    f"all-pairs candidates={candidate_all_pairs:,}"
                )
                continue

            sampled_pairs = sample_cell_pairs(
                cells,
                args.sample_pairs,
                args.path_seed,
            )

            for mode, enforce_f in PATH_CONFIGS:
                print(
                    f"Paths: mode={mode}, enforce_f={enforce_f}, "
                    f"pairs={len(sampled_pairs):,}"
                )

                with timed_rss_stage("sampled_path_search") as path_payload:
                    stats = profile_paths(
                        sampled_pairs,
                        num_nodes=n,
                        edge_u=edge_u,
                        edge_v=edge_v,
                        morse=morse,
                        mode=mode,
                        enforce_f=enforce_f,
                        eps=args.eps,
                        max_steps=args.max_steps,
                    )

                pair_count = len(sampled_pairs)
                elapsed = path_payload["seconds"]
                seconds_per_pair = elapsed / pair_count if pair_count else 0.0
                successful = stats["successful"]

                path_row = {
                    "size_requested": requested_size,
                    "nodes": n,
                    "edges": m,
                    "mode": mode,
                    "enforce_f": enforce_f,
                    "eps": args.eps,
                    "sampled_pairs": pair_count,
                    "successful_pairs": successful,
                    "successful_pair_ratio": (
                        successful / pair_count if pair_count else 0.0
                    ),
                    "unique_path_edges": len(stats["covered_edges"]),
                    "sampled_edge_coverage": (
                        len(stats["covered_edges"]) / m if m else 0.0
                    ),
                    "mean_path_length": (
                        float(stats["path_lengths"].mean())
                        if stats["path_lengths"].size
                        else 0.0
                    ),
                    "p50_path_length": percentile_or_zero(
                        stats["path_lengths"], 50
                    ),
                    "p95_path_length": percentile_or_zero(
                        stats["path_lengths"], 95
                    ),
                    "mean_bfs_steps": (
                        float(stats["bfs_steps"].mean())
                        if stats["bfs_steps"].size
                        else 0.0
                    ),
                    "p50_bfs_steps": percentile_or_zero(
                        stats["bfs_steps"], 50
                    ),
                    "p95_bfs_steps": percentile_or_zero(
                        stats["bfs_steps"], 95
                    ),
                    "seconds": elapsed,
                    "seconds_per_pair": seconds_per_pair,
                    **{
                        key: path_payload[key]
                        for key in (
                            "rss_start_mb",
                            "rss_end_mb",
                            "rss_peak_mb",
                            "rss_delta_mb",
                        )
                    },
                    "candidate_all_pairs": candidate_all_pairs,
                    "estimated_all_pairs_seconds": (
                        seconds_per_pair * candidate_all_pairs
                    ),
                    "status": "ok",
                    "error": "",
                }
                append_csv(paths_csv, PATH_FIELDS, [path_row])

                print(
                    f"  success={successful}/{pair_count}; "
                    f"time={elapsed:.3f}s; "
                    f"peak RSS={path_payload['rss_peak_mb']:.1f} MB; "
                    f"estimated all-pairs="
                    f"{path_row['estimated_all_pairs_seconds'] / 3600:.2f}h"
                )

            del morse, cells, sampled_pairs
            if requested_size != num_nodes:
                del edge_u, edge_v, sub_nodes
            gc.collect()

        except Exception as error:
            error_row = {
                field: "" for field in CONSTRUCTION_FIELDS
            }
            error_row.update(
                {
                    "size_requested": requested_size,
                    "stage": "failed",
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            append_csv(
                construction_csv,
                CONSTRUCTION_FIELDS,
                [error_row],
            )
            print(f"FAILED size={requested_size}: {error}")
            raise

    print("\nDone.")
    print(f"Construction CSV: {construction_csv.resolve()}")
    print(f"Path CSV:         {paths_csv.resolve()}")
    print(f"Metadata JSON:    {metadata_json.resolve()}")


if __name__ == "__main__":
    main()
