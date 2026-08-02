#!/usr/bin/env python3
"""Compare gradient-path search settings on all MG-Diff datasets.

Place this file in the ``MG-Diff_v1`` directory and run it from there.
The script does not train a model. For every requested dataset, it builds the
current degree-based discrete Morse function once and evaluates these four
gradient-path configurations:

1. mode="pairing", enforce_f=True
2. mode="pairing", enforce_f=False
3. mode="relaxed", enforce_f=True
4. mode="relaxed", enforce_f=False

For each configuration it reports the number of successful critical-cell
pairs, the union of graph edges on their paths, and the fraction of original
graph edges covered by that union.
"""

import argparse
import csv
import pickle
import time
import traceback
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np

from models.morsediffusion.morse_function import (
    build_clique_complex,
    construct_discrete_morse_function,
    construct_gradient_vector_field,
    find_between_criticals,
    identify_critical_simplices,
    initialize_vertex_weights,
)


DATASET_CONFIGS = {
    "PEMS03": {
        "type": "pickle",
        "path": "dataset/pems/PEMS03/adj_mx_03.pkl",
    },
    "PEMSBAY": {
        "type": "pickle",
        "path": "dataset/pemsbay/adj_mx_bay.pkl",
    },
    "BJAir": {
        "type": "npy",
        "path": "dataset/airquality/beijing/beijing_adj.npy",
    },
    "GZAir": {
        "type": "npy",
        "path": "dataset/airquality/guangzhou/guangzhou_adj.npy",
    },
}

DEFAULT_DATASETS = ["PEMS03", "PEMSBAY", "BJAir", "GZAir"]
PATH_CONFIGURATIONS = [
    ("pairing", True),
    ("pairing", False),
    ("relaxed", True),
    ("relaxed", False),
]

CSV_FIELDS = [
    "dataset",
    "mode",
    "enforce_f",
    "eps",
    "morse_seed",
    "max_dimension",
    "original_nodes",
    "original_edges",
    "graph_density",
    "critical_nodes",
    "critical_edges",
    "critical_faces",
    "total_critical_cells",
    "gradient_pairs",
    "candidate_critical_pairs",
    "successful_critical_pairs",
    "successful_pair_ratio",
    "skeleton_edges",
    "skeleton_edge_ratio",
    "skeleton_covered_nodes",
    "skeleton_node_ratio",
    "skeleton_isolated_nodes",
    "path_runtime_seconds",
    "status",
    "error",
]


def load_pickle(path):
    """Load an ordinary or Python-2-compatible pickle file."""
    with open(path, "rb") as file:
        try:
            return pickle.load(file)
        except UnicodeDecodeError:
            file.seek(0)
            return pickle.load(file, encoding="latin1")


def load_adjacency(dataset_name):
    """Load one configured adjacency matrix."""
    config = DATASET_CONFIGS[dataset_name]
    path = Path(config["path"])

    if not path.exists():
        raise FileNotFoundError(
            f"Adjacency file for {dataset_name} does not exist: {path}"
        )

    if config["type"] == "pickle":
        value = load_pickle(path)
        adjacency = value[-1] if isinstance(value, (list, tuple)) else value
    elif config["type"] == "npy":
        adjacency = np.load(path)
    else:
        raise ValueError(f"Unsupported adjacency type: {config['type']}")

    adjacency = np.asarray(adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            f"{dataset_name} adjacency must be square; got {adjacency.shape}"
        )

    return adjacency


def adjacency_to_graph(adjacency):
    """Binarize, explicitly undirect, and remove diagonal self-loops."""
    binary = np.logical_or(adjacency > 0, adjacency.T > 0)
    binary = binary.astype(np.float32)
    np.fill_diagonal(binary, 0)

    graph = nx.from_numpy_array(binary)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    if nx.number_of_selfloops(graph):
        raise RuntimeError("Self-loop removal failed")

    return graph


def build_morse_results(graph, max_dimension, morse_seed):
    """Build the current degree-based Morse function once for one graph."""
    complex_dict = build_clique_complex(
        graph,
        max_dimension=max_dimension,
    )

    # This reproduces the current model's g(v):
    # max_degree - degree(v) + Uniform(0, 0.5).
    vertex_weights = initialize_vertex_weights(
        graph,
        seed=morse_seed,
    )

    # This function also samples epsilon values for higher simplices.
    # Resetting the seed makes the preprocessing reproducible.
    np.random.seed(morse_seed)
    morse_function, flag = construct_discrete_morse_function(
        graph,
        complex_dict,
        vertex_weights,
        max_dimension=max_dimension,
    )

    is_critical = identify_critical_simplices(
        complex_dict,
        morse_function,
        max_dimension=max_dimension,
    )

    gradient_pairs, paired_with = construct_gradient_vector_field(
        complex_dict,
        morse_function,
        flag,
        max_dimension=max_dimension,
    )

    return {
        "complex": complex_dict,
        "vertex_weights": vertex_weights,
        "morse_function": morse_function,
        "Flag": flag,
        "IsCritical": is_critical,
        "paired_with": paired_with,
        "gradient_pairs": gradient_pairs,
    }


def cell_path_to_graph_edges(path):
    """Extract all ordinary graph edges (1-simplices) from a cell path."""
    edges = set()

    for simplex in path:
        simplex = (
            simplex
            if isinstance(simplex, frozenset)
            else frozenset(simplex)
        )

        if len(simplex) == 2:
            u, v = sorted(simplex)
            edges.add((int(u), int(v)))

    return edges


def critical_cells_up_to_edges(results):
    """Return critical vertices and edges, excluding critical 2-faces."""
    return [
        simplex
        if isinstance(simplex, frozenset)
        else frozenset(simplex)
        for simplex, is_critical in results["IsCritical"].items()
        if is_critical and len(simplex) <= 2
    ]


def extract_configured_skeleton(results, mode, enforce_f, eps):
    """Union path edges for one mode/enforce_f configuration."""
    criticals = critical_cells_up_to_edges(results)
    candidate_pairs = len(criticals) * (len(criticals) - 1) // 2
    successful_pairs = 0
    skeleton_edges = set()

    start_time = time.perf_counter()

    for start, end in combinations(criticals, 2):
        path, _ = find_between_criticals(
            start,
            end,
            results,
            mode=mode,
            enforce_f=enforce_f,
            eps=eps,
        )

        if not path:
            continue

        successful_pairs += 1
        skeleton_edges.update(cell_path_to_graph_edges(path))

    runtime = time.perf_counter() - start_time

    return {
        "candidate_pairs": candidate_pairs,
        "successful_pairs": successful_pairs,
        "skeleton_edges": skeleton_edges,
        "runtime": runtime,
    }


def count_critical_cells(results):
    """Count critical simplices by mathematical dimension."""
    counts = defaultdict(int)

    for simplex, is_critical in results["IsCritical"].items():
        if is_critical:
            counts[len(simplex) - 1] += 1

    return counts


def analyze_path_configuration(
    dataset_name,
    graph,
    results,
    mode,
    enforce_f,
    eps,
    morse_seed,
    max_dimension,
):
    """Measure one path-search configuration."""
    extraction = extract_configured_skeleton(
        results,
        mode=mode,
        enforce_f=enforce_f,
        eps=eps,
    )

    original_edges = {
        tuple(sorted((int(u), int(v))))
        for u, v in graph.edges()
        if u != v
    }
    skeleton_edges = extraction["skeleton_edges"]

    unexpected = skeleton_edges - original_edges
    if unexpected:
        raise RuntimeError(
            "Skeleton contains edges absent from the original graph: "
            f"{sorted(unexpected)[:5]}"
        )

    critical_counts = count_critical_cells(results)
    covered_nodes = {
        node for edge in skeleton_edges for node in edge
    }

    node_count = graph.number_of_nodes()
    edge_count = len(original_edges)
    candidate_pairs = extraction["candidate_pairs"]
    successful_pairs = extraction["successful_pairs"]

    return {
        "dataset": dataset_name,
        "mode": mode,
        "enforce_f": enforce_f,
        "eps": eps,
        "morse_seed": morse_seed,
        "max_dimension": max_dimension,
        "original_nodes": node_count,
        "original_edges": edge_count,
        "graph_density": nx.density(graph),
        "critical_nodes": critical_counts[0],
        "critical_edges": critical_counts[1],
        "critical_faces": critical_counts[2],
        "total_critical_cells": sum(critical_counts.values()),
        "gradient_pairs": len(results["gradient_pairs"]),
        "candidate_critical_pairs": candidate_pairs,
        "successful_critical_pairs": successful_pairs,
        "successful_pair_ratio": (
            successful_pairs / candidate_pairs
            if candidate_pairs
            else 0.0
        ),
        "skeleton_edges": len(skeleton_edges),
        "skeleton_edge_ratio": (
            len(skeleton_edges) / edge_count
            if edge_count
            else 0.0
        ),
        "skeleton_covered_nodes": len(covered_nodes),
        "skeleton_node_ratio": (
            len(covered_nodes) / node_count
            if node_count
            else 0.0
        ),
        "skeleton_isolated_nodes": node_count - len(covered_nodes),
        "path_runtime_seconds": extraction["runtime"],
        "status": "ok",
        "error": "",
    }


def make_error_row(
    dataset_name,
    graph,
    mode,
    enforce_f,
    eps,
    morse_seed,
    max_dimension,
    error,
):
    """Record a failed configuration without losing other results."""
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "dataset": dataset_name,
            "mode": mode,
            "enforce_f": enforce_f,
            "eps": eps,
            "morse_seed": morse_seed,
            "max_dimension": max_dimension,
            "original_nodes": graph.number_of_nodes(),
            "original_edges": graph.number_of_edges(),
            "graph_density": nx.density(graph),
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        }
    )
    return row


def save_csv(rows, output_path):
    """Incrementally save results so interrupted runs retain completed work."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    """Print the main rebuttal-relevant statistics."""
    print(
        "\n"
        f"{'dataset':<10} "
        f"{'mode':<9} "
        f"{'enforce':<8} "
        f"{'edges':>7} "
        f"{'crit-pairs':>12} "
        f"{'success':>9} "
        f"{'pair%':>8} "
        f"{'path-edges':>11} "
        f"{'edge%':>8} "
        f"{'time(s)':>9}"
    )
    print("-" * 108)

    for row in rows:
        if row["status"] != "ok":
            print(
                f"{row['dataset']:<10} "
                f"{row['mode']:<9} "
                f"{str(row['enforce_f']):<8} "
                f"{row['original_edges']:>7} "
                f"{'-':>12} {'-':>9} {'-':>8} "
                f"{'-':>11} {'-':>8} {'-':>9}"
            )
            print(f"  Error: {row['error']}")
            continue

        print(
            f"{row['dataset']:<10} "
            f"{row['mode']:<9} "
            f"{str(row['enforce_f']):<8} "
            f"{row['original_edges']:>7d} "
            f"{row['candidate_critical_pairs']:>12d} "
            f"{row['successful_critical_pairs']:>9d} "
            f"{100 * row['successful_pair_ratio']:>7.2f}% "
            f"{row['skeleton_edges']:>11d} "
            f"{100 * row['skeleton_edge_ratio']:>7.2f}% "
            f"{row['path_runtime_seconds']:>9.2f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare pairing/relaxed and enforce_f=True/False on all "
            "configured MG-Diff graphs without running model training."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=sorted(DATASET_CONFIGS),
    )
    parser.add_argument("--morse_seed", type=int, default=42)
    parser.add_argument("--max_dimension", type=int, default=2)
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument(
        "--output",
        default="morse_path_configuration_results.csv",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop immediately when one configuration raises an exception.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    all_rows = []

    for dataset_name in args.datasets:
        adjacency = load_adjacency(dataset_name)
        graph = adjacency_to_graph(adjacency)

        print(f"\n{'=' * 18} {dataset_name} {'=' * 18}")
        print(
            "[Input graph]",
            {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "density": nx.density(graph),
                "self_loops": nx.number_of_selfloops(graph),
            },
        )

        morse_start = time.perf_counter()
        results = build_morse_results(
            graph,
            max_dimension=args.max_dimension,
            morse_seed=args.morse_seed,
        )
        morse_runtime = time.perf_counter() - morse_start

        critical_counts = count_critical_cells(results)
        print(
            "[Morse structure]",
            {
                "critical_nodes": critical_counts[0],
                "critical_edges": critical_counts[1],
                "critical_faces": critical_counts[2],
                "gradient_pairs": len(results["gradient_pairs"]),
                "construction_seconds": morse_runtime,
            },
        )

        for mode, enforce_f in PATH_CONFIGURATIONS:
            print(
                f"\nRunning mode={mode}, "
                f"enforce_f={enforce_f}, eps={args.eps}"
            )

            try:
                row = analyze_path_configuration(
                    dataset_name=dataset_name,
                    graph=graph,
                    results=results,
                    mode=mode,
                    enforce_f=enforce_f,
                    eps=args.eps,
                    morse_seed=args.morse_seed,
                    max_dimension=args.max_dimension,
                )
            except Exception as error:
                if args.fail_fast:
                    raise

                traceback.print_exc()
                row = make_error_row(
                    dataset_name=dataset_name,
                    graph=graph,
                    mode=mode,
                    enforce_f=enforce_f,
                    eps=args.eps,
                    morse_seed=args.morse_seed,
                    max_dimension=args.max_dimension,
                    error=error,
                )

            all_rows.append(row)
            print("[Path configuration]", row)
            save_csv(all_rows, args.output)

    print_summary(all_rows)
    print(f"\nSaved results to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
