import networkx as nx
import random
from dataclasses import dataclass
from itertools import combinations
import torch
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)
from collections import defaultdict, deque
import numpy as np
Vertex = Hashable
Edge = Tuple[Vertex, Vertex]


def canon_edge(u, v):
    return (u, v) if u <= v else (v, u)

################
def validate_discrete_morse_on_graph(
    G, node_attr, edge_attr, allow_equal,):
    le = (lambda a, b: a <= b) if allow_equal else (lambda a, b: a < b)
    ge = (lambda a, b: a >= b) if allow_equal else (lambda a, b: a > b)

    report = {
        "missing_node_values": [],
        "missing_edge_values": [],
        "vertex_violations": [],
        "edge_violations": [],
    }

    # collect node values
    node_f = {}
    for v, data in G.nodes(data=True):
        if node_attr not in data:
            report["missing_node_values"].append(v)
        else:
            node_f[v] = float(data[node_attr])

    # collect edge values
    edge_f = {}
    for u, v, data in G.edges(data=True):
        if edge_attr not in data:
            report["missing_edge_values"].append((u, v))
        else:
            a, b = canon_edge(u, v)
            edge_f[(a, b)] = float(data[edge_attr])

    # vertex condition
    for v in G.nodes():
        if v not in node_f:
            continue
        fv = node_f[v]
        bad_edges = []
        for nbr in G.neighbors(v):
            a, b = canon_edge(v, nbr)
            if (a, b) not in edge_f:
                continue
            fe = edge_f[(a, b)]
            if le(fe, fv):
                bad_edges.append((a, b))
        if len(bad_edges) > 1:
            report["vertex_violations"].append(
                {"vertex": v, "f(v)": fv, "incident_edges_with_f(e)<=f(v)": bad_edges}
            )

    # edge condition
    for (a, b), fe in edge_f.items():
        endpoints_ge = []
        if a in node_f and ge(node_f[a], fe):
            endpoints_ge.append(a)
        if b in node_f and ge(node_f[b], fe):
            endpoints_ge.append(b)

        if len(endpoints_ge) > 1:
            report["edge_violations"].append(
                {
                    "edge": (a, b),
                    "f(e)": fe,
                    "endpoints_with_f(v)>=f(e)": endpoints_ge,
                    "f(a)": node_f.get(a),
                    "f(b)": node_f.get(b),
                }
            )

    is_valid = (
        len(report["missing_node_values"]) == 0
        and len(report["missing_edge_values"]) == 0
        and len(report["vertex_violations"]) == 0
        and len(report["edge_violations"]) == 0
    )
    return is_valid, report


def pretty_print_report(is_valid, report):
    print(f"VALID discrete Morse function on graph?  {is_valid}")
    if report["missing_node_values"]:
        print("\nMissing node values:")
        for v in report["missing_node_values"]:
            print(f"  - {v}")
    if report["missing_edge_values"]:
        print("\nMissing edge values:")
        for e in report["missing_edge_values"]:
            print(f"  - {e}")

    if report["vertex_violations"]:
        print("\nVertex violations (Condition V):")
        for item in report["vertex_violations"]:
            v = item["vertex"]
            fv = item["f(v)"]
            edges = item["incident_edges_with_f(e)<=f(v)"]
            print(
                f"  - vertex {v} with f(v)={fv}: "
                f"{len(edges)} incident edges have f(e)<=f(v): {edges}"
            )

    if report["edge_violations"]:
        print("\nEdge violations (Condition E):")
        for item in report["edge_violations"]:
            e = item["edge"]
            fe = item["f(e)"]
            endpoints = item["endpoints_with_f(v)>=f(e)"]
            print(
                f"  - edge {e} with f(e)={fe}: "
                f"both endpoints satisfy f(v)>=f(e): {endpoints}"
            )


###########
# Critical cells finder

@dataclass(frozen=True)
class Face:
    id: Hashable
    boundary_edges: Tuple[Edge, ...]


Cell = Union[Vertex, Edge, Face]
MorseValue = float
CriticalRecord = Tuple[Cell, MorseValue, int]  # (cell, value, dimension)



def make_attr_morse_function(
    G, node_attr, edge_attr,
    face_values: Optional[Dict[Hashable, float]] = None,
):
    def f(cell: Cell) -> MorseValue:
        if isinstance(cell, Face):
            if face_values is None:
                raise KeyError("face_values is None, but f(face) was requested.")
            return float(face_values[cell.id])

        if isinstance(cell, tuple) and len(cell) == 2:
            u, v = canon_edge(cell[0], cell[1])
            return float(G.edges[u, v][edge_attr])

        return float(G.nodes[cell][node_attr])

    return f

def identify_critical_cells_nx(
    G, faces, f,):
    vertices = list(G.nodes)
    edges = [canon_edge(u, v) for u, v in G.edges]
    faces = list(faces)

    # vertex -> incident edges
    incident_edges: Dict[Vertex, List[Edge]] = {v: [] for v in vertices}
    for u, v in edges:
        incident_edges[u].append((u, v))
        incident_edges[v].append((u, v))

    # edge -> upper faces
    upper_faces: Dict[Edge, List[Face]] = {e: [] for e in edges}
    for face in faces:
        for e in face.boundary_edges:
            ce = canon_edge(*e)
            upper_faces.setdefault(ce, []).append(face)

    critical: List[CriticalRecord] = []

    # 0-cells
    for v in vertices:
        Nv = incident_edges.get(v, [])
        if not Nv:
            critical.append((v, f(v), 0))
            continue
        min_edge_val = min(f(e) for e in Nv)
        if f(v) < min_edge_val:
            critical.append((v, f(v), 0))

    # 1-cells
    for e in edges:
        u, v = e
        if f(e) > max(f(u), f(v)):
            ups = upper_faces.get(e, [])
            if not ups:
                critical.append((e, f(e), 1))
            else:
                if f(e) < min(f(face) for face in ups):
                    critical.append((e, f(e), 1))

    critical.sort(key=lambda x: x[1])
    return critical


def identify_critical_cells_from_attrs(
    G, faces: Optional[Iterable[Face]] = None,
    node_attr: str = "f",
    edge_attr: str = "f",
    face_values: Optional[Dict[Hashable, float]] = None,
):
    if faces is None:
        faces = []
    f = make_attr_morse_function(G, node_attr=node_attr, edge_attr=edge_attr, face_values=face_values)
    return identify_critical_cells_nx(G, faces=faces, f=f)


################
# plug-in existing SBM graphs
def build_sbm_with_degree_valid_dmf_greedy_pair(
    G_adj,
    seed= 0,
    node_attr="f",
    edge_attr="f",
    eps=1e-3,
    chord_gap=10.0,
    pair_rate=0.2,
):
    rnd = random.Random(seed)
    G_adj = np.asarray(G_adj).copy()

    # Remove diagonal entries before constructing the graph.
    np.fill_diagonal(G_adj, 0)

    # The Morse implementation operates on an undirected graph.
    G_adj = np.logical_or(
        G_adj > 0,
        G_adj.T > 0,
    ).astype(np.float32)

    G = nx.from_numpy_array(G_adj)
    G.remove_edges_from(nx.selfloop_edges(G))

    G = nx.convert_node_labels_to_integers(
        G,
        first_label=0,
    )

    assert nx.number_of_selfloops(G) == 0

    # node values: degree + tie-breaker
    nodes = list(G.nodes())
    # stable unique rank: just node id order (or shuffle with seed if you want)
    rank = {v: i for i, v in enumerate(sorted(nodes))}

    for v in nodes:
        G.nodes[v][node_attr] = float(G.degree(v) + eps * rank[v])

    # Each node gets at most one paired incident edge
    paired_vertex = set()
    paired_edge = set()

    nodes_desc = sorted(nodes, key=lambda v: G.nodes[v][node_attr], reverse=True)

    for v in nodes_desc:
        if v in paired_vertex:
            continue
        if rnd.random() > pair_rate:
            continue

        fv = G.nodes[v][node_attr]
        # candidate neighbors strictly lower
        cands = [u for u in G.neighbors(v) if G.nodes[u][node_attr] < fv]
        if not cands:
            continue

        # pick the "lowest" neighbor (you can change this heuristic)
        u = min(cands, key=lambda x: G.nodes[x][node_attr])
        e = canon_edge(u, v)
        if e in paired_edge:
            continue

        # Pair v with edge (v,u): set f(e)=f(v)
        G.edges[e][edge_attr] = float(fv)
        paired_vertex.add(v)
        paired_edge.add(e)

    # assign values to all other edges (high)
    for j, (a, b) in enumerate(G.edges()):
        e = canon_edge(a, b)
        if e in paired_edge:
            continue
        fa = G.nodes[a][node_attr]
        fb = G.nodes[b][node_attr]
        G.edges[e][edge_attr] = float(max(fa, fb) + chord_gap + eps * (j + 1))

    return G, paired_vertex, paired_edge

################
# Critical-covering walk
from collections import deque

Cell1D = Union[Vertex, Edge]


def build_cell_adjacency_graph_1d(G):
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v in G.edges():
        e = canon_edge(u, v)
        H.add_node(e)
        H.add_edge(u, e)
        H.add_edge(v, e)
    return H


def bfs_shortest_path_H(H, s, t):
    if s == t:
        return [s]
    q = deque([s])
    parent: Dict[Cell1D, Optional[Cell1D]] = {s: None}

    while q:
        x = q.popleft()
        for y in H.neighbors(x):
            if y in parent:
                continue
            parent[y] = x
            if y == t:
                path = [t]
                cur: Optional[Cell1D] = t
                while parent[cur] is not None:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                return path
            q.append(y)
    return []


def critical_covering_walk_high_to_low(
    G, crit, f,):
    crit_cells: List[Cell1D] = [cell for (cell, _, dim) in crit if dim in (0, 1)]
    if not crit_cells:
        return []

    # choose endpoints
    start = max(crit_cells, key=lambda c: (float(f(c)), str(c)))          # highest
    end   = min(crit_cells, key=lambda c: (float(f(c)), str(c)))          # lowest

    # degenerate: only one critical cell
    if start == end:
        return [start]

    H = build_cell_adjacency_graph_1d(G)

    # visit all other criticals except start/end first
    unvisited: Set[Cell1D] = set(crit_cells)
    unvisited.discard(start)
    unvisited.discard(end)

    walk: List[Cell1D] = [start]
    cur = start

    while unvisited:
        dist = nx.single_source_shortest_path_length(H, cur)

        # pick nearest unvisited critical; tie-break by higher f
        target = min(
            unvisited,
            key=lambda c: (dist.get(c, float("inf")), -float(f(c)))
        )

        seg = bfs_shortest_path_H(H, cur, target)
        if not seg:
            break

        walk.extend(seg[1:])
        cur = target
        unvisited.remove(target)

    if walk[-1] != end:
        seg = bfs_shortest_path_H(H, cur, end)
        if seg:
            walk.extend(seg[1:])
        else:
            walk.append(end)

    return walk

def make_edge_mask(Edge_noncrit, drop_prob, seed, symmetric=True):
    device = Edge_noncrit.device
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    rand = torch.rand(Edge_noncrit.shape, device=device, generator=g)
    mask = (rand > drop_prob).float()

    mask = mask * (Edge_noncrit != 0).float()

    if symmetric:
        mask = torch.triu(mask, diagonal=1)
        mask = mask + mask.t()

    return mask


########################################################################################################
#     New Version of morse function
########################################################################################################


def build_clique_complex(G, max_dimension=2):
    """Build clique complex up to max_dimension from graph G."""
    complex_dict = defaultdict(list)

    # 0-simplices (vertices)
    for v in G.nodes():
        complex_dict[0].append(frozenset([v]))

    # 1-simplices (edges)
    for u, v in G.edges():
        if u == v:
            continue

        complex_dict[1].append(
            frozenset([u, v])
        )

    assert all(
        len(edge) == 2
        for edge in complex_dict[1]
    )

    # # Higher-dimensional cliques
    cliques = list(nx.find_cliques(G))
    # for clique in cliques:
    #     if len(clique) >= 3:
    #         complex_dict[len(clique)-1].append(frozenset(clique))

    for clique in cliques:
      dim = len(clique) - 1
      if 2 <= dim <= max_dimension:
        complex_dict[dim].append(frozenset(clique))
    
    # print('this is the complex dict\n', complex_dict)
    return complex_dict

def initialize_vertex_weights(G, seed=42):
    """g(v) = deg_max - degree(v) + ε (low weights for high-degree vertices)."""
    np.random.seed(seed)
    degrees = dict(G.degree())
    max_degree = max(degrees.values())

    g = {}
    for v in G.nodes():
        epsilon = np.random.uniform(0, 0.5)
        g[v] = max_degree - degrees[v] + epsilon
    return g

def get_faces(simplex, dimension, verbose=False):
    """Get all faces of simplex with specified dimension.

    Parameters
    ----------
    simplex : frozenset
    dimension : int
        Face dimension (e.g., for edges with dim=1, request faces of dim=0).
    verbose : bool
        If True, prints the faces for debugging.
    """
    vertices = list(simplex)
    faces = []
    for face_vertices in combinations(vertices, dimension + 1):
        faces.append(frozenset(face_vertices))

    if verbose:
        print('faces:', faces)
    return faces

def construct_discrete_morse_function(G, complex_dict, g, max_dimension=1):
    """Algorithm 1: Construct discrete Morse function f on all simplices."""
    f = {}
    Flag = {}

    # Initialize Flag for all simplices
    for p in range(max_dimension + 1):
        for simplex in complex_dict[p]:
            Flag[simplex] = 0

    # Assign weights to vertices (0-simplices)
    for simplex in complex_dict[0]:
        v = list(simplex)[0]
        f[simplex] = g[v]

    # Assign weights to higher simplices
    for p in range(1, max_dimension + 1):
        if p not in complex_dict or len(complex_dict[p]) == 0:
            continue

        for alpha in complex_dict[p]:
            Faces = get_faces(alpha, p-1)
            Faces_sorted = sorted(Faces, key=lambda face: f[face], reverse=True)

            gamma_0 = Faces_sorted[0]
            gamma_1 = Faces_sorted[1] if len(Faces_sorted) > 1 else gamma_0

            if Flag[gamma_0] == 0 and f[gamma_0] > f[gamma_1]:
                f[alpha] = (f[gamma_0] + f[gamma_1]) / 2.0
                Flag[gamma_0] = 1
            else:
                epsilon = np.random.uniform(0, 0.5)
                f[alpha] = f[gamma_0] + epsilon
    
    # print("this is f and flag", f, Flag)
    return f, Flag


def identify_critical_simplices(complex_dict, f, max_dimension=2):
    IsCritical = {}

    for p in range(max_dimension + 1):
        if p not in complex_dict:
            continue

        for alpha in complex_dict[p]:
            # U(alpha): cofaces with f(beta) <= f(alpha)
            U_alpha = []
            if p + 1 <= max_dimension and (p + 1) in complex_dict:
                for beta in complex_dict[p + 1]:
                    if alpha.issubset(beta) and f[beta] <= f[alpha]:
                        U_alpha.append(beta)

            # V(alpha): faces with f(gamma) >= f(alpha)
            V_alpha = []
            if p > 0:
                faces = get_faces(alpha, p - 1)
                for gamma in faces:
                    if f[gamma] >= f[alpha]:
                        V_alpha.append(gamma)

            # Critical iff both are empty
            IsCritical[alpha] = (len(U_alpha) == 0 and len(V_alpha) == 0)

    for p in range(max_dimension + 1):
        if p not in complex_dict:
            continue
        count = sum(1 for s in complex_dict[p] if IsCritical.get(s, False))

    return IsCritical

def construct_gradient_vector_field(complex_dict, f, Flag, max_dimension=1):
    """Build explicit gradient pairs from Morse function. FIXED VERSION."""
    gradient_pairs = []
    paired_with = {}


    # Initialize ALL simplices as unpaired FIRST
    for p in range(max_dimension + 1):
        if p not in complex_dict:
            continue
        for simplex in complex_dict[p]:
            paired_with[simplex] = None  # ← CRITICAL: Initialize EVERY simplex

    # Find pairings (higher → lower)
    for p in range(max_dimension + 1):
        if p not in complex_dict or p + 1 not in complex_dict:
            continue

        for tau in complex_dict[p + 1]:
            if paired_with[tau] is not None:  # Now safe - tau is guaranteed to exist
                continue

            faces = get_faces(tau, p)
            eligible_faces = [sigma for sigma in faces
                            if paired_with[sigma] is None and f[tau] <= f[sigma]]

            if eligible_faces:
                sigma = min(eligible_faces, key=lambda s: f[s])
                gradient_pairs.append((sigma, tau))
                paired_with[sigma] = tau
                paired_with[tau] = sigma
    
    # print("gradient pairs:", gradient_pairs)  # enable for debugging
    return gradient_pairs, paired_with

def run_complete_pipeline_with_visuals(G, max_dimension=2, seed=42):
    """Complete pipeline: Morse function → gradient field → visualizations [currently commented out but feel free to uncomment]."""

    # Original pipeline

    complex_dict = build_clique_complex(G, max_dimension)
    for p in range(max_dimension):
        if p in complex_dict:
            print(f" Dimension {p}: {len(complex_dict[p])} simplices")

    g = initialize_vertex_weights(G, seed)
    # print(f" Weight range: [{min(g.values()):.3f}, {max(g.values()):.3f}]")

    f, Flag = construct_discrete_morse_function(G, complex_dict, g, max_dimension)

    IsCritical = identify_critical_simplices(complex_dict, f, max_dimension)

    # Gradient field
    gradient_pairs, paired_with = construct_gradient_vector_field(
        complex_dict, f, Flag, max_dimension
    )

    results = {
        'complex': complex_dict,
        'vertex_weights': g,
        'morse_function': f,
        'Flag': Flag,
        'IsCritical': IsCritical,
        'paired_with': paired_with,
        'gradient_pairs': gradient_pairs
    }

    critical_edges = []
    critical_nodes = []
    for simplex, is_crit in results["IsCritical"].items():
        if not is_crit:
            continue

        if len(simplex) == 2:          # critical edge
            u, v = sorted(simplex)
            critical_edges.append((u, v))

        elif len(simplex) == 1:        # critical node
            u = next(iter(simplex))
            critical_nodes.append(u)

    return results, critical_edges, critical_nodes

def dim(sigma):
        return len(sigma) - 1

def as_fs(sigma):
    # ensure consistent hashing
    return sigma if isinstance(sigma, frozenset) else frozenset(sigma)

def paired_partner(sigma, paired_with):
    """Return the paired partner (as frozenset) if exists."""
    sigma = as_fs(sigma)
    partner = paired_with.get(sigma, None)
    return as_fs(partner) if partner is not None else None

def get_cofaces(alpha, complex_dict, p_plus_1):
    """All (p+1)-simplices that contain alpha."""
    alpha = as_fs(alpha)
    out = []
    for beta in complex_dict.get(p_plus_1, []):
        beta = as_fs(beta)
        if alpha.issubset(beta):
            out.append(beta)
    return out

def gradient_neighbors(
    sigma,
    *,
    complex_dict,
    paired_with,
    f,
    max_dimension=1,
    mode="pairing",
    enforce_f=True,
    eps=0.0,
):
    """
    Gradient-respecting neighbors in the incidence graph.

    mode="pairing" (strict-ish):
      - Up step sigma^p -> beta^{p+1} ONLY if (sigma, beta) is a gradient pair
      - Down step beta^{p+1} -> gamma^p to ANY face gamma != paired_face(beta)
    mode="relaxed":
      - Up step sigma^p -> beta^{p+1} allowed for ANY coface with f(beta) <= f(sigma)+eps
      - Down step same as above
    """
    sigma = as_fs(sigma)
    p = dim(sigma)
    out = []

    # ---------- UP moves (to cofaces) ----------
    if p + 1 <= max_dimension:
        if mode == "pairing":
            partner = paired_partner(sigma, paired_with)
            if partner is not None and dim(partner) == p + 1:
                # optional f check: paired coface should satisfy f(partner) <= f(sigma)
                if (not enforce_f) or (f[partner] <= f[sigma] + eps):
                    out.append(partner)
        elif mode == "relaxed":
            for beta in get_cofaces(sigma, complex_dict, p + 1):
                if (not enforce_f) or (f[beta] <= f[sigma] + eps):
                    out.append(beta)
        else:
            raise ValueError("mode must be 'pairing' or 'relaxed'")

    # ---------- DOWN moves (to faces) ----------
    if p > 0:
        faces = [as_fs(x) for x in get_faces(sigma, p - 1)]

        # If sigma is paired downward with a face, exclude that face (Forman V-path rule)
        paired_face = paired_partner(sigma, paired_with)
        if paired_face is not None and dim(paired_face) == p - 1:
            faces = [g for g in faces if g != paired_face]

        # optional f check: typically want f(face) < f(sigma)
        for gamma in faces:
            if (not enforce_f) or (f[gamma] < f[sigma] - eps):
                out.append(gamma)

    return out

def find_gradient_incidence_path(
    start,
    end,
    *,
    complex_dict,
    paired_with,
    f,
    max_dimension=1,
    mode="pairing",
    enforce_f=True,
    eps=0.0,
    max_steps=10_000,
):
    """
    BFS in the gradient-respecting incidence graph.
    Returns (path, length) or (None, None).
    """
    start, end = as_fs(start), as_fs(end)
    if start == end:
        return [start], 0

    q = deque([(start, [start])])
    seen = {start}
    steps = 0

    while q and steps < max_steps:
        cur, path = q.popleft()
        steps += 1

        for nxt in gradient_neighbors(
            cur,
            complex_dict=complex_dict,
            paired_with=paired_with,
            f=f,
            max_dimension=max_dimension,
            mode=mode,
            enforce_f=enforce_f,
            eps=eps,
        ):
            if nxt in seen:
                continue
            if nxt == end:
                return path + [nxt], len(path)
            seen.add(nxt)
            q.append((nxt, path + [nxt]))

    return None, None

def find_between_criticals(start, end, results, *, mode="pairing", enforce_f=True, eps=0.0):

    f = results["morse_function"]
    complex_dict = results["complex"]
    paired_with = results["paired_with"]
    max_dimension = max(complex_dict.keys()) if complex_dict else 1

    start, end = as_fs(start), as_fs(end)
    ds, de = dim(start), dim(end)

    # If you ask for "ascending" (lower->higher), compute reverse and flip for display.
    if ds < de:
        path, length = find_gradient_incidence_path(
            end, start,
            complex_dict=complex_dict,
            paired_with=paired_with,
            f=f,
            max_dimension=max_dimension,
            mode=mode,
            enforce_f=enforce_f,
            eps=eps,
        )
        if path:
            path = list(reversed(path))
            return path, length
        return None, None

    # otherwise do start->end directly
    return find_gradient_incidence_path(
        start, end,
        complex_dict=complex_dict,
        paired_with=paired_with,
        f=f,
        max_dimension=max_dimension,
        mode=mode,
        enforce_f=enforce_f,
        eps=eps,
    )



def get_critical_path(results):
    """Return the union of graph edges on valid gradient paths.

    Returns
    -------
    list[tuple[int, int]]
        Deduplicated undirected graph edges belonging to at least one
        valid gradient path between critical 0- or 1-cells.
    """

    def cell_path_to_graph_edges(path):
        """Extract every 1-simplex from a cell path."""
        edges = []

        for simplex in path:
            simplex = (
                simplex
                if isinstance(simplex, frozenset)
                else frozenset(simplex)
            )

            if len(simplex) == 2:
                u, v = sorted(simplex)
                edges.append((int(u), int(v)))

        return edges

    def _as_fs(cell):
        return (
            cell
            if isinstance(cell, frozenset)
            else frozenset(cell)
        )

    def _dim(cell):
        return len(cell) - 1

    criticals = [
        _as_fs(simplex)
        for simplex, is_critical
        in results["IsCritical"].items()
        if is_critical
    ]

    # Use a set because the same graph edge may occur in many paths.
    skeleton_edges = set()
    successful_pairs = 0

    have_find_between = (
        "find_between_criticals" in globals()
    )

    for a, b in combinations(criticals, 2):
        # This implementation only constructs a graph skeleton
        # between critical vertices and critical edges.
        if len(a) > 2 or len(b) > 2:
            continue

        if have_find_between:
            path, length = find_between_criticals(
                a,
                b,
                results,
                mode="pairing",
                enforce_f=True,
                eps=0.0,
            )

        else:
            dim_a = _dim(a)
            dim_b = _dim(b)

            if dim_a < dim_b:
                path, length = find_gradient_incidence_path(
                    b,
                    a,
                    complex_dict=results["complex"],
                    paired_with=results["paired_with"],
                    f=results["morse_function"],
                    max_dimension=max(
                        results["complex"].keys()
                    ),
                    mode="pairing",
                    enforce_f=True,
                    eps=0.0,
                )

                if path:
                    path = list(reversed(path))

            else:
                path, length = find_gradient_incidence_path(
                    a,
                    b,
                    complex_dict=results["complex"],
                    paired_with=results["paired_with"],
                    f=results["morse_function"],
                    max_dimension=max(
                        results["complex"].keys()
                    ),
                    mode="pairing",
                    enforce_f=True,
                    eps=0.0,
                )

        if path:
            successful_pairs += 1

            path_edges = cell_path_to_graph_edges(path)
            skeleton_edges.update(path_edges)

    skeleton_edges = sorted(skeleton_edges)

    print(
        "[Morse skeleton]",
        {
            "critical_cells": len(criticals),
            "successful_critical_pairs": successful_pairs,
            "skeleton_edges": len(skeleton_edges),
        },
    )

    return skeleton_edges