"""
Canonical graph topologies for the topology ablation experiment.

All graphs have 8 variables (X0..X7). Edge counts vary by topology
(this is intentional — different structures naturally have different
densities, and normalising edges would distort the topology).

Each topology is defined by:
- A fixed adjacency structure
- Weights drawn to match the structure's causal semantics
- A description of the structural motif being tested
"""

import numpy as np

N = 8
VARIABLES = [f"X{i}" for i in range(N)]


def _build(edges_with_weights: list, description: str, topo_order: list = None) -> dict:
    """Helper to build a topology definition."""
    adj = np.zeros((N, N), dtype=int)
    var_to_idx = {v: i for i, v in enumerate(VARIABLES)}
    edges = []

    for src, tgt, w in edges_with_weights:
        i, j = var_to_idx[src], var_to_idx[tgt]
        adj[i, j] = 1
        edges.append({
            "source": src, "target": tgt,
            "weight": w,
            "sign": "inhibitory" if w < 0 else "excitatory",
        })

    if topo_order is None:
        topo_order = list(VARIABLES)

    return {
        "description": description,
        "variables": {v: {"name": v} for v in VARIABLES},
        "topological_order": topo_order,
        "edges": edges,
        "adjacency_matrix": {"order": VARIABLES, "matrix": adj.tolist()},
        "scm_parameters": {
            "noise_variance": 0.3,
            "weights": {f"w_{e['source']}{e['target']}": e["weight"] for e in edges},
        },
    }


# =============================================================================
# 1. CHAIN: X0→X1→X2→X3→X4→X5→X6→X7
#    Pure sequential structure. 7 edges. Each variable has exactly 1 parent.
#    Prediction: easiest — each intervention isolates one edge.
# =============================================================================
CHAIN = _build(
    [("X0", "X1", 0.70), ("X1", "X2", 0.65), ("X2", "X3", 0.60),
     ("X3", "X4", -0.55), ("X4", "X5", 0.70), ("X5", "X6", 0.50),
     ("X6", "X7", 0.65)],
    "Chain: X0→X1→...→X7 (7 edges, max depth 7)"
)

# =============================================================================
# 2. FORK (Star): X0→{X1,X2,X3,X4,X5,X6,X7}
#    One root, all others are children. 7 edges.
#    Prediction: easy for root intervention, but children are exchangeable —
#    candidates that swap child edges are hard to distinguish.
# =============================================================================
FORK = _build(
    [("X0", "X1", 0.70), ("X0", "X2", -0.55), ("X0", "X3", 0.65),
     ("X0", "X4", 0.45), ("X0", "X5", -0.40), ("X0", "X6", 0.60),
     ("X0", "X7", 0.50)],
    "Fork/Star: X0→{X1..X7} (7 edges, depth 1)"
)

# =============================================================================
# 3. COLLIDER: X0,X1,X2 → X3 ← X4,X5; X3→X6→X7
#    Classic v-structure with 5 parents converging on X3.
#    8 edges. Collider creates explaining-away effects.
#    Prediction: moderate — v-structure identifiable from interventional data
#    but spurious correlations from conditioning on collider confuse candidates.
# =============================================================================
COLLIDER = _build(
    [("X0", "X3", 0.55), ("X1", "X3", 0.60), ("X2", "X3", -0.50),
     ("X4", "X3", 0.45), ("X5", "X3", -0.40),
     ("X3", "X6", 0.70), ("X6", "X7", 0.55),
     ("X0", "X1", 0.35)],  # weak edge between two parents for extra complexity
    "Collider: {X0,X1,X2,X4,X5}→X3→X6→X7 (8 edges, v-structure at X3)",
    topo_order=["X0", "X4", "X5", "X1", "X2", "X3", "X6", "X7"],
)

# =============================================================================
# 4. DIAMOND: Two paths from source to sink.
#    X0→X1→X3, X0→X2→X3, X3→X4→X6, X3→X5→X6, X6→X7
#    9 edges. Multiple colliders (X3, X6). Path interference.
#    Prediction: hard — direct vs indirect effects are confounded.
# =============================================================================
DIAMOND = _build(
    [("X0", "X1", 0.70), ("X0", "X2", 0.60),
     ("X1", "X3", 0.65), ("X2", "X3", -0.50),
     ("X3", "X4", 0.55), ("X3", "X5", -0.45),
     ("X4", "X6", 0.60), ("X5", "X6", 0.50),
     ("X6", "X7", 0.65)],
    "Diamond: X0→{X1,X2}→X3→{X4,X5}→X6→X7 (9 edges, double diamond)"
)

# =============================================================================
# 5. LAYERED: 2 roots → 3 middle → 3 leaves. Dense bipartite connections.
#    Roots: X0,X1. Middle: X2,X3,X4. Leaves: X5,X6,X7.
#    12 edges (2×3 + 3×2 = 12). Realistic for signalling cascades.
#    Prediction: hard — many edges, high connectivity, candidates overlap.
# =============================================================================
LAYERED = _build(
    [# Root → Middle (6 edges)
     ("X0", "X2", 0.60), ("X0", "X3", -0.45), ("X0", "X4", 0.50),
     ("X1", "X2", 0.55), ("X1", "X3", 0.65), ("X1", "X4", -0.40),
     # Middle → Leaf (6 edges)
     ("X2", "X5", 0.55), ("X2", "X6", 0.45),
     ("X3", "X5", -0.50), ("X3", "X7", 0.60),
     ("X4", "X6", 0.50), ("X4", "X7", -0.45)],
    "Layered: {X0,X1}→{X2,X3,X4}→{X5,X6,X7} (12 edges, bipartite)"
)

# =============================================================================
# 6. DENSE: Near-complete DAG. All forward edges i→j where i<j, minus a few.
#    ~24 edges out of 28 possible. Maximally connected.
#    Prediction: hardest — every variable influences most others,
#    removing/adding one edge barely changes the data distribution.
# =============================================================================
_dense_edges = []
_rng = np.random.default_rng(123)
for i in range(N):
    for j in range(i + 1, N):
        if _rng.random() < 0.85:  # 85% of possible edges
            w = _rng.choice([-1, 1]) * _rng.uniform(0.25, 0.55)
            _dense_edges.append((f"X{i}", f"X{j}", round(w, 3)))

DENSE = _build(
    _dense_edges,
    f"Dense: near-complete DAG ({len(_dense_edges)} edges out of {N*(N-1)//2} possible)"
)

# =============================================================================
# 7. TREE: Balanced binary tree. X0→{X1,X2}, X1→{X3,X4}, X2→{X5,X6}, X3→X7
#    7 edges. Each non-leaf has exactly 2 children. No colliders.
#    Prediction: easy — clean hierarchical structure, no confounding paths.
# =============================================================================
TREE = _build(
    [("X0", "X1", 0.70), ("X0", "X2", 0.65),
     ("X1", "X3", 0.60), ("X1", "X4", -0.50),
     ("X2", "X5", 0.55), ("X2", "X6", 0.60),
     ("X3", "X7", 0.50)],
    "Tree: balanced binary tree (7 edges, depth 3)"
)

# =============================================================================
# Registry
# =============================================================================
TOPOLOGIES = {
    "chain": CHAIN,
    "fork": FORK,
    "collider": COLLIDER,
    "diamond": DIAMOND,
    "layered": LAYERED,
    "dense": DENSE,
    "tree": TREE,
}


def get_topology(name: str) -> dict:
    return TOPOLOGIES[name]


def list_topologies() -> list:
    return list(TOPOLOGIES.keys())


if __name__ == "__main__":
    for name, topo in TOPOLOGIES.items():
        n_edges = len(topo["edges"])
        print(f"{name:10s}: {n_edges:>2} edges — {topo['description']}")
