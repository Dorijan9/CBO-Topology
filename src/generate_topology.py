"""
Generate ground truth and candidate graph JSON files for each topology.

Candidates are single-edge modifications (same as synthetic experiment):
K = min(2 * n_edges, 25) candidates, each differing by exactly one edge.
"""

import json
import numpy as np
from pathlib import Path
from src.topologies import TOPOLOGIES, VARIABLES, N


def _has_cycle(adj: np.ndarray) -> bool:
    n = adj.shape[0]
    WHITE, GREY, BLACK = 0, 1, 2
    color = [WHITE] * n

    def dfs(u):
        color[u] = GREY
        for v in range(n):
            if adj[u, v]:
                if color[v] == GREY:
                    return True
                if color[v] == WHITE and dfs(v):
                    return True
        color[u] = BLACK
        return False

    return any(color[u] == WHITE and dfs(u) for u in range(n))


def generate_candidates(adj_true: np.ndarray, n_candidates: int = 15,
                        seed: int = 42) -> list:
    """Generate single-edge modification candidates."""
    rng = np.random.default_rng(seed)
    n = adj_true.shape[0]

    candidates = [{
        "id": "G1",
        "confidence": 0.70,
        "description": "Ground truth",
        "adjacency": adj_true.tolist(),
    }]

    confidences = np.linspace(0.60, 0.35, n_candidates - 1)
    attempts = 0
    max_attempts = 3000

    while len(candidates) < n_candidates and attempts < max_attempts:
        attempts += 1
        adj_mod = adj_true.copy()

        op = rng.choice(["remove", "reverse", "add"], p=[0.35, 0.30, 0.35])

        if op == "remove":
            existing = list(zip(*np.where(adj_mod != 0)))
            if not existing:
                continue
            idx = rng.integers(len(existing))
            i, j = existing[idx]
            adj_mod[i, j] = 0

        elif op == "reverse":
            existing = list(zip(*np.where(adj_mod != 0)))
            if not existing:
                continue
            idx = rng.integers(len(existing))
            i, j = existing[idx]
            adj_mod[i, j] = 0
            adj_mod[j, i] = 1

        elif op == "add":
            non_existing = [(i, j) for i in range(n) for j in range(n)
                            if i != j and adj_mod[i, j] == 0 and adj_mod[j, i] == 0]
            if not non_existing:
                continue
            idx = rng.integers(len(non_existing))
            i, j = non_existing[idx]
            adj_mod[i, j] = 1

        if _has_cycle(adj_mod):
            continue
        if np.array_equal(adj_mod, adj_true):
            continue
        if any(np.array_equal(adj_mod, np.array(c["adjacency"])) for c in candidates):
            continue

        ci = len(candidates) - 1
        candidates.append({
            "id": f"G{len(candidates) + 1}",
            "confidence": float(round(confidences[ci], 3)) if ci < len(confidences) else 0.35,
            "description": f"Single-edge {op}",
            "adjacency": adj_mod.tolist(),
        })

    for i, c in enumerate(candidates):
        c["id"] = f"G{i + 1}"

    return candidates


def generate_all(output_dir: str = "data"):
    """Generate JSON files for all topologies."""
    for name, topo in TOPOLOGIES.items():
        n_edges = len(topo["edges"])
        n_candidates = min(2 * n_edges + 4, 25)

        adj_true = np.array(topo["adjacency_matrix"]["matrix"])
        candidates = generate_candidates(adj_true, n_candidates)

        # Ground truth JSON
        out = Path(output_dir) / name
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "ground_truth_dag.json", "w") as f:
            json.dump(topo, f, indent=2)

        # Candidates JSON
        cand_json = {
            "description": f"Candidates for {name} topology",
            "variable_order": VARIABLES,
            "candidates": [{
                "id": c["id"],
                "confidence": c["confidence"],
                "description": c["description"],
                "edges": [],
                "adjacency_matrix": c["adjacency"],
                "rationale": c["description"],
            } for c in candidates],
        }

        with open(out / "candidate_graphs.json", "w") as f:
            json.dump(cand_json, f, indent=2)

        print(f"  {name:10s}: {n_edges} edges, {len(candidates)} candidates")


if __name__ == "__main__":
    generate_all()
