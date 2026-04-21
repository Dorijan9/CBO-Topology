# CBO Topology Ablation Experiment

Tests how graph structure affects CBO recovery, with n=8 variables fixed across all topologies.

## Topologies Tested

| Topology | Edges | K | Structural Motif |
|----------|-------|---|------------------|
| Chain    | 7     | 18 | X0→X1→...→X7 (pure sequential) |
| Tree     | 7     | 18 | Balanced binary tree (depth 3) |
| Fork     | 7     | 18 | X0→{X1..X7} (star/broadcast) |
| Collider | 8     | 20 | {X0,X1,X2,X4,X5}→X3→X6→X7 + X0→X1 (v-structure) |
| Diamond  | 9     | 22 | X0→{X1,X2}→X3→{X4,X5}→X6→X7 |
| Layered  | 12    | 25 | 2 roots → 3 middle → 3 leaves |
| Dense    | 25    | 25 | Near-complete DAG (85% of edges) |

## Running

```bash
python -m src.generate_topology
python -m src.run_topology
python -m src.plot_topology logs/topology_results_<timestamp>.json
```
