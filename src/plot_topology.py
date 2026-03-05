"""
Plot topology ablation results.
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_results(results_path: str):
    with open(results_path) as f:
        results = json.load(f)

    # Order by expected difficulty
    order = ["chain", "tree", "fork", "diamond", "collider", "layered", "dense"]
    names = [n for n in order if n in results]

    n_edges = [results[n]["n_edges"] for n in names]
    n_cands = [results[n]["n_candidates"] for n in names]
    correct = [results[n]["correct_rate"] for n in names]
    mean_shd = [results[n]["mean_shd"] for n in names]
    std_shd = [results[n]["std_shd"] for n in names]
    mean_f1 = [results[n]["mean_f1"] for n in names]
    std_f1 = [results[n]["std_f1"] for n in names]
    mean_iters = [results[n]["mean_iterations"] for n in names]
    mean_wrmse = [results[n]["mean_weight_rmse"] for n in names]
    mean_wcov = [results[n]["mean_weight_coverage"] for n in names]
    conv_rate = [results[n]["convergence_rate"] for n in names]

    Path("plots").mkdir(exist_ok=True)

    # Colour scheme by topology family
    colors = {
        "chain": "#3498db", "tree": "#2ecc71", "fork": "#f39c12",
        "diamond": "#9b59b6", "collider": "#e74c3c",
        "layered": "#1abc9c", "dense": "#e67e22",
    }
    cols = [colors.get(n, "#95a5a6") for n in names]

    # ---- Figure 1: Main 2x2 comparison ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CBO Recovery by Graph Topology (n=8 variables)",
                 fontsize=16, fontweight="bold")

    # (a) Correct recovery rate
    ax = axes[0, 0]
    bar_colors = ["#2ecc71" if r >= 0.8 else "#e74c3c" if r < 0.5 else "#f39c12" for r in correct]
    ax.bar(range(len(names)), correct, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Correct Recovery Rate")
    ax.set_ylim(0, 1.1)
    ax.set_title("(a) Recovery Accuracy")
    for i, v in enumerate(correct):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10, fontweight="bold")

    # (b) SHD
    ax = axes[0, 1]
    ax.bar(range(len(names)), mean_shd, yerr=std_shd, color=cols,
           edgecolor="black", linewidth=0.8, capsize=4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("SHD")
    ax.set_title("(b) Structural Hamming Distance")

    # (c) F1
    ax = axes[1, 0]
    ax.bar(range(len(names)), mean_f1, yerr=std_f1, color=cols,
           edgecolor="black", linewidth=0.8, capsize=4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("(c) Edge Recovery F1")
    ax.set_ylim(0, 1.05)

    # (d) Iterations
    ax = axes[1, 1]
    ax.bar(range(len(names)), mean_iters, color=cols,
           edgecolor="black", linewidth=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Mean Iterations")
    ax.set_title("(d) Sample Efficiency")
    for i, v in enumerate(mean_iters):
        ax.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("plots/topology_summary.png", dpi=150, bbox_inches="tight")
    print("Saved plots/topology_summary.png")

    # ---- Figure 2: Edge count vs performance ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Effect of Graph Density on Recovery", fontsize=14, fontweight="bold")

    ax = axes[0]
    for i, n in enumerate(names):
        ax.scatter(n_edges[i], correct[i], color=cols[i], s=120, zorder=3,
                   edgecolor="black", linewidth=0.8)
        ax.annotate(n, (n_edges[i], correct[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)
    ax.set_xlabel("Number of Edges")
    ax.set_ylabel("Correct Recovery Rate")
    ax.set_title("(a) Accuracy vs Density")
    ax.set_ylim(0, 1.1)

    ax = axes[1]
    for i, n in enumerate(names):
        ax.scatter(n_edges[i], mean_iters[i], color=cols[i], s=120, zorder=3,
                   edgecolor="black", linewidth=0.8)
        ax.annotate(n, (n_edges[i], mean_iters[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)
    ax.set_xlabel("Number of Edges")
    ax.set_ylabel("Mean Iterations")
    ax.set_title("(b) Iterations vs Density")

    ax = axes[2]
    for i, n in enumerate(names):
        ax.scatter(n_edges[i], mean_wrmse[i], color=cols[i], s=120, zorder=3,
                   edgecolor="black", linewidth=0.8)
        ax.annotate(n, (n_edges[i], mean_wrmse[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)
    ax.set_xlabel("Number of Edges")
    ax.set_ylabel("Weight RMSE")
    ax.set_title("(c) Weight Error vs Density")

    plt.tight_layout()
    plt.savefig("plots/topology_density.png", dpi=150, bbox_inches="tight")
    print("Saved plots/topology_density.png")

    # ---- Figure 3: Posterior evolution per topology ----
    n_panels = len(names)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle("Posterior Evolution by Topology", fontsize=14, fontweight="bold")

    for idx, name in enumerate(names):
        ax = axes[idx]
        example = results[name].get("example_run", {})
        iters_data = example.get("iterations", [])
        if not iters_data:
            continue

        map_probs = [it["map_prob"] for it in iters_data]
        iter_nums = [it["iteration"] for it in iters_data]

        ax.plot(iter_nums, map_probs, "o-", color=colors.get(name, "#95a5a6"),
                linewidth=2, markersize=5)
        ax.axhline(y=0.90, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Iteration")
        if idx == 0:
            ax.set_ylabel("P(MAP)")
        ax.set_title(f"{name}\n({results[name]['n_edges']}e)")
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("plots/topology_posteriors.png", dpi=150, bbox_inches="tight")
    print("Saved plots/topology_posteriors.png")

    plt.close("all")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.plot_topology <results_json_path>")
        sys.exit(1)
    plot_results(sys.argv[1])
