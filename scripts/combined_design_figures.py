"""STEP 12 figures for Combined Design + Offline Replay.

Usage: python scripts/combined_design_figures.py
Writes results/pathmnist/combined_design_v1/figures/*.png
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "pathmnist", "combined_design_v1")
FIG_DIR = os.path.join(OUT_DIR, "figures")

CANDIDATES = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
FRONTIER = {"C3", "C4"}
UNSAFE = {"C1", "C5"}


def color_for(c: str) -> str:
    if c in UNSAFE:
        return "tab:red"
    if c in FRONTIER:
        return "tab:green"
    if c == "C2":
        return "tab:gray"
    return "tab:blue"


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    overall = pd.read_csv(os.path.join(OUT_DIR, "offline_replay_summary.csv"))
    static = overall[overall["variant"] == "static"].set_index("candidate")
    drift = overall[overall["variant"] == "drift_aware_norm_gate"].set_index("candidate")
    by_attack = pd.read_csv(os.path.join(OUT_DIR, "offline_replay_by_attack.csv"))
    by_partition = pd.read_csv(os.path.join(OUT_DIR, "offline_replay_by_partition.csv"))
    risk = pd.read_csv(os.path.join(OUT_DIR, "catastrophic_risk.csv")).set_index("candidate")
    ablation = pd.read_csv(os.path.join(OUT_DIR, "ablation_replay.csv"))

    # 1. honest-FPR comparison (static + drift-aware overlay)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CANDIDATES))
    vals_static = [static.loc[c, "honest_fpr"] for c in CANDIDATES]
    colors = [color_for(c) for c in CANDIDATES]
    ax.bar(x, vals_static, color=colors)
    for c in ["C2", "C3", "C4", "C6"]:
        if c in drift.index:
            i = CANDIDATES.index(c)
            ax.scatter([i], [drift.loc[c, "honest_fpr"]], color="black", marker="D", zorder=5,
                       label="drift-aware Norm gate" if c == "C2" else None)
    ax.set_xticks(x)
    ax.set_xticklabels(CANDIDATES)
    ax.set_ylabel("honest false-reject rate")
    ax.set_title("Fig 1: candidate honest FPR (bars=static, diamonds=drift-aware gate)\nred=catastrophic-risk, green=Pareto frontier, gray=redundant w/ C0")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "1_honest_fpr_comparison.png"), dpi=140)
    plt.close(fig)

    # 2. TPR by attack
    attacks = ["large_norm", "directional_poisoning", "full_sign_flip", "sparse_coordinate_attack", "low_norm"]
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.11
    xx = np.arange(len(attacks))
    for i, c in enumerate(CANDIDATES):
        vals = [by_attack[(by_attack.candidate == c) & (by_attack.attack == a)]["malicious_tpr"].values[0] for a in attacks]
        ax.bar(xx + i * width, vals, width, label=c, color=color_for(c))
    ax.set_xticks(xx + width * (len(CANDIDATES) - 1) / 2)
    ax.set_xticklabels(attacks, rotation=20, ha="right")
    ax.set_ylabel("TPR (malicious_active rows)")
    ax.set_title("Fig 2: candidate TPR by attack")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "2_tpr_by_attack.png"), dpi=140)
    plt.close(fig)

    # 3. performance vs Non-IID severity
    partitions = ["iid", "dirichlet_a1.0", "dirichlet_a0.5", "dirichlet_a0.1"]
    sub = by_partition[by_partition["regime"] == "A_condition_specific_oracle"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c in CANDIDATES:
        vals_fpr = [sub[(sub.candidate == c) & (sub.partition == p)]["honest_fpr"].values[0] for p in partitions]
        vals_tpr = [sub[(sub.candidate == c) & (sub.partition == p)]["malicious_tpr"].values[0] for p in partitions]
        axes[0].plot(range(4), vals_fpr, marker="o", label=c, color=color_for(c))
        axes[1].plot(range(4), vals_tpr, marker="o", label=c, color=color_for(c))
    for ax, title in zip(axes, ["honest FPR vs Non-IID severity", "malicious TPR vs Non-IID severity"]):
        ax.set_xticks(range(4))
        ax.set_xticklabels(partitions, rotation=20)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Fig 3: candidate performance vs Non-IID severity (Regime A)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "3_performance_vs_noniid.png"), dpi=140)
    plt.close(fig)

    # 4. catastrophic-risk comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [risk.loc[c, "large_norm_accepted"] for c in CANDIDATES]
    total = risk.loc[CANDIDATES[0], "large_norm_total"]
    colors = [color_for(c) for c in CANDIDATES]
    bars = ax.bar(x, vals, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(CANDIDATES)
    ax.set_ylabel(f"large_norm malicious updates ACCEPTED (/{total})")
    ax.set_title("Fig 4: catastrophic-risk comparison\n(any value > 0 = expected divergence per STEP5 evidence)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "4_catastrophic_risk.png"), dpi=140)
    plt.close(fig)

    # 5. Pareto scatter: honest FPR vs directional TPR
    dir_tpr = {}
    for c in CANDIDATES:
        sub2 = by_attack[(by_attack.candidate == c) & (by_attack.attack.isin(["full_sign_flip", "directional_poisoning"]))]
        dir_tpr[c] = sub2["malicious_tpr"].mean()
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in CANDIDATES:
        marker = "x" if c in UNSAFE else ("*" if c in FRONTIER else "o")
        size = 220 if c in FRONTIER else 140
        ax.scatter(static.loc[c, "honest_fpr"], dir_tpr[c], color=color_for(c), marker=marker, s=size, zorder=5)
        ax.annotate(c, (static.loc[c, "honest_fpr"], dir_tpr[c]), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("honest FPR (lower better)")
    ax.set_ylabel("directional-attack TPR (higher better)")
    ax.set_title("Fig 5: Pareto view — FPR vs directional coverage\n(x = catastrophic-risk/disqualified, * = Pareto frontier, gray=C2 redundant w/ C0)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "5_pareto_fpr_vs_directional.png"), dpi=140)
    plt.close(fig)

    # 6. ablation comparison (frontier candidates + C0 baseline)
    focus = ["C0", "C3", "C4"]
    removed_order = ["none", "norm", "cos", "sign"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    width = 0.25
    xx = np.arange(len(removed_order))
    for i, c in enumerate(focus):
        sub3 = ablation[ablation.candidate == c].set_index("removed_signal").reindex(removed_order)
        axes[0].bar(xx + i * width, sub3["honest_fpr"], width, label=c)
        axes[1].bar(xx + i * width, sub3["malicious_tpr"], width, label=c)
    for ax, title in zip(axes, ["honest FPR", "malicious TPR"]):
        ax.set_xticks(xx + width)
        ax.set_xticklabels(["full", "-Norm", "-Cosine", "-Sign"])
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend()
    fig.suptitle("Fig 6: logical ablation (signal removed), C0 baseline vs. Pareto-frontier C3/C4")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "6_ablation_comparison.png"), dpi=140)
    plt.close(fig)

    print(f"Saved 6 figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
