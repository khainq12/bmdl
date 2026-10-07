"""STEP 8 (analysis_v2): remaining required figures (1-4); figures 5-7
(temporal score plots) came from STEP 4, figure 8 (complementarity) from
STEP 7. Diverged cells are annotated, not hidden by averaging.

Usage: python scripts/analysis_v2_step8_figures.py
Writes results/pathmnist/analysis_v2/figures/{1_accuracy_heatmap,
2_divergence_heatmap, 3_tpr_by_attack, 4_fpr_vs_noniid}*.png
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ANALYSIS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "pathmnist", "analysis_v2")
FIG_DIR = os.path.join(ANALYSIS_DIR, "figures")

PARTITION_ORDER = ["iid", "dirichlet_a1.0", "dirichlet_a0.5", "dirichlet_a0.1"]
ATTACK_ORDER = ["no_attack", "large_norm", "low_norm", "full_sign_flip", "directional_poisoning", "sparse_coordinate_attack"]
DEFENSE_ORDER = ["fedavg", "norm", "cosine", "sign_consensus", "median", "multi_krum"]
DETECTOR_DEFENSES = ["norm", "cosine", "sign_consensus"]


def fig1_accuracy_heatmap(df: pd.DataFrame) -> None:
    """Figure 1: attack x defense accuracy heatmap, one panel per partition, Regime A/not_applicable."""
    sub = df[df["calibration_regime"].isin(["A_condition_specific_oracle", "not_applicable"])]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    for ax, partition in zip(axes, PARTITION_ORDER):
        psub = sub[sub["partition"] == partition]
        mat = psub.pivot_table(index="attack", columns="defense", values="acc_mean").reindex(index=ATTACK_ORDER, columns=DEFENSE_ORDER)
        div = psub.pivot_table(index="attack", columns="defense", values="diverged_fraction").reindex(index=ATTACK_ORDER, columns=DEFENSE_ORDER)
        im = ax.imshow(mat.values, cmap="RdYlGn", vmin=0.0, vmax=0.5, aspect="auto")
        ax.set_xticks(range(len(DEFENSE_ORDER)))
        ax.set_xticklabels(DEFENSE_ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(ATTACK_ORDER)))
        ax.set_yticklabels(ATTACK_ORDER, fontsize=8)
        ax.set_title(partition, fontsize=10)
        for i in range(len(ATTACK_ORDER)):
            for j in range(len(DEFENSE_ORDER)):
                v = mat.values[i, j]
                d = div.values[i, j]
                if pd.notna(v):
                    txt = f"{v:.2f}" + ("\n*DIV*" if pd.notna(d) and d > 0 else "")
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color="black")
    fig.colorbar(im, ax=axes, shrink=0.8, label="final accuracy")
    fig.suptitle("Figure 1: attack x defense final accuracy (Regime A / not_applicable), *DIV*=diverged in >=1 seed", fontsize=11)
    fig.savefig(os.path.join(FIG_DIR, "1_accuracy_heatmap.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig2_divergence_heatmap(df: pd.DataFrame) -> None:
    """Figure 2: attack x defense divergence-rate heatmap, one panel per partition."""
    sub = df[df["calibration_regime"].isin(["A_condition_specific_oracle", "not_applicable"])]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    for ax, partition in zip(axes, PARTITION_ORDER):
        psub = sub[sub["partition"] == partition]
        mat = psub.pivot_table(index="attack", columns="defense", values="diverged_fraction").reindex(index=ATTACK_ORDER, columns=DEFENSE_ORDER).fillna(0.0)
        im = ax.imshow(mat.values, cmap="Reds", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(DEFENSE_ORDER)))
        ax.set_xticklabels(DEFENSE_ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(ATTACK_ORDER)))
        ax.set_yticklabels(ATTACK_ORDER, fontsize=8)
        ax.set_title(partition, fontsize=10)
        for i in range(len(ATTACK_ORDER)):
            for j in range(len(DEFENSE_ORDER)):
                ax.text(j, i, f"{mat.values[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.8, label="fraction of 3 seeds diverged")
    fig.suptitle("Figure 2: attack x defense divergence rate (Regime A / not_applicable)", fontsize=11)
    fig.savefig(os.path.join(FIG_DIR, "2_divergence_heatmap.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig3_tpr_by_attack(df: pd.DataFrame) -> None:
    """Figure 3: TPR by attack, for Norm/Cosine/Sign, Regime A, one panel per partition."""
    sub = df[(df["calibration_regime"] == "A_condition_specific_oracle") & (df["defense"].isin(DETECTOR_DEFENSES)) & (df["attack"] != "no_attack")]
    attacks = [a for a in ATTACK_ORDER if a != "no_attack"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    colors = {"norm": "tab:blue", "cosine": "tab:orange", "sign_consensus": "tab:green"}
    for ax, partition in zip(axes, PARTITION_ORDER):
        psub = sub[sub["partition"] == partition]
        x = np.arange(len(attacks))
        width = 0.25
        for i, defense in enumerate(DETECTOR_DEFENSES):
            vals = [psub[(psub["defense"] == defense) & (psub["attack"] == a)]["tpr_mean"].values for a in attacks]
            vals = [v[0] if len(v) else np.nan for v in vals]
            ax.bar(x + (i - 1) * width, vals, width, label=defense, color=colors[defense])
        ax.set_xticks(x)
        ax.set_xticklabels(attacks, rotation=45, ha="right", fontsize=7)
        ax.set_title(partition, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("TPR")
    axes[0].legend(fontsize=8)
    fig.suptitle("Figure 3: TPR by attack, Norm/Cosine/Sign Consensus (Regime A)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "3_tpr_by_attack.png"), dpi=140)
    plt.close(fig)


def fig4_fpr_vs_noniid(df: pd.DataFrame) -> None:
    """Figure 4: FPR vs Non-IID severity (no_attack condition), Regime A and B, for the 3 detectors."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"norm": "tab:blue", "cosine": "tab:orange", "sign_consensus": "tab:green"}

    ax = axes[0]
    sub = df[(df["calibration_regime"] == "A_condition_specific_oracle") & (df["defense"].isin(DETECTOR_DEFENSES)) & (df["attack"] == "no_attack")]
    for defense in DETECTOR_DEFENSES:
        dsub = sub[sub["defense"] == defense].set_index("partition").reindex(PARTITION_ORDER)
        ax.plot(range(4), dsub["fpr_mean"], marker="o", label=defense, color=colors[defense])
    ax.set_xticks(range(4))
    ax.set_xticklabels(PARTITION_ORDER, rotation=20)
    ax.set_ylabel("FPR (no_attack)")
    ax.set_title("Regime A (condition-specific)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    sub_b = df[(df["calibration_regime"] == "B_iid_transfer_frozen") & (df["defense"].isin(DETECTOR_DEFENSES)) & (df["attack"] == "no_attack")]
    non_iid = ["dirichlet_a1.0", "dirichlet_a0.5", "dirichlet_a0.1"]
    for defense in DETECTOR_DEFENSES:
        dsub = sub_b[sub_b["defense"] == defense].set_index("partition").reindex(non_iid)
        ax2.plot(range(3), dsub["fpr_mean"], marker="s", label=defense, color=colors[defense])
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(non_iid, rotation=20)
    ax2.set_title("Regime B (IID-calibrated, frozen)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("Figure 4: FPR vs Non-IID severity (no_attack), Regime A vs B", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "4_fpr_vs_noniid.png"), dpi=140)
    plt.close(fig)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(ANALYSIS_DIR, "attack_defense_matrix.csv"))
    fig1_accuracy_heatmap(df)
    fig2_divergence_heatmap(df)
    fig3_tpr_by_attack(df)
    fig4_fpr_vs_noniid(df)
    print(f"Saved figures 1-4 to {FIG_DIR}/")


if __name__ == "__main__":
    main()
