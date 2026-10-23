"""R12 scientific audit: publication-quality figures for SCIENTIFIC_AUDIT.md /
ANALYSIS.md / ABLATION_RESULTS.md. Reads only already-computed artifacts
(results/rebenchmark_v2/full_benchmark/{run_manifest,per_round_results}.csv
and results/rebenchmark_v2/ablation_adaptive_norm_only/run_manifest.csv) --
no new training. Matches the report's established figure style (no titles
baked into the image -- captions live in the document; consistent color
per defense across all figures).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FB_DIR = os.path.join(ROOT, "results", "rebenchmark_v2", "full_benchmark")
ABL_DIR = os.path.join(ROOT, "results", "rebenchmark_v2", "ablation_adaptive_norm_only")
FIG_DIR = os.path.join(FB_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DEFENSE_ORDER = ["fedavg", "norm", "cosine", "sign_consensus", "median", "multi_krum", "c4_drift_aware"]
DEFENSE_LABEL = {"fedavg": "FedAvg", "norm": "Norm", "cosine": "Cosine", "sign_consensus": "Sign Consensus",
                  "median": "Median", "multi_krum": "Multi-Krum", "c4_drift_aware": "C4-DA v1"}
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
DEFENSE_COLOR = {d: COLORS[i] for i, d in enumerate(DEFENSE_ORDER)}
PARTITION_ORDER = ["iid", "dirichlet_a1.0", "dirichlet_a0.5", "dirichlet_a0.1"]


def fig1_accuracy_by_defense(manifest: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    incl = manifest.groupby("defense")["final_accuracy"].agg(["mean", "std"]).reindex(DEFENSE_ORDER)
    excl = manifest[~manifest["diverged"]].groupby("defense")["final_accuracy"].mean().reindex(DEFENSE_ORDER)
    x = np.arange(len(DEFENSE_ORDER))
    ax.bar(x - 0.18, incl["mean"], width=0.36, yerr=incl["std"], capsize=3,
           color=[DEFENSE_COLOR[d] for d in DEFENSE_ORDER], alpha=0.9, label="Bao gồm run diverge")
    ax.bar(x + 0.18, excl.values, width=0.36, color=[DEFENSE_COLOR[d] for d in DEFENSE_ORDER], alpha=0.45,
           label="Loại run diverge")
    ax.set_xticks(x)
    ax.set_xticklabels([DEFENSE_LABEL[d] for d in DEFENSE_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Accuracy cuối (trung bình ± std, 72 run/phương pháp)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 0.85)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_accuracy_by_defense.png"), dpi=200)
    plt.close(fig)


def fig2_divergence_rate(manifest: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rate = manifest.groupby("defense")["diverged"].mean().reindex(DEFENSE_ORDER)
    ax.bar(range(len(DEFENSE_ORDER)), rate.values * 100, color=[DEFENSE_COLOR[d] for d in DEFENSE_ORDER])
    ax.set_xticks(range(len(DEFENSE_ORDER)))
    ax.set_xticklabels([DEFENSE_LABEL[d] for d in DEFENSE_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Tỷ lệ run bị diverge (%) trong 72 run")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_divergence_rate_by_defense.png"), dpi=200)
    plt.close(fig)


def fig3_no_attack_curves(rounds: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, partition, title in zip(axes, ["iid", "dirichlet_a0.1"], ["IID", "Dirichlet α=0,1"]):
        sub = rounds[rounds["partition"] == partition]
        for d in DEFENSE_ORDER:
            g = sub[sub["defense"] == d].groupby("round")["accuracy"].mean()
            ax.plot(g.index, g.values, label=DEFENSE_LABEL[d], color=DEFENSE_COLOR[d], linewidth=1.3)
        ax.set_xlabel("Vòng")
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Validation accuracy (no_attack, trung bình 3 seed)")
    axes[1].legend(fontsize=7, loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_no_attack_convergence_curves.png"), dpi=200)
    plt.close(fig)


def fig4_freeze_illustration(rounds: pd.DataFrame, clients: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
    run_id = "norm__no_attack__iid__seed42"
    rr = rounds[rounds["run_id"] == run_id].sort_values("round")
    cc = clients[clients["run_id"] == run_id]
    accepted = cc.groupby("round")["accepted"].sum()
    axes[0].plot(rr["round"], rr["accuracy"], color="firebrick", linewidth=1.3)
    axes[0].set_ylabel("Accuracy")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].plot(accepted.index, accepted.values, color="steelblue", linewidth=1.3)
    axes[1].set_ylabel("Số client được accept / 5")
    axes[1].set_xlabel("Vòng")
    axes[1].set_ylim(-0.3, 5.3)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Norm tĩnh, no_attack, IID, seed 42 — minh họa đóng băng", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_norm_freeze_illustration.png"), dpi=200)
    plt.close(fig)


def fig5_ablation_tpr_fpr() -> None:
    abl = pd.read_csv(os.path.join(ABL_DIR, "run_manifest.csv"))
    defs = ["norm", "adaptive_norm_only", "c4_drift_aware"]
    labels = {"norm": "StaticNorm", "adaptive_norm_only": "AdaptiveNormOnly", "c4_drift_aware": "C4-DA (đầy đủ)"}
    attacks = ["no_attack", "large_norm", "directional_poisoning"]
    partitions = ["iid", "dirichlet_a0.5"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey=True)
    for i, partition in enumerate(partitions):
        for j, attack in enumerate(attacks):
            ax = axes[i][j]
            sub = abl[(abl.partition == partition) & (abl.attack == attack)]
            means = [sub[sub.defense == d]["final_accuracy"].mean() for d in defs]
            tprs = [sub[sub.defense == d]["tpr"].mean() for d in defs]
            x = np.arange(len(defs))
            bars = ax.bar(x, means, color=["#888888", "#1f77b4", "#d62728"])
            for k, (b, tpr) in enumerate(zip(bars, tprs)):
                tpr_s = f"TPR={tpr:.2f}" if pd.notna(tpr) else ""
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, tpr_s, ha="center", fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels([labels[d] for d in defs], rotation=25, ha="right", fontsize=7)
            ax.set_ylim(0, 0.95)
            if j == 0:
                ax.set_ylabel(f"{partition}\nAccuracy", fontsize=8)
            if i == 0:
                ax.set_title(attack, fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_ablation_accuracy_tpr.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    manifest = pd.read_csv(os.path.join(FB_DIR, "run_manifest.csv"))
    rounds = pd.read_csv(os.path.join(FB_DIR, "per_round_results.csv"))
    clients = pd.read_csv(os.path.join(FB_DIR, "client_decisions.csv"))
    fig1_accuracy_by_defense(manifest)
    fig2_divergence_rate(manifest)
    fig3_no_attack_curves(rounds)
    fig4_freeze_illustration(rounds, clients)
    fig5_ablation_tpr_fpr()
    print("Wrote 5 figures to", FIG_DIR)


if __name__ == "__main__":
    main()
