"""Build the 5 Chapter-4 figures for the R12 report revision, from verified
data only (results/rebenchmark_v2/{full_benchmark,ablation_adaptive_norm_only,
convergence_extension}/). Professional 2D style: no titles baked in (captions
live in Word), consistent color per method, no gradients/shadows/3D.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FB = os.path.join(ROOT, "results", "rebenchmark_v2", "full_benchmark")
ABL = os.path.join(ROOT, "results", "rebenchmark_v2", "ablation_adaptive_norm_only")
EXT = os.path.join(ROOT, "results", "rebenchmark_v2", "convergence_extension")
OUT = "/tmp/claude-1000/-home-ubuntu-Desktop/defbaefa-47cc-440c-8594-e8e12515a704/scratchpad/report_work/figs"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "axes.labelcolor": "#222222",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

ORDER = ["fedavg", "norm", "cosine", "sign_consensus", "median", "multi_krum", "c4_drift_aware"]
LABEL = {"fedavg": "FedAvg", "norm": "Norm", "cosine": "Cosine", "sign_consensus": "Sign\nConsensus",
         "median": "Median", "multi_krum": "Multi-\nKrum", "c4_drift_aware": "C4-DA v1"}
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
COLOR = {d: COLORS[i] for i, d in enumerate(ORDER)}
DETECTORS = ["norm", "cosine", "sign_consensus", "c4_drift_aware"]


def fig_B_overall_comparison():
    m = pd.read_csv(os.path.join(FB, "run_manifest.csv"))
    g = m.groupby("defense")["final_accuracy"].agg(["mean", "std"]).reindex(ORDER)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(ORDER))
    ax.bar(x, g["mean"], yerr=g["std"], capsize=3, color=[COLOR[d] for d in ORDER], width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[d] for d in ORDER], fontsize=9.5)
    ax.set_ylabel("Accuracy cuối (TB ± std)", fontsize=10.5)
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    fig.tight_layout(pad=1.3)
    fig.savefig(os.path.join(OUT, "fig_B_overall_comparison.png"), dpi=220)
    plt.close(fig)


def fig_C_tpr_fpr_divergence():
    m = pd.read_csv(os.path.join(FB, "run_manifest.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    for d in DETECTORS:
        sub = m[m.defense == d]
        tpr, fpr = sub["tpr"].mean(), sub["fpr"].mean()
        ax.scatter([fpr], [tpr], s=110, color=COLOR[d], label=LABEL[d].replace("\n", " "), zorder=3)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1, zorder=1)
    ax.set_xlabel("FPR trung bình (client lành bị từ chối)")
    ax.set_ylabel("TPR trung bình (client độc hại bị bắt)")
    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)

    ax2 = axes[1]
    rate = m.groupby("defense")["diverged"].mean().reindex(ORDER)
    x = np.arange(len(ORDER))
    ax2.bar(x, rate.values * 100, color=[COLOR[d] for d in ORDER], width=0.62)
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABEL[d] for d in ORDER], fontsize=9.5)
    ax2.set_ylabel("Tỷ lệ phân kỳ số trị (%, trong 72 lượt chạy)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_C_tpr_fpr_divergence.png"), dpi=220)
    plt.close(fig)


def fig_D_ablation():
    abl = pd.read_csv(os.path.join(ABL, "run_manifest.csv"))
    defs = ["norm", "adaptive_norm_only", "c4_drift_aware"]
    dlabel = {"norm": "Ngưỡng\ntĩnh", "adaptive_norm_only": "Ngưỡng\nthích nghi", "c4_drift_aware": "C4-DA\n(đầy đủ)"}
    dcolor = {"norm": "#888888", "adaptive_norm_only": "#1f77b4", "c4_drift_aware": "#d62728"}
    # "no_attack" has no real malicious client, so the "tpr" column there is not a
    # detection rate -- it is the false-rejection rate of honest clients that
    # happen to sit in the role-designated "attacker" index slot (see
    # benchmark/runner.py: n_mal is derived from malicious_ratio regardless of
    # attack.name, and "no_attack" leaves their delta unmodified). Labeling it
    # "TPR" there reads as a detection result and has caused reader confusion;
    # label it explicitly as a false-rejection rate instead, only for this column.
    attacks = ["no_attack", "large_norm", "directional_poisoning"]
    alabel = {
        "no_attack": "Không tấn công\n(tỷ lệ từ chối nhầm)",
        "large_norm": "Large-norm",
        "directional_poisoning": "Directional\npoisoning",
    }
    annot_prefix = {"no_attack": "Từ chối nhầm", "large_norm": "TPR", "directional_poisoning": "TPR"}
    partitions = ["iid", "dirichlet_a0.5"]
    plabel = {"iid": "IID", "dirichlet_a0.5": "Dirichlet α=0,5"}
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), sharey=True)
    for i, partition in enumerate(partitions):
        for j, attack in enumerate(attacks):
            ax = axes[i][j]
            sub = abl[(abl.partition == partition) & (abl.attack == attack)]
            means = [sub[sub.defense == d]["final_accuracy"].mean() for d in defs]
            tprs = [sub[sub.defense == d]["tpr"].mean() for d in defs]
            x = np.arange(len(defs))
            bars = ax.bar(x, means, color=[dcolor[d] for d in defs], width=0.65)
            for b, tpr in zip(bars, tprs):
                if pd.notna(tpr):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.025,
                            f"{annot_prefix[attack]} {tpr:.2f}",
                            ha="center", fontsize=7.5, color="#333333")
            ax.set_xticks(x)
            ax.set_xticklabels([dlabel[d] for d in defs], fontsize=8)
            ax.set_ylim(0, 1.0)
            if j == 0:
                ax.set_ylabel(f"{plabel[partition]}\nAccuracy", fontsize=9.5)
            if i == 0:
                ax.set_xlabel("")
                ax.set_title(alabel[attack], fontsize=9.5, pad=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_D_ablation.png"), dpi=220)
    plt.close(fig)


def fig_E_convergence_500():
    colors = {"fedavg": "#1f77b4", "multi_krum": "#8c564b", "c4_drift_aware": "#d62728"}
    labels = {"fedavg": "FedAvg", "multi_krum": "Multi-Krum", "c4_drift_aware": "C4-DA v1"}
    import glob
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, part, title in zip(axes, ["iid", "dirichlet_a0.5"], ["IID", "Dirichlet α = 0,5"]):
        for defense in ["fedavg", "multi_krum", "c4_drift_aware"]:
            curves = []
            for f in glob.glob(os.path.join(EXT, f"{defense}__{part}__seed*", "round_metrics.csv")):
                g = pd.read_csv(f).sort_values("round")
                curves.append(g["val_accuracy"].to_numpy())
            arr = np.mean(curves, axis=0)
            ax.plot(range(len(arr)), arr, label=labels[defense], color=colors[defense], linewidth=1.4)
        ax.axvline(280, color="#aaaaaa", linestyle="--", linewidth=0.9)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Vòng huấn luyện")
        ax.set_xlim(0, 500)
    axes[0].set_ylabel("Validation accuracy (TB 3 seed)")
    axes[1].legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_E_convergence_500.png"), dpi=220)
    plt.close(fig)


def fig_F_cosine_mechanism():
    clients = pd.read_csv(os.path.join(FB, "client_decisions.csv"))
    calib = pd.read_csv(os.path.join(FB, "calibration_results.csv"))
    rho = {p: calib[(calib.defense == "cosine") & (calib.partition == p)]["threshold"].iloc[0]
           for p in ["iid", "dirichlet_a0.5"]}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    branches = ["norm_fail", "cos_fail", "sign_peer_outlier"]
    blabel = ["Norm\n(thích nghi)", "Cosine\n(tĩnh)", "Sign\n(phụ trợ)"]
    width = 0.35
    x = np.arange(len(branches))
    for k, (part, color, dx) in enumerate([("iid", "#d62728", -width / 2), ("dirichlet_a0.5", "#1f77b4", width / 2)]):
        cd = clients[(clients.defense == "c4_drift_aware") & (clients.attack == "no_attack") &
                     (clients.partition == part) & (clients["round"] >= 230)]
        rates = [cd[b].mean() * 100 for b in branches]
        ax.bar(x + dx, rates, width=width, color=color, label="IID" if part == "iid" else "Dirichlet α=0,5")
    ax.set_xticks(x)
    ax.set_xticklabels(blabel, fontsize=9)
    ax.set_ylabel("Tỷ lệ kích hoạt nhánh từ chối (%)\n(vòng 230-280, no_attack)")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    data, positions, colors2 = [], [], []
    for i, part in enumerate(["iid", "dirichlet_a0.5"]):
        cd = clients[(clients.defense == "c4_drift_aware") & (clients.attack == "no_attack") &
                     (clients.partition == part) & (clients["round"] >= 230)]
        data.append(cd["cosine_score"].dropna().to_numpy())
        positions.append(i)
        colors2.append("#d62728" if part == "iid" else "#1f77b4")
    bp = ax2.boxplot(data, positions=positions, widths=0.5, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors2):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax2.axhline(rho["iid"], color="#d62728", linestyle="--", linewidth=1.1)
    ax2.text(1.32, rho["iid"] + 0.04, f"ρ(IID)={rho['iid']:.2f}", fontsize=8, color="#d62728", va="bottom")
    ax2.axhline(rho["dirichlet_a0.5"], color="#1f77b4", linestyle="--", linewidth=1.1)
    ax2.text(1.32, rho["dirichlet_a0.5"] + 0.04, f"ρ(Dir.)={rho['dirichlet_a0.5']:.2f}", fontsize=8, color="#1f77b4", va="bottom")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["IID", "Dirichlet\nα=0,5"], fontsize=9.5)
    ax2.set_ylabel("cosine_score thực tế (vòng 230-280)")
    ax2.set_xlim(-0.6, 1.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_F_cosine_mechanism.png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    fig_B_overall_comparison()
    fig_C_tpr_fpr_divergence()
    fig_D_ablation()
    fig_E_convergence_500()
    fig_F_cosine_mechanism()
    print("Wrote 5 figures to", OUT)
