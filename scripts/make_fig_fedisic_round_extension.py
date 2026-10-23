"""Fed-ISIC2019 round-budget re-verification figure, matching the house
style of scripts/report_r12_figures.py (no baked-in titles, DejaVu Sans,
consistent per-method colors). Built from round_extension.csv (3 defenses
x 2 seeds x 40 rounds, no_attack, same data split as main_benchmark)."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV = "/tmp/claude-1000/-home-ubuntu-Desktop/defbaefa-47cc-440c-8594-e8e12515a704/scratchpad/avis_fedisic/round_extension.csv"
OUT = "/tmp/claude-1000/-home-ubuntu-Desktop/defbaefa-47cc-440c-8594-e8e12515a704/scratchpad/avis_fedisic/fig_fedisic_round_extension.png"

plt.rcParams.update({
    "font.size": 11, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "axes.labelcolor": "#222222",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

COLORS = {"fedavg": "#1f77b4", "norm": "#2ca02c", "c4_drift_aware": "#d62728"}
LABELS = {"fedavg": "FedAvg", "norm": "Norm", "c4_drift_aware": "C4-DA v1"}

df = pd.read_csv(CSV)

fig, ax = plt.subplots(figsize=(7.6, 4.6))
for defense in ["fedavg", "norm", "c4_drift_aware"]:
    sub = df[df.defense == defense]
    rounds = sorted(sub["round"].unique())
    means = [sub[sub["round"] == r]["accuracy"].mean() for r in rounds]
    stds = [sub[sub["round"] == r]["accuracy"].std() for r in rounds]
    means_arr, stds_arr = np.array(means), np.array(stds)
    ax.plot(rounds, means_arr, label=LABELS[defense], color=COLORS[defense], linewidth=1.6)
    ax.fill_between(rounds, means_arr - stds_arr, means_arr + stds_arr, color=COLORS[defense], alpha=0.15)

ax.axvline(12, color="#999999", linestyle="--", linewidth=1.0)
ax.text(12.6, 0.30, "N_ROUNDS=12\n(giao thức gốc)", fontsize=8.5, color="#666666", va="bottom")

ax.set_xlabel("Vòng huấn luyện")
ax.set_ylabel("Accuracy trên tập test (TB ± std, 2 seed)")
ax.set_xlim(0, 40)
ax.set_ylim(0.25, 0.52)
ax.legend(fontsize=9.5, loc="lower right", framealpha=0.95)
fig.tight_layout()
fig.savefig(OUT, dpi=220)
plt.close(fig)
print("saved", OUT)
