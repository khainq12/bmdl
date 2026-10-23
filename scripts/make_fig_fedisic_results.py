"""Fed-ISIC2019 main results figure (accuracy + TPR by defense per attack),
built from the real 62-run main_benchmark data (run_manifest.csv +
client_decisions.csv), matching house style of scripts/report_r12_figures.py.
No fabricated/missing combos: cosine and sign_consensus are excluded here
since they only have directional_poisoning coverage (already in Bảng 4.5),
not a fair 6-attack comparison."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = "/home/ubuntu/Desktop/pq-zkfl-medical"
OUT = "/tmp/claude-1000/-home-ubuntu-Desktop/defbaefa-47cc-440c-8594-e8e12515a704/scratchpad/avis_fedisic/fig_fedisic_results.png"

plt.rcParams.update({
    "font.size": 11, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "axes.labelcolor": "#222222",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

m = pd.read_csv(os.path.join(ROOT, "results/fed_isic2019/main_benchmark/run_manifest.csv"))
c = pd.read_csv(os.path.join(ROOT, "results/fed_isic2019/main_benchmark/client_decisions.csv"))

DEFS = ["fedavg", "norm", "multi_krum", "c4_drift_aware"]
LABEL = {"fedavg": "FedAvg", "norm": "Norm", "multi_krum": "Multi-Krum", "c4_drift_aware": "C4-DA v1"}
COLOR = {"fedavg": "#888888", "norm": "#2ca02c", "multi_krum": "#8c564b", "c4_drift_aware": "#d62728"}
ATTACKS = ["no_attack", "large_norm", "low_norm", "full_sign_flip", "directional_poisoning", "sparse_coordinate_attack"]
ALABEL = {
    "no_attack": "Không\ntấn công", "large_norm": "Large-\nnorm", "low_norm": "Low-\nnorm",
    "full_sign_flip": "Full sign-\nflip", "directional_poisoning": "Directional\npoisoning",
    "sparse_coordinate_attack": "Sparse\ncoordinate",
}
DETECTORS = {"norm", "c4_drift_aware"}

data = {}
for d in DEFS:
    for a in ATTACKS:
        sub_m = m[(m.defense == d) & (m.attack == a)]
        acc = sub_m.final_accuracy.mean() if len(sub_m) else np.nan
        tpr = None
        if d in DETECTORS and a != "no_attack":
            sub_c = c[(c.defense == d) & (c.attack == a)]
            mal = sub_c[sub_c.delta_modified == True]
            if len(mal):
                tpr = (mal.accepted == False).mean()
        data[(d, a)] = (acc, tpr)

fig, ax = plt.subplots(figsize=(11, 5))
n_def = len(DEFS)
width = 0.19
x = np.arange(len(ATTACKS))

for j, d in enumerate(DEFS):
    accs = [data[(d, a)][0] for a in ATTACKS]
    offs = x + (j - (n_def - 1) / 2) * width
    bars = ax.bar(offs, accs, width=width * 0.92, color=COLOR[d], label=LABEL[d])
    for b, a in zip(bars, ATTACKS):
        tpr = data[(d, a)][1]
        if tpr is not None:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{tpr:.2f}",
                     ha="center", fontsize=7, color="#333333", rotation=0)

ax.set_xticks(x)
ax.set_xticklabels([ALABEL[a] for a in ATTACKS], fontsize=9)
ax.set_ylabel("Accuracy cuối (TB theo seed)")
ax.set_ylim(0, 0.62)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
ax.legend(fontsize=9, loc="lower left", ncol=4, framealpha=0.95, bbox_to_anchor=(0.0, -0.32))
ax.text(0.0, 1.02, "Số ghi trên cột (nếu có) = TPR", transform=ax.transAxes, fontsize=8, color="#666666")
fig.tight_layout()
fig.savefig(OUT, dpi=220, bbox_inches="tight")
plt.close(fig)
print("saved", OUT)
