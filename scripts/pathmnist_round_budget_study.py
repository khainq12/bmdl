"""Round-budget selection study: clean FedAvg ONLY, no attacks, no detector
calibration. Locked design: docs/BENCHMARK_PROTOCOL.md Addendum 2026-10-07
(h). Uses the fixed-K local training primitive (Addendum (g)).

Runs out to 25 rounds on IID and Dirichlet(alpha in {1.0,0.5,0.1}), 3
seeds, and records the learning curve so a round budget can be selected
from where it stabilizes — not from attack performance (none is run here).

Usage: python scripts/pathmnist_round_budget_study.py
Writes results/pathmnist/round_budget_study/{curves.csv, plot.png}
"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmark.datasets.pathmnist import _medmnist_split, build_or_load_partitions  # noqa: E402
from benchmark.models import local_training_fixed_steps  # noqa: E402
from benchmark.models.torch_cnn import TorchCNNFlat  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "pathmnist", "round_budget_study")

N_CLIENTS = 5
SEEDS = [42, 43, 44]
PARTITIONS = [("iid", None), ("dirichlet", 1.0), ("dirichlet", 0.5), ("dirichlet", 0.1)]
MAX_ROUNDS = 25
K_STEPS = 50
BATCH_SIZE = 128
LOCAL_LR = 0.01
SERVER_REF_FRACTION = 0.10
DEVICE = os.environ.get("ZKFL_DEVICE", "cuda")


def _tag(partition: str, alpha) -> str:
    return "iid" if partition == "iid" else f"dirichlet_a{alpha}"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.perf_counter()

    print("Loading official PathMNIST splits ...")
    X_train, y_train, info = _medmnist_split("train")
    X_test, y_test, _ = _medmnist_split("test")
    n_classes = len(info["label"])
    print(f"train={len(y_train)} test={len(y_test)} n_classes={n_classes} K={K_STEPS} batch_size={BATCH_SIZE} device={DEVICE}")

    rows: List[Dict[str, Any]] = []

    for partition, alpha in PARTITIONS:
        tag = _tag(partition, alpha)
        for seed in SEEDS:
            rec = build_or_load_partitions(
                len(X_train), y_train, partition, alpha if alpha is not None else 0.5, N_CLIENTS, SERVER_REF_FRACTION, seed
            )
            client_idx = [np.array(c) for c in rec["client_idx"]]
            partition_xy = [(X_train[c], y_train[c]) for c in client_idx]

            model = TorchCNNFlat(n_classes=n_classes, seed=seed, device=DEVICE)
            rng = np.random.default_rng(seed + 7000)

            for r in range(MAX_ROUNDS):
                t0 = time.perf_counter()
                gw = model.get_weights().copy()
                deltas = []
                for Xc, yc in partition_xy:
                    local = TorchCNNFlat(n_classes=n_classes, seed=seed, device=DEVICE)
                    local.set_weights(gw.copy())
                    d = local_training_fixed_steps(local, Xc, yc, K_STEPS, LOCAL_LR, BATCH_SIZE, rng)
                    deltas.append(d)
                update = np.mean(deltas, axis=0)
                model.set_weights(gw + update)
                acc, loss = model.evaluate(X_test, y_test)
                dt = time.perf_counter() - t0
                rows.append({"partition": tag, "seed": seed, "round": r, "accuracy": float(acc), "loss": float(loss), "runtime_s": float(dt)})

            print(f"{tag} seed={seed}: {MAX_ROUNDS} rounds done, final_acc={rows[-1]['accuracy']:.4f}, elapsed={time.perf_counter()-t_start:.0f}s")
            with open(os.path.join(OUT_DIR, "curves.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    # plot
    settings = [_tag(*p) for p in PARTITIONS]
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in settings:
        srows = [r for r in rows if r["partition"] == s]
        rounds = sorted(set(r["round"] for r in srows))
        means = [float(np.mean([r["accuracy"] for r in srows if r["round"] == rr])) for rr in rounds]
        stds = [float(np.std([r["accuracy"] for r in srows if r["round"] == rr])) for rr in rounds]
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        ax.plot(rounds, means, marker="o", markersize=3, label=s)
        ax.fill_between(rounds, means_arr - stds_arr, means_arr + stds_arr, alpha=0.15)
    for cand in (15, 20, 25):
        ax.axvline(x=cand - 1, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("round")
    ax.set_ylabel("test accuracy (mean +/- std over 3 seeds)")
    ax.set_title(f"Clean FedAvg learning curves, fixed K={K_STEPS} steps/round")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "plot.png"), dpi=150)
    plt.close(fig)

    total = time.perf_counter() - t_start
    print(f"DONE. {len(rows)} rows, total time {total:.0f}s ({total/60:.1f} min). Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
