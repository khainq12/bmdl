"""Fed-ISIC2019 round-budget selection study: clean FedAvg ONLY, no
attacks, no detector calibration. Locked design: docs/BENCHMARK_PROTOCOL.md
Addendum 2026-10-07 (j) -- mirrors scripts/pathmnist_round_budget_study.py's
methodology exactly (select N_ROUNDS from where the learning curve
stabilizes, not assumed), but K_STEPS is NOT copied from PathMNIST (K=50)
since Fed-ISIC2019's EfficientNet-b0 is ~33x slower per SGD step (measured:
~230ms vs ~7ms). K=10 chosen as the starting point (pretrained ImageNet
features need less local fine-tuning per round than PathMNIST's
from-scratch CNN); LR=0.01 matches both PathMNIST's own choice and
FLamby's own published FedAvg config for this exact model
(flamby/config_isic2019.json). No Dirichlet sweep -- natural centers are
the only condition (docs/BENCHMARK_PROTOCOL.md Sec 1).

Usage: python scripts/fedisic_round_budget_study.py
Writes results/fed_isic2019/round_budget_study/{curves.csv, plot.png}
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

from benchmark.datasets.fed_isic2019 import load_fed_isic2019_problem  # noqa: E402
from benchmark.models.local_training_image import local_training_fixed_steps_image  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "fed_isic2019", "round_budget_study")

SEEDS = [42, 43, 44]
MAX_ROUNDS = 15
K_STEPS = 10
BATCH_SIZE = 32
LOCAL_LR = 0.01


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.perf_counter()
    print(f"K_STEPS={K_STEPS} batch_size={BATCH_SIZE} lr={LOCAL_LR} max_rounds={MAX_ROUNDS} seeds={SEEDS}")

    rows: List[Dict[str, Any]] = []

    for seed in SEEDS:
        problem = load_fed_isic2019_problem(seed=seed)
        client_datasets = problem["client_datasets"]
        test_ds = problem["test"]
        model = problem["model_factory"](pretrained=True, seed=seed)
        rng = np.random.default_rng(seed + 7000)

        for r in range(MAX_ROUNDS):
            t0 = time.perf_counter()
            gw = model.get_weights().copy()
            deltas = []
            for ds, idx in client_datasets:
                # pretrained=False: weights are immediately overwritten by
                # set_weights(gw) below, so loading the pretrained archive
                # here would be pure waste, repeated every client every round.
                local = problem["model_factory"](pretrained=False, seed=seed)
                local.set_weights(gw.copy())
                d = local_training_fixed_steps_image(local, ds, idx, K_STEPS, LOCAL_LR, BATCH_SIZE, rng)
                deltas.append(d)
            update = np.mean(deltas, axis=0)
            model.set_weights(gw + update)
            acc, loss = model.evaluate(test_ds, batch_size=64)
            dt = time.perf_counter() - t0
            rows.append({"seed": seed, "round": r, "accuracy": float(acc), "loss": float(loss), "runtime_s": float(dt)})
            print(f"  seed={seed} round={r:2d} acc={acc:.4f} loss={loss:.4f} dt={dt:.1f}s")

            with open(os.path.join(OUT_DIR, "curves.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

        print(f"seed={seed}: {MAX_ROUNDS} rounds done, final_acc={rows[-1]['accuracy']:.4f}, elapsed={time.perf_counter()-t_start:.0f}s")

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    rounds = sorted(set(r["round"] for r in rows))
    means = [float(np.mean([r["accuracy"] for r in rows if r["round"] == rr])) for rr in rounds]
    stds = [float(np.std([r["accuracy"] for r in rows if r["round"] == rr])) for rr in rounds]
    means_arr, stds_arr = np.array(means), np.array(stds)
    ax.plot(rounds, means, marker="o", markersize=4)
    ax.fill_between(rounds, means_arr - stds_arr, means_arr + stds_arr, alpha=0.2)
    ax.set_xlabel("round")
    ax.set_ylabel("pooled test accuracy (mean +/- std over 3 seeds)")
    ax.set_title(f"Fed-ISIC2019 clean FedAvg learning curve, K={K_STEPS} steps/round")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "plot.png"), dpi=150)
    plt.close(fig)

    total = time.perf_counter() - t_start
    print(f"DONE. {len(rows)} rows, total time {total:.0f}s ({total/60:.1f} min). Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
