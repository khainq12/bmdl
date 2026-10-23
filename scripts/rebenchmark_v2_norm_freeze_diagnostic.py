"""R12 ablation, task 1: verify the Norm-freezing mechanism with real
per-round numbers (threshold, client delta norms, accepted-client count,
global update norm, accuracy, loss) -- the running avis-1 full-benchmark
job does NOT log raw norm scores for plain `norm` (NormFilter.aggregate
returns scores via `DefenseResult.scores`, but the full-benchmark script's
client-logging only captures `result.extra`, which NormFilter never sets).
This script re-derives those numbers with a fresh, fully-instrumented
single-combo replay -- same protocol constants (K=50, batch=128, lr=0.01,
N_ROUNDS=280, ATTACK_FROM_ROUND=100, same calibration procedure/seed) as
scripts/rebenchmark_v2_full_benchmark.py, so it's a faithful diagnostic of
what that job is doing, not a different experiment.

Runs on DEVICE=cpu by default to guarantee zero GPU contention with the
504-run sweep currently running on avis-1 (also fine to point at the local
GPU via ZKFL_DEVICE=cuda -- that GPU is idle and physically separate from
avis-1, so it would not compete either; cpu is simply the safest default
for a one-off diagnostic this cheap).

Writes results/rebenchmark_v2/norm_freeze_diagnostic/<defense>__<attack>__<partition>__seed<seed>.csv
Does not touch any existing results/ directory.

Usage: python3 scripts/rebenchmark_v2_norm_freeze_diagnostic.py [defense] [attack] [partition_tag] [seed]
  defaults: norm no_attack iid 42
"""
from __future__ import annotations

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from benchmark.attacks.suite import apply_attack  # noqa: E402
from benchmark.datasets.pathmnist import _medmnist_split, build_or_load_partitions  # noqa: E402
from benchmark.defenses import make_defense  # noqa: E402
from benchmark.models import local_training_fixed_steps  # noqa: E402
from benchmark.models.torch_cnn import TorchCNNFlat  # noqa: E402
from scripts.rebenchmark_v2_full_benchmark import (  # noqa: E402
    ATTACK_FROM_ROUND,
    BATCH_SIZE,
    K_STEPS,
    LOCAL_LR,
    MALICIOUS_RATIO,
    N_CLIENTS,
    N_ROUNDS,
    SERVER_REF_FRACTION,
    calibrate_for_condition,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "rebenchmark_v2", "norm_freeze_diagnostic")
DEVICE = os.environ.get("ZKFL_DEVICE", "cpu")


def _tag_to_partition(tag: str):
    if tag == "iid":
        return "iid", None
    alpha = float(tag.replace("dirichlet_a", ""))
    return "dirichlet", alpha


def main() -> None:
    defense_name = sys.argv[1] if len(sys.argv) > 1 else "norm"
    attack_name = sys.argv[2] if len(sys.argv) > 2 else "no_attack"
    partition_tag = sys.argv[3] if len(sys.argv) > 3 else "iid"
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    partition, alpha = _tag_to_partition(partition_tag)

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"defense={defense_name} attack={attack_name} partition={partition_tag} seed={seed} "
          f"N_ROUNDS={N_ROUNDS} ATTACK_FROM_ROUND={ATTACK_FROM_ROUND} device={DEVICE}")

    X_train, y_train, info = _medmnist_split("train")
    X_val, y_val, _ = _medmnist_split("val")
    X_test, y_test, _ = _medmnist_split("test")
    n_classes = len(info["label"])

    def model(seed_):
        return TorchCNNFlat(n_classes=n_classes, seed=seed_, device=DEVICE)

    def train(m, X, y, rng):
        return local_training_fixed_steps(m, X, y, K_STEPS, LOCAL_LR, BATCH_SIZE, rng)

    rec0 = build_or_load_partitions(len(X_train), y_train, partition, alpha if alpha is not None else 0.5,
                                     N_CLIENTS, SERVER_REF_FRACTION, 1000)
    server_ref_idx0 = np.array(rec0["server_ref_idx"])
    server_ref_xy0 = (X_train[server_ref_idx0], y_train[server_ref_idx0])
    calib = calibrate_for_condition(n_classes, "norm", partition, alpha, X_val, y_val, server_ref_xy0)
    static_tau = float(calib["threshold"])
    print(f"calibrated static tau = {static_tau:.6f} (val_fpr={calib['validation_fpr']:.3f}, "
          f"val_tpr={calib['validation_tpr']:.3f})")

    rec = build_or_load_partitions(len(X_train), y_train, partition, alpha if alpha is not None else 0.5,
                                    N_CLIENTS, SERVER_REF_FRACTION, seed)
    client_idx = [np.array(c) for c in rec["client_idx"]]
    partition_xy = [(X_train[c], y_train[c]) for c in client_idx]
    server_ref_idx = np.array(rec["server_ref_idx"])
    server_ref_xy = (X_train[server_ref_idx], y_train[server_ref_idx])

    n_mal = max(0, int(round(MALICIOUS_RATIO * N_CLIENTS)))
    malicious_client_ids = set(range(n_mal))

    defense = make_defense(defense_name)
    m = model(seed)
    rng = np.random.default_rng(seed + 11000)

    rows = []
    t0_all = time.perf_counter()
    for r in range(N_ROUNDS):
        gw = m.get_weights().copy()
        deltas = []
        for cid in range(N_CLIENTS):
            local = model(seed)
            local.set_weights(gw.copy())
            Xc, yc = partition_xy[cid]
            d = train(local, Xc, yc, rng)
            active = (cid in malicious_client_ids) and (r >= ATTACK_FROM_ROUND)
            if active:
                params = {"target_norm": static_tau} if attack_name == "directional_poisoning" else {}
                d = apply_attack(attack_name, d, rng, params)
            deltas.append(d)

        ctx = {"tau": static_tau, "rho": None, "kappa": None}
        if defense_name in ("cosine", "c4_drift_aware"):
            lm = model(seed)
            lm.set_weights(gw.copy())
            ctx["g_ref"] = train(lm, server_ref_xy[0], server_ref_xy[1], rng)

        result = defense.aggregate(deltas, ctx)
        m.set_weights(gw + result.update)

        acc, loss = m.evaluate(X_test, y_test)
        scores = result.scores if result.scores is not None else np.array([np.linalg.norm(d) for d in deltas])
        n_accepted = int(np.sum(result.accepted)) if result.accepted is not None else N_CLIENTS
        threshold_used = float(result.extra["tau_used"][0]) if (result.extra and "tau_used" in result.extra) else static_tau

        row = {
            "round": r, "static_tau": static_tau, "threshold_used": threshold_used,
            "client_norms": ";".join(f"{s:.4f}" for s in scores),
            "n_accepted": n_accepted, "n_clients": N_CLIENTS,
            "global_update_norm": float(np.linalg.norm(result.update)),
            "accuracy": float(acc), "loss": float(loss),
        }
        rows.append(row)
        if r < 5 or r % 10 == 0 or r == N_ROUNDS - 1:
            print(f"round {r:3d} thr={threshold_used:.4f} accepted={n_accepted}/{N_CLIENTS} "
                  f"||update||={row['global_update_norm']:.4f} acc={acc:.3f} loss={loss:.4f} "
                  f"norms=[{row['client_norms']}]")

    total = time.perf_counter() - t0_all
    print(f"DONE in {total:.0f}s")

    out_path = os.path.join(OUT_DIR, f"{defense_name}__{attack_name}__{partition_tag}__seed{seed}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
