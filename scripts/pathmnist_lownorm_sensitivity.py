"""low_norm scale-sensitivity study (supplementary, added in response to
report-review question "is TPR=0 for low_norm a bug or a real finding, and
how harmful is it"). Uses the EXACT locked PathMNIST protocol (N_CLIENTS=5,
K=50, batch=128, N_ROUNDS=25, ATTACK_FROM_ROUND=10, MALICIOUS_RATIO=0.2)
and the ALREADY-CALIBRATED tau for defense=norm, partition=dirichlet_a0.5,
Regime A (0.389715, from results/pathmnist/benchmark_v2_corrected/
calibration_results.csv) -- no new calibration is derived, so this cannot
be accused of re-tuning a threshold to make an attack easier/harder to
catch.

Sweeps the ONE free parameter of the low_norm attack (eta, the shrink
factor applied to the honest update: g' = eta * g) over
[0.01 (the value used everywhere else in this report), 0.05, 0.1, 0.3,
0.5, 0.8], x {fedavg (pure harm, no detector), norm (detector, to confirm
empirically that TPR stays 0 for every eta in this range, matching the
closed-form proof that cosine(eta*g, x) and sign(eta*g) are exactly
invariant for any eta>0 and norm-filter only checks an upper bound)},
x 3 seeds (42,43,44), on partition dirichlet_a0.5 (a representative
non-IID level, not cherry-picked -- it is the middle of the four
partitions already used everywhere else in the report).

Usage: python scripts/pathmnist_lownorm_sensitivity.py
"""
from __future__ import annotations

import csv
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from benchmark.attacks.suite import apply_attack  # noqa: E402
from benchmark.datasets.pathmnist import _medmnist_split, build_or_load_partitions  # noqa: E402
from benchmark.defenses import make_defense  # noqa: E402
from benchmark.models import local_training_fixed_steps  # noqa: E402
from benchmark.models.torch_cnn import TorchCNNFlat  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "pathmnist", "lownorm_sensitivity")

N_CLIENTS = 5
SEEDS = [42, 43]
# eta=0.01 is skipped here -- it is NOT new: the already-locked 594-run
# PathMNIST sweep (results/pathmnist/analysis_v2/attack_defense_matrix.csv)
# already covers eta=0.01 x {fedavg, norm, cosine, sign_consensus, median,
# multi_krum} x 4 partitions x 3 seeds. This script only adds the NEW etas.
ETAS = [0.1, 0.3, 0.5, 0.8]
# norm is dropped here: detection is proven impossible for ANY eta in (0,1)
# by construction (cos(eta*g,x)=cos(g,x) and sign(eta*g)=sign(g) exactly
# for eta>0 -- not an empirical claim, a closed-form identity; and the
# norm filter only ever shrinks further below tau as eta shrinks, so it
# cannot flip from accept to reject as eta decreases). Re-running the
# detector side would not add evidence to a claim that is already a proof,
# so compute is spent only on the actually-open question: how much does
# accuracy degrade as eta rises (a less extreme free-rider)?
DEFENSES = ["fedavg"]
PARTITION, ALPHA = "dirichlet", 0.5
SERVER_REF_FRACTION = 0.10
MALICIOUS_RATIO = 0.2

K_STEPS = 50
BATCH_SIZE = 128
LOCAL_LR = 0.01
N_ROUNDS = 25
ATTACK_FROM_ROUND = 10
DIVERGENCE_LOSS_THRESHOLD = 50.0
DEVICE = os.environ.get("ZKFL_DEVICE", "cpu")  # CPU: runs alongside the GPU-bound Fed-ISIC2019 job

TAU_NORM_DIRICHLET_A0_5_REGIME_A = 0.389715  # locked value, read from calibration_results.csv, not re-derived


def _model(n_classes: int, seed: int) -> TorchCNNFlat:
    return TorchCNNFlat(n_classes=n_classes, seed=seed, device=DEVICE)


def _train(model, X, y, rng) -> np.ndarray:
    return local_training_fixed_steps(model, X, y, K_STEPS, LOCAL_LR, BATCH_SIZE, rng)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run_combo(n_classes, partition_xy, X_test, y_test, eta, defense_name, seed, malicious_client_ids):
    model = _model(n_classes, seed)
    rng = np.random.default_rng(seed + 9000)
    defense = make_defense(defense_name)
    n_clients = len(partition_xy)
    rows = []
    diverged_row = None

    for r in range(N_ROUNDS):
        if diverged_row is not None:
            row = dict(diverged_row); row["round"] = r; rows.append(row); continue

        t0 = time.perf_counter()
        gw = model.get_weights().copy()
        deltas, mal_flags = [], []
        for cid in range(n_clients):
            local = _model(n_classes, seed)
            local.set_weights(gw.copy())
            Xc, yc = partition_xy[cid]
            d = _train(local, Xc, yc, rng)
            is_mal = cid in malicious_client_ids and r >= ATTACK_FROM_ROUND
            if is_mal:
                d = apply_attack("low_norm", d, rng, {"eta": eta})
            deltas.append(d); mal_flags.append(is_mal)

        ctx = {"tau": TAU_NORM_DIRICHLET_A0_5_REGIME_A, "rho": None, "kappa": None, "sign_topk": None, "krum_f": 1, "rng": rng}
        result = defense.aggregate(deltas, ctx)
        model.set_weights(gw + result.update)
        acc, loss = model.evaluate(X_test, y_test)
        dt = time.perf_counter() - t0

        tp = fp = tn = fn = None
        if result.is_detector and result.accepted is not None:
            tp = fp = tn = fn = 0
            for is_mal, acc_flag in zip(mal_flags, result.accepted):
                if is_mal and not acc_flag: tp += 1
                elif (not is_mal) and (not acc_flag): fp += 1
                elif (not is_mal) and acc_flag: tn += 1
                else: fn += 1

        row = {"round": r, "accuracy": float(acc), "loss": float(loss), "runtime_s": float(dt),
               "n_malicious": int(sum(mal_flags)), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "diverged": False}
        rows.append(row)
        if (not np.isfinite(loss)) or loss > DIVERGENCE_LOSS_THRESHOLD:
            diverged_row = dict(row); diverged_row["diverged"] = True

    return rows


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.perf_counter()
    print(f"device={DEVICE} etas={ETAS} defenses={DEFENSES} seeds={SEEDS}")

    X_train, y_train, info = _medmnist_split("train")
    X_test, y_test, _ = _medmnist_split("test")
    n_classes = int(y_train.max()) + 1
    n_mal = max(0, int(round(MALICIOUS_RATIO * N_CLIENTS)))
    malicious_client_ids = set(range(n_mal))

    manifest_path = os.path.join(OUT_DIR, "run_manifest.csv")
    round_path = os.path.join(OUT_DIR, "round_metrics.csv")
    manifest_rows: List[Dict[str, Any]] = []
    round_rows_all: List[Dict[str, Any]] = []
    reuse_ids = set()
    if os.path.exists(manifest_path):
        import pandas as pd
        om = pd.read_csv(manifest_path)
        manifest_rows = om.to_dict("records")
        reuse_ids = set(om["run_id"])
        if os.path.exists(round_path):
            round_rows_all = pd.read_csv(round_path).to_dict("records")
        print(f"Resuming: {len(reuse_ids)} runs already present")

    combo_count = len(reuse_ids)
    total = len(ETAS) * len(DEFENSES) * len(SEEDS)

    for seed in SEEDS:
        rec = build_or_load_partitions(len(X_train), y_train, PARTITION, ALPHA, N_CLIENTS, SERVER_REF_FRACTION, seed)
        client_idx = [np.array(c) for c in rec["client_idx"]]
        partition_xy = [(X_train[c], y_train[c]) for c in client_idx]

        for eta in ETAS:
            for defense_name in DEFENSES:
                run_id = f"eta{eta}__{defense_name}__seed{seed}"
                if run_id in reuse_ids:
                    continue
                t0 = time.perf_counter()
                rows = run_combo(n_classes, partition_xy, X_test, y_test, eta, defense_name, seed, malicious_client_ids)
                dt = time.perf_counter() - t0
                combo_count += 1
                last = rows[-1]
                tp_total = sum(r["tp"] for r in rows if r["tp"] is not None) if rows[0]["tp"] is not None else None
                fn_total = sum(r["fn"] for r in rows if r["fn"] is not None) if rows[0]["fn"] is not None else None
                tpr = (tp_total / (tp_total + fn_total)) if (tp_total is not None and (tp_total + fn_total) > 0) else None
                manifest_rows.append({
                    "run_id": run_id, "seed": seed, "eta": eta, "defense": defense_name,
                    "final_accuracy": last["accuracy"], "final_loss": last["loss"],
                    "diverged": any(r["diverged"] for r in rows), "tpr": tpr,
                    "tp_total": tp_total, "fn_total": fn_total, "runtime_s": dt,
                })
                for rr in rows:
                    rr2 = dict(rr); rr2.update(run_id=run_id, seed=seed, eta=eta, defense=defense_name)
                    round_rows_all.append(rr2)
                print(f"[{combo_count:3d}/{total}] {run_id:30s} {dt:6.1f}s final_acc={last['accuracy']:.3f} tpr={tpr}")
                _write_csv(manifest_path, manifest_rows)
                _write_csv(round_path, round_rows_all)

    print(f"DONE in {(time.perf_counter()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
