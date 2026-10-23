"""R12 BASELINE CONVERGENCE STUDY.

Clean FedAvg, no attack, no detector. Determines whether 25 rounds (the
previous protocol's round budget) under-trains the model, and if so, how
many rounds are actually needed. Compares sample-weighted vs unweighted
FedAvg aggregation under identical conditions.

Design:
  - partitions: iid, dirichlet alpha in {1.0, 0.5, 0.1}
  - aggregation: fedavg (unweighted), fedavg_weighted (n_i/n)
  - seeds: 42, 43, 44
  - rounds: up to MAX_ROUNDS (150), full curve recorded every round
  - train metric: fixed random subset of the official train split
    (TRAIN_EVAL_SUBSET images, same subset reused every round within a
    run), evaluated every round -- NOT used for any training decision.
  - validation metric: full official val split (10,004 images, never used
    for training), evaluated every round -- this is the curve the
    convergence criterion in analyze_convergence.py reads.
  - test metric: NOT evaluated per round. Official test split is reserved
    for a single final-round evaluation only (written to the manifest's
    final_test_accuracy/final_test_loss columns), matching the
    train/validate/test discipline requested for R12.

Output: results/rebenchmark_v2/baseline_convergence/
  - run_manifest.csv (one row per run, incl. final test accuracy/loss)
  - round_metrics.csv (one row per run x round: train/val accuracy+loss)

Old results under results/pathmnist/ are untouched.

Usage: python scripts/rebenchmark_v2_baseline_convergence.py
"""
from __future__ import annotations

import csv
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from benchmark.datasets.pathmnist import _medmnist_split, build_or_load_partitions  # noqa: E402
from benchmark.defenses import make_defense  # noqa: E402
from benchmark.models import local_training_fixed_steps  # noqa: E402
from benchmark.models.torch_cnn import TorchCNNFlat  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "rebenchmark_v2", "baseline_convergence")

N_CLIENTS = 5
PARTITIONS = [("iid", None), ("dirichlet", 1.0), ("dirichlet", 0.5), ("dirichlet", 0.1)]
SEEDS = [42, 43, 44]
CANDIDATES = ["fedavg", "fedavg_weighted"]

K_STEPS = 50
BATCH_SIZE = 128
LOCAL_LR = 0.01
SERVER_REF_FRACTION = 0.10
MAX_ROUNDS = int(os.environ.get("ZKFL_MAX_ROUNDS", "150"))
TRAIN_EVAL_SUBSET = 3000
DEVICE = os.environ.get("ZKFL_DEVICE", "cuda")
DIVERGENCE_LOSS_THRESHOLD = 50.0


def _tag(partition: str, alpha) -> str:
    return "iid" if partition == "iid" else f"dirichlet_a{alpha}"


def _model(n_classes: int, seed: int) -> TorchCNNFlat:
    return TorchCNNFlat(n_classes=n_classes, seed=seed, device=DEVICE)


def _train(model, X, y, rng) -> np.ndarray:
    return local_training_fixed_steps(model, X, y, K_STEPS, LOCAL_LR, BATCH_SIZE, rng)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        if not os.path.exists(path):
            open(path, "w", encoding="utf-8").close()
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


def run_one(n_classes, partition_xy, client_sizes, X_train_eval, y_train_eval, X_val, y_val, candidate, seed):
    model = _model(n_classes, seed)
    rng = np.random.default_rng(seed + 20000)
    defense = make_defense(candidate)
    n_clients = len(partition_xy)

    round_rows = []
    diverged_at = None

    for r in range(MAX_ROUNDS):
        t0 = time.perf_counter()
        gw = model.get_weights().copy()

        if diverged_at is not None:
            train_acc, train_loss = round_rows[-1]["train_accuracy"], round_rows[-1]["train_loss"]
            val_acc, val_loss = round_rows[-1]["val_accuracy"], round_rows[-1]["val_loss"]
            round_rows.append({"round": r, "train_accuracy": train_acc, "train_loss": train_loss,
                                "val_accuracy": val_acc, "val_loss": val_loss,
                                "runtime_s": 0.0, "diverged": True})
            continue

        deltas = []
        for cid in range(n_clients):
            local = _model(n_classes, seed)
            local.set_weights(gw.copy())
            Xc, yc = partition_xy[cid]
            d = _train(local, Xc, yc, rng)
            deltas.append(d)

        ctx = {"client_weights": client_sizes} if candidate == "fedavg_weighted" else {}
        result = defense.aggregate(deltas, ctx)
        model.set_weights(gw + result.update)

        train_acc, train_loss = model.evaluate(X_train_eval, y_train_eval)
        val_acc, val_loss = model.evaluate(X_val, y_val)
        dt = time.perf_counter() - t0

        round_rows.append({"round": r, "train_accuracy": float(train_acc), "train_loss": float(train_loss),
                            "val_accuracy": float(val_acc), "val_loss": float(val_loss),
                            "runtime_s": float(dt), "diverged": False})

        if (not np.isfinite(val_loss)) or val_loss > DIVERGENCE_LOSS_THRESHOLD:
            diverged_at = r

    return model, round_rows


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.perf_counter()
    print(f"MAX_ROUNDS={MAX_ROUNDS} K_STEPS={K_STEPS} candidates={CANDIDATES} seeds={SEEDS} device={DEVICE}")

    X_train, y_train, info = _medmnist_split("train")
    X_val, y_val, _ = _medmnist_split("val")
    X_test, y_test, _ = _medmnist_split("test")
    n_classes = len(info["label"])
    print(f"train={len(y_train)} val={len(y_val)} test={len(y_test)} n_classes={n_classes}")

    manifest_path = os.path.join(OUT_DIR, "run_manifest.csv")
    round_path = os.path.join(OUT_DIR, "round_metrics.csv")
    manifest_rows: List[Dict[str, Any]] = []
    all_round_rows: List[Dict[str, Any]] = []
    reuse_ids = set()
    if os.path.exists(manifest_path):
        om = pd.read_csv(manifest_path)
        manifest_rows = om.to_dict("records")
        reuse_ids = set(om["run_id"])
        if os.path.exists(round_path):
            all_round_rows = pd.read_csv(round_path).to_dict("records")
        print(f"Resuming: {len(reuse_ids)} runs already present")

    combos = [(p, a, c, s) for (p, a) in PARTITIONS for c in CANDIDATES for s in SEEDS]
    total = len(combos)
    combo_count = len(reuse_ids)
    print(f"Total planned combos: {total}")

    for partition, alpha, candidate, seed in combos:
        tag = _tag(partition, alpha)
        run_id = f"{candidate}__{tag}__seed{seed}"
        if run_id in reuse_ids:
            continue

        rec = build_or_load_partitions(len(X_train), y_train, partition, alpha if alpha is not None else 0.5,
                                        N_CLIENTS, SERVER_REF_FRACTION, seed)
        client_idx = [np.array(c) for c in rec["client_idx"]]
        partition_xy = [(X_train[c], y_train[c]) for c in client_idx]
        client_sizes = [len(c) for c in client_idx]

        # fixed train-eval subset, drawn deterministically from this run's
        # own seed, disjoint from nothing in particular (it's still part
        # of the official train split, same as all client data) -- used
        # ONLY to report a train-side accuracy/loss curve, never to train.
        eval_rng = np.random.default_rng(seed + 30000)
        eval_idx = eval_rng.choice(len(X_train), size=min(TRAIN_EVAL_SUBSET, len(X_train)), replace=False)
        X_train_eval, y_train_eval = X_train[eval_idx], y_train[eval_idx]

        t0 = time.perf_counter()
        model, round_rows = run_one(n_classes, partition_xy, client_sizes, X_train_eval, y_train_eval,
                                     X_val, y_val, candidate, seed)
        dt = time.perf_counter() - t0
        combo_count += 1

        test_acc, test_loss = model.evaluate(X_test, y_test)
        diverged = any(rr["diverged"] for rr in round_rows)
        last = round_rows[-1]
        manifest_rows.append({
            "run_id": run_id, "candidate": candidate, "seed": seed, "partition": tag,
            "n_rounds": MAX_ROUNDS, "diverged": diverged, "runtime_s": dt,
            "final_train_accuracy": last["train_accuracy"], "final_train_loss": last["train_loss"],
            "final_val_accuracy": last["val_accuracy"], "final_val_loss": last["val_loss"],
            "final_test_accuracy": float(test_acc), "final_test_loss": float(test_loss),
            "client_sizes": ";".join(str(s) for s in client_sizes),
        })
        for rr in round_rows:
            rr2 = dict(rr)
            rr2.update(run_id=run_id, candidate=candidate, seed=seed, partition=tag)
            all_round_rows.append(rr2)

        print(f"[{combo_count:3d}/{total}] {run_id:35s} {dt:7.1f}s "
              f"final_val_acc={last['val_accuracy']:.3f} final_test_acc={test_acc:.3f} diverged={diverged}")
        _write_csv(manifest_path, manifest_rows)
        _write_csv(round_path, all_round_rows)

    total_time = time.perf_counter() - t_start
    print(f"DONE. {combo_count} total runs, elapsed {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
