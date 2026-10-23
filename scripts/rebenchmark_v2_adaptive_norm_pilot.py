"""R12 ablation pilot: separates "threshold ADAPTATION" from "multi-signal
FUSION" as explanations for C4-DA v1's improvement over static Norm.

Three defenses, same drift-aware _drift_tau code path where applicable:
  - norm               : static tau, calibrated once (the baseline that freezes)
  - adaptive_norm_only : C4-DA's drift-aware norm gate ALONE (no cosine/sign)
  - c4_drift_aware      : the full locked C4-DA v1 (drift-aware norm + cosine + sign)

Grid (deliberately smaller than the main 504-run sweep -- this is a
targeted, controlled pilot, not a replacement for it):
  attacks:     no_attack, large_norm, directional_poisoning (tau-calibrated)
  partitions:  iid, dirichlet alpha=0.5
  seeds:       42, 43, 44
  rounds:      280 (same budget as the main sweep, for direct comparability)
  -> 3 defenses x 3 attacks x 2 partitions x 3 seeds = 54 runs

Reuses run_combo/calibrate_for_condition from rebenchmark_v2_full_benchmark.py
unmodified (imported, not copy-pasted) -- same K=50/batch=128/lr=0.01/
ATTACK_FROM_ROUND=100 protocol as the main sweep. Does NOT modify that
script, its CALIBRATED_DEFENSES set, or any of its output directories.

Output: results/rebenchmark_v2/ablation_adaptive_norm_only/
  (run_manifest.csv, per_round_results.csv, client_decisions.csv,
  calibration_results.csv) -- a new directory, nothing existing touched.

Usage:
  ZKFL_DEVICE=cuda python3 scripts/rebenchmark_v2_adaptive_norm_pilot.py
  ZKFL_SMOKE=1 ZKFL_DEVICE=cuda python3 scripts/rebenchmark_v2_adaptive_norm_pilot.py   # 1 combo, 3 rounds
"""
from __future__ import annotations

import csv
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from benchmark.datasets.pathmnist import _medmnist_split, build_or_load_partitions  # noqa: E402
from scripts.rebenchmark_v2_full_benchmark import (  # noqa: E402
    MALICIOUS_RATIO,
    N_CLIENTS,
    SERVER_REF_FRACTION,
    calibrate_for_condition,
    run_combo,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = os.environ.get("ZKFL_SMOKE", "0") == "1"
OUT_DIR = os.path.join(ROOT, "results", "rebenchmark_v2",
                        "ablation_adaptive_norm_only" + ("_smoke" if SMOKE else ""))

DEFENSES = ["norm", "adaptive_norm_only", "c4_drift_aware"]
ATTACKS = ["no_attack", "large_norm", "directional_poisoning"]
PARTITIONS = [("iid", None), ("dirichlet", 0.5)]
SEEDS = [42] if SMOKE else [42, 43, 44]
CALIB_SEED = 1000


def _tag(partition: str, alpha) -> str:
    return "iid" if partition == "iid" else f"dirichlet_a{alpha}"


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


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.perf_counter()
    print(f"SMOKE={SMOKE} defenses={DEFENSES} attacks={ATTACKS} "
          f"partitions={[_tag(*p) for p in PARTITIONS]} seeds={SEEDS} out={OUT_DIR}")

    X_train, y_train, info = _medmnist_split("train")
    X_val, y_val, _ = _medmnist_split("val")
    X_test, y_test, _ = _medmnist_split("test")
    n_classes = len(info["label"])
    print(f"train={len(y_train)} val={len(y_val)} test={len(y_test)} n_classes={n_classes}")

    manifest_path = os.path.join(OUT_DIR, "run_manifest.csv")
    round_path = os.path.join(OUT_DIR, "per_round_results.csv")
    client_path = os.path.join(OUT_DIR, "client_decisions.csv")
    calib_path = os.path.join(OUT_DIR, "calibration_results.csv")

    manifest_rows: List[Dict[str, Any]] = []
    all_round_rows: List[Dict[str, Any]] = []
    all_client_rows: List[Dict[str, Any]] = []
    calib_rows: List[Dict[str, Any]] = []
    reuse_ids = set()
    if os.path.exists(manifest_path):
        import pandas as pd
        om = pd.read_csv(manifest_path)
        manifest_rows = om.to_dict("records")
        reuse_ids = set(om["run_id"])
        if os.path.exists(round_path):
            all_round_rows = pd.read_csv(round_path).to_dict("records")
        if os.path.exists(client_path):
            all_client_rows = pd.read_csv(client_path).to_dict("records")
        if os.path.exists(calib_path):
            calib_rows = pd.read_csv(calib_path).to_dict("records")
        print(f"Resuming: {len(reuse_ids)} runs already present")

    thresholds: Dict[str, Dict[str, float]] = {}
    for partition, alpha in PARTITIONS:
        tag = _tag(partition, alpha)
        rec = build_or_load_partitions(len(X_train), y_train, partition, alpha if alpha is not None else 0.5,
                                        N_CLIENTS, SERVER_REF_FRACTION, CALIB_SEED)
        server_ref_idx = np.array(rec["server_ref_idx"])
        server_ref_xy = (X_train[server_ref_idx], y_train[server_ref_idx])
        per_partition = {}
        for defense_name, key_name in [("norm", "tau"), ("cosine", "rho"), ("sign_consensus", "kappa")]:
            cache_key = f"{defense_name}__{tag}"
            existing = [c for c in calib_rows if c.get("key") == cache_key]
            if existing:
                result = existing[0]
            else:
                result = calibrate_for_condition(n_classes, defense_name, partition, alpha, X_val, y_val, server_ref_xy)
                result.update(key=cache_key, defense=defense_name, partition=tag)
                calib_rows.append(dict(result))
                print(f"[calib] {defense_name:14s} {tag:16s} threshold={result['threshold']:.4f} "
                      f"val_fpr={result['validation_fpr']:.3f} val_tpr={result['validation_tpr']:.3f}")
            per_partition[key_name] = float(result["threshold"])
        thresholds[tag] = per_partition
    _write_csv(calib_path, calib_rows)
    print(f"Calibration done in {time.perf_counter()-t_start:.1f}s.")

    combos = [(p, a, atk, d, s) for (p, a) in PARTITIONS for atk in ATTACKS for d in DEFENSES for s in SEEDS]
    if SMOKE:
        smoke_n = int(os.environ.get("ZKFL_SMOKE_N", "1"))
        combos = combos[:smoke_n]
    total = len(combos)
    print(f"Total planned combos: {total}")
    n_mal = max(0, int(round(MALICIOUS_RATIO * N_CLIENTS)))
    malicious_client_ids = set(range(n_mal))
    combo_count = len(reuse_ids)

    for partition, alpha, attack, defense_name, seed in combos:
        tag = _tag(partition, alpha)
        run_id = f"{defense_name}__{attack}__{tag}__seed{seed}"
        if run_id in reuse_ids:
            continue

        rec = build_or_load_partitions(len(X_train), y_train, partition, alpha if alpha is not None else 0.5,
                                        N_CLIENTS, SERVER_REF_FRACTION, seed)
        server_ref_idx = np.array(rec["server_ref_idx"])
        client_idx = [np.array(c) for c in rec["client_idx"]]
        server_ref_xy = (X_train[server_ref_idx], y_train[server_ref_idx])
        partition_xy = [(X_train[c], y_train[c]) for c in client_idx]

        threshold_ctx = {"tau": thresholds[tag]["tau"], "rho": thresholds[tag]["rho"], "kappa": thresholds[tag]["kappa"]}

        t0 = time.perf_counter()
        round_rows, client_rows, peak_mem = run_combo(
            n_classes, partition_xy, server_ref_xy, X_test, y_test, attack, defense_name,
            threshold_ctx, seed, malicious_client_ids,
        )
        dt = time.perf_counter() - t0
        combo_count += 1

        diverged = any(rr["diverged"] for rr in round_rows)
        last = round_rows[-1]

        tp = fn = fp = tn = 0
        for cr in client_rows:
            if cr["accepted"] is None:
                continue
            if cr["attack_active"]:
                if cr["accepted"]:
                    fn += 1
                else:
                    tp += 1
            else:
                if cr["accepted"]:
                    tn += 1
                else:
                    fp += 1
        tpr = tp / (tp + fn) if (tp + fn) > 0 else None
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None

        manifest_rows.append({
            "run_id": run_id, "defense": defense_name, "attack": attack, "partition": tag, "seed": seed,
            "diverged": diverged, "runtime_s": dt,
            "final_accuracy": last["accuracy"], "final_loss": last["loss"],
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "tpr": tpr, "fpr": fpr,
            "peak_gpu_mem_bytes": peak_mem, "tau_used": threshold_ctx.get("tau"),
        })
        for rr in round_rows:
            rr2 = dict(rr)
            rr2.update(run_id=run_id, defense=defense_name, attack=attack, partition=tag, seed=seed)
            all_round_rows.append(rr2)
        for cr in client_rows:
            cr2 = dict(cr)
            cr2.update(run_id=run_id, defense=defense_name, attack=attack, partition=tag, seed=seed)
            all_client_rows.append(cr2)

        tpr_s = f"{tpr:.3f}" if tpr is not None else "n/a"
        fpr_s = f"{fpr:.3f}" if fpr is not None else "n/a"
        print(f"[{combo_count:3d}/{total}] {run_id:45s} {dt:7.1f}s acc={last['accuracy']:.3f} "
              f"tpr={tpr_s} fpr={fpr_s} diverged={diverged}")

        _write_csv(manifest_path, manifest_rows)
        _write_csv(round_path, all_round_rows)
        _write_csv(client_path, all_client_rows)

    total_time = time.perf_counter() - t_start
    print(f"DONE. {combo_count}/{total} total runs, elapsed {total_time:.0f}s ({total_time/3600:.2f} h)")


if __name__ == "__main__":
    main()
