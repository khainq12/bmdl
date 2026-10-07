"""Combined E2E minimal benchmark — the FIRST real end-to-end FL validation
of Combined Gradient Attestation (C3Static, C4Static, C4DriftAware),
promoted from offline replay (results/pathmnist/combined_design_v1/).

Same corrected protocol as scripts/pathmnist_benchmark_v2.py: K=50 SGD
steps/client/round, 25 rounds (ZKFL_N_ROUNDS/ZKFL_ATTACK_FROM_ROUND env
vars, locked via results/pathmnist/round_budget_study/SELECTION.md), 5
clients, MALICIOUS_RATIO=0.2, SERVER_REF_FRACTION=0.10, same divergence
safeguard. Only the defense changes. tau/rho/kappa reused from the existing
locked Regime-A calibration (results/pathmnist/benchmark_v2_corrected/
calibration_results.csv) -- NOT recalibrated for Combined.

Usage:
  python scripts/combined_e2e_benchmark.py --pilot   # 24 pilot runs
  python scripts/combined_e2e_benchmark.py            # full 144 runs
"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from benchmark.attacks.suite import apply_attack  # noqa: E402
from benchmark.datasets.pathmnist import _medmnist_split, build_or_load_partitions  # noqa: E402
from benchmark.defenses import make_defense  # noqa: E402
from benchmark.models import local_training_fixed_steps  # noqa: E402
from benchmark.models.torch_cnn import TorchCNNFlat  # noqa: E402

PILOT = "--pilot" in sys.argv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "pathmnist", "combined_e2e_minimal")
CALIB_CSV = os.path.join(ROOT, "results", "pathmnist", "benchmark_v2_corrected", "calibration_results.csv")

N_CLIENTS = 5
CONDITIONS = ["no_attack", "large_norm", "directional_poisoning", "sparse_coordinate_attack"]
CANDIDATES = ["c3_static", "c4_static", "c4_drift_aware"]

if PILOT:
    PARTITIONS = [("iid", None), ("dirichlet", 0.1)]
    SEEDS = [42]
else:
    PARTITIONS = [("iid", None), ("dirichlet", 1.0), ("dirichlet", 0.5), ("dirichlet", 0.1)]
    SEEDS = [42, 43, 44]

K_STEPS = 50
BATCH_SIZE = 128
LOCAL_LR = 0.01
N_ROUNDS = int(os.environ.get("ZKFL_N_ROUNDS", "0"))
ATTACK_FROM_ROUND = int(os.environ.get("ZKFL_ATTACK_FROM_ROUND", "0"))
MALICIOUS_RATIO = 0.2
SERVER_REF_FRACTION = 0.10
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


def load_thresholds():
    calib = pd.read_csv(CALIB_CSV)
    calib_A = calib[calib["calibration_regime"] == "A_condition_specific_oracle"]

    def th(defense, partition):
        row = calib_A[(calib_A["defense"] == defense) & (calib_A["partition"] == partition)]
        assert len(row) == 1, (defense, partition)
        return float(row.iloc[0]["threshold"])

    out = {}
    for partition, alpha in ([("iid", None), ("dirichlet", 1.0), ("dirichlet", 0.5), ("dirichlet", 0.1)]):
        tag = _tag(partition, alpha)
        out[tag] = {
            "tau": th("norm", tag),
            "rho": th("cosine", tag),
            "kappa": th("sign_consensus", tag),
        }
    return out


def run_one(n_classes, partition_xy, server_ref_xy, X_test, y_test, attack_name, candidate, thresh, seed, malicious_client_ids):
    model = _model(n_classes, seed)
    rng = np.random.default_rng(seed + 8000)
    defense = make_defense(candidate)
    n_clients = len(partition_xy)

    round_rows, client_rows, adaptive_rows = [], [], []
    diverged_at: Optional[int] = None
    last_state: Optional[Dict[str, Any]] = None

    for r in range(N_ROUNDS):
        if diverged_at is not None:
            rr = dict(last_state["round_row"])
            rr["round"] = r
            rr["diverged"] = True
            round_rows.append(rr)
            for crow in last_state["client_rows"]:
                crow2 = dict(crow)
                crow2["round"] = r
                client_rows.append(crow2)
            if candidate == "c4_drift_aware":
                arow = dict(last_state["adaptive_row"])
                arow["round"] = r
                adaptive_rows.append(arow)
            continue

        t0 = time.perf_counter()
        gw = model.get_weights().copy()
        deltas, mal_role, attack_active_flags, delta_modified_flags = [], [], [], []
        for cid in range(n_clients):
            local = _model(n_classes, seed)
            local.set_weights(gw.copy())
            Xc, yc = partition_xy[cid]
            d = _train(local, Xc, yc, rng)
            role = cid in malicious_client_ids
            active = role and r >= ATTACK_FROM_ROUND
            modified = active and attack_name != "no_attack"
            if active:
                d = apply_attack(attack_name, d, rng, {})
            deltas.append(d)
            mal_role.append(role)
            attack_active_flags.append(active)
            delta_modified_flags.append(modified)

        m = _model(n_classes, seed)
        m.set_weights(gw.copy())
        g_ref = _train(m, server_ref_xy[0], server_ref_xy[1], rng)

        ctx = {"tau": thresh["tau"], "rho": thresh["rho"], "kappa": thresh["kappa"], "g_ref": g_ref}
        result = defense.aggregate(deltas, ctx)
        model.set_weights(gw + result.update)

        acc, loss = model.evaluate(X_test, y_test)
        dt = time.perf_counter() - t0

        n_accepted = int(np.sum(result.accepted))
        round_row = {
            "round": r,
            "accuracy": float(acc),
            "loss": float(loss),
            "runtime_s": float(dt),
            "n_accepted": n_accepted,
            "n_rejected": n_clients - n_accepted,
            "n_malicious_role": int(sum(mal_role)),
            "n_attack_active": int(sum(attack_active_flags)),
            "diverged": False,
        }
        round_rows.append(round_row)

        crows_this_round = []
        for cid in range(n_clients):
            crow = {
                "round": r,
                "client_id": cid,
                "malicious_role": bool(mal_role[cid]),
                "attack_active": bool(attack_active_flags[cid]),
                "delta_modified": bool(delta_modified_flags[cid]),
                "accepted": bool(result.accepted[cid]),
            }
            if result.extra is not None:
                for k, v in result.extra.items():
                    crow[k] = float(v[cid]) if hasattr(v[cid], "item") else v[cid]
            client_rows.append(crow)
            crows_this_round.append(crow)

        adaptive_row = None
        if candidate == "c4_drift_aware" and result.extra is not None:
            adaptive_row = {
                "round": r,
                "median_norm": float(result.extra["median_norm"][0]),
                "mad_norm": float(result.extra["mad_norm"][0]),
                "tau_drift": float(result.extra["tau_used"][0]),
                "tau_static_fallback": float(result.extra["tau_static_fallback"][0]),
                "fallback_triggered": bool(result.extra["mad_norm"][0] == 0.0),
            }
            adaptive_rows.append(adaptive_row)

        last_state = {"round_row": round_row, "client_rows": crows_this_round, "adaptive_row": adaptive_row or {}}

        if (not np.isfinite(loss)) or loss > DIVERGENCE_LOSS_THRESHOLD:
            diverged_at = r

    return round_rows, client_rows, adaptive_rows


def main() -> None:
    if N_ROUNDS <= 0:
        raise SystemExit("N_ROUNDS not set. Pass ZKFL_N_ROUNDS=25 ZKFL_ATTACK_FROM_ROUND=10 in the environment.")

    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.perf_counter()
    print(f"PILOT={PILOT} N_ROUNDS={N_ROUNDS} ATTACK_FROM_ROUND={ATTACK_FROM_ROUND}")
    print(f"partitions={[_tag(*p) for p in PARTITIONS]} seeds={SEEDS} conditions={CONDITIONS} candidates={CANDIDATES}")

    thresholds = load_thresholds()

    # reuse valid prior runs (e.g. the 24-run pilot) by run_id -- do not rerun
    # them, per the prompt's explicit instruction. Only applies when no
    # code/threshold/rule change occurred since those rows were written.
    reuse_ids = set()
    manifest_rows: List[Dict[str, Any]] = []
    all_round_rows: List[Dict[str, Any]] = []
    all_client_rows: List[Dict[str, Any]] = []
    all_adaptive_rows: List[Dict[str, Any]] = []
    manifest_path = os.path.join(OUT_DIR, "run_manifest.csv")
    if (not PILOT) and os.path.exists(manifest_path) and os.environ.get("ZKFL_REUSE_PILOT", "1") == "1":
        prior_manifest = pd.read_csv(manifest_path)
        base_cols = ["run_id", "seed", "partition", "attack", "candidate", "n_rounds", "diverged", "runtime_s", "final_accuracy", "final_loss"]
        manifest_rows = prior_manifest[[c for c in base_cols if c in prior_manifest.columns]].to_dict("records")
        reuse_ids = set(prior_manifest["run_id"])
        for fname, store in [("round_metrics.csv", all_round_rows), ("client_decisions.csv", all_client_rows), ("adaptive_threshold_trace.csv", all_adaptive_rows)]:
            p = os.path.join(OUT_DIR, fname)
            if os.path.exists(p):
                store.extend(pd.read_csv(p).to_dict("records"))
        print(f"Reusing {len(reuse_ids)} prior run(s) from {manifest_path} (ZKFL_REUSE_PILOT=1): {sorted(reuse_ids)}")

    print("Loading official PathMNIST splits ...")
    X_train, y_train, info = _medmnist_split("train")
    X_test, y_test, _ = _medmnist_split("test")
    n_classes = len(info["label"])
    print(f"train={len(y_train)} test={len(y_test)} n_classes={n_classes} device={DEVICE}")

    n_mal = max(0, int(round(MALICIOUS_RATIO * N_CLIENTS)))
    malicious_client_ids = set(range(n_mal))

    combo_count = len(reuse_ids)
    total_combos = len(PARTITIONS) * len(CONDITIONS) * len(SEEDS) * len(CANDIDATES)

    for partition, alpha in PARTITIONS:
        tag = _tag(partition, alpha)
        thresh = thresholds[tag]
        for seed in SEEDS:
            rec = build_or_load_partitions(len(X_train), y_train, partition, alpha if alpha is not None else 0.5, N_CLIENTS, SERVER_REF_FRACTION, seed)
            server_ref_idx = np.array(rec["server_ref_idx"])
            client_idx = [np.array(c) for c in rec["client_idx"]]
            server_ref_xy = (X_train[server_ref_idx], y_train[server_ref_idx])
            partition_xy = [(X_train[c], y_train[c]) for c in client_idx]

            for attack in CONDITIONS:
                for candidate in CANDIDATES:
                    run_id = f"{candidate}__{tag}__{attack}__seed{seed}"
                    if run_id in reuse_ids:
                        print(f"[reused] {run_id}")
                        continue
                    t0 = time.perf_counter()
                    round_rows, client_rows, adaptive_rows = run_one(
                        n_classes, partition_xy, server_ref_xy, X_test, y_test, attack, candidate, thresh, seed, malicious_client_ids
                    )
                    dt = time.perf_counter() - t0
                    combo_count += 1

                    diverged = any(rr["diverged"] for rr in round_rows)
                    manifest_rows.append({
                        "run_id": run_id, "seed": seed, "partition": tag, "attack": attack, "candidate": candidate,
                        "n_rounds": N_ROUNDS, "diverged": diverged, "runtime_s": dt,
                        "final_accuracy": round_rows[-1]["accuracy"], "final_loss": round_rows[-1]["loss"],
                    })
                    for rr in round_rows:
                        rr2 = dict(rr)
                        rr2.update(run_id=run_id, seed=seed, partition=tag, attack=attack, candidate=candidate)
                        all_round_rows.append(rr2)
                    for cr in client_rows:
                        cr2 = dict(cr)
                        cr2.update(run_id=run_id, seed=seed, partition=tag, attack=attack, candidate=candidate)
                        all_client_rows.append(cr2)
                    for ar in adaptive_rows:
                        ar2 = dict(ar)
                        ar2.update(run_id=run_id, seed=seed, partition=tag, attack=attack, candidate=candidate)
                        all_adaptive_rows.append(ar2)

                    print(f"[{combo_count:3d}/{total_combos}] {run_id:60s} {dt:5.1f}s final_acc={round_rows[-1]['accuracy']:.3f} diverged={diverged}")

            _write_csv(os.path.join(OUT_DIR, "run_manifest.csv"), manifest_rows)
            _write_csv(os.path.join(OUT_DIR, "round_metrics.csv"), all_round_rows)
            _write_csv(os.path.join(OUT_DIR, "client_decisions.csv"), all_client_rows)
            _write_csv(os.path.join(OUT_DIR, "adaptive_threshold_trace.csv"), all_adaptive_rows)
            print(f"  -- checkpointed after {tag} seed={seed}: {len(manifest_rows)} runs so far, elapsed={time.perf_counter()-t_start:.0f}s --")

    _write_csv(os.path.join(OUT_DIR, "run_manifest.csv"), manifest_rows)
    _write_csv(os.path.join(OUT_DIR, "round_metrics.csv"), all_round_rows)
    _write_csv(os.path.join(OUT_DIR, "client_decisions.csv"), all_client_rows)
    _write_csv(os.path.join(OUT_DIR, "adaptive_threshold_trace.csv"), all_adaptive_rows)
    total = time.perf_counter() - t_start
    print(f"DONE. {combo_count} runs, total time {total:.0f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
