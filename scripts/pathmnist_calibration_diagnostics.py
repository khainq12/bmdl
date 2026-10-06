"""Calibration diagnostics gate — MUST pass before the full corrected sweep
is launched. Locked design: docs/BENCHMARK_PROTOCOL.md Addendum 2026-10-07
(g)/(h). Uses the fixed-K local training primitive throughout (calibration
pseudo-clients, g_ref, AND the real benchmark-client comparison trajectory
below use the identical K/batch_size).

Checks, per the explicit requirements:
  1. honest calibration Norm distribution vs honest benchmark-client Norm distribution
  2. Cosine distributions (same comparison)
  3. Sign Consensus distributions (same comparison)
  4. tau/rho/kappa (Regime A per partition, Regime B IID-frozen)
  5. expected clean-validation FPR: apply each threshold to REAL benchmark
     -client no_attack scores and report the resulting FPR directly (not
     just calibration's own self-reported FPR)

Usage: python scripts/pathmnist_calibration_diagnostics.py
Writes results/pathmnist/calibration_diagnostics_v2/{diagnostics.json, report.md}
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from benchmark.attacks.suite import apply_attack  # noqa: E402
from benchmark.calibration import calibrate_kappa, calibrate_rho, calibrate_tau  # noqa: E402
from benchmark.datasets.pathmnist import (  # noqa: E402
    _medmnist_split,
    build_or_load_partitions,
    partition_dirichlet_indices,
    partition_iid_indices,
)
from benchmark.defenses.cosine import cosine_similarity  # noqa: E402
from benchmark.defenses.sign_consensus import sign_consensus_scores  # noqa: E402
from benchmark.models import local_training_fixed_steps  # noqa: E402
from benchmark.models.torch_cnn import TorchCNNFlat  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "pathmnist", "calibration_diagnostics_v2")

N_CLIENTS = 5
PARTITIONS = [("iid", None), ("dirichlet", 1.0), ("dirichlet", 0.5), ("dirichlet", 0.1)]
SERVER_REF_FRACTION = 0.10
DEVICE = os.environ.get("ZKFL_DEVICE", "cuda")

# Locked, same as the full sweep (Addendum (g)):
K_STEPS = 50
BATCH_SIZE = 128
LOCAL_LR = 0.01

CALIB_ATTACK = "large_norm"
CALIB_ROUNDS = 5
CALIB_SEED = 1000
FPR_TARGET = 0.05
N_CALIB_CLIENTS = max(3, N_CLIENTS)
MALICIOUS_RATIO = 0.2

# Benchmark-client comparison trajectory: honest-only, same length as calibration
# (this script's job is to check SCALE/distribution match, not re-derive the
# full-sweep round budget — see pathmnist_round_budget_study.py for that).
COMPARISON_ROUNDS = 5
COMPARISON_SEED = 42

CALIB_FN = {"norm": calibrate_tau, "cosine": calibrate_rho, "sign_consensus": calibrate_kappa}
THRESH_KEY = {"norm": "tau", "cosine": "rho", "sign_consensus": "kappa"}


def _tag(partition: str, alpha) -> str:
    return "iid" if partition == "iid" else f"dirichlet_a{alpha}"


def _model(n_classes: int, seed: int) -> TorchCNNFlat:
    return TorchCNNFlat(n_classes=n_classes, seed=seed, device=DEVICE)


def _stats(values) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "median": float(np.median(v)),
        "p5": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def calibrate_for_condition(n_classes, defense_name, partition, alpha, X_val, y_val, server_ref_xy):
    rng = np.random.default_rng(CALIB_SEED)
    if partition == "iid":
        calib_idx = partition_iid_indices(len(y_val), N_CALIB_CLIENTS, seed=CALIB_SEED)
    else:
        calib_idx = partition_dirichlet_indices(len(y_val), y_val, N_CALIB_CLIENTS, alpha, seed=CALIB_SEED)
    calib_partitions = [(X_val[c], y_val[c]) for c in calib_idx]

    model = _model(n_classes, CALIB_SEED)
    n_mal = max(1, int(round(MALICIOUS_RATIO * N_CALIB_CLIENTS)))
    honest_scores: List[float] = []
    malicious_scores: List[float] = []
    gref_step_counts: List[int] = []
    client_step_counts: List[int] = []

    for r in range(CALIB_ROUNDS):
        gw = model.get_weights().copy()
        g_ref = None
        if defense_name == "cosine":
            Xr, yr = server_ref_xy
            m = _model(n_classes, CALIB_SEED)
            m.set_weights(gw.copy())
            counted = _CountingWrap(m)
            g_ref = local_training_fixed_steps(counted, Xr, yr, K_STEPS, LOCAL_LR, BATCH_SIZE, rng)
            gref_step_counts.append(counted.calls)

        deltas, mal_flags = [], []
        for cid in range(N_CALIB_CLIENTS):
            local = _model(n_classes, CALIB_SEED)
            local.set_weights(gw.copy())
            Xc, yc = calib_partitions[cid]
            counted = _CountingWrap(local)
            d = local_training_fixed_steps(counted, Xc, yc, K_STEPS, LOCAL_LR, BATCH_SIZE, rng)
            client_step_counts.append(counted.calls)
            is_mal = cid < n_mal and r >= 1
            if is_mal:
                d = apply_attack(CALIB_ATTACK, d, rng, {})
            deltas.append(d)
            mal_flags.append(is_mal)

        if defense_name == "norm":
            scores = [float(np.linalg.norm(d)) for d in deltas]
        elif defense_name == "cosine":
            scores = [cosine_similarity(d, g_ref) for d in deltas]
        else:
            scores = sign_consensus_scores(deltas).tolist()

        for s, m_ in zip(scores, mal_flags):
            (malicious_scores if m_ else honest_scores).append(float(s))

        honest_deltas = [d for d, m_ in zip(deltas, mal_flags) if not m_]
        update = np.mean(honest_deltas, axis=0) if honest_deltas else np.mean(deltas, axis=0)
        model.set_weights(gw + update)

    assert all(c == K_STEPS for c in client_step_counts), f"calibration client step-count mismatch: {set(client_step_counts)}"
    if gref_step_counts:
        assert all(c == K_STEPS for c in gref_step_counts), f"calibration g_ref step-count mismatch: {set(gref_step_counts)}"

    result = CALIB_FN[defense_name](np.array(honest_scores), np.array(malicious_scores), fpr_target=FPR_TARGET)
    result["param"] = THRESH_KEY[defense_name]
    result["honest_scores_raw"] = honest_scores
    result["malicious_scores_raw"] = malicious_scores
    return result


class _CountingWrap:
    """Wraps a TorchCNNFlat to count train_step calls for step-count verification."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def get_weights(self):
        return self.inner.get_weights()

    def set_weights(self, flat):
        self.inner.set_weights(flat)

    def train_step(self, X, y, lr):
        self.calls += 1
        return self.inner.train_step(X, y, lr)

    def evaluate(self, X, y):
        return self.inner.evaluate(X, y)


def benchmark_client_scores(n_classes, partition, alpha, X_train, y_train, server_ref_xy, seed):
    """Honest-only trajectory on REAL benchmark (train-pool) clients, same
    K/batch_size, to compare against calibration's score distributions."""
    rec = build_or_load_partitions(
        len(X_train), y_train, partition, alpha if alpha is not None else 0.5, N_CLIENTS, SERVER_REF_FRACTION, seed
    )
    client_idx = [np.array(c) for c in rec["client_idx"]]
    partition_xy = [(X_train[c], y_train[c]) for c in client_idx]

    model = _model(n_classes, seed)
    rng = np.random.default_rng(seed + 8000)
    norm_scores, cos_scores = [], []
    sign_scores_all = []
    client_step_counts = []

    for r in range(COMPARISON_ROUNDS):
        gw = model.get_weights().copy()
        Xr, yr = server_ref_xy
        m = _model(n_classes, seed)
        m.set_weights(gw.copy())
        g_ref = local_training_fixed_steps(m, Xr, yr, K_STEPS, LOCAL_LR, BATCH_SIZE, rng)

        deltas = []
        for Xc, yc in partition_xy:
            local = _model(n_classes, seed)
            local.set_weights(gw.copy())
            counted = _CountingWrap(local)
            d = local_training_fixed_steps(counted, Xc, yc, K_STEPS, LOCAL_LR, BATCH_SIZE, rng)
            client_step_counts.append(counted.calls)
            deltas.append(d)

        norm_scores.extend(float(np.linalg.norm(d)) for d in deltas)
        cos_scores.extend(cosine_similarity(d, g_ref) for d in deltas)
        sign_scores_all.extend(sign_consensus_scores(deltas).tolist())

        update = np.mean(deltas, axis=0)
        model.set_weights(gw + update)

    assert all(c == K_STEPS for c in client_step_counts), f"benchmark-client step-count mismatch: {set(client_step_counts)}"
    return {"norm": norm_scores, "cosine": cos_scores, "sign_consensus": sign_scores_all}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.perf_counter()

    print("Loading official PathMNIST splits ...")
    X_train, y_train, info = _medmnist_split("train")
    X_val, y_val, _ = _medmnist_split("val")
    n_classes = len(info["label"])
    print(f"n_classes={n_classes} K={K_STEPS} batch_size={BATCH_SIZE} device={DEVICE}")

    regimeA: Dict = {}
    regimeB: Dict = {}
    bench_scores: Dict[str, Dict[str, List[float]]] = {}

    for partition, alpha in PARTITIONS:
        tag = _tag(partition, alpha)
        rec = build_or_load_partitions(len(X_train), y_train, partition, alpha if alpha is not None else 0.5, N_CLIENTS, SERVER_REF_FRACTION, CALIB_SEED)
        server_ref_idx = np.array(rec["server_ref_idx"])
        server_ref_xy = (X_train[server_ref_idx], y_train[server_ref_idx])

        for defense_name in ("norm", "cosine", "sign_consensus"):
            r = calibrate_for_condition(n_classes, defense_name, partition, alpha, X_val, y_val, server_ref_xy)
            regimeA[(defense_name, tag)] = r
            print(f"[calib A] {defense_name:14s} {tag:16s} threshold={r['threshold']:.4f} val_fpr={r['validation_fpr']:.3f} val_tpr={r['validation_tpr']:.3f}")

        rec2 = build_or_load_partitions(len(X_train), y_train, partition, alpha if alpha is not None else 0.5, N_CLIENTS, SERVER_REF_FRACTION, COMPARISON_SEED)
        server_ref_idx2 = np.array(rec2["server_ref_idx"])
        server_ref_xy2 = (X_train[server_ref_idx2], y_train[server_ref_idx2])
        bench_scores[tag] = benchmark_client_scores(n_classes, partition, alpha, X_train, y_train, server_ref_xy2, COMPARISON_SEED)
        print(f"[bench]   {tag:16s} norm_mean={np.mean(bench_scores[tag]['norm']):.4f} "
              f"cos_mean={np.mean(bench_scores[tag]['cosine']):.4f} sign_mean={np.mean(bench_scores[tag]['sign_consensus']):.4f}")

    iid_rec = build_or_load_partitions(len(X_train), y_train, "iid", 0.5, N_CLIENTS, SERVER_REF_FRACTION, CALIB_SEED)
    iid_ref_idx = np.array(iid_rec["server_ref_idx"])
    iid_server_ref_xy = (X_train[iid_ref_idx], y_train[iid_ref_idx])
    for defense_name in ("norm", "cosine", "sign_consensus"):
        r = calibrate_for_condition(n_classes, defense_name, "iid", None, X_val, y_val, iid_server_ref_xy)
        regimeB[defense_name] = r
        print(f"[calib B] {defense_name:14s} iid_reference   threshold={r['threshold']:.4f} val_fpr={r['validation_fpr']:.3f} val_tpr={r['validation_tpr']:.3f}")

    # ---------- Diagnostics: distribution comparison + clean-validation FPR ----------
    diagnostics: Dict[str, Any] = {"regimeA": {}, "regimeB": {}}
    for defense_name in ("norm", "cosine", "sign_consensus"):
        for partition, alpha in PARTITIONS:
            tag = _tag(partition, alpha)
            calib_scores = regimeA[(defense_name, tag)]["honest_scores_raw"]
            bench = bench_scores[tag][defense_name]
            threshold = regimeA[(defense_name, tag)]["threshold"]
            direction = regimeA[(defense_name, tag)]["direction"]
            bench_arr = np.asarray(bench)
            clean_fpr = float(np.mean(bench_arr > threshold)) if direction == "reject_above" else float(np.mean(bench_arr < threshold))
            diagnostics["regimeA"][f"{defense_name}__{tag}"] = {
                "calibration_honest_scores": _stats(calib_scores),
                "benchmark_client_honest_scores": _stats(bench),
                "threshold": threshold,
                "calibration_time_validation_fpr": regimeA[(defense_name, tag)]["validation_fpr"],
                "expected_clean_validation_fpr_on_real_clients": clean_fpr,
            }

        threshold_b = regimeB[defense_name]["threshold"]
        direction_b = regimeB[defense_name]["direction"]
        for partition, alpha in PARTITIONS:
            if partition == "iid":
                continue
            tag = _tag(partition, alpha)
            bench = bench_scores[tag][defense_name]
            bench_arr = np.asarray(bench)
            clean_fpr = float(np.mean(bench_arr > threshold_b)) if direction_b == "reject_above" else float(np.mean(bench_arr < threshold_b))
            diagnostics["regimeB"][f"{defense_name}__{tag}"] = {
                "calibration_honest_scores_iid": _stats(regimeB[defense_name]["honest_scores_raw"]),
                "benchmark_client_honest_scores": _stats(bench),
                "threshold_frozen_from_iid": threshold_b,
                "expected_clean_validation_fpr_on_real_clients": clean_fpr,
            }

    with open(os.path.join(OUT_DIR, "diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, default=str)

    # ---------- Report ----------
    lines = ["# Calibration diagnostics v2 (fixed-K) — gate report\n"]
    lines.append(f"K={K_STEPS} batch_size={BATCH_SIZE} calib_rounds={CALIB_ROUNDS} comparison_rounds={COMPARISON_ROUNDS}\n")
    all_pass = True
    for defense_name in ("norm", "cosine", "sign_consensus"):
        lines.append(f"\n## {defense_name} — Regime A (condition-specific)")
        lines.append("| partition | threshold | calib honest mean | bench-client honest mean | calib-time FPR | REAL clean FPR |")
        lines.append("|---|---|---|---|---|---|")
        for partition, alpha in PARTITIONS:
            tag = _tag(partition, alpha)
            d = diagnostics["regimeA"][f"{defense_name}__{tag}"]
            flag = "PASS" if d["expected_clean_validation_fpr_on_real_clients"] <= 0.30 else "CHECK"
            if flag == "CHECK":
                all_pass = False
            lines.append(
                f"| {tag} | {d['threshold']:.4f} | {d['calibration_honest_scores'].get('mean', float('nan')):.4f} | "
                f"{d['benchmark_client_honest_scores'].get('mean', float('nan')):.4f} | "
                f"{d['calibration_time_validation_fpr']:.3f} | {d['expected_clean_validation_fpr_on_real_clients']:.3f} ({flag}) |"
            )

        lines.append(f"\n## {defense_name} — Regime B (IID-calibrated, frozen)")
        lines.append("| partition | frozen threshold | bench-client honest mean | REAL clean FPR |")
        lines.append("|---|---|---|---|")
        for partition, alpha in PARTITIONS:
            if partition == "iid":
                continue
            tag = _tag(partition, alpha)
            d = diagnostics["regimeB"][f"{defense_name}__{tag}"]
            lines.append(
                f"| {tag} | {d['threshold_frozen_from_iid']:.4f} | {d['benchmark_client_honest_scores'].get('mean', float('nan')):.4f} | "
                f"{d['expected_clean_validation_fpr_on_real_clients']:.3f} |"
            )

    lines.append(f"\n\n**Gate result: {'PASS — proceed to pilot' if all_pass else 'CHECK NEEDED — see CHECK-flagged rows above'}**")
    lines.append(
        "\nPASS threshold here is REAL clean FPR <= 0.30 per condition — a sanity bar, not the"
        " calibration.fpr_target=0.05 itself (that target is enforced on calibration data only, by"
        " design; some sampling/location-shift slack between 5k-ish calibration samples and the real"
        " client population is expected and acceptable, but FPR anywhere near the old ~1.0 pathology"
        " must not reappear here)."
    )
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n{'='*80}\nGATE RESULT: {'PASS' if all_pass else 'CHECK NEEDED'}\n{'='*80}")
    print(f"DONE in {time.perf_counter()-t0:.0f}s. Saved {OUT_DIR}/{{diagnostics.json,report.md}}")


if __name__ == "__main__":
    main()
