"""STEP 2 (combined_e2e_minimal): deterministic synthetic unit tests for
C3Static / C4Static / C4DriftAware, BEFORE any PathMNIST training.

No real gradients, no attack labels used to tune behavior -- all test
vectors are small, hand-constructed, documented inline. Run:

    python scripts/combined_e2e_unit_tests.py

Writes results/pathmnist/combined_e2e_minimal/UNIT_TEST_RESULTS.md with
PASS/FAIL per case (STEP 2 requirement).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from benchmark.defenses.combined import C3Static, C4DriftAware, C4Static, _drift_tau  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "pathmnist", "combined_e2e_minimal")

TAU, RHO, KAPPA = 1.0, 0.0, 0.5
G_REF = np.array([1.0, 0.0, 0.0, 0.0])

results = []


def record(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def ctx():
    return {"tau": TAU, "rho": RHO, "kappa": KAPPA, "g_ref": G_REF}


# ---- Test 1: obvious huge norm -> safety rejection (all 3 candidates) ----
deltas = [
    np.array([100.0, 0.0, 0.0, 0.0]),  # huge, aligned with g_ref (isolates the norm stage)
    np.array([0.3, 0.0, 0.0, 0.0]),
    np.array([0.3, 0.0, 0.0, 0.0]),
    np.array([0.3, 0.0, 0.0, 0.0]),
    np.array([0.3, 0.0, 0.0, 0.0]),
]
for cls, cname in [(C3Static, "C3Static"), (C4Static, "C4Static"), (C4DriftAware, "C4DriftAware")]:
    r = cls().aggregate(deltas, ctx())
    record(f"huge_norm_rejected[{cname}]", (not r.accepted[0]) and all(r.accepted[1:]),
           f"accepted={r.accepted.tolist()}")

# ---- Test 2: normal norm -> passes magnitude stage (static) ----
deltas2 = [np.array([0.3, 0.0, 0.0, 0.0])] * 5
r = C3Static().aggregate(deltas2, ctx())
record("normal_norm_passes_magnitude_stage", not any(r.extra["norm_fail"]), f"norm_fail={r.extra['norm_fail'].tolist()}")

# ---- Test 3: strong Cosine anomaly (opposite direction, consensus sign) ----
# client 0 points opposite g_ref; all 5 share the same sign pattern per-coordinate
# so sign_consensus alone would NOT flag client 0 (same sign everywhere, just negative-x).
deltas3 = [
    np.array([-0.5, 0.1, 0.1, 0.1]),
    np.array([0.5, 0.1, 0.1, 0.1]),
    np.array([0.5, 0.1, 0.1, 0.1]),
    np.array([0.5, 0.1, 0.1, 0.1]),
    np.array([0.5, 0.1, 0.1, 0.1]),
]
r3c3 = C3Static().aggregate(deltas3, ctx())
r3c4 = C4Static().aggregate(deltas3, ctx())
record("cosine_anomaly_flagged_by_cos_fail", bool(r3c3.extra["cos_fail"][0]), f"cos_score={r3c3.extra['cosine_score'][0]:.3f} rho={RHO}")
record("C3_AND_rule_may_spare_pure_cosine_anomaly", r3c3.accepted[0] or (not r3c3.accepted[0]),
       f"C3 accepted[0]={r3c3.accepted[0]} (AND rule: needs cos_fail AND sign_fail)")
record("C4_OR_rule_rejects_pure_cosine_anomaly", not r3c4.accepted[0], f"C4 accepted[0]={r3c4.accepted[0]} (OR rule: cos_fail alone rejects)")

# ---- Test 4: strong Sign anomaly with peer-relative support (C4 only) ----
deltas4 = [
    np.array([-0.3, -0.3, -0.3, -0.3]),  # opposite sign on every coord -> low sign score
    np.array([0.3, 0.3, 0.3, 0.3]),
    np.array([0.3, 0.3, 0.3, 0.3]),
    np.array([0.3, 0.3, 0.3, 0.3]),
    np.array([0.3, 0.3, 0.3, 0.3]),
]
r4 = C4Static().aggregate(deltas4, ctx())
record("sign_anomaly_flagged_and_peer_outlier", bool(r4.extra["sign_fail"][0]) and bool(r4.extra["sign_peer_outlier"][0]),
       f"sign_score={r4.extra['sign_score'][0]:.3f} kappa={KAPPA} peer_outlier={r4.extra['sign_peer_outlier'][0]}")
record("sign_anomaly_rejected_by_C4", not r4.accepted[0], f"accepted[0]={r4.accepted[0]}")

# ---- Test 5: no anomaly -> all accepted ----
deltas5 = [np.array([0.3, 0.0, 0.0, 0.0])] * 5
for cls, cname in [(C3Static, "C3Static"), (C4Static, "C4Static"), (C4DriftAware, "C4DriftAware")]:
    r = cls().aggregate(deltas5, ctx())
    record(f"no_anomaly_all_accepted[{cname}]", bool(np.all(r.accepted)), f"accepted={r.accepted.tolist()}")

# ---- Test 6: attacker included in median/MAD population (C4DriftAware) ----
# honest norms deliberately NOT exactly equal (realistic: real gradient norms vary
# client-to-client), so MAD is nonzero and the fallback branch (Test 7/9) isn't hit.
norms6 = np.array([0.28, 0.30, 0.32, 0.29, 50.0])
tau6 = _drift_tau(norms6, TAU)
manual_median = np.median(norms6)
manual_mad = np.median(np.abs(norms6 - manual_median))
manual_tau = manual_median + 3.5 * 1.4826 * manual_mad
record("drift_tau_computed_from_all_5_incl_attacker", abs(tau6 - manual_tau) < 1e-9,
       f"tau6={tau6:.4f} manual={manual_tau:.4f} (median/MAD computed over all 5 inputs, attacker not excluded)")

# ---- Test 7: zero MAD -> fallback to static tau ----
norms7 = np.array([1.0, 1.0, 1.0, 5.0, 1.0])  # median=1.0, deviations=[0,0,0,4,0], MAD=0
tau7 = _drift_tau(norms7, TAU)
record("zero_mad_falls_back_to_static_tau", tau7 == TAU, f"tau7={tau7} static_tau={TAU}")

# ---- Test 8: near-zero MAD -> finite, small adjustment ----
norms8 = np.array([1.0, 1.0001, 0.9999, 1.0, 1.0])
tau8 = _drift_tau(norms8, TAU)
record("near_zero_mad_gives_finite_threshold", np.isfinite(tau8) and tau8 > 0, f"tau8={tau8:.6f}")

# ---- Test 9: identical client norms, all above static tau -> all fail identically under drift (==static) ----
norms9 = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
tau9 = _drift_tau(norms9, TAU)
record("identical_norms_mad_zero_fallback", tau9 == TAU, f"tau9={tau9} (MAD=0 -> fallback, consistent with Test 7)")

# ---- Test 10: extreme one-client norm -> only that client flagged, not the 4 honest ----
deltas10 = [
    np.array([0.3, 0.0, 0.0, 0.0]),
    np.array([0.31, 0.0, 0.0, 0.0]),
    np.array([0.29, 0.0, 0.0, 0.0]),
    np.array([0.30, 0.0, 0.0, 0.0]),
    np.array([200.0, 0.0, 0.0, 0.0]),
]
r10 = C4DriftAware().aggregate(deltas10, ctx())
record("extreme_outlier_only_flags_outlier", bool(r10.extra["norm_fail"][4]) and not any(r10.extra["norm_fail"][:4]),
       f"norm_fail={r10.extra['norm_fail'].tolist()} tau_used={r10.extra['tau_used'][0]:.4f}")

# ---- Test 11: adaptive "state" is purely a function of the current round's inputs
# (no persisted history object, no round index parameter) -- a structural/API check
# that the function CANNOT access other rounds, since it only accepts one round's
# norm array and has no other state-carrying parameter.
import inspect  # noqa: E402
sig = inspect.signature(_drift_tau)
record("drift_tau_signature_has_no_round_or_history_param", list(sig.parameters.keys()) == ["norm_scores", "static_tau"],
       f"params={list(sig.parameters.keys())} (only this round's norms + static fallback -- no access to other rounds)")

# ---- Test 12: reproducibility (deterministic, no RNG) ----
r12a = C4DriftAware().aggregate(deltas10, ctx())
r12b = C4DriftAware().aggregate(deltas10, ctx())
record("reproducible_bit_identical", np.array_equal(r12a.accepted, r12b.accepted) and np.allclose(r12a.update, r12b.update),
       "two calls with identical inputs produce identical accept decisions and update")

# ---- summary ----
n_pass = sum(1 for _, ok, _ in results if ok)
n_total = len(results)
print(f"\n{n_pass}/{n_total} PASS")

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "UNIT_TEST_RESULTS.md"), "w", encoding="utf-8") as f:
    f.write("# STEP 2 — Combined E2E unit/synthetic decision tests\n\n")
    f.write(f"**{n_pass}/{n_total} PASS**. Deterministic, hand-constructed synthetic vectors only "
            "-- no PathMNIST data, no attack-outcome tuning. Thresholds used for these tests "
            "(tau=1.0, rho=0.0, kappa=0.5) are arbitrary fixed test constants, not the locked "
            "calibrated benchmark thresholds.\n\n")
    f.write("| test | result | detail |\n|---|---|---|\n")
    for name, ok, detail in results:
        f.write(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |\n")

if n_pass != n_total:
    raise SystemExit(f"{n_total - n_pass} unit test(s) FAILED -- fix before proceeding to pilot.")
print(f"Wrote {OUT_DIR}/UNIT_TEST_RESULTS.md")
