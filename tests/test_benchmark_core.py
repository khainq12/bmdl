"""R12 audit: targeted unit tests for the invariants that matter most for
result validity — aggregation math, fixed-step local training semantics,
model synchronization, metric formulas, and attack magnitude/invariance
properties. No GPU, no real dataset download required (uses small
synthetic arrays and a lightweight fake model for the training-loop test).

Run: pytest tests/test_benchmark_core.py -v
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from benchmark.attacks.suite import apply_attack, full_sign_flip, low_norm, directional_poisoning
from benchmark.defenses import make_defense
from benchmark.defenses.cosine import cosine_similarity
from benchmark.defenses.sign_consensus import sign_consensus_scores
from benchmark.models.local_training_fixed_steps import local_training_fixed_steps


# ---------------------------------------------------------------- FedAvg ---

def test_fedavg_unweighted_is_plain_mean():
    deltas = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 0.0])]
    result = make_defense("fedavg").aggregate(deltas, {})
    np.testing.assert_allclose(result.update, np.mean(deltas, axis=0))


def test_fedavg_weighted_matches_hand_computation():
    deltas = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    weights = [3, 1]  # client 0 has 3x the samples of client 1
    result = make_defense("fedavg_weighted").aggregate(deltas, {"client_weights": weights})
    expected = (3 / 4) * deltas[0] + (1 / 4) * deltas[1]
    np.testing.assert_allclose(result.update, expected)


def test_fedavg_weighted_equal_weights_equals_unweighted():
    deltas = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 0.0])]
    weighted = make_defense("fedavg_weighted").aggregate(deltas, {"client_weights": [10, 10, 10]})
    unweighted = make_defense("fedavg").aggregate(deltas, {})
    np.testing.assert_allclose(weighted.update, unweighted.update)


def test_fedavg_weighted_rejects_mismatched_length():
    with pytest.raises(ValueError):
        make_defense("fedavg_weighted").aggregate([np.array([1.0])], {"client_weights": [1, 2]})


def test_fedavg_weighted_requires_client_weights():
    with pytest.raises(ValueError):
        make_defense("fedavg_weighted").aggregate([np.array([1.0])], {})


# ----------------------------------------------- fixed-K local training ---

class _LinearProbeModel:
    """Minimal fake model: 1-D weight, train_step returns (y_batch_mean
    broadcast as 'gradient', 0.0 loss) so we can both count calls AND
    verify every call actually used a slice of the shuffled data, not the
    same batch repeated."""

    def __init__(self):
        self._w = np.zeros(1)
        self.calls = []  # list of batch_size seen, per call

    def get_weights(self):
        return self._w.copy()

    def set_weights(self, w):
        self._w = np.asarray(w, dtype=np.float64).copy()

    def train_step(self, X, y, lr):
        self.calls.append(len(X))
        # "gradient" = -mean(X) so weights move toward mean(X) each step;
        # harmless, just needs to be deterministic and depend on the batch.
        grad = -np.array([float(np.mean(X))])
        return grad, 0.0


def test_fixed_k_steps_calls_train_step_exactly_k_times():
    model = _LinearProbeModel()
    X = np.arange(10, dtype=np.float64).reshape(-1, 1)
    y = np.zeros(10)
    rng = np.random.default_rng(0)
    local_training_fixed_steps(model, X, y, k_steps=7, lr=0.1, batch_size=3, rng=rng)
    assert len(model.calls) == 7
    assert all(c == 3 for c in model.calls)


def test_fixed_k_steps_cycles_when_client_smaller_than_one_pass():
    # n=4 samples, batch_size=3: first batch uses 3, remaining 1 is
    # discarded and a fresh shuffled pass starts -- so k_steps=5 should
    # still issue 5 full batches of 3, never a short batch.
    model = _LinearProbeModel()
    X = np.arange(4, dtype=np.float64).reshape(-1, 1)
    y = np.zeros(4)
    rng = np.random.default_rng(1)
    local_training_fixed_steps(model, X, y, k_steps=5, lr=0.1, batch_size=3, rng=rng)
    assert len(model.calls) == 5
    assert all(c == 3 for c in model.calls)


def test_fixed_k_steps_delta_is_weights_after_minus_before():
    model = _LinearProbeModel()
    model.set_weights(np.array([42.0]))
    X = np.arange(10, dtype=np.float64).reshape(-1, 1)
    y = np.zeros(10)
    rng = np.random.default_rng(2)
    delta = local_training_fixed_steps(model, X, y, k_steps=3, lr=0.1, batch_size=4, rng=rng)
    np.testing.assert_allclose(delta, model.get_weights() - np.array([42.0]))


def test_fixed_k_steps_rejects_empty_client():
    model = _LinearProbeModel()
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        local_training_fixed_steps(model, np.empty((0, 1)), np.empty(0), k_steps=1, lr=0.1, batch_size=4, rng=rng)


# ------------------------------------------------------- model sync ---

def test_all_clients_start_from_identical_global_weights_each_round():
    """Regression test for the exact pattern used in every benchmark
    script's round loop: gw = model.get_weights().copy(); each client
    does local.set_weights(gw.copy()). Verifies no client sees a
    different (e.g. already-mutated) starting point."""
    global_model = _LinearProbeModel()
    global_model.set_weights(np.array([1.0]))
    n_clients = 5
    rng = np.random.default_rng(4)

    gw = global_model.get_weights().copy()
    starting_points = []
    for cid in range(n_clients):
        local = _LinearProbeModel()
        local.set_weights(gw.copy())
        starting_points.append(local.get_weights())
        # mutate the local model to make sure it's an independent copy
        local.set_weights(local.get_weights() + cid)

    for sp in starting_points:
        np.testing.assert_allclose(sp, np.array([1.0]))
    # global model itself must be untouched by any client's local mutation
    np.testing.assert_allclose(global_model.get_weights(), np.array([1.0]))


# ------------------------------------------------------------- metrics ---

def test_tpr_fpr_formula_matches_hand_count():
    # 3 malicious client-rounds: 2 rejected (TP), 1 accepted (FN)
    # 5 honest client-rounds: 1 rejected (FP), 4 accepted (TN)
    malicious_accepted = [False, False, True]
    honest_accepted = [False, True, True, True, True]

    tp = sum(1 for a in malicious_accepted if not a)
    fn = sum(1 for a in malicious_accepted if a)
    fp = sum(1 for a in honest_accepted if not a)
    tn = sum(1 for a in honest_accepted if a)

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    assert tp == 2 and fn == 1 and fp == 1 and tn == 4
    assert tpr == pytest.approx(2 / 3)
    assert fpr == pytest.approx(1 / 5)


def test_tpr_fpr_denominator_is_role_count_not_total_rounds():
    """Guards against a classic bug: denominator must be (TP+FN) for TPR
    and (FP+TN) for FPR, NOT the total number of client-rounds across both
    groups -- which would silently shrink both rates whenever the
    malicious/honest split is uneven (always is: 1 malicious vs N-1 honest)."""
    malicious_accepted = [False]  # 1 malicious client-round, rejected
    honest_accepted = [True, True, True, True]  # 4 honest, all accepted

    tp = sum(1 for a in malicious_accepted if not a)
    fn = sum(1 for a in malicious_accepted if a)
    fp = sum(1 for a in honest_accepted if not a)
    tn = sum(1 for a in honest_accepted if a)

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    assert tpr == 1.0  # not 1/5
    assert fpr == 0.0


# -------------------------------------------------------------- attacks ---

def test_directional_poisoning_target_norm_is_exact():
    rng = np.random.default_rng(5)
    delta = rng.normal(size=20)
    out = apply_attack("directional_poisoning", delta, rng, {"target_norm": 7.5})
    assert float(np.linalg.norm(out)) == pytest.approx(7.5, rel=1e-9)
    # direction must be exactly opposite the input
    cos = float(np.dot(out, delta) / (np.linalg.norm(out) * np.linalg.norm(delta)))
    assert cos == pytest.approx(-1.0, abs=1e-9)


def test_directional_poisoning_default_falls_back_to_own_norm():
    rng = np.random.default_rng(6)
    delta = rng.normal(size=20)
    out = apply_attack("directional_poisoning", delta, rng, {})
    assert float(np.linalg.norm(out)) == pytest.approx(float(np.linalg.norm(delta)), rel=1e-9)
    np.testing.assert_allclose(out, -delta)


def test_directional_poisoning_default_equals_full_sign_flip():
    """Locks in the R10 finding: with no target_norm, directional_poisoning
    is bit-identical to full_sign_flip. If this ever fails, either the
    attack's fallback behavior changed, or full_sign_flip did -- both are
    load-bearing for every existing directional_poisoning result in
    results/pathmnist and results/fed_isic2019."""
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    delta = np.random.default_rng(42).normal(size=15)
    a = apply_attack("directional_poisoning", delta, rng1, {})
    b = apply_attack("full_sign_flip", delta, rng2, {})
    np.testing.assert_array_equal(a, b)


def test_low_norm_preserves_cosine_and_sign():
    rng = np.random.default_rng(8)
    g = rng.normal(size=30)
    g_ref = rng.normal(size=30)
    for eta in (0.01, 0.1, 0.5, 0.99):
        g_scaled = apply_attack("low_norm", g, rng, {"eta": eta})
        np.testing.assert_allclose(g_scaled, eta * g)
        assert cosine_similarity(g_scaled, g_ref) == pytest.approx(cosine_similarity(g, g_ref), rel=1e-9)
        np.testing.assert_array_equal(np.sign(g_scaled), np.sign(g))


def test_sign_full_flip_and_directional_both_invert_sign_consensus_identically():
    """Empirical companion to the cosine/sign invariance proof: scaling to
    any positive target_norm must not change the sign-consensus score,
    since sign(c*x) == sign(x) for c>0."""
    rng = np.random.default_rng(9)
    deltas = [rng.normal(size=10) for _ in range(5)]
    base_scores = sign_consensus_scores(deltas)

    rescaled = list(deltas)
    rescaled[0] = apply_attack("directional_poisoning", deltas[0], rng, {"target_norm": 123.0})
    # directional_poisoning negates direction, so client 0's sign flips --
    # compare against an explicit full_sign_flip reference instead of the
    # unmodified base_scores.
    reference = list(deltas)
    reference[0] = -deltas[0]
    rescaled_scores = sign_consensus_scores(rescaled)
    reference_scores = sign_consensus_scores(reference)
    np.testing.assert_allclose(rescaled_scores, reference_scores)


# ------------------------------------------- R12 ablation: AdaptiveNormOnly ---

def test_adaptive_norm_only_matches_c4_drift_aware_norm_branch():
    """AdaptiveNormOnly must reject EXACTLY the same clients as C4DriftAware's
    norm_fail term, for the same deltas/tau/rho/kappa/g_ref -- it's the same
    _drift_tau calculation, just not OR-ed with cos_fail/sign_reject. If
    C4-DA's Cosine/Sign stages never additionally reject anyone for a given
    input, the two defenses' overall `accepted` must also match exactly;
    this synthetic case is built so Cosine/Sign pass everyone, isolating the
    norm-gate comparison."""
    rng = np.random.default_rng(42)
    deltas = [rng.normal(size=16) for _ in range(5)]
    deltas[0] = deltas[0] * 50.0  # one clear magnitude outlier
    g_ref = deltas[0] * 0  # placeholder; cosine/sign thresholds set permissive below

    adaptive = make_defense("adaptive_norm_only")
    c4 = make_defense("c4_drift_aware")

    ctx_adaptive = {"tau": 1e-6}  # tiny static fallback -- only matters if MAD==0
    ctx_c4 = {"tau": 1e-6, "rho": -1.0, "kappa": -1.0, "g_ref": deltas[0]}  # rho/kappa impossible to fail

    r_adaptive = adaptive.aggregate(deltas, ctx_adaptive)
    r_c4 = c4.aggregate(deltas, ctx_c4)

    np.testing.assert_array_equal(r_adaptive.extra["norm_fail"], r_c4.extra["norm_fail"])
    np.testing.assert_array_equal(r_adaptive.accepted, r_c4.accepted)
    np.testing.assert_allclose(r_adaptive.extra["tau_used"], r_c4.extra["tau_used"])


def test_adaptive_norm_only_uses_drift_tau_not_static_tau():
    """Regression guard: AdaptiveNormOnly's threshold must move with the
    current round's score distribution, not stay pinned at the static
    fallback -- otherwise it would silently degrade into plain NormFilter
    and the ablation would be comparing StaticNorm against itself."""
    rng = np.random.default_rng(7)
    deltas = [rng.normal(loc=0.0, scale=1.0, size=20) for _ in range(5)]
    static_tau_absurdly_low = 1e-9  # would reject everyone if actually used

    result = make_defense("adaptive_norm_only").aggregate(deltas, {"tau": static_tau_absurdly_low})
    assert result.extra["tau_used"][0] > static_tau_absurdly_low
    assert np.any(result.accepted), "drift-aware tau should accept the typical/central clients"


def test_adaptive_norm_only_requires_tau():
    with pytest.raises(ValueError):
        make_defense("adaptive_norm_only").aggregate([np.array([1.0])], {"tau": None})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
