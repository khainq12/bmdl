"""Combined Gradient Attestation candidates C3/C4, promoted from offline
replay (results/pathmnist/combined_design_v1/CANDIDATE_DEFINITIONS.md) for
the first real end-to-end FL validation
(results/pathmnist/combined_e2e_minimal/IMPLEMENTATION_AUDIT.md).

Three candidates, exactly as defined offline — no redesign:

- C3Static: hard Norm gate (static tau) + (Cosine fail AND Sign fail).
- C4Static: hard Norm gate (static tau) + Cosine fail OR
  (Sign fail AND peer-relative outlier).
- C4DriftAware: C4 architecture, but the Norm gate threshold is recomputed
  every round from THAT round's 5 submitted client norms (median + 3.5x
  1.4826x MAD, Iglewicz & Hoaglin 1993 convention), falling back to the
  static tau when MAD==0. This uses only information available at decision
  time for round t (current-round cross-sectional statistic) -- it is
  causal, but the malicious client's own norm (if present) contributes to
  the statistic used to judge it; see IMPLEMENTATION_AUDIT.md Critical
  Rule #2 for the self-influence distinction and self_influence_audit.csv
  for the diagnostic (never decision-affecting) leave-one-out check.

None of these three ever use ground-truth malicious labels or future-round
information in their accept/reject decision -- only norm_score/cosine_score
/sign_score computed from the delta actually submitted this round, and the
locked tau/rho/kappa (or, for C4DriftAware's Norm stage, the current
round's median/MAD).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .base import Defense, DefenseResult
from .cosine import cosine_similarity
from .sign_consensus import sign_consensus_scores

MODIFIED_Z_CUTOFF = 3.5
MAD_TO_STD = 1.4826


def _peer_median_excluding_self(values: np.ndarray) -> np.ndarray:
    """For each i, median of all OTHER entries (leave-one-out median)."""
    n = len(values)
    out = np.empty(n)
    for i in range(n):
        others = np.delete(values, i)
        out[i] = np.median(others)
    return out


def _drift_tau(norm_scores: np.ndarray, static_tau: float) -> float:
    m = float(np.median(norm_scores))
    mad = float(np.median(np.abs(norm_scores - m)))
    if mad == 0.0:
        return static_tau
    return m + MODIFIED_Z_CUTOFF * MAD_TO_STD * mad


def _aggregate_from_accept(deltas: List[np.ndarray], accepted: np.ndarray) -> np.ndarray:
    if not np.any(accepted):
        return np.zeros_like(deltas[0])
    return np.mean([d for d, a in zip(deltas, accepted) if a], axis=0)


class C3Static(Defense):
    """Hard Norm gate (static tau) + (Cosine fail AND Sign fail)."""

    name = "c3_static"

    def aggregate(self, deltas: List[np.ndarray], ctx: Dict[str, Any]) -> DefenseResult:
        tau, rho, kappa = ctx["tau"], ctx["rho"], ctx["kappa"]
        g_ref = ctx.get("g_ref")
        if tau is None or rho is None or kappa is None or g_ref is None:
            raise ValueError("C3Static requires ctx['tau'], ctx['rho'], ctx['kappa'], ctx['g_ref']")

        norm_scores = np.array([float(np.linalg.norm(d)) for d in deltas])
        cos_scores = np.array([cosine_similarity(d, g_ref) for d in deltas])
        sign_scores = sign_consensus_scores(deltas)

        norm_fail = norm_scores > tau
        cos_fail = cos_scores < rho
        sign_fail = sign_scores < kappa

        reject = norm_fail | (cos_fail & sign_fail)
        accepted = ~reject
        update = _aggregate_from_accept(deltas, accepted)

        extra = {
            "norm_score": norm_scores, "cosine_score": cos_scores, "sign_score": sign_scores,
            "norm_fail": norm_fail, "cos_fail": cos_fail, "sign_fail": sign_fail,
            "tau_used": np.full(len(deltas), tau, dtype=float),
        }
        return DefenseResult(update=update, scores=norm_scores, accepted=accepted, is_detector=True, extra=extra)


class C4Static(Defense):
    """Hard Norm gate (static tau) + Cosine fail OR (Sign fail AND peer-relative outlier)."""

    name = "c4_static"

    def aggregate(self, deltas: List[np.ndarray], ctx: Dict[str, Any]) -> DefenseResult:
        tau, rho, kappa = ctx["tau"], ctx["rho"], ctx["kappa"]
        g_ref = ctx.get("g_ref")
        if tau is None or rho is None or kappa is None or g_ref is None:
            raise ValueError("C4Static requires ctx['tau'], ctx['rho'], ctx['kappa'], ctx['g_ref']")

        norm_scores = np.array([float(np.linalg.norm(d)) for d in deltas])
        cos_scores = np.array([cosine_similarity(d, g_ref) for d in deltas])
        sign_scores = sign_consensus_scores(deltas)

        norm_fail = norm_scores > tau
        cos_fail = cos_scores < rho
        sign_fail = sign_scores < kappa
        peer_median = _peer_median_excluding_self(sign_scores)
        sign_peer_outlier = sign_scores < peer_median
        sign_reject = sign_fail & sign_peer_outlier

        reject = norm_fail | cos_fail | sign_reject
        accepted = ~reject
        update = _aggregate_from_accept(deltas, accepted)

        extra = {
            "norm_score": norm_scores, "cosine_score": cos_scores, "sign_score": sign_scores,
            "norm_fail": norm_fail, "cos_fail": cos_fail, "sign_fail": sign_fail,
            "sign_peer_outlier": sign_peer_outlier, "tau_used": np.full(len(deltas), tau, dtype=float),
        }
        return DefenseResult(update=update, scores=norm_scores, accepted=accepted, is_detector=True, extra=extra)


class C4DriftAware(Defense):
    """C4 architecture, Norm gate uses current-round median+MAD (causal,
    current-round statistic -- see module docstring and
    IMPLEMENTATION_AUDIT.md for the self-influence distinction)."""

    name = "c4_drift_aware"

    def aggregate(self, deltas: List[np.ndarray], ctx: Dict[str, Any]) -> DefenseResult:
        static_tau, rho, kappa = ctx["tau"], ctx["rho"], ctx["kappa"]
        g_ref = ctx.get("g_ref")
        if static_tau is None or rho is None or kappa is None or g_ref is None:
            raise ValueError("C4DriftAware requires ctx['tau'] (fallback), ctx['rho'], ctx['kappa'], ctx['g_ref']")

        norm_scores = np.array([float(np.linalg.norm(d)) for d in deltas])
        cos_scores = np.array([cosine_similarity(d, g_ref) for d in deltas])
        sign_scores = sign_consensus_scores(deltas)

        tau_drift = _drift_tau(norm_scores, static_tau)
        norm_fail = norm_scores > tau_drift
        cos_fail = cos_scores < rho
        sign_fail = sign_scores < kappa
        peer_median = _peer_median_excluding_self(sign_scores)
        sign_peer_outlier = sign_scores < peer_median
        sign_reject = sign_fail & sign_peer_outlier

        reject = norm_fail | cos_fail | sign_reject
        accepted = ~reject
        update = _aggregate_from_accept(deltas, accepted)

        # self-influence diagnostic: leave-one-out tau, NEVER used in the decision above
        n = len(deltas)
        tau_without = np.empty(n)
        for i in range(n):
            others = np.delete(norm_scores, i)
            tau_without[i] = _drift_tau(others, static_tau) if len(others) > 1 else static_tau

        extra = {
            "norm_score": norm_scores, "cosine_score": cos_scores, "sign_score": sign_scores,
            "norm_fail": norm_fail, "cos_fail": cos_fail, "sign_fail": sign_fail,
            "sign_peer_outlier": sign_peer_outlier,
            "tau_used": np.full(n, tau_drift, dtype=float),
            "tau_static_fallback": np.full(n, static_tau, dtype=float),
            "median_norm": np.full(n, float(np.median(norm_scores)), dtype=float),
            "mad_norm": np.full(n, float(np.median(np.abs(norm_scores - np.median(norm_scores)))), dtype=float),
            "tau_without_self": tau_without,  # diagnostic only, see IMPLEMENTATION_AUDIT.md
        }
        return DefenseResult(update=update, scores=norm_scores, accepted=accepted, is_detector=True, extra=extra)


class C4DriftAwareNoNorm(Defense):
    """End-to-end ablation of locked C4-DA v1: magnitude safety gate
    removed entirely (no norm_fail term at all -- not masked post-hoc,
    structurally absent). Directional stage unchanged. Tests whether the
    Norm signal is actually necessary, not assumed to be."""

    name = "c4_drift_aware_no_norm"

    def aggregate(self, deltas: List[np.ndarray], ctx: Dict[str, Any]) -> DefenseResult:
        rho, kappa = ctx["rho"], ctx["kappa"]
        g_ref = ctx.get("g_ref")
        norm_scores = np.array([float(np.linalg.norm(d)) for d in deltas])
        cos_scores = np.array([cosine_similarity(d, g_ref) for d in deltas])
        sign_scores = sign_consensus_scores(deltas)
        cos_fail = cos_scores < rho
        sign_fail = sign_scores < kappa
        peer_median = _peer_median_excluding_self(sign_scores)
        sign_reject = sign_fail & (sign_scores < peer_median)
        reject = cos_fail | sign_reject
        accepted = ~reject
        update = _aggregate_from_accept(deltas, accepted)
        extra = {"norm_score": norm_scores, "cosine_score": cos_scores, "sign_score": sign_scores,
                 "cos_fail": cos_fail, "sign_fail": sign_fail, "sign_peer_outlier": sign_scores < peer_median}
        return DefenseResult(update=update, scores=norm_scores, accepted=accepted, is_detector=True, extra=extra)


class C4DriftAwareNoCosine(Defense):
    """End-to-end ablation of locked C4-DA v1: Cosine stage removed
    entirely. Magnitude gate (drift-aware) and Sign stage unchanged."""

    name = "c4_drift_aware_no_cosine"

    def aggregate(self, deltas: List[np.ndarray], ctx: Dict[str, Any]) -> DefenseResult:
        static_tau, kappa = ctx["tau"], ctx["kappa"]
        norm_scores = np.array([float(np.linalg.norm(d)) for d in deltas])
        sign_scores = sign_consensus_scores(deltas)
        tau_drift = _drift_tau(norm_scores, static_tau)
        norm_fail = norm_scores > tau_drift
        sign_fail = sign_scores < kappa
        peer_median = _peer_median_excluding_self(sign_scores)
        sign_reject = sign_fail & (sign_scores < peer_median)
        reject = norm_fail | sign_reject
        accepted = ~reject
        update = _aggregate_from_accept(deltas, accepted)
        extra = {"norm_score": norm_scores, "sign_score": sign_scores,
                 "norm_fail": norm_fail, "sign_fail": sign_fail, "sign_peer_outlier": sign_scores < peer_median,
                 "tau_used": np.full(len(deltas), tau_drift, dtype=float)}
        return DefenseResult(update=update, scores=norm_scores, accepted=accepted, is_detector=True, extra=extra)


class C4DriftAwareNoSign(Defense):
    """End-to-end ablation of locked C4-DA v1: Sign stage removed entirely.
    Magnitude gate (drift-aware) and Cosine stage unchanged."""

    name = "c4_drift_aware_no_sign"

    def aggregate(self, deltas: List[np.ndarray], ctx: Dict[str, Any]) -> DefenseResult:
        static_tau, rho = ctx["tau"], ctx["rho"]
        g_ref = ctx.get("g_ref")
        norm_scores = np.array([float(np.linalg.norm(d)) for d in deltas])
        cos_scores = np.array([cosine_similarity(d, g_ref) for d in deltas])
        tau_drift = _drift_tau(norm_scores, static_tau)
        norm_fail = norm_scores > tau_drift
        cos_fail = cos_scores < rho
        reject = norm_fail | cos_fail
        accepted = ~reject
        update = _aggregate_from_accept(deltas, accepted)
        extra = {"norm_score": norm_scores, "cosine_score": cos_scores,
                 "norm_fail": norm_fail, "cos_fail": cos_fail,
                 "tau_used": np.full(len(deltas), tau_drift, dtype=float)}
        return DefenseResult(update=update, scores=norm_scores, accepted=accepted, is_detector=True, extra=extra)
