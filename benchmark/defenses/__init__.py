"""Defense registry. Combined Attestation candidates (c3_static, c4_static,
c4_drift_aware) were promoted from offline replay — see
results/pathmnist/combined_design_v1/ (design) and
results/pathmnist/combined_e2e_minimal/ (first real end-to-end validation).
docs/BENCHMARK_PROTOCOL.md Sec 5 scoped Combined out of the original
single-method benchmark pass; that pass is complete and preserved
unmodified in results/pathmnist/benchmark_v2_corrected/."""

from __future__ import annotations

from .base import Defense, DefenseResult
from .combined import C3Static, C4DriftAware, C4DriftAwareNoCosine, C4DriftAwareNoNorm, C4DriftAwareNoSign, C4Static
from .cosine import CosineFilter
from .fedavg import FedAvg
from .krum import MultiKrum
from .median import CoordMedian
from .norm import NormFilter
from .sign_consensus import SignConsensus

DEFENSES = {
    "fedavg": FedAvg,
    "norm": NormFilter,
    "cosine": CosineFilter,
    "sign_consensus": SignConsensus,
    "median": CoordMedian,
    "multi_krum": MultiKrum,
    "c3_static": C3Static,
    "c4_static": C4Static,
    "c4_drift_aware": C4DriftAware,
    "c4_drift_aware_no_norm": C4DriftAwareNoNorm,
    "c4_drift_aware_no_cosine": C4DriftAwareNoCosine,
    "c4_drift_aware_no_sign": C4DriftAwareNoSign,
}

DETECTOR_DEFENSES = {
    "norm", "cosine", "sign_consensus", "c3_static", "c4_static", "c4_drift_aware",
    "c4_drift_aware_no_norm", "c4_drift_aware_no_cosine", "c4_drift_aware_no_sign",
}


def make_defense(name: str) -> Defense:
    if name not in DEFENSES:
        raise ValueError(
            f"unknown or not-yet-implemented defense '{name}'. Registered: {sorted(DEFENSES)}. "
            "Combined Attestation is deferred — see docs/BENCHMARK_PROTOCOL.md Sec 5/11."
        )
    return DEFENSES[name]()


__all__ = ["Defense", "DefenseResult", "DEFENSES", "DETECTOR_DEFENSES", "make_defense"]
