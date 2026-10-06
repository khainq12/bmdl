"""Defense registry. Combined Attestation is intentionally absent — see
docs/BENCHMARK_PROTOCOL.md Sec 5 ("out of scope for this pass")."""

from __future__ import annotations

from .base import Defense, DefenseResult
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
}

DETECTOR_DEFENSES = {"norm", "cosine", "sign_consensus"}


def make_defense(name: str) -> Defense:
    if name not in DEFENSES:
        raise ValueError(
            f"unknown or not-yet-implemented defense '{name}'. Registered: {sorted(DEFENSES)}. "
            "Combined Attestation is deferred — see docs/BENCHMARK_PROTOCOL.md Sec 5/11."
        )
    return DEFENSES[name]()


__all__ = ["Defense", "DefenseResult", "DEFENSES", "DETECTOR_DEFENSES", "make_defense"]
