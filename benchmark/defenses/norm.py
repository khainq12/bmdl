"""Plain (non-cryptographic) norm-threshold filter. Deliberately independent
of crypto.zkp_norm.ZKPNormBound — see docs/BENCHMARK_PROTOCOL.md Sec 5."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .base import Defense, DefenseResult


class NormFilter(Defense):
    name = "norm"

    def aggregate(self, deltas: List[np.ndarray], ctx: Dict[str, Any]) -> DefenseResult:
        tau = ctx["tau"]
        if tau is None:
            raise ValueError("NormFilter requires ctx['tau'] (calibrate first; see docs/BENCHMARK_PROTOCOL.md Sec 8)")
        scores = np.array([float(np.linalg.norm(d)) for d in deltas])
        accepted = scores <= tau
        if not np.any(accepted):
            update = np.zeros_like(deltas[0])
        else:
            update = np.mean([d for d, a in zip(deltas, accepted) if a], axis=0)
        return DefenseResult(update=update, scores=scores, accepted=accepted, is_detector=True)
