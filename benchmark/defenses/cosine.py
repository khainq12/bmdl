"""Cosine-similarity filter against a trusted server reference gradient.
g_ref construction/disjointness is locked in docs/BENCHMARK_PROTOCOL.md
Sec 3 and Sec 6 — this module only consumes ctx['g_ref'], it does not build it.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .base import Defense, DefenseResult


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class CosineFilter(Defense):
    name = "cosine"

    def aggregate(self, deltas: List[np.ndarray], ctx: Dict[str, Any]) -> DefenseResult:
        g_ref = ctx.get("g_ref")
        rho = ctx["rho"]
        if g_ref is None:
            raise ValueError("CosineFilter requires ctx['g_ref'] (see docs/BENCHMARK_PROTOCOL.md Sec 6)")
        if rho is None:
            raise ValueError("CosineFilter requires ctx['rho'] (calibrate first; see docs/BENCHMARK_PROTOCOL.md Sec 8)")
        scores = np.array([cosine_similarity(d, g_ref) for d in deltas])
        accepted = scores >= rho
        if not np.any(accepted):
            update = np.zeros_like(deltas[0])
        else:
            update = np.mean([d for d, a in zip(deltas, accepted) if a], axis=0)
        return DefenseResult(update=update, scores=scores, accepted=accepted, is_detector=True)
