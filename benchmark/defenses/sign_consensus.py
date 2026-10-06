"""Coordinate-wise majority-sign-agreement score. The literature-grounded
definition (signSGD majority vote, adapted as a per-client consensus/anomaly
score) is locked and documented in docs/BENCHMARK_PROTOCOL.md Sec 7 — read
that section for the formula derivation, tie handling, and the documented
limitation (a large colluding fraction can poison the majority itself).
This module is a direct implementation of that locked formula; it does not
introduce any scoring rule not already written down there.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .base import Defense, DefenseResult


def sign_consensus_scores(deltas: List[np.ndarray], sign_topk: Optional[int] = None) -> np.ndarray:
    mat = np.stack(deltas, axis=0)
    signs = np.sign(mat)
    majority = np.sign(signs.sum(axis=0))

    coord_mask = np.ones(mat.shape[1], dtype=bool)
    if sign_topk:
        mean_abs = np.mean(np.abs(mat), axis=0)
        top_idx = np.argsort(-mean_abs)[: int(sign_topk)]
        coord_mask = np.zeros(mat.shape[1], dtype=bool)
        coord_mask[top_idx] = True

    valid = (majority != 0) & coord_mask
    denom = max(1, int(valid.sum()))
    scores = np.array(
        [float(np.sum((signs[i] == majority) & valid)) / denom for i in range(len(deltas))]
    )
    return scores


class SignConsensus(Defense):
    name = "sign_consensus"

    def aggregate(self, deltas: List[np.ndarray], ctx: Dict[str, Any]) -> DefenseResult:
        kappa = ctx["kappa"]
        if kappa is None:
            raise ValueError("SignConsensus requires ctx['kappa'] (calibrate first; see docs/BENCHMARK_PROTOCOL.md Sec 8)")
        sign_topk = ctx.get("sign_topk")
        scores = sign_consensus_scores(deltas, sign_topk=sign_topk)
        accepted = scores >= kappa
        if not np.any(accepted):
            update = np.zeros_like(deltas[0])
        else:
            update = np.mean([d for d, a in zip(deltas, accepted) if a], axis=0)
        return DefenseResult(update=update, scores=scores, accepted=accepted, is_detector=True)
