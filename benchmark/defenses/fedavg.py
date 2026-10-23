from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .base import Defense, DefenseResult


class FedAvg(Defense):
    """Unweighted mean of client deltas. Deliberate choice for this
    benchmark's fixed-K-step local training (benchmark/models/
    local_training_fixed_steps.py): every client performs the same number
    of gradient steps regardless of dataset size specifically so that
    delta magnitude is comparable across client sizes; re-weighting by n_i
    afterward would partially undo that and reintroduce a size bias the
    fixed-step design was built to remove. Not equivalent to the classic
    n_i/n-weighted FedAvg formula (McMahan et al. 2017) — see FedAvgWeighted
    below for that variant, used only for the explicit weighted-vs-unweighted
    baseline comparison (R12)."""

    name = "fedavg"

    def aggregate(self, deltas: List[np.ndarray], ctx: Dict[str, Any]) -> DefenseResult:
        update = np.mean(deltas, axis=0)
        return DefenseResult(update=update, scores=None, accepted=None, is_detector=False)


class FedAvgWeighted(Defense):
    """Classic sample-size-weighted FedAvg: update = sum_i (n_i/n) * delta_i,
    matching McMahan et al. (2017) eq. and the formula shown in the report's
    Chapter 2. Requires ctx['client_weights'] = list of n_i (number of local
    samples used to produce each delta, same order as `deltas`)."""

    name = "fedavg_weighted"

    def aggregate(self, deltas: List[np.ndarray], ctx: Dict[str, Any]) -> DefenseResult:
        weights = ctx.get("client_weights")
        if weights is None:
            raise ValueError("FedAvgWeighted requires ctx['client_weights'] (per-client n_i)")
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != len(deltas):
            raise ValueError(f"client_weights length {len(w)} != deltas length {len(deltas)}")
        if w.sum() <= 0:
            raise ValueError("client_weights must sum to a positive value")
        w = w / w.sum()
        update = np.sum([wi * d for wi, d in zip(w, deltas)], axis=0)
        return DefenseResult(update=update, scores=None, accepted=None, is_detector=False)
