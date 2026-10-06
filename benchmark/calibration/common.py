"""Shared validation-only threshold calibration. See
docs/BENCHMARK_PROTOCOL.md Sec 8 for the locked decision rule.

Locked rule: among all candidate thresholds, pick the one maximizing TPR on
malicious scores subject to pooled validation FPR <= fpr_target. If no
candidate meets the FPR target, fall back to the threshold minimizing FPR
and mark used_fallback=True. Thresholds are never chosen from test-set data.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def calibrate_threshold(
    honest_scores: np.ndarray,
    malicious_scores: np.ndarray,
    direction: str,
    fpr_target: float = 0.05,
) -> Dict:
    """direction='reject_above' (norm: lower score = more honest) or
    'reject_below' (cosine/sign: higher score = more honest)."""
    if direction not in ("reject_above", "reject_below"):
        raise ValueError(f"unknown direction: {direction}")

    honest_scores = np.asarray(honest_scores, dtype=np.float64)
    malicious_scores = np.asarray(malicious_scores, dtype=np.float64)
    if honest_scores.size == 0:
        raise ValueError("calibrate_threshold: no honest scores collected")

    candidates = np.unique(np.concatenate([honest_scores, malicious_scores]))
    best = None
    fallback = None
    for t in candidates:
        if direction == "reject_above":
            fpr = float(np.mean(honest_scores > t))
            tpr = float(np.mean(malicious_scores > t)) if malicious_scores.size else 0.0
        else:
            fpr = float(np.mean(honest_scores < t))
            tpr = float(np.mean(malicious_scores < t)) if malicious_scores.size else 0.0

        if fallback is None or fpr < fallback["fpr"]:
            fallback = {"threshold": float(t), "fpr": fpr, "tpr": tpr}
        if fpr <= fpr_target and (best is None or tpr > best["tpr"]):
            best = {"threshold": float(t), "fpr": fpr, "tpr": tpr}

    used_fallback = best is None
    chosen = best or fallback
    return {
        "threshold": chosen["threshold"],
        "validation_fpr": chosen["fpr"],
        "validation_tpr": chosen["tpr"],
        "fpr_target": fpr_target,
        "used_fallback": used_fallback,
        "direction": direction,
        "n_honest": int(honest_scores.size),
        "n_malicious": int(malicious_scores.size),
    }
