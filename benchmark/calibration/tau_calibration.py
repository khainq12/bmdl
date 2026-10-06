from __future__ import annotations

import numpy as np

from .common import calibrate_threshold


def calibrate_tau(honest_norms: np.ndarray, malicious_norms: np.ndarray, fpr_target: float = 0.05):
    out = calibrate_threshold(honest_norms, malicious_norms, direction="reject_above", fpr_target=fpr_target)
    out["param"] = "tau"
    out["method"] = "validation_grid_fpr_target"
    return out
