from __future__ import annotations

import numpy as np

from .common import calibrate_threshold


def calibrate_rho(honest_cos: np.ndarray, malicious_cos: np.ndarray, fpr_target: float = 0.05):
    out = calibrate_threshold(honest_cos, malicious_cos, direction="reject_below", fpr_target=fpr_target)
    out["param"] = "rho"
    out["method"] = "validation_grid_fpr_target"
    return out
