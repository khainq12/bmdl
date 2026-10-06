from __future__ import annotations

import numpy as np

from .common import calibrate_threshold


def calibrate_kappa(honest_sign: np.ndarray, malicious_sign: np.ndarray, fpr_target: float = 0.05):
    out = calibrate_threshold(honest_sign, malicious_sign, direction="reject_below", fpr_target=fpr_target)
    out["param"] = "kappa"
    out["method"] = "validation_grid_fpr_target"
    return out
