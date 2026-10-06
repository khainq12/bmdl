from .common import calibrate_threshold
from .kappa_calibration import calibrate_kappa
from .rho_calibration import calibrate_rho
from .tau_calibration import calibrate_tau

NEEDS_CALIBRATION = {
    "norm": ("tau", calibrate_tau),
    "cosine": ("rho", calibrate_rho),
    "sign_consensus": ("kappa", calibrate_kappa),
}

__all__ = [
    "calibrate_threshold",
    "calibrate_tau",
    "calibrate_rho",
    "calibrate_kappa",
    "NEEDS_CALIBRATION",
]
