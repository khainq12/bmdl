from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .base import Defense, DefenseResult


class FedAvg(Defense):
    name = "fedavg"

    def aggregate(self, deltas: List[np.ndarray], ctx: Dict[str, Any]) -> DefenseResult:
        update = np.mean(deltas, axis=0)
        return DefenseResult(update=update, scores=None, accepted=None, is_detector=False)
