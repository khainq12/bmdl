"""Multi-Krum — robust aggregator, not a detector. Wraps the one canonical
implementation (fl_core.robust_agg.multi_krum), see
docs/IMPLEMENTATION_PLAN.md Sec 3.3."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from fl_core.robust_agg import multi_krum

from .base import Defense, DefenseResult


class MultiKrum(Defense):
    name = "multi_krum"

    def aggregate(self, deltas: List[np.ndarray], ctx: Dict[str, Any]) -> DefenseResult:
        f = int(ctx.get("krum_f", 1))
        update = multi_krum(deltas, f=f)
        return DefenseResult(update=update, scores=None, accepted=None, is_detector=False)
