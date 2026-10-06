"""Coordinate-wise median — robust aggregator, not a detector. Wraps the one
canonical implementation (fl_core.robust_agg.coord_median) chosen in
docs/IMPLEMENTATION_PLAN.md Sec 3.3, rather than adding a fourth divergent
copy alongside the three already in the upstream experiments/ scripts."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from fl_core.robust_agg import coord_median

from .base import Defense, DefenseResult


class CoordMedian(Defense):
    name = "median"

    def aggregate(self, deltas: List[np.ndarray], ctx: Dict[str, Any]) -> DefenseResult:
        update = coord_median(deltas)
        return DefenseResult(update=update, scores=None, accepted=None, is_detector=False)
