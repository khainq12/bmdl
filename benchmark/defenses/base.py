"""Defense interface. See docs/BENCHMARK_PROTOCOL.md Sec 5.

Per the task brief: Median and Multi-Krum are robust aggregators, not
detectors — they must set is_detector=False and leave scores/accepted as
None, so the metrics layer never computes TPR/FPR/precision for them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DefenseResult:
    update: np.ndarray
    scores: Optional[np.ndarray]  # per-client score; None if not a detector
    accepted: Optional[np.ndarray]  # boolean mask; None if not a detector
    is_detector: bool
    extra: Optional[Dict[str, Any]] = None  # richer per-signal diagnostics (Combined defenses only)


class Defense:
    name: str = "base"

    def aggregate(self, deltas: List[np.ndarray], ctx: Dict[str, Any]) -> DefenseResult:
        raise NotImplementedError
