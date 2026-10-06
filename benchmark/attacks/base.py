"""Attack interface. See docs/BENCHMARK_PROTOCOL.md Sec 4 for the locked
formulas each registered attack must implement.

An attack is any callable (honest_delta, rng, params) -> poisoned_delta.
`rng` is always a caller-supplied np.random.Generator seeded by the runner —
an attack must never touch global NumPy random state (this is the fix for
the upstream reproducibility gap documented in docs/IMPLEMENTATION_PLAN.md
Sec 3.9).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

AttackFn = "Callable[[np.ndarray, np.random.Generator, Dict[str, Any]], np.ndarray]"
