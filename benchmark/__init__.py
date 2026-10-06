"""Shared gradient-defense benchmark framework.

See docs/BENCHMARK_PROTOCOL.md for the locked protocol this package
implements, and docs/IMPLEMENTATION_PLAN.md for the overall architecture.
This package is additive: nothing under crypto/, fl_core/, experiments/, or
formal/ is modified or imported-and-mutated by anything here.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
