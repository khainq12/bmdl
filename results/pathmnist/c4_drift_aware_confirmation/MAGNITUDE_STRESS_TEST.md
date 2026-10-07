# Targeted Confirmation — C4DriftAware Magnitude Stress Test

Analytical/diagnostic only. Uses the REAL honest-cluster norm distribution from `combined_e2e_minimal/client_decisions.csv` (1200 honest (run,round) groups, 1200 sampled), injects a hypothetical 5th attacker norm at a range of multipliers of that round's honest median, and recomputes `tau_drift` via the unmodified `benchmark/defenses/combined.py::_drift_tau`. **This script never modifies the defense; results are diagnostic, not used to retune C4DriftAware.**

## Pass rate and mean threshold inflation by attacker magnitude multiplier

| multiplier (x honest median) | attacker pass rate | mean threshold inflation |
|---|---|---|
| 1x | 1.0000 | -0.0294 |
| 2x | 0.1792 | 0.0559 |
| 3x | 0.0233 | 0.0565 |
| 5x | 0.0008 | 0.0565 |
| 7x | 0.0000 | 0.0565 |
| 10x | 0.0000 | 0.0565 |
| 15x | 0.0000 | 0.0565 |
| 20x | 0.0000 | 0.0565 |
| 30x | 0.0000 | 0.0565 |
| 50x | 0.0000 | 0.0565 |
| 75x | 0.0000 | 0.0565 |
| 100x | 0.0000 | 0.0565 |
| 150x | 0.0000 | 0.0565 |
| 200x | 0.0000 | 0.0565 |

**Largest multiplier at which the injected attacker norm was ever observed to pass: 5x** (sampled across 1200 real honest-round contexts, 14 multiplier levels from 1x to 200x).

## Interpretation (OBSERVED, not re-tuned)

Per ChatGPT's review: the correct claim is not 'self-influence is negligible' but 'the attacker materially shifts its own judging threshold (median/MAD breakdown-point-bounded), but in every magnitude region tested here, that shift was never enough to let the attacker's own update simultaneously inflate the threshold AND fall under it.' This sweep directly tests for a hypothetical 'sweet spot' where self-influence could become exploitable and reports whether one was found, at what magnitude, and how often.
