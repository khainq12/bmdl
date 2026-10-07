# STEP 10 — Fair Baseline Comparison (reused, not rerun)

Source: `baseline_comparability.csv`, joining the 144 Combined runs against
`results/pathmnist/benchmark_v2_corrected/per_round_results.csv` (read-only
— never modified or rerun) on exact (partition, seed, attack, defense,
calibration_regime) match. **864/864 join rows (144 runs × 6 baseline
defenses) matched exactly — 0 forced or approximate comparisons.** Protocol
fields (K=50, 25 rounds, model, optimizer, LR, 5 clients, attack start
round 10, attacker = client 0, evaluation protocol) are identical between
the original 594-combo sweep and this benchmark by construction (both
derive these constants from the same locked protocol addenda) — see
`baseline_comparability.csv`'s `protocol_fields` column for the explicit
statement attached to every row.

## Mean final accuracy, pooled across attacks, by partition

| partition | FedAvg | Cosine | Sign Consensus | Median | Norm | Multi-Krum | c3_static | c4_static | **c4_drift_aware** |
|---|---|---|---|---|---|---|---|---|---|
| iid | 0.315 | 0.436 | 0.365 | 0.447 | 0.260 | 0.448 | 0.254 | 0.255 | **0.452** |
| α=1.0 | 0.304 | 0.333 | 0.329 | 0.369 | 0.419 | 0.438 | 0.458 | 0.317 | **0.442** |
| α=0.5 | 0.252 | 0.270 | 0.282 | 0.298 | 0.360 | **0.387** | 0.352 | 0.296 | 0.364 |
| α=0.1 | 0.135 | 0.178 | 0.166 | 0.133 | 0.259 | **0.272** | 0.221 | 0.200 | 0.266 |

`c4_drift_aware` is the single best method at IID (0.452, narrowly ahead of
Multi-Krum's 0.448), second-best at α=1.0 (0.442 vs Multi-Krum's 0.438 —
effectively tied), and close behind Multi-Krum at α=0.5/α=0.1 (0.364 vs
0.387; 0.266 vs 0.272). **Across all four partitions, `c4_drift_aware`
tracks Multi-Krum closely** (within 2.3 percentage points in every case) —
this is the most precise statement supportable by this data: *on par with*
the strongest existing robust aggregator, not a decisive win over it.

## large_norm-specific (catastrophic-safety-relevant) comparison

| defense | divergence rate | mean final accuracy (large_norm only) |
|---|---|---|
| FedAvg | 100% | 0.086 |
| Sign Consensus | 100% | 0.086 |
| Cosine | 75% | 0.196 |
| Median | 0% | 0.323 |
| Norm | 0% | 0.342 |
| Multi-Krum | 0% | 0.390 |
| c3_static | 0% | 0.356 |
| c4_static | 0% | 0.299 |
| **c4_drift_aware** | **0%** | **0.396** |

`c4_drift_aware` has the single highest `large_norm` accuracy of every
method tested, single-signal or Combined — narrowly ahead of Multi-Krum
(0.396 vs 0.390) — while matching the zero-divergence group exactly.

## Fairness notes (explicit, per the prompt's requirement)

Median and Multi-Krum are robust aggregators, not detectors — no TPR/FPR is
reported for them here or anywhere in this comparison; they are compared
only on model utility, divergence, and (identically-protocoled) runtime,
consistent with the rest of this project's convention. `detector_tpr_fpr_table.csv`
(reused from `benchmark_v2_corrected`, unmodified) provided the single
-signal Cosine/Sign-Consensus TPR figures quoted in `REPORT.md` §8/§9
(directional_poisoning: Cosine TPR 0.889; sparse_coordinate_attack: Norm/Sign
TPR 1.000 each) for the direct per-attack retained-capability comparison.
