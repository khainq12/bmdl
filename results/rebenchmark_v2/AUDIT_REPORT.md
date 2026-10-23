# R12 audit report

Scope: training loop, aggregation, attacks, defenses, evaluation, data
partitions, metrics. Model sync, data leakage, reproducibility, metric
correctness. 17 pytest unit tests added (`tests/test_benchmark_core.py`,
all passing).

## Findings

| # | Area | Finding | Status |
|---|---|---|---|
| 1 | FedAvg aggregation | `benchmark/defenses/fedavg.py` does unweighted `mean(deltas)`; report's eq. (2.2)/(2.3) describes classic n_i/n-weighted FedAvg. Confirmed deliberate given the fixed-K-step local training design (every client does the same number of gradient steps regardless of dataset size, specifically to make delta magnitude comparable across client sizes — re-weighting by n_i afterward would partially undo that). | Added `FedAvgWeighted` class (`ctx['client_weights']`) for explicit comparison, kept `FedAvg` unchanged. Report math to be corrected in Chapter 2 once baseline comparison confirms which variant the full benchmark should use. |
| 2 | Model synchronization | Every round: `gw = model.get_weights().copy()`; every client does `local.set_weights(gw.copy())` before training. Confirmed by code read (`pathmnist_benchmark_v2.py:150-161`) and by unit test `test_all_clients_start_from_identical_global_weights_each_round`. | VERIFIED, no bug. |
| 3 | Gradient/update definition | `delta = model.get_weights() - initial_weights`, consistent in every script and in `local_training_fixed_steps.py`. Covered by `test_fixed_k_steps_delta_is_weights_after_minus_before`. | VERIFIED, no bug. |
| 4 | Data leakage | `results/pathmnist/characterization/calibration_disjointness_proof.csv` (pre-existing) confirms zero overlap between server_ref, client training data, calibration data, and test data for every partition/seed checked. | VERIFIED via existing artifact, no new leakage introduced by R12 additions (train-eval subset for the new baseline study is drawn from the training pool itself, by definition — not a leakage case). |
| 5 | Reproducibility | Every RNG is seeded (`np.random.default_rng(seed + offset)` pattern); client partitions persisted to disk keyed by `(partition, alpha, n_clients, seed)`, reused bit-identically across experiments touching the same seed. | VERIFIED, no bug. |
| 6 | TPR/FPR metric correctness | Formula uses TP/(TP+FN) and FP/(FP+TN) with the correct per-role denominators (not total client-rounds). Covered by `test_tpr_fpr_formula_matches_hand_count` and `test_tpr_fpr_denominator_is_role_count_not_total_rounds` (guards specifically against the classic uneven-split denominator bug). | VERIFIED, no bug. |
| 7 | K-step local training | Exactly K=50 `train_step` calls per client per round regardless of dataset size, cycling via reshuffled passes. Covered by `test_fixed_k_steps_calls_train_step_exactly_k_times` and `test_fixed_k_steps_cycles_when_client_smaller_than_one_pass`. | VERIFIED, no bug. |
| 8 | directional_poisoning / full_sign_flip | Confirmed (locked into regression test `test_directional_poisoning_default_equals_full_sign_flip`) that without `target_norm`, directional_poisoning is bit-identical to full_sign_flip — matches the R10/R11 finding, now has a permanent regression guard. `target_norm` behavior itself verified exact (`test_directional_poisoning_target_norm_is_exact`). | VERIFIED, already fixed in report (R10/R11); now also guarded by test. |
| 9 | low_norm / Cosine / Sign invariance | cos(ηg,g_ref)=cos(g,g_ref) and sgn(ηg)=sgn(g) for η>0, confirmed by `test_low_norm_preserves_cosine_and_sign` and `test_sign_full_flip_and_directional_both_invert_sign_consensus_identically`. | VERIFIED, matches closed-form proof already in the report. |

## No confirmed bugs requiring a code fix beyond #1

Items 2–9 were audited and found correct; only item 1 (FedAvg weighting)
represented a genuine code-vs-documentation mismatch, resolved by adding
an explicit alternative (`fedavg_weighted`) rather than changing existing
behavior, so the R12 baseline convergence study can measure which variant
the full benchmark should standardize on — not assumed in either
direction.

## Files changed

- `benchmark/defenses/fedavg.py` — added `FedAvgWeighted`.
- `benchmark/defenses/__init__.py` — registered `fedavg_weighted`.
- `tests/test_benchmark_core.py` — new, 17 tests, all passing (`pytest tests/ -v`).
- `scripts/rebenchmark_v2_baseline_convergence.py` — new baseline convergence study.
- `scripts/rebenchmark_v2_analyze_convergence.py` — new convergence-criterion analysis.

No existing file under `results/pathmnist/` or `results/fed_isic2019/` was
modified or deleted.
