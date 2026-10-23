# Baseline convergence analysis (R12)

Criterion: PATIENCE=15 rounds, ACC_EPS=0.005, LOSS_EPS=0.002/round (fixed before inspecting results).

| run_id | candidate | partition | seed | converged_at_round | final_val_acc | final_val_loss | final_test_acc |
|---|---|---|---|---|---|---|---|
| fedavg__dirichlet_a0.1__seed42 | fedavg | dirichlet_a0.1 | 42 | NOT CONVERGED | 0.4711 | 1.3950 | 0.4253 |
| fedavg__dirichlet_a0.1__seed43 | fedavg | dirichlet_a0.1 | 43 | NOT CONVERGED | 0.5169 | 1.2206 | 0.5872 |
| fedavg__dirichlet_a0.1__seed44 | fedavg | dirichlet_a0.1 | 44 | NOT CONVERGED | 0.5120 | 1.2140 | 0.5692 |
| fedavg__dirichlet_a0.5__seed42 | fedavg | dirichlet_a0.5 | 42 | NOT CONVERGED | 0.6042 | 1.0555 | 0.6305 |
| fedavg__dirichlet_a0.5__seed43 | fedavg | dirichlet_a0.5 | 43 | NOT CONVERGED | 0.6473 | 0.9592 | 0.6657 |
| fedavg__dirichlet_a0.5__seed44 | fedavg | dirichlet_a0.5 | 44 | NOT CONVERGED | 0.6149 | 1.0168 | 0.6542 |
| fedavg__dirichlet_a1.0__seed42 | fedavg | dirichlet_a1.0 | 42 | NOT CONVERGED | 0.6491 | 0.9724 | 0.6521 |
| fedavg__dirichlet_a1.0__seed43 | fedavg | dirichlet_a1.0 | 43 | NOT CONVERGED | 0.6586 | 0.9302 | 0.7107 |
| fedavg__dirichlet_a1.0__seed44 | fedavg | dirichlet_a1.0 | 44 | NOT CONVERGED | 0.6404 | 0.9851 | 0.6593 |
| fedavg__iid__seed42 | fedavg | iid | 42 | NOT CONVERGED | 0.6743 | 0.9139 | 0.6957 |
| fedavg__iid__seed43 | fedavg | iid | 43 | NOT CONVERGED | 0.6813 | 0.8968 | 0.6997 |
| fedavg__iid__seed44 | fedavg | iid | 44 | NOT CONVERGED | 0.6520 | 0.9419 | 0.6876 |
| fedavg_weighted__dirichlet_a0.1__seed42 | fedavg_weighted | dirichlet_a0.1 | 42 | NOT CONVERGED | 0.4866 | 1.3206 | 0.5455 |
| fedavg_weighted__dirichlet_a0.1__seed43 | fedavg_weighted | dirichlet_a0.1 | 43 | NOT CONVERGED | 0.5810 | 1.0797 | 0.6166 |
| fedavg_weighted__dirichlet_a0.1__seed44 | fedavg_weighted | dirichlet_a0.1 | 44 | NOT CONVERGED | 0.5275 | 1.1980 | 0.5719 |
| fedavg_weighted__dirichlet_a0.5__seed42 | fedavg_weighted | dirichlet_a0.5 | 42 | NOT CONVERGED | 0.6111 | 1.0575 | 0.6570 |
| fedavg_weighted__dirichlet_a0.5__seed43 | fedavg_weighted | dirichlet_a0.5 | 43 | NOT CONVERGED | 0.6493 | 0.9522 | 0.6675 |
| fedavg_weighted__dirichlet_a0.5__seed44 | fedavg_weighted | dirichlet_a0.5 | 44 | NOT CONVERGED | 0.6477 | 0.9552 | 0.6710 |
| fedavg_weighted__dirichlet_a1.0__seed42 | fedavg_weighted | dirichlet_a1.0 | 42 | NOT CONVERGED | 0.6686 | 0.9339 | 0.6947 |
| fedavg_weighted__dirichlet_a1.0__seed43 | fedavg_weighted | dirichlet_a1.0 | 43 | NOT CONVERGED | 0.6655 | 0.9152 | 0.7138 |
| fedavg_weighted__dirichlet_a1.0__seed44 | fedavg_weighted | dirichlet_a1.0 | 44 | NOT CONVERGED | 0.6437 | 0.9706 | 0.6606 |
| fedavg_weighted__iid__seed42 | fedavg_weighted | iid | 42 | NOT CONVERGED | 0.6729 | 0.9158 | 0.6933 |
| fedavg_weighted__iid__seed43 | fedavg_weighted | iid | 43 | NOT CONVERGED | 0.6821 | 0.8969 | 0.7010 |
| fedavg_weighted__iid__seed44 | fedavg_weighted | iid | 44 | NOT CONVERGED | 0.6530 | 0.9383 | 0.6883 |

## Per (candidate, partition) summary across seeds

| candidate | partition | n_converged/n_seeds | median_converged_round | mean_final_val_acc | std_final_val_acc |
|---|---|---|---|---|---|
| fedavg | dirichlet_a0.1 | 0/3 | n/a | 0.5000 | 0.0251 |
| fedavg | dirichlet_a0.5 | 0/3 | n/a | 0.6221 | 0.0225 |
| fedavg | dirichlet_a1.0 | 0/3 | n/a | 0.6494 | 0.0091 |
| fedavg | iid | 0/3 | n/a | 0.6692 | 0.0153 |
| fedavg_weighted | dirichlet_a0.1 | 0/3 | n/a | 0.5317 | 0.0473 |
| fedavg_weighted | dirichlet_a0.5 | 0/3 | n/a | 0.6360 | 0.0217 |
| fedavg_weighted | dirichlet_a1.0 | 0/3 | n/a | 0.6593 | 0.0136 |
| fedavg_weighted | iid | 0/3 | n/a | 0.6694 | 0.0149 |

## Additional analysis: late-training slope (rounds 140-149)

Strict criterion above returns 0/24 CONVERGED — confirmed by direct inspection
of the tail slope (val_accuracy change per round, rounds 140->149):

| run (representative) | partition | acc slope/round | loss slope/round |
|---|---|---|---|
| fedavg seed43 | dirichlet_a0.5 | +0.00044 | -0.00202 |
| fedavg seed43 | iid | +0.00199 | -0.00182 |
| fedavg seed43 | dirichlet_a0.1 | +0.00325 | -0.00924 |

**dirichlet_a0.1 is still improving FASTEST at round 150** (not slowest) --
counterintuitive but consistent with it starting from the worst position
(most label skew = hardest optimization problem = most room left to
improve). 150 rounds is closer to sufficient for IID/a1.0/a0.5 (slopes
near the practical noise floor) but clearly insufficient for a0.1.

## Weighted vs unweighted FedAvg (mean final val_accuracy +- std, 3 seeds)

| partition | unweighted | weighted |
|---|---|---|
| iid | 0.6692 +- 0.0153 | 0.6694 +- 0.0149 |
| dirichlet_a1.0 | 0.6494 +- 0.0091 | 0.6593 +- 0.0136 |
| dirichlet_a0.5 | 0.6221 +- 0.0225 | 0.6360 +- 0.0217 |
| dirichlet_a0.1 | 0.5000 +- 0.0251 | 0.5317 +- 0.0473 |

Weighted FedAvg is consistently slightly higher, with the largest (but
noisiest, highest-std) gap at the most severe non-IID level (a0.1) --
consistent with weighting mattering most when client dataset sizes vary
most (Dirichlet partitioning concentrates samples unevenly as alpha
shrinks). With only 3 seeds the gap is not clearly significant except
possibly at a0.1. No strong evidence to switch the full benchmark's
default aggregation away from unweighted FedAvg; recommend keeping
`fedavg` (unweighted) as the primary condition for comparability with
the fixed-K-step protocol's rationale, and correcting the report's
eq. (2.2)/(2.3) description to match, while optionally including
`fedavg_weighted` as a documented alternative.
