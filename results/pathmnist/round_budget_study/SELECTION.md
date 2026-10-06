# Round budget selection — rationale (locked)

Study: `scripts/pathmnist_round_budget_study.py`. Clean FedAvg only, no
attacks, no detector calibration, fixed K=50 SGD steps/client/round,
batch_size=128, 4 partition settings × 3 seeds, out to 25 rounds. Raw
curves: `curves.csv`; plot: `plot.png`.

## Per-round accuracy (mean over 3 seeds)

| round | iid | α=1.0 | α=0.5 | α=0.1 |
|---|---|---|---|---|
| 10 | 0.273 | 0.227 | 0.117 | 0.117 |
| 15 | 0.283 | 0.329 | 0.286 | 0.179 |
| 20 | 0.433 | 0.461 | 0.353 | 0.192 |
| 24 | 0.442 | 0.454 | 0.406 | 0.196 |

## Reading

- **IID and α=1.0** flatten clearly from ~round 18–20 onward (IID: std
  drops from 0.068 at round 15 to 0.008 at round 24 — a genuine plateau,
  not just a noisy mean). 15 rounds is clearly too early for either (both
  still mid-climb).
- **α=0.1** plateaus by ~round 14 (into a noisy flat band around
  0.19–0.21) — but at a much lower absolute level, consistent with the
  honest-gradient characterization study's finding that severe Non-IID
  honest clients barely form a coherent aggregate direction. More rounds
  do not meaningfully move this condition past round ~16.
- **α=0.5 is the outlier: still climbing at round 24** (0.406, the
  single highest point in its entire curve — not a plateau). Of the three
  candidate budgets, none fully stabilizes this condition, but 25 is
  closer than 20, which is closer than 15.

## Decision

**N_ROUNDS = 25**, the largest of the three candidate budgets offered,
because it is the only one where the two best-behaved conditions (IID,
α=1.0) are unambiguously plateaued and the other two (α=0.5, α=0.1) are
each closer to their eventual level than at 15 or 20 rounds — even though
α=0.5 has not fully converged even at 25. This is reported as a known,
honest limitation of this round budget for that one condition, not
papered over: comparisons involving α=0.5 downstream should be read with
the understanding that its honest baseline may still be a few points
below where it would eventually settle with a longer budget.

**ATTACK_FROM_ROUND = 10**, chosen so the model has moved past the
near-flat early-training region (rounds 0–9, where even IID is only at
0.27 and α=0.1 has not started moving at all) before any attack is
introduced, leaving 15 rounds (10–24) of active-attack exposure — far
more than the invalid v1 sweep's 4 rounds, giving both accuracy
-degradation and detector TPR/FPR measurements room to show a real signal
rather than early-training noise.

These two numbers (`N_ROUNDS=25`, `ATTACK_FROM_ROUND=10`) are now frozen
for the calibration diagnostics gate, the pilot, and the full corrected
sweep — selected from this clean-FedAvg-only curve data, before any
detector, threshold, or attack result existed, per
`docs/BENCHMARK_PROTOCOL.md` Addendum 2026-10-07 (h).
