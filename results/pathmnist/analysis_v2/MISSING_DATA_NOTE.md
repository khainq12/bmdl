# What's missing from `benchmark_v2_corrected/` for STEPs 4/5/7, and the targeted traces that fill it

Per-round pooled data (`per_round_results.csv`) already contains, **no new
compute needed**:
- `tp/fp/tn/fn` per round → FPR/TPR **per round** are directly computable,
  not just pooled over the active-attack window.
- `accuracy`/`loss` per round and the `diverged` flag → the round a combo's
  loss first crosses the divergence threshold, and loss immediately
  before/after, are directly readable.
- `calibration_results.csv` → the frozen τ/ρ/κ per `(defense, partition,
  regime)`, constant across rounds, usable as a threshold line on any plot.

**Not present anywhere, because `pathmnist_benchmark_v2.py` only ever
logged the pooled accept/reject counts, never the individual scores that
produced them:**
- Per-client raw score (‖gᵢ‖, cos(gᵢ,g_ref), or sign-consensus score) by
  round — needed for STEP 4's honest-vs-malicious score-distribution plots.
- The specific malicious update's norm/score at the exact round a
  catastrophic miss occurred — needed for STEP 5.
- Three detectors' scores on the *same* generated update in the same round
  — no single real sweep run computes more than one detector's score per
  client (each defense's run is independent), so STEP 7's overlap
  (rejected-by-Norm-only vs Cosine-only vs Sign-only, etc.) cannot be
  derived from `benchmark_v2_corrected/` at all.

**Two small, targeted, instrumented traces fill this — not a rerun of the
594-combo matrix:**

- **Trace A** (`scripts/analysis_v2_trace_A.py`): reproduces each real
  per-defense trajectory for `{norm, cosine, sign_consensus} × 4 partitions
  × 6 attacks`, Regime A only, single seed (`42`), with the *identical*
  config (K=50, batch=128, 25 rounds, attack from round 10, same frozen
  thresholds from `calibration_results.csv`) as the real sweep — the only
  change is logging each client's raw score/accept-decision/delta-norm
  per round, in addition to accuracy/loss. Regime B is intentionally
  **not** traced in detail: its aggregate story (FPR≈1.0 from a static,
  badly-mismatched frozen threshold) is already fully characterized by
  existing per-round FPR, and a round-level score trace would not add
  qualitatively new information there — this is a scope decision, stated
  here rather than silently made.
- **Trace B** (`scripts/analysis_v2_trace_B.py`): one stable, non-diverging
  trajectory per `(partition, attack)` (24 combos), driven by Coordinate
  Median (the one aggregator with zero divergence across the entire real
  sweep — see `divergence_table.csv`), so the same sequence of honest +
  malicious updates can be scored by all three detectors *simultaneously*
  without the trajectory itself diverging differently depending on which
  single signal would have driven it. This is the control STEP 7's
  complementarity question needs; it is explicitly a diagnostic
  construction, not a reproduction of any real defense's own trajectory
  (that fidelity is Trace A's job, for STEP 4/5 specifically).

**Stated limitation carried into STEP 5:** the real sweep's divergence
safeguard halts local training once loss crosses the threshold, so
"whether recovery occurred" cannot be observed past that round in either
the real sweep or Trace A (both use the same safeguard, for consistency
with the reported numbers) — this question is answered as "not
observable from this data" rather than guessed at.
