# Result integrity check — `results/pathmnist/benchmark_v2_corrected/`

Verified against the design locked in `docs/BENCHMARK_PROTOCOL.md` Addenda
2026-10-07 (g)/(h)/(i) and the run's own `REPORT.md`. **Gate: PASS** — the
corrected result set is complete and internally consistent; analysis
proceeds on it.

## 1. Row/combo counts

- `per_round_results.csv`: **14,850 rows** = 594 combos × 25 rounds exactly.
  Every round value (0–24) has exactly 594 rows — no partial/truncated
  combos.
- **Combo breakdown matches the designed grid exactly**:
  - 3 detector defenses (`norm`, `cosine`, `sign_consensus`) × 4 partitions ×
    6 attacks × 3 seeds = **72 combos each** under `A_condition_specific_oracle`.
  - Same 3 defenses × 3 non-IID partitions (α=1.0/0.5/0.1; IID excluded by
    design, §Addendum (f): Regime B at IID is definitionally identical to
    Regime A) × 6 attacks × 3 seeds = **54 combos each** under
    `B_iid_transfer_frozen`.
  - 3 non-detector defenses (`fedavg`, `median`, `multi_krum`) × 4 partitions
    × 6 attacks × 3 seeds = **72 combos each** under `not_applicable`.
  - Total: `3×(72+54) + 3×72 = 378 + 216 = 594`. ✅ Matches exactly.
- `calibration_results.csv`: **15 rows** = 12 Regime A (3 detectors × 4
  partitions) + 3 Regime B (3 detectors × 1 IID reference). ✅ Matches.
- `divergence_table.csv`: **198 rows** = 594 combos ÷ 3 seeds (grouped
  across seeds, `n_seeds=3` per row). ✅ Matches.

## 2. Seeds, attacks, defenses, partitions, regimes — all present, nothing extra

- Seeds: `{42, 43, 44}` — exactly 3, every combo has all 3 (verified: 0
  combos with `!=3` distinct seeds).
- Attacks: `{no_attack, large_norm, low_norm, full_sign_flip,
  directional_poisoning, sparse_coordinate_attack}` — all 6 required,
  nothing extra.
- Defenses: `{fedavg, norm, cosine, sign_consensus, median, multi_krum}` —
  all 6 required, nothing extra. No Combined Attestation present (correct —
  it was not implemented).
- Partitions: `{iid, dirichlet_a1.0, dirichlet_a0.5, dirichlet_a0.1}` — all
  4 required.
- Calibration regimes: `{A_condition_specific_oracle,
  B_iid_transfer_frozen, not_applicable}` — exactly the three locked
  categories, correctly scoped (B absent for non-detectors; B absent for
  IID, by design).
- Every `(partition, attack, defense, regime, seed)` trajectory has
  **exactly 25 rows** (round 0–24) — 0 trajectories short or long.

## 3. Missing / NaN / corrupted data

- No NaN in `round, accuracy, loss, runtime_s, n_malicious, diverged,
  partition, seed, attack, defense, calibration_regime`.
- `tp, fp, tn, fn, threshold_used` are NaN for exactly **5,400** rows —
  this is correct and expected, not missing data: `216 non-detector combos
  × 25 rounds = 5,400`. Non-detector defenses (`fedavg`, `median`,
  `multi_krum`) produce no score/accept-mask by design (§5 of
  `BENCHMARK_PROTOCOL.md`: "Median and Multi-Krum are robust aggregators,
  not detectors"), so these fields are correctly absent, not corrupted.
- `accuracy` range: `[0.044, 0.520]` — sane for 9-class PathMNIST (chance =
  0.111); no out-of-range values.
- `loss`: non-diverged rows range `[1.49, 1.85×10¹¹]` — the upper end looks
  alarming in isolation but is explained in §4 below; it is not a defect.
  **Zero** diverged rows have `loss < 50` (the divergence threshold) — the
  divergence flag is internally consistent with its own trigger condition
  in every one of the 603 flagged rows.

## 4. Divergence flag consistency

603 rows flagged `diverged=True`, broken down by attack:

| attack | diverged rows |
|---|---|
| large_norm | 588 |
| full_sign_flip | 5 |
| sparse_coordinate_attack | 7 |
| directional_poisoning | 3 |
| low_norm | 0 |
| no_attack | 0 |

This matches `REPORT.md`'s §1 finding (large_norm dominates divergence) and
correctly shows **zero** divergence for `low_norm`/`no_attack`, as expected
— neither injects unbounded magnitude. The "non-diverged max loss =
1.8×10¹¹" from §3 is explained here: by design
(`docs/BENCHMARK_PROTOCOL.md` Addendum (i)), the **triggering round itself**
is recorded with `diverged=False` (it's the round where the huge loss is
first observed and evaluated); only the subsequent copy-forwarded rounds
are flagged `diverged=True`. This is intentional — it is exactly the round
needed for the STEP 5 catastrophic-miss analysis (identifying *when* the
fatal update was accepted) — not a labeling bug.

## 5. Attacker-assignment / partition consistency

`n_malicious` (count of clients flagged as the designated malicious client
that round) is `1` for 1,485 rows and `0` for 990 rows, **identically**
across all 6 attacks including `no_attack`. One methodological nuance
worth flagging explicitly (not a bug, but affects interpretation of
`no_attack` TPR):

- The malicious-client *assignment* (`cid < n_mal`, active from
  `ATTACK_FROM_ROUND=10`) is independent of which attack function is
  applied. Under `no_attack`, the designated client is still labeled
  "malicious" in `tp/fp/tn/fn` bookkeeping from round 10 onward, even
  though `apply_attack("no_attack", ...)` is a no-op and its update is
  statistically indistinguishable from any honest client's.
  **Consequence:** `no_attack` condition's `TPR` is not a meaningful
  detection-quality number (there is nothing to detect — any "catch" is
  the detector occasionally flagging an ordinary client that happens to be
  labeled); its `FPR` is very slightly conservative, since one of five
  clients per round ≥10 is excluded from the true-honest pool. This does
  not affect any `attack != no_attack` condition's TPR/FPR, and does not
  affect accuracy/loss/divergence for any condition. Flagged here for
  anyone reading `no_attack` TPR numbers downstream; STEP 2 onward will
  treat `no_attack` TPR as not meaningful and lead with `no_attack` FPR
  instead.

Partitions are reused unchanged from the persisted, disjointness-proven
files in `data/partitions/pathmnist/` (STEP 5) for every seed — not
regenerated per run — so "identical partitions ... across all defenses"
(this turn's requirement) holds by construction: `build_or_load_partitions`
is deterministic given `(partition, alpha, n_clients, server_ref_fraction,
seed)` and was called with the same arguments for every defense sharing a
given `(partition, alpha, seed)`.

## 6. Old diagnostic sweep stays separate

- No file under `benchmark_v2_corrected/` references or loads anything
  from `benchmark_diagnostic_INVALID_v1/`, and vice versa (checked by
  grep; the only matches are the two reports' own prose cross-references
  to each other's *names*, not data loads).
- `benchmark_diagnostic_INVALID_v1/*.csv` file modification times (Oct 5,
  prior session) predate this session — confirmed untouched.
- This analysis (STEPs 2 onward) reads exclusively from
  `benchmark_v2_corrected/`.

## Verdict

**PASS.** The corrected result set is complete (594/594 combos, 14,850/14,850
rows, no truncated trajectories), internally consistent (divergence flags
match their own trigger condition; combo breakdown matches the locked
design exactly), and cleanly separated from the preserved invalid sweep.
Proceeding to STEP 2.
