# PathMNIST single-method benchmark — report

Design: `docs/BENCHMARK_PROTOCOL.md` Addendum 2026-10-06 (f). Script:
`scripts/pathmnist_benchmark.py` (sweep) + `scripts/pathmnist_benchmark_aggregate.py`
(aggregation). Full run: 594 combos, 3,564 per-round rows, 151.4 minutes.
Raw data: `per_round_results.csv`, `calibration_results.csv`; derived tables:
`utility_table.csv`, `robustness_table.csv`, `detector_tpr_fpr_table.csv`,
`runtime_table.csv`.

**Headline: a real bug was found during analysis that invalidates the Norm
defense's numbers and the absolute accuracy numbers across the board. It is
reported here, not hidden. Combined Attestation and Fed-ISIC2019 were not
started, per instruction.**

---

## 1. What was run

- **Calibration (locked, Addendum (f)):** τ/ρ/κ for `norm`/`cosine`/`sign_consensus`,
  calibrated against a fixed canonical attack (`large_norm`) only, never the
  test-time attack. **Regime A** (condition-specific, per `(partition, alpha)`,
  12 calibrations) and **Regime B** (once, IID reference, frozen, 3
  calibrations, reused at α=1.0/0.5/0.1).
- **Attack suite:** `no_attack, large_norm, low_norm, full_sign_flip,
  directional_poisoning, sparse_coordinate_attack` (all 5 required + clean
  baseline).
- **Defenses:** `fedavg, norm, cosine, sign_consensus, median, multi_krum`
  (no Combined Attestation).
- **Partitions:** `iid, dirichlet(α=1.0), dirichlet(α=0.5), dirichlet(α=0.1)`.
- **Seeds:** 3 (`42,43,44`), identical partitions/attacker-assignment
  (first `n_mal=1` client index) reused across all 6 defenses for a given
  `(partition, attack, seed)` — apples-to-apples per instruction item 4.
- **Rounds:** 6 per run, attack active from round 2 (4 active rounds).

## 2. Critical finding: an uncontrolled-SGD-step confound, found during analysis

**The calibration pseudo-clients and the test-time clients train on very
different amounts of local data per round, which confounds every
norm-based comparison in this sweep.** Confirmed directly:

| | samples/client | batches/epoch (batch=256) |
|---|---|---|
| Calibration pseudo-client (val split ÷ 5) | 2,000 | **8** |
| IID test client (train pool ÷ 5) | 16,200 | **64** |
| Dirichlet(α=0.1) test clients (same round!) | 7,439 – 33,151 | **30 – 130** |

`local_epochs=1` means "one full pass over whatever data this client has" —
so a client with 8× more data performs 8× more SGD steps per round, which
mechanically produces a larger-magnitude weight delta, independent of any
attack. Two consequences, both confirmed in the data:

1. **Norm defense is non-functional across this entire sweep.** Regime A
   calibration (8-batch pseudo-clients) produces τ as small as `0.0129`
   (IID). Applied to 64-batch test clients, this rejects **everyone**:
   `detector_tpr_fpr_table.csv` shows `fp=47-48, tn=0-1` (FPR ≈ 96–100%)
   for **every** partition × attack combination, including `no_attack` at
   IID under Regime A — the condition Regime A is supposed to handle best.
   This is not a Non-IID finding; it reproduces even where calibration and
   test conditions nominally match, because the confound is about
   calibration-pool-size vs. train-pool-size, not about heterogeneity.
   **Every Norm-defense accuracy, TPR, and FPR number in this sweep should
   be treated as an artifact of this bug, not a finding.**
2. **Even within a single Dirichlet(0.1) test round**, clients differ in
   SGD-step count by over 4× (30 vs. 130 batches) purely from partition
   imbalance — meaning part of the "honest gradient norm variance under
   severe Non-IID" reported in the earlier characterization study
   (`results/pathmnist/characterization/FINDINGS.md`) is now suspect too:
   some of that variance is client-size-driven, not purely directional
   heterogeneity. The characterization study's **cosine** and
   **sign-consensus** findings are less affected (both are scale-invariant
   or coarsely quantized), but its **norm** statistics and the "norm
   sensitivity" framing should be re-read with this caveat.

**Root cause, precisely:** `benchmark/models/local_training.py::local_training_seeded`
runs `n_epochs` full passes over the client's *entire* local array; the
number of SGD steps is therefore proportional to that client's sample
count, not fixed. This was invisible in every prior PathMNIST run (STEP 5
baselines, characterization study) because none of those compared
magnitude-sensitive scores (raw norm thresholds) across datasets or
partitions of very different sizes within the same decision — the
characterization study reported norm *distributions* without thresholding
them, and STEP 5 only reported accuracy, which is far less sensitive to
this than a hard norm cutoff is.

**Not yet fixed.** Fixing it (e.g., a fixed number of local steps per round,
independent of client dataset size) is a real design decision — sampling
with replacement vs. truncating vs. re-deriving what "one local epoch"
should mean when comparing clients of very different sizes — and changes
the locked local-training primitive every other result in this project
depends on. Per "stop after the single-method PathMNIST benchmark and
report results," this is flagged here rather than silently patched and
rerun.

## 3. What is, and isn't, trustworthy from this run

| | Status |
|---|---|
| **Norm defense** (all accuracy/TPR/FPR) | **Not trustworthy — confirmed artifact of the bug above.** |
| **Absolute accuracy numbers** (all defenses) | **Not trustworthy for a second, independent reason:** 6 rounds is too short on PathMNIST. `no_attack` FedAvg reaches only 0.12–0.18 after 6 rounds (chance = 0.111) vs. 50.7%/45.9% at 15 rounds in STEP 5. The model barely leaves the "early training" regime, so attack-vs-no-attack and defense-vs-defense accuracy deltas are mostly noise around a near-chance baseline. |
| **Robustness table** (accuracy drop vs. no_attack) | Inherits the above — not usable as evidence. |
| Calibration table itself (τ/ρ/κ, validation FPR/TPR) | **Usable with the above caveat on absolute τ magnitude for Norm** — the *qualitative* pattern (Cosine Regime A's val TPR collapses from 1.0 at IID to 0.0 at every tested Non-IID level; Sign Consensus's small-sample calibration, 4 malicious observations, is noisy) is informative and consistent with the characterization study. |
| Cosine / Sign Consensus TPR/FPR patterns | **Directionally informative, not precise.** Both are less sensitive to the step-count confound than Norm (cosine is scale-normalized; sign only uses ±1), but calibration was still done on 8-batch pseudo-clients vs. 64-batch test clients, which can shift *direction*, not just magnitude, as training depth differs. Treat the qualitative story (Regime B transfer fails badly; Regime A struggles against `low_norm`/`sparse_coordinate_attack`) as a hypothesis the step-count fix should re-test, not a confirmed result. |
| **Runtime** | **Trustworthy** — unaffected by the above. Mean per-round runtime: cosine ≈2.76s (extra `g_ref` pass), all others ≈2.47–2.49s. Cosine's ~12% overhead vs. FedAvg is the only runtime differentiation; Norm/Sign/Median/Krum add negligible overhead over plain FedAvg at this model/data scale. |
| Four-way partition/seed/attacker-assignment identity across defenses | **Trustworthy** — confirmed by construction (same `build_or_load_partitions` call, same seed, per combo). |

## 4. The one genuinely load-bearing result: Regime A vs. Regime B, qualitatively

Despite the confound above, one pattern is robust to it because it shows up
as a **directional** effect, not a magnitude threshold: **Regime B
(IID-calibrated, frozen) cosine and norm thresholds reject the overwhelming
majority of honest clients under every tested Non-IID condition — including
when there is no attack at all.** E.g. `dirichlet_a0.1 / no_attack / cosine /
B_iid_transfer_frozen`: `fp=12, tn=0` → 100% of honest clients rejected,
every round, with zero attacker present. This is the exact failure mode the
honest-gradient characterization study's Q5 answer predicted, and it
reproduces here under an actual (if accuracy-degraded) federated run, not
just a static score distribution. It is also visible, independent of the
norm-scale bug, because it is about *relative* threshold-vs-score position
(IID-calibrated ρ sits above the entire Non-IID honest distribution), which
the step-count confound shifts but does not invent.

## 5. Explicit answers to items 5–7

5. **Model utility, robustness, TPR/FPR, runtime** — reported above; utility
   and robustness are **not usable** this round (round-budget + norm bug);
   TPR/FPR usable qualitatively only, pending the fix; runtime is usable and
   shows Cosine as the only defense with measurable overhead.
6. **IID vs. α=1.0/0.5/0.1 comparison** — the one trustworthy cross-partition
   result is §4: Regime B's honest-rejection catastrophe gets worse as α
   shrinks (more severe Non-IID → larger gap between the frozen IID
   threshold and the true honest distribution), consistent with the
   characterization study.
7. **Regime A and Regime B kept separate** — done throughout; every row in
   every output table carries an explicit `calibration_regime` column, and
   no table merges or averages across regimes.

## 6. Recommended next step (not yet taken)

Fix `local_training_seeded` (or add a variant) so the number of local SGD
steps per round is controlled rather than proportional to client dataset
size — the natural options are (a) a fixed step count per round sampled
with replacement from each client's local data, or (b) normalizing the
reported delta by step count before any norm-based comparison. This needs
to be decided and locked in `docs/BENCHMARK_PROTOCOL.md` before rerunning,
since it changes a primitive every prior PathMNIST number (STEP 5,
characterization study, this sweep) was built on. Also worth reconsidering
alongside it: a longer round budget (closer to STEP 5's 15 rounds) so
utility/robustness numbers move off the near-chance floor.

Stopping here, as instructed. Not proceeding to Fed-ISIC2019 or Combined
Attestation.
