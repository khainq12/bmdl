# STEP 4 — Temporal threshold drift: diagnosis

Data: Trace A (`trace_A_client_scores.csv`, raw per-client scores, seed=42)
for score-vs-round plots; real sweep's `per_round_results.csv` (3 seeds,
pooled tp/fp/tn/fn) for FPR/TPR-vs-round. Plots in `figures/temporal_*.png`
(12 defense×partition combinations generated; `large_norm` shown as the
primary attack since it is where every defense's behavior is most
diagnostic — the divergence-causing attack). Summary table:
`step4_temporal_drift_summary.csv`.

Four candidate explanations were given: (A) calibration/test distribution
mismatch, (B) temporal drift during FL training, (C) Non-IID heterogeneity,
(D) attack/benign score overlap. **The evidence shows the dominant cause is
different for each defense, and sometimes different across partitions for
the same defense** — there is no single answer.

## Norm — dominant cause differs by partition: (B) at IID, (C)+(B) at Non-IID

**OBSERVED** (`temporal_norm_iid_large_norm.png`): the honest norm score at
IID follows a clear **U-shape** over the 25-round trajectory — starts
~0.055, dips to a minimum ~0.03 by round ~7-9, then climbs steadily past
the threshold (0.065) by round ~17-18, reaching ~0.07 by round 24. FPR
tracks this shape almost exactly: 0.47→0.33 (rounds 0-7, dip) →0.67 (rounds
8-9, **before the attack starts at round 10**) →1.00 (round 18 onward,
after the honest score crosses the static threshold). **The FPR climb from
0.33 to 0.67 happens entirely in the clean pre-attack window** — this is
not an attack-interaction effect.

**OBSERVED** (`temporal_norm_dirichlet_a0.1_large_norm.png`): honest scores
form two visible clusters (~0.65-0.80 and ~0.1-0.3) from round 0 onward —
a Non-IID heterogeneity signature, not present at IID. FPR **decreases**
monotonically from 0.67 (round 0) to ~0.13-0.17 by round 6, then stays
**flat** for the remaining 19 rounds — no late climb like IID shows.

**INTERPRETATION:** at IID, (A) calibration mismatch is largely resolved
by the fixed-K fix (round-0 honest scores already sit close to the
threshold, not off by an order of magnitude as in the invalid v1 sweep) —
but (B) temporal drift, specifically a non-monotonic U-shaped honest-score
trajectory that a short 5-round calibration window cannot see, is the
clear remaining driver of the elevated IID FPR. At severe Non-IID, (C)
heterogeneity sets a wide initial score spread, but the dominant later
-round behavior is stabilization, not further drift — Non-IID's effect
here is mostly on the *starting* FPR, not a growing one.

**HYPOTHESIS** (not proven by this trace): the late-stage IID norm climb
plausibly reflects the model beginning to fit per-client idiosyncrasies
more strongly after ~1,250 cumulative local SGD steps (25 rounds × 50
steps) over a comparatively small per-client IID partition (~16k samples),
rather than continuing to reduce a shared/easy loss component — i.e. a
mild overfitting-driven gradient-magnitude regrowth. This is consistent
with, but not confirmed by, the shape observed; no independent evidence
(e.g. train/test loss gap) was collected here to confirm it.

## Cosine — dominant cause: (D), directly caused by (C)

**OBSERVED** (`temporal_cosine_dirichlet_a0.1_large_norm.png`): once the
attack starts, the 4 honest clients' cosine scores settle into four nearly
flat, stable horizontal bands (≈0.46, 0.23, 0.10, −0.06) — each client has
a characteristic, stable alignment with `g_ref` once local training
stabilizes. The malicious client's `large_norm` score sits at **≈0.00,
interleaved directly between the honest bands at 0.10 and −0.06** —
statistically indistinguishable from, and literally positioned inside,
the honest score range.

**INTERPRETATION:** this is a direct, visible case of (D) attack/benign
score overlap. It is not a calibration artifact (scores are flat/stable,
not drifting) and not really a *drift* phenomenon (B) — it is that (C)
Non-IID heterogeneity spreads the honest cosine distribution wide enough
(here, −0.06 to 0.46) that it **brackets** the ≈0 score an isotropic
random vector produces by construction (an isotropic Gaussian in ~207k
dimensions has expected cosine ≈0 with any fixed direction, by
concentration of measure). Under IID, where honest cosine is tightly
clustered near 0.96, this same ≈0 attack score is far outside the honest
band and trivially caught (TPR=1.0 at IID, confirmed in the summary
table). **C causes D here** — they are not independent explanations for
this defense.

## Sign Consensus — dominant cause: (D), and the malicious score is *higher* than honest, not just overlapping

**OBSERVED** (`temporal_sign_consensus_iid_large_norm.png`): once the
attack starts, the malicious client's score (≈0.675) is **the highest of
all five clients, every round** — strictly above the four honest clients'
scores (≈0.61-0.645). The threshold (0.532) sits below all of them, so
nothing is rejected (TPR=0 at every partition, confirmed in the summary
table, including IID).

**INTERPRETATION:** this is not score overlap in the usual sense (scores
interleaved) — the attack score **exceeds** the honest range entirely,
making rejection-by-threshold structurally impossible without also
rejecting every honest client first. This is still categorized as (D)
(the signal does not separate attacker from honest) but the mechanism is
distinct from Cosine's "attack lands in the middle" case.

**HYPOTHESIS** (not proven by this trace): sign-consensus's majority vote
is computed including the attacker's own vote; an isotropic random sign
vector may coincidentally agree with a self-inclusive majority at a rate
at least as high as correlated-but-imperfect honest votes do under this
model/data, especially early in training when per-client gradients are
less mutually aligned than the majority-defining aggregate. This would
need a dedicated ablation (e.g., majority computed excluding each client's
own vote) to confirm — not run here.

## Summary table (large_norm, Regime A)

| defense | partition | threshold | honest score round 0→24 | FPR round 0→24 | TPR round 10→24 | dominant cause |
|---|---|---|---|---|---|---|
| norm | iid | 0.065 | 0.055→0.070 (U-shape, dips then climbs) | 0.47→1.00 | 1.0→1.0 | (B) temporal drift |
| norm | α=1.0 | 0.304 | 0.227→0.187 | 0.47→0.25 | 1.0→1.0 | (C) heterogeneity (initial), stabilizes |
| norm | α=0.5 | 0.390 | 0.473→0.507 | 0.60→0.25 | 1.0→1.0 | (C) heterogeneity (initial), stabilizes |
| norm | α=0.1 | 0.547 | 0.609→0.462 | 0.67→0.17 | 1.0→1.0 | (C) heterogeneity (initial), stabilizes |
| cosine | iid | 0.651 | 0.960→0.983 | 0.00→0.00 | 1.0→1.0 | n/a (well-separated) |
| cosine | α=1.0 | −0.514 | 0.235→0.177 | 0.00→0.00 | 0.0→0.0 | (D), caused by (C) |
| cosine | α=0.5 | −0.437 | 0.151→0.130 | 0.00→0.00 | 0.0→0.0 | (D), caused by (C) |
| cosine | α=0.1 | −0.544 | 0.078→0.184 | 0.00→0.08 | 0.0→0.0 | (D), caused by (C) |
| sign_consensus | iid | 0.532 | 0.863→0.628 | 0.00→0.00 | 0.0→0.0 | (D) — attack score *exceeds* honest |
| sign_consensus | α=1.0 | 0.416 | 0.661→0.450 | 0.00→0.17 | 0.0→0.0 | (D), with some (B) FPR drift |
| sign_consensus | α=0.5 | 0.400 | 0.606→0.459 | 0.00→0.17 | 0.0→0.0 | (D), with some (B) FPR drift |
| sign_consensus | α=0.1 | 0.378 | 0.586→0.406 | 0.00→0.42 | 0.0→0.0 | (D), with some (B) FPR drift |

## What this does NOT claim

- These mechanisms are read from a **single seed's** trace (seed=42) — the
  qualitative shapes (U-curve, cluster separation, score ranking) are
  distinctive enough to trust as real phenomena, but exact FPR/TPR
  percentages at any given round should be read from the 3-seed pooled
  `per_round_results.csv` line, not the single-seed scatter.
- No adaptive threshold is proposed or implied by any of the above —
  this is diagnosis only, as instructed.
- The overfitting hypothesis for Norm's IID U-shape is explicitly labeled
  a hypothesis, not confirmed.
