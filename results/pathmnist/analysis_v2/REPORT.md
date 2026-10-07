# PathMNIST corrected-benchmark deep analysis — report

Scope: deep analysis of the already-complete corrected PathMNIST sweep
(`results/pathmnist/benchmark_v2_corrected/`), to understand it well
enough to justify a future Combined Attestation design. **No Combined
Attestation was implemented. No Fed-ISIC2019 work was started. The
594-combo matrix was not rerun.** Two small, targeted, instrumented
traces (72 + 24 combos, both far smaller than the 594-combo matrix) were
run to recover per-client score data the original sweep never logged —
see `MISSING_DATA_NOTE.md` for exactly what was missing and why.

---

## 1. Experimental evidence used

- `results/pathmnist/benchmark_v2_corrected/per_round_results.csv` — the
  corrected sweep itself: 594 combos, 14,850 rows, 3 seeds, verified
  complete in STEP 1.
- `calibration_results.csv`, `divergence_table.csv` from the same sweep.
- **Trace A** (`trace_A_client_scores.csv`, new, 9,000 rows): reproduces
  each real per-defense trajectory for `{norm, cosine, sign_consensus} ×
  4 partitions × 6 attacks`, Regime A, seed=42, identical config to the
  real sweep, with per-client score/accept/delta-norm logged per round.
- **Trace B** (`trace_B_complementarity.csv`, new, 3,000 rows): one
  Coordinate-Median-driven (never-diverging) trajectory per `(partition,
  attack)`, all three detectors scored simultaneously on the same
  updates, for the complementarity analysis.
- The preserved, invalid v1 diagnostic sweep
  (`benchmark_diagnostic_INVALID_v1/`) was **not read or used anywhere**
  in this analysis (confirmed in STEP 1).

## 2. Result integrity

**PASS** (`RESULT_INTEGRITY.md`). 594/594 combos, 14,850/14,850 rows, no
truncated trajectories, no unexplained NaNs, divergence flags internally
consistent with their own trigger condition, old sweep confirmed
untouched. One labeling nuance found and documented (not a defect): the
`no_attack` condition's designated "malicious" client-slot is still
tracked in `tp/fp/tn/fn` bookkeeping even though its update is unmodified
— `no_attack` TPR is therefore not a meaningful detection-quality number
(nothing is being detected); `no_attack` FPR is the meaningful one and is
used throughout this report.

## 3. Overall attack x defense comparison

(`attack_defense_matrix.csv`/`.md`, Figures 1–2.) Clean-utility ordering
(`no_attack`) is **iid > α=1.0 > α=0.5 > α=0.1** for every defense except
Norm, whose IID cell is dragged down by elevated FPR (§5). Norm, Median,
and Multi-Krum **never diverge** anywhere in the 594-combo matrix.
FedAvg, Cosine, and Sign Consensus do, but in different, non-overlapping
conditions (Figure 2): FedAvg always under `large_norm`; Cosine only
outside IID under `large_norm` (plus 2 rare `sparse_coordinate_attack`
seeds); Sign Consensus at **every** partition including IID under
`large_norm` (plus 1 rare non-`large_norm` seed).

## 4. Effect of Non-IID heterogeneity

Per-defense progressions (IID→α=1.0→α=0.5→α=0.1), Regime A:

- **Clean utility**: declines monotonically for every defense (e.g.
  FedAvg 0.442→0.450→0.409→0.196).
- **Attack robustness**: declines for directional attacks across all
  detectors, but Norm/`sparse_coordinate_attack` and Sign/`sparse_coordinate_attack`
  stay at TPR=1.0 regardless of heterogeneity — robustness loss is
  attack-specific, not universal.
- **Detector TPR**: Cosine's directional-attack TPR degrades gently
  (1.00→0.76); Sign's degrades steeply (1.00→0.00 by α=0.5); Norm's
  `large_norm`/`sparse` TPR is flat at 1.00 throughout; Cosine's and
  Sign's `large_norm` TPR is 0 outside IID (Cosine) or everywhere (Sign).
- **Detector FPR**: Norm's `no_attack` FPR *improves* with heterogeneity
  (0.63→0.22, the opposite of intuition — see §6 for why); Cosine/Sign's
  stay low (≤0.03) in Regime A throughout.
- **Which methods become unreliable as heterogeneity increases**: Cosine
  and Sign Consensus both lose `large_norm` detection entirely outside
  IID (Sign loses it even at IID); Sign additionally loses directional
  -attack detection by α=0.5. Norm does not lose any detection capability
  it had, but its FPR problem (already present at IID) does not get
  worse with heterogeneity.

## 5. Threshold-transfer results (Regime A vs Regime B)

Kept strictly separate throughout (every table/figure carries a
`calibration_regime` column; Figure 4 plots them side by side).
**IID-calibrated, frozen thresholds (Regime B) do not transfer**: Norm
and Cosine both show `no_attack` FPR ≈0.99–1.00 at every tested Non-IID
level — rejecting essentially every honest client, attack or no attack.
Sign Consensus transfers far better (FPR 0.003→0.085) — the one clearly
differentiated result between detectors on this specific question. This
confirms, under an actual 25-round federated run (not just the earlier
static score-distribution study), the honest-gradient characterization
study's prediction that a single global/IID-derived threshold is not
safe to assume generalizes.

## 6. Temporal threshold drift

(`step4_temporal_drift_diagnosis.md`, Figures in `temporal_*.png`.) The
dominant cause differs by defense and sometimes by partition for the same
defense — there is no single answer:

- **Norm at IID**: (B) temporal drift dominates. Honest score follows a
  U-shape (dips then climbs past threshold by round ~17), and the FPR
  climb from 0.33→0.67 happens **before the attack even starts** (rounds
  7-9) — a pure training-dynamics effect, not an attack interaction.
- **Norm at Non-IID**: (C) heterogeneity sets a wide initial two-cluster
  score spread; FPR then **stabilizes** rather than climbing further —
  explaining why Norm's FPR is paradoxically lower at severe Non-IID.
- **Cosine**: (D) score overlap, directly caused by (C) — Non-IID widens
  the honest cosine band wide enough to bracket the ≈0 score an isotropic
  `large_norm` vector produces by construction; at IID the same ≈0 attack
  score is far outside the tight honest band and trivially caught.
- **Sign Consensus**: (D), but as *inversion* rather than overlap — the
  malicious `large_norm` score is the **highest of all 5 clients, every
  round**, strictly above the honest range, not interleaved with it.

## 7. Catastrophic missed-detection analysis

(`step5_catastrophic_miss_analysis.md`.) **OBSERVED**: `P(divergence | 0
malicious accepted) = 0.0000` (n=156, exact) — a miss is always necessary.
`P(divergence | ≥1 malicious accepted) = 0.2138` overall, but this
collapses into a sharp attack-magnitude split: **`large_norm`:
P=1.000 (n=30)** — one miss is certain death, confirmed identically at
round 10 across all 3 seeds for every diverging combo. **Bounded-magnitude
attacks (`full_sign_flip`, `directional_poisoning`): P≈0.03**, and when
divergence does happen it requires **7–8 consecutive** undetected
acceptances, not one. `low_norm`: never divergence-relevant (not a
poisoning attack). **The hypothesis "a single false negative is
catastrophic" is TRUE for unbounded-magnitude attacks and FALSE (sustained
low recall matters instead) for bounded ones** — this is an observed
split, not an assumption.

## 8. Failure modes of each defense

(`step6_failure_mode_table.md`.) Summary: Norm's strength is
magnitude-bounded attacks (`large_norm`, `sparse_coordinate_attack`,
TPR=1.0 at every partition for both); its failure is temporal-drift-driven
FPR and blindness to `low_norm`. Cosine's strength is directional attacks,
degrading gently with heterogeneity; its failure is total blindness to
`large_norm`/`sparse_coordinate_attack` outside IID. Sign Consensus's
strength is uniquely robust `sparse_coordinate_attack` detection
(TPR=1.0 at every partition); its failure is total blindness to
`large_norm` even at IID and the steepest heterogeneity-driven collapse
on directional attacks of the three. Median/Multi-Krum never diverge and
degrade gracefully under every attack, but provide no detection
information at all.

## 9. Complementarity of Norm, Cosine, and Sign Consensus

(`step7_complementarity_analysis.md`, `figures/complementarity_overlap.png`.)
Using Trace B (all three signals scored on the identical, non-diverging,
median-driven update sequence), **now validated across all 3 seeds
(42/43/44) — see "Addendum" below**: **each signal has a distinct
required attack category, and the three required categories do not
overlap.** Norm ↔ `large_norm` (75% Norm-alone, identical in every seed).
Cosine ↔ directional attacks (Norm alone is never independently
sufficient there — always co-occurs with Cosine, in every seed). Sign
jointly with Norm ↔ `sparse_coordinate_attack` (Cosine contributes
**zero** rejections there, identical in every seed). On the honest side,
Norm accounts for the large majority of false rejections (16.1% of
honest client-rounds alone, 3-seed pooled, vs. Cosine's 1.2% and Sign's
3.1%) — relevant to how any future combination rule should weigh the
three.

**Addendum (2026-10-06, external review):** the user consulted a second
AI (ChatGPT) as an independent reviewer of this analysis. It endorsed the
overall conclusion but specifically asked whether the complementarity
percentages — originally measured on seed=42 only — were seed-specific
before being used to justify a Combined Attestation design. Seeds 43 and
44 were added in response (72 more Trace B combos, pooled to 9,000 rows
total). Result: the three attack-specific "required signal" findings
above (`large_norm`→Norm, `sparse_coordinate_attack`→Norm+Sign,
directional attacks→Cosine-necessary) are **exactly seed-invariant** —
identical composition in all 3 seeds individually, not just on average.
The honest-side numbers shifted moderately on pooling (e.g.
`accepted_by_all` was 85.1% at seed=42 alone, 79.6% pooled, range 75.6–
85.1% across seeds) — reported as found, not smoothed into the original
draft. Full detail: `step7_complementarity_analysis.md`'s "Multi-seed
consistency check" section, `step7_per_seed_consistency.csv`.

## 10. Implications for Combined Gradient Attestation

**This section is the input to a future design decision, not the
decision itself.** The complementarity evidence (§9) supports combining
signals: no single signal covers the attack suite, and the three
"required" categories are genuinely disjoint, not redundant measurements.
But three cautions are equally load-bearing evidence, not footnotes:

1. **Norm's elevated FPR (§6) would be inherited by any rule requiring
   only one signal to reject** (an OR-of-rejections combination) — the
   asymmetry between Norm's noisiness and Cosine/Sign's quietness on
   honest updates needs to be designed for explicitly, not averaged away.
2. **A single miss is only catastrophic for unbounded-magnitude attacks
   (§7)** — a Combined rule's main measurable benefit may be less about
   improving average TPR and more about ensuring the magnitude-sensitive
   signal (Norm, currently the only one with TPR=1.0 against `large_norm`
   at every partition) is never the single point of failure for that one
   attack category.
3. **Trace B's complementarity picture was deliberately measured on a
   never-diverging trajectory** — it says nothing about whether a Combined
   rule would itself inherit the single-miss catastrophic failure mode
   found in §7 for Cosine/Sign under `large_norm`. That needs a dedicated
   follow-up once a specific Combined rule is chosen and implemented.

## 11. Limitations

- **Trace A** (STEP 4/5: temporal drift, catastrophic-miss detail) still
  uses a **single seed (42)** — the qualitative shapes (U-curve, cluster
  separation, score ranking, round-10 miss timing) are distinctive
  enough to trust, and the round-10-miss timing was independently
  confirmed across all 3 seeds using the real sweep's own pooled data
  (§7), but the exact score values plotted in `temporal_*.png` are
  seed=42-specific. **Trace B** (STEP 7: complementarity) was extended to
  all 3 seeds after external review (§9 Addendum) — its percentages are
  now pooled and seed-checked, not single-seed. The real sweep's 3-seed
  `per_round_results.csv` remains the authority for any FPR/TPR/accuracy
  percentage not sourced from Trace B.
- "Whether recovery occurred" after a catastrophic miss is **not
  observable** in this data — the divergence safeguard halts training at
  the trigger round in both the real sweep and Trace A, by design.
- Regime B was not traced for temporal score drift at round-level detail
  (§`MISSING_DATA_NOTE.md`) — its aggregate story (FPR≈1.0 throughout) is
  well-characterized without it, but this is a stated scope choice, not
  an oversight.
- The overfitting hypothesis offered for Norm's IID U-shape (§6/STEP 4)
  is explicitly unconfirmed — no independent train/test-gap measurement
  was collected to test it.
- Trace B's median-driven trajectory is a deliberate experimental control
  for complementarity, not a claim about what any real three-signal
  Combined rule's trajectory would look like.

## 12. What should be tested next

- A dedicated ablation on Sign Consensus's majority-vote construction
  (excluding each client's own vote from its own comparison) to test the
  self-inclusion hypothesis from §6/STEP 4 for why its `large_norm` score
  exceeds the honest range.
- Independent train/test loss tracking during a clean IID run, to test
  the overfitting hypothesis for Norm's U-shaped score trajectory.
- Once a specific Combined Attestation rule is designed (next phase, not
  this one): re-run the STEP 5-style catastrophic-miss analysis *on the
  Combined rule itself*, since Trace B's complementarity result does not
  establish whether a combined signal inherits or avoids the single-miss
  failure mode.
- A multi-seed version of Traces A/B if any STEP 6/7 finding needs
  tighter confidence intervals before being load-bearing for a design
  decision (current n=1 seed per trace is sufficient for the qualitative
  patterns reported here, not for precise percentages).

---

## Four questions, answered

**1. What does Norm do well, and where does it fail?**
Norm is the only signal with **perfect (TPR=1.0) detection of
magnitude-based attacks at every tested heterogeneity level** —
`large_norm` and `sparse_coordinate_attack` alike — and it is the only
detector that **never once caused catastrophic model divergence** across
the entire 594-combo matrix. It fails on two fronts: it is **structurally
blind to `low_norm`** (TPR=0.0 everywhere — a magnitude-only rule cannot
catch an attack defined by *reducing* magnitude), and it carries a
**temporal-drift-driven false-positive problem**, worst at IID (FPR=0.63),
caused by the honest score's own U-shaped trajectory over 25 rounds
drifting past a threshold calibrated from a short window that never saw
the later climb — not by Non-IID (Non-IID heterogeneity actually makes
Norm's FPR *better*, the reverse of every other defense studied).

**2. What does Cosine do well, and where does it fail?**
Cosine is the strongest detector for **global directional attacks**
(`directional_poisoning`, `full_sign_flip`), with the gentlest
heterogeneity-driven degradation of the three detectors (TPR 1.00→0.76
from IID to α=0.1). It fails **completely outside IID against
`large_norm` and `sparse_coordinate_attack`** (TPR=0.0) — not because of
calibration error, but because severe Non-IID genuinely widens the honest
cosine-to-`g_ref` distribution until it brackets the near-zero score an
isotropic or sparse attack produces by construction. When it misses, the
consequence is severe: **100% divergence rate** whenever it fails to
detect `large_norm` outside IID.

**3. What does Sign Consensus do well, and where does it fail?**
Sign Consensus is the **only signal with heterogeneity-flat, perfect
detection of `sparse_coordinate_attack`** (TPR=1.0 at every partition,
jointly necessary with Norm — Cosine contributes nothing there) and has
the lowest honest-client false-positive rate of the three in Regime A.
It fails hardest of any defense studied on `large_norm`: **TPR=0.0 at
every partition including IID** — the malicious score doesn't just
overlap the honest range, it **exceeds** every honest client's score,
making threshold-based rejection structurally impossible without
rejecting honest clients first. It also shows the **steepest**
heterogeneity-driven collapse on directional attacks of the three
detectors (TPR 1.00→0.00 by α=0.5, versus Cosine's gentler 1.00→0.76).

**4. Is there empirical evidence that their failure modes are
complementary enough to justify building Combined Attestation?**
**Yes.** The complementarity trace (§9) shows each signal has a distinct,
non-overlapping "required" attack category — Norm for unbounded
magnitude, Cosine for global direction, Sign (jointly with Norm) for
sparse/local coordinate attacks — and no signal is even *involved* in
rejecting its blind-spot attack's updates, not just weaker at it. This is
closer to genuine complementary coverage than to three redundant
measurements of the same underlying signal. That said, the evidence also
surfaces two real design constraints a Combined rule must account for,
not just the opportunity: Norm's false-positive rate would contaminate
any simple OR-combination, and the catastrophic-single-miss failure mode
(§7) has only been characterized for the *individual* signals, not for
whatever Combined rule is eventually chosen — that is the next, separate
piece of work.
