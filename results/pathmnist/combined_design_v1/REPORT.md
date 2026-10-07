# Combined Gradient Attestation — Design + Offline Replay (results/pathmnist/combined_design_v1/)

**Scope of this report, stated up front:** this is an **offline replay** of
candidate Combined decision rules against the existing 9,000-row Trace B
(`results/pathmnist/analysis_v2/trace_B_complementarity_pooled3seeds.csv`,
seeds 42/43/44, Coordinate-Median-driven stable trajectories). **No new FL
model was trained. The 594-combination PathMNIST matrix was not rerun.
Fed-ISIC2019 was not touched.** See "Critical scientific limitation" (§14)
for exactly what this analysis can and cannot claim.

---

## 1. Motivation from single-signal failure modes

From `results/pathmnist/analysis_v2/REPORT.md` and its STEP 4/5/7 findings:

- **Norm**: TPR=1.0 on `large_norm`/`sparse_coordinate_attack` at every
  Non-IID level, never diverges the model on these attacks — but a single
  accepted `large_norm` update diverges the model with probability 1.000
  (n=30, identical across 3 seeds), and Norm's static τ accrues honest FPR
  via temporal drift (worst at IID), independent of Non-IID.
- **Cosine**: the only signal that ever independently rejects
  `directional_poisoning`/`full_sign_flip` (STEP 7, seed-invariant
  qualitative finding), but degrades under Non-IID and is structurally blind
  to `large_norm`/`sparse_coordinate_attack` outside IID.
- **Sign Consensus**: TPR=1.0 on `sparse_coordinate_attack` at every Non-IID
  level, cleanest honest-side signal (pooled FPR 3.1%) — but its large_norm
  score was found to be *inverted* (malicious score higher than honest) at
  IID, so no threshold on Sign alone can catch `large_norm`.
- Each signal has a "home turf" attack with no overlap with the others'
  (STEP 7, confirmed seed-invariant for `large_norm`/`sparse_coordinate_attack`/
  `low_norm`); honest-side cost is asymmetric (Norm 16.1% pooled FPR vs.
  Cosine 1.2% / Sign 3.1%). This is the empirical basis for attempting a
  Combined rule, and the reason a naive OR was flagged in advance as
  inheriting Norm's FPR.

## 2. Design objectives

G1–G8 locked in `CANDIDATE_DEFINITIONS.md` before any candidate was scored.
G1 (preserve hard large_norm protection) is treated as a **non-negotiable
constraint**, not a soft objective, given STEP 5's P(diverge|≥1 accept)=1.000
finding for this attack specifically.

## 3. Candidate Combined rules

Seven candidates (C0–C6) defined in `CANDIDATE_DEFINITIONS.md`: naive OR,
majority vote, hard-gate+OR, hard-gate+AND, hierarchical+peer-rank,
normalized weighted score, and gate+weighted-directional. All use the
existing, unchanged τ/ρ/κ (Regime A) unless marked "drift-aware" (§10).

## 4. Offline replay methodology

Every candidate's accept/reject decision was computed independently for all
9,000 Trace B rows from the raw `norm_score`/`cosine_score`/`sign_score`
columns (not by reusing the single-signal `accept_*` columns, except as a
consistency check — see `TRACE_AUDIT.md`). Honest/malicious-active
membership uses the **corrected** definition
(`honest := (~is_malicious) | (attack=="no_attack")`), documented and
justified in `TRACE_AUDIT.md` (recovers 180 rows mislabeled malicious under
`no_attack` due to a role-vs-perturbation labeling nuance in the trace
generator — not a bug in this analysis, a documented property of the source
data, corrected transparently before any candidate was scored).

## 5. Overall results (Regime A, static variant, pooled 3 seeds × 4 partitions)

| candidate | honest FPR | malicious TPR (pooled, incl. `low_norm`) | catastrophic-risk |
|---|---|---|---|
| C0 (naive OR) | 0.2044 | 0.780 | safe |
| C1 (majority) | **0.0000** | 0.544 | **UNSAFE — 135/180 large_norm accepted** |
| C2 (gate+OR) | 0.2044 | 0.780 | safe — **identical to C0, see §9** |
| C3 (gate+AND) | 0.1622 | 0.694 | safe |
| C4 (hierarchical) | 0.1969 | 0.780 | safe |
| C5 (weighted, no gate) | **0.0000** | 0.293 | **UNSAFE — 180/180 large_norm accepted** |
| C6 (gate+weighted dir.) | 0.1622 | 0.649 | safe |

Full table: `offline_replay_summary.csv`. The two candidates with the most
attractive honest-FPR numbers (C1, C5) are exactly the two that fail the
hard safety constraint — this is the central, expected-in-advance tension
this whole design step exists to navigate (§2, G1 vs. G4).

## 6. Attack-specific results

`offline_replay_by_attack.csv`. Headline TPRs:

| attack | C0 | C1 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| large_norm | 1.00 | 0.25 | 1.00 | 1.00 | 0.00 | 1.00 |
| directional_poisoning | 0.95 | 0.75 | 0.75 | 0.95 | 0.23 | 0.62 |
| full_sign_flip | 0.95 | 0.72 | 0.72 | 0.95 | 0.23 | 0.62 |
| sparse_coordinate_attack | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| low_norm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**OBSERVED, non-obvious**: `sparse_coordinate_attack` TPR is 1.00 for *every*
candidate, including gate-less C5. This is explained in §11 (ablation): in
this trace, **Norm alone already achieves perfect TPR on
`sparse_coordinate_attack`** (its magnitude happens to also exceed τ), so no
candidate's handling of this attack is actually discriminating — STEP 7's
"Norm+Sign jointly required" finding described *co-occurring* rejections,
not *independently necessary* ones once Norm is already present as an
OR-term. This refines, not contradicts, the STEP 7 complementarity claim.
`low_norm` is 0.00 for every candidate by construction (not a harmful attack
in the tested setup; STEP 5 found misses here are never catastrophic).

## 7. Non-IID results

`offline_replay_by_partition.csv`, Regime A. The honest FPR of every
static-gate candidate is **dominated by the IID condition specifically**
(e.g. C0: 0.483 at IID vs. 0.091–0.140 at the three Dirichlet levels) —
consistent with `analysis_v2`'s finding that Norm's temporal-drift FPR
problem is *worst at IID*, not worsened by Non-IID (Non-IID, if anything,
makes Norm's FPR better; it is Cosine's FPR that Non-IID would be expected
to worsen, but Cosine's own FPR contribution here is small, see §11).
Regime B (IID-calibrated, frozen) was additionally checked for C0/C1/C2/C3
on the three non-IID partitions (`offline_replay_by_partition.csv`,
`regime=B_iid_transfer_frozen`); C4/C5/C6 were not re-derived under Regime B
(documented limitation, §14) since their R-score/peer-rank machinery was
defined against Regime-A τ/ρ/κ only.

## 8. Honest false-rejection analysis

Pooled honest FPR ranges from 0.000 (C1, C5 — both disqualified) to 0.204
(C0/C2). The two Pareto-frontier survivors (§9) sit at 0.162 (C3) and 0.197
(C4) — both still meaningfully above Cosine/Sign's own single-signal FPRs
(1.2%/3.1%), confirming Norm's FPR is the dominant cost driver in *every*
safe candidate, exactly as STEP 7 predicted before any candidate was built.

## 9. Catastrophic-risk analysis

`catastrophic_risk.csv`. C1 and C5 are **disqualifying failures**: C1
(majority vote) accepts 135/180 `large_norm` updates because Norm is the
*sole* rejecting signal on the large majority of these rows (STEP 7: Sign
never contributes, Cosine only 25% of the time) — a 2-of-3 rule overturns
exactly the single-signal-necessary case G1 was written to protect. C5
(pure weighted score) accepts **all** 180/180, because its clipped
`R_norm` saturates at 1.0 for any score ≥2× τ, diluting a norm of 22,809
down to the same weight as a norm barely over threshold, then averaging it
away against two near-zero directional components. All four safe candidates
(C0, C2, C3, C4) accept exactly 0/180. Maximum consecutive accepted
directional-attack updates (a secondary sustained-miss risk per STEP 5,
P≈0.03 for single misses but rising with sustained ones): C0=1, C3=11, C4=1,
C6=15 — C3 and C6's AND-type stage 2 can let a long run of
`directional_poisoning`/`full_sign_flip` updates through in the conditions
where Cosine alone would have caught them but Sign does not concur.

## 10. Temporal-drift implications (STEP 5 design study — PROPOSED, not yet FL-validated)

Substituting the drift-aware Norm gate (per-round robust median+MAD, 3.5×
modified-z cutoff, label-free, fallback to static τ when MAD=0 —
`CANDIDATE_DEFINITIONS.md` §5) into the Norm-gate stage of C2/C3/C4/C6,
**while keeping Cosine/Sign unchanged**:

| candidate | honest FPR static → drift-aware | malicious TPR static → drift-aware | large_norm accepted (drift-aware) |
|---|---|---|---|
| C2 (≡C0) | 0.204 → **0.055** | 0.780 → 0.780 (unchanged) | 0/180 |
| C3 | 0.162 → **0.014** | 0.694 → 0.634 (**−0.060**) | 0/180 |
| C4 | 0.197 → **0.048** | 0.780 → 0.780 (**unchanged**) | 0/180 |
| C6 | 0.162 → **0.014** | 0.649 → 0.567 (**−0.082**) | 0/180 |

**OBSERVED**: the drift-aware gate fully preserves large_norm safety (0/180
accepted in all four) while cutting honest FPR by 3–12×. **C4 gets this
improvement for free** (TPR unchanged — verified per-attack: both
directional attacks stay at 0.95 exactly, because C4's `cos_fail` OR-term
already provides full directional coverage independent of the Norm gate's
exact shape). **C3 and C6 pay a real TPR cost** (directional TPR drops from
0.736→≈0.586 for C3) because their AND-gated stage 2 was relying in part on
the *static* gate's incidental over-triggering to catch some directional
updates that stage 2 alone would miss — removing that slack removes some of
those incidental catches. **INFERRED**: Norm's honest-FPR problem is
substantially a static-calibration/temporal-drift artifact, not an
inherent property of using Norm as a hard gate (directly answering STEP 5's
question) — but this conclusion rests on a trajectory (Trace B) that never
diverges by construction (Coordinate-Median-driven); whether a drift-aware
gate remains stable once it is *actually* steering acceptance/rejection in a
live FL loop is explicitly untested and is the top recommended next
experiment (§15).

## 11. Logical ablation

`ablation_replay.csv`. For C0 (=C2): removing Norm lets 135/180 `large_norm`
updates through (confirms G1's dependency on Norm) and drops pooled TPR from
0.780→0.630; removing Cosine drops TPR 0.780→0.694 (directional detection
loss, honest FPR barely moves: 0.204→0.192); **removing Sign leaves TPR
unchanged at 0.780 exactly** and honest FPR drops only marginally
(0.204→0.174) — in this specific OR-structured trace, Sign's marginal
contribution is fully subsumed by Norm, which independently already catches
100% of `sparse_coordinate_attack` (§6). For C3/C6 (AND-gated stage 2):
removing *either* Cosine or Sign collapses stage 2 to always-false (an AND
with one side forced False never fires), so both ablations land on exactly
the same numbers as each other and as "Norm alone" — and, notably, **honest
FPR is unchanged from the full rule** in both cases (0.162→0.162), because
stage 2 (`cos_fail AND sign_fail`) **never fires on a single honest update in
this trace** — directly reproducing STEP 7's multi-seed finding that "no
honest update was ever rejected by two or more signals simultaneously, in
any seed." This is an internal cross-validation between this offline replay
and the independently-generated STEP 7 analysis, not an assumption.

## 12. Pareto analysis

Full writeup in `PARETO_ANALYSIS.md`. Summary: C1 and C5 are disqualified on
the hard safety constraint (G1); C2 is dropped as a verified duplicate of
C0; C0 is dominated by C4 (equal-or-better on every axis); C6 is dominated
by C3 (equal FPR, strictly worse directional TPR). **Pareto frontier:
{C3, C4}** — a genuine honest-FPR-vs-directional-recall tradeoff, not a
single winner.

## 13. Candidate selection (at most two, per STOP CONDITION — not by final
model accuracy, since none was trained)

**Candidate A (primary): C4 (hierarchical gate + Cosine + peer-rank-confirmed
Sign), with the drift-aware Norm gate as the specific refinement to carry
into the next real benchmark.** Justification: matches C0's full directional
TPR (0.95) and full large_norm/sparse safety while already reducing static
honest FPR (0.197 vs 0.204), and — uniquely among the four safe
candidates — gains the drift-aware gate's large FPR reduction (0.197→0.048)
with **zero TPR cost**, because its OR-based Cosine term does not depend on
the Norm gate's exact shape.

**Candidate B (alternative): C3 (hard Norm gate + Cosine AND Sign), static
variant.** Simpler (two raw booleans, no peer-rank machinery), lowest honest
FPR among the disqualification-safe non-duplicate candidates (0.162) at the
cost of real directional-attack recall (0.736, and up to 11 consecutive
directional misses observed in the worst condition). Recommended as the
"prioritize honest-client burden over directional recall" alternative,
**not** paired with its own drift-aware variant by default, since that
variant was shown (§10) to trade away additional directional TPR rather
than gaining it for free.

Both selections exclude C1/C5 (disqualified) and C6 (dominated); neither
claims superiority over Median/Multi-Krum or any accuracy improvement — see
§14.

## 14. Critical scientific limitation

Offline replay evaluates per-update **accept/reject behavior against a
fixed, Coordinate-Median-driven trajectory that never diverges by
construction**. It does **not** show what the global model trajectory would
do once a Combined rule is actually driving acceptance/rejection — a
different aggregation decision each round changes the *subsequent* round's
honest gradients, scores, and even whether Norm's drift pattern looks the
same. Accordingly this report makes **no claim** that any candidate
"improves model accuracy," "prevents divergence," or "outperforms
Median/Multi-Krum" — those require Combined running inside the actual FL
loop, which is explicitly out of scope here (STOP CONDITION). The
drift-aware Norm gate (§10) is a **design study**, not a validated
mechanism — it was evaluated only along the Trace B observational
trajectory, never by actually substituting it into training. Regime B was
only checked for C0/C1/C2/C3 (§7), not C4/C5/C6. Ablation (§11) used a
mechanical "force this signal's fail flag False" definition, which is the
standard ablation technique but is not identical to retraining a 2-signal
system from scratch.

## 15. Recommended next experiment

A **minimal, bounded** end-to-end FL benchmark — not the full 594-matrix —
testing exactly:
- Candidate A (C4, static Norm gate) and Candidate A′ (C4 + drift-aware Norm
  gate) inside the real FL loop, replacing the single-signal defenses, on
  the 4 partitions × `{large_norm, directional_poisoning,
  sparse_coordinate_attack}` (the 3 attacks with genuinely different
  per-signal behavior; `full_sign_flip` tracks `directional_poisoning`
  closely enough in every prior analysis to be optional here) × 3 seeds —
  36 runs.
- Candidate B (C3, static) on the same grid as a second arm — 36 more runs.
- Primary outcomes to measure for the first time in a live loop: final
  accuracy, divergence rate, and — specifically for Candidate A′ — whether
  the drift-aware gate remains stable (does not itself start rejecting
  honest updates at a growing rate) once it is actually steering which
  updates get aggregated, which Trace B's static observational trajectory
  cannot test.
- Do not expand to Fed-ISIC2019 or the full matrix until this minimal
  benchmark's results are reviewed.

---

## Decision memo (per STOP CONDITION — answering the 7 requested questions)

**1. Which candidate rules were clearly bad, and why?**
C1 (majority vote) and C5 (pure weighted score, no hard gate) are both
disqualified: C1 accepts 135/180 `large_norm` updates because a 2-of-3 rule
overturns the single-signal-necessary case for this attack; C5 accepts all
180/180 because its clipped, averaged score structurally cannot give a
single catastrophic outlier the weight it needs. C2 is not "bad," but is
redundant — proven byte-for-byte identical to C0 (9,000/9,000 rows) under
static thresholds, so it is not a distinct design.

**2. Which candidate has the best catastrophic-safety profile?**
C0, C3, C4, C6 are tied at 0/180 `large_norm` accepted (and so is every
drift-aware-gate variant of C2/C3/C4/C6). Among these, C4 and C0 additionally
have the fewest consecutive-miss opportunities on `directional_poisoning`/
`full_sign_flip` (max 1 consecutive accepted update, vs. 11 for C3 and 15 —
the full attack window — for C6).

**3. Which candidate has the best honest-FPR / attack-coverage tradeoff?**
No single winner — C3 (FPR 0.162, directional TPR 0.736) and C4 (FPR 0.197,
directional TPR 0.950) are both Pareto-optimal; the choice is a real
tradeoff (§12). With the drift-aware Norm gate added, C4 strictly improves
(FPR 0.197→0.048 at **no** TPR cost) while C3's equivalent gain costs real
TPR — so if the drift-aware refinement is adopted, **C4 becomes the clearer
single best choice**.

**4. Does static Norm gating remain acceptable?**
Yes as a safety mechanism (0 large_norm misses in every safe candidate), but
its honest-FPR cost (16–20 percentage points, concentrated almost entirely
at IID) is the binding constraint on every safe candidate's overall FPR —
confirming §1's motivating observation rather than resolving it on its own.

**5. Do we need drift-aware Norm calibration before end-to-end Combined?**
Not strictly required to get a *safe* Combined rule (static C3/C4 are both
already safe), but the offline evidence (§10) is strong enough — large FPR
reduction, zero TPR cost for C4 specifically, safety fully preserved — that
it is the single highest-value refinement to validate next, rather than
something to defer indefinitely.

**6. Which ONE or TWO candidates should be tested in the real FL loop next?**
C4 (static) and C4 + drift-aware Norm gate, as Candidate A / A′; C3 (static)
as Candidate B if a simpler two-boolean rule is preferred for the report's
narrative. See §13 for full justification.

**7. What exact minimal benchmark should be run next to validate them?**
4 partitions × 3 representative attacks (`large_norm`,
`directional_poisoning`, `sparse_coordinate_attack`) × 3 seeds × {C4-static,
C4-drift-aware, C3-static} = 108 runs, measuring final accuracy, divergence
rate, and (new, not available from Trace B) whether the drift-aware gate's
FPR stays low once it is actually driving which updates get aggregated. No
expansion to Fed-ISIC2019 or the full 594-combination matrix until this is
reviewed.
