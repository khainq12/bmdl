# STEP 8 — Pareto Analysis (Combined Design candidates, static variant, Regime A, pooled)

## Axes

| candidate | honest FPR (↓ better) | large_norm safe? (hard constraint) | directional TPR (↑ better) | sparse TPR (↑ better) | interpretability (↑ better, 1-5) | complexity (↓ better, 1-5) |
|---|---|---|---|---|---|---|
| C0 (naive OR) | 0.2044 | yes | 0.950 | 1.000 | 5 | 1 |
| C1 (majority) | 0.0000 | **no** | 0.736 | 1.000 | 5 | 1 |
| C2 (hard gate + OR) | 0.2044 | yes | 0.950 | 1.000 | 5 | 1 |
| C3 (hard gate + AND) | 0.1622 | yes | 0.736 | 1.000 | 4 | 2 |
| C4 (hierarchical + peer-rank) | 0.1969 | yes | 0.950 | 1.000 | 3 | 3 |
| C5 (weighted, no gate) | 0.0000 | **no** | 0.233 | 1.000 | 3 | 3 |
| C6 (gate + weighted directional) | 0.1622 | yes | 0.622 | 1.000 | 3 | 3 |

`sparse_tpr` is 1.000 for every candidate, including gate-less C5 — it is not
discriminating here (see `REPORT.md` §11 ablation discussion: in this trace,
Norm alone already achieves perfect TPR on `sparse_coordinate_attack`, so it
cannot separate candidates).

## Hard constraint first (G1, non-negotiable per STEP 5/7 catastrophic-miss
evidence: P(diverge | ≥1 large_norm accept) = 1.000)

`catastrophic_risk_flag = True` (C1, C5) is treated as **disqualifying**, not
one more axis to trade off — an candidate that ever accepts a `large_norm`
update is excluded from the frontier regardless of how attractive its honest
FPR looks. This is a deliberate asymmetric-cost choice (G1), not an oversight:
C1 and C5 both show a tempting honest FPR of exactly 0.000, but C1 accepts
135/180 and C5 accepts **all 180/180** `large_norm` updates — in the real FL
loop (per STEP 5 of `analysis_v2`), either would be expected to diverge on
first contact with this attack.

**C1 and C5 are eliminated.**

## Redundancy check

C2's decisions are byte-for-byte identical to C0's (verified: 9,000/9,000
rows match — see `offline_replay_summary.csv`, same row twice). This was
predicted in `CANDIDATE_DEFINITIONS.md` before replay: writing the same OR
as two stages does not change the truth table under static thresholds.
**C2 adds nothing over C0 and is dropped as a duplicate**, not scored
separately below.

## Pareto dominance among the remaining 4 safe, non-redundant candidates
(C0, C3, C4, C6)

A dominates B if A is no worse on every axis and strictly better on at least
one (axes: honest FPR, directional TPR — sparse TPR and large_norm-safety
are tied/satisfied by all four and so cannot discriminate):

- **C4 vs. C0**: honest FPR 0.1969 < 0.2044 (C4 better), directional TPR tied
  at 0.950, sparse tied, both safe. **C4 weakly dominates C0.** → C0 eliminated.
- **C3 vs. C6**: honest FPR tied at 0.1622, directional TPR 0.736 > 0.622
  (C3 better), sparse tied, both safe. **C3 dominates C6.** → C6 eliminated.
- **C4 vs. C3**: honest FPR 0.1969 > 0.1622 (C3 better) but directional TPR
  0.950 > 0.736 (C4 better) — **neither dominates**; this is a genuine
  FPR-vs-directional-recall tradeoff. **Both remain on the frontier.**

## Resulting Pareto frontier: **{C3, C4}**

| | honest FPR | directional TPR | interpretability | complexity |
|---|---|---|---|---|
| C3 | 0.162 (lower) | 0.736 (lower) | simpler (AND of two booleans) | lower |
| C4 | 0.197 (higher) | 0.950 (higher) | peer-rank adds a moving part | higher |

Neither is strictly best; the choice is a direct tradeoff between honest-client
burden and directional-attack recall. Both are carried into STEP 9 (ablation)
and STEP 11 (candidate selection) below; see `REPORT.md` for how the
drift-aware Norm gate (STEP 5 design study) changes this picture again before
a final recommendation is made.
