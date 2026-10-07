# STEP 7 — Complementarity of Norm, Cosine, Sign Consensus

Data: Trace B — one stable, Coordinate-Median-driven trajectory per
`(partition, attack)` (median never diverges anywhere in the real sweep,
so the same honest+malicious update sequence can be scored by all three
signals without the trajectory itself collapsing differently depending on
which signal would have driven it; see `MISSING_DATA_NOTE.md` for why this
control was necessary). **Updated 2026-10-06 per external review**: the
original version of this analysis used seed=42 only; a reviewer (ChatGPT,
consulted by the user as a second opinion on this analysis) specifically
asked whether the complementarity percentages were seed-specific before
treating them as load-bearing for a Combined Attestation decision. Traced
seeds 43 and 44 were added (`trace_B_complementarity_seed{42,43,44}.csv`,
pooled into `trace_B_complementarity_pooled3seeds.csv`, 9,000 rows) and
this document now reports the pooled 3-seed numbers, with the per-seed
breakdown (`step7_per_seed_consistency.csv`) used to state explicitly
which findings are seed-invariant and which have real seed-to-seed
variation — rather than silently swapping one single-seed number for
another. Full counts: `step7_complementarity_tables.csv`. Figure:
`figures/complementarity_overlap.png`. All numbers below are Regime A
(condition-specific thresholds), pooled across the 4 partitions and now
3 seeds (n=900 malicious client-rounds, n=7,920 honest client-rounds)
unless stated otherwise.

## Multi-seed consistency check (what changed, what didn't)

- **Seed-invariant (identical composition in all 3 seeds individually,
  not just on average)**: `large_norm` (75% Norm-alone / 25% Norm+Cosine
  in every seed), `sparse_coordinate_attack` (75% Norm+Sign / 25% all
  three in every seed), `low_norm` (100% accepted-by-all in every seed).
  These are the attacks this report's strongest claims rest on, and they
  do not move at all across seeds.
- **Qualitatively stable, quantitatively variable**: `directional_poisoning`
  and `full_sign_flip` — in every seed, Cosine is involved in every
  rejection and Norm/Sign are never independently sufficient without it
  (the qualitative claim in this document), but the exact split between
  "Cosine alone" vs. "Norm+Cosine" vs. "Cosine+Sign" vs. "all three" shifts
  by up to ~15 percentage points between seeds (e.g. Cosine-alone:
  20.0%/seed42, varies to 3.3%/seed44 for `full_sign_flip`-adjacent
  patterns — see `step7_per_seed_consistency.csv` for the full per-seed
  table). The pooled numbers below are the right ones to quote; a single
  seed's exact split should not be.
- **Corrected by pooling**: honest-update `accepted_by_all` was **85.1%
  at seed=42 alone**; the 3-seed pooled figure is **79.6%** (range across
  individual seeds: 75.6%–85.1%). The single-seed number was on the
  optimistic end, not the center — reported here as found, not smoothed
  into the earlier draft.

## Malicious updates: each signal has a distinct "home turf" attack, and none covers everything alone

| attack | dominant rejecting signal(s) | accepted by all 3 |
|---|---|---|
| `large_norm` | **Norm** (75% Norm-alone, 25% Norm+Cosine — identical in all 3 seeds) — Sign **never** contributes | 0% |
| `directional_poisoning` | **Cosine** (20% alone, 27% with Norm, 18% with Sign, 30% all three — pooled; composition varies by seed, Cosine's necessity does not) | 5% |
| `full_sign_flip` | **Cosine** (23% alone, 29% with Norm, 15% with Sign, 28% all three — pooled) | 5% |
| `sparse_coordinate_attack` | **Norm+Sign jointly** (75%), rising to all three (25%) — identical in all 3 seeds — **Cosine alone or Cosine+anything never appears** | 0% |
| `low_norm` | none — not a magnitude/direction attack by construction | 100% |

**OBSERVED, directly from the counts (not inferred):**
- For `large_norm`, **Sign Consensus contributes to zero rejections** — every
  rejection is Norm-alone or Norm+Cosine. This is the same result as
  STEP 4/5 (Sign's large_norm score exceeds the honest range, so it never
  flags this attack), now confirmed in a controlled setting where the
  trajectory itself is healthy.
- For `directional_poisoning` and `full_sign_flip`, **Norm alone never
  uniquely rejects** — every Norm-involved rejection for these two attacks
  co-occurs with Cosine. Cosine is the necessary ingredient; Norm adds
  incremental coverage on top of it, not independently.
- For `sparse_coordinate_attack`, **Cosine contributes to zero
  rejections** — every rejection is Norm+Sign or all three. This is the
  cleanest "Sign catches what Cosine structurally cannot" result: a sparse
  spike in a handful of coordinates barely shifts the *overall* direction
  (cosine is a whole-vector measure) but is directly visible to a
  coordinate-wise sign-agreement check.
- `low_norm` is accepted by all three, always — confirms this attack
  (a free-rider scaling-down, not a poisoning attack) is not something any
  of the three signals is designed to catch, and none does.

## Honest updates: Norm is the noisiest signal for false positives

| rejected by | fraction of honest client-rounds (3-seed pooled) | range across individual seeds |
|---|---|---|
| (accepted by all) | 79.6% | 75.6% – 85.1% |
| Norm alone | 16.1% | 11.9% – 18.9% |
| Sign alone | 3.1% | 2.3% – 3.5% |
| Cosine alone | 1.2% | 0.6% – 2.0% |
| (any multi-signal rejection) | 0% — not observed in any seed of this trace | — |

**OBSERVED:** in this trace (all 3 seeds), every honest-update rejection
was by exactly one signal — no honest update was ever rejected by two or
more signals simultaneously, in any seed. Norm accounts for the large
majority (16.1 of ~20.4 percentage points) of all honest false
rejections, consistent with STEP 3/4's finding that Norm's FPR is
elevated by temporal drift
independent of Non-IID or the attack present.

## Does this justify Combined Attestation? — the empirical case, stated precisely

**Yes, there is real empirical justification for combining signals, with
a specific, evidenced shape — not a vague "more signals are better"
argument:**

1. **No single signal is sufficient across the attack suite.** Norm misses
   `directional_poisoning`/`full_sign_flip` (never the sole or joint-necessary
   rejector without Cosine) and contributes nothing to `sparse_coordinate_attack`'s
   Cosine-component (it isn't needed there either, but Sign is). Cosine
   contributes nothing to `sparse_coordinate_attack` and nothing to `large_norm`
   beyond what Norm already provides alone 75% of the time. Sign
   contributes nothing to `large_norm` at all and nothing to the two
   directional attacks beyond what Cosine already provides. **Every signal
   has at least one required attack category and at least one attack
   category where it is entirely absent from the rejection set.**
2. **The three "required" categories do not overlap**: Norm ↔ unbounded
   magnitude (`large_norm`); Cosine ↔ global direction (`directional_poisoning`,
   `full_sign_flip`); Sign (jointly with Norm) ↔ sparse/coordinate-local
   (`sparse_coordinate_attack`). This is close to a textbook complementary
   -coverage pattern, not three redundant measurements of the same thing.
3. **The honest-side cost is asymmetric, which matters for how signals
   should be combined**: Norm is responsible for most false rejections, so
   a combination rule that requires *all* signals to agree before
   rejecting (AND-of-rejections / majority-vote-to-reject) would inherit
   Norm's FPR problem; a rule that requires only one signal to flag
   (OR-of-rejections) would compound all three signals' FPRs rather than
   averaging them out. **This observed asymmetry is itself an argument for
   why the Combined design needs to be chosen deliberately from this
   evidence, not assumed** — which is explicitly out of scope for this
   analysis pass.

**What this analysis does NOT establish:** it does not test whether a
specific combination rule (OR, AND, majority, weighted) would actually
improve on the best single signal once both attack-detection *and*
honest-client-FPR are weighed together — that requires implementing and
evaluating the rule, which is Combined Attestation itself, explicitly
deferred. It also does not cover Regime B or non-`large_norm` catastrophic
-divergence interactions in the complementarity frame (Trace B's stable
median-driven trajectory was deliberately chosen to be divergence-free, so
it cannot speak to whether a Combined rule would itself be vulnerable to
the same single-miss catastrophe found in STEP 5 — that would need a
dedicated follow-up once a Combined rule exists).
