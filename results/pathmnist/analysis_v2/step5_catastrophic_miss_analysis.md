# STEP 5 — Catastrophic-miss analysis

Data: `step5_catastrophic_miss_table.csv` (Trace A, seed=42, per-client
detail for the 7 large_norm divergences) + `per_round_results.csv` (all 3
seeds, pooled, for the quantitative P(divergence|...) claims and the
non-large_norm divergence cases).

## Per-divergence detail (large_norm, Regime A, seed=42 — Trace A)

| defense | partition | round | malicious score | threshold | malicious delta norm | loss before | loss after |
|---|---|---|---|---|---|---|---|
| cosine | α=0.1 | 10 | −0.0007 | −0.544 | 22,790 | 2.30 | 1.67×10⁸ |
| cosine | α=0.5 | 10 | 0.0029 | −0.437 | 22,792 | 2.25 | 9.62×10⁷ |
| cosine | α=1.0 | 10 | −0.0021 | −0.514 | 22,804 | 2.22 | 9.89×10⁷ |
| sign_consensus | α=0.1 | 10 | 0.848 | 0.378 | 22,803 | 2.29 | 2.40×10⁸ |
| sign_consensus | α=0.5 | 10 | 0.793 | 0.400 | 22,800 | 2.26 | 9.82×10⁷ |
| sign_consensus | α=1.0 | 10 | 0.810 | 0.416 | 22,817 | 2.21 | 1.93×10⁸ |
| sign_consensus | iid | 10 | 0.676 | 0.532 | 22,800 | 2.18 | 1.70×10⁸ |

**OBSERVED:** all 7 — confirmed identically across all 3 seeds for all 7
combos (21/21 checked in `per_round_results.csv`, not just the seed=42
trace) — fail at **round 10, the very first round the attack is active**.
Malicious delta norm is consistently ≈22,790–22,817 (matches the
`large_norm` attack's own definition: `N(0, 50²)` over ~207,081
parameters → expected norm `50·√207081 ≈ 22,751`). Loss jumps from a
normal ≈2.2–2.3 to 9.6×10⁷–2.4×10⁸ in that single round. "Whether
recovery occurred" is **not observable**: the divergence safeguard halts
further training once the threshold is crossed (§`MISSING_DATA_NOTE.md`),
so no post-divergence trajectory exists in this data to check for
recovery, for any defense, in either the real sweep or Trace A.

## Quantified: P(divergence | ≥1 accepted malicious update)

Computed over **every** detector-defense trajectory (`norm`, `cosine`,
`sign_consensus` × 4 partitions × 2 regimes where applicable × 3 seeds ×
5 attacks excluding `no_attack` = 315 trajectories), using the pooled
`fn` (malicious-accepted) counts already in `per_round_results.csv` — no
new compute needed for this part.

| | diverged | not diverged |
|---|---|---|
| **≥1 malicious accepted** (active rounds) | 34 | 125 |
| **0 malicious accepted** | 0 | 156 |

- **P(divergence \| 0 malicious accepted) = 0.0000** (n=156) — a clean,
  large-sample **necessary condition**: in this entire dataset, zero
  trajectories ever diverged without the detector accepting at least one
  malicious update first. No defense "spontaneously" fails.
- **P(divergence \| ≥1 malicious accepted) = 0.2138** (n=159) — but a miss
  is **not sufficient** in general: fewer than a quarter of
  "at-least-one-miss" trajectories actually diverge.

**The sufficiency of a single miss is sharply attack-dependent:**

| attack | P(divergence \| ≥1 miss) | n | mean accepted-malicious rounds observed when it diverges |
|---|---|---|---|
| large_norm | **1.000** | 30 | 15 (= 1 real event at round 10 + 14 copy-forwarded repeats, not 15 independent events — see caveat below) |
| low_norm | 0.000 | 54 | — |
| directional_poisoning | 0.030 | 33 | 8.2 |
| full_sign_flip | 0.030 | 33 | 8.2 |
| sparse_coordinate_attack | 0.222 | 9 | 15 (same copy-forward caveat) |

**Caveat on "mean accepted-malicious rounds":** once a trajectory
diverges, the safeguard copy-forwards the triggering round's `tp/fp/tn/fn`
values for every remaining round (Addendum (i)) — so a `large_norm`
divergence at round 10 shows `fn>0` for all 15 rounds 10–24 in the
pooled data, but this is **one** real failure event, not 15 independent
ones. For the rarer `full_sign_flip`/`directional_poisoning` cases, the
underlying per-round trace (checked directly, not inferred) shows **7–8
genuinely consecutive accepted-malicious rounds before the loss actually
jumps** (e.g. `sign_consensus/α=0.1/full_sign_flip/seed=43`: first miss at
round 10, loss jump at round 19 — 7 real accumulated misses in between,
not an artifact).

## HYPOTHESIS TEST: "a high average TPR is insufficient when a single false negative can catastrophically destroy the model"

**Result: conditionally true, and the condition is attack magnitude.**

- **For `large_norm` (unbounded-magnitude attack): TRUE without
  qualification.** `P(divergence | ≥1 miss) = 1.000`. A detector with
  TPR=0.999 against this attack would still be expected to fail
  catastrophically within ~1,000 rounds of exposure on this evidence — one
  miss is sufficient, full stop. This is the sharpest form of the
  hypothesis, and the data supports it unreservedly for this attack.
- **For magnitude-bounded attacks (`full_sign_flip`,
  `directional_poisoning`, `sparse_coordinate_attack`): FALSE as stated —
  a single miss is usually survivable** (`P(divergence | ≥1 miss)` is
  0.03–0.22, not 1.0) — **but sustained low recall is still dangerous**:
  every observed divergence under these attacks required 7+ consecutive
  undetected acceptances, not one. The correct generalization of the
  hypothesis for bounded attacks is "sustained near-zero TPR over many
  consecutive rounds is sufficient to destroy the model," not "any single
  miss is."
- **For `low_norm`: the hypothesis does not apply** — this attack is not
  magnitude-damaging regardless of acceptance (it scales the honest update
  toward zero), so "acceptance" here causes no harm by construction; 54
  trajectories accepted it with 0 divergences.

**This is an OBSERVED, quantified result, not a hypothesis** — the
attack-magnitude-dependent split is read directly from `per_round_results.csv`
with no new data collection, and the three-seed consistency (21/21
large_norm cases diverging at exactly round 10) rules out single-seed
noise as the explanation.
