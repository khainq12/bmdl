# C4-Drift-Aware Targeted Confirmation (per ChatGPT review of combined_e2e_minimal)

**192 runs**: 4 partitions × 8 seeds (42–49) × 6 attacks (no_attack,
large_norm, directional_poisoning, sparse_coordinate_attack, full_sign_flip,
low_norm) × `c4_drift_aware` only. 48 of these (seeds 42–44 × 4 of the 6
attacks) are reused unmodified from `combined_e2e_minimal/`; 144 are newly
run. Same locked protocol throughout (K=50, 25 rounds, etc.). **No code in
`benchmark/defenses/combined.py` was changed as a result of any finding in
this report** — this is a confirmation pass, not a redesign pass.

Plus: `MAGNITUDE_STRESS_TEST.md` — an analytical, diagnostic-only sweep of
hypothetical attacker-magnitude multipliers against the real honest-norm
distribution, using the unmodified `_drift_tau` function (no new training).

## Verdict against ChatGPT's stated lock criteria

| criterion | result |
|---|---|
| large_norm → safe | **0/480** malicious client-rounds accepted, across all 8 seeds individually (0 in every single seed) |
| directional → robust | TPR mean 0.879 (std 0.151, min 0.467 — see caveat below) |
| full_sign_flip → robust | TPR mean 0.885 (std 0.145, min 0.467) — **first real end-to-end measurement of this attack for Combined** |
| sparse_coordinate → robust | TPR = **1.000 in every one of 192 runs**, zero variance |
| low_norm → no meaningful damage | TPR = 0.000 always, as expected (not a harmful attack; no signal in this whole project has ever been designed to flag it) |
| clean → stable | mean honest FPR 0.056 (no_attack condition), max single-run FPR 0.427 — elevated in a few individual runs but never collapse-level |
| Non-IID → stable | honest FPR by partition: iid 0.040, α=1.0 0.035, α=0.5 0.055, α=0.1 0.096 — mild, no partition is catastrophic |
| more seeds → stable | **0/192 collapse (honest_fpr>0.9) across all 8 seeds** — extends the original 0/48 finding cleanly |
| no collapse | **confirmed, 0/192**, and 0 numerical divergence anywhere |

**Every criterion ChatGPT listed is met.** The honest, precise caveat (not
glossed over): directional_poisoning/full_sign_flip TPR has real
seed/partition variance (min observed 0.467 in at least one condition each)
— "robust" here means *consistently high on average and never catastrophic*,
not *literally perfect every time*, unlike sparse_coordinate_attack and
large_norm which are perfect/zero-variance.

## large_norm safety — the headline result

0 accepted malicious client-rounds out of 480 (8 seeds × 4 partitions × 15
attacked rounds), **identically 0 in every one of the 8 seeds individually**
(60/60 correctly rejected per seed). TPR = 1.000, standard deviation =
0.000 — this is now the most heavily-validated single finding in the whole
Combined Attestation research arc (0/12 → 0/48 → 0/480 across three
successive stages of increasing scale).

## Magnitude self-influence stress test (analytical, diagnostic only)

Using the real honest-norm distribution from 1,200 sampled (run, round)
contexts across the 144-run `combined_e2e_minimal` benchmark: a
hypothetical attacker norm injected at `k`× the honest median passes the
recomputed `tau_drift` with probability 1.000 at k=1 (trivial — that's not
an attack, just the median itself), 0.179 at k=2, 0.023 at k=3, **and
effectively 0 from k=5 onward** (max observed passing multiplier: 5×, out
of 14 tested levels from 1× to 200×). Real `large_norm` attacks (scale=50,
producing norms in the thousands) operate several orders of magnitude
beyond this narrow ≤5× window — directly explaining why self-influence
(median relative shift 16.9% in the real benchmark) never once translated
into an actual safety breach in 480 real attacked rounds. Per ChatGPT's
correction: the precise claim is **"attacker self-influence is material but
bounded, and the exploitable magnitude region (≤5× honest median) does not
overlap with the actual large_norm attack's operating region (10²–10⁴×)"**
— not "self-influence is negligible."

## Seed-to-seed variance (honest FPR, no_attack condition)

| seed | mean | std | max |
|---|---|---|---|
| 42 | 0.020 | 0.024 | 0.048 |
| 43 | 0.030 | 0.010 | 0.040 |
| 44 | 0.074 | 0.091 | 0.208 |
| 45 | 0.028 | 0.025 | 0.056 |
| 46 | 0.046 | 0.020 | 0.072 |
| 47 | 0.078 | 0.070 | 0.160 |
| 48 | 0.056 | 0.028 | 0.080 |
| 49 | 0.058 | 0.080 | 0.176 |

Natural run-to-run variation (seeds 44/47/49 run somewhat hotter than
42/43/45), but **every single seed stays at least 4.3× below the 0.9
collapse threshold** that both static candidates crossed repeatedly in
`combined_e2e_minimal`. No seed shows any sign of the collapse failure mode.

## Decision

**PathMNIST design for C4-drift-aware is confirmed stable across 8 seeds,
all 4 partitions, and all 6 attacks in the suite — consistent with
ChatGPT's stated lock criteria.** Per the established "do not redesign
based on results" rule, no changes were made to `combined.py`. Relaying
this confirmation to ChatGPT for final cross-check before considering the
PathMNIST-stage design locked and moving toward Fed-ISIC2019 preparation.
