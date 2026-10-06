# PathMNIST honest-gradient characterization — findings

Design locked in `docs/BENCHMARK_PROTOCOL.md` Addendum 2026-10-06 (e);
produced by `scripts/pathmnist_characterization.py`. **No attack was
injected anywhere in this study and no threshold (τ/ρ/κ) was selected.**
All numbers below are pooled across 3 seeds × 10 rounds × 5 clients
(`n=150` per client-level metric, `n=300` for pairwise cosine, per
partition setting) unless stated otherwise. Full per-row data is in the
CSVs alongside this file; `summary.json` has every statistic in machine
-readable form. Plots referenced below are in `plots/`.

**Standing caveat, repeated from the protocol addendum:** the 53.7%
(centralized), 50.7% (FedAvg IID), and 45.9% (FedAvg Dirichlet(0.5))
accuracy figures from STEP 5 are pipeline sanity checks only. Nothing in
this document treats them as evidence for anything.

## Summary statistics

| Partition | Metric | mean | std | median | p5 | p25 | p75 | p95 | min | max |
|---|---|---|---|---|---|---|---|---|---|---|
| iid | ‖g‖₂ | 0.206 | 0.105 | 0.178 | 0.080 | 0.101 | 0.305 | 0.374 | 0.074 | 0.399 |
| iid | cos(g, g_ref) | **0.959** | 0.029 | 0.967 | 0.907 | 0.942 | 0.982 | 0.990 | 0.856 | 0.994 |
| iid | sign-consensus | 0.875 | 0.026 | 0.877 | 0.820 | 0.865 | 0.892 | 0.912 | 0.796 | 0.926 |
| iid | pairwise cos(gᵢ,gⱼ) | 0.979 | 0.016 | 0.983 | 0.948 | 0.971 | 0.991 | 0.996 | 0.914 | 0.998 |
| dirichlet α=1.0 | ‖g‖₂ | 0.334 | 0.092 | 0.315 | 0.208 | 0.260 | 0.400 | 0.491 | 0.183 | 0.570 |
| dirichlet α=1.0 | cos(g, g_ref) | 0.486 | 0.299 | 0.541 | **-0.093** | 0.290 | 0.717 | 0.881 | -0.309 | 0.954 |
| dirichlet α=1.0 | sign-consensus | 0.706 | 0.058 | 0.703 | 0.607 | 0.668 | 0.736 | 0.807 | 0.566 | 0.859 |
| dirichlet α=1.0 | pairwise cos(gᵢ,gⱼ) | 0.295 | 0.282 | 0.315 | -0.229 | 0.111 | 0.516 | 0.724 | -0.502 | 0.853 |
| dirichlet α=0.5 | ‖g‖₂ | 0.399 | 0.120 | 0.362 | 0.251 | 0.312 | 0.466 | 0.632 | 0.198 | 0.797 |
| dirichlet α=0.5 | cos(g, g_ref) | 0.360 | 0.289 | 0.365 | **-0.103** | 0.134 | 0.577 | 0.872 | -0.274 | 0.924 |
| dirichlet α=0.5 | sign-consensus | 0.653 | 0.087 | 0.647 | 0.528 | 0.592 | 0.705 | 0.813 | 0.486 | 0.894 |
| dirichlet α=0.5 | pairwise cos(gᵢ,gⱼ) | 0.140 | 0.283 | 0.134 | -0.279 | -0.074 | 0.322 | 0.651 | -0.452 | 0.856 |
| dirichlet α=0.1 | ‖g‖₂ | 0.448 | 0.172 | 0.428 | 0.166 | 0.336 | 0.543 | 0.753 | 0.122 | 0.941 |
| dirichlet α=0.1 | cos(g, g_ref) | **0.178** | 0.329 | 0.218 | **-0.362** | -0.067 | 0.446 | 0.665 | -0.630 | 0.759 |
| dirichlet α=0.1 | sign-consensus | 0.604 | 0.073 | 0.614 | 0.486 | 0.544 | 0.656 | 0.727 | 0.462 | 0.766 |
| dirichlet α=0.1 | pairwise cos(gᵢ,gⱼ) | **-0.009** | 0.267 | -0.029 | -0.396 | -0.211 | 0.212 | 0.430 | -0.582 | 0.640 |

## g_ref size sensitivity (nested subsets of the reserved 10% pool)

| Partition | pool fraction | abs. samples | class dist. (9 classes) | mean cos(honest, g_ref) | std |
|---|---|---|---|---|---|
| iid | 0.10 | 899 | [89,93,111,107,68,124,76,99,132] | 0.399 | 0.153 |
| iid | 0.25 | 2249 | [212,213,271,284,204,307,193,241,324] | 0.829 | 0.101 |
| iid | 0.50 | 4499 | [446,469,542,539,397,586,390,480,650] | 0.916 | 0.048 |
| iid | 1.00 | 8999 | [928,941,1099,1067,810,1159,778,913,1304] | 0.960 | 0.028 |
| dirichlet α=0.1 | 0.10 | 899 | (same draws as iid row — identical seeded subsets) | 0.001 | 0.329 |
| dirichlet α=0.1 | 0.25 | 2249 | — | 0.047 | 0.348 |
| dirichlet α=0.1 | 0.50 | 4499 | — | 0.106 | 0.346 |
| dirichlet α=0.1 | 1.00 | 8999 | — | 0.177 | 0.331 |

Class proportions at every size closely track the official PathMNIST
train set's overall class balance (the carve-out is a uniform random
sample, not stratified, but PathMNIST's 9 classes are not wildly
imbalanced, so this falls out naturally rather than by design).

## Five questions, answered

### 1. How aligned are honest clients with `g_ref`?

**Only under IID.** Mean `cos(honest, g_ref) = 0.959`, and even the 5th
percentile is `0.907` (`cos_ref_by_partition.png`) — honest IID clients
are tightly clustered near `g_ref`'s direction, with essentially no
negative-cosine clients across 150 observations. This is the only regime
in this study where "honest clients align with `g_ref`" is unambiguously
true.

### 2. How does Non-IID change that alignment?

It collapses, monotonically and substantially, as Dirichlet α shrinks:
mean `cos(honest, g_ref)` goes `0.959 (iid) → 0.486 (α=1.0) → 0.360
(α=0.5) → 0.178 (α=0.1)`. This is not just added noise around a stable
center — the 5th percentile goes **negative** at every tested Non-IID
level (`α=1.0: -0.093`, `α=0.5: -0.103`, `α=0.1: -0.362`), meaning a
real, non-trivial fraction of genuinely honest clients are already
anti-correlated with `g_ref` well before any attacker is introduced.
`cos_ref_vs_round.png` shows this is not a transient early-training
artifact that trains away: each partition setting's trajectory plateaus
by round ~5–6 and stays there for the rest of training (IID plateaus
near 0.95–0.98; α=1.0 near 0.63–0.69; α=0.5 near 0.48–0.51; α=0.1 near
0.25–0.28) — the gap between regimes is structural, not transient.
Pairwise honest-client cosine shows the same collapse even more starkly:
mean `cos(gᵢ,gⱼ)` goes `0.979 → 0.295 → 0.140 → -0.009` — at α=0.1,
honest clients are on average **orthogonal to each other**, not just to
`g_ref`.

### 3. How sensitive is `g_ref` to trusted-server dataset size?

Highly sensitive, and the sensitivity itself depends on heterogeneity in
a way that matters for how to read it. Under IID, `g_ref` representativeness
improves sharply and monotonically with size: mean alignment goes `0.399
(≈1% of train) → 0.829 (≈2.5%) → 0.916 (≈5%) → 0.960 (≈10%, our locked
default)`, with std shrinking from `0.153` to `0.028` — a clear case where
"use a bigger trusted reference set" is a real, working lever
(`gref_size_sensitivity.png`). Under severe Non-IID (α=0.1), the same
4× size increase (899→8999 samples) only moves mean alignment from
`0.001` to `0.177` — size helps, but nowhere near enough to reach even
the IID *small*-reference-set number. **The bottleneck under severe
Non-IID is not reference-set size** — it's that no single client's local
update resembles an IID-like aggregate direction at all, so a bigger
`g_ref` (which stays IID-like-balanced at every size tested) cannot
become "more similar" to a client that is, by Dirichlet(0.1)
construction, concentrated in one or two classes almost exclusively
(confirmed by the STEP 5 dataset audit's per-client class counts, e.g. a
client with zero samples in 2 of 9 classes). Bigger `g_ref` fixes a
sampling-noise problem; it does not fix a structural-mismatch problem.

### 4. How stable is Sign Consensus among honest clients?

More stable than Cosine in relative terms, but still degrades
substantially. Mean sign-consensus score: `0.875 (iid) → 0.706 (α=1.0) →
0.653 (α=0.5) → 0.604 (α=0.1)` (`sign_score_by_partition.png`). Unlike
cosine, the mean never crosses below 0.5 even in the worst tested
regime, and the 5th percentile only dips to `0.486` at α=0.1 (barely
below half, vs. cosine's 5th percentile of `-0.362` in the same
condition). Sign agreement is a coarser signal (per-coordinate ±1 vs.
continuous direction), which plausibly explains why it tolerates
heterogeneity better here — but "better than cosine" is not the same as
"stable": a drop from 0.875 to 0.604 is still a large shift in the
honest-score distribution that any fixed κ would have to accommodate.

### 5. Is there evidence that a single global cosine threshold could cause honest-client false positives?

Yes, clearly. Take the IID honest distribution (p5 = `0.907`) as a
plausible calibration target — a `ρ` picked to be reasonably permissive
for IID honest clients would sit somewhere in the 0.85–0.95 range. Applied
unchanged to α=0.1 clients, whose own 75th percentile is only `0.446` and
whose median is `0.218`, that same `ρ` would reject the overwhelming
majority of genuinely honest α=0.1 clients — a severe false-positive
rate, every round, with no attacker present. The only way to avoid that
with a *single* global `ρ` is to calibrate loose enough to admit the
α=0.1 distribution's lower tail (down to `-0.362`), at which point the
threshold accepts almost any direction at all under IID conditions too,
which is exactly the "a global cosine threshold becomes meaningless
under heterogeneity" failure mode this question was asking about. This
is evidence — not yet a conclusion, since no attack or calibration has
been run — that Cosine filtering likely needs a Non-IID-aware (e.g.
per-`(dataset, partition, alpha)`) threshold rather than one global `ρ`,
which the locked calibration protocol (§8) already does per
`(dataset, partition, alpha)` — this finding is a concrete argument for
why that per-setting calibration granularity is the right call, not an
over-engineering choice.

## What this study does not claim

- No defense-quality number (TPR/FPR/accuracy-under-attack) — that
  requires the attack suite and calibration step, explicitly deferred.
- No claim about PathMNIST test accuracy beyond "the pipeline learns,"
  already established in STEP 5 and not re-litigated here.
- No claim about Fed-ISIC2019 — a dataset with natural (not synthetic
  Dirichlet) heterogeneity may or may not show the same pattern; that is
  STEP 6+ work.
- The g_ref size sub-study only covers `iid` and `dirichlet(α=0.1)` (the
  two ends of the tested spectrum, per the locked design) — the
  intermediate α=1.0/0.5 settings' size-sensitivity was not measured, to
  bound compute; the two extremes already establish the qualitative
  pattern (IID: size matters a lot; severe Non-IID: size helps only
  marginally).
