# STEP 2–5 — Design Objectives, Base Signals, Candidate Combined Rules

## STEP 2 — Design objectives (locked before any candidate is evaluated)

G1. Preserve hard protection against catastrophic unbounded-magnitude attacks
    (`large_norm`): STEP 5 of `analysis_v2` found exactly one accepted
    `large_norm` update was sufficient for divergence in every observed case
    (P(diverge | ≥1 accept) = 1.000, n=30, identical in all 3 seeds).

G2. Preserve Cosine's ability to catch global directional attacks
    (`full_sign_flip`, `directional_poisoning`) — the only signal for which
    these two attacks are ever rejected (STEP 7: Norm/Sign never sufficient
    alone for these two attacks, in any of the 3 seeds).

G3. Preserve Sign Consensus's complementary value for localized/sparse
    manipulation (`sparse_coordinate_attack`) — STEP 7: Cosine contributes 0%
    of rejections for this attack in any seed; Norm+Sign is required.

G4. Avoid simply inheriting Norm's high honest false-rejection rate (pooled
    16.1%, vs. Cosine 1.2% / Sign 3.1% — STEP 7 multi-seed table).

G5. Remain reasonably robust as IID → α=1.0 → 0.5 → 0.1 (do not silently
    assume Regime-A condition-specific recalibration is available at
    deployment; Regime B — IID-frozen — is the deployment-realistic case).

G6. Avoid dependence on attack labels at deployment (every candidate below
    is defined using only per-round, per-client observable quantities: raw
    scores, static calibration thresholds, and — for the drift-aware Norm
    variant — the other clients' scores in the same round. None use
    `is_malicious` or `attack` as an input to the decision itself).

G7. Keep the rule interpretable enough to explain in a federated-learning /
    security report (no opaque learned classifier).

G8. Do not optimize against final test attacks in a way that leaks
    attack-test information into threshold or weight calibration. All
    weights/margins/constants chosen below are either (a) the existing
    locked τ/ρ/κ from `calibration_results.csv` (validation-only, already
    locked long before this step), (b) equal weighting, (c) the geometric
    midpoint of a [0,1] normalized score, or (d) a published, off-the-shelf
    robust-statistics constant (Iglewicz & Hoaglin's modified z-score cutoff)
    — never a value fit by scanning attack outcomes.

## STEP 3 — Base signals (unchanged from the existing benchmark)

For client update `g_i` at a given round, with server reference `g_ref`
trained on the disjoint server-reference split:

- **Norm**: `n_i = ||g_i||_2`. Suspicious (fail) if `n_i > tau`.
- **Cosine**: `c_i = cos(g_i, g_ref)`. Suspicious (fail) if `c_i < rho`.
- **Sign Consensus**: `s_i` = existing benchmark coordinate-sign-agreement
  score (`benchmark/defenses/sign_consensus.py`). Suspicious (fail) if
  `s_i < kappa`.

These definitions are **not changed** in this step. τ/ρ/κ are taken verbatim
from `results/pathmnist/benchmark_v2_corrected/calibration_results.csv`
(verified to reproduce Trace B's stored accept/reject columns exactly — see
`TRACE_AUDIT.md`):

**Regime A (condition-specific), by partition:**

| partition | tau (τ) | rho (ρ) | kappa (κ) |
|---|---|---|---|
| iid | 0.065314 | 0.650783 | 0.532159 |
| dirichlet_a1.0 | 0.304172 | -0.514188 | 0.416304 |
| dirichlet_a0.5 | 0.389715 | -0.437277 | 0.400319 |
| dirichlet_a0.1 | 0.547492 | -0.543603 | 0.377527 |

**Regime B (IID-calibrated, frozen, applied to non-IID only):**
τ=0.065314, ρ=0.649566, κ=0.531668 for all of dirichlet_a1.0/0.5/0.1.

Score ranges observed in Trace B (for normalization in STEP 4):
`norm_score ∈ [0.0003, 22809.9]` (unbounded — `large_norm` is isotropic
noise with no magnitude cap), `cosine_score ∈ [-0.995, 0.995]`,
`sign_score ∈ [0.006, 0.929]`.

## STEP 4 — Candidate Combined rules

All candidates below use Regime A (condition-specific) τ/ρ/κ as the primary
view, matching the rest of `analysis_v2`; Regime B is checked separately in
`offline_replay_by_partition.csv` wherever defined (non-IID partitions only).

**C0 — Naive OR (baseline, not expected to be optimal):**
`reject iff norm_fail OR cos_fail OR sign_fail`

**C1 — Majority voting:** `reject iff (norm_fail + cos_fail + sign_fail) >= 2`.
Expected risk: unsafe for `large_norm` whenever Norm is the *only* failing
signal (STEP 7 confirms Sign contributes 0% and Cosine only 25% of the time
for `large_norm` — so Norm-alone rejections, which are the majority of
`large_norm` catches, would be **overturned to accept** under majority vote).

**C2 — Hard Norm gate + directional OR:**
Stage 1: `if norm_fail: reject`. Stage 2: `if cos_fail OR sign_fail: reject`.
Else accept. **This is logically identical to C0** under unchanged static
thresholds — restating `(A) OR (B OR C)` as two stages does not change the
truth table. Replay must demonstrate 0 decision differences from C0; this is
treated as a sanity check on the replay engine, not evidence that two-stage
phrasing alone adds value.

**C3 — Hard Norm gate + directional AND:**
Stage 1: `if norm_fail: reject`. Stage 2: `if cos_fail AND sign_fail: reject`.
Else accept. Expected weakness: a `directional_poisoning`/`full_sign_flip`
update caught by Cosine alone (STEP 7: pooled 20–23% of these attacks'
rejections are Cosine-alone) would **pass** C3's stage 2, since Sign would
not also need to fail. Quantified directly in the replay.

**C4 — Hierarchical/contextual rule**, going beyond C2/C3 by adding a
label-free *peer-relative* confirmation step for Sign (motivated by STEP 4's
finding that Sign's absolute score can be pushed below κ by Non-IID alone,
not just by an attack):
```
Stage 1 (magnitude safety): if norm_fail: reject
Stage 2 (directional):
    if cos_fail: reject                          # strong global anomaly
    elif sign_fail AND sign_score < median(other 4 clients' sign_score
                                            this round):   # peer-relative
                                                             outlier check
        reject
    else: accept
```
The peer-relative condition uses only information already visible to the
server each round (all 5 clients' scores that round) — no attack labels, no
new constant invented for this task; it is a *relative rank* test, which is
the only way to add nuance to C3 without introducing an arbitrary new
threshold.

**C5 — Normalized weighted score (no hard gate):**
```
R_norm  = clip((norm_score - tau) / tau, 0, 1)
R_cos   = clip((rho - cosine_score) / (rho + 1), 0, 1)
R_sign  = clip((kappa - sign_score) / kappa, 0, 1)
R_i = (1/3) R_norm + (1/3) R_cos + (1/3) R_sign
reject iff R_i > 0.5
```
Equal weights and a 0.5 midpoint cutoff are the attack-agnostic defaults
(G8) — explicitly **not** grid-searched against attack outcomes. `R_norm` is
clipped to [0,1] (saturates once the score is 2× over threshold), which is
the mechanism by which C5 is expected to under-protect against
unbounded-magnitude `large_norm`: a norm score of 22,809 contributes the same
`R_norm=1` as a score barely over τ, diluted by averaging with two other
components that are very likely near 0 for a pure-magnitude attack.

**C6 — Safety gate + weighted directional score** (the architecture implied
by the empirical complementarity evidence: Norm = magnitude safety, Cosine =
global direction, Sign = fine-grained direction):
```
Stage 1 (magnitude safety): if norm_fail: reject
Stage 2: D_i = 0.5 R_cos + 0.5 R_sign; reject iff D_i > 0.5
```
`R_cos`/`R_sign` as defined under C5. This candidate is not assumed to win;
it is evaluated on equal footing with the others in STEP 6–8.

## STEP 5 — Norm temporal drift: static vs. drift-aware gate

**Variant A (static, default for all candidates above):** τ fixed per
partition from `calibration_results.csv`, unchanged.

**Variant B (drift-aware, PROPOSED / design-study only, not used to select a
final candidate):** a per-round, label-free robust threshold computed from
the 5 client norm scores actually submitted that round, using the standard
modified z-score cutoff (Iglewicz & Hoaglin, 1993; cutoff 3.5 is their
published convention, not fit for this task):
```
m_r   = median(norm_score of the 5 clients this round)
MAD_r = median(|norm_score_i - m_r|)
tau_drift(r) = m_r + 3.5 * 1.4826 * MAD_r      # 1.4826: MAD→std scale factor
                                                 # for a normal distribution
fallback: if MAD_r == 0, tau_drift(r) := tau (static Regime-A value)
```
This only ever substitutes for τ in the Norm-gate stage of C2/C3/C4/C6 (the
candidates that use Norm as a hard gate); it is **not** applied to C0/C1/C5
in this report, since isolating its effect to the gate candidates is the
direct test of STEP 5's question ("is Norm FPR caused by the hard-gate idea
or by static calibration?"). Results are reported side-by-side (`_driftaware`
suffix) in `offline_replay_summary.csv`, never merged into the primary
Variant-A numbers used for candidate selection.
