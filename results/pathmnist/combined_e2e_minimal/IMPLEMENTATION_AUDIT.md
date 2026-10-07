# STEP 1 — Implementation Audit (before any GPU training)

Implementation: [benchmark/defenses/combined.py](../../../benchmark/defenses/combined.py).
Registered in the shared defense registry (`benchmark/defenses/__init__.py`)
as `c3_static`, `c4_static`, `c4_drift_aware`, alongside the existing six
single-method defenses — reusing the exact same `Defense.aggregate(deltas,
ctx) -> DefenseResult` interface, so the real FL runner can treat Combined
exactly like any other defense (no special-cased training loop).

## 1–4. Pseudocode for all three candidates (reproduced from
`results/pathmnist/combined_design_v1/CANDIDATE_DEFINITIONS.md` — **not
redesigned** for this stage)

```
# shared signals, computed identically for all three from the delta
# actually submitted this round:
n_i = ||delta_i||_2                          # Norm
c_i = cos(delta_i, g_ref)                    # Cosine
s_i = sign_consensus_score(delta_i, all deltas this round)   # Sign

norm_fail(i)  := n_i > tau
cos_fail(i)   := c_i < rho
sign_fail(i)  := s_i < kappa

# C3Static:
reject(i) := norm_fail(i) OR (cos_fail(i) AND sign_fail(i))
# tau/rho/kappa: locked Regime-A calibration values (static, unchanged).

# C4Static:
peer_median(i) := median(s_j for j != i, this round)   # leave-one-out
sign_reject(i) := sign_fail(i) AND (s_i < peer_median(i))
reject(i) := norm_fail(i) OR cos_fail(i) OR sign_reject(i)
# tau/rho/kappa: locked Regime-A calibration values (static, unchanged).

# C4DriftAware: identical directional stage to C4Static; ONLY the Norm
# gate's threshold changes, from static tau to a per-round statistic:
m_r   := median(n_j for ALL 5 clients this round)       # incl. attacker
MAD_r := median(|n_j - m_r| for ALL 5 clients this round)
tau_drift(r) := m_r + 3.5 * 1.4826 * MAD_r   if MAD_r != 0
             := tau (static fallback)         if MAD_r == 0
norm_fail(i) := n_i > tau_drift(r)             # replaces static norm_fail
reject(i) := norm_fail(i) OR cos_fail(i) OR sign_reject(i)
```

accepted = ~reject; `update = mean(delta_i for i where accepted(i))`,
or the zero vector if every client is rejected (same convention as every
existing single-method detector defense — `NormFilter`/`CosineFilter`/
`SignConsensus`, see `benchmark/defenses/norm.py` etc.).

## 5. Threshold values and score direction — verified, not re-derived

τ/ρ/κ are read from `results/pathmnist/benchmark_v2_corrected/calibration_results.csv`
(Regime A, condition-specific per partition) exactly as the six existing
single-method defenses already do — this script does not recompute or
retune them. Direction convention (`norm_fail := score > tau`, `cos_fail :=
score < rho`, `sign_fail := score < kappa`) is copied verbatim from
`benchmark/defenses/{norm,cosine,sign_consensus}.py` and from
`CANDIDATE_DEFINITIONS.md` STEP 3 — unchanged in this stage.

## 6. No attack labels enter deployment decisions — verified

`C3Static.aggregate`, `C4Static.aggregate`, `C4DriftAware.aggregate` in
`benchmark/defenses/combined.py` take only `deltas` (the actual submitted
updates) and `ctx` (τ/ρ/κ, `g_ref`) as input. None of the three reads
`is_malicious`, `malicious_client_ids`, or any ground-truth field — the
runner script does not pass them into `ctx`. This is identical to how the
six existing defenses are already called (see `run_combo()` in
`scripts/pathmnist_benchmark_v2.py`); Combined changes nothing about this.

## 7. No future information enters adaptive state — verified

`_drift_tau(norm_scores, static_tau)` (the only function computing the
adaptive threshold) takes a single round's 5 norm scores and the static
fallback constant — nothing else. Its signature has no `round` index, no
history buffer, no reference to any other round's data (verified
mechanically in `scripts/combined_e2e_unit_tests.py`, test
`drift_tau_signature_has_no_round_or_history_param`). **There is no
persisted adaptive state object carried between rounds at all** — the
"adaptive state" at round *t* is recomputed from scratch, purely as a
function of round *t*'s own 5 submitted norms. This is the strictest form
of causality: round *t*'s decision cannot even in principle depend on any
other round, because the function has no mechanism to receive other
rounds' data.

**This directly resolves ChatGPT's "Critical Rule #2 / current-round vs.
previous-round" distinction**: the offline design (`combined_design_v1`)
and this implementation both use **current-round (B)** statistics, not
previous-round (A) ones — preserved unchanged for reproducibility, exactly
as the review requested ("If the existing offline C4-drift-aware definition
used current-round statistics, preserve that as a named variant... but
document its self-influence property"). **This means the attacker's own
norm, when present, contributes to the median/MAD used to judge it in the
same round** — a real self-influence property, not a flaw hidden by
causality. STEP 9 (`self_influence_audit.csv`) quantifies exactly how much
this matters, using a leave-one-out diagnostic (`tau_without_self`) that is
computed and logged for every round but **never fed back into the
accept/reject decision** (verified: `extra["tau_without_self"]` is
populated after `reject`/`accepted` are already finalized in
`C4DriftAware.aggregate` — see the source).

## 8. Same client update scored by all required components — verified

All three candidates compute `norm_score`, `cosine_score`, `sign_score` from
the *same* `deltas` list in a single `aggregate()` call — there is no
per-signal re-sampling or re-training; this mirrors Trace B's own
same-update-scored-by-all-signals design from `combined_design_v1`.

## Self-influence breakdown-point argument (addressing the threat-model
concern directly, not just procedurally)

With `MALICIOUS_RATIO=0.2` (1 malicious of 5 clients, unchanged from the
existing protocol), the median has a breakdown point of ⌊(5-1)/2⌋=2 — a
single outlier cannot move the median of 5 values outside the honest
range. MAD (itself a median of absolute deviations) has the same
breakdown-point property. **One attacker among five cannot, by
construction, arbitrarily move `tau_drift` — but it CAN shift it by a
bounded amount**, since the attacker's deviation from the honest median
still occupies one of the 5 ranked deviation slots. STEP 9 measures this
bounded shift empirically (`delta_threshold`, `relative_shift`) rather than
asserting it is negligible. **This guarantee is specific to
`MALICIOUS_RATIO <= ~0.4`** (beyond which median/MAD breakdown is no
longer guaranteed) — stated here as an explicit boundary condition of the
mechanism, not assumed silently.
