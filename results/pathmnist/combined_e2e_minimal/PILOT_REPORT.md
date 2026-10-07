# STEP 3 — Pilot Report (24 runs: 2 partitions x 4 conditions x seed 42 x 3 candidates)

## Verdict: PROCEED to the full 144-run benchmark, C3/C4/C4-drift-aware
**unchanged** from their locked offline definitions. One critical-looking
gate (Gate 7) was investigated in depth and root-caused to a genuine,
implementation-faithful candidate behavior — not a bug — and is preserved,
not patched, per the explicit "do not redesign based on results" rule. Two
implementation bugs in the *analysis scripts* (not the defenses) were found
and fixed during this pilot pass — exactly what the pilot step exists to
catch before committing GPU time to the full 144 runs.

## Bugs found and fixed during pilot analysis (scripts, not defenses)

1. **`combined_e2e_analysis.py` non-idempotent manifest merge**: rerunning
   the analysis script appended duplicate `_x`/`_y` detection columns into
   `run_manifest.csv` instead of replacing them. Fixed by dropping any
   previously-merged detection columns before re-merging. Verified
   idempotent over 3 consecutive reruns.
2. **`self_influence_audit()` used `attack_active` instead of
   `delta_modified`**: `attack_active` is role+round only and is `True`
   under `no_attack` too (same gating, identity perturbation) — using it
   would have silently mixed genuinely-perturbed malicious rows with
   unperturbed `no_attack` rows into the self-influence diagnostic, exactly
   the ambiguity the prompt explicitly asked to avoid repeating from Trace
   B. Fixed to filter on `delta_modified` (90 rows, down from 120 — the 30
   `no_attack`-condition rows correctly excluded).

## Gate-by-gate

**GATE 1 (no unexplained NaN/Inf):** PASS. `run_manifest.csv`: 0 NaN in
raw columns (6 expected NaN in `malicious_tpr` for the 6 `no_attack` runs,
where there are 0 malicious-active rows by construction — explained, not a
defect). `round_metrics.csv`: 0 NaN. `client_decisions.csv`: NaN only in
candidate-specific `extra` columns that a given candidate doesn't compute
(e.g. `sign_peer_outlier` is NaN for all 1000 `c3_static` rows, which
doesn't use peer-rank; `tau_without_self`/`median_norm`/`mad_norm`/
`tau_static_fallback` are NaN for the 2000 `c3_static`+`c4_static` rows,
which don't use the drift-aware gate) — every NaN is explained by which
candidate produced the row, not missing/corrupted data. 0 Inf anywhere.

**GATE 2 (all expected logs exist):** PASS. 24/24 runs x 25 rounds = 600
round rows (exact). 24 x 25 x 5 clients = 3,000 client rows (exact). 8
`c4_drift_aware` runs x 25 rounds = 200 adaptive-threshold rows (exact).

**GATE 3 (causal audit, no future-information leakage):** PASS, verified
pre-training in `IMPLEMENTATION_AUDIT.md` + `UNIT_TEST_RESULTS.md` (19/19).
`_drift_tau`'s signature structurally cannot access any round but the one
passed to it.

**GATE 4 (ground-truth labels never enter defense decisions):** PASS,
verified by code inspection (`IMPLEMENTATION_AUDIT.md` §6) — `ctx` passed to
every candidate's `aggregate()` contains only τ/ρ/κ/g_ref, never
`malicious_client_ids` or any label.

**GATE 5 (adaptive threshold logged every round):** PASS — 200/200 expected
`c4_drift_aware` round-level adaptive rows present, each with
`median_norm`/`mad_norm`/`tau_drift`/`tau_static_fallback`/
`fallback_triggered`.

**GATE 6 (self-influence auditable):** PASS — 90 attacked-round diagnostic
rows in `self_influence_audit.csv` (2 partitions x 3 real-attack conditions
x 15 attacked rounds, `c4_drift_aware` only). Pilot-stage self-influence
summary: `relative_shift` mean=0.100, median=0.068, max=0.483 (the
attacker's own presence shifts its judging threshold by a bounded, nonzero
amount, consistent with the median/MAD breakdown-point argument in
`IMPLEMENTATION_AUDIT.md`); `malicious_norm_over_threshold` median=61x,
i.e. even the *self-inflated* threshold is nowhere close to letting
`large_norm` updates through (confirmed also by 0/2 `large_norm` pilot runs
ever accepting a malicious update — see Gate 7 data table below).

**GATE 7 (no_attack runs produce plausible FL learning trajectories):
CONDITIONAL PASS, with a significant finding, investigated and root-caused,
not silently patched.** `c4_static` at `dirichlet_a0.1`/`no_attack` is
**not** plausible by the literal reading of this gate: `n_accepted=0` from
round 1 onward for all 25 rounds, i.e. every single round's update collapses
to the zero vector and the global model literally freezes at its round-0
state (accuracy/loss bit-identical for all of rounds 1-24: `0.171727` /
`2.23445`). **Root cause, verified via per-client score inspection (not
assumed):** at round 1, 4/5 honest clients have `norm_score > tau` (the
*locked, unchanged* Regime-A static τ=0.547492 for this partition) and the
5th fails `cos_fail` — i.e. **every honest client fails at least one gate
simultaneously**, which under C4Static's `OR` structure
(`reject := norm_fail OR cos_fail OR sign_reject`) rejects all 5 clients at
once. Once the aggregate is the zero vector, the model never updates, so
the next round's local training starts from the identical frozen weights
and reproduces a similar (or worse) rejection pattern — a **self-perpetuating
lock-in**, not a one-off fluke. **This is not an implementation bug**: the
candidate logic matches `CANDIDATE_DEFINITIONS.md` exactly (re-verified
line-by-line against `benchmark/defenses/combined.py`), and τ/ρ/κ are the
unmodified, already-locked Regime-A values shared with the original
single-method Norm/Cosine defenses. It is a genuine property of combining
Norm+Cosine with `OR` logic once the *real* trajectory's honest gradient
norms exceed a threshold calibrated on a short, separate calibration
simulation — a failure mode **invisible to `combined_design_v1`'s offline
replay**, because Trace B's Coordinate-Median-driven trajectory never
collapses by construction, so it could never surface this interaction
between Combined's own acceptance decisions and the trajectory they then
produce. **Per the explicit rule "if a rule fails, preserve the failure" /
"do not redesign based on these results," `c4_static` is carried into the
full 144-run benchmark completely unchanged.** This is flagged as one of
the two headline pilot findings, not swept into a footnote.

Critically, **`c4_drift_aware` does *not* reproduce this collapse** at the
same partition/condition (`n_accepted` stays between 2-5 every round, final
accuracy 0.234 vs `c4_static`'s frozen 0.172) — because its adaptive,
per-round Norm gate does not inherit the static gate's early-round
mismatch. This is the first real end-to-end (not offline-replay) evidence
that the drift-aware refinement's benefit is not merely an FPR-percentage
artifact of Trace B — it changes the *qualitative* outcome (learns at all,
vs. permanently frozen) for a candidate that otherwise shares the identical
directional-stage logic.

**GATE 8 (identical init/partition across candidates, same seed):** PASS
by construction — `_model(n_classes, seed)` and
`build_or_load_partitions(..., seed)` are called identically regardless of
`candidate` (see `scripts/combined_e2e_benchmark.py::run_one`/`main`); round-0
*accuracy* legitimately differs across candidates at `dirichlet_a0.1`
(0.1717 for c3/c4-static, 0.1864 for c4-drift-aware) because each
candidate's own round-0 accept/reject decision can differ even from
identical initial weights — this is expected candidate behavior, not an
initialization mismatch. At `iid`, round-0 accuracy happens to match across
all three (0.0586) because round-0 acceptance is unanimous there.

**GATE 9 (consistent attack generation up to trajectory-caused
differences):** PASS by construction — the attack RNG stream position at a
given (seed, round, client) depends only on the *number* of prior local
-training draws consumed, which is identical across candidates (same K=50
steps every round regardless of accept/reject outcomes or weight values);
`large_norm`'s injected noise is therefore the same draw across candidates
at a given (seed, round). Attacks whose output depends on the actual
client delta (`directional_poisoning`) legitimately differ across
candidates once their trajectories diverge — exactly the documented,
expected exception.

**GATE 10 (feasible runtime):** PASS — 1,270s (21.2 min) for 24 runs,
~52.9s/run. Full 144 runs projected at ~127 min (~2.1 hours), consistent
with the pre-pilot estimate from a 3-round smoke test.

## Catastrophic safety (pilot-scale, `large_norm` only)

| candidate | large_norm runs | runs with >=1 accepted | P(diverge\|>=1 accept) | runs with 0 accepted | P(diverge\|0 accept) |
|---|---|---|---|---|---|
| c3_static | 2 | 0 | n/a | 2 | 0.000 |
| c4_static | 2 | 0 | n/a | 2 | 0.000 |
| c4_drift_aware | 2 | 0 | n/a | 2 | 0.000 |

**Zero catastrophic misses across all three candidates in the pilot** — no
`large_norm` update was ever accepted, in either partition, by any
candidate. (`p_diverge_given_0_accept = 0.000` and no numerical divergence
flag was ever set anywhere in the pilot, including during `c4_static`'s
collapse — the collapse is a *stall*, not a numerical blow-up.)

## Honest FPR / malicious TPR snapshot (pilot-scale, informs but does not
replace the full 144-run analysis)

`c4_drift_aware` has the lowest honest FPR in **every one of the 8 pilot
conditions it ran** (e.g. iid/no_attack: 0.048 vs 0.272 for both static
candidates; `dirichlet_a0.1`/no_attack: 0.032 vs c3's 0.440 and c4-static's
0.992), while matching or exceeding their malicious TPR in 6 of 8 attacked
conditions (one exception: `dirichlet_a0.1`/directional_poisoning, where
c4-drift-aware's trajectory becomes volatile — oscillating between ~0.10
and ~0.33 accuracy rather than cleanly converging or freezing — ending at a
lower final accuracy (0.103) than c3-static (0.264) despite its own lower
FPR/higher TPR; full `round_metrics.csv` shows this is oscillation, not a
single collapse). This volatility is the second headline pilot finding,
alongside c4_static's collapse, and both are carried forward for deeper
characterization in the full 144-run analysis (3 seeds will show whether
either is seed-specific or systematic).

## Decision

All 10 gates pass or conditionally pass with a fully root-caused, non-bug
explanation. Proceeding to the full 144-run benchmark with all three
candidates **unmodified**. The 24 valid pilot runs are reused (not rerun)
in the full dataset per the prompt's explicit instruction, since no
code/threshold/rule change occurred after the pilot.
