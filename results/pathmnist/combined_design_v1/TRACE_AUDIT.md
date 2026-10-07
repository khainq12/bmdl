# STEP 1 — Trace B Audit for Combined Design Offline Replay

Source: `results/pathmnist/analysis_v2/trace_B_complementarity_pooled3seeds.csv`
(unmodified — this step reads it, does not rewrite it).

## Verdict: PASS — Trace B supports valid same-update offline replay, with one
labeling nuance documented below and corrected for in this analysis.

## Checks performed

| Check | Result |
|---|---|
| Row count | 9,000 exactly (3 seeds × 4 partitions × 6 attacks × 25 rounds × 5 clients) |
| Seeds present | {42, 43, 44}, 3,000 rows each |
| Partitions present | iid, dirichlet_a1.0, dirichlet_a0.5, dirichlet_a0.1 |
| Attacks present | no_attack, large_norm, low_norm, full_sign_flip, directional_poisoning, sparse_coordinate_attack |
| Round range | 0–24 (25 rounds, matches locked N_ROUNDS) |
| Client ids | 0–4 (N_CLIENTS=5) |
| Duplicate (seed, partition, attack, round, client_id) keys | 0 |
| NaN in seed/partition/attack/round/client_id/is_malicious/norm_score/cosine_score/sign_score/accept_\*_A | 0 |
| NaN in accept_\*_B | 2,250 (expected — Regime B is `iid_reference_frozen`, not applicable to the `iid` partition itself; 3 seeds × 1 partition × 6 attacks × 25 rounds × 5 clients = 2,250) |

## Per-row score → decision consistency (Regime A)

Trace B stores `accept_norm_A` / `accept_cosine_A` / `accept_sign_A` as
booleans but does not store τ/ρ/κ per row. These are deterministic given
`partition` under Regime A (condition-specific, constant across round/seed),
so they were reconstructed by joining
`results/pathmnist/benchmark_v2_corrected/calibration_results.csv`
(`calibration_regime == "A_condition_specific_oracle"`) on `(defense, partition)`,
and re-deriving the accept decision from `norm_score`/`cosine_score`/`sign_score`
directly:

- `accept_norm_A  == (norm_score  <= tau_A)`
- `accept_cosine_A == (cosine_score >= rho_A)`
- `accept_sign_A   == (sign_score  >= kappa_A)`

**Verified exact match (100%, 0 mismatches) for all 4 partitions.** This
confirms the stored accept/reject columns are internally consistent with the
stored scores and the locked calibration thresholds — no silent drift between
what was logged and what the thresholds actually imply. τ/ρ/κ for Regime A per
partition, and the single frozen Regime B values, are reproduced in
`CANDIDATE_DEFINITIONS.md` (STEP 3).

## Attack/honest label consistency

- Malicious rows only ever occur at `client_id == 0` (matches
  `MALICIOUS_RATIO=0.2` × 5 clients = 1 malicious client) and only at
  `round >= 10` (matches `ATTACK_FROM_ROUND`). 0 malicious rows before round 10.
- **Labeling nuance (documented, not a defect in the source script):**
  `is_malicious` in Trace B reflects *role assignment* (`client_id in
  malicious_client_ids and round >= ATTACK_FROM_ROUND`), independent of which
  attack function is applied. For `attack == "no_attack"`, `apply_attack` is
  the identity function (`benchmark/attacks/suite.py::no_attack`), so the 180
  rows where `attack == "no_attack" and is_malicious == True` carry an
  **unperturbed, honest-equivalent delta** despite the `is_malicious=True` flag.
  This affects only `no_attack`; all 5 other attacks have a real perturbation
  whenever `is_malicious=True`.
- Consequence for this analysis: a naive honest filter (`~is_malicious`) would
  **under-count true-honest rows by exactly 180/9,000 (2.0%)** — all drawn from
  client 0, rounds 10–24, under `no_attack`. The malicious-side filter used in
  the prior STEP 7 analysis (`is_malicious & round>=10 & attack!="no_attack"`)
  was already unaffected by this, since it explicitly excludes `no_attack`.
  **This offline-replay analysis uses the corrected honest definition**
  `honest := (~is_malicious) | (attack == "no_attack")` throughout, recovering
  the 180 rows as honest. This is a data-fidelity correction (using the actual
  delta's provenance, not the role label) made transparently before any
  candidate is scored — it was not tuned per-candidate and does not touch
  `attack != "no_attack"` rows at all.

| Definition | Honest rows | Malicious-active rows (round≥10, attack≠no_attack) |
|---|---|---|
| Naive (`~is_malicious`) | 7,920 | 900 |
| Corrected (used here) | 8,100 | 900 |

## Conclusion

Trace B contains, per row: seed, round, client, attack, partition, honest/
malicious provenance (corrected), and all three raw scores. τ/ρ/κ for both
regimes are exactly reconstructable via a deterministic join (verified, not
assumed). This is sufficient to replay every candidate rule's accept/reject
decision on the identical 9,000 client-round updates used in STEP 7, for both
Regime A and (where defined) Regime B. **Proceeding to STEP 2.**
