# STEP 2 — Combined E2E unit/synthetic decision tests

**19/19 PASS**. Deterministic, hand-constructed synthetic vectors only -- no PathMNIST data, no attack-outcome tuning. Thresholds used for these tests (tau=1.0, rho=0.0, kappa=0.5) are arbitrary fixed test constants, not the locked calibrated benchmark thresholds.

| test | result | detail |
|---|---|---|
| huge_norm_rejected[C3Static] | PASS | accepted=[False, True, True, True, True] |
| huge_norm_rejected[C4Static] | PASS | accepted=[False, True, True, True, True] |
| huge_norm_rejected[C4DriftAware] | PASS | accepted=[False, True, True, True, True] |
| normal_norm_passes_magnitude_stage | PASS | norm_fail=[False, False, False, False, False] |
| cosine_anomaly_flagged_by_cos_fail | PASS | cos_score=-0.945 rho=0.0 |
| C3_AND_rule_may_spare_pure_cosine_anomaly | PASS | C3 accepted[0]=True (AND rule: needs cos_fail AND sign_fail) |
| C4_OR_rule_rejects_pure_cosine_anomaly | PASS | C4 accepted[0]=False (OR rule: cos_fail alone rejects) |
| sign_anomaly_flagged_and_peer_outlier | PASS | sign_score=0.000 kappa=0.5 peer_outlier=True |
| sign_anomaly_rejected_by_C4 | PASS | accepted[0]=False |
| no_anomaly_all_accepted[C3Static] | PASS | accepted=[True, True, True, True, True] |
| no_anomaly_all_accepted[C4Static] | PASS | accepted=[True, True, True, True, True] |
| no_anomaly_all_accepted[C4DriftAware] | PASS | accepted=[True, True, True, True, True] |
| drift_tau_computed_from_all_5_incl_attacker | PASS | tau6=0.4038 manual=0.4038 (median/MAD computed over all 5 inputs, attacker not excluded) |
| zero_mad_falls_back_to_static_tau | PASS | tau7=1.0 static_tau=1.0 |
| near_zero_mad_gives_finite_threshold | PASS | tau8=1.000000 |
| identical_norms_mad_zero_fallback | PASS | tau9=1.0 (MAD=0 -> fallback, consistent with Test 7) |
| extreme_outlier_only_flags_outlier | PASS | norm_fail=[False, False, False, False, True] tau_used=0.3519 |
| drift_tau_signature_has_no_round_or_history_param | PASS | params=['norm_scores', 'static_tau'] (only this round's norms + static fallback -- no access to other rounds) |
| reproducible_bit_identical | PASS | two calls with identical inputs produce identical accept decisions and update |
