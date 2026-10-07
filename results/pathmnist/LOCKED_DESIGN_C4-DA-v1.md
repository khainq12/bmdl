# C4-DA v1 — Locked PathMNIST Design

**Status: LOCKED**, per ChatGPT cross-check of
`results/pathmnist/c4_drift_aware_confirmation/CONFIRMATION_REPORT.md`
(192-run confirmation: 8 seeds × 4 partitions × 6 attacks, 0/192 collapse,
0/480 large_norm catastrophic misses across every individual seed,
full_sign_flip TPR 0.885 newly confirmed, directional_poisoning TPR 0.879,
sparse_coordinate TPR 1.000, low_norm/clean/Non-IID all stable). All 9
criteria set for locking were met. **No further experiments are needed to
decide this design.**

## Definition (implementation: `benchmark/defenses/combined.py::C4DriftAware`, unchanged)

```
Magnitude Safety:
    tau_drift(round) = median(norm_i, all 5 clients this round)
                        + 3.5 * 1.4826 * MAD(norm_i, all 5 clients this round)
                        (fallback to static locked tau if MAD == 0)
    norm_fail(i) := norm_score(i) > tau_drift(round)

Directional / Fine-grained Attestation (C4 hierarchical rule, unchanged
from combined_design_v1):
    cos_fail(i)  := cosine_score(i) < rho       (locked Regime-A rho)
    sign_fail(i) := sign_score(i) < kappa        (locked Regime-A kappa)
    peer_median(i) := median(sign_score(j), j != i, this round)
    sign_reject(i) := sign_fail(i) AND (sign_score(i) < peer_median(i))

Decision:
    reject(i) := norm_fail(i) OR cos_fail(i) OR sign_reject(i)
```

Causal: `tau_drift` uses only the current round's 5 submitted norms — no
future-round information (verified structurally, `IMPLEMENTATION_AUDIT.md`).
Self-influence: the attacker's own norm (when present) contributes to the
statistic that judges it; this is documented, bounded (median/MAD breakdown
point for 5 clients, 1 attacker), and empirically characterized (median
relative shift 16.9%, exploitable region ≤5× the honest median — see
`c4_drift_aware_confirmation/MAGNITUDE_STRESS_TEST.md` — well below real
`large_norm`'s operating region of 10²–10⁴×).

## What "locked" means from here forward

1. **The formula, code, and threshold policy are frozen.** No further edits
   to `C4DriftAware` based on PathMNIST results, Fed-ISIC2019 results, or
   any subsequent benchmark — per the standing "do not redesign based on
   test results" rule that has governed every stage of this research arc.
2. **Fed-ISIC2019 must use this exact locked version unchanged.** If
   cross-dataset performance is weak, that is a valid, reportable
   generalization-limitation finding — not a trigger to retune C4-DA while
   still calling it the same version. A retuned version would need a new
   name (e.g. "C4-DA v2") and would need its own validation cycle, starting
   over from design justification, not silently inheriting v1's evidence.
3. **An important interpretive nuance to preserve, not smooth over** (per
   ChatGPT's explicit correction): `low_norm` TPR = 0 across every run in
   this entire research arc does **not** mean C4-DA (or any single
   signal) defends against low-norm/free-rider attacks. The only
   defensible claim is that **the specific low_norm attack implementation
   tested** (`delta * eta`, `eta=0.01`) did not cause measurable harm in
   any tested configuration. A stronger, adaptive low-norm attack (e.g.
   one that mimics honest-client statistics more closely) remains
   untested and is an explicit limitation / future-work item, not a
   closed question.

## Evidence this design rests on (chronological, each stage building on the
last, nothing skipped)

1. Single-defense characterization (`analysis_v2/`) — identified Norm's
   temporal-drift FPR problem, Cosine's Non-IID fragility, Sign's
   large_norm blind spot, and genuine signal complementarity (STEP 7).
2. Offline design + replay (`combined_design_v1/`) — 7 candidates (C0–C6)
   screened via 9,000-row Trace B; eliminated catastrophically unsafe
   candidates (C1, C5); identified the drift-aware Norm gate's predicted
   FPR benefit.
3. Minimal end-to-end validation (`combined_e2e_minimal/`) — 144 real FL
   runs; confirmed C4-drift-aware eliminates a 25%-of-runs collapse failure
   mode that afflicts both static candidates, while matching or exceeding
   single-signal detection capability and tracking Multi-Krum's utility.
4. Targeted confirmation (`c4_drift_aware_confirmation/`) — 192 more real
   FL runs (8 seeds) + an analytical magnitude stress test; extended every
   finding's evidence base and newly validated `full_sign_flip`.

## Next stages (per ChatGPT's agreed roadmap)

```
C4-DA v1 LOCKED                          ← this document
        ↓
Final PathMNIST evaluation + end-to-end ablation
   (C4-DA vs C4-DA-without-Norm vs -without-Cosine vs -without-Sign;
    fair comparison vs FedAvg/Norm/Cosine/Sign/Median/Multi-Krum)
        ↓
Fed-ISIC2019 (locked C4-DA v1, unchanged)
        ↓
Cross-dataset generalization conclusions
        ↓
Security + limitations
        ↓
Paper/report finalization
```

Per ChatGPT's explicit efficiency instruction: the Final PathMNIST
Evaluation stage must **audit and reuse** the already-valid 594 baseline
runs (`benchmark_v2_corrected/`) + 144 minimal runs
(`combined_e2e_minimal/`) + 192 confirmation runs
(`c4_drift_aware_confirmation/`) wherever a cell already has a valid
result, and only run genuinely missing cells (primarily: the three new
single-signal-removed ablation variants, which do not yet exist in any
prior stage).
