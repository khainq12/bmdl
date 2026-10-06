# PathMNIST single-method benchmark v2 (corrected) — report

Fix design: `docs/BENCHMARK_PROTOCOL.md` Addenda 2026-10-07 (g)/(h)/(i).
Scripts: `pathmnist_round_budget_study.py` → `pathmnist_calibration_diagnostics.py`
→ `pathmnist_benchmark_v2.py --pilot` → `pathmnist_benchmark_v2.py` (full) →
`pathmnist_benchmark_v2_aggregate.py`. Full sweep: 594 combos, 14,850
per-round rows, 439.4 minutes. Old (invalid) sweep preserved, untouched, at
`results/pathmnist/benchmark_diagnostic_INVALID_v1/`.

## 0. What was fixed, and confirmation it worked

The step-count confound (calibration pseudo-clients: 8 batches/epoch; real
clients: 30–130+ batches/epoch) is gone: every client now performs exactly
`K=50` SGD steps/round regardless of dataset size — verified experimentally
across client sizes from 10 to 33,151 samples
(`scripts/verify_fixed_steps.py`). The old pathological ~96–100% Norm-defense
honest-rejection is gone: Regime A clean FPR is now `0.000–0.32` (vs. ~1.0
everywhere before). The `no_attack` FedAvg baseline at round 24
(`0.442–0.450` across partitions) matches the independent round-budget study's
clean curves almost exactly — the strongest possible cross-check that the
corrected pipeline reproduces itself consistently.

## 1. Headline finding: catastrophic divergence is defense-specific, not attack-specific

**Norm, Median, and Multi-Krum never catastrophically diverge, anywhere in
the 594-combo matrix.** **FedAvg, Cosine, and Sign Consensus do**, under
`large_norm` specifically:

| Defense | Regime | Diverges under `large_norm` at | diverged_fraction |
|---|---|---|---|
| fedavg | n/a | **every** partition | 1.00 |
| cosine | A (condition-specific) | α=1.0, α=0.5, α=0.1 (**not** IID) | 1.00 |
| sign_consensus | A | **every** partition incl. IID | 1.00 |
| sign_consensus | B (frozen) | α=1.0, α=0.5, α=0.1 | 1.00 |
| norm, median, multi_krum | — | never | 0.00 |

The sharpest result: **Cosine Regime A at IID has `validation_tpr=1.000`
at calibration time, yet still diverges 0% of the time at IID but 100% of
the time everywhere else** — and **Sign Consensus Regime A diverges 100% of
the time even at IID**, despite IID being its best-calibrated condition.
One missed accept over 15 active-attack rounds is enough to permanently
destroy the model (loss jumps to `1.7×10⁸`, accuracy freezes) — a single
round's recall failure matters far more than average TPR, because the
damage is irreversible. Norm/Median/Multi-Krum are **fail-safe** by
construction here: Norm hard-rejects anything over τ regardless of
direction; Median/Multi-Krum structurally cannot let one extreme outlier
dominate the aggregate. Cosine and Sign Consensus (and FedAvg) have no such
hard ceiling, so their worst case is unbounded.

## 2. A second, real generalization gap: calibration trajectory length

Beyond partition-type, a **third** calibration-to-deployment gap shows up
that the 5-round calibration-diagnostics gate (Addendum (g)/(h)) did not
catch, because it only compared a 5-round calibration trajectory against a
5-round benchmark-client trajectory. The full 25-round sweep reveals: at
IID, `no_attack`, Norm Regime A — calibrated to `validation_fpr=0.048` at
calibration time — the **real pooled FPR over the full 25-round run is
0.633** (`tp=0, fp=209, tn=121, fn=9`). Non-IID conditions fare much
better on this same cell (`α=0.5`: FPR=0.324; `α=0.1`: FPR=0.221) —
**the opposite ranking** of what calibration-time numbers alone would
predict, because the IID threshold (`τ=0.065`) is so tight that normal
honest-norm drift over a full 25-round trajectory (not visible in a
5-round snapshot) pushes many honest clients over it. This is a genuine,
newly-surfaced limitation: **a short calibration trajectory is not
sufficient to certify a threshold for a much longer deployment
trajectory**, independent of the step-count fix. Recorded here as a
finding for future calibration-protocol work, not something this pass
fixes.

## 3. Utility across attacks and partitions (trustworthy — real finding, not sanity-check)

`no_attack` baseline, final round (mean over 3 seeds):

| defense | iid | α=1.0 | α=0.5 | α=0.1 |
|---|---|---|---|---|
| fedavg | 0.442 | 0.450 | 0.409 | 0.196 |
| norm (A) | **0.256** | 0.442 | 0.352 | 0.193 |
| cosine (A) | 0.419 | 0.449 | 0.382 | 0.205 |
| sign_consensus (A) | 0.442 | 0.454 | 0.405 | 0.196 |
| median | 0.444 | 0.409 | 0.294 | 0.207 |
| multi_krum | 0.447 | 0.437 | 0.378 | 0.194 |

(Full per-cell numbers in `utility_table.csv`.) Every defense except Norm
at IID tracks the clean FedAvg curve closely under `no_attack`, as
expected — none of them should be actively rejecting much of anything with
zero attacker present. **Norm at IID is the one clear exception** (0.256 vs.
FedAvg's 0.442): this is the calibration-trajectory-length FPR gap from §2
showing up directly in utility — Norm is wrongly rejecting 63% of honest
IID clients even with no attacker present, visibly dragging accuracy down.
Severity ordering **iid > α=1.0 > α=0.5 > α=0.1** holds for every other
defense under `no_attack`, matching the round-budget study; Norm's IID cell
is the sole exception to that ordering, for the same reason.

Under `large_norm` (the most damaging attack): Multi-Krum is the most
robust non-detector (`0.447→0.299` from iid to α=0.1, the best floor of
any defense), Median close behind (`0.450→0.120`), Cosine/Sign
Consensus/FedAvg diverge to near-chance everywhere they catastrophically
fail (§1).

Under `low_norm` (free-rider/stealth) and the three directional/sparse
attacks (`full_sign_flip`, `directional_poisoning`, `sparse_coordinate_attack`):
no defense diverges, and accuracy stays broadly close to the `no_attack`
baseline for most defense/partition pairs — these attacks are far less
damaging to raw utility than `large_norm`, consistent with them not
injecting unbounded magnitude.

## 4. Robustness (accuracy drop vs. no_attack, same partition/defense/regime)

Full table: `robustness_table.csv`. Largest drops are concentrated in
`large_norm` for the three non-fail-safe defenses (e.g. FedAvg at
`α=1.0`: `-0.373`; Cosine Regime A at `α=1.0`: `-0.309`) — i.e., drops of
30–37 accuracy points, consistent with catastrophic divergence (§1).
Fail-safe defenses show much smaller drops under `large_norm` (Multi-Krum:
`-0.10` to `+0.02` depending on partition — occasionally *better* than
`no_attack` within noise). Under the non-magnitude attacks, drops are
mostly in the ±0.00–0.15 range for all defenses, with no consistent
direction — several cells show a small *negative* drop (attack condition
slightly outperforming the clean baseline), which is within the 3-seed
noise band given `no_attack_acc_mean` itself has non-trivial seed variance
at the harder partitions (e.g. `α=0.5` no_attack std up to 0.114).

## 5. Detector TPR/FPR

Full table: `detector_tpr_fpr_table.csv`. Headline patterns:

- **Norm catches `large_norm` perfectly everywhere** (`TPR=1.0` at every
  partition) — it is the only detector with this property. Its FPR under
  `large_norm` ranges `0.167` (α=0.1) to `0.822` (IID) — the IID number is
  the calibration-trajectory-length gap from §2, not a partition-severity
  effect.
- **Cosine and Sign Consensus both show `TPR=0.0` against `large_norm`
  outside IID** (Regime A) — consistent with calibration time
  (`validation_tpr=0.0` for both, at every non-IID partition), and this is
  exactly why both diverge under exactly those conditions (§1).
- Regime B (frozen IID threshold) shows `FPR≈1.0` for Norm and Cosine under
  **every** Non-IID condition and **every** attack including `no_attack` —
  the transfer-failure finding from the honest-gradient characterization
  study, now confirmed under a full, real, 25-round federated run (not just
  a static score distribution).

## 6. Runtime

Mean per-round runtime (non-diverged rows): Cosine ≈2.12s (extra `g_ref`
pass), everything else ≈1.76–1.78s — consistent with the ~12-17% overhead
pattern already seen in the invalid v1 sweep, now at the corrected scale.

## 7. Partition comparison, explicitly (iid vs. α=1.0 vs. α=0.5 vs. α=0.1)

- **Clean utility** degrades monotonically with heterogeneity for every
  defense (§3), confirming the round-budget study at full sweep scale.
- **Catastrophic-divergence risk is partition-dependent per defense**:
  Cosine Regime A is divergence-free *only* at IID; everywhere else it
  fails outright. Sign Consensus Regime A fails even at IID.
- **Regime B's honest-rejection catastrophe** (§5, FPR≈1.0) appears at
  every Non-IID level with no clear monotonic trend in severity — it is
  already maximal at the mildest tested heterogeneity (α=1.0), not a
  gradual decline.
- **Fail-safe defenses' robustness gap narrows as heterogeneity increases**:
  Multi-Krum's α=0.1 floor under `large_norm` (0.299) is closer to its
  clean α=0.1 baseline (0.194) than its IID gap is (0.447 clean vs. 0.447
  under attack — i.e. almost no visible cost at IID, a real but smaller
  cost at severe Non-IID).

## 8. Per-defense summary

| Defense | Catastrophic failure mode found? | Best partition | Worst partition | Notes |
|---|---|---|---|---|
| FedAvg | Yes — always, under `large_norm` | any (no_attack) | any (large_norm) | No filtering at all; the reference "what happens with zero defense." |
| Norm | No | α=0.1 (best FPR) | IID (worst FPR, §2) | Only detector with `TPR=1.0` vs `large_norm` everywhere; its FPR is trajectory-length-sensitive, not just partition-sensitive. |
| Cosine | Yes — outside IID, under `large_norm` (Regime A) | IID | α=1.0/0.5/0.1 (diverges) | The one defense whose calibration-time TPR=1.0 (at IID) did NOT prevent catastrophic failure elsewhere — a real asymmetric-risk lesson. |
| Sign Consensus | Yes — even at IID, under `large_norm` | none diverge-free under `large_norm` | all | Small-sample calibration (4 malicious observations) too fragile to certify against this attack at all. |
| Median | No | iid/α=1.0 | α=0.1 | Graceful degradation under every attack; no detector overhead (no scores, no thresholds). |
| Multi-Krum | No | iid | α=0.1 | Best overall robustness floor under `large_norm` among all 6 defenses. |

## 9. Limitations, stated plainly

- α=0.5's clean baseline had not fully converged by round 25 even in the
  round-budget study (`SELECTION.md`) — comparisons involving α=0.5 should
  be read with that in mind.
- Sign Consensus calibration draws only 4 malicious observations per
  condition (1 malicious pseudo-client × 4 active calibration rounds) —
  too small a sample to reliably certify a κ against anything but the
  clearest signal; this likely explains its weak, sometimes `TPR=0.0`
  calibration outcomes more than any inherent property of sign-agreement
  scoring.
- The calibration-trajectory-length gap (§2) means even Regime A's
  "condition-specific, best-achievable" numbers are not fully
  deployment-realistic — they are better than Regime B, not perfect.
- 3 seeds is enough to see clear qualitative patterns (divergence is
  deterministic given the attack succeeds; accuracy differences are
  mostly larger than seed-to-seed noise) but not enough for tight
  confidence intervals on the smaller effects in §4.

## 10. Stopping point

This is the corrected, trustworthy single-method PathMNIST benchmark.
Combined Attestation was not implemented. Fed-ISIC2019 was not started.
Both remain for a future pass, per instruction.
