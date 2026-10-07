# Fed-ISIC2019 Cross-Dataset Generalization Validation

**Research question** (per ChatGPT's framing when this stage was scoped):
*does `C4-DA v1` — designed and completely locked on PathMNIST, with no
retuning permitted — hold up when moved to 6 real, natural medical-imaging
centers?* This is **not** a repeat of PathMNIST's ablation/mechanism study
(already answered there); it is a generalization test.

**34 runs**: 2 seeds (42, 43) × {`C4-DA v1`, FedAvg, Norm, Multi-Krum} × 4
conditions (no_attack, large_norm, directional_poisoning,
sparse_coordinate_attack) + Cosine × directional_poisoning only (1
condition, as the direct single-signal baseline for C4-DA's directional
component). `benchmark/defenses/combined.py::C4DriftAware` — **zero lines
changed** from the PathMNIST-locked version. Protocol: 6 FLamby natural
centers (no re-partitioning), EfficientNet-b0 (FLamby's own pretrained
baseline, not a new architecture), K=10 steps/round, N_ROUNDS=12,
ATTACK_FROM_ROUND=5 — all locked via their own dataset-appropriate
sizing study (`docs/BENCHMARK_PROTOCOL.md` Addendum (j)), **not** copied
from PathMNIST's K=50/N=25. τ/ρ/κ calibrated once
(`scripts/fedisic_calibration.py`, canonical `large_norm` calibration
attack, same procedure PathMNIST used) **before** this benchmark ran —
never retuned against these results.

## Headline result: catastrophic safety generalizes perfectly

| attack | FedAvg accuracy | FedAvg diverged | C4-DA / Norm / Multi-Krum accuracy | diverged |
|---|---|---|---|---|
| large_norm | **0.183** (both seeds) | **2/2** | 0.475 / 0.474 / 0.476 | 0/2 each |
| sparse_coordinate_attack | **0.221** (mean) | 0/2 (loss stayed finite, but accuracy destroyed) | 0.475 / 0.474 / 0.472 | 0/2 each |
| directional_poisoning | 0.474 | 0/2 | 0.478 / 0.474 / 0.474 | 0/2 each |
| no_attack | 0.472 | — | 0.470 / 0.479 / 0.475 | — |

**FedAvg diverges in both seeds under `large_norm`** (final accuracy
0.183, matching PathMNIST's own FedAvg-diverges-under-large_norm finding
almost to three decimal places) — **`C4-DA v1`, Norm, and Multi-Krum all
stay completely safe, in both seeds**, on real medical imaging data this
design never saw during its PathMNIST-only design/lock process. This is
the single most important confirmation this stage could produce, and it
replicates cleanly. `sparse_coordinate_attack` also substantially degrades
unprotected FedAvg here (0.221, a new-to-this-dataset finding — PathMNIST's
FedAvg was not specifically tested this way in the comparable baseline
table) while all three protected methods remain unaffected.

**Clean (no_attack) utility: `C4-DA v1` pays no meaningful accuracy cost**
— 0.470 vs. FedAvg 0.472, Norm 0.479, Multi-Krum 0.475, all within normal
seed-to-seed noise. The PathMNIST finding that Combined designs don't
sacrifice clean utility generalizes.

## Important nuance #1: Sign Consensus, not Cosine, is the operative
directional signal on this dataset — a genuine role inversion from
PathMNIST

| defense | directional_poisoning TPR | directional_poisoning honest FPR |
|---|---|---|
| **Cosine (standalone)** | **0.000** | 0.000 |
| Norm (standalone) | 0.000 | 0.046 |
| **C4-DA v1** | **1.000** | 0.115–0.123 |

**OBSERVED, directly traced** (`client_decisions.csv`, 7 attacked rounds ×
2 seeds for the malicious center): `cos_fail` is `False` in every single
attacked round (`cosine_score` stays ~0.2–0.43, nowhere near ρ=-0.393) —
**standalone Cosine is completely blind to `directional_poisoning` on
Fed-ISIC2019**, a flat reversal of PathMNIST's finding that Cosine was
*the* directional specialist (TPR 0.889–0.900 there). `C4-DA v1` still
achieves perfect TPR (1.000, both seeds) on this same attack — but tracing
*why* shows `sign_fail=True` and `sign_peer_outlier=True` in every single
attacked round: **Sign Consensus's peer-rank mechanism is what actually
rejects the attacker here, not Cosine.**

**INFERRED (reasonable interpretation, not directly measured):** this is
plausibly a curse-of-dimensionality effect — `directional_poisoning`
rescales a direction-reversed delta to match the honest update's own norm
in a ~4,017,796-parameter space (vs. PathMNIST's much smaller model), and
cosine similarity naturally concentrates toward 0 as dimensionality grows,
compressing the honest/malicious separation that a lower-dimensional model
would show more sharply. This was not tested directly (would require a
dedicated dimensionality sweep, out of scope here) and is flagged as
interpretation, not fact.

**Why this matters for the thesis's central argument:** PathMNIST's own
ablation (`results/pathmnist/c4_da_ablation/FINAL_PATHMNIST_REPORT.md`)
found Sign's contribution "not shown to be necessary in this attack
suite" and explicitly flagged as future work the question "would Sign
become necessary against attacks Norm+Cosine can't cover?" **Fed-ISIC2019
answers that question concretely: yes** — on real medical imaging data,
Sign Consensus is the signal standing between a directional attack and a
successful compromise, exactly the complementary-coverage argument the
whole Combined Attestation design was built on, now demonstrated outside
the dataset it was designed on.

## Important nuance #2: C4-DA v1's honest-FPR advantage over standalone
Norm does **not** clearly replicate here

| condition | C4-DA v1 honest FPR | Norm (standalone) honest FPR |
|---|---|---|
| no_attack | 0.181 (mean; 0.236/0.125 by seed) | **0.042** (identical both seeds) |
| sparse_coordinate_attack | 0.238 | **0.046** |
| large_norm | 0.046 | 0.046 (tied) |
| directional_poisoning | 0.115–0.123 | 0.046 |

**OBSERVED, stated plainly, not smoothed over:** on PathMNIST, the
drift-aware Norm gate's whole point was a large honest-FPR *reduction*
relative to static-threshold designs. Here, on 3 of 4 conditions, `C4-DA
v1`'s honest FPR is 2.5–5x **higher** than standalone Norm's, not lower.
Norm's own FPR is remarkably stable across every condition (0.042–0.046,
near-identical regardless of attack) — the drift-aware magnitude gate
itself is behaving well. The elevated FPR in `C4-DA v1` is therefore not
coming from the Norm component; by elimination (and consistent with
nuance #1 above), it is most plausibly the **Sign Consensus peer-rank
stage** generating extra false rejections against honest centers' natural
heterogeneity — the same mechanism that successfully catches
`directional_poisoning` also appears to cost more honest-client false
rejections on this dataset's real (not synthetic) Non-IID structure than
it did on PathMNIST's Dirichlet partitions.

**This is reported as a genuine, measured limitation, not reinterpreted
to look better.** Per the locked rule, `C4-DA v1`'s formula is **not**
retuned in response to this finding — it remains exactly as locked. This
is Fed-ISIC2019 doing its job: surfacing a real generalization gap that
PathMNIST's synthetic Dirichlet partitions did not surface.

## Seed consistency (2 seeds, per ChatGPT's locked plan)

Every TPR value is **bit-identical across both seeds** (0.0 or 1.0 exactly,
zero variance) for every defense/attack combination. FPR values are
stable within a few percentage points except `C4-DA v1`'s `no_attack`
condition (0.236 vs. 0.125) — a real quantitative swing, but not a
qualitative one: both seeds show the same direction (clearly elevated
above Norm's stable 0.042) and no seed shows anything resembling the
PathMNIST-stage collapse pattern (honest_fpr > 0.9). **Per ChatGPT's
explicit stop rule ("if 2 seeds consistent → STOP; only add a 3rd if
variance/collapse is concerning"), a 3rd seed is not run** — no TPR
instability, no collapse, and the one real FPR variance doesn't change
any directional conclusion.

## Answering the research question

**Does `C4-DA v1`, locked on PathMNIST, hold up on 6 natural medical
centers?** Yes on the property that matters most (catastrophic `large_norm`
safety, which generalizes with zero degradation) and on clean utility
(no accuracy cost). **No, not uniformly** — the specific mechanism behind
directional-attack detection shifts from Cosine-led (PathMNIST) to
Sign-led (Fed-ISIC2019), and the honest-FPR advantage that was `C4-DA v1`'s
headline PathMNIST result does not clearly carry over, with Sign's
peer-rank stage appearing to be the source of the gap. Both the positive
and negative findings are real, evidenced, and reported without
rebalancing the narrative toward "it worked."

## Limitations

- 2 seeds, not PathMNIST's later 8-seed depth — scoped down deliberately
  given the ~18x per-round cost; TPR stability across both seeds is
  strong evidence, but FPR's single-condition variance (no_attack) means
  the *exact* FPR magnitude should be read as a range, not a point
  estimate, same caution PathMNIST itself applied to single-seed numbers
  before its own multi-seed extension.
- Only 3 attacks tested (`large_norm`, `directional_poisoning`,
  `sparse_coordinate_attack`) plus `no_attack` — `full_sign_flip` and
  `low_norm` were not run here (per the locked scope decision, these were
  deprioritized as lower-value given PathMNIST's own findings that they
  track the tested attacks closely or are low-impact).
- No ablation was run on Fed-ISIC2019 (deliberately, per ChatGPT's
  guidance — PathMNIST's ablation already answers the mechanism question;
  this stage is generalization-only). The Sign-vs-Cosine role-inversion
  finding above is therefore based on direct per-round score tracing, not
  a formal Fed-ISIC2019 ablation.
- Only one designated malicious center (center 1, fixed) was used across
  all conditions — matching PathMNIST's MALICIOUS_RATIO≈0.2 convention,
  but not varied across different centers to check whether the finding
  depends on which center plays the attacker role.
