# Final Report — Gradient Defenses for Byzantine-Robust Federated Learning,
# from ZKFL-PQ Reproduction to Combined Gradient Attestation (C4-DA v1)

**Status: experimental phase complete**, per cross-check with an independent
reviewer (ChatGPT, consulted throughout this research arc as a second
opinion). This document synthesizes the full arc: upstream reproduction →
single-defense characterization → complementarity discovery → Combined
Gradient Attestation design → PathMNIST end-to-end validation → Fed-ISIC2019
cross-dataset validation → final conclusions.

---

## 1. Executive summary

Starting from the `edlansiaux/pq-zkfl-medical` ("ZKFL-PQ") paper/codebase —
a post-quantum, zero-knowledge federated-learning system using norm-bound
ZK proofs plus threshold homomorphic encryption — this project (a) audited
the upstream cryptographic implementation independently, (b) built a
shared gradient-defense benchmark (attacks × defenses × calibration,
dataset-agnostic), (c) characterized three gradient-based detection
signals (Norm, Cosine, Sign Consensus) on a controlled image benchmark
(PathMNIST), (d) found they have genuinely complementary — not redundant
— failure modes, (e) designed and empirically validated a layered Combined
Gradient Attestation mechanism (**C4-DA v1**: a causal, drift-aware
magnitude-safety gate plus a hierarchical Cosine+Sign directional stage),
and (f) tested whether that PathMNIST-locked design generalizes, unmodified,
to a real multi-hospital natural federation (Fed-ISIC2019, FLamby).

**The central, evidence-backed finding**: *no single gradient signal is
reliable across every data distribution.* Norm's magnitude evidence was
stable across every experiment in this project. Directional evidence was
not — Cosine was the directional specialist on PathMNIST (TPR 0.889) but
was **completely blind** to the same class of attack on Fed-ISIC2019's
real natural federation (TPR 0.000), while Sign Consensus — whose
contribution could not be shown to be necessary anywhere in the PathMNIST
study — became the operative directional defense there (TPR 1.000).
Combining multiple signal types let `C4-DA v1` maintain coverage against
the tested harmful attack classes across both datasets despite this role
swap underneath it, at
the cost of a real, measured increase in honest-client false rejections
on the natural federation. Both the success (coverage held) and the cost
(FPR rose) are reported without rebalancing the narrative toward either
extreme.

---

## 2. Research arc

```
Upstream ZKFL-PQ reproduction + independent security audit
        ↓
Shared attack/defense/calibration benchmark framework (dataset-agnostic)
        ↓
Synthetic data (debug/interface validation only)
        ↓
PathMNIST — single-defense characterization
  Norm / Cosine / Sign Consensus / FedAvg / Median / Multi-Krum
  6 attacks × 4 partitions (IID, α=1.0/0.5/0.1) × 3 seeds, 594 combos
        ↓
Deep failure-mode + complementarity analysis (analysis_v2/)
  Norm → magnitude specialist, temporal-drift FPR problem
  Cosine → directional specialist, Non-IID fragility
  Sign → sparse/local-robust, redundant with Norm in this attack suite
        ↓
Combined Gradient Attestation — offline design + replay (combined_design_v1/)
  7 candidate rules screened on 9,000-row same-update trace
  Eliminated catastrophically unsafe candidates (majority vote, naive
  weighted score); identified drift-aware Norm gate's predicted benefit
        ↓
C4-DA v1 — first real end-to-end FL validation (combined_e2e_minimal/)
  144 real training runs; confirmed drift-aware gate eliminates a
  25%-of-runs self-rejection collapse failure mode static designs showed
        ↓
8-seed confirmation + analytical self-influence stress test
  (c4_drift_aware_confirmation/) — 192 more runs, all lock criteria met
        ↓
C4-DA v1 LOCKED (results/pathmnist/LOCKED_DESIGN_C4-DA-v1.md)
        ↓
Final PathMNIST evaluation + end-to-end ablation (c4_da_ablation/)
  216 runs: Norm and Cosine proven necessary; Sign's necessity NOT shown
  in this attack suite (explicitly flagged as open question, not hidden)
        ↓
Fed-ISIC2019 — cross-dataset natural-federation validation (fed_isic2019/)
  34 runs, locked C4-DA v1 UNCHANGED, own dataset-appropriate protocol
  large_norm safety generalizes perfectly; Cosine/Sign roles INVERT;
  honest-FPR advantage does NOT clearly generalize
        ↓
THIS DOCUMENT
```

Every stage locked its protocol in writing *before* implementation
(`docs/BENCHMARK_PROTOCOL.md`, with dated, additive addenda — never a
silent edit of an already-locked rule), and every GPU-expensive stage was
preceded by a pilot and an explicit integrity/gate check before scaling up.

---

## 3. Upstream findings (kept separate from the gradient-attestation
contribution, per `docs/UPSTREAM_SECURITY_FINDINGS.md`)

An independent audit of the original ZKFL-PQ crypto stack found, among
other things:

- The **main experiment scripts** (`run_experiment.py`,
  `run_baselines.py`, `run_backdoor.py`, `run_scale.py`,
  `run_medmnist.py`) use a classical Fiat-Shamir ZK proof with **~8-bit
  (1/256) soundness**, not the 128-bit class the README's threat table
  implies for "ZKP" generally — only 5 of the ~10 experiment scripts use
  the actual 128-bit-class Unruh-transform proof.
- **The threshold-HE "privacy" layer and the median/Krum robust
  aggregator are mutually exclusive in practice** — `run_target_protocol.py`
  individually threshold-decrypts every accepted client's own ciphertext
  before handing plaintext deltas to `robust_aggregate`, because
  coordinate-median/Krum need cross-client plaintext comparison. Under the
  default `ZKFL_ROBUST_AGG=median`, the server sees every accepted
  client's full gradient in the clear every round — the same visibility
  as having no HE at all for those clients, despite `SECURITY.md`
  presenting threshold decryption as the mitigation for exactly the
  "curious aggregator" threat this configuration does not actually close.
  **This is the most significant, previously-undisclosed finding in that
  audit.**
- `run_medmnist_cnn.py` never resets the model to the round's global
  weights before each client's local training (cascading/sequential SGD
  rather than parallel federated updates, plus a double-application of
  the aggregate) — a likely explanation for the 74.2%-vs-claimed-93.3%
  accuracy gap reproduced at STEP 3.
- A norm bound structurally cannot distinguish an in-bound malicious
  direction from an honest one (confirmed from the repo's own backdoor
  results, ASR ≈98% under ZKP-ℓ2-only) — the exact motivating gap this
  entire project's gradient-defense work addresses.

**These are properties/bugs of the upstream crypto system, not of the
gradient-attestation contribution this project built.** Nothing upstream
was silently patched; `C4-DA v1` and the whole benchmark operate as a
plain (non-cryptographic) gradient-defense layer, evaluated on its own
terms, not as a fix for the above.

---

## 4. PathMNIST: single-defense characterization and complementarity

Full detail: `results/pathmnist/analysis_v2/REPORT.md`. Headline,
evidence-based findings (594 runs, 6 attacks × 4 partitions × 3 seeds):

| Signal | Strength | Failure mode |
|---|---|---|
| **Norm** | TPR=1.0 on `large_norm`/`sparse_coordinate_attack` at every Non-IID level; never diverges the model on these attacks | Blind to `low_norm` (TPR=0 always); static-threshold FPR drifts over the training trajectory, worst at IID (63%) — a temporal-drift problem, **not** a Non-IID problem (Non-IID actually improves Norm's FPR, opposite of the other two signals) |
| **Cosine** | Best directional-attack detector on PathMNIST; TPR 1.00→0.76 as Non-IID increases | Structurally blind (TPR=0) to `large_norm`/`sparse_coordinate_attack` outside IID; a single missed `large_norm` update is catastrophic |
| **Sign Consensus** | Only signal holding TPR=1.0 on `sparse_coordinate_attack` at every Non-IID level; cleanest honest-side signal (lowest single-signal FPR) | Worst single result in the whole PathMNIST study: TPR=0 on `large_norm` at *every* partition including IID — the malicious score is inverted (higher than honest), not merely overlapping |

**Catastrophic-miss evidence** (`STEP 5`): P(diverge | ≥1 `large_norm`
accepted) = **1.000** (n=30, identical across 3 seeds, always at the first
attacked round) — a single false negative is catastrophic specifically for
unbounded-magnitude attacks, not for magnitude-bounded ones (which require
*sustained* misses, P≈0.03 for a single miss).

**Complementarity evidence** (`STEP 7`, validated across 3 seeds, 9,000
trace rows): each signal has a non-overlapping "home turf" — Norm for
`large_norm`, Cosine (necessarily) for directional attacks, Norm+Sign
jointly for `sparse_coordinate_attack` — with Norm responsible for the
large majority (16.1 of ~20.4 percentage points, pooled 3-seed) of all
honest false rejections. This complementary-but-unequal-cost pattern is
the direct empirical motivation for a *layered*, not flat-voting, Combined
design.

---

## 5. C4-DA v1: design, lock, and PathMNIST end-to-end validation

**Definition** (`results/pathmnist/LOCKED_DESIGN_C4-DA-v1.md`,
`benchmark/defenses/combined.py::C4DriftAware`):

```
Magnitude Safety:  causal, label-free, per-round median+MAD Norm gate
                   (current-round statistic; self-influence bounded by
                   the breakdown point of median/MAD for 1 attacker of 5)
Directional:       hierarchical Cosine OR (Sign-fail AND Sign-peer-outlier)
```

**Offline design** (7 candidates screened on a 9,000-row same-update
trace) eliminated majority-voting and naive-weighted-score candidates for
accepting `large_norm` updates 135/180 and 180/180 times respectively
despite an attractive 0% honest FPR — exactly the "looks good on one
metric, catastrophic on another" trap the Pareto-style multi-axis
screening was designed to catch.

**End-to-end validation** (144 → 192 → 216 real training runs across
three successive stages):
- **Large-norm catastrophic safety: 0 misses in 528 total `large_norm`
  runs/client-rounds across every stage** (0/12 → 0/48 → 0/480 across
  increasing seed counts, then reconfirmed in the 216-run ablation).
- **Drift-aware gating eliminated a real failure mode static designs
  showed**: both static-threshold candidates (`C3-static`, `C4-static`)
  exhibited a **total self-rejection collapse** (model frozen permanently
  at its round-0 state) in up to 25% of runs, triggered whenever the real
  training trajectory's honest gradient statistics drifted past a
  locked-too-early threshold — a failure mode invisible to offline replay
  (which used a trajectory that never collapses by construction) and only
  discovered because the pilot-before-full-run discipline caught it before
  144 further GPU-hours were spent on a flawed design. `c4_drift_aware`
  showed this collapse in **0 of 48** of its own runs.
- **End-to-end ablation** (216 runs, 3 signal-removed variants): Norm and
  Cosine are each independently necessary (removing Norm: `large_norm` TPR
  1.000→0.250, 9/12 Non-IID runs diverge, 0/3 IID runs diverge — exactly
  reproducing Cosine's own IID-only large_norm blind spot once Norm is
  gone; removing Cosine: directional TPR 0.900→0.544, nothing else
  affected). **Sign's necessity was not established in this attack
  suite** — removing it changed no measured outcome — reported honestly
  as an open question, not papered over, and explicitly *not* "fixed" by
  inventing a new attack to rescue it post-hoc.
- Accuracy is comparable to Multi-Krum (0.381 vs. 0.386, the best
  single existing baseline) — **not** a demonstrated accuracy win, framed
  throughout as "comparable utility, additional interpretability" (explicit
  per-client, per-signal accept/reject reasoning Multi-Krum does not
  provide), never as "beats Multi-Krum."

---

## 6. Fed-ISIC2019: cross-dataset generalization validation

Full detail: `results/fed_isic2019/FEDISIC_REPORT.md`. 34 real runs on 6
natural FLamby medical centers (23,247 real dermoscopy images, no
synthetic re-partitioning), `C4-DA v1` carried over with **zero code
changes**; only the dataset-appropriate protocol (K=10 steps, N_ROUNDS=12,
its own calibration) was freshly derived, locked *before* any attack
result was seen.

**What generalized cleanly:**
- `large_norm` catastrophic safety: FedAvg diverges in 2/2 seeds (accuracy
  collapses to 0.183, closely matching the PathMNIST-scale collapse);
  `C4-DA v1`, Norm, and Multi-Krum stay completely safe in 2/2 seeds each.
- Clean (no-attack) utility: `C4-DA v1` achieved similar clean utility in
  the two-seed evaluation (0.470), compared with FedAvg (0.472), Norm
  (0.479), and Multi-Krum (0.475).

**What did not generalize, reported with the same rigor as the positive
result:**
- **Cosine and Sign swap roles.** Standalone Cosine is *completely blind*
  (TPR=0.000, both seeds) to `directional_poisoning` on Fed-ISIC2019 —
  the opposite of its PathMNIST specialist role (TPR 0.889) — while Sign
  Consensus, whose contribution PathMNIST's own ablation could not
  establish as necessary, is the signal directly traced (via per-round
  `cos_fail`/`sign_fail`/`sign_peer_outlier` logging) to be catching this
  attack on the real federation (TPR 1.000). The precise, scoped claim
  this supports: *Sign Consensus demonstrated unique marginal utility on
  Fed-ISIC2019 by detecting directional poisoning that the locked Cosine
  detector completely missed* — not a general claim that Sign is always
  necessary.
- **`C4-DA v1`'s honest-FPR advantage over standalone Norm — the headline
  PathMNIST result — does not clearly carry over.** On 3 of 4 tested
  conditions, `C4-DA v1`'s honest FPR (0.115–0.238) is 2.5–5x *higher*
  than standalone Norm's (0.042–0.046, remarkably stable across every
  condition — the drift-aware magnitude gate itself continues to work
  well). Sign's peer-rank mechanism is the prime suspect for the
  additional false rejections (the same mechanism that successfully
  catches `directional_poisoning` above) — but this is **inferred, not
  proven**: no dedicated Fed-ISIC2019 ablation was run to confirm Sign is
  the sole cause, and that causal claim is not made more strongly than
  the evidence supports.

**Seed scope**: 2 seeds (not PathMNIST's eventual 8), a deliberate,
cost-justified scope decision (~18x higher per-round compute cost than
PathMNIST). Every TPR value was bit-identical across both seeds; FPR
showed one real but non-qualitative variance (`C4-DA v1`'s `no_attack`
FPR: 0.236 vs. 0.125 — same direction, no collapse, no flip). A third seed
was not run, per the pre-agreed stop rule (2 seeds consistent → stop;
only extend on collapse or qualitative disagreement).

**Cross-dataset summary.** The table below condenses this section's
findings into the single comparison that carries this project's core
cross-dataset contribution:

| Property | PathMNIST | Fed-ISIC2019 |
|---|---|---|
| Norm → large-norm | Strong | Strong |
| Cosine → directional | Strong | Failed |
| Sign unique contribution | Not demonstrated | Demonstrated |
| C4-DA catastrophic safety | Preserved | Preserved |
| C4-DA honest FPR | Low after adaptation | Higher / variable |
| Clean utility | Preserved | Preserved |

---

## 7. Core contribution and research insight

The strongest, most defensible conclusion this project supports is **not**
"Combined Gradient Attestation beats every alternative" — it is:

> **No single gradient-based signal is reliable across every data
> distribution.** Magnitude evidence (Norm) was the most stable signal
> across every experiment in this project, on both datasets. Directional
> evidence was not: which signal (Cosine or Sign) actually provides
> reliable directional-attack coverage depends on the data distribution
> in ways neither this project nor, as far as these experiments show, the
> broader literature's single-signal framings fully anticipate. A layered
> design that combines magnitude and (multiple) directional evidence types
> maintained catastrophic-magnitude and directional/sparse attack coverage
> across the evaluated settings on two structurally different datasets
> *despite* this underlying role change — but that resilience
> was not free: it came with measurably higher honest-client false
> rejections on the natural federation than any single signal alone would
> have produced there.

This is a stronger and more honest claim than a simple performance
leaderboard, precisely because it is falsifiable in a way "Combined is
best" is not — the PathMNIST ablation's open question about Sign's
necessity was answered concretely, not rhetorically, by a second dataset,
and the answer came with a real cost attached.

---

## 8. Comprehensive limitations

Consolidated from every stage's own limitations section, not re-litigated
here — restated once, completely, in one place:

1. **Sign Consensus's necessity is dataset-dependent, confirmed, not
   resolved.** It showed no measurable contribution in PathMNIST's
   6-attack suite but became the operative directional defense on
   Fed-ISIC2019. Whether this is attack-suite-specific or a genuine
   distributional effect is not separable with the experiments run here.
2. **Fed-ISIC2019 used 2 seeds**, not PathMNIST's eventual 8 — a
   deliberate cost tradeoff (~18x per-round compute). TPR findings are
   strongly evidenced (bit-identical across both seeds); FPR magnitudes
   should be read as a range, not a point estimate.
3. **Self-influence on the drift-aware Norm gate is real and bounded, not
   eliminated.** The attacker's own update measurably shifts its judging
   threshold (median relative shift 16.9% in the PathMNIST 144-run
   benchmark); an analytical stress test found the exploitable region
   (attacker magnitude that both inflates the threshold *and* passes it)
   caps at ≤5× the honest median — real `large_norm` attacks operate
   10²–10⁴× beyond that region in every test here, but a hypothetical
   attack specifically tuned to stay within that narrow band was not
   constructed or tested.
4. **`low_norm` (free-rider) harmlessness is an artifact of the tested
   implementation, not a proven defense property.** TPR=0 for every
   signal against every `low_norm` variant tested throughout this project
   means only that *this specific* low-norm attack (`delta * eta`,
   `eta=0.01`) caused no measurable harm — not that `C4-DA v1` or any
   component defends against a stronger, more adaptive free-rider attack
   designed to mimic honest-client statistics more closely. Untested,
   explicit future work.
5. **Cosine's `g_ref` requires a trusted server-held reference split**,
   carved disjoint from client data by construction and asserted
   programmatically — but the trust assumption itself (the server
   constructing an uncorrupted reference set) is not evaluated as an
   attack surface anywhere in this project.
6. **`C4-DA v1`'s honest-FPR increase on the natural federation is a
   measured cost, not a resolved one.** No Fed-ISIC2019-specific ablation
   was run to confirm Sign's peer-rank mechanism is the sole or even
   primary cause. The Sign peer-rank mechanism is a plausible contributor,
   supported by mechanism overlap with the directional-detection behavior,
   but this was not isolated experimentally.
7. **Upstream ZKFL-PQ's cryptographic weaknesses (§3) are independent of
   this project's gradient-attestation contribution** and must not be
   conflated — `C4-DA v1` is a plain (non-cryptographic) gradient defense,
   evaluated at the same privacy tier as the upstream's own median/Krum
   composition (§3's finding that threshold-HE and cross-client robust
   aggregation are already mutually exclusive in the upstream code),
   not a cryptographic fix for anything in §3.
8. **No claim that `C4-DA v1` outperforms Multi-Krum is supported by this
   data.** Their final accuracies are statistically close on both datasets
   (PathMNIST: 0.381 vs. 0.386; the two were not compared head-to-head on
   Fed-ISIC2019's attack conditions beyond the shared matrix already
   reported). The differentiating value, where it exists, is
   interpretability (explicit per-client, per-signal attestation) versus
   Multi-Krum's opaque robust aggregation — a different kind of
   contribution, not a better one by the metrics tested.
9. **Protocol scale (K-steps, N_ROUNDS, batch size) was independently
   derived per dataset**, not transferred — correct per the locked
   "procedure, not numbers" principle, but means PathMNIST's and
   Fed-ISIC2019's absolute round-counts/training budgets are not directly
   comparable as training-compute figures.
10. **A single, fixed, pre-declared malicious client/center was used in
    every attacked run** (consistent `MALICIOUS_RATIO≈0.2` convention) —
    the effect of *which* client plays the attacker role, or of multiple
    simultaneous attackers, was not varied or tested.

---

## 9. Final research questions, answered

1. **Does a norm bound alone defend against directional/backdoor attacks?**
   No — confirmed both from the upstream repo's own backdoor results (ASR
   ≈98% under ZKP-ℓ2-only) and from this project's own Cosine/Sign
   characterization; this structural gap is the motivating problem the
   whole gradient-defense benchmark addresses.
2. **Do Norm, Cosine, and Sign Consensus have genuinely complementary, not
   redundant, failure modes?** Yes, on PathMNIST (STEP 7, 3-seed
   validated) and **confirmed again, with a twist, on Fed-ISIC2019** —
   complementary, but *which* signal covers the directional gap is itself
   dataset-dependent.
3. **Can a layered Combined design (magnitude gate + directional stage)
   preserve each signal's strength without inheriting Norm's static-FPR
   weakness?** Yes for the magnitude side (the drift-aware gate's FPR
   reduction was large and clean on PathMNIST, and the gate itself stayed
   well-behaved on Fed-ISIC2019 too) — but the Combined system's *overall*
   FPR did not inherit that improvement cleanly once Sign's directional
   contribution became load-bearing on the second dataset.
4. **Does a design locked entirely on one dataset generalize to a
   structurally different, real natural federation without retuning?**
   Partially, and specifically: the catastrophic-safety property
   generalizes perfectly; the detection-mechanism attribution and the
   FPR profile do not transfer as cleanly — exactly the kind of nuanced,
   evidenced answer a real cross-dataset test is supposed to produce.
5. **Is Combined Gradient Attestation ready to claim superiority over
   existing robust aggregators (Median, Multi-Krum)?** No — utility is
   comparable, not better; the contribution is interpretability and
   multi-signal attack coverage, stated as such, not inflated.

---

## 10. What was deliberately not done (and why that's documented, not a gap)

- No Combined Attestation design or implementation happened until the
  single-defense complementarity evidence existed to justify it (PathMNIST
  STEP 7) — avoiding "combine signals because it sounds good" reasoning.
- No threshold was ever retuned after seeing an attack-test result, at any
  stage, on either dataset — every calibration was locked from
  validation-only data before the corresponding benchmark ran.
- No new attack was invented to "rescue" Sign's PathMNIST ablation result
  — Fed-ISIC2019's organic finding that Sign matters there was left to
  stand on its own, not engineered.
- The 594-combination PathMNIST matrix, the 144/192/216-run Combined
  validation stages, and the Fed-ISIC2019 34-run matrix were each sized
  to their own pilot/round-budget evidence, never assumed from a prior
  dataset's numbers.
- Upstream cryptographic code was never silently patched while building
  or benchmarking the gradient-defense layer.

---

## 11. Conclusion

This project traced a complete, disciplined arc from reproducing and
independently auditing an existing post-quantum federated-learning system,
through characterizing why its norm-only gradient check is structurally
insufficient, to designing, locking, and validating a layered Combined
Gradient Attestation mechanism on two structurally different datasets —
one controlled (PathMNIST), one a real multi-hospital natural federation
(Fed-ISIC2019). The resulting evidence supports a specific, bounded claim:
`C4-DA v1` prevented the catastrophic failure mode of the tested
unbounded-magnitude attack in all evaluated PathMNIST and Fed-ISIC2019
runs, maintains
clean-data utility competitive with the strongest existing robust
aggregator, and preserves directional-attack coverage across a real
distribution shift that caused its best single alternative signal to fail
completely — at a real, measured, and reported cost in honest-client false
rejections that is not yet fully understood or resolved. Every positive
and negative finding here is backed by logged, inspectable experimental
data (`results/pathmnist/`, `results/fed_isic2019/`,
`results/pathmnist/LOCKED_DESIGN_C4-DA-v1.md`), not asserted.
