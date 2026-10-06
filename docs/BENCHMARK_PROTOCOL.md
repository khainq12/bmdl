# Benchmark Protocol — v0.1 (locked before implementation)

This document is written and frozen **before** any `benchmark/` source file is
written, per explicit instruction. It encodes the decisions below as the
contract the implementation must follow. Changes to any locked rule in this
document must be a new, dated, additive section — never a silent edit of an
already-locked rule, and never a change made to fit a result we already saw.

Scope of this version: **common benchmark interfaces + transparent
single-method baselines only** (FedAvg, Norm filtering, Cosine filtering,
Sign Consensus, Coordinate-wise Median, Multi-Krum). Weighted Combined
Attestation, and even the simpler unweighted/transparent Combined
Attestation ablation grid (task brief §9/§12), are explicitly **out of
scope for this implementation pass**. They are a separate, later step once
the six single-method baselines are implemented, calibrated, and sanity
checked. Dataset loaders for PathMNIST and Fed-ISIC2019 are also a later
step (STEP 5/6); this version wires up only the synthetic dataset end to
end, since synthetic's whole role is interface debugging (see §1).

---

## 1. Dataset roles (locked)

| Dataset | Role | Notes |
|---|---|---|
| **Synthetic** (`fl_core.model.generate_synthetic_medical_data` + `partition_non_iid`) | **Reproduction / debugging only.** Used to validate the benchmark interfaces (this pass) and for fast iteration. **Never used for headline defense-comparison claims.** | Already used for STEP 3 upstream reproduction; reused here under the new, properly-seeded runner (`benchmark/models/local_training.py`), not the upstream unseeded `local_training()`. |
| **PathMNIST** (MedMNIST) | **Main controlled benchmark.** Official train/val/test splits. IID + Dirichlet(α) partitions generated once per seed and persisted to disk. | STEP 5, not yet implemented. Model: a new torch CNN (`benchmark/models/torch_cnn.py`, GPU-capable) for all six defenses. The full ZKFL-PQ crypto stack (ZKP+HE+Unruh) is **not** run at PathMNIST scale by default — per `docs/IMPLEMENTATION_PLAN.md` §4, it is cost-prohibitive at this model size on this hardware; any crypto-stack cell for PathMNIST will either use a deliberately compact model and say so, or be marked "not run — crypto stack cost-prohibitive at this scale," never silently omitted. |
| **Fed-ISIC2019** (FLamby) | **Natural-client validation.** Native FLamby center partition is preserved exactly — no re-partitioning, no Dirichlet re-sampling of centers. | STEP 6, not yet implemented; requires `pip install flamby` plus a manual dataset-acquisition/license step that cannot be scripted. One exception to "no re-partitioning" is carved out for the cosine defense's trusted reference set — see §3. |

## 2. Model per dataset (locked)

- Synthetic: `fl_core.model.SimpleMLP` (flat-vector `get_weights`/`set_weights`/`get_gradient_vector`), reused as-is.
- PathMNIST / Fed-ISIC2019: a new torch CNN (GPU), independent of the flat-vector crypto contract, since these two datasets are evaluated only on the gradient-defense arms (FedAvg/Norm/Cosine/Sign/Median/Krum), not the crypto stack, by default (see §1).

## 3. Data splits and partition protocol (locked)

For every dataset, the **official training pool** (never the official test
split) is divided, in this fixed order, into four disjoint subsets before
any client ever sees anything:

1. **Server trusted reference subset** (`server_ref_fraction`, default 10% of the training pool) — used only to build `g_ref` for the Cosine defense. Carved out first, before clients are assigned data, so it can never overlap a client partition by construction.
2. **Calibration/validation subset** (`calib_fraction`, default 15% of the remaining pool) — used only to calibrate τ/ρ/κ (§8). Never touched after calibration completes for a given run.
3. **Client training partitions** — everything left over, split across `n_clients` clients via IID or Dirichlet(α) (`fl_core.model.partition_non_iid`, reused as-is).
4. **Official test split** — untouched by 1–3, used only for final accuracy/loss reporting.

**Hard rule, enforced in code (not just documented):** the loader for every
dataset must assert these four subsets are pairwise disjoint before
returning. `benchmark/datasets/synthetic.py` does this with an explicit
index-set-union assertion; the same assertion is required of the PathMNIST
and Fed-ISIC2019 loaders when they are written (STEP 5/6).

**Fed-ISIC2019 exception (documented, not silent):** FLamby's native center
partition has no "server-held" split by construction. For the Cosine
defense only, a fixed, pre-declared designated center (the largest center,
by sample count, chosen once and recorded in the run's `config.json` — never
re-chosen per condition) has a `server_ref_fraction` slice carved out of its
data for `g_ref`; the remaining samples from that center still participate
in FL training as that center's client data. Every other defense sees the
fully untouched natural center partition. This is a deliberate, narrow,
documented deviation from "preserve the natural partition," scoped to
exactly the one defense that structurally requires a trusted reference set.

**Partitions are generated once per `(dataset, partition, alpha, seed)` and
persisted to disk** (for PathMNIST/Fed-ISIC2019, STEP 5/6) so every defense
in a comparison sees byte-identical client data. For the synthetic debug
role this pass, partitions are regenerated deterministically from the seed
rather than cached to disk, since synthetic data generation itself is cheap
and seed-reproducible — this is an intentional simplification for the
debug-only dataset, not a relaxation of the disk-persistence rule for
PathMNIST/Fed-ISIC2019.

## 4. Attack suite (locked formulas)

All attacks operate on one client's already-computed honest local delta
`g ∈ R^d` and take `(g, rng, params) -> g'`. `rng` is always a
caller-supplied `np.random.Generator` seeded by the runner — no attack may
touch global NumPy random state (this is the fix for the upstream
reproducibility gap documented in `docs/IMPLEMENTATION_PLAN.md` §3.9).

| Name | Formula | Notes |
|---|---|---|
| `no_attack` | `g' = g` | Control condition. |
| `large_norm` | `g' ~ N(0, scale² I)` | `scale` configurable (default 50.0, matching the upstream convention). Classic oversized-update attack. |
| `low_norm` | `g' = η · g`, `η ∈ [0,1)` (default 0.01) | **Free-rider / stealth attack**: client contributes almost nothing while trivially passing any norm-based or cosine-based check (small positive-scaled honest direction). Tests whether a defense can tell "barely participating" apart from "actively malicious" — by design, most of our defenses are not supposed to catch this (it's not Byzantine-harmful to model quality, just free-riding); recorded for completeness per the task brief's required scenario list, not because every defense is expected to reject it. |
| `full_sign_flip` | `g' = -g` | Negates **every** coordinate. Distinct from `directional_poisoning` below — this is the literal per-coordinate sign flip the name promises. (Upstream's `"sign_flip"` in `experiments/run_baselines.py` is actually the scaled-opposite-direction construction; we do not reuse that name here to avoid the ambiguity flagged in `docs/IMPLEMENTATION_PLAN.md` §3.) |
| `directional_poisoning` | `g' = -g · (r / ‖g‖)`, `r` = a configurable target norm (default: the current round's calibrated τ, falling back to `‖g‖` if no τ is available) | Opposite direction, magnitude pinned to stay in-bound against a norm gate — this is upstream's original `"sign_flip"` construction, correctly renamed. |
| `sparse_coordinate_attack` | Zero out all but `k` coordinates (default `k = max(1, d/100)`), selected either by largest `|g|` (`selection="topk"`, default) or uniformly at random (`selection="random"`); set those `k` coordinates to `±spike` (default `spike = 10·max|g|`) | Generalizes the single-spike construction in `experiments/run_innovation_pack.backdoor_linf_ablation` to a configurable coordinate count; tests coordinate-level (ℓ∞ / sign) defenses specifically. |

`malicious_ratio` (fraction of clients malicious) and `attack_from_round`
(round index at which the attack activates) are run-level config, not part
of the attack function itself, mirroring the upstream convention.

## 5. Defenses in scope now (locked formulas)

All defenses take the full set of submitted client deltas `{g_i}` for a
round plus a context dict `ctx` (round RNG, calibrated thresholds, `g_ref`
if applicable) and return an aggregated `update`, plus — for the three that
are explicit detectors — a per-client `score` and boolean `accepted` mask.
Per the task brief: **Median and Multi-Krum are robust aggregators, not
detectors** — they never produce a `score`/`accepted` output, and must never
be scored with TPR/FPR/precision in the metrics layer.

| Defense | Type | Rule |
|---|---|---|
| `fedavg` | aggregator | `update = mean_i(g_i)`. No filtering. |
| `norm` | **detector** | `score_i = ‖g_i‖₂`; accept iff `score_i ≤ τ`; `update = mean` over accepted. (This is deliberately *not* ZK-proved — a plain non-cryptographic threshold check, since the task brief treats "Norm" as a signal, distinct from `crypto.zkp_norm.ZKPNormBound`.) |
| `cosine` | **detector** | `score_i = cos(g_i, g_ref)`; accept iff `score_i ≥ ρ`; `update = mean` over accepted. `g_ref` construction is locked in §3/§6. |
| `sign_consensus` | **detector** | See §7 for the full, literature-grounded definition. Accept iff `score_i ≥ κ`; `update = mean` over accepted. |
| `median` | aggregator (not a detector) | Coordinate-wise median of **all** submitted `g_i`, via the one canonical implementation `fl_core.robust_agg.coord_median` (reused, not reimplemented — see `docs/IMPLEMENTATION_PLAN.md` §3.3 on the three divergent upstream Multi-Krum copies; we are not adding a fourth divergent median). |
| `multi_krum` | aggregator (not a detector) | `fl_core.robust_agg.multi_krum` (the one canonical implementation), `f` configurable (default 1). |

**Out of scope for this pass:** `combined` (any form — unweighted or
weighted). `benchmark/defenses/combined.py` does not exist yet;
`make_defense()` raises a clear error naming this document if requested.

## 6. Trusted server reference set for Cosine (`g_ref`) — locked

- **Hard rule: `g_ref` is built only from the server trusted reference subset (§3), and must never overlap the test split or any client's training partition.** Enforced by the same disjointness assertion as §3.
- `g_ref` is **recomputed every round** from the *current* global weights: one local-SGD pass (same `local_epochs`/`lr`/`batch_size` as clients, for scale comparability) over the reference subset, using `benchmark/models/local_training.py`'s seeded trainer. It is not a static, round-0-only vector.
- Reference-set size (`server_ref_fraction`) is a **documented sensitivity axis** (task brief §7/§13): the default (10%) is a starting point to be revisited once we have calibration-set results, never tuned against test accuracy.
- We will separately measure and report `cos(honest, g_ref)` vs. `cos(malicious, g_ref)` distributions (task brief §7) once real attack/defense runs exist — not fabricated now.

## 7. Sign Consensus — literature-grounded definition (locked, written before any code)

**Basis.** Coordinate-wise majority-sign agreement is the scoring mechanism
underlying sign-based distributed SGD, most directly **signSGD**
(Bernstein, Wang, Azizzadenesheli, Anandkumar, *"signSGD: Compressed
Optimisation for Non-Convex Problems,"* ICML 2018), where the server
aggregates by taking the coordinate-wise majority vote of client sign
vectors. We adapt the same majority-vote object, but as a **per-client
anomaly score relative to that majority** (an accept/reject consensus
check) rather than as the aggregation rule itself — i.e., we use it the way
sign-based Byzantine-robustness analyses of signSGD-style schemes use it:
a client whose update disagrees with the majority sign on most coordinates
is flagged, rather than silently out-voted.

**Formula.** Given the `n` submitted deltas for a round, stacked as a
matrix `G ∈ R^{n×d}`:

```
sign_i,k      = sign(G[i,k])                      ∈ {-1, 0, +1}
majority_k    = sign( Σ_i sign_i,k )               (coordinate-wise majority vote)
K             = { k : majority_k ≠ 0 } ∩ TopK_mask  (valid, non-tied coordinates, optionally restricted — see below)
score_i       = (1 / |K|) · Σ_{k∈K} 1[ sign_i,k == majority_k ]
accept_i      ⟺ score_i ≥ κ
```

- **Tie handling:** a coordinate where the signed vote sums to exactly 0 contributes to no client's score (excluded from `K`) — it is not evidence for or against any client that round.
- **Top-K option (configurable, `sign_topk`):** when set, `K` is restricted to the `sign_topk` coordinates with the largest mean `|G[:,k]|` across the submitted set, focusing the consensus check on the coordinates carrying the most signal (near-zero coordinates have unstable, low-information signs). When `sign_topk` is `None` (default), all `d` coordinates are used.
- **Consensus source:** the majority vote is computed **only from the currently submitted clients' deltas in that round** — no trusted server data, no cross-round memory. This satisfies the task brief's "consensus derived only from allowed training/client information."

**Documented limitation (stated now, not discovered later):** because the
majority vote is computed from the submitted set *before* any filtering,
this score inherits the same structural weakness signSGD-style majority
votes have: if the malicious fraction of clients is large enough to flip
the per-coordinate majority (informally, ≥50% colluding on a coordinate's
sign), the "majority" itself is poisoned and this score will misclassify
honest clients as the outliers. This is a known, inherent property of
*any* majority-based consensus signal, not an implementation defect — it
belongs in the sensitivity analysis (task brief §13, "malicious-client
ratio") rather than being silently assumed away.

## 8. Threshold calibration protocol for τ, ρ, κ — locked

**Hard rule: thresholds are chosen only from the calibration/validation
subset (§3), under attacks and seeds disjoint from the test-time run, and
then frozen — never re-tuned per test condition, never informed by a look
at test-set accuracy or test-set attack realizations.**

1. **Calibration loop.** For a `(dataset, partition, alpha)` configuration, run `calibration.calib_rounds` (default 5) rounds of FL on the calibration subset, split into its own pseudo-clients (same `n_clients`, same partition scheme, but a seed (`calibration.calib_seed`) and malicious-client assignment independent of the test-time run). The same attack named in the run's `attack.name` is injected starting from calibration round 1 (round 0 is always clean, to guarantee at least one clean round of honest-only scores).
2. **Score collection.** Every client-round produces one scalar score (raw `‖g_i‖` for `norm`, `cos(g_i, g_ref)` for `cosine`, the Sign Consensus score of §7 for `sign_consensus`), labeled honest or malicious from the known calibration-time injection (never from the defense's own accept/reject decision — that would be circular).
3. **Decision rule (locked, identical for all three thresholds, direction-adjusted):** among all candidate thresholds (every distinct observed score value), choose the one **maximizing TPR on malicious scores subject to pooled validation FPR ≤ `calibration.fpr_target`** (default 0.05). "Reject" direction is `score > τ` for Norm (lower = more honest) and `score < ρ` / `score < κ` for Cosine/Sign (higher = more honest). If no candidate threshold achieves the FPR target, fall back to the threshold minimizing FPR and set `used_fallback: true` in the recorded output — this must never be silently swapped for a different rule.
4. **Freeze.** The resulting numeric τ/ρ/κ is reused for **every** test-time attack type and malicious ratio evaluated under that same `(dataset, partition, alpha)` configuration. A different `alpha` or dataset gets its own independent calibration.
5. **Always recorded** (task brief §6/§7/§8): the selected threshold, `calibration.fpr_target`, the resulting validation FPR/TPR, `used_fallback`, and the calibration config (rounds, seed, `n_honest`/`n_malicious` samples pooled).

## 9. Output file schema (locked)

Every run of `benchmark/runner.run()` via `scripts/run_one.py` writes, into
one `out_dir`:

- `config.json` — the full `RunConfig`, serialized verbatim.
- `metrics.json` — `{"final_accuracy": ..., "calibration": {...} | null}`.
- `per_round.csv` — one row per round: `round, accuracy, loss, runtime_s, n_malicious, tp, fp, tn, fn` (confusion counts are `null`/absent fields when the defense is not a detector — never fabricated zeros).
- `client_scores.csv` — one row per `(round, client)`, **only for detector defenses**: `round, client_id, score, is_malicious, accepted`. Empty file (header-only or absent) for `fedavg`/`median`/`multi_krum`.
- `timing.csv` — one row per round: `round, runtime_s`.

This is the same convention named in the task brief §14 and
`docs/IMPLEMENTATION_PLAN.md` §5; this document is where the exact column
names are frozen so `benchmark/metrics.py` and any later aggregation script
agree on them without renegotiation.

## 10. Standing decisions carried over from earlier steps

- **The `run_medmnist_cnn.py` per-client weight-reset bug (`docs/UPSTREAM_SECURITY_FINDINGS.md` §5.8) is not fixed and will not be fixed.** It remains a documented upstream reproduction issue. The new PathMNIST path (STEP 5) is entirely new code (a new torch model + a new runner path), not a reuse of `run_medmnist_cnn.py`, so this bug cannot leak into new-benchmark numbers by construction — but we are not going back to "correct" the upstream script or its already-reported numbers either.
- No parameter in this document was chosen by looking at a test-set result. Where a default (10% reference fraction, 15% calibration fraction, 5% FPR target, `scale=50` for `large_norm`, etc.) is a round starting number rather than something derived from data, it is marked as such and is itself in scope for the sensitivity analysis (task brief §13), not presented as a tuned optimum.

## 11. Versioning

This is **v0.1**. Any future change — adding Combined Attestation,
changing a calibration rule, changing an attack formula — is a new, dated
section appended below this line, not an edit to the text above.

---

### Addendum 2026-10-05 — calibration trajectory must advance on honest-only deltas

Found during initial end-to-end validation of the `norm`/`large_norm`
config (benchmark/runner.py `_calibrate`): the calibration loop originally
advanced the model each round via a plain, unfiltered mean of *all*
submitted deltas (honest + malicious), matching naive FedAvg under attack.
Under `large_norm`, this caused the calibration model itself to diverge
after round 1 — the same collapse pattern seen in the STEP 3 upstream
reproduction (`results/upstream_reproduction/run_experiment.stdout.log`,
Experiment 1, rounds 4–10). A subsequent **honest** client's local gradient
on the now-diverged model then exploded numerically, contaminating the
honest-score sample with an instability artifact (observed: a calibrated τ
of ~2.9×10¹⁶ with validation TPR=0.0 — the fallback threshold was pinned
above the malicious scores by one corrupted "honest" outlier, not by any
real separation failure between honest and malicious score distributions).

**Fix (locked as of this addendum):** the calibration simulation advances
its own model trajectory using the **mean of honest-labeled deltas only**
each round. This uses no test-time information and no privileged knowledge
beyond what the calibration simulation already knows about its own
injected labels (we are the ones injecting the attack in calibration, so
the label is not "looking at the answer" — it is simulation ground truth,
the same way `experiments/run_backdoor.py` knows which upstream clients it
poisoned). The point of calibration is to characterize the honest-vs
-malicious score distributions **under a model trajectory representative
of normal training**, not under a trajectory that has already been
destroyed by the very attack we have not yet decided how to defend
against — an unfiltered calibration trajectory conflates "this score
distribution is hard to separate" with "the model already diverged," which
are different failure modes and must not be reported as the same number.

---

### Addendum 2026-10-05 (b) — calibration ground truth is an offline-only privilege; synthetic numbers so far are framework validation, not findings

Two clarifications requested explicitly and locked here, not discovered —
stated up front before PathMNIST work begins:

1. **The honest/malicious labels used throughout §8's calibration
   procedure (and in the `_calibrate` trajectory fix above) are available
   only because calibration is an offline simulation where *we* choose
   which pseudo-clients to poison.** At deployment / test time, a defense
   never has access to a ground-truth malicious label — it only ever sees
   the submitted score (`‖g_i‖`, `cos(g_i, g_ref)`, or the sign-consensus
   score) and must decide accept/reject from that alone, exactly as
   `benchmark/defenses/*.py` already does (no defense's `aggregate()` method
   reads a label). The calibration step is the *only* place in this
   framework ground-truth labels are consulted, and it is validation-split
   only (§8, §3) — this is standard train/val/test discipline, not a
   privileged shortcut carried into test-time evaluation. Any future code
   that lets a `Defense.aggregate()` implementation see `mal_flags` directly
   would violate this and must be treated as a bug.
2. **Every FPR/TPR/threshold number produced on the synthetic dataset so
   far (`configs/synthetic_*.yaml`, this session) is a framework-validation
   result — proof that the pipeline computes a threshold, writes the
   locked file schema, and does not crash — not a research finding about
   defense quality.** Synthetic's role is fixed as debug-only (§1); no
   accuracy, FPR, TPR, or detection-rate number produced on it should be
   quoted in the eventual report's Results section. The first numbers that
   count as research findings are PathMNIST's (STEP 5 onward), and only
   once calibration has been run under the real locked protocol on real
   data — which has explicitly not happened yet as of this addendum (STEP
   5 scope is dataset/model/partition plumbing and a clean/no-attack
   sanity check only; no τ/ρ/κ calibration and no attacks are run in STEP
   5, by instruction).

### Addendum 2026-10-05 (c) — PathMNIST official-split-to-role mapping (locked before STEP 5 code)

MedMNIST's PathMNIST already ships three official, disjoint splits:
`train`, `val`, `test`. Rather than re-deriving a calibration subset by
carving one out of `train` (as the synthetic loader does, since the
synthetic generator has no built-in val split), PathMNIST maps roles onto
the **existing** official splits directly, to preserve them exactly rather
than reshuffle them:

| Protocol role (Sec 3) | PathMNIST source |
|---|---|
| Test set | official `test` split — untouched |
| Calibration/validation subset | official `val` split, used **as-is**, in full — not further subsampled |
| Server trusted reference subset | a `server_ref_fraction` slice carved from official `train` (same mechanism as synthetic, Sec 3) |
| Client training partitions | the remainder of official `train` after the server-reference slice is removed, split IID or Dirichlet(α) |

This is a stricter reading of "preserve the official split" than the
synthetic loader's approach (which has no official val split to preserve
in the first place), and is the mapping used by
`benchmark/datasets/pathmnist.py`. The only index-overlap risk is between
`server_ref` and `client_training` (both carved from `train`); that pair,
plus all four subsets pairwise, are checked by **both** an index-set
assertion (same mechanism as `benchmark/datasets/synthetic.py`) **and** a
content-hash duplicate check across the full `train`+`val`+`test` arrays
(catching the hypothetical case of a duplicated image appearing under two
different official-split indices, which index-set disjointness alone would
not catch) — see the dataset audit report for the result of both checks.

---

### Addendum 2026-10-06 (d) — calibration-client construction from the official val split (locked)

"Val used as-is" (Addendum (c)) specifies which *samples* calibration draws
from, but not how those samples become multiple "calibration clients" for
the per-client-round scoring in §8's calibration loop. Locked here,
matching what `benchmark/runner.py::_calibrate` already implements:

1. **Source:** calibration pseudo-clients are built **only** from the
   official `val` split (`problem["calib"]`, 10,004 PathMNIST samples) —
   never from `train` (which already feeds `server_ref` + client training
   partitions) and never from `test`.
2. **Partition scheme:** the val split is partitioned into
   `n_calib_clients = max(3, cfg.dataset.n_clients)` pseudo-clients using
   the **same** partition family (`iid` or `dirichlet(alpha)`) as the
   production run being calibrated — so a Dirichlet(0.5) test-time run is
   calibrated against Dirichlet(0.5) pseudo-clients, not an IID mismatch.
3. **Seed independence:** the partition is seeded by
   `cfg.calibration.calib_seed` (default 1000), deliberately **independent**
   of `cfg.dataset.seed` — calibration pseudo-client boundaries never
   coincide with test-time client boundaries, even though both ultimately
   partition "the same kind of split" conceptually (they don't share
   samples at all here, since val vs. train are different arrays, but the
   seed independence also matters if this mapping is ever extended to a
   dataset without a separate official val split, e.g. a future
   carve-from-train design).
4. **Disjointness proof, made explicit (not just implied by "different
   official split"):** `scripts/pathmnist_characterization.py`'s setup
   step asserts, by content hash (same mechanism as
   `assert_four_way_disjoint`), that no calibration pseudo-client's images
   hash-match any image in `server_ref`, any client training partition, or
   `test`. Because calibration pseudo-clients are index-subsets of `val`
   alone, and `val` was already proven disjoint from `train`/`test` in the
   STEP 5 dataset audit, this check is expected to be — and is required to
   be — trivially zero; running it anyway turns "disjoint by construction"
   into "disjoint, proven again at the point of use," which is the
   standard this protocol holds itself to elsewhere (§3).
5. **What calibration pseudo-clients are used for:** per round, each
   pseudo-client's score (raw norm / `cos(·, g_ref)` / sign-consensus) is
   labeled honest or malicious from the calibration loop's own synthetic
   injection (§8.1) — never from `val`'s true PathMNIST tissue-class labels,
   which play no role in calibration beyond ordinary supervised loss on
   whatever classification task the client is nominally training.

---

### Addendum 2026-10-06 (e) — honest-gradient characterization study (locked before code)

Before any τ/ρ/κ is selected and before any attack is injected on
PathMNIST, we characterize the **honest-only** score distributions these
thresholds will eventually be calibrated against. This is purely
descriptive: no threshold is chosen here, no attack runs here, and no
number from this study is a defense-quality claim — it is the input to
deciding whether Cosine/Sign Consensus are even well-posed on this dataset
before we spend any effort calibrating them.

**Design:**

- **Partition grid:** `iid`, `dirichlet(alpha=1.0)`, `dirichlet(alpha=0.5)`,
  `dirichlet(alpha=0.1)` — a spread from near-IID to severe heterogeneity,
  all at `n_clients=5`, `server_ref_fraction=0.10` (the locked default).
- **Seeds:** 3 independent seeds per partition setting (`42, 43, 44`),
  governing both the client-partition draw and the honest-training RNG
  stream — "multiple seeds" per instruction, scoped to 3 for tractability;
  more can be added later without invalidating what's here, since each
  seed's rows are independently labeled in the output.
- **Rounds:** 10 rounds of **honest-only** FL training per
  `(partition, seed)` — all 5 clients are honest, the global model
  advances via the plain mean of all 5 deltas each round (no attacker to
  filter), `local_epochs=1, lr=0.01, batch_size=128` (matching the STEP 5
  clean FedAvg baseline configuration, so this study's trajectory is the
  same kind of trajectory already sanity-checked there).
- **`g_ref`:** recomputed every round from the current global weights over
  the full default-size (10%) server reference subset, exactly per §6.
- **Metrics collected per `(partition, seed, round, client)`:**
  `‖g_i‖₂`; `cos(g_i, g_ref)`; the Sign Consensus score of §7 computed over
  that round's 5 honest deltas (no malicious clients to poison the
  majority here — this measures honest-vs-honest sign agreement directly).
- **Pairwise metric, per `(partition, seed, round)`:** `cos(g_i, g_j)` for
  every client pair `i<j` (10 pairs at `n_clients=5`).
- **`g_ref` size sub-study (answers "how sensitive is `g_ref` to reference
  size"):** run only for two representative settings — `iid` and
  `dirichlet(alpha=0.1)` (mildest and most severe heterogeneity in the
  grid) — to bound the cost. Sizes tested are **nested subsets of the
  already-reserved 10% server-reference pool** (fractions `{0.1, 0.25,
  0.5, 1.0}` of that pool, i.e. roughly `{1%, 2.5%, 5%, 10%}` of the full
  training set), drawn via one fixed seeded permutation of the reserved
  pool's indices so smaller sizes are literal prefixes of larger ones —
  deliberately **not** independent resamples, so a shift in `cos(·,
  g_ref)` across sizes reflects reference-set size per se, not resampling
  noise. These sizes stay strictly inside the pool already carved out for
  `server_ref` in §3/§6 — **client training data is never touched by this
  sub-study**, so no new disjointness risk is introduced; we still log the
  class-count distribution at each size as direct evidence for the
  representativeness question.
- **No threshold selection, no attacks:** this study never calls
  `calibrate_tau/rho/kappa`, never injects any `benchmark.attacks` function,
  and produces no `accepted`/`score>=threshold` decision of any kind.

**Outputs:** per-row CSVs (not just summary numbers, so the full
distribution is always re-derivable), summary statistics (mean, std,
median, p5/p25/p75/p95/min/max) per metric per partition setting, and
distribution plots — all under `results/pathmnist/characterization/`. The
written findings (`FINDINGS.md`) answer five fixed questions (alignment
with `g_ref`, Non-IID's effect on that alignment, `g_ref` size
sensitivity, Sign Consensus stability among honest clients, and whether a
single global cosine threshold risks honest-client false positives) from
these numbers — and only these numbers; STEP 5's 53.7%/50.7%/45.9%
accuracy figures remain pipeline-sanity checks, not evidence for anything
claimed here.

---

### Addendum 2026-10-06 (f) — calibration-attack ambiguity fixed; two explicit calibration regimes locked

**Problem found:** §8.1, as originally written, says calibration injects
"the same attack named in the run's attack.name." Read literally, this
means a test run evaluating (say) `directional_poisoning` would calibrate
its τ/ρ/κ against *`directional_poisoning`-injected* validation data —
i.e., the threshold would be tuned against the very attack signature it
is about to be tested on. §8.4 already said the opposite ("the resulting
numeric τ/ρ/κ is reused for every test-time attack type"), so the
original text was internally inconsistent, and the attack-specific
reading risks violating both the original brief's "threshold selection
must be performed without looking at final test results" (§6) and this
turn's explicit "do not optimize thresholds using attack test results."
This was not caught at implementation time because the only code path
exercising `_calibrate()` so far (the `configs/synthetic_*.yaml` framework
-validation runs) always had `attack.name` match the single attack under
test, so the two readings never diverged in practice until now, when one
calibration run needs to serve multiple test-time attacks.

**Fix (locked):** calibration **always** injects one fixed, canonical
calibration-attack — `large_norm` — regardless of which attack(s) will be
evaluated at test time. `large_norm` is chosen because it is the most
generic "obviously anomalous update" signature (isotropic Gaussian noise
at a large scale), not tuned to resemble any one of the five test-time
attacks' specific structure (sign pattern, sparsity, direction). The
resulting τ/ρ/κ is frozen and reused across the full attack suite —
`no_attack, large_norm, low_norm, full_sign_flip, directional_poisoning,
sparse_coordinate_attack` — for that calibration condition. This
resolves §8.1/§8.4's inconsistency in §8.4's favor and is the version
implemented in the PathMNIST benchmark sweep (`scripts/pathmnist_benchmark.py`);
`benchmark/runner.py::_calibrate`'s synthetic-framework-validation path
(which used `cfg.attack.name`) is superseded by this fixed rule for all
PathMNIST work from this point on — a correction, logged here rather than
silently changed.

**Two calibration regimes, locked as explicitly separate, always
-labeled outputs — neither is "the" result on its own:**

- **Regime A — condition-specific (oracle / best-achievable).** τ/ρ/κ are
  calibrated against `large_norm`-injected validation pseudo-clients drawn
  under the **same** `(partition, alpha)` as the test condition being
  evaluated. This answers "how well could this detector do if it knew the
  deployment's heterogeneity level in advance and got to calibrate for
  it." Every row produced under this regime is labeled
  `calibration_regime: "A_condition_specific_oracle"` and must be reported
  with the explicit caveat that it is **not** a deployment-realistic
  number — a real deployment does not get to recalibrate its threshold
  every time client heterogeneity shifts.
- **Regime B — transfer / generalization.** τ/ρ/κ are calibrated **once**,
  against `large_norm`-injected **IID** validation pseudo-clients (IID is
  the locked reference condition — the implicit assumption a practitioner
  makes if they calibrate without accounting for client heterogeneity,
  and the condition the honest-gradient characterization study showed
  gives the cleanest, most separable honest-vs-anomalous signal). That
  frozen threshold is then applied **unchanged** when evaluating
  `α=1.0, 0.5, 0.1` test conditions. This answers "does a threshold
  calibrated under an optimistic/naive assumption remain usable as
  heterogeneity increases" — directly testing the failure mode the
  characterization study's Q5 answer predicted. Every row is labeled
  `calibration_regime: "B_iid_transfer_frozen"`. At the IID test
  condition itself, Regime B is definitionally identical to Regime A (both
  calibrate against IID `large_norm` data) — it is still recorded, but
  flagged `redundant_with_regime_a: true` rather than silently duplicated
  as if it were new evidence.
- Non-detector defenses (`fedavg`, `median`, `multi_krum`) need no
  threshold and therefore have no Regime A/B split — they are evaluated
  once per `(partition, attack, seed)` and labeled
  `calibration_regime: "not_applicable"`.

**Still locked, unchanged from §8:** thresholds are selected only from
calibration-split data, never from the attack-suite test runs; the
decision rule (maximize TPR subject to validation FPR ≤
`calibration.fpr_target`) is unchanged; validation FPR/TPR achieved at
calibration time is still always recorded alongside the chosen threshold.

---

### Addendum 2026-10-07 (g) — fixed local-step count K, locked (fixes the 8-vs-64-batch confound)

**Bug found during analysis of the first full sweep** (preserved,
unmodified, at `results/pathmnist/benchmark_diagnostic_INVALID_v1/`, see
`INVALID_README.md` there): `benchmark/models/local_training.py::local_training_seeded`
runs one full local epoch — i.e., a number of SGD steps proportional to
`len(client_data) / batch_size`. Calibration pseudo-clients (built from the
10,004-sample val split ÷ `n_calib_clients`) and real benchmark clients
(built from the much larger train pool, further split unevenly by
Dirichlet(α)) therefore performed wildly different numbers of gradient
steps per round — 8 vs. 64 batches/epoch at IID alone, and 30–130
batches/epoch *within a single Dirichlet(α=0.1) round* from partition-size
imbalance. Gradient-delta magnitude scales with step count, so every
Norm-threshold comparison across these populations was comparing
apples (few-step, small-magnitude deltas) to oranges (many-step,
large-magnitude deltas), independent of any real heterogeneity or attack
signal.

**Fix (locked).** A new primitive, `local_training_fixed_steps`, replaces
`local_training_seeded` as the benchmark's local-training primitive
everywhere a client-vs-client or calibration-vs-test comparison is made.
`local_training_seeded` itself is **not deleted** — it remains exactly as
it was, since the preserved diagnostic sweep and the synthetic
framework-validation configs from earlier in this project depend on its
exact behavior as historical record; superseding it silently in place
would make those artifacts unreproducible from the code as it then stood.

**Sampling procedure, precisely (this is the part that must be
unambiguous, not just "fixed K"):**

1. A client with local data `(X, y)` of size `n`, a step budget `K`, and a
   batch size `B` performs **exactly `K` SGD steps**, regardless of `n`.
2. Maintain one shuffled permutation of the client's own `n` local indices
   at a time (seeded by the caller-supplied `rng`, so fully reproducible).
   Draw successive batches of size `B` from the front of the current
   permutation.
3. **Wraparound rule:** before drawing a batch, if fewer than `B` indices
   remain in the current permutation (including the common case `n < B`,
   where the current permutation is smaller than one batch from the
   start), **discard the remainder, draw a fresh shuffled permutation of
   all `n` indices, and continue from its start.** This is the same
   epoch-boundary convention ordinary multi-epoch training uses
   (unused tail samples from one pass are not stitched into the next),
   generalized from "stop at the end of one pass" to "keep cycling until
   `K` steps are reached."
4. **No rebalancing.** A client's own `(X, y)` is never subsampled,
   padded, or resized to match any other client's size — a Dirichlet
   client with 500 samples and one with 30,000 samples both perform
   exactly `K` steps of batch size `B`; the 500-sample client simply
   revisits its data roughly `K·B/500` times over the round where the
   30,000-sample client revisits its data roughly `K·B/30000` times. This
   preserves each client's actual local data distribution exactly — fixing
   the step-count confound must not come at the cost of silently
   editing what Non-IID means.
5. `K` and `B` are the same two numbers for **every** caller: IID clients,
   Dirichlet clients (at every α), calibration pseudo-clients, and (per
   Addendum (h) below) the `g_ref` computation. No caller may use a
   different `K`.

**Locked defaults:** `K=50` local SGD steps, `batch_size=128` (reverting
to the batch size validated in STEP 5 / the characterization study, rather
than the `256` used in the now-invalid first sweep, for continuity with
already-validated sane learning curves). `K=50` is a deliberate, chosen
starting point — not derived from the old (buggy) per-epoch numbers — sized
so that `K·B = 6,400` samples/round is close to one full pass for the
*smallest* real Dirichlet(α=0.1) client observed in the STEP 5 audit
(~7,400 samples) and several passes for the largest. It is not swept or
tuned here; if the round-budget study (Addendum (h)) shows learning is too
slow or unstable under this `K`, that will be visible in the learning
curves and is grounds for revisiting `K` explicitly, not silently.

**Verification requirement (must be run and reported, not assumed):**
`scripts/verify_fixed_steps.py` counts the actual number of `train_step`
calls `local_training_fixed_steps` makes for a battery of client sizes
spanning smaller-than-one-batch, smaller-than-K-batches, and much-larger
-than-K-batches, and asserts the count equals `K` in every case.

---

### Addendum 2026-10-07 (h) — g_ref uses the identical fixed-K procedure; round budget to be selected empirically, not assumed

**g_ref is not automatically fixed by Addendum (g).** The original §6 said
`g_ref` uses "the same `local_epochs`/`lr`/`batch_size` as clients, for
scale comparability" — but under the old epoch-based primitive, the server
reference pool (fixed at `server_ref_fraction` of train, e.g. 8,999
samples at the default 10%) *also* had its own fixed-but-uncontrolled step
count (≈35 batches/epoch at `batch_size=256` in the invalid sweep),
incidentally different from every client population it was being compared
against. This was never separately verified — it rode along with the same
bug.

**Locked:** `g_ref`'s local training call uses `local_training_fixed_steps`
with the **same** `K` and `batch_size` as the client calls in that exact
experimental condition — not a separately-tuned value, not the old
per-epoch rule. This is now mechanically guaranteed (both call sites pass
the same module-level `K`/`batch_size` constants), not just stated in
prose, and `scripts/pathmnist_calibration_diagnostics.py` (Addendum (f)'s
gate, re-run under this fix) explicitly checks that `g_ref`'s own
step-count matches `K` alongside the client checks.

**Round budget is an open parameter, to be selected from data, not
assumed.** The invalid sweep used `N_ROUNDS=6` chosen for compute-budget
reasons, not learning-curve evidence, and the resulting near-chance
accuracy (§ diagnostic README) shows that was too short. Before any
attack/defense run under the fixed-`K` primitive, `scripts/pathmnist_round_budget_study.py`
runs **clean FedAvg only — no attacks, no detector calibration** — on IID
and Dirichlet(α∈{1.0,0.5,0.1}), 3 seeds, out to 25 rounds, and the round
budget for everything downstream (calibration diagnostics, pilot, full
corrected sweep) is fixed from where those curves stabilize. The selection
rationale, with the actual curve data, is recorded in
`results/pathmnist/round_budget_study/SELECTION.md` before any other code
in this fix depends on the chosen number.

---

### Addendum 2026-10-07 (i) — divergence safeguard, found via the pilot

The pilot (step 5 of the fix request) surfaced genuine numerical
divergence in two of its 22 combos: `iid/large_norm/fedavg` (expected —
FedAvg has no filtering at all, so an unfiltered `large_norm` update
directly enters the aggregate) and `iid/large_norm/sign_consensus` under
Regime A (consistent with that combo's calibration already showing
`validation_tpr=0.000` at IID — a genuine detection failure, not a pilot
bug). In both, one accepted `large_norm` update (isotropic noise at
`scale=50` over ~207k parameters) was enough to permanently destroy the
model — loss jumped to `1.7×10⁸` (or literal `NaN`) and accuracy froze,
for all 15 remaining rounds. This is a legitimate simulation outcome (a
defense that fails to catch a large-norm update *should* show
catastrophic failure), not something to suppress or paper over — but
letting `K=50`-step local training continue for 15 more rounds on an
already-destroyed model wastes GPU time across hundreds of combos and
produces rows that would corrupt a naive mean (e.g. averaging a `1.7×10⁸`
loss into a seed-mean is meaningless).

**Fix (locked, `scripts/pathmnist_benchmark_v2.py::run_combo`):** after
each round's evaluation, if `loss` is non-finite or exceeds `50.0` (sane
9-class cross-entropy is O(1)–O(3); `50.0` is unambiguously catastrophic,
not a borderline tuning choice), the combo stops training and copies the
diverged row forward for all remaining rounds, each flagged
`diverged: true`. Every combo still contributes exactly `N_ROUNDS` rows
(schema stays consistent for aggregation), but at a fraction of the
compute, and `diverged` lets the aggregation step treat "the model was
destroyed" as its own reportable category rather than silently averaging
huge-loss rows into a utility number.
