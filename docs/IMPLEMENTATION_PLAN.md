# Implementation Plan — ZKFL-PQ Benchmark Extension

Status: STEP 1–2 deliverable. Written after reading every file in `crypto/`,
`fl_core/`, `experiments/`, `formal/`, `results/*.json`, `README.md`,
`SECURITY.md`. No source code has been modified yet.

---

## 1. Current architecture (as found)

### 1.1 `fl_core/` — model + FL primitives (pure NumPy, CPU-only)

| File | Contents |
|---|---|
| `model.py` | `SimpleMLP` (2-hidden-layer MLP, default `784→128→64→4`, configurable `hidden=(h1,h2)`); flat-vector `get_weights`/`set_weights`/`get_gradient_vector` contract used by **every** crypto gadget downstream; `generate_synthetic_medical_data` (Gaussian-blob-per-class + AR(1)-style feature correlation); `load_breast_cancer_medical` (UCI via `sklearn`, always offline); `load_medical_dataset` dispatcher (`synthetic`/`breast_cancer`/`medmnist*`); `partition_non_iid` (Dirichlet client split). |
| `cnn.py` | `ConvNet28` — small NumPy CNN (`Conv(1→8)→Pool→Conv(8→16)→Pool→FC→FC`) for 28×28 imaging, same flat-vector API as `SimpleMLP`. No GPU path. |
| `clip.py` | `clip_infty`, `dual_norm_ok` (ℓ∞ clip + dual ℓ2/ℓ∞ membership check). |
| `robust_agg.py` | `coord_median`, `multi_krum`, `robust_aggregate` dispatcher (`median` default / `krum` / `mean`, env `ZKFL_ROBUST_AGG`). |
| `adaptive_tau.py` | `AdaptiveTau` — quantile-based public τ schedule from *accepted* norms history (EMA-smoothed), bound into next round's associated data. |

### 1.2 `crypto/` — PQ transport, ZKP, HE (pure NumPy / Python `object` big-int, CPU-only)

| File | Contents |
|---|---|
| `ml_kem.py` | `MLKEM768` — simplified FIPS-203-shaped ML-KEM-768 (k=3, n=256, q=3329), naive O(n²) polynomial multiply (no NTT); AES-256-CTR symmetric wrapper. |
| `zkp_norm.py` | `ZKPNormBound` — Fiat-Shamir Σ-protocol proving `‖Δw‖₂ ≤ τ` via an SIS-style `LatticeCommitment`; challenge binds `associated_data` (HE ciphertext bytes); rejection sampling with `max_attempts=10`. `ZKPBatchNormBound` chunks over `chunk_size`. |
| `qrom_nizk.py` | `UnruhNormNIZK` — Unruh transform of the above, `reps=128` default binary parallel sessions, invertible RO points `(ρ, H(ρ))`, **one shared** `LatticeCommitment` matrix `A` across all sessions. |
| `enc_consistency.py` | `EncConsistencyGadget` — classical-FS Σ-proof that `(coins, plaintext)` correctly produced a BFV ciphertext chunk. |
| `unruh_enc_consistency.py` | `UnruhEncConsistency` — Unruh-lifted version of the above, `reps=16` default. |
| `partial_dec_nizk.py` | `PartialDecryptNIZK` — proves a threshold party's partial decryption `μ_i = c1⋆s_eff` is correctly formed; `threshold_decrypt_with_nizk` aborts the whole open on any failing party. |
| `round_transcript.py` | `RoundTranscript` — SHA3 hash chain; `advance()` folds the previous round's accept-set into next round's associated data (prevents cross-round replay/splicing). |
| `homomorphic.py` | `BFVScheme` (full negacyclic-ring BFV, two HE presets `classic128`/`classic128_demo` via `ZKFL_HE_PRESET`); `ThresholdBFV` ((t,n) Shamir-shared secret key, **never reconstructs** `sk`, Lagrange-weighted partial decrypt + smudging noise); `GradientHEManager` (chunks a length-`d` gradient into `HE_N`-sized polynomials, threaded encrypt). |
| `fused_he.py` | `FusedSealThresholdHE` — primary path is always NumPy `ThresholdBFV`; if TenSEAL is importable, a parallel SEAL ciphertext is also produced and a **post-hoc numerical consistency check** (not a real dual-aggregation) is logged. `create_he_manager()` factory keyed on `ZKFL_HE_BACKEND` (`fused`/`numpy`/`tenseal`). |
| `seal_backend.py` | `TenSEALGradientHE` — SEAL-only path, **single decryptor** (no threshold), only usable if `tenseal` is installed. |
| `lattice_security.py` | `report_he_security()` — prints the preset's `(n, log2 q, claimed_bits)`; optional `lattice_estimator` integration (not installed here) for a real estimate. Explicitly documents that the NumPy `n=512` demo is *not* a certified Classic-128 instance. |
| `keccak_f1600.py` | Bit-level Keccak-f[1600] + SHA3-256 sponge, matches `hashlib` byte-for-byte; used only by `formal/` checkers, not by the hot path (which uses `hashlib.sha3_256`). |

### 1.3 `formal/` — mechanized checks, **not** a full automated proof

`run_formal_ci.py` runs, in order: `check_unruh_soundness` (exhaustive combinatorial check of the 2⁻ʳ Unruh bound for small r, asserts the r=128 claim), `check_unruh_qrom_games` (hand-written G0→G3 game-hop mechanization in Python, not EasyCrypt), `check_sha3_qrom` (FIPS test vector + **static substring grep** of the `.ec` files for required lemma/operator names), `check_keccak_bitlevel` (bit-level Keccak vs `hashlib`, algebraic lane lemmas, **static grep** of `KeccakF1600.ec`). It then checks the `.ec` files exist and contain certain strings, and *optionally* shells out to `lake`/`easycrypt` binaries **only if found on PATH** (neither is installed in this environment — confirmed). **The EasyCrypt/Lean sources are not type-checked by this repository's CI as configured here; "formal" results are (a) Python-mechanized combinatorics/game-hops and (b) static presence-checks of lemma names in `.ec`/`.lean` source text.** This distinction must be preserved verbatim in reporting — do not describe `FORMAL_CI_OK=1` as "proof discharged."

### 1.4 `experiments/` — entry points (each duplicates its own CONFIG + training loop)

| Script | Role | Key config |
|---|---|---|
| `run_experiment.py` | **Main/original protocol.** 3 arms (`standard` FedAvg, `+ML-KEM`, `+ZKP+HE` hybrid) + 2 ablations (n_malicious, τ). | `n_clients=5, n_rounds=10, n_samples=1000, n_features=784, n_classes=4, dirichlet_alpha=0.5, norm_threshold=5.0 (τ), malicious_client_id=3, malicious_scale=50.0, seed=42`. **This exactly matches the "original baseline" described in the task brief** (1000/784/4/5/α=0.5/τ=5) → this is the upstream-reproduction target for STEP 3. |
| `run_baselines.py` | 5-seed (42–46) × 7-method (`fedavg, clip, multi_krum, he_only, zkp_only, hybrid, hybrid_median`) × 2-attack (`large_norm, sign_flip`) sweep on synthetic data. Imports `CONFIG`/`local_training` from `run_experiment.py` via `importlib`. |
| `run_backdoor.py` | Separate trigger/backdoor study (own CONFIG: 15 clients, 25 rounds, label+pixel trigger poisoning), compares `fedavg/zkp_l2/krum/hybrid_zkp_krum/hybrid_zkp_median`. |
| `run_target_protocol.py` | "Default composition" demo on UCI breast cancer: fused HE + Unruh NIZK + Enc-consistency + threshold decrypt + `robust_aggregate`. |
| `run_innovation_pack.py` | Smoke-tests Enc-consistency/PartialDecrypt-NIZK/AdaptiveTau/RoundTranscript/dual-norm gadgets + a sparse-ℓ∞ ablation + one UCI run exercising all of them together. |
| `run_medmnist.py` / `run_medmnist_cnn.py` / `run_medmnist_fullres.py` | MedMNIST (PneumoniaMNIST by default) variants: random-projected MLP, full-res `ConvNet28`, full-res MLP. **`medmnist` package is not installed in this environment** (confirmed below); these currently fail or fall back to `breast_cancer`. |
| `run_scale.py` | N=20/T=30 scale study, methods `fedavg/clip/krum/zkp/hybrid` (hybrid uses "hybrid-lite": ZKP verify gate + plaintext mean, full HE only if `ZKFL_SCALE_FULL_HE=1`). |
| `plot_figures.py`, `make_excellence_figure.py` | Plot `results/*.json` into `figures/`. |
| `smoke_residuals.py` | Enc-consistency unit test + optional TenSEAL test (skipped, not installed) + formal Unruh check. |

### 1.5 `results/*.json` — no shared schema

Each script writes its own flat JSON with its own keys (confirmed by inspection: `experiment_results.json`, `baseline_results.json`, `target_protocol_results.json`, `innovation_pack_results.json`, `backdoor_results.json`, `scale_results.json`, `medmnist*_results.json`, `hybrid_median_ablation.json`, `excellence_summary.json` — 11 files, 11 different top-level shapes). There is **no `config.json`/`per_round.csv`/`client_scores.csv` convention** anywhere in the repo today — this is new infrastructure we must build (STEP 14).

### 1.6 Environment actually available (checked, not assumed)

- Python 3.10.12, `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `cryptography` installed (match `requirements.txt`).
- `medmnist`, `tenseal`, `flamby`, `lattice_estimator` — **not installed** (all four `ImportError`). Required before STEP 5/6.
- GPU: NVIDIA GeForce GTX 1660 SUPER, 6 GB VRAM, driver CUDA 13.0; `torch==2.13.0+cu130` installed, `torch.cuda.is_available() == True`.
- **The entire upstream stack (`fl_core/`, `crypto/`) is pure NumPy / CPU and has zero GPU hooks.** The GPU is only usable if we add a separate torch model path for the new real-dataset arms (see §3, §4).

---

## 2. Reusable components (use as-is, do not modify)

- `fl_core.model.SimpleMLP` / `fl_core.cnn.ConvNet28` — flat-vector contract is exactly what the common benchmark runner needs for the gradient-defense arms.
- `fl_core.model.partition_non_iid` — Dirichlet partitioner; reusable for the synthetic and PathMNIST IID/non-IID splits, as long as we persist its output (see §4).
- `fl_core.robust_agg.robust_aggregate` — gives us FedAvg (`mean`), Coordinate-wise Median, and one Multi-Krum implementation for free as 3 of the 7 required defense baselines.
- `experiments.run_experiment.local_training` — the one local-SGD primitive; every entry point already reuses it via `importlib`. The new benchmark runner will do the same.
- All of `crypto/` — untouched, reused verbatim for the "ZKFL-PQ" row of the defense comparison and for the upstream-reproduction path. We are not auditing or fixing it as part of the benchmark (separate findings doc, §6).

## 3. Components that need extension (gaps vs. the task brief)

1. **No attack interface.** Poisoning logic is duplicated ad hoc in 5 places (`run_experiment.py` inline, `run_baselines._poison`, `run_scale._poison`, `run_backdoor.poison_client_data`, `run_medmnist_cnn.py` inline), and none of the 6 required scenarios (`no_attack, large_norm, low_norm, full_sign_flip, directional_poisoning, sparse_coordinate_attack`) exist as a reusable, independently-callable module. The existing `"sign_flip"` in `run_baselines.py` is actually a **directional attack scaled to exactly τ** (`-delta * tau/‖delta‖`), not a true per-coordinate sign flip — we will implement both `full_sign_flip` (negate every coordinate of the honest delta) and `directional_poisoning` (the existing τ-scaled-opposite-direction attack) as distinct, correctly-named scenarios.
2. **No defense interface.** Norm-filtering-as-a-standalone-defense, Cosine filtering, and Sign Consensus do not exist anywhere in the repo. `ZKPNormBound` conflates a cryptographic ZK proof with the plain norm-threshold decision it's proving — we need a **non-cryptographic** norm filter (`‖g_i‖ ≤ τ → accept`) as its own ablation-independent baseline, since the task brief treats "Norm" as a signal, not a ZK proof.
3. **Two non-canonical Multi-Krum implementations already disagree in detail** (`fl_core.robust_agg.multi_krum` uses `n - f - 2` neighbours; `experiments.run_baselines.multi_krum_aggregate` uses `max(1, n-f-2)`; `experiments.run_backdoor.multi_krum` returns a *list* not an aggregate). We will **not** silently "fix" the upstream copies — we pick `fl_core.robust_agg.multi_krum` as the one canonical implementation for the new benchmark and leave the upstream scripts exactly as they are.
4. **No g_ref (trusted server reference) construction anywhere.** Cosine filtering is impossible without one; we must build a trusted-server-dataset split that is disjoint from both client training data and the test set (task brief §7 is explicit that it must never be test data).
5. **No config/YAML system.** Every script hardcodes a Python `dict`. We need a schema (dataclass) + loader and one `configs/*.yaml` per experiment cell.
6. **No persisted/deterministic partitions.** `partition_non_iid` is called fresh, in-memory, per script invocation. The task brief requires partitions "saved to disk so every defense receives EXACTLY the same client data" — currently nothing is saved.
7. **No GPU path for real datasets.** PathMNIST and Fed-ISIC2019 at realistic scale need a torch CNN/ResNet to be remotely practical on a 6 GB GPU; the existing NumPy `ConvNet28` would be the fallback for small-scale/CPU-only verification but is not the primary model for STEP 5/6.
8. **No PathMNIST or Fed-ISIC2019 loaders.** `load_medical_dataset` only knows `synthetic`/`breast_cancer`/a generic `medmnist*` key that defaults unrecognized names to `pneumoniamnist`. PathMNIST-specific loading (official train/val/test split, 9-class) and FLamby Fed-ISIC2019 (natural center partition) do not exist and must be written new.
9. **Reproducibility gaps in the upstream scripts (must be preserved, not fixed, in the reproduction path; must be fixed in the new framework):**
   - `experiments/run_experiment.py`'s malicious-delta injection (`run_standard_fl`, `run_fl_mlkem`, `run_fl_hybrid`, both ablations) draws from the **global, unseeded** `np.random.normal(...)`, not a `seed`-derived generator.
   - `local_training()` (shared by *every* entry point) shuffles minibatches with an **unseeded** `rng = np.random.default_rng()`.
   - Net effect: re-running `python experiments/run_experiment.py` with the same `CONFIG['seed']=42` does **not** reproduce bit-identical numbers run-to-run. Only the model init (`SimpleMLP(..., seed)`), data generation, and partitioning are actually seeded. This must be stated plainly in the upstream-reproduction report (§ "what could not be reproduced exactly") rather than silently patched.

## 4. Compatibility risks

- **Crypto-path cost does not scale to real imaging models.** The README's own numbers (ConvNet28, `d≈51618`, HE encrypt ≈3.5s/client/round, Unruh prove/verify ≈0.10/0.06s at `r=4`) are already at the edge of interactive latency for a *compact* CNN. A realistic PathMNIST or Fed-ISIC2019 model (10⁴–10⁷ params) through the full ZKP+HE+Unruh+threshold stack is not computationally tractable in this environment. **Decision:** the full crypto stack ("ZKFL-PQ" row) stays scoped to the synthetic/UCI/compact-MLP arms for which it was designed; on PathMNIST/Fed-ISIC2019 we report the crypto-stack numbers only if a reduced/compact-head model is used and we say so explicitly, and otherwise mark that cell "not run — crypto stack cost-prohibitive at this scale" rather than faking a number.
- **Flat-vector contract.** Every crypto gadget binds to `dim == len(gradient)` exactly (`ZKPNormBound.__init__`, `GradientHEManager.gradient_dim`). Any new torch-based model used in a crypto-stack arm would have to export the identical flat `get_weights`/`set_weights`/`get_gradient_vector` API; for the pure gradient-defense arms (no crypto) this constraint does not apply and a standard torch training loop is fine.
- **Optional dependencies gate entire experiments.** `medmnist`, `tenseal`, `flamby` must be installed before STEP 5/6; `tenseal` additionally changes the *default* HE backend (`ZKFL_HE_BACKEND` defaults to `fused` only if TenSEAL is importable) — every run must log the actually-used backend (the target-protocol script already does this; we will carry the same field into the new runner).
- **FLamby/Fed-ISIC2019 licensing.** Unlike MedMNIST, FLamby's ISIC2019 loader requires a manual, separate dataset download and per-source license acknowledgement; this cannot be scripted blindly and will need a manual confirmation step before STEP 6.
- **Partial parameter coverage in the existing hybrid protocol.** `run_fl_hybrid`'s "unprotected" coordinates beyond `PROTECTED_DIM = min(HE_N, n_params)` are aggregated by a **plain mean of accepted clients' full deltas**, not gated by any crypto check — the in-code comment calls this out as not closing suffix-poisoning. This is a genuine upstream limitation we will document in `UPSTREAM_SECURITY_FINDINGS.md`, not silently patch.
- **`ZKFL_HE_BACKEND=fused` default depends on `tenseal`.** Since `tenseal` is not installed here, all our crypto-stack runs will silently use `numpy`-only threshold HE unless we `pip install tenseal`. This is fine but must be recorded per-run, not assumed.

## 5. Proposed directory structure (additive — upstream tree untouched)

```
pq-zkfl-medical/
├── crypto/ fl_core/ experiments/ formal/ manuscript/ figures/   # UNTOUCHED
├── results/
│   ├── upstream_reproduction/     # NEW — raw outputs of unmodified experiments/*.py only
│   ├── synthetic/                 # NEW — common-framework synthetic-dataset runs
│   ├── pathmnist/                 # NEW
│   └── fed_isic2019/              # NEW
├── benchmark/                      # NEW shared framework package
│   ├── config.py                  # dataclass schema + YAML loader, seed plumbing
│   ├── datasets/
│   │   ├── synthetic.py           # wraps fl_core.model synthetic + partition, persists to disk
│   │   ├── pathmnist.py           # MedMNIST PathMNIST, official splits, IID + Dirichlet(α) partitions persisted
│   │   └── fed_isic2019.py        # FLamby natural-center loader (centers preserved, no re-partitioning)
│   ├── models/
│   │   ├── flat_mlp.py            # thin re-export of fl_core.model.SimpleMLP for the crypto-stack arms
│   │   └── torch_cnn.py           # NEW torch CNN/ResNet for PathMNIST/Fed-ISIC2019 gradient-defense arms (GPU)
│   ├── attacks/
│   │   ├── base.py                # Attack protocol: attack(delta, rng, cfg) -> delta
│   │   └── suite.py               # no_attack, large_norm, low_norm, full_sign_flip, directional_poisoning, sparse_coordinate_attack
│   ├── defenses/
│   │   ├── base.py                # Defense protocol: aggregate(deltas, ctx) -> (update, per_client_scores/accept_mask)
│   │   ├── fedavg.py, norm.py, cosine.py, sign_consensus.py, median.py, krum.py, combined.py
│   ├── calibration/
│   │   ├── tau_calibration.py     # validation-only τ selection, FPR/TPR logging
│   │   ├── rho_calibration.py     # g_ref construction + cosine threshold ρ, sensitivity to ref-set size
│   │   └── kappa_calibration.py   # sign-consensus threshold κ
│   ├── runner.py                  # one FL loop: dataset+model+attack+defense from one config, writes standard outputs
│   └── metrics.py                 # utility/security/cost metrics, config.json/metrics.json/per_round.csv/client_scores.csv/timing.csv writers
├── configs/                        # NEW — one YAML per experiment cell
├── scripts/                        # NEW — thin CLIs over benchmark.runner (smoke/small/full)
├── docs/
│   ├── IMPLEMENTATION_PLAN.md      # this file
│   ├── BENCHMARK_PROTOCOL.md       # STEP 19 deliverable
│   └── UPSTREAM_SECURITY_FINDINGS.md
└── plots/                          # NEW — generated figures for the benchmark (kept separate from figures/ and manuscript/.../figures/)
```

## 6. Exact experimental plan (maps task-brief §§1–16 to concrete work)

1. **STEP 3 — Upstream reproduction.** Run `python experiments/run_experiment.py` unmodified, `CONFIG` as shipped (seed=42, τ=5, 1000/784/4/5/α=0.5). Save raw stdout + the emitted `results/experiment_results.json` copy into `results/upstream_reproduction/` with a `config.json` snapshot and a note on the two unseeded-RNG gaps from §3.9. Do **not** average across reruns and call it reproduced-exactly; report the single-run numbers and flag non-determinism.
2. **STEP 4 — Common framework.** Build `benchmark/config.py`, `benchmark/runner.py`, `benchmark/metrics.py`, `benchmark/attacks/`, `benchmark/defenses/{fedavg,norm,median,krum}.py` (the 4 that don't need a trusted reference) first, validated on the synthetic dataset against the upstream `fedavg`/`clip`/`multi_krum` numbers as a sanity cross-check (not a hard equality requirement, since RNG streams differ by construction).
3. **STEP 5 — PathMNIST.** `pip install medmnist`; write `benchmark/datasets/pathmnist.py` using official train/val/test; generate + persist IID and Dirichlet(α∈{0.1,0.5,1.0}) partitions once per seed; add `benchmark/models/torch_cnn.py` (GPU).
4. **STEP 6 — Fed-ISIC2019.** `pip install flamby` + manual dataset acquisition/license step (cannot be scripted without user action); write `benchmark/datasets/fed_isic2019.py` preserving native FLamby centers verbatim.
5. **STEP 7 — Attacks.** Implement all 6 scenarios in `benchmark/attacks/suite.py` with a configurable malicious-ratio; unit-test each against a trivial honest/malicious norm check before wiring into the runner.
6. **STEP 8 — Defenses.** `norm.py` (plain threshold, no ZK), `cosine.py` (needs g_ref, built in calibration step), `sign_consensus.py` (new, literature-grounded coordinate-sign-agreement score, documented explicitly), `median.py`/`krum.py` (wrap `fl_core.robust_agg`), `combined.py` (configurable signal subset for the ablation grid in §12 of the brief).
7. **STEP 9 — Calibration.** `tau_calibration.py`/`rho_calibration.py`/`kappa_calibration.py` select thresholds on a validation split only, before any test-set run; g_ref construction/validation (trusted server dataset, size sensitivity) happens here.
8. **STEP 10 — Combined Attestation + ablations.** 7-way signal-subset grid (Norm / Cosine / Sign / N+C / N+S / C+S / N+C+S), transparent (unweighted/documented) combination rule first; a weighted/adaptive variant, if added later, is a separate experiment with its own validation-based weight-selection writeup.
9. **STEP 11–12 — Small pilot.** One seed, one dataset (synthetic), reduced rounds, inspect `results/.../{config,metrics,per_round,client_scores,timing}` files for correctness before anything bigger.
10. **STEP 13 — Full matrix.** Only after STEP 11–12 pass and compute-cost is estimated (dataset × partition × attack × ratio × defense × seed count), consistent with the brief's explicit "do not launch a huge matrix blindly" instruction.
11. **STEP 14–15 — Aggregation + plots.** CSV rollups + the plot list in the brief §15, generated from the standard per-run files only (no fabricated points for missing cells).

## 7. What this plan deliberately does NOT do yet

- No source file under `crypto/`, `fl_core/`, `experiments/`, or `formal/` has been modified.
- No results have been generated yet (STEP 3 reproduction is queued next).
- No parameters have been tuned on any test set.
- `docs/UPSTREAM_SECURITY_FINDINGS.md` is tracked as a separate deliverable and will list crypto-layer concerns independently of this plan (per task-brief §17, implementation bugs vs. conceptual limitations are kept in separate sections there).
