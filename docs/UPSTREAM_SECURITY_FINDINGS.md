# Upstream Security Findings — pq-zkfl-medical

Independent read of `crypto/`, `fl_core/`, `formal/`, `README.md`, `SECURITY.md`
as of commit `e6a6d5a` (HEAD of `main` at clone time). This document is
deliberately **separate** from the benchmark work: it records what we found
while reading the cryptographic implementation, not whether our new gradient
defenses outperform it. Nothing here has been fixed in-place — see
`docs/IMPLEMENTATION_PLAN.md` for why (task brief: do not silently fix
upstream crypto while benchmarking gradient defenses).

Several of these points are already disclosed by the authors in
`SECURITY.md`/`README.md` (the "Parameter honesty" and "We do not claim"
tables are unusually candid). Where that's true we say so and add only what
is *not* already stated. Where we found something not previously disclosed,
we say that too.

---

## 1. ZKP soundness concerns

- **Two different soundness regimes are in use across the experiment scripts, and most scripts use the weaker one.** `crypto/zkp_norm.ZKPNormBound` is a single-shot classical Fiat-Shamir Σ-protocol whose challenge is derived as `int.from_bytes(sha3_256(...)[:4], 'little') % 256 + 1` — a **256-ary challenge space**, giving soundness error on the order of `1/256` per proof, not the "128-bit" figure the README advertises. The actual 128-bit-class soundness comes only from `crypto/qrom_nizk.UnruhNormNIZK`, which repeats `reps=128` independent binary sessions under the Unruh transform. Checking which experiment scripts use which: `run_target_protocol.py`, `run_innovation_pack.py`, `run_medmnist_cnn.py`, and `run_medmnist_fullres.py` use `UnruhNormNIZK`. **`run_experiment.py`, `run_baselines.py`, `run_backdoor.py`, `run_scale.py`, and `run_medmnist.py` call `ZKPNormBound` directly** — i.e. the main/original protocol script and its 5-seed baseline sweep run at ~8-bit (1/256) soundness, not the 128-bit class the README's threat table implies for "ZKP"/"Hybrid" generally. This distinction is not called out anywhere in `README.md`'s "Parameter honesty" table, which only discusses the Unruh `r` knob, not which scripts use Unruh at all. **Not previously disclosed.**
- **No stated bit-security for the SIS lattice commitment.** `LatticeCommitment` (`COMMIT_N=128, COMMIT_Q=7681, COMMIT_M=256`) underlies every proof (classical and Unruh). Unlike the HE presets, which carry an explicit `claimed_bits` field and an honesty caveat, the commitment scheme's own SIS hardness is never estimated or documented anywhere in the repo. **Not previously disclosed.**
- The rejection-sampling bound `B_reject = sigma_mask * sqrt(dim) * 1.5` (in `ZKPNormBound.__init__`) is a heuristic constant, not a value derived from a stated zero-knowledge simulation argument; generation retries up to `max_attempts=10` and — correctly — the verifier always recomputes `z_norm` itself rather than trusting `proof["accepted"]`, so this does not weaken verification, but the "1.5" slack factor's origin is undocumented.

## 2. HE limitations

Most of the headline caveats here are already disclosed candidly in
`README.md`/`SECURITY.md` (the `classic128_demo` n=512 preset is explicitly
flagged as "not a HomomorphicEncryption.org certificate"). What we add:

- **`FusedSealThresholdHE`'s "consistency check" does not check what its name implies.** `decrypt_aggregated` in `crypto/fused_he.py` re-encrypts the *already-recovered* NumPy plaintext mean under SEAL and immediately decrypts it back, then reports the residual as `last_seal_consistency_err`. This trivially succeeds (it is testing SEAL's own round-trip, not whether the SEAL-side and NumPy-side aggregates from the *actual client ciphertexts* agree). The in-code comment acknowledges this directly: `"Stronger: aggregate last-round client SEAL cts if stored"` — i.e., the stronger check is not implemented. Any reported `seal_consistency_err≈0` should not be read as cross-backend validation. **Not previously disclosed** (the weakness is visible in code but not stated in prose anywhere).
- **Threshold-decryption smudging noise uses the same σ as encryption noise.** `ThresholdBFV.threshold_decrypt`/`threshold_decrypt_with_nizk` add smudging noise via `_he_sample_error(..., sigma=HE_SIGMA)`, i.e. `σ=3.2`, identical to the scheme's own encryption-noise distribution. The threshold-HE literature (e.g. smudging-based simulation arguments for partial decryption) generally wants smudging noise several orders of magnitude wider than the ciphertext noise to statistically hide a single party's share-dependent contribution; reusing the encryption σ is a plausible under-provisioning of that guarantee. This is a parameter choice, not a crash bug, and we are not able to quantify the actual leakage without a dedicated analysis — flagging it as worth checking before any privacy claim is made about the threshold-open step specifically (as opposed to the aggregate-sum step).
- No automated noise-budget assertion exists anywhere in `homomorphic.py`: `decode()`'s `int(round(int(x) * t / q)) % t` rescale has no check that the accumulated noise after `homomorphic_add_many` across `n_clients` ciphertexts stayed inside the correctness bound for the modulus/plaintext-space pair in use. The repo does empirically log `he_reconstruction_errors`/`seal_consistency_err` per run (good practice), but there is no analytic guard if someone scales `n_clients`, `scale`, or the gradient magnitude beyond what's been empirically tested.

## 3. ML-KEM concerns

- **`crypto/ml_kem.py` implements IND-CPA-shaped Kyber math, not IND-CCA2 ML-KEM.** FIPS 203 (and the Kyber submission it derives from) achieves CCA2 security via a Fujisaki-Okamoto-style re-encryption check inside decapsulation (recompute the ciphertext from the recovered message and compare; on mismatch, return a pseudorandom "implicit rejection" value instead of the real shared secret). `MLKEM768.decaps()` here does neither: it decodes `m_recovered` from `v - s·u` by nearest-point rounding and hashes it directly, with no re-encryption check and no implicit-rejection fallback. The module's own docstring is honest about scope ("Simplified implementation **faithful to FIPS 203 mathematical structure**"), but `README.md`'s threat-coverage table lists "ML-KEM-768 | PQ transport (FIPS 203 L3)" without a CCA caveat, which could be read as a stronger security claim than the code supports. **Not previously disclosed at the claims-table level**, though the code-level docstring is candid.
- **`_poly_mul_ntt_naive` contains no NTT.** It is a plain O(n²) double-loop schoolbook polynomial multiplication; the name is misleading (no number-theoretic transform is used). Purely a naming/documentation issue — the arithmetic result is correct — but worth fixing if this code is ever used as a reference for an actual NTT implementation.
- **Public matrix `A` is sampled with a weaker entropy source than FIPS 203 specifies.** `_sample_matrix_A` reseeds a fresh `np.random.default_rng(int.from_bytes(h[:8], 'little'))` per `(i,j)` cell from only the first 8 bytes of a SHA-256 hash, rather than using a single continuous XOF (SHAKE-128) stream as FIPS 203 does. Because `A` is public, this does not directly threaten MLWE hardness, but it is a structural deviation worth recording since it reduces the effective sampling entropy per cell to 64 bits.
- `symmetric_encrypt`/`symmetric_decrypt` use AES-256-**CTR** with a random IV and no MAC — confidentiality only, not authenticated encryption. Payload tampering at this layer alone would go undetected (the downstream ZKP/Enc-consistency binding separately protects the *gradient* payload's integrity in the full-protocol scripts, but the raw KEM/symmetric "channel" primitive itself is not AEAD).

## 4. Privacy / robust-aggregation conflict

This is the most significant finding in this document, and it is **not**
stated anywhere in `README.md` or `SECURITY.md` as currently written.

`SECURITY.md` lists `ZKFL_ROBUST_AGG=median` as the fix for "Sign-flip /
in-bound backdoors vs ℓ2 alone," and separately lists threshold `(t,n)`
partial decryption as closing the "curious aggregator" / "single decryptor"
residual. **These two claims are not jointly true the way the code is
wired.** In `experiments/run_target_protocol.py` (the "target protocol"
script, i.e. the paper's own default composition demo):

```python
if accepted_cts:
    opened = []
    for cts in accepted_cts:
        vec, _ = he.decrypt_aggregated(cts, 1)   # <-- decrypts ONE client's own ciphertext
        opened.append(vec)
    update = robust_aggregate(opened, method=robust, f=CONFIG["robust_f"])
```

`coord_median`/`multi_krum` need per-client plaintext values to compute a
coordinate-wise median or pairwise distances — there is no way to run them
on a still-encrypted sum. The code resolves this by **individually
threshold-decrypting every accepted client's own ciphertext** before handing
the plaintext deltas to `robust_aggregate`. The server therefore sees each
accepted client's full individual gradient in the clear, every round, under
`ZKFL_ROBUST_AGG=median` or `krum` — functionally the same server-side
visibility as having no HE at all for those clients. The threshold/HE layer
in this configuration only protects values in transit and against a
non-colluding external observer; it does **not** protect an individual
client's gradient from the server itself, despite `SECURITY.md`'s "Single
decryptor" and "curious aggregator" rows implying that threshold decryption
is the mitigation for exactly that threat. The mitigation and the
default robust aggregator are mutually exclusive in practice, and the
median default (`ZKFL_ROBUST_AGG` defaults to `median` per
`fl_core/robust_agg.py`) is the common case, not an edge case.

This is a direct, practical reason the new benchmark's gradient-level
defenses (which also require visibility into individual client deltas) are
not a strictly weaker privacy posture than the "secure" median-aggregated
hybrid protocol already shipped here — both see raw per-client gradients at
the server.

## 5. Implementation bugs (fixable; not fundamental to the design)

1. Three divergent Multi-Krum implementations exist (`fl_core/robust_agg.multi_krum`, `experiments/run_baselines.multi_krum_aggregate`, `experiments/run_backdoor.multi_krum`), differing in neighbour-count guards and return type (aggregate vs. filtered list). Not wrong individually, but inconsistent across the same repo.
2. `_poly_mul_ntt_naive` is a mislabeled schoolbook multiply (see §3).
3. `FusedSealThresholdHE.decrypt_aggregated`'s SEAL "consistency check" is tautological (see §2).
4. `run_experiment.py`'s hybrid arm (and its two ablation functions) truncates crypto protection to `PROTECTED_DIM = min(HE_N, n_params)` and aggregates the remaining coordinates by plaintext mean of accepted full deltas — the in-code comment calls this "not private; pass-the-proof adversaries can still poison suffix." Notably, **every other full-protocol script in the repo** (`run_target_protocol.py`, `run_innovation_pack.py`, `run_medmnist.py`, `run_medmnist_cnn.py`, `run_medmnist_fullres.py`) already chunks the *entire* gradient through `GradientHEManager`/`create_he_manager` with no truncation — so this is a scope limitation specific to one script, already shown to be fixable within the same codebase, not a fundamental barrier.
5. Smudging-noise σ reuse and absent SIS/commitment bit-security estimate (see §§1–2) are parameter/documentation gaps, not structural flaws.
6. AES-CTR-without-MAC in the raw symmetric wrapper (see §3) — switching to AES-GCM is a small, local fix.
7. Experimental (non-cryptographic) reproducibility gap, noted for completeness since it affects how any of the above should be re-measured: the malicious-delta injection in `run_experiment.py` and the minibatch shuffle in the shared `local_training()` both draw from the unseeded global NumPy RNG rather than a seed-derived generator, so re-running the same `CONFIG['seed']` does not reproduce bit-identical numbers. Full detail and the plan for the new framework's seeding are in `docs/IMPLEMENTATION_PLAN.md` §3.9; listed here only as context for anyone trying to re-verify the findings above numerically.
8. **`experiments/run_medmnist_cnn.py` never resets the model to the round's global weights before each client's local training — found during our STEP 3 reproduction, not previously disclosed.** Every other full-protocol script (`run_experiment.py`, `run_baselines.py`, `run_backdoor.py`, `run_target_protocol.py`, `run_medmnist.py`, `run_medmnist_fullres.py`) explicitly snapshots `gw = model.get_weights().copy()` once per round and trains a **fresh** local model from `gw.copy()` for every client. `run_medmnist_cnn.py`'s round loop instead calls `local_training(model, Xi, yi, ...)` passing the **shared global `ConvNet28` instance directly**, with no per-client reset:
   ```python
   for cid, (Xi, yi) in enumerate(parts):
       delta = local_training(model, Xi, yi, cfg["local_epochs"], cfg["local_lr"], cfg["batch_size"])
   ```
   Because `local_training` both trains *and* leaves the model mutated at its new weights (it returns `delta = model.get_weights() - initial_weights` but does not restore `initial_weights` afterward), each successive client in the same round continues training from the **previous client's already-updated weights** rather than from a common round-start point — this is cascading/sequential SGD across clients, not parallel federated local updates. The round loop then **adds the aggregated per-client deltas on top of the already-cascaded weights** (`model.set_weights(model.get_weights() + bar)`), effectively double-applying the updates. We reproduced this script as-shipped (STEP 3, unmodified) and measured final accuracy 74.2% vs. the README's claimed ≈93.3% at the same `n_params=51618`; this bug is a plausible (though not certain, given the separate RNG non-determinism in §5.7) explanation for most of that gap. See `results/upstream_reproduction/REPRODUCTION_SUMMARY.md` for the full comparison. We have not patched this file — fixing it is out of scope for the gradient-defense benchmark per the task brief's instruction not to silently change upstream behavior.

## 6. Conceptual limitations of norm-only (ℓ2) attestation

These are properties of the *design*, not bugs — several are already
acknowledged by the authors (SECURITY.md's "sign-flip / in-bound backdoors"
row, the README's "crypto detection of sign-flip under ℓ2 alone is 0%"
line). We confirm them independently from the code/results and add detail.

1. **A norm bound cannot distinguish an in-bound malicious direction from an honest one.** Confirmed empirically in the repo's own `results/backdoor_results.json` / `experiments/run_backdoor.py`: FedAvg backdoor ASR ≈98%, ZKP-ℓ2-only does not meaningfully reduce it (by construction — a backdoor update sized within τ passes), coordinate-median brings it down to ≈55% (not eliminated). This is the authors' own, already-disclosed result; we are not claiming a new attack, just confirming the mechanism of failure is structural (any attack whose norm stays ≤ τ is invisible to a norm gate by definition, regardless of implementation quality).
2. **Dual-norm (ℓ2+ℓ∞) closes one specific sparse-spike construction, evaluated narrowly.** `experiments/run_innovation_pack.backdoor_linf_ablation` tests a single-coordinate spike (`poison[0] = 4.5`) against fixed `(τ2=5, τ∞=0.2)` on `d=256` synthetic honest noise. It does not evaluate: (a) a distributed sparse attack that spreads mass across many coordinates each individually under τ∞ while still exceeding τ∞ in aggregate effect, (b) non-IID real data where honest per-coordinate magnitudes vary more, or (c) an adversary that knows τ∞ and adapts. The result that dual-norm helps is real within its tested scope; generalizing it further is untested in this repo.
3. **The robust-aggregation/privacy conflict in §4 is itself a conceptual limitation of pairing norm-only crypto attestation with Byzantine-robust aggregation**, not merely an implementation slip: *any* aggregator that needs cross-client comparison (median, Krum, cosine/sign consensus — including the ones we are about to add in the new benchmark) needs per-client plaintext visibility, which is in tension with the "never decrypt an individual gradient" privacy framing used elsewhere in the paper. This motivates why our benchmark treats gradient-level defenses as operating at the same privacy tier as the existing median/Krum composition, not a strictly worse one.
4. **A single-round norm test has no memory.** `AdaptiveTau` adapts the *threshold* from the accepted-norm history, but there is no per-client reputation or cross-round correlation anywhere in the stack. An adversary that stays just under τ every round (rather than a one-shot large-norm spike) is invisible to the norm gate by construction, every round, independent of how τ is calibrated.
5. **Threshold tightness trades detection for false positives, and this trade-off is only characterized at one heterogeneity level.** `run_experiment.run_ablation_threshold` already shows FPR rising as τ shrinks, but only at `dirichlet_alpha=0.5` on synthetic data. How this trade-off shifts under more severe non-IID splits (lower α) or on real imaging data is exactly the sensitivity analysis the new benchmark (task brief §13) is designed to fill in — it is a known-open question, not an upstream defect.
