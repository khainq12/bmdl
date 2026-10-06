# Upstream Reproduction Summary

All runs below used the **unmodified** scripts in `experiments/` and `formal/`
as shipped at commit `e6a6d5a` (HEAD of `main` at clone time), with their
shipped default `CONFIG` — no parameters were changed, and none were tuned
against any test set. Raw stdout and JSON outputs for each run are in this
directory (`run_*.stdout.log`, `*_results.json`).

**Environment:** Python 3.10.12, numpy (per `pip show`), scipy 1.15.3,
scikit-learn 1.7.2, cryptography 3.4.8, matplotlib 3.10.9. `medmnist==3.0.2`
and `tenseal==0.3.16` were installed partway through this session (see
per-row notes below for which runs predate that). GPU present (GTX 1660
SUPER) but unused — none of `crypto/`/`fl_core/` have a GPU path.

**Standing caveat (applies to every row):** `local_training()` (shared by
every script) shuffles minibatches with an unseeded
`np.random.default_rng()`, and `run_experiment.py`'s malicious-delta
injection draws from the unseeded global `np.random.normal`. Neither is
derived from `CONFIG['seed']`. Re-running any of these scripts will **not**
reproduce bit-identical numbers; only the qualitative pattern is expected to
replicate. Full detail in `docs/IMPLEMENTATION_PLAN.md` §3.9 and
`docs/UPSTREAM_SECURITY_FINDINGS.md` §5.7.

| Claim (README.md) | Claimed | Reproduced (this run) | Match? |
|---|---|---|---|
| UCI innovation pack, 3 rounds | 89.5% → 93.0% → 92.1%; τ 8→2 | 89.47% → 92.98% → 92.98%; τ 8.0→4.31→2.0 (final 2.0) | **Close** — same qualitative path, small numeric drift consistent with the RNG caveat above. |
| Dual-norm sparse poison | Reject 100% vs ℓ2-only 0%; ~265× lower coord spike | dual reject=100%, ℓ2-only reject=0%; **265.18×** reduction | **Match**, essentially exact. |
| Tampered partial μ_i | Rejected by PartialDecrypt NIZK | Rejected (`tampered_partial_rejected=True`) | **Match.** |
| Wall-clock, UCI pack | ≈20–23 s/round on laptop CPU | ≈8.0–8.3 s/round | **Faster than claimed**, not a failure to reproduce — this machine's CPU/thread count differs from whatever "laptop CPU" the paper was benchmarked on (`GradientHEManager` uses an 8-worker thread pool; core count matters a lot here). |
| Synthetic Hybrid+median, 5 seeds | large-norm & sign-flip final acc 100%; crypto det. large-norm ≈97%, sign-flip 0% | large_norm acc=100.0±0.0%, det=97.1%; sign_flip acc=100.0±0.0%, det=0.0% | **Match**, essentially exact. |
| UCI legacy target (`run_target_protocol.py`) | 88.6% → 92.1% → 92.1%; **fused** HE + median + Unruh r=64 | 89.47% → 92.98% → 92.98%; **numpy**-only backend (ran before `tenseal` was installed this session) + median + Unruh r=64 | **Close, with a known backend difference** — our run used the numpy-threshold-only HE path, not the fused SEAL sidecar, because `tenseal` wasn't yet installed when this script ran. Numbers are close but this run should not be read as validating the fused path specifically. |
| Backdoor ASR | FedAvg/ℓ2 ≈98% → ZKP+median ≈55% (clean ≈99%); Multi-Krum ≈46% | fedavg ASR=97.9% (clean=99.2%); zkp_l2 ASR=97.7%; krum ASR=47.7% (clean=99.1%); hybrid_zkp_median ASR=54.7% (clean=99.1%) | **Match**, within a point or two on every arm. |
| ConvNet28 PneumoniaMNIST smoke | ≈93.3%; d≈51618; Unruh r=4 | **74.2%**; d=51618 (exact); r=4 | **Could not reproduce the accuracy claim.** Parameter count matches exactly; final accuracy does not. See "Not reproduced" below — we traced this to a likely script-specific bug, not just RNG variance. |
| Scale study (N=20,T=30) | *(no single headline number in README; sanity-checked against `run_scale.py`'s own internal "closes A5 scale residual" note)* | fedavg/large_norm acc=27.1%; clip/krum/zkp/hybrid all acc=100% det=100% on large_norm; on sign_flip only krum reaches 100% (others 76–80%, consistent with ℓ2-bound attacks evading norm-only detection) | **Consistent with the repo's own stated purpose** (demonstrate robustness at larger N,T; sign-flip evading norm-only defenses is the expected, documented failure mode). |
| Formal CI (`formal/run_formal_ci.py`) | `FORMAL_CI_OK=1` | `FORMAL_CI_OK=1`; `lake`/`easycrypt` binaries confirmed **not installed** (both checks skipped, as the script itself reports) | **Match** — and confirms, independently, that the "formal" layer as actually exercised here is the Python-mechanized combinatorics + static `.ec`/`.lean` source-text presence checks described in `docs/IMPLEMENTATION_PLAN.md` §1.3, not a discharged EasyCrypt/Lean proof. |

## What could not be reproduced exactly

**ConvNet28 / PneumoniaMNIST accuracy (93.3% claimed vs. 74.2% reproduced).**
The parameter count (`d=51618`) and Unruh repetition count (`r=4`) match the
claim exactly, so this is the same model/config, not a different experiment.
While we cannot rule out that the README's 93.3% came from a differently
seeded run under the same non-deterministic RNG behavior described above,
reading `experiments/run_medmnist_cnn.py` turned up a concrete,
script-specific bug that is a more likely explanation: the round loop never
resets the model to the round's global weights before each client's local
training (every other full-protocol script in the repo does this reset;
this one script does not — see `docs/UPSTREAM_SECURITY_FINDINGS.md` §5.8 for
the exact code and mechanism). This makes client training cascade
sequentially within a round and then double-applies the aggregated update on
top. We have not patched this to see what the "corrected" accuracy would be,
per the instruction to reproduce upstream as-is rather than silently fix it;
we are reporting the discrepancy honestly rather than claiming successful
reproduction of this one number.

## What this reproduction run deliberately did not attempt

- `run_medmnist.py` and `run_medmnist_fullres.py` (the two other MedMNIST
  variants) were not separately reproduced in this pass — `run_medmnist_cnn.py`
  already exercises the full crypto stack on real MedMNIST data at the
  largest parameter count of the three, and was prioritized given time
  constraints. Can be run on request with the same `python experiments/run_medmnist.py` /
  `run_medmnist_fullres.py` commands.
- `experiments/plot_figures.py` / `make_excellence_figure.py` were not run —
  they regenerate figures from the `results/*.json` already reproduced here
  and don't add new numeric claims to verify.
