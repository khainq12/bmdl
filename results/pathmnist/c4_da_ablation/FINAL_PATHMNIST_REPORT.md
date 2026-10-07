# Final PathMNIST Evaluation — End-to-End Ablation of Locked C4-DA v1

Per ChatGPT's agreed roadmap after locking `C4-DA v1`
(`results/pathmnist/LOCKED_DESIGN_C4-DA-v1.md`): real end-to-end ablation
(not offline replay) of each signal, answering whether C4-DA protects
better than individual detectors, whether each signal is actually
necessary, and how it trades off against Median/Multi-Krum.

**216 new runs**: `c4_drift_aware_no_norm`, `c4_drift_aware_no_cosine`,
`c4_drift_aware_no_sign` × 4 partitions × 3 seeds (42–44) × 6 attacks. The
full `C4-DA` baseline at the same seeds/partitions/attacks is **reused, not
rerun**, from `c4_drift_aware_confirmation/` (which already covers this
exact grid). `benchmark/defenses/combined.py::C4DriftAware` was **not
modified** — three new structurally-ablated variants were added
alongside it (`C4DriftAwareNoNorm/NoCosine/NoSign`), each genuinely never
computing the removed signal (not post-hoc masking).

## Headline ablation table

| removed signal | large_norm TPR | directional_poisoning TPR | full_sign_flip TPR | sparse_coordinate TPR | honest FPR (no_attack) |
|---|---|---|---|---|---|
| (none — full C4-DA) | 1.000 | 0.900 | 0.900 | 1.000 | 0.041 |
| **Norm** | **0.250** | 0.878 | 0.883 | 1.000 | 0.009 |
| **Cosine** | 1.000 | **0.544** | **0.544** | 1.000 | 0.040 |
| **Sign** | 1.000 | 0.900 | 0.900 | 1.000 | 0.043 |

## 1. Removing Norm — catastrophic, Non-IID-dependent

`large_norm` TPR collapses 1.000→0.250; **9/12 `large_norm` runs
numerically diverge** (`run_manifest.csv`). The precise pattern is itself
informative: **divergence occurs in all 9 Non-IID runs (α=1.0, α=0.5,
α=0.1 × 3 seeds each) but in 0/3 IID runs.** This exactly reproduces, in
the real ablated end-to-end loop, the single-signal Cosine finding from
`analysis_v2` (Cosine alone catches `large_norm` at IID but goes blind
under any Non-IID) — with Norm removed, C4-DA's residual Cosine+Sign stage
inherits exactly that IID-only blind spot. **Directional/sparse TPR are
only mildly affected** (0.900→0.878, 0.900→0.883, 1.000→1.000) since
Cosine/Sign are untouched. **Honest FPR actually improves** (0.041→0.009)
— Norm is, as established throughout this entire research arc, the
noisiest signal; removing it costs nothing on the honest side but costs
everything on `large_norm` safety under Non-IID. **Norm is necessary,
specifically for Non-IID large_norm safety** — not a redundant or
decorative component.

## 2. Removing Cosine — large, clean, exactly as predicted

`directional_poisoning`/`full_sign_flip` TPR both collapse 0.900→0.544 —
the single largest effect of any ablation, affecting only the two attacks
Cosine is specifically responsible for. `large_norm`/`sparse_coordinate`
TPR are completely unaffected (1.000→1.000 both). Honest FPR is
essentially unchanged (0.041→0.040). **Cosine is necessary, specifically
and exclusively for directional-attack detection** — the cleanest,
most textbook-exact ablation result of the three, with zero side effects
on any other axis.

## 3. Removing Sign — no measurable effect on anything

Every TPR value is identical or within noise to the full C4-DA baseline
(large_norm 1.000=1.000, directional 0.900=0.900, full_sign_flip
0.900=0.900, sparse_coordinate 1.000=1.000), and honest FPR is unchanged
(0.041→0.043, within seed-to-seed noise). **OBSERVED, not assumed**: this
directly confirms, now in a real end-to-end training loop rather than
offline replay, the finding first surfaced in `combined_e2e_minimal/REPORT.md`
§11 (ablation on C0): Sign's contribution to `sparse_coordinate_attack`
detection is **fully subsumed by Norm** — Norm alone already achieves
perfect TPR on this attack (its spike magnitude also exceeds τ), so Sign's
peer-rank mechanism never gets a chance to matter. **This is an honest,
somewhat unflattering finding for the locked design, reported as found**:
Sign is not harmful (no honest-side cost either), but its presence in
C4-DA v1 is not shown to be *necessary* against any attack in this suite.
It remains in the locked design (per the "do not redesign based on
results" rule) and is reported transparently as a limitation/open
question — a future attack that defeats Norm's magnitude check while
still being sparse/coordinate-local (unlike `sparse_coordinate_attack` as
implemented here, whose spike also happens to inflate the overall norm)
could be the scenario where Sign's removal would actually matter; this
benchmark does not test that scenario.

## Answering ChatGPT's three final questions

**(1) Does C4-DA protect the model better than each individual detector,
end-to-end?** Yes, on `large_norm` specifically (TPR 1.000 vs single-Cosine's
1.000-at-IID-only/0-elsewhere, single-Sign's 0 everywhere — see
`analysis_v2`) and matches-or-exceeds single-Cosine's own TPR on
directional attacks (0.900 vs 0.889) despite also carrying magnitude and
sparse protection single-Cosine lacks entirely. It is not uniformly "better
than every detector on every axis" — it is the only one of the four
(three single signals + C4-DA) that is simultaneously non-catastrophic on
`large_norm`, strong on directional attacks, and perfect on
`sparse_coordinate_attack`.

**(2) Is each of Norm/Cosine/Sign actually necessary, via end-to-end
ablation?** **Norm: yes** (necessary for Non-IID large_norm safety —
removal causes 9/12 divergences). **Cosine: yes** (necessary for
directional-attack detection — removal halves TPR on exactly those two
attacks). **Sign: not shown to be necessary in this attack suite** —
removal changes no measured outcome. This is reported honestly rather than
forcing a "yes, all three are needed" narrative the data doesn't support.

**(3) How does C4-DA trade off utility, detection capability, and overhead
against Median/Multi-Krum?** From `combined_e2e_minimal/BASELINE_COMPARISON.md`
(reused, not rerun here): C4-DA's mean accuracy (0.381) is within 0.5
percentage points of Multi-Krum's (0.386, the best single existing
baseline) and exceeds Median (0.312), while additionally providing
explicit per-client, per-signal accept/reject reasoning that neither
Median nor Multi-Krum provide. Runtime overhead is negligible (<1s/run
difference across all Combined variants, from both this and the prior
stage). The honest framing (per ChatGPT's own correction) is: **comparable
utility and safety to the best existing robust aggregator, with added
interpretability/attestation value — not a demonstrated accuracy
advantage.**

## What this does NOT establish (explicit limitation)

This ablation used the same 6-attack suite as every prior stage. It cannot
rule out an attack specifically designed to defeat Norm+Cosine while
remaining undetectable to a peer-rank Sign check — `sparse_coordinate_attack`
as implemented here happens to also trip the Norm gate, which is why Sign
appeared redundant. A stronger, more targeted sparse/coordinate attack
that keeps its norm inside τ is untested and would be the natural
follow-up for probing whether Sign's apparent redundancy is a suite
artifact or a genuine design property.

## Status

Final PathMNIST stage complete. `C4-DA v1` remains locked and unmodified.
Per the agreed roadmap, next stage is Fed-ISIC2019 validation using this
exact locked version.
