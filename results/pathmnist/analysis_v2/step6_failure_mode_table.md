# STEP 6 — Failure-mode table (evidence-based)

Populated only from STEP 2/3/4/5/7's observed results. No method is called
"best," "fail-safe," or "universally robust" — strengths and failure modes
are listed per attack/condition, because every method examined has both.

| Method | Strength (observed) | Failure mode (observed) | Sensitive to Non-IID? | Catastrophic miss possible? | Attacks it handles well |
|---|---|---|---|---|---|
| **FedAvg** | None — reference baseline, zero filtering by design | Accepts every update unconditionally | Utility degrades with heterogeneity like every method, but no attack-specific sensitivity (nothing is filtered either way) | **Yes — always**, under `large_norm` (100% divergence, every partition, STEP 2/5) | None (not a defense) |
| **Norm** | `large_norm`: TPR=1.0 at **every** partition. `sparse_coordinate_attack`: TPR=1.0 at every partition too (STEP 6 TPR table) | `low_norm`: TPR=0.0 everywhere — structurally undetectable by a magnitude-only rule, since the attack scales magnitude *down*. Directional attacks (`directional_poisoning`/`full_sign_flip`): partial, degrading TPR (0.71 IID → 0.47 at α=0.1) — not reliable. **Elevated honest-client FPR everywhere, worst at IID (0.63)**, caused by temporal drift (STEP 4: honest score follows a U-shaped trajectory a short calibration window cannot see), not by Non-IID itself (FPR is *lower* at severe Non-IID than at IID for Norm, the reverse of every other defense) | Yes, but counterintuitively — FPR *decreases* with heterogeneity due to a different mechanism (initial cluster separation, STEP 4), while directional-attack TPR *decreases* with heterogeneity normally | **No** — zero divergences anywhere in 594 combos (STEP 1/2) | `large_norm`, `sparse_coordinate_attack` |
| **Cosine** | `directional_poisoning`/`full_sign_flip`: strong and comparatively Non-IID-tolerant (TPR 1.00→0.76 from IID to α=0.1 — the smallest degradation of any detector for these two attacks). `large_norm` at IID: TPR=1.0, FPR=0.0 | `large_norm` and `sparse_coordinate_attack` outside IID: **TPR=0.0** — the attack's near-zero cosine score (large_norm: isotropic noise; sparse: a few spiked coordinates barely move the aggregate direction) falls inside the Non-IID-widened honest score range (STEP 4 §Cosine, STEP 7: Cosine contributes to **zero** sparse-attack rejections). Depends on `g_ref` every round (extra ~12-17% runtime overhead, STEP 2) | Yes, but attack-specific: directional-attack TPR degrades gracefully; `large_norm`/`sparse` TPR collapses to 0 outright outside IID | **Yes, outside IID** — 100% divergence under `large_norm` at every Non-IID partition when it's blind to it (STEP 5); never diverges at IID or under attacks it actually detects | `directional_poisoning`, `full_sign_flip` (robustly); `large_norm`, `sparse_coordinate_attack` (IID only) |
| **Sign Consensus** | `sparse_coordinate_attack`: **TPR=1.0 at every partition** — the most Non-IID-robust detector for this one attack (STEP 6 TPR table, STEP 7: necessary alongside Norm). Near-zero honest FPR under `no_attack` at every partition in Regime A (STEP 3) | `large_norm`: **TPR=0.0 at every partition including IID** — its single worst result of any defense/attack/partition cell in this study; STEP 4 showed the malicious score *exceeds* the honest range entirely (not overlap — inversion), and STEP 7 confirmed Sign contributes to zero `large_norm` rejections. Directional attacks collapse steeply with heterogeneity (TPR 1.00 IID → 0.00 by α=0.5) — the *steepest* degradation of any detector for these attacks, worse than Cosine's | Yes, severely for directional attacks (steeper drop than Cosine); not for `sparse_coordinate_attack` (flat TPR=1.0) | **Yes — at every partition**, including IID, under `large_norm` (the only detector that fails even at IID) | `sparse_coordinate_attack` (robustly, uniquely so) |
| **Median** | Never diverges under any attack/partition (STEP 1/2). Near-zero accuracy degradation vs. `no_attack` across all 5 attacks and all 4 partitions (mostly within ±0.01-0.15, no consistent direction — STEP 6) | Not a detector — no TPR/FPR, no attacker accountability, no rejected-client count. Provides no information about *who* was malicious | Utility degrades with heterogeneity like every method, but attack-robustness itself does not visibly worsen with heterogeneity | **No** | All 5 attacks, uniformly (graceful, not selective) |
| **Multi-Krum** | Same divergence-free, low-degradation profile as Median, slightly better floor under `large_norm` specifically at α=0.1 (0.299 vs Median's 0.120, STEP 2) | Same lack of detector semantics as Median | Same as Median | **No** | All 5 attacks, uniformly (graceful, not selective), best floor under `large_norm` of the two robust aggregators |

## Cross-cutting observations (not specific to one method)

- **No detector is Non-IID-robust across every attack it is nominally
  good at.** Norm's FPR improves with heterogeneity (unusual) while its
  directional-attack TPR worsens (usual). Cosine's directional-attack TPR
  degrades gently; its `large_norm`/sparse TPR collapses outright. Sign's
  sparse-attack TPR is flat; its directional-attack TPR collapses
  steepest of all three.
- **Every detector has exactly one attack where it is uninvolved in
  rejection at all** (STEP 7): Sign for `large_norm`, Cosine for
  `sparse_coordinate_attack`, and (within the directional-attack pair)
  Norm is never independently sufficient without Cosine.
- **Catastrophic divergence is not spread evenly**: Norm (0 divergences),
  Cosine (divergent only outside IID, only under `large_norm`/rare sparse
  cases), Sign (divergent at every partition under `large_norm`, plus rare
  directional cases) — the two aggregators (Median/Multi-Krum) never
  diverge at all, which is itself the most consistent evidence in this
  study that structural-robustness (hard caps, outlier resistance)
  outperforms score-threshold detection on worst-case safety, even though
  the detectors can out-perform them on ordinary-case utility and do
  provide attacker accountability the aggregators cannot.
