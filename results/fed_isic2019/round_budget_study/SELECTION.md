# Fed-ISIC2019 round budget — rationale (locked)

Study: `scripts/fedisic_round_budget_study.py`. Clean FedAvg only, no
attacks, no detector calibration, fixed K=10 SGD steps/client/round
(justified in `docs/BENCHMARK_PROTOCOL.md` Addendum (j) — EfficientNet-b0
measured at ~33x PathMNIST's per-step cost, so K=50 was never attempted),
batch_size=32, 6 natural FLamby centers (no partition sweep — natural
federation is the one condition under test), 3 seeds, out to 15 rounds.
Raw curves: `curves.csv`; plot: `plot.png`.

## Per-round accuracy (mean over 3 seeds, pooled test set)

| round | mean acc | std across seeds |
|---|---|---|
| 0 | 0.290 | 0.014 |
| 5 | 0.449 | 0.009 |
| 8 | 0.465 | 0.006 |
| 10 | 0.471 | 0.004 |
| 11 | 0.473 | 0.003 |
| 12 | 0.474 | 0.002 |
| 14 | 0.476 | 0.004 |

## Reading

Convergence is **much faster and cleaner than PathMNIST's** (which needed
25 rounds and still had a non-plateaued condition at that point) — the
expected effect of starting from ImageNet-pretrained EfficientNet-b0
features rather than training a CNN from scratch. The curve has a clear
knee around round 8–10 and is essentially flat from round 10 onward: only
+0.5 percentage points of mean accuracy gain across rounds 10→14 (4 more
rounds), while cross-seed std keeps shrinking (0.004→0.002 at round 12,
the single lowest value observed) — both the accuracy plateau and the
variance minimum point to round 12 as the right stopping point, not an
arbitrary round-count pick.

**N_ROUNDS = 12** (captures the plateau with one round of margin past the
std minimum at round 12 itself; rounds 13–14 were measured and add only
+0.2pp — not worth the extra ~38s/round × however many runs the main
benchmark needs).

**ATTACK_FROM_ROUND = 5**: at round 5 the model has already reached
0.449/0.476 ≈ 94% of its eventual round-14 accuracy — clearly
"post-initial-learning," matching PathMNIST's own design principle (attack
a model that has already learned something substantive, not a near-random
one) without needlessly delaying attack exposure on a dataset that
converges this much faster. 7 of 12 rounds (58%) are attack-exposed,
comparable to PathMNIST's 15/25 (60%).

## Cost reality (for scoping the main benchmark — flagged, not yet decided)

Measured: **~38.5s/round** for 6 clients × K=10 steps (clean FedAvg, no
server-ref/g_ref training, no attack/defense overhead). This is **~18x**
PathMNIST's measured ~2.1s/round. One full N_ROUNDS=12 run ≈ 7.7 minutes
*before* adding Cosine's extra server-ref training step or any
defense/attack-specific overhead. PathMNIST's 594-combination original
sweep or even the 144-run Combined minimal benchmark are **not** directly
affordable at this per-run cost without a much smaller matrix. This
reality is relayed to ChatGPT before committing to any specific
defense x attack x seed matrix for the main Fed-ISIC2019 benchmark.
