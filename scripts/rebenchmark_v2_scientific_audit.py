"""R12 scientific audit of results/rebenchmark_v2/full_benchmark/ (the 504-run
sweep) and results/rebenchmark_v2/ablation_adaptive_norm_only/ (the 54-run
ablation). Pure data analysis -- no GPU, no new training. Writes all
intermediate numbers to results/rebenchmark_v2/full_benchmark/analysis_artifacts/
so every number quoted in SCIENTIFIC_AUDIT.md / ANALYSIS.md / ABLATION_RESULTS.md
traces back to a file here, not to an assertion.

Usage: python3 scripts/rebenchmark_v2_scientific_audit.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FB_DIR = os.path.join(ROOT, "results", "rebenchmark_v2", "full_benchmark")
ABL_DIR = os.path.join(ROOT, "results", "rebenchmark_v2", "ablation_adaptive_norm_only")
OUT_DIR = os.path.join(FB_DIR, "analysis_artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

DEFENSES = ["fedavg", "norm", "cosine", "sign_consensus", "median", "multi_krum", "c4_drift_aware"]
ATTACKS = ["no_attack", "large_norm", "low_norm", "full_sign_flip", "directional_poisoning", "sparse_coordinate_attack"]
PARTITIONS = ["iid", "dirichlet_a1.0", "dirichlet_a0.5", "dirichlet_a0.1"]
SEEDS = [42, 43, 44]
DETECTOR_DEFENSES = {"norm", "cosine", "sign_consensus", "c4_drift_aware"}
N_ROUNDS = 280


def log(msg: str) -> None:
    print(msg)


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    manifest = pd.read_csv(os.path.join(FB_DIR, "run_manifest.csv"))
    rounds = pd.read_csv(os.path.join(FB_DIR, "per_round_results.csv"))
    clients = pd.read_csv(os.path.join(FB_DIR, "client_decisions.csv"))
    report: dict = {}

    # ---------------------------------------------------- 1. COMPLETENESS ---
    section("1. COMPLETENESS")
    expected_ids = {
        f"{d}__{a}__{p}__seed{s}"
        for d in DEFENSES for a in ATTACKS for p in PARTITIONS for s in SEEDS
    }
    actual_ids = set(manifest["run_id"])
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids
    dup_ids = manifest["run_id"].duplicated().sum()
    log(f"expected combos: {len(expected_ids)}; manifest rows: {len(manifest)}; "
        f"unique run_ids: {manifest['run_id'].nunique()}")
    log(f"missing combos: {len(missing)}; unexpected extra combos: {len(extra)}; duplicate run_ids: {dup_ids}")
    exp_rounds = len(manifest) * N_ROUNDS
    exp_clients = exp_rounds * 5
    log(f"per_round_results rows: {len(rounds)} (expected {exp_rounds}, match={len(rounds) == exp_rounds})")
    log(f"client_decisions rows: {len(clients)} (expected {exp_clients}, match={len(clients) == exp_clients})")
    rows_per_run = rounds.groupby("run_id").size()
    bad_round_counts = rows_per_run[rows_per_run != N_ROUNDS]
    log(f"runs with round count != {N_ROUNDS}: {len(bad_round_counts)}")
    report["completeness"] = {
        "expected_combos": len(expected_ids), "manifest_rows": len(manifest),
        "unique_run_ids": int(manifest["run_id"].nunique()), "missing": sorted(missing),
        "extra": sorted(extra), "duplicate_run_ids": int(dup_ids),
        "per_round_rows": len(rounds), "per_round_rows_expected": exp_rounds,
        "client_decision_rows": len(clients), "client_decision_rows_expected": exp_clients,
        "runs_with_wrong_round_count": int(len(bad_round_counts)),
    }

    # seed coverage per combo (every (defense, attack, partition) should have exactly 3 seeds)
    combo_seed_counts = manifest.groupby(["defense", "attack", "partition"])["seed"].nunique()
    bad_seed_coverage = combo_seed_counts[combo_seed_counts != 3]
    log(f"(defense, attack, partition) combos without exactly 3 seeds: {len(bad_seed_coverage)}")
    report["completeness"]["combos_with_wrong_seed_count"] = int(len(bad_seed_coverage))

    # -------------------------------------------------- 2. NaN / MISSING ---
    section("2. NaN / MISSING DATA")
    na_counts = manifest[["final_accuracy", "final_loss", "runtime_s", "tp", "fp", "tn", "fn", "tpr", "fpr"]].isna().sum()
    log(na_counts.to_string())
    # tp/fp/tn/fn/tpr/fpr should be NaN ONLY for non-detector defenses (fedavg/median/multi_krum)
    detector_rows = manifest[manifest["defense"].isin(DETECTOR_DEFENSES)]
    non_detector_rows = manifest[~manifest["defense"].isin(DETECTOR_DEFENSES)]
    detector_na_tpr = detector_rows["tpr"].isna().sum()
    nondetector_non_na_tpr = non_detector_rows["tpr"].notna().sum()
    log(f"detector-defense rows with NaN tpr (should be 0 or explainable): {detector_na_tpr}")
    log(f"non-detector-defense rows with non-NaN tpr (should be 0): {nondetector_non_na_tpr}")
    if detector_na_tpr > 0:
        bad = detector_rows[detector_rows["tpr"].isna()][["run_id", "tp", "fn"]]
        log("  rows: " + ", ".join(bad["run_id"].tolist()[:10]))
    report["nan_audit"] = {
        "na_counts": na_counts.to_dict(),
        "detector_rows_with_nan_tpr": int(detector_na_tpr),
        "nondetector_rows_with_non_nan_tpr": int(nondetector_non_na_tpr),
    }

    # ---------------------------------------- 3. RECOMPUTE TPR/FPR FROM RAW ---
    section("3. RECOMPUTE TPR/FPR/PRECISION FROM client_decisions.csv (cross-check vs manifest)")
    mism = []
    for _, row in manifest[manifest["defense"].isin(DETECTOR_DEFENSES)].iterrows():
        cd = clients[clients["run_id"] == row["run_id"]]
        cd = cd[cd["accepted"].notna()]
        tp = int(((cd["attack_active"]) & (~cd["accepted"].astype(bool))).sum())
        fn = int(((cd["attack_active"]) & (cd["accepted"].astype(bool))).sum())
        fp = int(((~cd["attack_active"]) & (~cd["accepted"].astype(bool))).sum())
        tn = int(((~cd["attack_active"]) & (cd["accepted"].astype(bool))).sum())
        recomputed_tpr = tp / (tp + fn) if (tp + fn) > 0 else None
        recomputed_fpr = fp / (fp + tn) if (fp + tn) > 0 else None
        stored_tpr = None if pd.isna(row["tpr"]) else float(row["tpr"])
        stored_fpr = None if pd.isna(row["fpr"]) else float(row["fpr"])
        ok_tp = tp == row["tp"]
        ok_fp = fp == row["fp"]
        ok_tn = tn == row["tn"]
        ok_fn = fn == row["fn"]
        if not (ok_tp and ok_fp and ok_tn and ok_fn):
            mism.append({"run_id": row["run_id"], "stored": (row["tp"], row["fp"], row["tn"], row["fn"]),
                         "recomputed": (tp, fp, tn, fn)})
    log(f"runs checked: {len(manifest[manifest['defense'].isin(DETECTOR_DEFENSES)])}")
    log(f"mismatches between stored tp/fp/tn/fn and recomputed from client_decisions.csv: {len(mism)}")
    for m in mism[:10]:
        log(f"  {m}")
    report["tpr_fpr_recompute_audit"] = {"n_checked": int(len(manifest[manifest['defense'].isin(DETECTOR_DEFENSES)])),
                                          "n_mismatches": len(mism), "examples": mism[:10]}

    # ---------------------------------------------- 4. DIVERGENCE AUDIT ---
    section("4. DIVERGENCE AUDIT")
    log(f"total diverged runs: {int(manifest['diverged'].sum())} / {len(manifest)}")
    by_def = manifest.groupby("defense")["diverged"].agg(["sum", "mean", "count"]).round(3)
    by_atk = manifest.groupby("attack")["diverged"].agg(["sum", "mean", "count"]).round(3)
    by_part = manifest.groupby("partition")["diverged"].agg(["sum", "mean", "count"]).round(3)
    log("-- by defense --\n" + by_def.to_string())
    log("-- by attack --\n" + by_atk.to_string())
    log("-- by partition --\n" + by_part.to_string())
    by_def.to_csv(os.path.join(OUT_DIR, "divergence_by_defense.csv"))
    by_atk.to_csv(os.path.join(OUT_DIR, "divergence_by_attack.csv"))
    by_part.to_csv(os.path.join(OUT_DIR, "divergence_by_partition.csv"))

    # check: does a diverged run's round-level loss actually exceed the divergence threshold (50.0) or is non-finite,
    # and does the per-round data correctly flatline after the divergence round (copy-forward semantics)?
    diverged_runs = manifest[manifest["diverged"]]["run_id"].tolist()
    flatline_violations = []
    threshold_violations = []
    for rid in diverged_runs:
        rr = rounds[rounds["run_id"] == rid].sort_values("round")
        div_rounds = rr[rr["diverged"]]
        if len(div_rounds) == 0:
            flatline_violations.append(rid)
            continue
        first_div_round = div_rounds["round"].min()
        tail = rr[rr["round"] >= first_div_round]
        if tail["accuracy"].nunique() > 1 or tail["loss"].nunique() > 1:
            flatline_violations.append(rid)
        first_loss = rr[rr["round"] == first_div_round]["loss"].iloc[0]
        if np.isfinite(first_loss) and first_loss <= 50.0:
            threshold_violations.append((rid, float(first_loss)))
    log(f"diverged runs whose post-divergence accuracy/loss is NOT flat (violates copy-forward semantics): "
        f"{len(flatline_violations)}")
    log(f"diverged runs whose divergence round does NOT satisfy loss>50 or non-finite: {len(threshold_violations)}")
    if threshold_violations:
        log("  " + str(threshold_violations[:10]))
    report["divergence_audit"] = {
        "total_diverged": int(manifest["diverged"].sum()), "total_runs": len(manifest),
        "flatline_violations": flatline_violations, "threshold_violations": threshold_violations,
    }

    # does including diverged runs' final_accuracy change the headline defense-level means much?
    log("\n-- effect of excluding diverged runs from defense-level mean accuracy --")
    incl = manifest.groupby("defense")["final_accuracy"].mean()
    excl = manifest[~manifest["diverged"]].groupby("defense")["final_accuracy"].mean()
    cmp_df = pd.DataFrame({"acc_incl_diverged": incl, "acc_excl_diverged": excl,
                           "n_diverged": manifest.groupby("defense")["diverged"].sum()}).round(3)
    log(cmp_df.to_string())
    cmp_df.to_csv(os.path.join(OUT_DIR, "divergence_inclusion_effect.csv"))

    # --------------------------------------- 5. METHOD x ATTACK x PARTITION ---
    section("5. METHOD x ATTACK, METHOD x PARTITION TABLES")
    ma = manifest.groupby(["defense", "attack"]).agg(
        acc_mean=("final_accuracy", "mean"), acc_std=("final_accuracy", "std"),
        tpr_mean=("tpr", "mean"), fpr_mean=("fpr", "mean"), diverge_rate=("diverged", "mean"),
        n=("run_id", "count"),
    ).round(4)
    mp = manifest.groupby(["defense", "partition"]).agg(
        acc_mean=("final_accuracy", "mean"), acc_std=("final_accuracy", "std"),
        tpr_mean=("tpr", "mean"), fpr_mean=("fpr", "mean"), diverge_rate=("diverged", "mean"),
        n=("run_id", "count"),
    ).round(4)
    ma.to_csv(os.path.join(OUT_DIR, "method_x_attack.csv"))
    mp.to_csv(os.path.join(OUT_DIR, "method_x_partition.csv"))
    log("method x attack -> " + os.path.join(OUT_DIR, "method_x_attack.csv"))
    log("method x partition -> " + os.path.join(OUT_DIR, "method_x_partition.csv"))

    # full 4-way table (defense, attack, partition) with per-seed values spelled out for paired comparison
    full = manifest.pivot_table(index=["attack", "partition", "defense"], columns="seed",
                                 values="final_accuracy").reset_index()
    full.columns = [str(c) for c in full.columns]
    full.to_csv(os.path.join(OUT_DIR, "method_x_attack_x_partition_x_seed_accuracy.csv"), index=False)
    log("per-seed accuracy pivot -> method_x_attack_x_partition_x_seed_accuracy.csv")

    # --------------------------------------------- 6. PAIRED SEED COMPARISON ---
    section("6. PAIRED COMPARISON: c4_drift_aware vs multi_krum vs median (same seed, same condition)")
    pivot = manifest.pivot_table(index=["attack", "partition", "seed"], columns="defense", values="final_accuracy")
    paired_rows = []
    for (atk, part, seed), row in pivot.iterrows():
        if pd.notna(row.get("c4_drift_aware")) and pd.notna(row.get("multi_krum")):
            paired_rows.append({"attack": atk, "partition": part, "seed": seed,
                                 "c4_drift_aware": row["c4_drift_aware"], "multi_krum": row["multi_krum"],
                                 "c4_minus_mkrum": row["c4_drift_aware"] - row["multi_krum"]})
    paired_df = pd.DataFrame(paired_rows)
    n_c4_better = (paired_df["c4_minus_mkrum"] > 0).sum()
    n_mkrum_better = (paired_df["c4_minus_mkrum"] < 0).sum()
    log(f"c4_drift_aware vs multi_krum, paired by (attack,partition,seed): n={len(paired_df)}, "
        f"c4_better={n_c4_better}, multi_krum_better={n_mkrum_better}, "
        f"mean_diff={paired_df['c4_minus_mkrum'].mean():.4f}, std_diff={paired_df['c4_minus_mkrum'].std():.4f}")
    paired_df.to_csv(os.path.join(OUT_DIR, "paired_c4_vs_multikrum.csv"), index=False)
    report["paired_c4_vs_multikrum"] = {
        "n": len(paired_df), "c4_better": int(n_c4_better), "multi_krum_better": int(n_mkrum_better),
        "mean_diff": float(paired_df["c4_minus_mkrum"].mean()), "std_diff": float(paired_df["c4_minus_mkrum"].std()),
    }

    # -------------------------------------------- 7. NO-ATTACK CONVERGENCE ---
    section("7. NO-ATTACK CONVERGENCE AT 280 ROUNDS (per defense x partition, seed-averaged curve)")
    no_atk_runs = manifest[manifest["attack"] == "no_attack"]["run_id"]
    na_rounds = rounds[rounds["run_id"].isin(no_atk_runs)]
    curve = na_rounds.groupby(["defense", "partition", "round"])["accuracy"].mean().reset_index()
    conv_rows = []
    for (d, p), g in curve.groupby(["defense", "partition"]):
        g = g.sort_values("round")
        acc = g["accuracy"].to_numpy()
        last30_slope = np.polyfit(np.arange(30), acc[-30:], 1)[0]
        conv_rows.append({"defense": d, "partition": p, "acc_round50": acc[50] if len(acc) > 50 else None,
                           "acc_round150": acc[150] if len(acc) > 150 else None, "acc_round279": acc[-1],
                           "last30_round_slope": last30_slope})
    conv_df = pd.DataFrame(conv_rows).round(5)
    conv_df.to_csv(os.path.join(OUT_DIR, "no_attack_convergence_check.csv"), index=False)
    log(conv_df.to_string())
    curve.to_csv(os.path.join(OUT_DIR, "no_attack_accuracy_curves.csv"), index=False)

    # ----------------------------------------------- 8. FREEZE DETECTION ---
    section("8. FREEZE DETECTION (last 30 rounds: n_accepted==0 for ALL rounds, among detector defenses)")
    det_clients = clients[clients["defense"].isin(DETECTOR_DEFENSES)]
    accepted_per_round = det_clients.groupby(["run_id", "round"])["accepted"].sum().reset_index()
    freeze_rows = []
    for rid, g in accepted_per_round.groupby("run_id"):
        g = g.sort_values("round")
        tail = g[g["round"] >= g["round"].max() - 29]
        is_frozen = bool((tail["accepted"] == 0).all())
        freeze_rows.append({"run_id": rid, "frozen_last30": is_frozen})
    freeze_df = pd.DataFrame(freeze_rows).merge(manifest[["run_id", "defense", "attack", "partition", "seed", "diverged"]], on="run_id")
    freeze_df.to_csv(os.path.join(OUT_DIR, "freeze_detection.csv"), index=False)
    log(f"total detector runs: {len(freeze_df)}; frozen_last30=True: {freeze_df['frozen_last30'].sum()}")
    log("-- frozen count by (defense, partition) --")
    fbp = freeze_df.groupby(["defense", "partition"])["frozen_last30"].sum()
    log(fbp[fbp > 0].to_string())
    log("-- freeze vs diverged overlap (should be mostly disjoint: freeze=stable-but-stuck, diverge=numerically blown up) --")
    overlap = pd.crosstab(freeze_df["frozen_last30"], freeze_df["diverged"])
    log(overlap.to_string())
    report["freeze_audit"] = {
        "total_detector_runs": len(freeze_df), "frozen_last30_count": int(freeze_df["frozen_last30"].sum()),
        "freeze_diverge_crosstab": overlap.to_dict(),
    }

    # ----------------------------------------- 9. ABLATION PILOT CROSS-CHECK ---
    section("9. ABLATION PILOT (results/rebenchmark_v2/ablation_adaptive_norm_only/) INTEGRITY")
    if os.path.exists(os.path.join(ABL_DIR, "run_manifest.csv")):
        abl = pd.read_csv(os.path.join(ABL_DIR, "run_manifest.csv"))
        exp_abl = {f"{d}__{a}__{p}__seed{s}"
                   for d in ["norm", "adaptive_norm_only", "c4_drift_aware"]
                   for a in ["no_attack", "large_norm", "directional_poisoning"]
                   for p in ["iid", "dirichlet_a0.5"] for s in SEEDS}
        log(f"ablation expected combos: {len(exp_abl)}; actual rows: {len(abl)}; "
            f"missing: {len(exp_abl - set(abl['run_id']))}")
        # cross-check: does the ablation's norm/no_attack/iid result match the main benchmark's?
        main_match = manifest[(manifest.defense == "norm") & (manifest.attack == "no_attack") & (manifest.partition == "iid")]
        abl_match = abl[(abl.defense == "norm") & (abl.attack == "no_attack") & (abl.partition == "iid")]
        cross = main_match.merge(abl_match, on="seed", suffixes=("_main", "_ablation"))[
            ["seed", "final_accuracy_main", "final_accuracy_ablation"]]
        cross["abs_diff"] = (cross["final_accuracy_main"] - cross["final_accuracy_ablation"]).abs()
        log("cross-check norm/no_attack/iid, main benchmark vs ablation pilot (independent reruns, same seeds):")
        log(cross.to_string())
        cross.to_csv(os.path.join(OUT_DIR, "ablation_vs_main_crosscheck.csv"), index=False)
        report["ablation_integrity"] = {
            "expected_combos": len(exp_abl), "actual_rows": len(abl),
            "missing": len(exp_abl - set(abl["run_id"])),
            "crosscheck_max_abs_diff": float(cross["abs_diff"].max()),
        }
    else:
        log("ABLATION MANIFEST NOT FOUND")

    with open(os.path.join(OUT_DIR, "audit_summary.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"\nWrote {os.path.join(OUT_DIR, 'audit_summary.json')}")


if __name__ == "__main__":
    main()
