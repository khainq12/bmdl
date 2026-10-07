"""Combined Design + Offline Replay (results/pathmnist/combined_design_v1/).

Offline-only: replays candidate Combined Attestation rules C0-C6 against the
existing 9,000-row Trace B (results/pathmnist/analysis_v2/trace_B_complementarity_pooled3seeds.csv).
No new FL training. No full-matrix rerun. See CANDIDATE_DEFINITIONS.md for the
locked rule definitions this script implements verbatim.

Usage: python scripts/combined_design_offline_replay.py
Writes CSVs into results/pathmnist/combined_design_v1/.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_PATH = os.path.join(ROOT, "results", "pathmnist", "analysis_v2", "trace_B_complementarity_pooled3seeds.csv")
CALIB_PATH = os.path.join(ROOT, "results", "pathmnist", "benchmark_v2_corrected", "calibration_results.csv")
OUT_DIR = os.path.join(ROOT, "results", "pathmnist", "combined_design_v1")

ATTACK_FROM_ROUND = 10
MODIFIED_Z_CUTOFF = 3.5  # Iglewicz & Hoaglin (1993) published convention
MAD_TO_STD = 1.4826

CANDIDATES = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
SIGNALS = ["norm", "cos", "sign"]


def load_trace() -> pd.DataFrame:
    df = pd.read_csv(TRACE_PATH)
    calib = pd.read_csv(CALIB_PATH)
    calib_A = calib[calib["calibration_regime"] == "A_condition_specific_oracle"]
    calib_B = calib[calib["calibration_regime"] == "B_iid_transfer_frozen"]

    def th(table, defense, partition):
        row = table[(table["defense"] == defense) & (table["partition"] == partition)]
        assert len(row) == 1
        return float(row.iloc[0]["threshold"])

    tau_B = th(calib_B, "norm", "iid_reference_frozen")
    rho_B = th(calib_B, "cosine", "iid_reference_frozen")
    kappa_B = th(calib_B, "sign_consensus", "iid_reference_frozen")

    tau_A, rho_A, kappa_A = {}, {}, {}
    for p in df["partition"].unique():
        tau_A[p] = th(calib_A, "norm", p)
        rho_A[p] = th(calib_A, "cosine", p)
        kappa_A[p] = th(calib_A, "sign_consensus", p)

    df["tau_A"] = df["partition"].map(tau_A)
    df["rho_A"] = df["partition"].map(rho_A)
    df["kappa_A"] = df["partition"].map(kappa_A)
    df["tau_B"] = tau_B
    df["rho_B"] = rho_B
    df["kappa_B"] = kappa_B

    # corrected honest / malicious_active flags (see TRACE_AUDIT.md)
    df["honest"] = (~df["is_malicious"]) | (df["attack"] == "no_attack")
    df["malicious_active"] = df["is_malicious"] & (df["round"] >= ATTACK_FROM_ROUND) & (df["attack"] != "no_attack")

    # base fail flags, Regime A (verified == stored accept_*_A columns in TRACE_AUDIT.md)
    df["norm_fail_A"] = df["norm_score"] > df["tau_A"]
    df["cos_fail_A"] = df["cosine_score"] < df["rho_A"]
    df["sign_fail_A"] = df["sign_score"] < df["kappa_A"]

    # Regime B fail flags (non-IID rows only; NaN on iid rows, left as NaN)
    is_non_iid = df["partition"] != "iid"
    df["norm_fail_B"] = np.where(is_non_iid, df["norm_score"] > df["tau_B"], np.nan)
    df["cos_fail_B"] = np.where(is_non_iid, df["cosine_score"] < df["rho_B"], np.nan)
    df["sign_fail_B"] = np.where(is_non_iid, df["sign_score"] < df["kappa_B"], np.nan)

    # normalized anomaly scores (Regime A), for C5/C6
    df["R_norm"] = ((df["norm_score"] - df["tau_A"]) / df["tau_A"]).clip(lower=0, upper=1)
    df["R_cos"] = ((df["rho_A"] - df["cosine_score"]) / (df["rho_A"] + 1.0)).clip(lower=0, upper=1)
    df["R_sign"] = ((df["kappa_A"] - df["sign_score"]) / df["kappa_A"]).clip(lower=0, upper=1)

    # peer-relative Sign outlier check (C4): median of the OTHER 4 clients' sign_score
    # in the same (seed, partition, attack, round) group
    grp_cols = ["seed", "partition", "attack", "round"]

    def other_median(g: pd.Series) -> pd.Series:
        n = len(g)
        total = g.sum()
        # median of others excluding self is awkward with pandas median directly;
        # for n=5 this is fine to compute by leave-one-out on the sorted array.
        vals = g.values
        out = np.empty(n)
        for i in range(n):
            others = np.delete(vals, i)
            out[i] = np.median(others)
        return pd.Series(out, index=g.index)

    df["sign_peer_median"] = df.groupby(grp_cols)["sign_score"].transform(other_median)
    df["sign_peer_outlier"] = df["sign_score"] < df["sign_peer_median"]

    # drift-aware Norm gate (Variant B, STEP 5 design study)
    def tau_drift(g: pd.Series) -> pd.Series:
        m = g.median()
        mad = (g - m).abs().median()
        if mad == 0:
            return pd.Series(np.nan, index=g.index)  # signal fallback to static tau
        t = m + MODIFIED_Z_CUTOFF * MAD_TO_STD * mad
        return pd.Series(t, index=g.index)

    df["tau_drift_raw"] = df.groupby(grp_cols)["norm_score"].transform(tau_drift)
    df["tau_drift"] = df["tau_drift_raw"].fillna(df["tau_A"])  # fallback per CANDIDATE_DEFINITIONS.md
    df["norm_fail_drift"] = df["norm_score"] > df["tau_drift"]

    return df


def apply_signal_removal(flags: dict, remove: str | None) -> dict:
    """Ablation: force a signal's fail flag False / R component to 0."""
    f = dict(flags)
    if remove == "norm":
        f["norm_fail"] = pd.Series(False, index=f["norm_fail"].index)
        f["R_norm"] = pd.Series(0.0, index=f["R_norm"].index)
    elif remove == "cos":
        f["cos_fail"] = pd.Series(False, index=f["cos_fail"].index)
        f["R_cos"] = pd.Series(0.0, index=f["R_cos"].index)
    elif remove == "sign":
        f["sign_fail"] = pd.Series(False, index=f["sign_fail"].index)
        f["sign_peer_outlier"] = pd.Series(False, index=f["sign_peer_outlier"].index)
        f["R_sign"] = pd.Series(0.0, index=f["R_sign"].index)
    return f


def decide(candidate: str, df: pd.DataFrame, remove: str | None = None, drift_aware: bool = False) -> pd.Series:
    flags = {
        "norm_fail": df["norm_fail_drift"] if drift_aware else df["norm_fail_A"],
        "cos_fail": df["cos_fail_A"],
        "sign_fail": df["sign_fail_A"],
        "sign_peer_outlier": df["sign_peer_outlier"],
        "R_norm": df["R_norm"],
        "R_cos": df["R_cos"],
        "R_sign": df["R_sign"],
    }
    flags = apply_signal_removal(flags, remove)
    nf, cf, sf = flags["norm_fail"], flags["cos_fail"], flags["sign_fail"]

    if candidate == "C0":
        reject = nf | cf | sf
    elif candidate == "C1":
        reject = (nf.astype(int) + cf.astype(int) + sf.astype(int)) >= 2
    elif candidate == "C2":
        reject = nf | cf | sf  # logically == C0, see CANDIDATE_DEFINITIONS.md
    elif candidate == "C3":
        reject = nf | (cf & sf)
    elif candidate == "C4":
        sign_reject = sf & flags["sign_peer_outlier"]
        reject = nf | cf | sign_reject
    elif candidate == "C5":
        R = (flags["R_norm"] + flags["R_cos"] + flags["R_sign"]) / 3.0
        reject = R > 0.5
    elif candidate == "C6":
        D = 0.5 * flags["R_cos"] + 0.5 * flags["R_sign"]
        reject = nf | (D > 0.5)
    else:
        raise ValueError(candidate)
    return reject  # True = reject (flag as malicious)


def summarize(df: pd.DataFrame, reject: pd.Series, group_cols: list[str] | None = None) -> pd.DataFrame:
    tmp = df.copy()
    tmp["reject"] = reject
    rows = []
    if group_cols is None:
        groups = [((), tmp)]
    else:
        groups = list(tmp.groupby(group_cols))
    for key, g in groups:
        honest = g[g["honest"]]
        mal = g[g["malicious_active"]]
        row = {}
        if group_cols is not None:
            if not isinstance(key, tuple):
                key = (key,)
            for c, v in zip(group_cols, key):
                row[c] = v
        row["n_honest"] = len(honest)
        row["honest_fpr"] = honest["reject"].mean() if len(honest) else np.nan
        row["n_malicious_active"] = len(mal)
        row["malicious_tpr"] = mal["reject"].mean() if len(mal) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_trace()
    print(f"Loaded trace: {df.shape[0]} rows. Honest={df['honest'].sum()} malicious_active={df['malicious_active'].sum()}")

    # ---------------- sanity: C2 == C0 ----------------
    r_c0 = decide("C0", df)
    r_c2 = decide("C2", df)
    assert (r_c0 == r_c2).all(), "C2 must be logically identical to C0 under static thresholds"
    print("Sanity check passed: C2 decisions identical to C0 (static thresholds) for all 9000 rows.")

    # ---------------- STEP 6: offline replay summaries ----------------
    overall_rows, by_attack_rows, by_partition_rows, by_seed_rows = [], [], [], []
    decisions = {}
    for cand in CANDIDATES:
        reject = decide(cand, df)
        decisions[cand] = reject

        o = summarize(df, reject)
        o.insert(0, "candidate", cand)
        o["variant"] = "static"
        overall_rows.append(o)

        a = summarize(df, reject, ["attack"])
        a.insert(0, "candidate", cand)
        by_attack_rows.append(a)

        p = summarize(df, reject, ["partition"])
        p.insert(0, "candidate", cand)
        by_partition_rows.append(p)

        s = summarize(df, reject, ["seed"])
        s.insert(0, "candidate", cand)
        by_seed_rows.append(s)

        # drift-aware variant, only for hard-Norm-gate candidates (STEP 5)
        if cand in ("C2", "C3", "C4", "C6"):
            reject_d = decide(cand, df, drift_aware=True)
            od = summarize(df, reject_d)
            od.insert(0, "candidate", cand)
            od["variant"] = "drift_aware_norm_gate"
            overall_rows.append(od)

    overall = pd.concat(overall_rows, ignore_index=True)
    by_attack = pd.concat(by_attack_rows, ignore_index=True)
    by_partition = pd.concat(by_partition_rows, ignore_index=True)
    by_seed = pd.concat(by_seed_rows, ignore_index=True)

    # Regime B check (non-IID only), static variant only, primary candidates
    by_partition_B_rows = []
    non_iid = df[df["partition"] != "iid"].copy()
    for cand in CANDIDATES:
        flags_B = {
            "norm_fail": non_iid["norm_fail_B"].astype(bool),
            "cos_fail": non_iid["cos_fail_B"].astype(bool),
            "sign_fail": non_iid["sign_fail_B"].astype(bool),
        }
        nf, cf, sf = flags_B["norm_fail"], flags_B["cos_fail"], flags_B["sign_fail"]
        if cand in ("C0", "C2"):
            reject_B = nf | cf | sf
        elif cand == "C1":
            reject_B = (nf.astype(int) + cf.astype(int) + sf.astype(int)) >= 2
        elif cand == "C3":
            reject_B = nf | (cf & sf)
        else:
            continue  # C4/C5/C6 require R_* / peer-rank not recomputed for Regime B (documented limitation)
        pB = summarize(non_iid, reject_B, ["partition"])
        pB.insert(0, "candidate", cand)
        pB["regime"] = "B_iid_transfer_frozen"
        by_partition_B_rows.append(pB)
    by_partition["regime"] = "A_condition_specific_oracle"
    by_partition = pd.concat([by_partition] + by_partition_B_rows, ignore_index=True)

    overall.to_csv(os.path.join(OUT_DIR, "offline_replay_summary.csv"), index=False)
    by_attack.to_csv(os.path.join(OUT_DIR, "offline_replay_by_attack.csv"), index=False)
    by_partition.to_csv(os.path.join(OUT_DIR, "offline_replay_by_partition.csv"), index=False)
    by_seed.to_csv(os.path.join(OUT_DIR, "offline_replay_by_seed.csv"), index=False)
    print("Wrote offline_replay_summary/by_attack/by_partition/by_seed.csv")

    # ---------------- STEP 7: catastrophic-risk analysis ----------------
    risk_rows = []
    for cand in CANDIDATES:
        reject = decisions[cand]
        tmp = df.copy()
        tmp["reject"] = reject
        mal = tmp[tmp["malicious_active"]]

        ln = mal[mal["attack"] == "large_norm"]
        ln_accept = (~ln["reject"]).sum()
        ln_total = len(ln)

        dir_attacks = mal[mal["attack"].isin(["full_sign_flip", "directional_poisoning"])]
        dir_fn_total = (~dir_attacks["reject"]).sum()
        dir_total = len(dir_attacks)

        sparse = mal[mal["attack"] == "sparse_coordinate_attack"]
        sparse_fn = (~sparse["reject"]).sum()
        sparse_total = len(sparse)

        # max consecutive accepted (not rejected) run, per (seed,partition,attack) trajectory
        def max_consec_accept(g: pd.DataFrame) -> int:
            g = g.sort_values("round")
            accepted = (~g["reject"]).values
            best = cur = 0
            for a in accepted:
                cur = cur + 1 if a else 0
                best = max(best, cur)
            return best

        max_consec = (
            mal[mal["attack"].isin(["full_sign_flip", "directional_poisoning"])]
            .groupby(["seed", "partition", "attack"])
            .apply(max_consec_accept)
        )
        max_consec_directional = int(max_consec.max()) if len(max_consec) else 0

        risk_rows.append({
            "candidate": cand,
            "large_norm_accepted": int(ln_accept),
            "large_norm_total": int(ln_total),
            "catastrophic_risk_flag": bool(ln_accept > 0),
            "directional_fn_total": int(dir_fn_total),
            "directional_total": int(dir_total),
            "directional_fn_rate": dir_fn_total / dir_total if dir_total else np.nan,
            "max_consecutive_directional_accepts": max_consec_directional,
            "sparse_fn_total": int(sparse_fn),
            "sparse_total": int(sparse_total),
            "sparse_fn_rate": sparse_fn / sparse_total if sparse_total else np.nan,
        })
    risk_df = pd.DataFrame(risk_rows)
    risk_df.to_csv(os.path.join(OUT_DIR, "catastrophic_risk.csv"), index=False)
    print("Wrote catastrophic_risk.csv")
    print(risk_df.to_string(index=False))

    # ---------------- STEP 9: logical ablation ----------------
    ablation_rows = []
    for cand in CANDIDATES:
        for remove in [None, "norm", "cos", "sign"]:
            reject = decide(cand, df, remove=remove)
            o = summarize(df, reject)
            o.insert(0, "candidate", cand)
            o.insert(1, "removed_signal", remove if remove else "none")
            # catastrophic check under this ablation
            tmp = df.copy()
            tmp["reject"] = reject
            mal_ln = tmp[(tmp["malicious_active"]) & (tmp["attack"] == "large_norm")]
            o["large_norm_accepted"] = int((~mal_ln["reject"]).sum())
            ablation_rows.append(o)
    ablation_df = pd.concat(ablation_rows, ignore_index=True)
    ablation_df.to_csv(os.path.join(OUT_DIR, "ablation_replay.csv"), index=False)
    print("Wrote ablation_replay.csv")

    print("\nDONE — offline replay complete.")


if __name__ == "__main__":
    main()
