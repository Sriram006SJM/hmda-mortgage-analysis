"""
prepare_viz.py — Pre-computes all aggregations needed for the visualization dashboard.
Reads processed parquets (only relevant columns) and writes tiny viz-ready parquets.

Run once after pipeline.py is complete. Takes ~5–8 min.
Output: output/viz_*.parquet
"""

import gc
import time
from pathlib import Path

import pandas as pd

PROCESSED = Path("processed")
OUTPUT = Path("output")
YEARS = list(range(2007, 2018))

# action_taken codes
APPROVED  = {1, 2, 8}
DENIED    = {3, 7}
DECIDED   = {1, 2, 3, 7, 8}

# loan_type mapping
LOAN_TYPE_MAP = {
    1: "Conventional (Private)",
    2: "FHA-Insured (Govt)",
    3: "VA-Guaranteed (Govt)",
    4: "FSA/RHS (Govt)",
}

# loan_purpose mapping
LOAN_PURPOSE_MAP = {
    1: "Home Purchase",
    2: "Home Improvement",
    3: "Refinancing",
}


def load_year(year: int, cols: list, retries: int = 3) -> pd.DataFrame:
    path = PROCESSED / f"hmda_cleaned_{year}.parquet"
    for attempt in range(retries):
        try:
            df = pd.read_parquet(path, columns=cols)
            return df
        except Exception as e:
            if attempt < retries - 1:
                print(f"    Retry {attempt+1}/{retries-1} for {year}: {e}", flush=True)
                gc.collect()
                time.sleep(3)
            else:
                raise


def compute_all():
    state_rows = []
    loan_type_rows = []
    loan_purpose_rows = []
    msa_rows = []
    national_rows = []

    for year in YEARS:
        t = time.time()
        print(f"  Processing {year}…", flush=True)

        df = load_year(year, [
            "as_of_year", "action_taken", "loan_type", "loan_purpose",
            "loan_amount_000s", "state_abbr", "state_name",
            "msa_md", "msa_md_name", "is_originated",
        ])

        df["is_decided"] = df["action_taken"].isin(DECIDED).astype("int8")
        df["is_denied"]  = df["action_taken"].isin(DENIED).astype("int8")
        df["is_approved"] = df["action_taken"].isin(APPROVED).astype("int8")
        df["is_govt"] = df["loan_type"].isin([2, 3, 4]).astype("int8")

        # ── 1. State-level denial & origination rates ─────────────────────
        sg = df.groupby("state_abbr").agg(
            state_name=("state_name", "first"),
            total_apps=("is_decided", "sum"),
            total_denied=("is_denied", "sum"),
            total_approved=("is_approved", "sum"),
            total_originated=("is_originated", "sum"),
            avg_loan=("loan_amount_000s", "mean"),
        ).reset_index()
        sg["denial_rate"]  = (sg["total_denied"]    / sg["total_apps"].clip(1)).round(4)
        sg["orig_rate"]    = (sg["total_originated"] / sg["total_apps"].clip(1)).round(4)
        sg["year"] = year
        state_rows.append(sg)

        # ── 2. National loan type mix ──────────────────────────────────────
        lt = df.groupby("loan_type").agg(
            count=("loan_type", "size"),
            originated=("is_originated", "sum"),
        ).reset_index()
        lt["year"] = year
        lt["loan_type_label"] = lt["loan_type"].map(LOAN_TYPE_MAP)
        loan_type_rows.append(lt)

        # ── 3. National loan purpose mix ──────────────────────────────────
        lp = df.groupby("loan_purpose").agg(
            count=("loan_purpose", "size"),
            originated=("is_originated", "sum"),
            denied=("is_denied", "sum"),
            approved=("is_approved", "sum"),
            decided=("is_decided", "sum"),
        ).reset_index()
        lp["year"] = year
        lp["loan_purpose_label"] = lp["loan_purpose"].map(LOAN_PURPOSE_MAP)
        loan_purpose_rows.append(lp)

        # ── 4. MSA-level originations ──────────────────────────────────────
        msa = df[df["msa_md"].notna() & (df["msa_md"] > 0)].groupby(
            ["msa_md", "msa_md_name"]
        ).agg(
            total_apps=("is_decided", "sum"),
            total_originated=("is_originated", "sum"),
            avg_loan=("loan_amount_000s", "mean"),
        ).reset_index()
        msa["year"] = year
        msa_rows.append(msa)

        # ── 5. National summary ────────────────────────────────────────────
        national_rows.append({
            "year": year,
            "total_apps": int(df["is_decided"].sum()),
            "total_originated": int(df["is_originated"].sum()),
            "total_denied": int(df["is_denied"].sum()),
            "total_govt_loans": int(df["is_govt"].sum()),
            "total_loans": int(len(df)),
            "avg_loan_000s": float(df["loan_amount_000s"].mean()),
        })

        del df; gc.collect()
        print(f"    Done in {time.time()-t:.0f}s", flush=True)

    # ── Save all ───────────────────────────────────────────────────────────
    print("Saving viz parquets…")

    state_df = pd.concat(state_rows, ignore_index=True)
    state_df.to_parquet(OUTPUT / "viz_state_denial.parquet", compression="snappy", index=False)
    print(f"  viz_state_denial.parquet  ({len(state_df):,} rows)")

    lt_df = pd.concat(loan_type_rows, ignore_index=True)
    lt_df.to_parquet(OUTPUT / "viz_loan_type.parquet", compression="snappy", index=False)
    print(f"  viz_loan_type.parquet  ({len(lt_df):,} rows)")

    lp_df = pd.concat(loan_purpose_rows, ignore_index=True)
    lp_df.to_parquet(OUTPUT / "viz_loan_purpose.parquet", compression="snappy", index=False)
    print(f"  viz_loan_purpose.parquet  ({len(lp_df):,} rows)")

    msa_df = pd.concat(msa_rows, ignore_index=True)
    msa_df.to_parquet(OUTPUT / "viz_msa.parquet", compression="snappy", index=False)
    print(f"  viz_msa.parquet  ({len(msa_df):,} rows)")

    nat_df = pd.DataFrame(national_rows)
    nat_df["orig_rate"] = nat_df["total_originated"] / nat_df["total_apps"].clip(1)
    nat_df["govt_share"] = nat_df["total_govt_loans"] / nat_df["total_loans"].clip(1)
    nat_df.to_parquet(OUTPUT / "viz_national.parquet", compression="snappy", index=False)
    print(f"  viz_national.parquet  ({len(nat_df):,} rows)")

    # ── MSA Recovery Index (2007 baseline = 100) ───────────────────────────
    msa_2007 = msa_df[msa_df["year"] == 2007][["msa_md", "msa_md_name", "total_originated", "total_apps"]].copy()
    msa_2007 = msa_2007.rename(columns={"total_originated": "orig_2007", "total_apps": "apps_2007"})

    msa_all = msa_df.merge(msa_2007[["msa_md", "orig_2007", "apps_2007"]], on="msa_md", how="inner")
    msa_all["recovery_index"] = (msa_all["total_originated"] / msa_all["orig_2007"].clip(1) * 100).round(1)

    # Keep MSAs with meaningful 2007 volume
    valid_msas = msa_2007[msa_2007["orig_2007"] >= 500]["msa_md"]
    msa_recovery = msa_all[msa_all["msa_md"].isin(valid_msas)]
    msa_recovery.to_parquet(OUTPUT / "viz_msa_recovery.parquet", compression="snappy", index=False)
    print(f"  viz_msa_recovery.parquet  ({len(msa_recovery):,} rows)")

    print("\nAll viz data ready.")


if __name__ == "__main__":
    print("Preparing visualization data…")
    t0 = time.time()
    compute_all()
    print(f"Total time: {(time.time()-t0)/60:.1f} min")
