"""
extract_income_stats.py — Computes income & affordability statistics from
processed parquets. No re-downloading needed.

Outputs:
  output/viz_income_national.parquet   — mean/median income & loan by year
  output/viz_income_purpose.parquet    — by year × loan purpose
  output/viz_income_state.parquet      — by year × state (for map)
  output/viz_affordability.parquet     — loan-to-income ratio over years
"""

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("processed")
OUTPUT    = Path("output")
YEARS     = list(range(2007, 2018))

LOAN_PURPOSE_MAP = {1: "Home Purchase", 2: "Home Improvement", 3: "Refinancing"}

# Filter out HMDA cap codes & unreasonable values
INCOME_MAX = 9998   # 9999 = HMDA "not available" cap
INCOME_MIN = 1      # $1K minimum
LOAN_MAX   = 9998   # $9.998M — filters coded caps
LOAN_MIN   = 1


def load_year(year: int) -> pd.DataFrame:
    path = PROCESSED / f"hmda_cleaned_{year}.parquet"
    for attempt in range(3):
        try:
            df = pd.read_parquet(path, columns=[
                "as_of_year", "action_taken", "loan_purpose",
                "loan_amount_000s", "applicant_income_000s",
                "state_abbr", "state_name", "is_originated",
            ])
            return df
        except Exception as e:
            if attempt < 2:
                print(f"    Retry {attempt+1} for {year}: {e}", flush=True)
                gc.collect()
                time.sleep(3)
            else:
                raise


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove cap codes and nulls from income and loan amount."""
    df = df.dropna(subset=["applicant_income_000s", "loan_amount_000s"])
    df = df[
        df["applicant_income_000s"].between(INCOME_MIN, INCOME_MAX) &
        df["loan_amount_000s"].between(LOAN_MIN, LOAN_MAX)
    ]
    return df


def stats(series: pd.Series) -> dict:
    return {
        "mean":   round(series.mean(), 2),
        "median": round(series.median(), 2),
        "p25":    round(series.quantile(0.25), 2),
        "p75":    round(series.quantile(0.75), 2),
    }


def compute_all():
    national_rows  = []
    purpose_rows   = []
    state_rows     = []

    for year in YEARS:
        t = time.time()
        print(f"  Processing {year}…", flush=True)

        df_raw = load_year(year)
        df     = clean(df_raw)

        originated = df[df["is_originated"] == 1]
        denied     = df_raw[df_raw["action_taken"].isin([3, 7])].pipe(clean)

        df["loan_to_income"]         = df["loan_amount_000s"] / df["applicant_income_000s"]
        originated["loan_to_income"] = originated["loan_amount_000s"] / originated["applicant_income_000s"]

        # ── 1. National ────────────────────────────────────────────────────────
        row = {"year": year}

        for prefix, subset in [("all", df), ("approved", originated), ("denied", denied)]:
            inc = stats(subset["applicant_income_000s"])
            loa = stats(subset["loan_amount_000s"])
            row[f"{prefix}_income_mean"]   = inc["mean"]
            row[f"{prefix}_income_median"] = inc["median"]
            row[f"{prefix}_income_p25"]    = inc["p25"]
            row[f"{prefix}_income_p75"]    = inc["p75"]
            row[f"{prefix}_loan_mean"]     = loa["mean"]
            row[f"{prefix}_loan_median"]   = loa["median"]
            row[f"{prefix}_count"]         = len(subset)

        lti = stats(originated["loan_to_income"])
        row["lti_mean"]   = lti["mean"]
        row["lti_median"] = lti["median"]

        # Income needed per $100K loan (reciprocal of LTI)
        row["income_per_100k_loan"] = round(100 / lti["median"], 1)

        national_rows.append(row)

        # ── 2. By loan purpose ─────────────────────────────────────────────────
        for purpose_code, purpose_label in LOAN_PURPOSE_MAP.items():
            sub_all  = df[df["loan_purpose"] == purpose_code]
            sub_orig = originated[originated["loan_purpose"] == purpose_code]
            if sub_orig.empty:
                continue
            purpose_rows.append({
                "year":                  year,
                "loan_purpose":          purpose_code,
                "loan_purpose_label":    purpose_label,
                "approved_income_mean":  round(sub_orig["applicant_income_000s"].mean(), 2),
                "approved_income_median":round(sub_orig["applicant_income_000s"].median(), 2),
                "approved_loan_mean":    round(sub_orig["loan_amount_000s"].mean(), 2),
                "approved_loan_median":  round(sub_orig["loan_amount_000s"].median(), 2),
                "lti_median":            round(sub_orig["loan_to_income"].median(), 2),
                "all_income_median":     round(sub_all["applicant_income_000s"].median(), 2),
                "count":                 len(sub_orig),
            })

        # ── 3. By state ────────────────────────────────────────────────────────
        sg = originated.groupby("state_abbr").agg(
            state_name=("state_name", "first"),
            approved_income_mean=("applicant_income_000s", "mean"),
            approved_income_median=("applicant_income_000s", "median"),
            approved_loan_mean=("loan_amount_000s", "mean"),
            approved_loan_median=("loan_amount_000s", "median"),
            count=("is_originated", "count"),
        ).reset_index()
        sg["lti_median"] = (
            originated.groupby("state_abbr")
            .apply(lambda x: (x["loan_amount_000s"] / x["applicant_income_000s"]).median())
            .reset_index(drop=True)
        )
        sg["year"] = year
        state_rows.append(sg)

        del df_raw, df, originated, denied; gc.collect()
        print(f"    Done in {time.time()-t:.0f}s", flush=True)

    # ── Save ───────────────────────────────────────────────────────────────────
    print("\nSaving…")

    nat_df = pd.DataFrame(national_rows)
    nat_df.to_parquet(OUTPUT / "viz_income_national.parquet", compression="snappy", index=False)
    print(f"  viz_income_national.parquet  ({len(nat_df)} rows)")
    print(nat_df[["year","approved_income_median","approved_loan_median","lti_median","income_per_100k_loan"]].to_string(index=False))

    lp_df = pd.DataFrame(purpose_rows)
    lp_df.to_parquet(OUTPUT / "viz_income_purpose.parquet", compression="snappy", index=False)
    print(f"\n  viz_income_purpose.parquet  ({len(lp_df)} rows)")

    st_df = pd.concat(state_rows, ignore_index=True)
    # Fix lti_median alignment issue from groupby apply
    st_df["lti_median"] = pd.to_numeric(st_df["lti_median"], errors="coerce")
    st_df.to_parquet(OUTPUT / "viz_income_state.parquet", compression="snappy", index=False)
    print(f"  viz_income_state.parquet  ({len(st_df)} rows)")


if __name__ == "__main__":
    t0 = time.time()
    print("Extracting income & affordability stats…")
    compute_all()
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
