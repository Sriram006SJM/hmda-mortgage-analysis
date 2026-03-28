"""
analyze.py — Reads all yearly aggregated parquets and produces:
  1. A unified multi-year aggregated parquet
  2. Console trend report identifying:
     - Market crash bottom (lowest originations)
     - Recovery inflection point
     - Application vs origination rate trend

Usage:
    python analyze.py
"""

import sys
from pathlib import Path

import pandas as pd

from config import OUTPUT_DIR, YEARS

LOAN_PURPOSE_MAP = {
    1: "Home Purchase",
    2: "Home Improvement",
    3: "Refinancing",
}


def load_all() -> pd.DataFrame:
    frames = []
    for year in YEARS:
        p = OUTPUT_DIR / f"hmda_agg_{year}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            frames.append(df)
        else:
            print(f"  WARNING: {p.name} not found — run pipeline.py first")
    if not frames:
        print("No aggregated files found. Run pipeline.py first.")
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)


def national_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Roll up to national level by year."""
    nat = (
        df.groupby("as_of_year")
        .agg(
            total_applications=("total_applications", "sum"),
            total_originations=("total_originations", "sum"),
            avg_loan_amount_000s=("avg_loan_amount_000s", "mean"),
        )
        .reset_index()
        .sort_values("as_of_year")
    )
    nat["origination_rate"] = (
        nat["total_originations"] / nat["total_applications"]
    ).round(4)
    nat["yoy_origination_chg"] = nat["total_originations"].pct_change().round(4)
    return nat


def main():
    print("=" * 70)
    print("  HMDA NATIONAL TREND ANALYSIS — 2007–2017")
    print("=" * 70)

    df = load_all()
    print(f"\n  Loaded {len(df):,} aggregated rows across {df['as_of_year'].nunique()} years")

    # ── National Trend ────────────────────────────────────────────────────────
    nat = national_trend(df)

    print("\n  NATIONAL: Applications vs Originations by Year")
    print(f"  {'Year':<6} {'Applications':>15} {'Originations':>14} "
          f"{'Rate':>7} {'YoY Chg':>9}  {'Avg Loan ($K)':>13}")
    print("  " + "─" * 68)
    for _, row in nat.iterrows():
        yoy = f"{row['yoy_origination_chg']*100:+.1f}%" if pd.notna(row['yoy_origination_chg']) else "   n/a"
        print(
            f"  {int(row['as_of_year']):<6}"
            f"{row['total_applications']:>15,.0f}"
            f"{row['total_originations']:>14,.0f}"
            f"{row['origination_rate']:>8.1%}"
            f"{yoy:>9}"
            f"{row['avg_loan_amount_000s']:>14,.0f}"
        )

    # ── Crash Bottom ─────────────────────────────────────────────────────────
    bottom_row = nat.loc[nat["total_originations"].idxmin()]
    print(f"\n  MARKET CRASH BOTTOM:")
    print(f"  Year {int(bottom_row['as_of_year'])} had the LOWEST originations: "
          f"{bottom_row['total_originations']:,.0f}")

    # ── Recovery Inflection ───────────────────────────────────────────────────
    after_bottom = nat[nat["as_of_year"] > bottom_row["as_of_year"]]
    recovery = after_bottom[after_bottom["yoy_origination_chg"] > 0]
    if not recovery.empty:
        inflection_year = int(recovery.iloc[0]["as_of_year"])
        inflection_chg = recovery.iloc[0]["yoy_origination_chg"]
        print(f"\n  RECOVERY INFLECTION POINT:")
        print(f"  Year {inflection_year} — first year with positive origination growth "
              f"after bottom ({inflection_chg*100:+.1f}% YoY)")

    # ── By Loan Purpose ───────────────────────────────────────────────────────
    if "loan_purpose" in df.columns:
        purpose_trend = (
            df.groupby(["as_of_year", "loan_purpose"])
            .agg(total_originations=("total_originations", "sum"))
            .reset_index()
        )
        purpose_trend["purpose_label"] = purpose_trend["loan_purpose"].map(LOAN_PURPOSE_MAP)

        print(f"\n  ORIGINATIONS BY LOAN PURPOSE (nationwide totals):")
        pivot = purpose_trend.pivot_table(
            index="as_of_year",
            columns="purpose_label",
            values="total_originations",
            aggfunc="sum",
        ).fillna(0).astype(int)
        print(pivot.to_string())

    # ── Save unified output ───────────────────────────────────────────────────
    unified_path = OUTPUT_DIR / "hmda_national_trend_2007_2017.parquet"
    nat.to_parquet(unified_path, compression="snappy", index=False)
    print(f"\n  Saved: {unified_path}")

    unified_all_path = OUTPUT_DIR / "hmda_unified_agg_2007_2017.parquet"
    df.to_parquet(unified_all_path, compression="snappy", index=False)
    print(f"  Saved: {unified_all_path}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
