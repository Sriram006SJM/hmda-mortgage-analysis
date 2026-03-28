"""
run_all.py — Monitors downloads and processes each year as soon as
its zip is fully downloaded. Prints a summary update after each year.

Usage:
    python run_all.py
"""

import gc
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import psutil

from config import (
    CHUNK_SIZE, CORE_COLS, NUMERIC_COLS, ORIGINATED_CODE,
    REQUIRED_COLS, YEARS, csv_name, parquet_path, raw_zip,
    OUTPUT_DIR, LOG_DIR,
)

# Remote file sizes in bytes (fetched once at start)
REMOTE_SIZES = {
    2007: 1_717_011_265,
    2008: 1_056_045_903,
    2009: 1_289_565_624,
    2010: 1_192_453_843,
    2011: 1_078_289_194,
    2012: 1_398_451_738,
    2013: 1_271_344_098,
    2014:   862_921_728,
    2015: 1_209_576_651,
    2016: 1_202_825_364,
    2017:   985_997_940,
}

COLUMN_DECISIONS = {
    "as_of_year":            ("REQUIRED", "Primary time dimension"),
    "action_taken":          ("REQUIRED", "Application vs origination metric"),
    "action_taken_name":     ("REQUIRED", "Human-readable action label"),
    "loan_type":             ("REQUIRED", "Loan category"),
    "loan_type_name":        ("REQUIRED", "Human-readable loan type"),
    "loan_purpose":          ("REQUIRED", "Purchase / refi / improvement"),
    "loan_purpose_name":     ("REQUIRED", "Human-readable loan purpose"),
    "loan_amount_000s":      ("REQUIRED", "Loan size in $000s"),
    "applicant_income_000s": ("REQUIRED", "Applicant income"),
    "msamd":                 ("REQUIRED", "MSA/MD market code"),
    "msamd_name":            ("REQUIRED", "MSA/MD name"),
    "state_code":            ("REQUIRED", "State FIPS code"),
    "state_name":            ("REQUIRED", "State name"),
    "state_abbr":            ("REQUIRED", "State abbreviation"),
    "county_code":           ("REQUIRED", "County FIPS"),
    "county_name":           ("REQUIRED", "County name"),
    "owner_occupancy":       ("OPTIONAL", "Owner vs investor"),
    "owner_occupancy_name":  ("OPTIONAL", "Human-readable occupancy"),
    "property_type":         ("OPTIONAL", "Property type"),
    "property_type_name":    ("OPTIONAL", "Human-readable property type"),
    "lien_status":           ("OPTIONAL", "First vs subordinate lien"),
    "lien_status_name":      ("OPTIONAL", "Human-readable lien status"),
    "purchaser_type":        ("OPTIONAL", "Secondary market purchaser"),
    "purchaser_type_name":   ("OPTIONAL", "Human-readable purchaser type"),
}


def is_complete(year: int) -> bool:
    p = raw_zip(year)
    if not p.exists():
        return False
    return p.stat().st_size >= REMOTE_SIZES.get(year, 0)


def process_year(year: int) -> dict:
    zip_path = raw_zip(year)
    out_path = parquet_path(year)
    agg_path = OUTPUT_DIR / f"hmda_agg_{year}.parquet"

    t_start = time.time()

    # ── Read header ───────────────────────────────────────────────────────────
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(csv_name(year)) as f:
            all_cols = f.readline().decode("latin-1").strip().split(",")

    kept = [c for c in all_cols if COLUMN_DECISIONS.get(c, ("DROPPED",))[0] in ("REQUIRED","OPTIONAL")]
    dropped = [c for c in all_cols if c not in kept]

    print(f"\n{'═'*70}")
    print(f"  YEAR {year} — COLUMN ANALYSIS")
    print(f"{'═'*70}")
    print(f"\nCOLUMNS FOUND: {len(all_cols)}")
    print(f"\nCOLUMNS KEPT ({len(kept)}):")
    for c in kept:
        dec, reason = COLUMN_DECISIONS.get(c, ("OPTIONAL","kept"))
        print(f"  ✓ [{dec:8}] {c:<35} — {reason}")
    print(f"\nCOLUMNS DROPPED ({len(dropped)}):")
    for c in dropped:
        _, reason = COLUMN_DECISIONS.get(c, ("DROPPED", "Not needed for macro trend analysis"))
        print(f"  ✗            {c:<35} — {reason}")

    # ── Stream process ────────────────────────────────────────────────────────
    print(f"\nPROCESSING {year} in {CHUNK_SIZE:,}-row chunks…")
    chunks = []
    rows_in = rows_out = chunk_n = 0

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(csv_name(year)) as f:
            for chunk in pd.read_csv(
                f, usecols=kept, chunksize=CHUNK_SIZE,
                encoding="latin-1", low_memory=False, dtype=str,
            ):
                chunk_n += 1
                rows_in += len(chunk)
                chunk.columns = chunk.columns.str.strip().str.lower()
                if "msamd" in chunk.columns:
                    chunk = chunk.rename(columns={"msamd": "msa_md", "msamd_name": "msa_md_name"})
                for col in NUMERIC_COLS:
                    c = "msa_md" if col == "msamd" else col
                    if c in chunk.columns:
                        chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
                req = [c for c in REQUIRED_COLS if c in chunk.columns]
                chunk = chunk.dropna(subset=req)
                rows_out += len(chunk)
                chunk["is_application"] = 1
                chunk["is_originated"] = (chunk["action_taken"] == ORIGINATED_CODE).astype("int8")
                chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()

    # ── Aggregation ───────────────────────────────────────────────────────────
    geo = "msa_md" if "msa_md" in df.columns else "state_code"
    agg = (
        df.groupby(["as_of_year", geo, "state_code", "loan_purpose"])
        .agg(
            total_applications=("is_application","sum"),
            total_originations=("is_originated","sum"),
            avg_loan_amount_000s=("loan_amount_000s","mean"),
            avg_income_000s=("applicant_income_000s","mean"),
        )
        .reset_index()
    )
    agg["origination_rate"] = (agg["total_originations"] / agg["total_applications"]).round(4)

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_parquet(out_path, compression="snappy", index=False)
    agg.to_parquet(agg_path, compression="snappy", index=False)

    # ── Delete raw ────────────────────────────────────────────────────────────
    zip_size_mb = zip_path.stat().st_size / 1e6
    zip_path.unlink()

    elapsed = time.time() - t_start
    total_apps = int(df["is_application"].sum())
    total_orig = int(df["is_originated"].sum())
    orig_rate  = total_orig / total_apps if total_apps else 0
    avg_loan   = df["loan_amount_000s"].mean()

    del df; gc.collect()

    result = {
        "year": year,
        "rows_in": rows_in,
        "rows_kept": rows_out,
        "rows_dropped": rows_in - rows_out,
        "cols_kept": len(kept),
        "cols_dropped": len(dropped),
        "total_applications": total_apps,
        "total_originations": total_orig,
        "origination_rate": orig_rate,
        "avg_loan_amount_000s": avg_loan,
        "parquet_mb": out_path.stat().st_size / 1e6,
        "agg_rows": len(agg),
        "elapsed_s": elapsed,
        "zip_freed_mb": zip_size_mb,
    }

    print(f"\nFINAL DATASET SHAPE: {rows_out:,} rows × {len(kept)+2} columns")
    print(f"  Rows dropped: {rows_in - rows_out:,}")
    print(f"FILE SAVED:    processed/hmda_cleaned_{year}.parquet  ({result['parquet_mb']:.1f} MB)")
    print(f"FILE SAVED:    output/hmda_agg_{year}.parquet")
    print(f"RAW FILE DELETED  ({zip_size_mb:.0f} MB freed)")

    return result


def print_update(results: list[dict]) -> None:
    print(f"\n{'━'*70}")
    print(f"  CUMULATIVE UPDATE — {len(results)} year(s) processed")
    print(f"{'━'*70}")
    print(f"  {'Year':<6} {'Applications':>14} {'Originations':>14} {'Rate':>7} {'AvgLoan$K':>10} {'Time':>7}")
    print(f"  {'─'*64}")
    for r in results:
        print(
            f"  {r['year']:<6}"
            f"{r['total_applications']:>14,.0f}"
            f"{r['total_originations']:>14,.0f}"
            f"{r['origination_rate']:>8.1%}"
            f"{r['avg_loan_amount_000s']:>10.0f}"
            f"{r['elapsed_s']:>6.0f}s"
        )
    print(f"{'━'*70}\n")


def wait_for_year(year: int, poll_sec: int = 15) -> None:
    if is_complete(year):
        return
    p = raw_zip(year)
    remote = REMOTE_SIZES.get(year, 0)
    print(f"  Waiting for {year} download… (target {remote/1e9:.2f} GB)", flush=True)
    while not is_complete(year):
        local = p.stat().st_size if p.exists() else 0
        pct = local / remote * 100 if remote else 0
        print(f"\r    {year}: {local/1e9:.2f}/{remote/1e9:.2f} GB  ({pct:.1f}%)", end="", flush=True)
        time.sleep(poll_sec)
    print(f"\r    {year}: download complete.{' '*30}")


def main():
    print(f"\n{'═'*70}")
    print(f"  HMDA PIPELINE — 2007 to 2017 — process-as-download")
    print(f"{'═'*70}\n")

    # 2017 already processed — load its result from parquet for the summary
    results = []
    if parquet_path(2017).exists():
        agg = pd.read_parquet(OUTPUT_DIR / "hmda_agg_2017.parquet")
        nat = agg.groupby("as_of_year").agg(
            total_applications=("total_applications","sum"),
            total_originations=("total_originations","sum"),
            avg_loan_amount_000s=("avg_loan_amount_000s","mean"),
        ).reset_index()
        r = nat.iloc[0]
        results.append({
            "year": 2017,
            "rows_in": 14285496, "rows_kept": 14083384, "rows_dropped": 202112,
            "cols_kept": 24, "cols_dropped": 54,
            "total_applications": int(r["total_applications"]),
            "total_originations": int(r["total_originations"]),
            "origination_rate": r["total_originations"] / r["total_applications"],
            "avg_loan_amount_000s": r["avg_loan_amount_000s"],
            "parquet_mb": parquet_path(2017).stat().st_size / 1e6,
            "agg_rows": len(agg),
            "elapsed_s": 162,
            "zip_freed_mb": 0,
        })
        print(f"  [2017] Already processed — loaded from cache.")
        print_update(results)

    process_order = [y for y in YEARS if y != 2017]

    for year in process_order:
        # Skip already-processed years
        if parquet_path(year).exists():
            print(f"  [{year}] Parquet already exists — loading from cache.", flush=True)
            agg_p = OUTPUT_DIR / f"hmda_agg_{year}.parquet"
            if agg_p.exists():
                agg = pd.read_parquet(agg_p)
                nat = agg.groupby("as_of_year").agg(
                    total_applications=("total_applications","sum"),
                    total_originations=("total_originations","sum"),
                    avg_loan_amount_000s=("avg_loan_amount_000s","mean"),
                ).reset_index()
                r = nat.iloc[0]
                results.append({
                    "year": year,
                    "rows_in": 0, "rows_kept": 0, "rows_dropped": 0,
                    "cols_kept": 24, "cols_dropped": 54,
                    "total_applications": int(r["total_applications"]),
                    "total_originations": int(r["total_originations"]),
                    "origination_rate": r["total_originations"] / r["total_applications"],
                    "avg_loan_amount_000s": r["avg_loan_amount_000s"],
                    "parquet_mb": parquet_path(year).stat().st_size / 1e6,
                    "agg_rows": len(agg),
                    "elapsed_s": 0,
                    "zip_freed_mb": 0,
                })
            results_sorted = sorted(results, key=lambda x: x["year"])
            print_update(results_sorted)
            continue

        wait_for_year(year)
        try:
            result = process_year(year)
            results.append(result)
            results_sorted = sorted(results, key=lambda x: x["year"])
            print_update(results_sorted)
        except Exception as e:
            print(f"\n  [{year}] ERROR: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()

    # ── Final unified output ──────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  ALL YEARS COMPLETE — FINAL SUMMARY")
    print(f"{'═'*70}")
    results_sorted = sorted(results, key=lambda x: x["year"])
    for r in results_sorted:
        print(f"  {r['year']}: {r['total_originations']:>10,.0f} originations  "
              f"rate={r['origination_rate']:.1%}  {r['parquet_mb']:.0f} MB parquet")
    print(f"{'═'*70}")
    print("  Run: python analyze.py  for crash/recovery trend analysis")


if __name__ == "__main__":
    main()
