"""
extract_denial_reasons.py — Downloads each year's raw zip, extracts ONLY
denied applications with denial reason codes, saves a compact parquet, then
deletes the zip immediately to save disk space.

Run time: ~30-60 min (download dominated).
Output:   output/viz_denial_reasons.parquet

Denial reason codes:
  1 = Debt-to-income ratio
  2 = Employment history
  3 = Credit history
  4 = Collateral
  5 = Insufficient cash (downpayment/closing costs)
  6 = Unverifiable information
  7 = Credit application incomplete
  8 = Mortgage insurance denied
  9 = Other
"""

import gc
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from config import YEARS, URL_TEMPLATE, raw_zip, csv_name

OUTPUT   = Path("output")
RAW_DIR  = Path("raw")
RAW_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "HMDA-Pipeline/1.0"}

CHUNK_SIZE = 500_000

# action_taken codes for denied applications
DENIED_CODES = {"3", "7"}

DENIAL_REASON_MAP = {
    1: "Debt-to-income ratio",
    2: "Employment history",
    3: "Credit history",
    4: "Collateral",
    5: "Insufficient cash",
    6: "Unverifiable information",
    7: "Credit application incomplete",
    8: "Mortgage insurance denied",
    9: "Other",
}

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


def download_year(year: int) -> Path:
    url  = URL_TEMPLATE.format(year=year)
    dest = raw_zip(year)

    remote = REMOTE_SIZES.get(year, 0)
    local  = dest.stat().st_size if dest.exists() else 0

    if local >= remote * 0.999:
        print(f"  [{year}] Already downloaded.", flush=True)
        return dest

    resume = {}
    mode   = "wb"
    if 0 < local < remote:
        resume = {"Range": f"bytes={local}-"}
        mode   = "ab"
        print(f"  [{year}] Resuming from {local/1e6:.0f} MB…", flush=True)
    else:
        print(f"  [{year}] Downloading {remote/1e9:.2f} GB…", flush=True)

    t0 = time.time()
    with requests.get(url, headers={**HEADERS, **resume}, stream=True, timeout=120) as r:
        r.raise_for_status()
        downloaded = local
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                pct   = downloaded / remote * 100 if remote else 0
                speed = (downloaded - local) / max(time.time() - t0, 0.1) / 1e6
                print(f"\r    {pct:.1f}%  {downloaded/1e9:.2f}/{remote/1e9:.2f} GB  {speed:.1f} MB/s",
                      end="", flush=True)
    print(f"\n  [{year}] Download done in {time.time()-t0:.0f}s", flush=True)
    return dest


def extract_denial_reasons(year: int) -> pd.DataFrame:
    """Stream the zip, keep only denied rows, extract denial reason cols."""
    zip_path = raw_zip(year)
    name     = csv_name(year)

    # Discover available columns
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(name) as f:
            header = f.readline().decode("latin-1").strip().split(",")

    header_lower = [c.strip().lower() for c in header]

    # Map original → lower names we need
    NEED = {"action_taken", "loan_type", "loan_purpose", "state_abbr",
            "denial_reason_1", "denial_reason_2", "denial_reason_3"}

    # Some years use msamd instead of msa_md — we only need state here
    available = [c for c in header if c.strip().lower() in NEED]
    missing   = NEED - {c.strip().lower() for c in available}
    if missing:
        print(f"    [{year}] Missing cols: {missing}", flush=True)

    chunks_out = []
    rows_read  = 0

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(name) as f:
            for chunk in pd.read_csv(
                f, usecols=available, chunksize=CHUNK_SIZE,
                encoding="latin-1", low_memory=False, dtype=str,
            ):
                rows_read += len(chunk)
                chunk.columns = chunk.columns.str.strip().str.lower()

                # Keep only denied rows
                denied = chunk[chunk["action_taken"].isin(DENIED_CODES)].copy()
                if denied.empty:
                    continue

                # Cast denial reason codes to numeric
                for col in ["denial_reason_1", "denial_reason_2", "denial_reason_3"]:
                    if col in denied.columns:
                        denied[col] = pd.to_numeric(denied[col], errors="coerce")

                denied["year"] = year
                chunks_out.append(denied)

    df = pd.concat(chunks_out, ignore_index=True) if chunks_out else pd.DataFrame()
    del chunks_out; gc.collect()
    print(f"    [{year}] {rows_read:,} rows read → {len(df):,} denied kept", flush=True)
    return df


def aggregate_denial_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt denial_reason_1/2/3 into long form, count by year + reason.
    Each denied application can have up to 3 reasons — count each separately.
    """
    reason_cols = [c for c in ["denial_reason_1","denial_reason_2","denial_reason_3"]
                   if c in df.columns]

    melted = df.melt(
        id_vars=["year", "loan_purpose", "state_abbr"],
        value_vars=reason_cols,
        var_name="reason_slot",
        value_name="reason_code",
    ).dropna(subset=["reason_code"])

    melted["reason_code"] = melted["reason_code"].astype(int)
    melted = melted[melted["reason_code"].between(1, 9)]
    melted["reason_label"] = melted["reason_code"].map(DENIAL_REASON_MAP)

    agg = (
        melted.groupby(["year", "reason_code", "reason_label", "loan_purpose", "state_abbr"])
        .size()
        .reset_index(name="count")
    )
    return agg


def main():
    all_rows = []

    for year in YEARS:
        print(f"\n{'─'*60}", flush=True)
        print(f"  YEAR {year}", flush=True)
        t = time.time()

        # 1. Download
        download_year(year)

        # 2. Extract
        denied_df = extract_denial_reasons(year)

        # 3. Aggregate
        if not denied_df.empty:
            agg = aggregate_denial_reasons(denied_df)
            all_rows.append(agg)
            print(f"    [{year}] {len(agg):,} aggregated rows", flush=True)
        del denied_df; gc.collect()

        # 4. Delete zip immediately
        zp = raw_zip(year)
        if zp.exists():
            freed = zp.stat().st_size / 1e6
            zp.unlink()
            print(f"    [{year}] Zip deleted ({freed:.0f} MB freed)", flush=True)

        print(f"    [{year}] Done in {time.time()-t:.0f}s", flush=True)

    # Save
    final = pd.concat(all_rows, ignore_index=True)

    # Also save national rollup (no state breakdown) for easy charting
    national = (
        final.groupby(["year", "reason_code", "reason_label"])["count"]
        .sum().reset_index()
    )

    out = OUTPUT / "viz_denial_reasons.parquet"
    final.to_parquet(out, compression="snappy", index=False)
    print(f"\nSaved: {out}  ({len(final):,} rows)")

    out_nat = OUTPUT / "viz_denial_reasons_national.parquet"
    national.to_parquet(out_nat, compression="snappy", index=False)
    print(f"Saved: {out_nat}  ({len(national):,} rows)")

    # Quick preview
    print("\nTop denial reasons (all years, all purposes):")
    top = national.groupby("reason_label")["count"].sum().sort_values(ascending=False)
    print(top.to_string())


if __name__ == "__main__":
    t0 = time.time()
    print("Extracting denial reasons from HMDA 2007-2017…")
    main()
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
