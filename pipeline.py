"""
pipeline.py — HMDA Data Engineering Pipeline (2007–2017)

Processes each year's nationwide LAR CSV (from zip) into:
  1. Cleaned row-level parquet  →  processed/hmda_cleaned_{year}.parquet
  2. Aggregated summary parquet →  output/hmda_agg_{year}.parquet

Follows the 8-step process:
  STEP 1: LOAD  |  STEP 2: COLUMN ANALYSIS  |  STEP 3: SELECT
  STEP 4: TRANSFORM  |  STEP 5: DERIVED METRICS
  STEP 6: AGGREGATE  |  STEP 7: SAVE  |  STEP 8: MEMORY

Usage:
    python pipeline.py                   # process all years with downloaded zips
    python pipeline.py --years 2017      # single year
    python pipeline.py --no-delete       # keep raw zip after processing
"""

import argparse
import gc
import logging
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import psutil

from config import (
    CHUNK_SIZE,
    CORE_COLS,
    NUMERIC_COLS,
    ORIGINATED_CODE,
    REQUIRED_COLS,
    YEARS,
    csv_name,
    parquet_path,
    raw_zip,
    LOG_DIR,
    OUTPUT_DIR,
)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "pipeline.log"),
    ],
)
log = logging.getLogger(__name__)

# ── All 78 columns with KEEP/DROP decision ────────────────────────────────────

COLUMN_DECISIONS = {
    # REQUIRED — core macro-trend fields
    "as_of_year":            ("REQUIRED", "Primary time dimension for all trend analysis"),
    "action_taken":          ("REQUIRED", "Core metric: determines application vs origination"),
    "action_taken_name":     ("REQUIRED", "Human-readable label for action_taken"),
    "loan_type":             ("REQUIRED", "Loan category (conventional, FHA, VA, etc.)"),
    "loan_type_name":        ("REQUIRED", "Human-readable loan type label"),
    "loan_purpose":          ("REQUIRED", "Purpose: purchase / refi / improvement"),
    "loan_purpose_name":     ("REQUIRED", "Human-readable loan purpose label"),
    "loan_amount_000s":      ("REQUIRED", "Loan size in $000s — key economic indicator"),
    "applicant_income_000s": ("REQUIRED", "Applicant income — affordability analysis"),
    "msamd":                 ("REQUIRED", "MSA/MD code — geographic market identifier"),
    "msamd_name":            ("REQUIRED", "MSA/MD name — human-readable geography"),
    "state_code":            ("REQUIRED", "State FIPS code — geographic aggregation"),
    "state_name":            ("REQUIRED", "State name — human-readable"),
    "state_abbr":            ("REQUIRED", "State abbreviation — compact geo label"),
    "county_code":           ("REQUIRED", "County FIPS — sub-state geography"),
    "county_name":           ("REQUIRED", "County name — human-readable"),
    # OPTIONAL — enriching context
    "owner_occupancy":       ("OPTIONAL", "Occupancy type — filters investor vs owner-occupied"),
    "owner_occupancy_name":  ("OPTIONAL", "Human-readable occupancy label"),
    "property_type":         ("OPTIONAL", "Property type — 1-4 family, multifamily, etc."),
    "property_type_name":    ("OPTIONAL", "Human-readable property type label"),
    "lien_status":           ("OPTIONAL", "Lien position — first vs subordinate lien"),
    "lien_status_name":      ("OPTIONAL", "Human-readable lien status label"),
    "purchaser_type":        ("OPTIONAL", "Secondary market purchaser — GSE, FHA, etc."),
    "purchaser_type_name":   ("OPTIONAL", "Human-readable purchaser type label"),
    # DROPPED — not needed for macro trend analysis
    "respondent_id":         ("DROPPED", "Lender identifier — institution-level, not needed"),
    "agency_name":           ("DROPPED", "Regulatory agency — not relevant to macro trends"),
    "agency_abbr":           ("DROPPED", "Regulatory agency abbreviation — not needed"),
    "agency_code":           ("DROPPED", "Regulatory agency code — not needed"),
    "preapproval":           ("DROPPED", "Preapproval flag — niche subset, not macro"),
    "preapproval_name":      ("DROPPED", "Preapproval label — dropped with preapproval"),
    "census_tract_number":   ("DROPPED", "Census tract — too granular for macro analysis"),
    "applicant_ethnicity":   ("DROPPED", "Demographic — outside scope of macro trends"),
    "applicant_ethnicity_name": ("DROPPED", "Demographic label — outside scope"),
    "co_applicant_ethnicity":   ("DROPPED", "Demographic — outside scope"),
    "co_applicant_ethnicity_name": ("DROPPED", "Demographic label — outside scope"),
    "applicant_race_name_1": ("DROPPED", "Demographic — outside scope of macro analysis"),
    "applicant_race_1":      ("DROPPED", "Demographic — outside scope"),
    "applicant_race_name_2": ("DROPPED", "Demographic — outside scope"),
    "applicant_race_2":      ("DROPPED", "Demographic — outside scope"),
    "applicant_race_name_3": ("DROPPED", "Demographic — outside scope"),
    "applicant_race_3":      ("DROPPED", "Demographic — outside scope"),
    "applicant_race_name_4": ("DROPPED", "Demographic — outside scope"),
    "applicant_race_4":      ("DROPPED", "Demographic — outside scope"),
    "applicant_race_name_5": ("DROPPED", "Demographic — outside scope"),
    "applicant_race_5":      ("DROPPED", "Demographic — outside scope"),
    "co_applicant_race_name_1": ("DROPPED", "Demographic — outside scope"),
    "co_applicant_race_1":   ("DROPPED", "Demographic — outside scope"),
    "co_applicant_race_name_2": ("DROPPED", "Demographic — outside scope"),
    "co_applicant_race_2":   ("DROPPED", "Demographic — outside scope"),
    "co_applicant_race_name_3": ("DROPPED", "Demographic — outside scope"),
    "co_applicant_race_3":   ("DROPPED", "Demographic — outside scope"),
    "co_applicant_race_name_4": ("DROPPED", "Demographic — outside scope"),
    "co_applicant_race_4":   ("DROPPED", "Demographic — outside scope"),
    "co_applicant_race_name_5": ("DROPPED", "Demographic — outside scope"),
    "co_applicant_race_5":   ("DROPPED", "Demographic — outside scope"),
    "applicant_sex":         ("DROPPED", "Demographic — outside scope"),
    "applicant_sex_name":    ("DROPPED", "Demographic — outside scope"),
    "co_applicant_sex":      ("DROPPED", "Demographic — outside scope"),
    "co_applicant_sex_name": ("DROPPED", "Demographic — outside scope"),
    "denial_reason_name_1":  ("DROPPED", "Denial reasons captured separately if needed"),
    "denial_reason_1":       ("DROPPED", "Denial reasons — out of scope for origination trend"),
    "denial_reason_name_2":  ("DROPPED", "Denial reasons — out of scope"),
    "denial_reason_2":       ("DROPPED", "Denial reasons — out of scope"),
    "denial_reason_name_3":  ("DROPPED", "Denial reasons — out of scope"),
    "denial_reason_3":       ("DROPPED", "Denial reasons — out of scope"),
    "rate_spread":           ("DROPPED", "Interest rate detail — not needed for volume trends"),
    "hoepa_status":          ("DROPPED", "HOEPA flag — regulatory subset, not macro"),
    "hoepa_status_name":     ("DROPPED", "HOEPA label — dropped with hoepa_status"),
    "edit_status":           ("DROPPED", "Data quality flag — internal CFPB use"),
    "edit_status_name":      ("DROPPED", "Edit status label — internal CFPB use"),
    "sequence_number":       ("DROPPED", "Row sequence — not analytically useful"),
    "population":            ("DROPPED", "Census tract population — too granular"),
    "minority_population":   ("DROPPED", "Census tract demographic — out of scope"),
    "hud_median_family_income": ("DROPPED", "Tract-level income — use applicant_income instead"),
    "tract_to_msamd_income": ("DROPPED", "Tract ratio — too granular for macro analysis"),
    "number_of_owner_occupied_units": ("DROPPED", "Tract-level housing count — too granular"),
    "number_of_1_to_4_family_units":  ("DROPPED", "Tract-level unit count — too granular"),
    "application_date_indicator": ("DROPPED", "Filing timing flag — not needed"),
}


def mem_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def print_separator(char="─", width=70):
    log.info(char * width)


def process_year(year: int, delete_raw: bool = True) -> bool:
    zip_path = raw_zip(year)
    if not zip_path.exists():
        log.info(f"  [{year}] ZIP not found: {zip_path}. Run download.py first.")
        return False

    out_path = parquet_path(year)
    agg_path = OUTPUT_DIR / f"hmda_agg_{year}.parquet"

    print_separator("═")
    log.info(f"  YEAR: {year}")
    print_separator("═")

    # ── STEP 1: LOAD ──────────────────────────────────────────────────────────
    log.info("\nSTEP 1: LOAD DATA")
    with zipfile.ZipFile(zip_path) as zf:
        inner = csv_name(year)
        with zf.open(inner) as f:
            header_line = f.readline().decode("latin-1").strip()
    all_cols = [c.strip() for c in header_line.split(",")]

    log.info(f"\nCOLUMNS FOUND ({len(all_cols)} total):")
    for i, c in enumerate(all_cols, 1):
        log.info(f"  {i:3}. {c}")

    # ── STEP 2: COLUMN ANALYSIS ───────────────────────────────────────────────
    log.info("\nSTEP 2: COLUMN ANALYSIS")

    keep_cols = []
    drop_cols = []
    for col in all_cols:
        decision, reason = COLUMN_DECISIONS.get(col, ("DROPPED", "Not in analysis schema"))
        if decision in ("REQUIRED", "OPTIONAL"):
            keep_cols.append((col, decision, reason))
        else:
            drop_cols.append((col, reason))

    log.info(f"\nCOLUMNS KEPT ({len(keep_cols)}):")
    for col, dec, reason in keep_cols:
        log.info(f"  ✓ [{dec:8}] {col:<35} — {reason}")

    log.info(f"\nCOLUMNS DROPPED ({len(drop_cols)}):")
    for col, reason in drop_cols:
        log.info(f"  ✗            {col:<35} — {reason}")

    # ── STEP 3: SELECT COLUMNS ────────────────────────────────────────────────
    log.info("\nSTEP 3: SELECT COLUMNS")
    selected = [c for c, _, _ in keep_cols]
    selected = [c for c in selected if c in all_cols]
    log.info(f"  Selecting {len(selected)} columns from {len(all_cols)} total")

    # ── STEP 4–7: STREAM PROCESS IN CHUNKS ───────────────────────────────────
    log.info(f"\nSTEP 4–7: PROCESSING IN {CHUNK_SIZE:,}-ROW CHUNKS")

    chunks_processed = []
    total_rows_in = 0
    total_rows_out = 0
    chunk_num = 0
    mem_before = mem_mb()

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(csv_name(year)) as f:
            reader = pd.read_csv(
                f,
                usecols=selected,
                chunksize=CHUNK_SIZE,
                encoding="latin-1",
                low_memory=False,
                dtype=str,   # read all as str first; cast below
            )

            for chunk in reader:
                chunk_num += 1
                total_rows_in += len(chunk)

                # STEP 4: Transform
                chunk.columns = chunk.columns.str.strip().str.lower()

                # Rename msamd → msa_md
                if "msamd" in chunk.columns:
                    chunk = chunk.rename(columns={"msamd": "msa_md", "msamd_name": "msa_md_name"})

                # Cast numeric columns
                for col in NUMERIC_COLS:
                    if col in chunk.columns:
                        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
                    # handle renamed msamd
                    if col == "msamd" and "msa_md" in chunk.columns:
                        chunk["msa_md"] = pd.to_numeric(chunk["msa_md"], errors="coerce")

                # Drop rows missing critical fields
                req_present = [c for c in REQUIRED_COLS if c in chunk.columns]
                chunk = chunk.dropna(subset=req_present)

                total_rows_out += len(chunk)

                # STEP 5: Derived metrics
                chunk["is_application"] = 1
                chunk["is_originated"] = (chunk["action_taken"] == ORIGINATED_CODE).astype("int8")

                chunks_processed.append(chunk)

                if chunk_num % 5 == 0:
                    log.info(f"  Chunk {chunk_num:3}: {total_rows_in:>12,} rows in | "
                             f"{total_rows_out:>12,} rows kept | mem {mem_mb():.0f} MB")

    # Combine all chunks
    log.info(f"  Combining {chunk_num} chunks…")
    df = pd.concat(chunks_processed, ignore_index=True)
    del chunks_processed
    gc.collect()

    log.info(f"\nFINAL DATASET SHAPE: {df.shape[0]:,} rows × {df.shape[1]} columns")
    log.info(f"  Rows dropped (missing critical fields): {total_rows_in - total_rows_out:,}")

    # STEP 6: AGGREGATION
    log.info("\nSTEP 6: AGGREGATION")
    geo_col = "msa_md" if "msa_md" in df.columns else "state_code"
    agg = (
        df.groupby(["as_of_year", geo_col, "state_code", "loan_purpose"])
        .agg(
            total_applications=("is_application", "sum"),
            total_originations=("is_originated", "sum"),
            avg_loan_amount_000s=("loan_amount_000s", "mean"),
            avg_income_000s=("applicant_income_000s", "mean"),
        )
        .reset_index()
    )
    agg["origination_rate"] = (agg["total_originations"] / agg["total_applications"]).round(4)
    log.info(f"  Aggregated shape: {agg.shape[0]:,} rows × {agg.shape[1]} columns")

    # STEP 7: SAVE
    log.info("\nSTEP 7: SAVE OUTPUT")
    df.to_parquet(out_path, compression="snappy", index=False)
    log.info(f"FILE SAVED: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    agg.to_parquet(agg_path, compression="snappy", index=False)
    log.info(f"FILE SAVED: {agg_path}  ({agg_path.stat().st_size / 1e6:.1f} MB)")

    # STEP 8: MEMORY MANAGEMENT
    log.info("\nSTEP 8: MEMORY MANAGEMENT")
    size_before_del = zip_path.stat().st_size / 1e6
    del df
    gc.collect()

    if delete_raw:
        zip_path.unlink()
        log.info(f"RAW FILE DELETED: {zip_path.name}  ({size_before_del:.0f} MB freed from disk)")
    else:
        log.info(f"RAW FILE KEPT: {zip_path.name}  (--no-delete flag set)")

    mem_after = mem_mb()
    log.info(f"Memory: {mem_before:.0f} MB → {mem_after:.0f} MB  "
             f"(Δ {mem_after - mem_before:+.0f} MB)")

    print_separator()
    log.info(f"  [{year}] COMPLETE\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="HMDA Pipeline 2007–2017")
    parser.add_argument("--years", type=int, nargs="+", default=YEARS)
    parser.add_argument("--no-delete", action="store_true",
                        help="Keep raw zip files after processing")
    args = parser.parse_args()

    years = sorted(args.years)
    log.info(f"\n{'═'*70}")
    log.info(f"  HMDA DATA ENGINEERING PIPELINE — {years[0]}–{years[-1]}")
    log.info(f"{'═'*70}")
    log.info(f"  Years to process: {years}")
    log.info(f"  Chunk size: {CHUNK_SIZE:,} rows")
    log.info(f"  Delete raw after processing: {not args.no_delete}")
    log.info("")

    results = {}
    total_start = time.time()
    for year in years:
        t = time.time()
        ok = process_year(year, delete_raw=not args.no_delete)
        results[year] = ("OK" if ok else "SKIPPED", round(time.time() - t))

    log.info(f"\n{'═'*70}")
    log.info("  PIPELINE SUMMARY")
    log.info(f"{'═'*70}")
    for year, (status, secs) in results.items():
        log.info(f"  {year}: {status}  ({secs}s)")
    log.info(f"\n  Total time: {(time.time()-total_start)/60:.1f} min")
    log.info(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
