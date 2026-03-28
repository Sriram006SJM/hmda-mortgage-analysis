"""
config.py — HMDA Pipeline Configuration
Source: https://www.consumerfinance.gov/data-research/hmda/historic-data/
        ?geo=nationwide&records=all-records&field_descriptions=labels
"""

from pathlib import Path

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
RAW_DIR    = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR    = BASE_DIR / "logs"

for d in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, LOG_DIR):
    d.mkdir(exist_ok=True)

# ── Years ─────────────────────────────────────────────────────────────────────
YEARS = list(range(2007, 2018))   # 2007–2017 inclusive

# ── Download URL template ─────────────────────────────────────────────────────
URL_TEMPLATE = (
    "https://files.consumerfinance.gov/hmda-historic-loan-data/"
    "hmda_{year}_nationwide_all-records_labels.zip"
)

def raw_zip(year: int) -> Path:
    return RAW_DIR / f"hmda_{year}_nationwide_all-records_labels.zip"

def csv_name(year: int) -> str:
    return f"hmda_{year}_nationwide_all-records_labels.csv"

def parquet_path(year: int) -> Path:
    return PROCESSED_DIR / f"hmda_cleaned_{year}.parquet"

# ── Chunk size for streaming CSV reads ───────────────────────────────────────
CHUNK_SIZE = 500_000     # rows per chunk

# ── Column mapping — historic names → standard names ─────────────────────────
# Historic HMDA (2007-2017) uses slightly different column names than post-2018
COLUMN_MAP = {
    "loan_amount_000s": "loan_amount_000s",   # kept as-is (in $000s)
    "applicant_income_000s": "applicant_income_000s",
    "msamd": "msa_md",
    "msamd_name": "msa_md_name",
}

# ── Required columns (must be present; row dropped if null) ───────────────────
REQUIRED_COLS = [
    "as_of_year",
    "action_taken",
    "loan_purpose",
    "loan_type",
    "loan_amount_000s",
    "state_code",
]

# ── Core columns to keep (required + optional) ────────────────────────────────
CORE_COLS = [
    # Required
    "as_of_year",
    "action_taken",
    "loan_purpose",
    "loan_type",
    "loan_amount_000s",
    "applicant_income_000s",
    "msamd",           # will be renamed to msa_md
    "state_code",
    "county_code",
    # Optional
    "owner_occupancy",
    "property_type",
    "lien_status",
    "purchaser_type",
    # Label columns (human-readable)
    "action_taken_name",
    "loan_purpose_name",
    "loan_type_name",
    "msamd_name",
    "state_name",
    "county_name",
]

# ── Numeric columns to cast ───────────────────────────────────────────────────
NUMERIC_COLS = [
    "loan_amount_000s",
    "applicant_income_000s",
    "msamd",
    "state_code",
    "county_code",
    "action_taken",
    "loan_purpose",
    "loan_type",
    "owner_occupancy",
    "property_type",
    "lien_status",
    "purchaser_type",
]

# ── action_taken = 1 → originated ────────────────────────────────────────────
ORIGINATED_CODE = 1
