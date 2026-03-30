# tests/test_significance.py
import pandas as pd
import math
from pathlib import Path

SIG_PATH = Path("output/viz_significance.parquet")


def test_parquet_exists():
    assert SIG_PATH.exists(), "viz_significance.parquet not found — run extract_significance.py first"


def test_exactly_six_rows():
    df = pd.read_parquet(SIG_PATH)
    assert len(df) == 6, f"Expected 6 rows, got {len(df)}"


def test_no_nulls_in_required_columns():
    df = pd.read_parquet(SIG_PATH)
    # p_value intentionally stores float("nan") in the NaN fallback path — exclude it here
    for col in ["confidence_pct", "is_significant", "verdict", "effect_size_label", "detail"]:
        assert df[col].notna().all(), f"Null found in column: {col}"


def test_confidence_pct_in_range():
    df = pd.read_parquet(SIG_PATH)
    assert (df["confidence_pct"] >= 0).all() and (df["confidence_pct"] <= 100).all(), \
        "confidence_pct out of [0, 100] range"


def test_is_significant_matches_p_value():
    df = pd.read_parquet(SIG_PATH)
    # NaN p_values get is_significant=False via safe_row fallback
    for _, row in df.iterrows():
        if math.isfinite(row["p_value"]):
            expected = row["p_value"] < 0.05
            assert row["is_significant"] == expected, \
                f"Mismatch for {row['finding']}: p={row['p_value']}, is_significant={row['is_significant']}"


def test_effect_size_is_finite():
    df = pd.read_parquet(SIG_PATH)
    for _, row in df.iterrows():
        assert math.isfinite(row["effect_size"]), \
            f"Non-finite effect_size for {row['finding']}: {row['effect_size']}"
