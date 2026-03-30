# Statistical Significance Layer — Design Spec
**Date:** 2026-03-30
**Project:** HMDA Mortgage Market Analysis (2007–2017)
**Goal:** Prove or disprove statistical significance of every major finding from the data visualizations.

---

## Problem Statement

The HMDA dashboards reveal compelling visual patterns — application volumes dropped, LTI ratios rose, denial rates diverged by state. But visual patterns alone are not proof. This spec defines a statistical significance layer that rigorously tests each finding and presents results in plain English for a non-technical executive audience.

---

## Scope

6 findings to test:

1. Application volume drop over time
2. Approval rate structural break post-2008
3. LTI ratio increase 2007–2017
4. Denial rate differences across states
5. Recovery gap between top and bottom MSAs
6. Denial reason mix shift post-2008

---

## Architecture

### New File: `extract_significance.py`

Standalone Python script. Reads from existing `output/` parquets — no raw data re-download needed. Runs all statistical tests and writes one output file.

**Input files (already exist):**
- `output/viz_national.parquet` — national yearly aggregates
- `output/viz_state_denial.parquet` — state-level denial rates by year
- `output/viz_msa_recovery.parquet` — MSA recovery indices
- `output/viz_income_national.parquet` — LTI and income by year
- `output/viz_denial_reasons_national.parquet` — denial reason counts by year

**Output:** `output/viz_significance.parquet`

**Schema:**

| Column | Type | Example |
|---|---|---|
| `finding` | str | "Application Volume Drop" |
| `test_name` | str | "Linear Regression + Chow Test" |
| `confidence_pct` | float | 99.7 |
| `is_significant` | bool | True |
| `verdict` | str | "The drop in applications was not by chance. We are 99.7% confident." |
| `p_value` | float | 0.003 |
| `detail` | str | "Applications fell by ~890K per year (R²=0.81). Structural break detected at 2008." |

One row per finding (6 rows total).

---

## Statistical Tests

### 1. Application Volume Drop
- **Linear Regression** (time → total_applications)
  - Gives: slope (magnitude of drop per year), R², p-value
  - Answers: "How many applications were lost each year, and is that trend real?"
- **Chow Test**
  - Splits series at each candidate year (2008–2012), finds where F-statistic is maximised
  - Answers: "Exactly which year did the structural break happen?"

### 2. Approval Rate Post-2008 Structural Break
- **Interrupted Time Series (ITS)**
  - Models: approval_rate ~ time + post2008 + time_after_2008
  - Gives: level shift at 2008, pre-trend slope, post-trend slope
  - Answers: "Did 2008 cause a sudden shift AND a change in trajectory?"
- **Difference-in-means with trend control**
  - Compares mean approval rate pre-2008 vs post-2008 while controlling for the natural time trend
  - Answers: "Is the difference real after accounting for gradual drift?"

### 3. LTI Ratio Increase
- **Mann-Kendall Trend Test**
  - Non-parametric test for monotonic trend in LTI time series
  - Answers: "Is LTI consistently rising or could this be noise?"
- **OLS Regression: LTI ~ Time**
  - Gives: slope (LTI increase per year), R², p-value
  - Answers: "By how much does LTI rise each year, and is that statistically real?"

### 4. Denial Rate Differences Across States
- **One-way ANOVA**
  - Groups: all states, dependent variable: denial rate
  - Answers: "Are state denial rates genuinely different, or could they all come from the same distribution?"

### 5. Recovery Gap: Top vs Bottom MSAs
- **Mann-Whitney U Test**
  - Compares recovery index distribution of top 10 vs bottom 10 MSAs across all years
  - Non-parametric — does not assume normal distribution
  - Answers: "Are the top and bottom groups truly different populations?"

### 6. Denial Reason Mix Shift Post-2008
- **Chi-Square Test of Independence**
  - Contingency table: year group (pre/post 2008) × denial reason
  - Answers: "Did the mix of denial reasons change significantly after the crash?"

---

## Dashboard Changes

### A — Significance Badges on Existing Charts

Each relevant chart gets a small annotation box added via `fig.add_annotation()`:

```
🟢 STATISTICALLY SIGNIFICANT
99.7% confident — not by chance
```

Badge placement:

| Chart | Position |
|---|---|
| National Crash & Recovery | Top-left corner |
| Approval Rate line | Next to 2008 annotation |
| LTI bar chart | Top-right corner |
| State denial choropleth | Below title |
| MSA top/bottom bar chart | Above each subplot |
| Denial reasons stacked bar | Top-left corner |

Badge colour: green border for significant, red border for not significant.

### B — Statistical Proof Summary Panel

New section at the bottom of `viz_interactive.py`, added after Panel 6.

**Components:**
1. Summary table — one row per finding, showing: Finding | Confidence | 🟢/🔴 | Plain-English Verdict
2. Plain-English callout box per finding (using `st.success()` or `st.info()`)
3. Methodology note (collapsible `st.expander`) for anyone who wants the test names

---

## Implementation Steps

1. `pip install pymannkendall scipy statsmodels` — add to `requirements.txt`
2. Write `extract_significance.py`
3. Run script → verify `output/viz_significance.parquet` is correct
4. Add badges to `viz_interactive.py` charts
5. Add Statistical Proof Summary panel to bottom of `viz_interactive.py`
6. Push updated files to GitHub

---

## Success Criteria

- All 6 findings have a computed p-value and confidence percentage
- Every badge on every chart matches the significance result
- A non-technical viewer can read the proof panel and understand each finding without knowing what a p-value is
- Script runs in under 60 seconds (reads from parquets, no raw data)
