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
| `test_name` | str | "Linear Regression + Structural Break (2008)" |
| `confidence_pct` | float | 99.7 |
| `is_significant` | bool | True |
| `verdict` | str | "The drop in applications was not by chance. We are 99.7% confident." |
| `p_value` | float | 0.003 |
| `effect_size` | float | -890000.0 |
| `effect_size_label` | str | "applications lost per year (slope)" |
| `detail` | str | "Applications fell by ~890K per year (R²=0.81). Structural break confirmed at 2008." |

One row per finding (6 rows total).

**NaN fallback:** If any test produces a NaN p-value (e.g., due to zero variance or insufficient data), set `is_significant=False`, `confidence_pct=0.0`, `verdict="Insufficient data to test this finding."` — never let a NaN reach the dashboard.

---

## Statistical Tests

### 1. Application Volume Drop

**Derivation:** `total_loans` column from `viz_national.parquet` (all filed applications per year).

- **Linear Regression** (time → total_applications via `scipy.stats.linregress`)
  - Gives: slope (magnitude of drop per year), R², p-value
  - `effect_size` = slope (applications lost per year)
  - Answers: "How many applications were lost each year, and is that trend real?"

- **Structural Break at 2008** (indicator-variable OLS via `statsmodels`)
  - Model: `total_apps ~ year + post2008 + year_after2008`
  - `post2008` = 1 if year >= 2008, else 0
  - `year_after2008` = (year - 2008) * post2008
  - Fixed break at 2008 — do not scan candidate years (only 11 data points; scanning produces degenerate subseries with fewer than 4 points)
  - Gives: level shift coefficient at 2008, pre-trend slope, post-trend slope, p-value on the break indicator
  - Answers: "Did 2008 cause a sudden level shift in application volume?"

### 2. Approval Rate Post-2008 Structural Break

**Derivation:** `approval_rate = (total_apps - total_denied) / total_apps` — computed directly from `viz_national.parquet` columns `total_apps` and `total_denied`. `total_apps` = all filed applications (action_taken codes 1–8, every disposition). `total_denied` = applications where action_taken ∈ {3, 7} (explicitly denied by lender). This is the web-standard industry definition: denial rate = denied/all_filed, approval rate = 1 − denial rate. Withdrawn (4), incomplete (5), and purchased (6) applications remain in the denominator, consistent with how the dashboard was built.

- **Interrupted Time Series (ITS)** via `statsmodels.formula.api.ols`
  - Model: `approval_rate ~ time + post2008 + time_after_2008`
  - `time` = year - 2007 (0-indexed)
  - `post2008` = 1 if year >= 2008, else 0
  - `time_after_2008` = (year - 2008) * post2008
  - Gives: level shift at 2008, pre-trend slope, post-trend slope, p-value on break indicator
  - `effect_size` = level shift coefficient (percentage point change at 2008)
  - Answers: "Did 2008 cause a sudden shift AND a change in trajectory?"

- **Difference-in-means with trend control** (Welch's t-test, `scipy.stats.ttest_ind`)
  - Pre-2008 group: years 2007–2008; Post-2008 group: years 2009–2017
  - Controls for trend by detrending the series first (subtract linear fit from full series)
  - Answers: "Is the mean difference real after removing the natural time trend?"

### 3. LTI Ratio Increase

**Derivation:** `lti_median` column from `viz_income_national.parquet`.

- **Mann-Kendall Trend Test** via `pymannkendall.original_test`
  - Non-parametric test for monotonic trend
  - Note: LTI dips slightly in 2010–2012 before resuming upward — Mann-Kendall correctly accounts for this non-monotonic segment. Report Kendall's tau alongside p-value in `detail` to convey "strong but not perfectly linear."
  - Answers: "Is LTI consistently rising or could this be noise?"

- **OLS Regression: LTI ~ Time** via `scipy.stats.linregress`
  - Gives: slope (LTI increase per year), R², p-value
  - `effect_size` = slope (LTI units per year)
  - Answers: "By how much does LTI rise each year, and is that statistically real?"

### 4. Denial Rate Differences Across States

**Derivation:** Collapse `viz_state_denial.parquet` to one row per state by computing each state's mean `denial_rate` across all years. This gives 50 independent observations (US states only — exclude territories: PR, DC, GU, VI, etc. by keeping only rows where `state_abbr` is a standard 2-letter US state code from a fixed allowlist of 50 states).

- **One-way ANOVA** via `scipy.stats.f_oneway`
  - Groups: all 50 states, dependent variable: mean denial rate
  - Independent observations guaranteed by collapsing to per-state mean
  - Gives: F-statistic, p-value
  - `effect_size` = standard deviation of state means (spread across states)
  - Answers: "Are state denial rates genuinely different, or could they all come from the same distribution?"

### 5. Recovery Gap: Top vs Bottom MSAs

**Derivation:** `recovery_index` from `viz_msa_recovery.parquet`, year=2017. Top 10 = 10 highest recovery_index; Bottom 10 = 10 lowest.

- **Mann-Whitney U Test** via `scipy.stats.mannwhitneyu`
  - Compares recovery index distribution of top 10 vs bottom 10 MSAs
  - Non-parametric — does not assume normal distribution
  - `effect_size` = difference in median recovery indices (top median - bottom median)
  - Answers: "Are the top and bottom groups truly different populations?"

### 6. Denial Reason Mix Shift Post-2008

**Derivation:** `viz_denial_reasons_national.parquet` grouped by year and reason_label. Exclude reason codes 8 ("Mortgage insurance denied") and 9 ("Other") before building the contingency table — these categories are not interpretable as consistent bank policy signals.

- **Chi-Square Test of Independence** via `scipy.stats.chi2_contingency`
  - Contingency table: year group (pre-2008: 2007–2008 vs post-2008: 2009–2017) × denial reason (7 remaining codes)
  - Gives: chi-square statistic, p-value, degrees of freedom
  - `effect_size` = Cramér's V (standardised effect size for chi-square)
  - Answers: "Did the mix of denial reasons change significantly after the crash?"

---

## Dashboard Changes

### A — Significance Badges on Existing Charts

Each relevant chart gets a small annotation added via `fig.add_annotation()` using paper-relative coordinates (`xref="paper", yref="paper"`) to ensure correct positioning regardless of data range or slider state.

Badge format:
```
🟢 STATISTICALLY SIGNIFICANT
99.7% confident — not by chance
```
Green border for significant (`is_significant=True`), red border for not significant.

Badge placement:

| Chart | `x` | `y` | `xanchor` |
|---|---|---|---|
| National Crash & Recovery | 0.01 | 0.97 | "left" |
| Approval Rate line | 0.01 | 0.85 | "left" |
| LTI bar chart | 0.99 | 0.97 | "right" |
| State denial choropleth | 0.01 | 0.97 | "left" |
| MSA top/bottom bar charts | 0.01 | 0.97 | "left" |
| Denial reasons stacked bar | 0.01 | 0.97 | "left" |

### B — Statistical Proof Summary Panel

New section at the bottom of `viz_interactive.py`, added after Panel 6.

**Components:**
1. Summary table — one row per finding: Finding | Confidence | 🟢/🔴 | Plain-English Verdict
2. `st.success()` callout per significant finding, `st.error()` per non-significant finding
3. `st.expander("Methodology")` — collapsible section listing test names and package versions for technical readers

---

## Dependencies

Add to `requirements.txt` (pinned):
```
scipy==1.13.0
statsmodels==0.14.1
pymannkendall==1.4.3
```

---

## Implementation Steps

1. Add pinned dependencies to `requirements.txt` and run `pip install`
2. Write `extract_significance.py`
3. Run script and verify `output/viz_significance.parquet`:
   - Exactly 6 rows
   - No null values in `p_value`, `confidence_pct`, `is_significant`, `verdict`
   - All `confidence_pct` values between 0 and 100
   - `is_significant` matches `p_value < 0.05` for each row
   - `effect_size` is a finite float (not NaN) for all rows
4. Add significance badges to `viz_interactive.py` charts
5. Add Statistical Proof Summary panel to bottom of `viz_interactive.py`
6. Push updated files to GitHub

---

## Success Criteria

- All 6 findings have a computed p-value and confidence percentage
- Every badge on every chart matches the significance result
- A non-technical viewer can read the proof panel and understand each finding without knowing what a p-value is
- Script runs in under 60 seconds (reads from parquets, no raw data)
- No NaN values reach the dashboard under any circumstance
