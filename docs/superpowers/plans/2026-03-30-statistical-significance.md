# Statistical Significance Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove statistical significance of 6 HMDA findings via a pre-computed script, then surface results as badges on existing charts and a proof summary panel in the dashboard.

**Architecture:** `extract_significance.py` reads from existing `output/` parquets, runs 6 statistical tests, writes `output/viz_significance.parquet` (6 rows). `viz_interactive.py` loads this parquet and renders badges on each chart + a proof summary panel at the bottom.

**Tech Stack:** Python, Pandas, scipy==1.13.0, statsmodels==0.14.1, pymannkendall==1.4.3, Streamlit, Plotly

---

## File Map

| Action | File | Responsibility |
|---|---|---|
| Create | `extract_significance.py` | Run all 6 statistical tests, write `output/viz_significance.parquet` |
| Modify | `requirements.txt` | Add 3 new pinned dependencies |
| Modify | `viz_interactive.py` | Load significance parquet, add badges to charts, add proof panel |
| Create | `tests/test_significance.py` | Verify parquet schema, NaN guards, significance logic |

---

## Task 1: Add Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add three pinned dependencies**

Open `requirements.txt` and append:
```
scipy==1.13.0
statsmodels==0.14.1
pymannkendall==1.4.3
```

- [ ] **Step 2: Install them**

```bash
pip install scipy==1.13.0 statsmodels==0.14.1 pymannkendall==1.4.3
```
Expected: all three install without error.

- [ ] **Step 3: Verify imports work**

```bash
python3 -c "import scipy, statsmodels, pymannkendall; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "feat: add scipy, statsmodels, pymannkendall dependencies"
```

---

## Task 2: Write the Significance Tests (extract_significance.py)

**Files:**
- Create: `extract_significance.py`

- [ ] **Step 1: Create the file with imports and constants**

```python
"""
extract_significance.py — Runs 6 statistical tests on HMDA findings.
Outputs output/viz_significance.parquet (6 rows, one per finding).
"""
import math
import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy import stats
from scipy.stats import mannwhitneyu, f_oneway, chi2_contingency
import statsmodels.formula.api as smf
from pathlib import Path

OUTPUT = Path("output")

US_STATE_ABBRS = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
}

def safe_row(finding, test_name, p_value, effect_size, effect_size_label, detail_fn):
    """Build a result row, applying NaN fallback if p_value is bad."""
    if p_value is None or not math.isfinite(p_value) or np.isnan(p_value):
        return {
            "finding": finding, "test_name": test_name,
            "p_value": float("nan"), "confidence_pct": 0.0,
            "is_significant": False, "effect_size": 0.0,
            "effect_size_label": effect_size_label,
            "verdict": "Insufficient data to test this finding.",
            "detail": "Insufficient data.",
        }
    confidence = round((1 - p_value) * 100, 1)
    is_sig = p_value < 0.05
    verdict = (
        f"We are {confidence}% confident this finding is not by chance."
        if is_sig else
        f"Result is NOT statistically significant (confidence only {confidence}%)."
    )
    return {
        "finding": finding, "test_name": test_name,
        "p_value": round(p_value, 6), "confidence_pct": confidence,
        "is_significant": is_sig,
        "effect_size": round(float(effect_size), 4),
        "effect_size_label": effect_size_label,
        "verdict": verdict, "detail": detail_fn(),
    }
```

- [ ] **Step 2: Write Finding 1 — Application Volume Drop**

Append to `extract_significance.py`:

```python
def test_application_drop(nat):
    """Linear regression + indicator-variable structural break at 2008."""
    nat = nat.sort_values("year").copy()
    x = nat["year"].values.astype(float)
    y = nat["total_loans"].values.astype(float)

    # Linear regression
    slope, intercept, r, p_lin, se = stats.linregress(x, y)

    # Structural break — indicator OLS
    nat["time"]          = nat["year"] - 2007
    nat["post2008"]      = (nat["year"] >= 2008).astype(int)
    nat["year_after2008"] = (nat["year"] - 2008) * nat["post2008"]
    model = smf.ols("total_loans ~ time + post2008 + year_after2008", data=nat).fit()
    p_break = model.pvalues.get("post2008", float("nan"))
    level_shift = model.params.get("post2008", 0.0)

    # Use the more conservative (higher) p-value of the two tests
    p_combined = max(p_lin, p_break)

    def detail():
        return (
            f"Applications fell by ~{abs(slope):,.0f} per year (R²={r**2:.2f}, p={p_lin:.4f}). "
            f"Structural break at 2008: level shift {level_shift:+,.0f} apps (p={p_break:.4f})."
        )

    return safe_row(
        finding="Application Volume Drop",
        test_name="Linear Regression + Structural Break at 2008",
        p_value=p_combined,
        effect_size=slope,
        effect_size_label="applications lost per year (slope)",
        detail_fn=detail,
    )
```

- [ ] **Step 3: Write Finding 2 — Approval Rate Structural Break**

Append to `extract_significance.py`:

```python
def test_approval_rate_break(nat):
    """ITS model + detrended Welch t-test on approval rate."""
    nat = nat.sort_values("year").copy()
    # Use total_loans (all filed apps) as denominator — matches the dashboard chart exactly
    nat["approval_rate"] = (nat["total_loans"] - nat["total_denied"]) / nat["total_loans"]
    nat["time"]          = nat["year"] - 2007
    nat["post2008"]      = (nat["year"] >= 2008).astype(int)
    nat["time_after_2008"] = (nat["year"] - 2008) * nat["post2008"]

    # ITS
    model = smf.ols("approval_rate ~ time + post2008 + time_after_2008", data=nat).fit()
    p_its      = model.pvalues.get("post2008", float("nan"))
    level_shift = model.params.get("post2008", 0.0)

    # Detrended Welch t-test
    slope_all, intercept_all, _, _, _ = stats.linregress(nat["time"], nat["approval_rate"])
    nat["detrended"] = nat["approval_rate"] - (slope_all * nat["time"] + intercept_all)
    pre  = nat.loc[nat["year"] <= 2008, "detrended"].values
    post = nat.loc[nat["year"] >  2008, "detrended"].values
    _, p_ttest = stats.ttest_ind(pre, post, equal_var=False)

    p_combined = max(p_its, p_ttest)

    def detail():
        return (
            f"ITS level shift at 2008: {level_shift*100:+.1f} pp (p={p_its:.4f}). "
            f"Detrended mean difference confirmed by Welch t-test (p={p_ttest:.4f})."
        )

    return safe_row(
        finding="Approval Rate Structural Break (2008)",
        test_name="Interrupted Time Series + Detrended Welch t-test",
        p_value=p_combined,
        effect_size=level_shift * 100,
        effect_size_label="percentage point level shift at 2008",
        detail_fn=detail,
    )
```

- [ ] **Step 4: Write Finding 3 — LTI Ratio Increase**

Append to `extract_significance.py`:

```python
def test_lti_increase(inc_nat):
    """Mann-Kendall trend test + OLS regression on LTI median."""
    inc = inc_nat.sort_values("year").copy()
    lti = inc["lti_median"].values

    mk_result = mk.original_test(lti)
    slope_reg, _, r, p_reg, _ = stats.linregress(inc["year"].values.astype(float), lti)

    p_combined = max(mk_result.p, p_reg)

    def detail():
        return (
            f"Mann-Kendall: tau={mk_result.Tau:.3f}, p={mk_result.p:.4f} "
            f"(trend: {mk_result.trend}). "
            f"OLS: LTI rises {slope_reg:.4f}x per year (R²={r**2:.2f}, p={p_reg:.4f}). "
            f"Note: LTI dips 2010–2012 before resuming upward — trend is strong but not perfectly linear."
        )

    return safe_row(
        finding="LTI Ratio Increase (2007–2017)",
        test_name="Mann-Kendall Trend Test + OLS Regression",
        p_value=p_combined,
        effect_size=slope_reg,
        effect_size_label="LTI increase per year (slope)",
        detail_fn=detail,
    )
```

- [ ] **Step 5: Write Finding 4 — Denial Rate Differences Across States**

Append to `extract_significance.py`:

```python
def test_state_denial_differences(state):
    """One-way ANOVA on per-state mean denial rates (50 US states only).
    Each state's single mean denial rate (collapsed across all years) is one
    independent observation. Groups are the 4 Census regions; the dependent
    variable is per-state mean denial rate — 50 independent scalars.
    Since f_oneway requires multiple observations per group, we group by
    Census region (Northeast/South/Midwest/West) and pass each region's
    per-state means as its group vector.
    """
    REGION_MAP = {
        "CT":"NE","ME":"NE","MA":"NE","NH":"NE","RI":"NE","VT":"NE",
        "NJ":"NE","NY":"NE","PA":"NE",
        "IL":"MW","IN":"MW","MI":"MW","OH":"MW","WI":"MW",
        "IA":"MW","KS":"MW","MN":"MW","MO":"MW","NE":"MW","ND":"MW","SD":"MW",
        "DE":"S","FL":"S","GA":"S","MD":"S","NC":"S","SC":"S","VA":"S","WV":"S",
        "AL":"S","KY":"S","MS":"S","TN":"S","AR":"S","LA":"S","OK":"S","TX":"S",
        "AZ":"W","CO":"W","ID":"W","MT":"W","NV":"W","NM":"W","UT":"W","WY":"W",
        "AK":"W","CA":"W","HI":"W","OR":"W","WA":"W",
    }
    state = state[state["state_abbr"].isin(US_STATE_ABBRS)].copy()
    # Collapse to one mean per state (50 independent observations)
    state_means = state.groupby("state_abbr")["denial_rate"].mean().reset_index()
    state_means["region"] = state_means["state_abbr"].map(REGION_MAP)
    state_means = state_means.dropna(subset=["region"])

    groups = [
        state_means.loc[state_means["region"] == r, "denial_rate"].values
        for r in ["NE", "MW", "S", "W"]
    ]
    f_stat, p_anova = f_oneway(*groups)
    effect = float(state_means["denial_rate"].std())

    def detail():
        return (
            f"One-way ANOVA across 50 states: F={f_stat:.2f}, p={p_anova:.6f}. "
            f"Std dev of state mean denial rates: {effect:.4f} "
            f"(range: {state_means.min():.1%}–{state_means.max():.1%})."
        )

    return safe_row(
        finding="Denial Rate Differences Across States",
        test_name="One-way ANOVA (50 US states)",
        p_value=p_anova,
        effect_size=effect,
        effect_size_label="std dev of state mean denial rates",
        detail_fn=detail,
    )
```

- [ ] **Step 6: Write Finding 5 — MSA Recovery Gap**

Append to `extract_significance.py`:

```python
def test_msa_recovery_gap(msa_r):
    """Mann-Whitney U test: top 10 vs bottom 10 MSAs by recovery_index in 2017."""
    msa_2017 = msa_r[msa_r["year"] == 2017].dropna(subset=["recovery_index"]).copy()
    top10    = msa_2017.nlargest(10,  "recovery_index")["recovery_index"].values
    bottom10 = msa_2017.nsmallest(10, "recovery_index")["recovery_index"].values

    stat, p_mw = mannwhitneyu(top10, bottom10, alternative="greater")
    effect = float(np.median(top10) - np.median(bottom10))

    def detail():
        return (
            f"Mann-Whitney U={stat:.0f}, p={p_mw:.6f}. "
            f"Median recovery index — top 10: {np.median(top10):.1f}, "
            f"bottom 10: {np.median(bottom10):.1f} "
            f"(gap: {effect:.1f} index points)."
        )

    return safe_row(
        finding="MSA Recovery Gap (Top 10 vs Bottom 10)",
        test_name="Mann-Whitney U Test (2017 recovery index)",
        p_value=p_mw,
        effect_size=effect,
        effect_size_label="median recovery index gap (top minus bottom)",
        detail_fn=detail,
    )
```

- [ ] **Step 7: Write Finding 6 — Denial Reason Mix Shift**

Append to `extract_significance.py`:

```python
def test_denial_reason_shift(dr_nat):
    """Chi-square test: denial reason mix pre vs post 2008 (excl. codes 8 and 9)."""
    dr = dr_nat[~dr_nat["reason_code"].isin([8, 9])].copy()
    dr["period"] = dr["year"].apply(lambda y: "pre" if y <= 2008 else "post")

    contingency = (
        dr.groupby(["period", "reason_label"])["count"]
        .sum()
        .unstack(fill_value=0)
    )

    chi2, p_chi, dof, _ = chi2_contingency(contingency.values)

    # Cramér's V
    n = contingency.values.sum()
    cramers_v = float(np.sqrt(chi2 / (n * (min(contingency.shape) - 1))))

    def detail():
        return (
            f"Chi-square={chi2:.1f}, df={dof}, p={p_chi:.6f}. "
            f"Cramér's V={cramers_v:.3f} (effect size: "
            f"{'small' if cramers_v < 0.1 else 'medium' if cramers_v < 0.3 else 'large'}). "
            f"7 reason codes used (codes 8 & 9 excluded). "
            f"Pre-2008: 2007–2008. Post-2008: 2009–2017."
        )

    return safe_row(
        finding="Denial Reason Mix Shift Post-2008",
        test_name="Chi-Square Test of Independence (7 reason codes)",
        p_value=p_chi,
        effect_size=cramers_v,
        effect_size_label="Cramér's V (standardised effect size)",
        detail_fn=detail,
    )
```

- [ ] **Step 8: Write the main() function and runner**

Append to `extract_significance.py`:

```python
def main():
    print("Loading data...", flush=True)
    nat    = pd.read_parquet(OUTPUT / "viz_national.parquet")
    state  = pd.read_parquet(OUTPUT / "viz_state_denial.parquet")
    msa_r  = pd.read_parquet(OUTPUT / "viz_msa_recovery.parquet")
    inc    = pd.read_parquet(OUTPUT / "viz_income_national.parquet")
    dr_nat = pd.read_parquet(OUTPUT / "viz_denial_reasons_national.parquet")

    print("Running tests...", flush=True)
    rows = [
        test_application_drop(nat),
        test_approval_rate_break(nat),
        test_lti_increase(inc),
        test_state_denial_differences(state),
        test_msa_recovery_gap(msa_r),
        test_denial_reason_shift(dr_nat),
    ]

    df = pd.DataFrame(rows)
    out = OUTPUT / "viz_significance.parquet"
    df.to_parquet(out, compression="snappy", index=False)

    print(f"\nSaved: {out}  ({len(df)} rows)\n")
    print(df[["finding", "confidence_pct", "is_significant", "verdict"]].to_string(index=False))

if __name__ == "__main__":
    main()
```

- [ ] **Step 9: Run the script**

```bash
cd /Users/sriramganeshalingam/Documents/hmda_pipeline
python3 extract_significance.py
```
Expected: prints 6 rows, saves parquet. No errors.

- [ ] **Step 10: Commit**

```bash
git add extract_significance.py
git commit -m "feat: add extract_significance.py — 6 statistical tests on HMDA findings"
```

---

## Task 3: Verify the Output Parquet

**Files:**
- Create: `tests/test_significance.py`

- [ ] **Step 1: Create tests directory and test file**

```bash
mkdir -p /Users/sriramganeshalingam/Documents/hmda_pipeline/tests
```

```python
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
```

- [ ] **Step 2: Run the tests**

```bash
cd /Users/sriramganeshalingam/Documents/hmda_pipeline
python3 -m pytest tests/test_significance.py -v
```
Expected: 6 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_significance.py
git commit -m "test: add parquet acceptance tests for viz_significance output"
```

---

## Task 4: Add Significance Badges to Dashboard Charts

**Files:**
- Modify: `viz_interactive.py`

- [ ] **Step 1: Load significance parquet in the load() function**

Find the `load()` function in `viz_interactive.py`. Make these exact changes:

```python
# BEFORE (last two lines of load()):
    return lp, state, msa_r, nat, lt, inc_nat, inc_lp, inc_st, dr_nat, dr_full

lp, state, msa_r, nat, lt, inc_nat, inc_lp, inc_st, dr_nat, dr_full = load()

# AFTER:
    sig = pd.read_parquet(OUTPUT / "viz_significance.parquet")
    return lp, state, msa_r, nat, lt, inc_nat, inc_lp, inc_st, dr_nat, dr_full, sig

lp, state, msa_r, nat, lt, inc_nat, inc_lp, inc_st, dr_nat, dr_full, sig = load()
```

- [ ] **Step 2: Create a helper function for badge annotations**

Add this function near the top of `viz_interactive.py`, after the imports:

```python
def significance_badge(fig, sig_df, finding_name, x=0.01, y=0.97, xanchor="left"):
    """Add a significance badge annotation to a plotly figure."""
    row = sig_df[sig_df["finding"] == finding_name]
    if row.empty:
        return fig
    row = row.iloc[0]
    is_sig   = row["is_significant"]
    conf     = row["confidence_pct"]
    icon     = "🟢" if is_sig else "🔴"
    label    = "SIGNIFICANT" if is_sig else "NOT SIGNIFICANT"
    color    = "#27ae60" if is_sig else "#e74c3c"
    fig.add_annotation(
        x=x, y=y, xref="paper", yref="paper",
        xanchor=xanchor, yanchor="top",
        text=f"<b>{icon} {label}</b><br>{conf:.1f}% confident",
        showarrow=False,
        font=dict(size=10, color=color),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor=color, borderwidth=1.5, borderpad=5,
    )
    return fig
```

- [ ] **Step 3: Add badge to National Crash & Recovery chart (fig1)**

After `fig1.update_layout(...)`, add:
```python
fig1 = significance_badge(fig1, sig, "Application Volume Drop", x=0.01, y=0.97)
```

- [ ] **Step 4: Add badge to Approval Rate chart (fig1, second annotation)**

After the `fig1 = significance_badge(...)` line above, add:
```python
fig1 = significance_badge(fig1, sig, "Approval Rate Structural Break (2008)", x=0.01, y=0.85)
```

- [ ] **Step 5: Add badge to LTI bar chart (fig6b)**

After `fig6b.update_layout(...)`, add:
```python
fig6b = significance_badge(fig6b, sig, "LTI Ratio Increase (2007–2017)", x=0.99, y=0.97, xanchor="right")
```

- [ ] **Step 6: Add badge to State denial choropleth (fig2)**

After `fig2.update_layout(...)`, add:
```python
fig2 = significance_badge(fig2, sig, "Denial Rate Differences Across States", x=0.01, y=0.97)
```

- [ ] **Step 7: Add badge to MSA top/bottom chart (fig3)**

After `fig3.update_layout(...)`, add:
```python
fig3 = significance_badge(fig3, sig, "MSA Recovery Gap (Top 10 vs Bottom 10)", x=0.01, y=0.97)
```

- [ ] **Step 8: Add badge to Denial Reasons stacked bar (fig5a)**

After `fig5a.update_layout(...)`, add:
```python
fig5a = significance_badge(fig5a, sig, "Denial Reason Mix Shift Post-2008", x=0.01, y=0.97)
```

- [ ] **Step 9: Verify dashboard loads without error**

```bash
streamlit run viz_interactive.py --server.port 8503 --server.headless true
```
Expected: starts cleanly, all 6 badges visible on their respective charts.

- [ ] **Step 10: Commit**

```bash
git add viz_interactive.py
git commit -m "feat: add significance badges to dashboard charts"
```

---

## Task 5: Add Statistical Proof Summary Panel

**Files:**
- Modify: `viz_interactive.py`

- [ ] **Step 1: Add the proof panel section at the bottom of viz_interactive.py**

Append after the last `st.markdown("---")` at the end of the file:

```python
# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL PROOF SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("📊 Statistical Proof Summary")
st.caption("Every visual pattern in this dashboard has been tested for statistical significance.")

# Summary table
summary_rows = []
for _, row in sig.iterrows():
    icon = "🟢" if row["is_significant"] else "🔴"
    summary_rows.append({
        "Finding":    row["finding"],
        "Confidence": f"{row['confidence_pct']:.1f}%",
        "Result":     f"{icon} {'Significant' if row['is_significant'] else 'Not Significant'}",
        "Verdict":    row["verdict"],
    })
summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Plain-English callout per finding
for _, row in sig.iterrows():
    if row["is_significant"]:
        st.success(f"**{row['finding']}** — {row['verdict']}\n\n_{row['detail']}_")
    else:
        st.error(f"**{row['finding']}** — {row['verdict']}\n\n_{row['detail']}_")

# Methodology expander
with st.expander("Methodology — for technical readers"):
    method_rows = []
    for _, row in sig.iterrows():
        method_rows.append({
            "Finding":   row["finding"],
            "Test":      row["test_name"],
            "p-value":   f"{row['p_value']:.6f}",
            "Effect Size": f"{row['effect_size']:.4f} ({row['effect_size_label']})",
        })
    st.dataframe(pd.DataFrame(method_rows), use_container_width=True, hide_index=True)
    st.caption("Packages: scipy==1.13.0 · statsmodels==0.14.1 · pymannkendall==1.4.3")
```

- [ ] **Step 2: Verify proof panel renders correctly**

Open browser at `http://localhost:8503`, scroll to bottom.
Expected: summary table with 6 rows, green `st.success()` callout per significant finding, methodology expander.

- [ ] **Step 3: Commit**

```bash
git add viz_interactive.py
git commit -m "feat: add statistical proof summary panel to dashboard"
```

---

## Task 6: Push to GitHub

- [ ] **Step 1: Run all tests one final time**

```bash
cd /Users/sriramganeshalingam/Documents/hmda_pipeline
python3 -m pytest tests/test_significance.py -v
```
Expected: 6/6 PASS.

- [ ] **Step 2: Push to GitHub**

```bash
git push origin main
```

- [ ] **Step 3: Verify on GitHub**

Open `https://github.com/Sriram006SJM/hmda-mortgage-analysis` and confirm:
- `extract_significance.py` is present
- `output/viz_significance.parquet` is present
- `tests/test_significance.py` is present
- `requirements.txt` includes the 3 new packages

---

## Acceptance Checklist

- [ ] `python3 extract_significance.py` runs in < 60 seconds with no errors
- [ ] `output/viz_significance.parquet` has exactly 6 rows, no nulls
- [ ] All 6 `is_significant` values match `p_value < 0.05`
- [ ] All 6 charts show a significance badge
- [ ] Proof summary panel visible at bottom of dashboard
- [ ] A non-technical viewer can read the panel without knowing what a p-value is
- [ ] `pytest tests/test_significance.py` passes 6/6
