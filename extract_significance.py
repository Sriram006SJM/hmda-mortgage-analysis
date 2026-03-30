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


def test_application_drop(nat):
    """Linear regression + indicator-variable structural break at 2008."""
    nat = nat.sort_values("year").copy()
    x = nat["year"].values.astype(float)
    y = nat["total_loans"].values.astype(float)

    # Linear regression
    slope, intercept, r, p_lin, se = stats.linregress(x, y)

    # Structural break — indicator OLS
    nat["time"]           = nat["year"] - 2007
    nat["post2008"]       = (nat["year"] >= 2008).astype(int)
    nat["year_after2008"] = (nat["year"] - 2008) * nat["post2008"]
    model   = smf.ols("total_loans ~ time + post2008 + year_after2008", data=nat).fit()
    p_break = model.pvalues.get("post2008", float("nan"))
    level_shift = model.params.get("post2008", 0.0)

    # Use the more conservative (higher) p-value of the two tests
    p_combined = max(p_lin, p_break)

    def detail():
        return (
            f"Applications fell by ~{abs(slope):,.0f} per year (R\u00b2={r**2:.2f}, p={p_lin:.4f}). "
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


def test_approval_rate_break(nat):
    """ITS model + detrended Welch t-test on approval rate."""
    nat = nat.sort_values("year").copy()
    # Use total_loans (all filed apps) as denominator — matches the dashboard chart exactly
    nat["approval_rate"]   = (nat["total_loans"] - nat["total_denied"]) / nat["total_loans"]
    nat["time"]            = nat["year"] - 2007
    nat["post2008"]        = (nat["year"] >= 2008).astype(int)
    nat["time_after_2008"] = (nat["year"] - 2008) * nat["post2008"]

    # ITS
    model      = smf.ols("approval_rate ~ time + post2008 + time_after_2008", data=nat).fit()
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
            f"OLS: LTI rises {slope_reg:.4f}x per year (R\u00b2={r**2:.2f}, p={p_reg:.4f}). "
            f"Note: LTI dips 2010\u20132012 before resuming upward \u2014 trend is strong but not perfectly linear."
        )

    return safe_row(
        finding="LTI Ratio Increase (2007\u20132017)",
        test_name="Mann-Kendall Trend Test + OLS Regression",
        p_value=p_combined,
        effect_size=slope_reg,
        effect_size_label="LTI increase per year (slope)",
        detail_fn=detail,
    )


def test_state_denial_differences(state):
    """One-way ANOVA on per-state mean denial rates (50 US states only).
    Each state's single mean denial rate (collapsed across all years) is one
    independent observation. Groups are the 4 Census regions; the dependent
    variable is per-state mean denial rate — 50 independent scalars.
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
        min_dr = state_means["denial_rate"].min()
        max_dr = state_means["denial_rate"].max()
        return (
            f"One-way ANOVA across 50 states: F={f_stat:.2f}, p={p_anova:.6f}. "
            f"Std dev of state mean denial rates: {effect:.4f} "
            f"(range: {min_dr:.1%}\u2013{max_dr:.1%})."
        )

    return safe_row(
        finding="Denial Rate Differences Across States",
        test_name="One-way ANOVA (50 US states)",
        p_value=p_anova,
        effect_size=effect,
        effect_size_label="std dev of state mean denial rates",
        detail_fn=detail,
    )


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
            f"Median recovery index \u2014 top 10: {np.median(top10):.1f}, "
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
            f"Cram\u00e9r's V={cramers_v:.3f} (effect size: "
            f"{'small' if cramers_v < 0.1 else 'medium' if cramers_v < 0.3 else 'large'}). "
            f"7 reason codes used (codes 8 & 9 excluded). "
            f"Pre-2008: 2007\u20132008. Post-2008: 2009\u20132017."
        )

    return safe_row(
        finding="Denial Reason Mix Shift Post-2008",
        test_name="Chi-Square Test of Independence (7 reason codes)",
        p_value=p_chi,
        effect_size=cramers_v,
        effect_size_label="Cramér's V (standardised effect size)",
        detail_fn=detail,
    )


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
