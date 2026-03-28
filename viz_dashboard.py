"""
viz_dashboard.py — HMDA Market Crash & Recovery Dashboard (2007–2017)

4 panels:
  1. Crash & Recovery: Applications vs Originations (annotated)
  2. Geographic: Denial Rate by State (choropleth + time slider)
  3. Government Backstop: Conventional vs Govt loan-type share by year + loan purpose
  4. MSA Recovery Index: city-level recovery relative to 2007 baseline

Usage:
    streamlit run viz_dashboard.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

OUTPUT = Path("output")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HMDA Market Crash & Recovery (2007–2017)",
    page_icon="🏠",
    layout="wide",
)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    nat     = pd.read_parquet(OUTPUT / "viz_national.parquet")
    state   = pd.read_parquet(OUTPUT / "viz_state_denial.parquet")
    lt      = pd.read_parquet(OUTPUT / "viz_loan_type.parquet")
    lp      = pd.read_parquet(OUTPUT / "viz_loan_purpose.parquet")
    msa_r   = pd.read_parquet(OUTPUT / "viz_msa_recovery.parquet")
    dr_nat  = pd.read_parquet(OUTPUT / "viz_denial_reasons_national.parquet")
    dr_full = pd.read_parquet(OUTPUT / "viz_denial_reasons.parquet")
    inc_nat = pd.read_parquet(OUTPUT / "viz_income_national.parquet")
    inc_lp  = pd.read_parquet(OUTPUT / "viz_income_purpose.parquet")
    inc_st  = pd.read_parquet(OUTPUT / "viz_income_state.parquet")
    return nat, state, lt, lp, msa_r, dr_nat, dr_full, inc_nat, inc_lp, inc_st

nat, state, lt, lp, msa_r, dr_nat, dr_full, inc_nat, inc_lp, inc_st = load_data()

YEARS = sorted(nat["year"].unique())
BOTTOM_YEAR = int(nat.loc[nat["total_originated"].idxmin(), "year"])

# ── Freddie Mac PMMS 30-year fixed rate annual averages (public data) ──
MORTGAGE_RATES = pd.DataFrame({
    "year":       list(range(2007, 2018)),
    "rate_30yr":  [6.34, 6.03, 5.04, 4.69, 4.45, 3.66, 3.98, 4.17, 3.85, 3.65, 3.99],
    # Federal Funds Effective Rate annual averages — FRED (FEDFUNDS)
    "fed_funds":  [5.02, 1.93, 0.24, 0.18, 0.10, 0.14, 0.11, 0.09, 0.13, 0.40, 1.00],
})
INFLECTION_YEAR = int(
    nat[nat["year"] > BOTTOM_YEAR]
    .assign(yoy=lambda d: d["total_originated"].pct_change())
    .query("yoy > 0")
    .iloc[0]["year"]
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🏠 HMDA Mortgage Market: Crash & Recovery (2007–2017)")
st.caption(
    "Source: CFPB / FFIEC Home Mortgage Disclosure Act historic data  •  "
    f"Crash bottom: **{BOTTOM_YEAR}**  •  Recovery inflection: **{INFLECTION_YEAR}**"
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — Crash & Recovery National Trend
# ══════════════════════════════════════════════════════════════════════════════
st.header("1. National Crash & Recovery")

# Web-standard: approved out of total_loans (all filed applications incl. withdrawals)
# Both rates use total_loans (all filed apps) as denominator — consistent with loan purpose/type charts
nat["approval_rate"] = ((nat["total_apps"] - nat["total_denied"]) / nat["total_loans"].clip(1)).round(4)
nat["orig_rate_all"]  = (nat["total_originated"] / nat["total_loans"].clip(1)).round(4)

fig1 = make_subplots(specs=[[{"secondary_y": True}]])

# Stack originated ON TOP of applications so both use same baseline and totals are comparable
fig1.add_trace(go.Bar(
    x=nat["year"],
    y=nat["total_loans"] - nat["total_originated"],
    name="Not Originated (withdrawn/denied/incomplete)",
    marker_color="rgba(180,200,255,0.5)",
), secondary_y=False)

fig1.add_trace(go.Bar(
    x=nat["year"], y=nat["total_originated"],
    name="Total Originated",
    marker_color="rgba(30,100,220,0.85)",
), secondary_y=False)

fig1.add_trace(go.Scatter(
    x=nat["year"], y=(nat["orig_rate_all"] * 100).round(1),
    name="Origination Rate (%)",
    mode="lines+markers+text",
    line=dict(color="#e64d1f", width=2.5),
    marker=dict(size=8),
    text=(nat["orig_rate_all"] * 100).round(1).astype(str) + "%",
    textposition="top center",
    textfont=dict(size=10),
), secondary_y=True)

fig1.add_trace(go.Scatter(
    x=nat["year"], y=(nat["approval_rate"] * 100).round(1),
    name="Approval Rate (%)",
    mode="lines+markers+text",
    line=dict(color="#8e44ad", width=2.5, dash="dot"),
    marker=dict(size=8, symbol="diamond"),
    text=(nat["approval_rate"] * 100).round(1).astype(str) + "%",
    textposition="bottom center",
    textfont=dict(size=10),
), secondary_y=True)

# Crash bottom annotation
fig1.add_vline(x=BOTTOM_YEAR, line_dash="dash", line_color="#e64d1f", line_width=1.5)
fig1.add_annotation(
    x=BOTTOM_YEAR, y=1, yref="paper",
    text=f"<b>Crash Bottom</b><br>{BOTTOM_YEAR}",
    showarrow=True, arrowhead=2, arrowcolor="#e64d1f",
    ax=40, ay=-60, font=dict(color="#e64d1f", size=12),
    bgcolor="rgba(255,255,255,0.85)",
)

# "Survival of the Fittest" callout on approval rate line at 2008
approval_2008 = float(((nat[nat["year"]==2008]["total_apps"].values[0] - nat[nat["year"]==2008]["total_denied"].values[0]) / nat[nat["year"]==2008]["total_loans"].values[0]) * 100)
fig1.add_annotation(
    x=2008, y=approval_2008, yref="y2",
    text="<b>🎯 Survival of the Fittest</b><br>Only high-quality borrowers remained.",
    showarrow=True, arrowhead=2, arrowcolor="#8e44ad",
    ax=80, ay=-55,
    font=dict(color="#8e44ad", size=11),
    bgcolor="rgba(255,255,255,0.92)",
    bordercolor="#8e44ad",
    borderwidth=1.5,
    borderpad=6,
)

# Inflection annotation
fig1.add_vline(x=INFLECTION_YEAR, line_dash="dot", line_color="#2ab548", line_width=1.5)
fig1.add_annotation(
    x=INFLECTION_YEAR, y=1, yref="paper",
    text=f"<b>Recovery Inflection</b><br>{INFLECTION_YEAR}",
    showarrow=True, arrowhead=2, arrowcolor="#2ab548",
    ax=-50, ay=-60, font=dict(color="#2ab548", size=12),
    bgcolor="rgba(255,255,255,0.85)",
)

fig1.update_layout(
    barmode="stack",
    height=420,
    legend=dict(orientation="h", y=1.1),
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(tickmode="linear", dtick=1, title="Year"),
    yaxis=dict(title="Loan Count", tickformat=","),
    yaxis2=dict(title="Rate (%)", range=[40, 105]),
    margin=dict(t=40, b=40),
)
fig1.update_xaxes(showgrid=False)
fig1.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.07)", secondary_y=False)

st.plotly_chart(fig1, use_container_width=True)

# ── Interest Rate Chart ────────────────────────────────────────────────────────
st.subheader("30-Year Fixed Mortgage Rate (Freddie Mac PMMS)")
st.caption("Source: Freddie Mac Primary Mortgage Market Survey — annual averages. Explains why origination volume spikes in 2009 and 2012.")

nat_rate = nat.merge(MORTGAGE_RATES, on="year")

fig1r = make_subplots(specs=[[{"secondary_y": True}]])

# Interest rate line — primary axis
fig1r.add_trace(go.Scatter(
    x=nat_rate["year"],
    y=nat_rate["rate_30yr"],
    name="30-Yr Fixed Rate (%)",
    mode="lines+markers+text",
    line=dict(color="#c0392b", width=3),
    marker=dict(size=10),
    text=nat_rate["rate_30yr"].astype(str) + "%",
    textposition="top center",
    textfont=dict(size=11, color="#c0392b"),
    fill="tozeroy",
    fillcolor="rgba(192,57,43,0.08)",
), secondary_y=False)

# Originations bar — secondary axis (context)
fig1r.add_trace(go.Bar(
    x=nat_rate["year"],
    y=nat_rate["total_originated"],
    name="Total Originated",
    marker_color="rgba(30,100,220,0.25)",
    marker_line_width=0,
), secondary_y=True)

# Fed Funds Rate line
fig1r.add_trace(go.Scatter(
    x=nat_rate["year"],
    y=nat_rate["fed_funds"],
    name="Fed Funds Rate (%)",
    mode="lines+markers+text",
    line=dict(color="#1a7abf", width=2.5, dash="dash"),
    marker=dict(size=8, symbol="square"),
    text=nat_rate["fed_funds"].astype(str) + "%",
    textposition="bottom center",
    textfont=dict(size=10, color="#1a7abf"),
), secondary_y=False)

# Annotate key events
RATE_EVENTS = [
    (2008, "Fed cuts rates<br>to near zero", "#e67e22", 30, -50),
    (2009, "Refi Boom #1<br>5.04%", "#27ae60", -50, -50),
    (2012, "Refi Boom #2<br>Record low 3.66%", "#27ae60", -60, -50),
    (2013, "Taper Tantrum<br>rates jump", "#c0392b", 40, -50),
    (2014, "Mortgage bottom<br>refi boom ends", "#c0392b", 40, -50),
]

for year, text, color, ax, ay in RATE_EVENTS:
    rate_val = MORTGAGE_RATES[MORTGAGE_RATES["year"] == year]["rate_30yr"].values[0]
    fig1r.add_annotation(
        x=year, y=rate_val, yref="y",
        text=f"<b>{text}</b>",
        showarrow=True, arrowhead=2, arrowcolor=color,
        ax=ax, ay=ay,
        font=dict(color=color, size=10),
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor=color,
        borderwidth=1,
    )

fig1r.update_layout(
    height=400,
    legend=dict(orientation="h", y=1.1),
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(tickmode="linear", dtick=1, title="Year", showgrid=False),
    yaxis=dict(title="30-Yr Fixed Rate (%)", range=[2.5, 7.5],
               ticksuffix="%", showgrid=True, gridcolor="rgba(0,0,0,0.07)"),
    yaxis2=dict(title="Originations", tickformat=",", showgrid=False),
    margin=dict(t=40, b=40),
)
st.plotly_chart(fig1r, use_container_width=True)

# Rate + origination table
rate_table = nat_rate[["year", "fed_funds", "rate_30yr", "total_originated", "total_loans"]].copy()
rate_table["orig_M"]    = (rate_table["total_originated"] / 1e6).round(2).astype(str) + "M"
rate_table["apps_M"]    = (rate_table["total_loans"]      / 1e6).round(2).astype(str) + "M"
rate_table["rate_30yr"] = rate_table["rate_30yr"].astype(str) + "%"
rate_table["fed_funds"] = rate_table["fed_funds"].astype(str) + "%"
rate_table["spread"]    = (nat_rate["rate_30yr"] - nat_rate["fed_funds"]).round(2).astype(str) + "%"
rate_table = rate_table[["year", "fed_funds", "rate_30yr", "spread", "apps_M", "orig_M"]]
rate_table = rate_table.drop(columns=["total_loans"], errors="ignore")
rate_table.columns = ["Year", "Fed Funds Rate", "30-Yr Mortgage Rate", "Spread", "Applications", "Originated"]
st.dataframe(rate_table.reset_index(drop=True), use_container_width=True, hide_index=True)

# ── Applications by Loan Purpose ──────────────────────────────────────────────
st.subheader("Applications by Loan Purpose Over the Years")
st.caption("Shows exactly what drove spikes and crashes in total application volume.")

lp_apps = lp.pivot_table(index="year", columns="loan_purpose_label", values="count", aggfunc="sum").fillna(0).reset_index()

COLORS_LP2 = {
    "Refinancing":      "#e74c3c",
    "Home Purchase":    "#2980b9",
    "Home Improvement": "#f39c12",
}

fig1p = make_subplots(specs=[[{"secondary_y": False}]])

for label, color in COLORS_LP2.items():
    if label in lp_apps.columns:
        fig1p.add_trace(go.Bar(
            x=lp_apps["year"],
            y=lp_apps[label],
            name=label,
            marker_color=color,
            hovertemplate=f"<b>{label}</b><br>Year: %{{x}}<br>Applications: %{{y:,.0f}}<extra></extra>",
        ))

# Mark refi boom peaks and collapse
fig1p.add_vline(x=2009, line_dash="dot", line_color="#27ae60", line_width=1.5)
fig1p.add_annotation(x=2009, y=1, yref="paper",
    text="<b>Refi Boom #1</b>", showarrow=False,
    font=dict(color="#27ae60", size=10), bgcolor="rgba(255,255,255,0.85)", ay=-10)

fig1p.add_vline(x=2012, line_dash="dot", line_color="#27ae60", line_width=1.5)
fig1p.add_annotation(x=2012, y=1, yref="paper",
    text="<b>Refi Boom #2</b>", showarrow=False,
    font=dict(color="#27ae60", size=10), bgcolor="rgba(255,255,255,0.85)", ay=-10)

fig1p.add_vline(x=2014, line_dash="dash", line_color="#e64d1f", line_width=1.5)
fig1p.add_annotation(x=2014, y=1, yref="paper",
    text="<b>Refi Collapses</b>", showarrow=False,
    font=dict(color="#e64d1f", size=10), bgcolor="rgba(255,255,255,0.85)", ay=-10)

fig1p.update_layout(
    barmode="stack",
    height=420,
    legend=dict(orientation="h", y=1.1),
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(tickmode="linear", dtick=1, title="Year", showgrid=False),
    yaxis=dict(title="Number of Applications", tickformat=",",
               showgrid=True, gridcolor="rgba(0,0,0,0.07)"),
    margin=dict(t=50, b=40),
)
st.plotly_chart(fig1p, use_container_width=True)

# Summary table
lp_tbl = lp_apps.copy()
for col in ["Home Purchase", "Refinancing", "Home Improvement"]:
    if col in lp_tbl.columns:
        lp_tbl[col] = lp_tbl[col].astype(int).map("{:,}".format)
lp_tbl["Total"] = lp.groupby("year")["count"].sum().reset_index()["count"].astype(int).map("{:,}".format)
st.dataframe(lp_tbl.reset_index(drop=True), use_container_width=True, hide_index=True)

# ── Conventional vs Govt: The Hidden Lender Tightening ────────────────────────
st.subheader("Conventional vs Government-Backed: Unmasking Lender Tightening")
st.caption(
    "The aggregate approval rate masked two opposite trends. "
    "Conventional banks tightened dramatically — but only better applicants applied. "
    "Government loans absorbed the riskier borrowers conventional banks rejected."
)

lt["category"] = lt["loan_type"].map({
    1: "Conventional (Private)",
    2: "Govt-Backed (FHA/VA/FSA)",
    3: "Govt-Backed (FHA/VA/FSA)",
    4: "Govt-Backed (FHA/VA/FSA)",
})
cat_df = lt.groupby(["year","category"]).agg(count=("count","sum"), originated=("originated","sum")).reset_index()
cat_df["orig_rate"] = (cat_df["originated"] / cat_df["count"].clip(1)).round(4)

fig1c = make_subplots(rows=1, cols=2,
    subplot_titles=["Origination Rate: Conventional vs Govt (%)", "Application Volume: Conventional vs Govt"],
    horizontal_spacing=0.10)

COLORS_CAT = {"Conventional (Private)": "#1a7abf", "Govt-Backed (FHA/VA/FSA)": "#e67e22"}

for cat, color in COLORS_CAT.items():
    subset = cat_df[cat_df["category"] == cat].sort_values("year")
    # Rate chart
    fig1c.add_trace(go.Scatter(
        x=subset["year"], y=(subset["orig_rate"]*100).round(1),
        name=cat, mode="lines+markers",
        line=dict(color=color, width=2.5),
        marker=dict(size=8),
        legendgroup=cat,
        hovertemplate=f"<b>{cat}</b><br>%{{y:.1f}}%<extra></extra>",
    ), row=1, col=1)
    # Volume chart
    fig1c.add_trace(go.Bar(
        x=subset["year"], y=subset["count"],
        name=cat, marker_color=color,
        legendgroup=cat, showlegend=False,
        hovertemplate=f"<b>{cat}</b><br>Apps: %{{y:,.0f}}<extra></extra>",
    ), row=1, col=2)

# Add annotation: conventional rate rising = self-selection, not lender generosity
fig1c.add_annotation(
    x=2012, y=56, xref="x", yref="y",
    text="<b>Rising rate = only<br>qualified applicants left</b>",
    showarrow=True, arrowhead=2, arrowcolor="#1a7abf",
    ax=60, ay=-40, font=dict(color="#1a7abf", size=10),
    bgcolor="rgba(255,255,255,0.88)",
)
fig1c.add_annotation(
    x=2009, y=1, xref="x2", yref="paper",
    text="<b>Govt loans 4× in 2 years</b>",
    showarrow=False, font=dict(color="#e67e22", size=10),
    bgcolor="rgba(255,255,255,0.88)", row=1, col=2,
)

for col_idx in [1, 2]:
    fig1c.add_vline(x=2008, line_dash="dash", line_color="#e64d1f", line_width=1.2, col=col_idx, row=1)

fig1c.update_layout(
    height=400, barmode="stack",
    legend=dict(orientation="h", y=1.12),
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(t=50, b=40),
)
fig1c.update_xaxes(tickmode="linear", dtick=1, showgrid=False)
fig1c.update_yaxes(title_text="Origination Rate (%)", ticksuffix="%",
                   showgrid=True, gridcolor="rgba(0,0,0,0.06)", row=1, col=1)
fig1c.update_yaxes(title_text="Applications", tickformat=",",
                   showgrid=True, gridcolor="rgba(0,0,0,0.06)", row=1, col=2)
st.plotly_chart(fig1c, use_container_width=True)

# Key callouts
c1, c2, c3 = st.columns(3)
conv_2007 = cat_df[(cat_df["year"]==2007) & (cat_df["category"]=="Conventional (Private)")].iloc[0]
conv_2009 = cat_df[(cat_df["year"]==2009) & (cat_df["category"]=="Conventional (Private)")].iloc[0]
govt_2007 = cat_df[(cat_df["year"]==2007) & (cat_df["category"]=="Govt-Backed (FHA/VA/FSA)")].iloc[0]
govt_2009 = cat_df[(cat_df["year"]==2009) & (cat_df["category"]=="Govt-Backed (FHA/VA/FSA)")].iloc[0]
c1.metric("Conventional Apps 2007→2009",
          f"{int(conv_2009['count']):,}",
          f"{int(conv_2009['count']-conv_2007['count']):,} ({(conv_2009['count']/conv_2007['count']-1):.0%})")
c2.metric("Govt-Backed Apps 2007→2009",
          f"{int(govt_2009['count']):,}",
          f"+{int(govt_2009['count']-govt_2007['count']):,} (+{(govt_2009['count']/govt_2007['count']-1):.0%})")
c3.metric("Conventional Orig Rate 2007→2017",
          f"{conv_2009['orig_rate']:.1%} (2009)",
          f"Started at {conv_2007['orig_rate']:.1%} in 2007 — rose as weak applicants left")

st.info(
    "**The paradox explained:** Conventional banks DID tighten dramatically after 2008 — "
    "but the riskiest borrowers had already stopped applying. The remaining applicant pool was "
    "so much stronger that the approval rate actually *rose*, masking the tightening. "
    "Meanwhile, government-backed loans absorbed 4× more applicants — the people conventional "
    "banks turned away. The HMDA data cannot show the millions of *discouraged borrowers* who "
    "never applied at all — they are the invisible victims of the crash."
)

col1, col2, col3, col4 = st.columns(4)
bottom_nat = nat[nat["year"] == BOTTOM_YEAR].iloc[0]
inflect_nat = nat[nat["year"] == INFLECTION_YEAR].iloc[0]
prev_nat = nat[nat["year"] == INFLECTION_YEAR - 1].iloc[0]
yoy = (inflect_nat["total_originated"] - prev_nat["total_originated"]) / prev_nat["total_originated"]
col1.metric("Crash Bottom Year", str(BOTTOM_YEAR))
col2.metric("Originations at Bottom", f"{int(bottom_nat['total_originated']):,}")
col3.metric("Recovery Inflection Year", str(INFLECTION_YEAR))
col4.metric("YoY Growth at Inflection", f"+{yoy:.1%}")

# ── Approval Rate by Loan Purpose ─────────────────────────────────────────────
st.subheader("Approval & Origination Rate by Loan Purpose")
st.caption("Approval Rate = bank-approved loans / all applications (incl. withdrawals). Origination Rate = fully closed loans / all applications. Matches industry-reported figures.")

# Web-standard definition: originated / all applications (matches industry reports)
# Denominator includes withdrawals + incomplete — reflects real-world access to credit
lp["approval_rate"] = (lp["approved"]   / lp["count"].clip(1)).round(4)
lp["orig_rate"]     = (lp["originated"] / lp["count"].clip(1)).round(4)

COLORS_PURPOSE = {
    "Home Purchase":    ("#2980b9", "#aed6f1"),
    "Refinancing":      ("#e74c3c", "#f5b7b1"),
    "Home Improvement": ("#f39c12", "#fde8a1"),
}

fig1b = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Approval Rate by Loan Purpose (%)", "Origination Rate by Loan Purpose (%)"],
    horizontal_spacing=0.10,
)

for label, (solid, light) in COLORS_PURPOSE.items():
    subset = lp[lp["loan_purpose_label"] == label].sort_values("year")
    if subset.empty:
        continue
    fig1b.add_trace(go.Scatter(
        x=subset["year"], y=(subset["approval_rate"] * 100).round(1),
        name=label,
        mode="lines+markers",
        line=dict(color=solid, width=2.5),
        marker=dict(size=7),
        legendgroup=label,
        hovertemplate=f"<b>{label}</b><br>Year: %{{x}}<br>Approval Rate: %{{y:.1f}}%<extra></extra>",
    ), row=1, col=1)
    fig1b.add_trace(go.Scatter(
        x=subset["year"], y=(subset["orig_rate"] * 100).round(1),
        name=label,
        mode="lines+markers",
        line=dict(color=solid, width=2.5, dash="dot"),
        marker=dict(size=7, symbol="diamond"),
        legendgroup=label,
        showlegend=False,
        hovertemplate=f"<b>{label}</b><br>Year: %{{x}}<br>Orig Rate: %{{y:.1f}}%<extra></extra>",
    ), row=1, col=2)

for col_idx in [1, 2]:
    fig1b.add_vline(x=BOTTOM_YEAR, line_dash="dash", line_color="#e64d1f", line_width=1.2, col=col_idx, row=1)
    fig1b.add_vline(x=INFLECTION_YEAR, line_dash="dot", line_color="#2ab548", line_width=1.2, col=col_idx, row=1)

fig1b.update_layout(
    height=380,
    legend=dict(orientation="h", y=1.12),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(t=50, b=40),
)
fig1b.update_xaxes(tickmode="linear", dtick=1, showgrid=False)
fig1b.update_yaxes(title_text="Rate (%)", ticksuffix="%", showgrid=True,
                   gridcolor="rgba(0,0,0,0.06)", range=[30, 100])

st.plotly_chart(fig1b, use_container_width=True)

# Summary table
lp_latest = lp[lp["year"] == 2017][["loan_purpose_label", "count", "originated", "denied", "approval_rate", "orig_rate"]].copy()
lp_latest.columns = ["Loan Purpose", "Applications", "Originated", "Denied", "Approval Rate", "Orig Rate"]
lp_latest["Approval Rate"] = (lp_latest["Approval Rate"] * 100).round(1).astype(str) + "%"
lp_latest["Orig Rate"]     = (lp_latest["Orig Rate"]     * 100).round(1).astype(str) + "%"
lp_latest["Applications"]  = lp_latest["Applications"].map("{:,}".format)
lp_latest["Originated"]    = lp_latest["Originated"].map("{:,}".format)
lp_latest["Denied"]        = lp_latest["Denied"].map("{:,}".format)
st.caption("2017 snapshot:")
st.dataframe(lp_latest.reset_index(drop=True), use_container_width=True, hide_index=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — Denial Rate Choropleth Map
# ══════════════════════════════════════════════════════════════════════════════
st.header("2. Geographic Denial Rates by State")
st.caption("Use the slider to move through years one at a time.")

selected_year = st.slider("Select Year", min_value=min(YEARS), max_value=max(YEARS), value=2008, step=1)

state_year = state[state["year"] == selected_year].copy()
state_year["denial_pct"] = (state_year["denial_rate"] * 100).round(2)
state_year["orig_pct"]   = (state_year["orig_rate"]   * 100).round(2)
state_year["hover"] = (
    "<b>" + state_year["state_name"] + "</b><br>"
    "Denial Rate: " + state_year["denial_pct"].astype(str) + "%<br>"
    "Orig Rate: "   + state_year["orig_pct"].astype(str) + "%<br>"
    "Avg Loan: $"   + (state_year["avg_loan"].round(0).astype(int) * 1000).map("{:,}".format)
)

fig2 = go.Figure(go.Choropleth(
    locations=state_year["state_abbr"],
    z=state_year["denial_pct"],
    locationmode="USA-states",
    colorscale=[
        [0.0, "#d1fae5"], [0.2, "#6ee7b7"], [0.4, "#34d399"],
        [0.6, "#f59e0b"], [0.8, "#ef4444"], [1.0, "#7f1d1d"],
    ],
    zmin=5, zmax=50,
    colorbar=dict(
        title="Denial Rate (%)",
        ticksuffix="%",
        thickness=15,
        len=0.7,
    ),
    hovertext=state_year["hover"],
    hoverinfo="text",
))

fig2.update_layout(
    geo=dict(scope="usa", projection_type="albers usa", showlakes=True, lakecolor="lightblue"),
    height=450,
    margin=dict(t=30, b=10, l=10, r=10),
    title=dict(text=f"Mortgage Denial Rates by State — {selected_year}", font=dict(size=15)),
    paper_bgcolor="white",
)

st.plotly_chart(fig2, use_container_width=True)

# Year-over-year denial rate change table
if selected_year > min(YEARS):
    prev_state = state[state["year"] == selected_year - 1][["state_abbr", "denial_rate"]].rename(columns={"denial_rate": "prev_denial"})
    merged = state_year.merge(prev_state, on="state_abbr")
    merged["delta"] = ((merged["denial_rate"] - merged["prev_denial"]) * 100).round(2)
    top_increase = merged.nlargest(5, "delta")[["state_name", "denial_pct", "delta"]]
    top_decrease = merged.nsmallest(5, "delta")[["state_name", "denial_pct", "delta"]]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Top 5 Denial Rate Increases vs {selected_year-1}")
        top_increase.columns = ["State", "Denial Rate (%)", "Change (pp)"]
        st.dataframe(top_increase.reset_index(drop=True), use_container_width=True, hide_index=True)
    with c2:
        st.subheader(f"Top 5 Denial Rate Decreases vs {selected_year-1}")
        top_decrease.columns = ["State", "Denial Rate (%)", "Change (pp)"]
        st.dataframe(top_decrease.reset_index(drop=True), use_container_width=True, hide_index=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3 — Government Backstop
# ══════════════════════════════════════════════════════════════════════════════
st.header("3. The Government Backstop: When Private Capital Fled")

# Loan type share by year
lt_pivot = lt.pivot_table(index="year", columns="loan_type_label", values="count", aggfunc="sum").fillna(0)
total_by_year = lt_pivot.sum(axis=1)
lt_share = lt_pivot.div(total_by_year, axis=0) * 100

# Loan purpose by year (originations only)
lp_pivot = lp.pivot_table(index="year", columns="loan_purpose_label", values="originated", aggfunc="sum").fillna(0)
lp_total = lp_pivot.sum(axis=1)
lp_share = lp_pivot.div(lp_total, axis=0) * 100

fig3 = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Loan Type Share (% of Applications)", "Loan Purpose (% of Originations)"],
    horizontal_spacing=0.10,
)

COLORS_LT = {
    "Conventional (Private)": "#1e64dc",
    "FHA-Insured (Govt)": "#e67e22",
    "VA-Guaranteed (Govt)": "#27ae60",
    "FSA/RHS (Govt)": "#8e44ad",
}

for label, color in COLORS_LT.items():
    if label in lt_share.columns:
        fig3.add_trace(go.Scatter(
            x=lt_share.index, y=lt_share[label].round(1),
            name=label,
            stackgroup="lt",
            mode="lines",
            line=dict(width=0.5, color=color),
            fillcolor=color.replace(")", ",0.75)").replace("rgb", "rgba") if color.startswith("rgb") else color,
            hovertemplate="%{y:.1f}%<extra>" + label + "</extra>",
        ), row=1, col=1)

COLORS_LP = {
    "Home Purchase": "#2980b9",
    "Refinancing": "#e74c3c",
    "Home Improvement": "#f39c12",
}
for label, color in COLORS_LP.items():
    if label in lp_share.columns:
        fig3.add_trace(go.Scatter(
            x=lp_share.index, y=lp_share[label].round(1),
            name=label,
            stackgroup="lp",
            mode="lines",
            line=dict(width=0.5, color=color),
            hovertemplate="%{y:.1f}%<extra>" + label + "</extra>",
        ), row=1, col=2)

# Mark 2008 on both subplots
for col_idx in [1, 2]:
    fig3.add_vline(x=2008, line_dash="dash", line_color="#e64d1f", line_width=1.2, col=col_idx, row=1)

fig3.update_layout(
    height=420,
    legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(t=50, b=80),
)
fig3.update_xaxes(tickmode="linear", dtick=1, showgrid=False)
fig3.update_yaxes(title_text="Share (%)", ticksuffix="%", showgrid=True, gridcolor="rgba(0,0,0,0.06)", row=1, col=1)
fig3.update_yaxes(title_text="Share (%)", ticksuffix="%", showgrid=True, gridcolor="rgba(0,0,0,0.06)", row=1, col=2)

st.plotly_chart(fig3, use_container_width=True)

# Govt share callout
govt_2007 = nat[nat["year"] == 2007].iloc[0]["govt_share"]
govt_peak  = nat.loc[nat["govt_share"].idxmax()]
conv_2007  = lt[(lt["year"] == 2007) & (lt["loan_type_label"] == "Conventional (Private)")]["count"].sum()
conv_2009  = lt[(lt["year"] == 2009) & (lt["loan_type_label"] == "Conventional (Private)")]["count"].sum()

c1, c2, c3 = st.columns(3)
c1.metric("Govt Loan Share (2007)", f"{govt_2007:.1%}", help="FHA + VA + FSA share of all applications")
c2.metric(f"Govt Loan Share Peak ({int(govt_peak['year'])})", f"{govt_peak['govt_share']:.1%}", f"+{(govt_peak['govt_share']-govt_2007)*100:.1f} pp vs 2007")
c3.metric("Conventional Apps Drop (2007→2009)", f"{int(conv_2009 - conv_2007):,}", f"{(conv_2009/conv_2007 - 1):.1%}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 4 — MSA Recovery Index
# ══════════════════════════════════════════════════════════════════════════════
st.header("4. MSA Recovery Index (2007 = 100)")
st.caption("Only MSAs with ≥ 500 originations in 2007 are shown. Recovery index = (year originations / 2007 originations) × 100.")

col_yr, col_top = st.columns([2, 1])
with col_yr:
    idx_year = st.select_slider("Select Year for Recovery Index", options=YEARS, value=2017)
with col_top:
    top_n = st.number_input("Show Top/Bottom N MSAs", min_value=10, max_value=50, value=20, step=5)

msa_year = msa_r[msa_r["year"] == idx_year].copy()
msa_year = msa_year.sort_values("recovery_index", ascending=False)
total_msas = len(msa_year)
recovered = (msa_year["recovery_index"] >= 100).sum()

# Full scatter (all MSAs)
msa_scatter = msa_r.copy()
msa_scatter["color"] = msa_scatter["recovery_index"].apply(
    lambda v: "#27ae60" if v >= 100 else ("#e67e22" if v >= 70 else "#e74c3c")
)

fig4a = go.Figure()

for label, color, condition in [
    ("Recovered (≥100)", "#27ae60", msa_year["recovery_index"] >= 100),
    ("Partial (70–99)", "#e67e22", (msa_year["recovery_index"] >= 70) & (msa_year["recovery_index"] < 100)),
    ("Lagging (<70)", "#e74c3c", msa_year["recovery_index"] < 70),
]:
    subset = msa_year[condition]
    fig4a.add_trace(go.Scatter(
        x=subset["total_apps"],
        y=subset["recovery_index"],
        mode="markers",
        name=label,
        marker=dict(color=color, size=7, opacity=0.7, line=dict(width=0.5, color="white")),
        hovertemplate="<b>%{customdata}</b><br>Recovery Index: %{y:.1f}<br>Apps: %{x:,}<extra></extra>",
        customdata=subset["msa_md_name"],
    ))

fig4a.add_hline(y=100, line_dash="dash", line_color="#555", line_width=1.5,
                annotation_text="2007 Baseline (100)", annotation_position="top right")

fig4a.update_layout(
    height=420,
    xaxis=dict(title="Total Applications (log scale)", type="log", showgrid=True, gridcolor="rgba(0,0,0,0.07)"),
    yaxis=dict(title="Recovery Index (2007=100)", showgrid=True, gridcolor="rgba(0,0,0,0.07)"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(orientation="h", y=1.07),
    margin=dict(t=40, b=50),
    title=dict(text=f"MSA Recovery Index — {idx_year}  ({recovered}/{total_msas} MSAs at or above 2007 levels)", font=dict(size=14)),
)
st.plotly_chart(fig4a, use_container_width=True)

# Top / Bottom bars
c1, c2 = st.columns(2)

with c1:
    top = msa_year.nlargest(top_n, "recovery_index")
    fig_top = go.Figure(go.Bar(
        y=top["msa_md_name"], x=top["recovery_index"],
        orientation="h",
        marker_color=top["recovery_index"].apply(lambda v: "#27ae60" if v >= 100 else "#e67e22"),
        text=top["recovery_index"].round(1),
        textposition="outside",
    ))
    fig_top.add_vline(x=100, line_dash="dash", line_color="#555")
    fig_top.update_layout(
        title=f"Top {top_n} Most Recovered MSAs ({idx_year})",
        height=max(400, top_n * 22),
        margin=dict(t=40, b=30, l=10, r=60),
        xaxis=dict(title="Recovery Index", range=[0, max(top["recovery_index"].max() * 1.1, 120)]),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_top, use_container_width=True)

with c2:
    bottom = msa_year.nsmallest(top_n, "recovery_index")
    fig_bot = go.Figure(go.Bar(
        y=bottom["msa_md_name"], x=bottom["recovery_index"],
        orientation="h",
        marker_color="#e74c3c",
        text=bottom["recovery_index"].round(1),
        textposition="outside",
    ))
    fig_bot.add_vline(x=100, line_dash="dash", line_color="#555")
    fig_bot.update_layout(
        title=f"Bottom {top_n} Lagging MSAs ({idx_year})",
        height=max(400, top_n * 22),
        margin=dict(t=40, b=30, l=10, r=60),
        xaxis=dict(title="Recovery Index", range=[0, 130]),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_bot, use_container_width=True)

# Recovery trend for selected MSA
st.subheader("MSA Recovery Trend Over Time")
all_msa_names = sorted(msa_r["msa_md_name"].dropna().unique())
default_msas = ["San Jose - San Francisco - Oakland, CA"] if "San Jose - San Francisco - Oakland, CA" in all_msa_names else all_msa_names[:3]
selected_msas = st.multiselect("Select MSA(s) to compare", all_msa_names, default=default_msas[:3])

if selected_msas:
    trend_data = msa_r[msa_r["msa_md_name"].isin(selected_msas)]
    fig4b = go.Figure()
    for msa_name in selected_msas:
        subset = trend_data[trend_data["msa_md_name"] == msa_name].sort_values("year")
        fig4b.add_trace(go.Scatter(
            x=subset["year"], y=subset["recovery_index"],
            mode="lines+markers", name=msa_name,
            hovertemplate="%{y:.1f}<extra>" + msa_name + "</extra>",
        ))
    fig4b.add_hline(y=100, line_dash="dash", line_color="#555", line_width=1.5,
                    annotation_text="2007 Baseline (100)", annotation_position="top right")
    fig4b.update_layout(
        height=360,
        xaxis=dict(tickmode="linear", dtick=1, title="Year", showgrid=False),
        yaxis=dict(title="Recovery Index (2007=100)", showgrid=True, gridcolor="rgba(0,0,0,0.07)"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig4b, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 5 — Denial Reasons
# ══════════════════════════════════════════════════════════════════════════════
st.header("5. Why Were Loans Denied? (2007–2017)")
st.caption(
    "Each denied application can cite up to 3 reasons. "
    "Counts reflect total reason citations, not unique applications."
)

REASON_COLORS = {
    "Credit history":                "#e74c3c",
    "Debt-to-income ratio":          "#e67e22",
    "Collateral":                    "#f1c40f",
    "Other":                         "#95a5a6",
    "Credit application incomplete": "#3498db",
    "Unverifiable information":      "#9b59b6",
    "Insufficient cash":             "#1abc9c",
    "Employment history":            "#2ecc71",
    "Mortgage insurance denied":     "#34495e",
}

# ── 5a. Stacked bar — denial reasons by year ──────────────────────────────────
st.subheader("Denial Reasons Over the Years")

dr_pivot = dr_nat.pivot_table(
    index="year", columns="reason_label", values="count", aggfunc="sum"
).fillna(0).reset_index()

fig5a = go.Figure()
for reason, color in REASON_COLORS.items():
    if reason in dr_pivot.columns:
        fig5a.add_trace(go.Bar(
            x=dr_pivot["year"], y=dr_pivot[reason],
            name=reason,
            marker_color=color,
            hovertemplate=f"<b>{reason}</b><br>Year: %{{x}}<br>Count: %{{y:,.0f}}<extra></extra>",
        ))

fig5a.add_vline(x=2008, line_dash="dash", line_color="#c0392b", line_width=1.5)
fig5a.add_annotation(x=2008, y=1, yref="paper", text="<b>2008 Crash</b>",
    showarrow=False, font=dict(color="#c0392b", size=10),
    bgcolor="rgba(255,255,255,0.85)")

fig5a.update_layout(
    barmode="stack", height=450,
    legend=dict(orientation="h", y=1.12, font=dict(size=11)),
    plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(tickmode="linear", dtick=1, title="Year", showgrid=False),
    yaxis=dict(title="Denial Reason Citations", tickformat=",",
               showgrid=True, gridcolor="rgba(0,0,0,0.07)"),
    margin=dict(t=60, b=40),
)
st.plotly_chart(fig5a, use_container_width=True)

# ── 5b. Share chart — how the MIX changed over time ──────────────────────────
st.subheader("How the Mix of Denial Reasons Shifted")
st.caption("Shows % share of each reason — reveals how bank priorities changed after the crash.")

dr_total = dr_pivot.drop(columns=["year"]).sum(axis=1)
dr_share = dr_pivot.copy()
for col in dr_pivot.columns:
    if col != "year":
        dr_share[col] = (dr_pivot[col] / dr_total * 100).round(2)

fig5b = go.Figure()
for reason, color in REASON_COLORS.items():
    if reason in dr_share.columns:
        fig5b.add_trace(go.Scatter(
            x=dr_share["year"], y=dr_share[reason],
            name=reason,
            stackgroup="one",
            mode="lines",
            line=dict(width=0.5, color=color),
            hovertemplate=f"<b>{reason}</b>: %{{y:.1f}}%<extra></extra>",
        ))

fig5b.add_vline(x=2008, line_dash="dash", line_color="#c0392b", line_width=1.5)
fig5b.update_layout(
    height=400,
    legend=dict(orientation="h", y=1.12, font=dict(size=11)),
    plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(tickmode="linear", dtick=1, title="Year", showgrid=False),
    yaxis=dict(title="Share of Denial Citations (%)", ticksuffix="%",
               showgrid=True, gridcolor="rgba(0,0,0,0.07)"),
    margin=dict(t=60, b=40),
)
st.plotly_chart(fig5b, use_container_width=True)

# ── 5c. By loan purpose — with year selector ─────────────────────────────────
st.subheader("Denial Reasons by Loan Purpose")

LOAN_PURPOSE_MAP2 = {1: "Home Purchase", 2: "Home Improvement", 3: "Refinancing"}
dr_full["purpose_label"] = pd.to_numeric(dr_full["loan_purpose"], errors="coerce").map(LOAN_PURPOSE_MAP2)

purpose_yr_col, purpose_mode_col = st.columns([2, 1])
with purpose_yr_col:
    purpose_year = st.select_slider(
        "Select Year (Denial Reasons by Purpose)",
        options=["All Years"] + sorted(dr_full["year"].unique().tolist()),
        value="All Years",
    )
with purpose_mode_col:
    show_raw = st.checkbox("Show raw counts instead of %", value=False)

# Filter by year or use all
if purpose_year == "All Years":
    dr_filtered = dr_full.copy()
    year_label  = "2007–2017 (All Years)"
else:
    dr_filtered = dr_full[dr_full["year"] == purpose_year].copy()
    year_label  = str(purpose_year)

purpose_reason = (
    dr_filtered.groupby(["purpose_label", "reason_label"])["count"]
    .sum().reset_index()
)
purpose_reason = purpose_reason[purpose_reason["purpose_label"].notna()]

purpose_total = purpose_reason.groupby("purpose_label")["count"].transform("sum")
purpose_reason["share"] = (purpose_reason["count"] / purpose_total * 100).round(1)

purposes = ["Home Purchase", "Refinancing", "Home Improvement"]
fig5c = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f"{p}" for p in purposes],
    horizontal_spacing=0.08,
)

for i, purpose in enumerate(purposes, 1):
    subset = purpose_reason[purpose_reason["purpose_label"] == purpose].sort_values(
        "count" if show_raw else "share", ascending=True
    )
    x_val  = subset["count"] if show_raw else subset["share"]
    x_text = subset["count"].map("{:,}".format) if show_raw else (subset["share"].astype(str) + "%")
    x_max  = x_val.max() * 1.3 if not subset.empty else 10

    fig5c.add_trace(go.Bar(
        y=subset["reason_label"],
        x=x_val,
        orientation="h",
        marker_color=[REASON_COLORS.get(r, "#aaa") for r in subset["reason_label"]],
        text=x_text,
        textposition="outside",
        showlegend=False,
        hovertemplate="<b>%{y}</b><br>" + ("Count: %{x:,.0f}" if show_raw else "Share: %{x:.1f}%") + "<extra></extra>",
    ), row=1, col=i)
    fig5c.update_xaxes(range=[0, x_max], row=1, col=i)

fig5c.update_layout(
    height=380,
    title=dict(text=f"Denial Reasons by Loan Purpose — {year_label}", font=dict(size=13)),
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(t=70, b=30, l=10, r=80),
)
x_title = "Count" if show_raw else "Share (%)"
x_suffix = "" if show_raw else "%"
fig5c.update_xaxes(title_text=x_title, ticksuffix=x_suffix,
                   showgrid=True, gridcolor="rgba(0,0,0,0.06)")
fig5c.update_yaxes(showgrid=False)
st.plotly_chart(fig5c, use_container_width=True)

# ── 5d. KPI summary ───────────────────────────────────────────────────────────
total_denials = dr_nat["count"].sum()
top_reason    = dr_nat.groupby("reason_label")["count"].sum().idxmax()
top_count     = dr_nat.groupby("reason_label")["count"].sum().max()
peak_year_dtr = dr_nat[dr_nat["reason_label"]=="Debt-to-income ratio"].groupby("year")["count"].sum().idxmax()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Denial Citations (2007–17)", f"{total_denials:,.0f}")
c2.metric("Most Common Reason (all years)", top_reason, f"{top_count:,.0f} citations")
c3.metric("DTI Denials Peak Year", str(peak_year_dtr))
c4.metric("Unique Reason Categories", "9")

# ── 5e. Year-by-year table ────────────────────────────────────────────────────
st.subheader("Full Breakdown Table")
tbl = dr_nat.pivot_table(index="year", columns="reason_label", values="count", aggfunc="sum").fillna(0).astype(int)
tbl["Total"] = tbl.sum(axis=1)
tbl = tbl.reset_index()
tbl.columns.name = None
st.dataframe(tbl, use_container_width=True, hide_index=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 6 — Income & Affordability Analysis
# ══════════════════════════════════════════════════════════════════════════════
st.header("6. Borrower Income & Affordability (2007–2017)")
st.caption(
    "Based on approved (originated) loans only. Income values in $000s. "
    "Outliers (HMDA cap code 9999) excluded. Loan-to-income ratio = loan amount ÷ annual income."
)

# ── 6a. Mean vs Median Income over years ──────────────────────────────────────
st.subheader("Mean & Median Income of Approved Borrowers")

fig6a = make_subplots(specs=[[{"secondary_y": True}]])

fig6a.add_trace(go.Scatter(
    x=inc_nat["year"], y=inc_nat["approved_income_median"],
    name="Median Income ($K)", mode="lines+markers+text",
    line=dict(color="#2980b9", width=3),
    marker=dict(size=9),
    text=("$" + inc_nat["approved_income_median"].astype(int).astype(str) + "K"),
    textposition="top center", textfont=dict(size=10),
))
fig6a.add_trace(go.Scatter(
    x=inc_nat["year"], y=inc_nat["approved_income_mean"],
    name="Mean Income ($K)", mode="lines+markers+text",
    line=dict(color="#2980b9", width=2, dash="dot"),
    marker=dict(size=7, symbol="diamond"),
    text=("$" + inc_nat["approved_income_mean"].round(0).astype(int).astype(str) + "K"),
    textposition="bottom center", textfont=dict(size=10, color="#2980b9"),
))
fig6a.add_trace(go.Scatter(
    x=inc_nat["year"], y=inc_nat["denied_income_median"],
    name="Denied — Median Income ($K)", mode="lines+markers",
    line=dict(color="#e74c3c", width=2, dash="dash"),
    marker=dict(size=7),
))
fig6a.add_trace(go.Scatter(
    x=inc_nat["year"], y=inc_nat["approved_loan_median"],
    name="Median Loan Amount ($K)", mode="lines+markers+text",
    line=dict(color="#27ae60", width=2.5),
    marker=dict(size=8, symbol="square"),
    text=("$" + inc_nat["approved_loan_median"].astype(int).astype(str) + "K"),
    textposition="top center", textfont=dict(size=10, color="#27ae60"),
), secondary_y=True)

fig6a.add_vline(x=2008, line_dash="dash", line_color="#e64d1f", line_width=1.5)
fig6a.add_annotation(x=2008, y=1, yref="paper", text="<b>2008 Crash</b>",
    showarrow=False, font=dict(color="#e64d1f", size=10), bgcolor="rgba(255,255,255,0.85)")

fig6a.update_layout(
    height=420,
    legend=dict(orientation="h", y=1.12),
    plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(tickmode="linear", dtick=1, title="Year", showgrid=False),
    yaxis=dict(title="Income ($K)", showgrid=True, gridcolor="rgba(0,0,0,0.07)"),
    yaxis2=dict(title="Loan Amount ($K)", showgrid=False),
    margin=dict(t=50, b=40),
)
st.plotly_chart(fig6a, use_container_width=True)

# KPIs
c1, c2, c3, c4 = st.columns(4)
inc_2007 = inc_nat[inc_nat["year"]==2007].iloc[0]
inc_2017 = inc_nat[inc_nat["year"]==2017].iloc[0]
c1.metric("Median Income 2007", f"${int(inc_2007['approved_income_median']):,}K")
c2.metric("Median Income 2017", f"${int(inc_2017['approved_income_median']):,}K",
          f"{((inc_2017['approved_income_median']/inc_2007['approved_income_median'])-1)*100:+.1f}%")
c3.metric("Median Loan 2007",   f"${int(inc_2007['approved_loan_median']):,}K")
c4.metric("Median Loan 2017",   f"${int(inc_2017['approved_loan_median']):,}K",
          f"{((inc_2017['approved_loan_median']/inc_2007['approved_loan_median'])-1)*100:+.1f}%")

st.markdown("---")

# ── 6b. Loan-to-Income Ratio (affordability) ──────────────────────────────────
st.subheader("Loan-to-Income Ratio & Income Required per $100K Loan")
st.caption(
    "LTI = loan amount ÷ annual income.  "
    "LTI of 2.0 means the loan is 2× your annual salary.  "
    "Higher LTI = less affordable.  "
    "Most lenders use a max LTI of 4–4.5×."
)

fig6b = make_subplots(specs=[[{"secondary_y": True}]])

fig6b.add_trace(go.Bar(
    x=inc_nat["year"], y=inc_nat["lti_median"],
    name="Median LTI Ratio",
    marker_color=[
        "#27ae60" if v < 2.2 else ("#f39c12" if v < 2.5 else "#e74c3c")
        for v in inc_nat["lti_median"]
    ],
    text=inc_nat["lti_median"].round(2).astype(str) + "×",
    textposition="outside",
))
fig6b.add_trace(go.Scatter(
    x=inc_nat["year"], y=inc_nat["income_per_100k_loan"],
    name="Income needed per $100K loan ($K)",
    mode="lines+markers+text",
    line=dict(color="#8e44ad", width=2.5),
    marker=dict(size=8),
    text=("$" + inc_nat["income_per_100k_loan"].astype(str) + "K"),
    textposition="top center", textfont=dict(size=10, color="#8e44ad"),
), secondary_y=True)

fig6b.add_vline(x=2008, line_dash="dash", line_color="#e64d1f", line_width=1.5)
fig6b.update_layout(
    height=400,
    legend=dict(orientation="h", y=1.1),
    plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(tickmode="linear", dtick=1, title="Year", showgrid=False),
    yaxis=dict(title="LTI Ratio (×)", range=[0, 3.2],
               showgrid=True, gridcolor="rgba(0,0,0,0.07)"),
    yaxis2=dict(title="Income needed per $100K ($K)", range=[35, 60], showgrid=False),
    margin=dict(t=40, b=40),
)
st.plotly_chart(fig6b, use_container_width=True)

st.info(
    "**How to read this:** In 2007, the median approved borrower needed **$48.3K income to borrow $100K**. "
    "By 2017 that dropped to **$41.7K per $100K** — meaning loans grew faster than incomes. "
    "The LTI ratio rising from 2.07× to 2.40× confirms houses became less affordable relative to income "
    "even as the market 'recovered.' A rising LTI is an early warning sign of a new affordability bubble."
)

st.markdown("---")

# ── 6c. Income by Loan Purpose ────────────────────────────────────────────────
st.subheader("Median Income & Loan by Loan Purpose")

fig6c = make_subplots(rows=1, cols=2,
    subplot_titles=["Median Income of Approved Borrowers ($K)", "Median Loan Amount ($K)"],
    horizontal_spacing=0.10)

COLORS_PURPOSE2 = {"Home Purchase": "#2980b9", "Refinancing": "#e74c3c", "Home Improvement": "#f39c12"}

for label, color in COLORS_PURPOSE2.items():
    sub = inc_lp[inc_lp["loan_purpose_label"] == label].sort_values("year")
    fig6c.add_trace(go.Scatter(
        x=sub["year"], y=sub["approved_income_median"],
        name=label, mode="lines+markers",
        line=dict(color=color, width=2.5), marker=dict(size=7),
        legendgroup=label,
        hovertemplate=f"<b>{label}</b><br>Income: $%{{y:.0f}}K<extra></extra>",
    ), row=1, col=1)
    fig6c.add_trace(go.Scatter(
        x=sub["year"], y=sub["approved_loan_median"],
        name=label, mode="lines+markers",
        line=dict(color=color, width=2.5, dash="dot"), marker=dict(size=7),
        legendgroup=label, showlegend=False,
        hovertemplate=f"<b>{label}</b><br>Loan: $%{{y:.0f}}K<extra></extra>",
    ), row=1, col=2)

for col_idx in [1, 2]:
    fig6c.add_vline(x=2008, line_dash="dash", line_color="#e64d1f", line_width=1, col=col_idx, row=1)

fig6c.update_layout(
    height=380, legend=dict(orientation="h", y=1.12),
    plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=50, b=40),
)
fig6c.update_xaxes(tickmode="linear", dtick=1, showgrid=False)
fig6c.update_yaxes(title_text="Income ($K)", tickprefix="$", ticksuffix="K",
                   showgrid=True, gridcolor="rgba(0,0,0,0.06)", row=1, col=1)
fig6c.update_yaxes(title_text="Loan ($K)", tickprefix="$", ticksuffix="K",
                   showgrid=True, gridcolor="rgba(0,0,0,0.06)", row=1, col=2)
st.plotly_chart(fig6c, use_container_width=True)

st.markdown("---")

# ── 6d. State income map ──────────────────────────────────────────────────────
st.subheader("Median Income of Approved Borrowers by State")

inc_yr = st.select_slider("Select Year (Income Map)", options=YEARS, value=2017)
inc_st_yr = inc_st[inc_st["year"] == inc_yr].copy()
inc_st_yr["hover"] = (
    "<b>" + inc_st_yr["state_abbr"] + "</b><br>"
    "Median Income: $" + inc_st_yr["approved_income_median"].round(0).astype(int).astype(str) + "K<br>"
    "Median Loan: $" + inc_st_yr["approved_loan_median"].round(0).astype(int).astype(str) + "K<br>"
    "LTI Ratio: " + inc_st_yr["lti_median"].round(2).astype(str) + "×"
)

c_inc, c_loan = st.columns(2)
map_metric = c_inc.radio("Map shows:", ["Median Income", "Median Loan Amount", "LTI Ratio"],
                         horizontal=True)
metric_col = {"Median Income": "approved_income_median",
              "Median Loan Amount": "approved_loan_median",
              "LTI Ratio": "lti_median"}[map_metric]
metric_label = {"Median Income": "Median Income ($K)",
                "Median Loan Amount": "Median Loan ($K)",
                "LTI Ratio": "LTI Ratio (×)"}[map_metric]

fig6d = go.Figure(go.Choropleth(
    locations=inc_st_yr["state_abbr"],
    z=inc_st_yr[metric_col].round(1),
    locationmode="USA-states",
    colorscale="Blues",
    colorbar=dict(title=metric_label, thickness=15, len=0.7),
    hovertext=inc_st_yr["hover"],
    hoverinfo="text",
))
fig6d.update_layout(
    geo=dict(scope="usa", projection_type="albers usa", showlakes=True, lakecolor="lightblue"),
    height=430,
    title=dict(text=f"{map_metric} of Approved Borrowers — {inc_yr}", font=dict(size=14)),
    margin=dict(t=40, b=10, l=10, r=10),
    paper_bgcolor="white",
)
st.plotly_chart(fig6d, use_container_width=True)

# ── 6e. Mean Income vs Mean Loan Amount ───────────────────────────────────────
st.subheader("Mean Income vs Mean Loan Amount by Year")
st.caption(
    "Mean is pulled up by high-income/high-loan outliers — always higher than median. "
    "The gap between mean and median reveals income inequality among approved borrowers."
)

fig6e = make_subplots(rows=1, cols=2,
    subplot_titles=["Mean vs Median Income ($K) — Approved Borrowers",
                    "Mean vs Median Loan Amount ($K) — Originated Loans"],
    horizontal_spacing=0.10)

# Income chart
fig6e.add_trace(go.Scatter(
    x=inc_nat["year"], y=inc_nat["approved_income_mean"],
    name="Mean Income", mode="lines+markers+text",
    line=dict(color="#1a7abf", width=3),
    marker=dict(size=9),
    text=("$" + inc_nat["approved_income_mean"].round(0).astype(int).astype(str) + "K"),
    textposition="top center", textfont=dict(size=10),
), row=1, col=1)
fig6e.add_trace(go.Scatter(
    x=inc_nat["year"], y=inc_nat["approved_income_median"],
    name="Median Income", mode="lines+markers+text",
    line=dict(color="#1a7abf", width=2, dash="dot"),
    marker=dict(size=7),
    text=("$" + inc_nat["approved_income_median"].round(0).astype(int).astype(str) + "K"),
    textposition="bottom center", textfont=dict(size=10, color="#1a7abf"),
), row=1, col=1)

# Loan amount chart
fig6e.add_trace(go.Scatter(
    x=inc_nat["year"], y=inc_nat["approved_loan_mean"],
    name="Mean Loan", mode="lines+markers+text",
    line=dict(color="#27ae60", width=3),
    marker=dict(size=9),
    text=("$" + inc_nat["approved_loan_mean"].round(0).astype(int).astype(str) + "K"),
    textposition="top center", textfont=dict(size=10, color="#27ae60"),
), row=1, col=2)
fig6e.add_trace(go.Scatter(
    x=inc_nat["year"], y=inc_nat["approved_loan_median"],
    name="Median Loan", mode="lines+markers+text",
    line=dict(color="#27ae60", width=2, dash="dot"),
    marker=dict(size=7),
    text=("$" + inc_nat["approved_loan_median"].round(0).astype(int).astype(str) + "K"),
    textposition="bottom center", textfont=dict(size=10, color="#27ae60"),
), row=1, col=2)

for col_idx in [1, 2]:
    fig6e.add_vline(x=2008, line_dash="dash", line_color="#e64d1f", line_width=1.2, col=col_idx, row=1)

fig6e.update_layout(
    height=400,
    legend=dict(orientation="h", y=1.12),
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(t=60, b=40),
)
fig6e.update_xaxes(tickmode="linear", dtick=1, showgrid=False)
fig6e.update_yaxes(tickprefix="$", ticksuffix="K", showgrid=True,
                   gridcolor="rgba(0,0,0,0.06)", row=1, col=1)
fig6e.update_yaxes(tickprefix="$", ticksuffix="K", showgrid=True,
                   gridcolor="rgba(0,0,0,0.06)", row=1, col=2)
st.plotly_chart(fig6e, use_container_width=True)

# ── 6f. Full table ────────────────────────────────────────────────────────────
st.subheader("Full Income & Loan Summary Table (Approved Borrowers)")
tbl6 = inc_nat[[
    "year",
    "approved_income_mean", "approved_income_median",
    "approved_loan_mean",   "approved_loan_median",
    "lti_mean",             "lti_median",
    "income_per_100k_loan",
]].copy()
tbl6.columns = [
    "Year",
    "Mean Income ($K)", "Median Income ($K)",
    "Mean Loan ($K)",   "Median Loan ($K)",
    "Mean LTI",         "Median LTI",
    "Income/$100K Loan ($K)",
]
for col in tbl6.columns[1:]:
    tbl6[col] = tbl6[col].round(1)
st.dataframe(tbl6.reset_index(drop=True), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("HMDA data from CFPB/FFIEC, 2007–2017. Pipeline: Python / Pandas / Parquet. Dashboard: Streamlit / Plotly.")
