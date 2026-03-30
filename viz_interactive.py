"""
viz_interactive.py — Unified 4-panel interactive dashboard.
One year slider controls all charts simultaneously.

Panel 1 — Behaviour Shift: Home Purchase vs Refinancing vs Improvement
Panel 2 — Region Map: YoY change in originations by state
Panel 3 — Recovery Bar Race: Top/Bottom MSAs vs 2007 baseline
Panel 4 — Risk Appetite: Private lending share, LTI ratio, approval trends

Usage:
    streamlit run viz_interactive.py
"""

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

OUTPUT = Path("output")


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

st.set_page_config(
    page_title="HMDA Interactive Explorer",
    page_icon="📊",
    layout="wide",
)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    lp      = pd.read_parquet(OUTPUT / "viz_loan_purpose.parquet")
    state   = pd.read_parquet(OUTPUT / "viz_state_denial.parquet")
    msa_r   = pd.read_parquet(OUTPUT / "viz_msa_recovery.parquet")
    nat     = pd.read_parquet(OUTPUT / "viz_national.parquet")
    lt      = pd.read_parquet(OUTPUT / "viz_loan_type.parquet")
    inc_nat = pd.read_parquet(OUTPUT / "viz_income_national.parquet")
    inc_lp  = pd.read_parquet(OUTPUT / "viz_income_purpose.parquet")
    inc_st  = pd.read_parquet(OUTPUT / "viz_income_state.parquet")
    dr_nat  = pd.read_parquet(OUTPUT / "viz_denial_reasons_national.parquet")
    dr_full = pd.read_parquet(OUTPUT / "viz_denial_reasons.parquet")
    sig = pd.read_parquet(OUTPUT / "viz_significance.parquet")
    return lp, state, msa_r, nat, lt, inc_nat, inc_lp, inc_st, dr_nat, dr_full, sig

lp, state, msa_r, nat, lt, inc_nat, inc_lp, inc_st, dr_nat, dr_full, sig = load()

YEARS = sorted(nat["year"].unique().tolist())

# ── Pre-compute YoY state change ───────────────────────────────────────────────
state_sorted = state.sort_values(["state_abbr","year"])
state_sorted["prev_originated"] = state_sorted.groupby("state_abbr")["total_originated"].shift(1)
state_sorted["yoy_pct"] = (
    (state_sorted["total_originated"] - state_sorted["prev_originated"])
    / state_sorted["prev_originated"].clip(1) * 100
).round(1)

# ── Pre-compute risk appetite components ───────────────────────────────────────
lt["category"] = lt["loan_type"].map({1:"Conventional",2:"Govt",3:"Govt",4:"Govt"})
conv_share = (
    lt.groupby(["year","category"])["count"].sum()
    .reset_index()
    .pivot_table(index="year", columns="category", values="count")
    .fillna(0)
    .assign(conv_share=lambda d: d["Conventional"] / (d["Conventional"] + d["Govt"]) * 100)
    .reset_index()[["year","conv_share"]]
)
risk_df = (
    nat.merge(conv_share, on="year")
    .merge(inc_nat[["year","lti_median","approved_income_median","approved_loan_median"]], on="year")
)
risk_df["denial_rate"] = (risk_df["total_denied"] / risk_df["total_apps"] * 100).round(1)
risk_df["approval_rate"] = (100 - risk_df["denial_rate"]).round(1)
# Composite risk score (0–100): high = banks taking more risk
# Components: conv_share (40%) + LTI normalized (30%) + approval_rate normalized (30%)
risk_df["lti_norm"]      = ((risk_df["lti_median"] - 2.0) / 0.6 * 100).clip(0, 100)
risk_df["approval_norm"] = ((risk_df["approval_rate"] - 55) / 25 * 100).clip(0, 100)
risk_df["conv_norm"]     = ((risk_df["conv_share"] - 50) / 40 * 100).clip(0, 100)
risk_df["risk_score"]    = (
    risk_df["conv_norm"] * 0.4 +
    risk_df["lti_norm"]  * 0.3 +
    risk_df["approval_norm"] * 0.3
).round(1)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Market Hysteresis: Why the 2008 Ghost Still Haunts American Lending")
st.caption("HMDA Data Analysis 2007–2017  •  150M+ loan applications across 50 states")

# ══════════════════════════════════════════════════════════════════════════════
# MASTER YEAR SLIDER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
col_sl, col_info = st.columns([3, 1])
with col_sl:
    year = st.select_slider(
        "📅 Select Year — controls all panels",
        options=YEARS,
        value=2009,
    )
with col_info:
    prev_year = year - 1 if year > min(YEARS) else None
    nat_row  = nat[nat["year"] == year].iloc[0]
    nat_prev = nat[nat["year"] == prev_year].iloc[0] if prev_year else None
    orig_yoy = (
        f"{(nat_row['total_originated']/nat_prev['total_originated']-1)*100:+.1f}% vs {prev_year}"
        if nat_prev is not None else "—"
    )
    st.metric(f"Originations {year}", f"{int(nat_row['total_originated']):,}", orig_yoy)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1: Behaviour Shift | Region Map
# ══════════════════════════════════════════════════════════════════════════════
col1, col2 = st.columns([1, 1])

# ── Panel 1: Behaviour Shift ──────────────────────────────────────────────────
with col1:
    st.subheader("1. Behaviour Shift: Buy vs Refi vs Improve")

    lp_pivot = lp.pivot_table(
        index="year", columns="loan_purpose_label", values="count", aggfunc="sum"
    ).fillna(0).reset_index()

    COLORS_LP = {
        "Home Purchase":    "#2980b9",
        "Refinancing":      "#e74c3c",
        "Home Improvement": "#f39c12",
    }

    fig1 = go.Figure()
    for label, color in COLORS_LP.items():
        if label not in lp_pivot.columns:
            continue
        opacities = [1.0 if y == year else 0.25 for y in lp_pivot["year"]]
        fig1.add_trace(go.Bar(
            x=lp_pivot["year"],
            y=lp_pivot[label],
            name=label,
            marker_color=color,
            opacity=1.0,
            marker=dict(
                color=[color] * len(lp_pivot),
                opacity=opacities,
            ),
        ))

    # Highlight selected year
    fig1.add_vline(x=year, line_color="black", line_width=2, line_dash="dot")

    # Annotate dominant behaviour for selected year
    yr_data = lp_pivot[lp_pivot["year"] == year].iloc[0]
    purposes = {k: yr_data[k] for k in COLORS_LP if k in yr_data}
    dominant = max(purposes, key=purposes.get)
    total_yr = sum(purposes.values())
    dom_pct  = purposes[dominant] / total_yr * 100

    fig1.add_annotation(
        x=year, y=total_yr * 1.05,
        text=f"<b>{dominant.split()[0]}<br>dominates<br>{dom_pct:.0f}%</b>",
        showarrow=False, font=dict(size=10, color=COLORS_LP[dominant]),
        bgcolor="rgba(255,255,255,0.9)", bordercolor=COLORS_LP[dominant], borderwidth=1,
    )

    fig1.update_layout(
        barmode="stack", height=380,
        legend=dict(orientation="h", y=1.08, font=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(tickmode="linear", dtick=1, showgrid=False, title="Year"),
        yaxis=dict(title="Applications", tickformat=",",
                   showgrid=True, gridcolor="rgba(0,0,0,0.07)"),
        margin=dict(t=50, b=40, l=10, r=10),
    )
    fig1 = significance_badge(fig1, sig, "Application Volume Drop", x=0.01, y=0.97)
    fig1 = significance_badge(fig1, sig, "Approval Rate Structural Break (2008)", x=0.01, y=0.85)
    st.plotly_chart(fig1, use_container_width=True)

    # Mini stats
    if yr_data is not None:
        c1a, c1b, c1c = st.columns(3)
        for (label, color), col_m in zip(COLORS_LP.items(), [c1a, c1b, c1c]):
            val = purposes.get(label, 0)
            col_m.metric(label.split()[0], f"{val/1e6:.2f}M",
                         f"{val/total_yr*100:.1f}% share")

# ── Panel 2: Region Map YoY Change ────────────────────────────────────────────
with col2:
    st.subheader("2. Region-wise Origination Change (YoY %)")

    state_yr = state_sorted[state_sorted["year"] == year].copy()
    if prev_year:
        map_title = f"Origination Change vs {prev_year} (%)"
        z_col     = "yoy_pct"
        colorscale = [
            [0.0, "#d32f2f"], [0.3, "#ef9a9a"], [0.45, "#ffcdd2"],
            [0.5, "#f5f5f5"],
            [0.55, "#c8e6c9"], [0.7, "#66bb6a"], [1.0, "#1b5e20"],
        ]
        zmid  = 0
        zmin  = max(state_yr["yoy_pct"].min(), -60)
        zmax  = min(state_yr["yoy_pct"].max(),  60)
        cbar_title = "YoY Change (%)"
        cbar_suffix = "%"
    else:
        map_title = f"Originations {year}"
        z_col     = "total_originated"
        colorscale = "Blues"
        zmid  = None
        zmin  = state_yr["total_originated"].min()
        zmax  = state_yr["total_originated"].max()
        cbar_title = "Originations"
        cbar_suffix = ""

    state_yr["hover"] = (
        "<b>" + state_yr["state_name"] + "</b><br>" +
        ("YoY Change: " + state_yr["yoy_pct"].astype(str) + "%<br>" if prev_year else "") +
        "Originations: " + state_yr["total_originated"].map("{:,}".format) + "<br>" +
        "Denial Rate: " + (state_yr["denial_rate"] * 100).round(1).astype(str) + "%"
    )

    fig2 = go.Figure(go.Choropleth(
        locations=state_yr["state_abbr"],
        z=state_yr[z_col],
        locationmode="USA-states",
        colorscale=colorscale,
        zmid=zmid,
        zmin=zmin, zmax=zmax,
        colorbar=dict(title=cbar_title, ticksuffix=cbar_suffix, thickness=14, len=0.6),
        hovertext=state_yr["hover"],
        hoverinfo="text",
    ))
    fig2.update_layout(
        geo=dict(scope="usa", projection_type="albers usa",
                 showlakes=True, lakecolor="lightblue"),
        height=380,
        title=dict(text=map_title, font=dict(size=13)),
        margin=dict(t=40, b=5, l=5, r=5),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig2, use_container_width=True)

    if prev_year:
        gaining = (state_yr["yoy_pct"] > 0).sum()
        losing  = (state_yr["yoy_pct"] < 0).sum()
        best    = state_yr.nlargest(1,"yoy_pct").iloc[0]
        worst   = state_yr.nsmallest(1,"yoy_pct").iloc[0]
        c2a, c2b, c2c = st.columns(3)
        c2a.metric("States Growing", f"{gaining}", f"{losing} declining")
        c2b.metric(f"Best: {best['state_abbr']}", f"+{best['yoy_pct']:.1f}%")
        c2c.metric(f"Worst: {worst['state_abbr']}", f"{worst['yoy_pct']:.1f}%")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2: Recovery Bar Race | Risk Appetite
# ══════════════════════════════════════════════════════════════════════════════
col3, col4 = st.columns([1, 1])

# ── Panel 3: Recovery Bar Race ────────────────────────────────────────────────
with col3:
    st.subheader("3. Recovery Race: MSAs vs 2007 Baseline")

    msa_all = msa_r[msa_r["year"] == year].dropna(subset=["msa_md_name"]).copy()

    top10    = msa_all.nlargest(10, "recovery_index").sort_values("recovery_index", ascending=True)
    bottom10 = msa_all.nsmallest(10, "recovery_index").sort_values("recovery_index", ascending=True)

    # shorten city names to first city only
    top10["city"]    = top10["msa_md_name"].str.split("-").str[0].str.strip()
    bottom10["city"] = bottom10["msa_md_name"].str.split("-").str[0].str.strip()

    fig3 = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"🟢 Top 10 Recovering Cities — {year}",
            f"🔴 Bottom 10 Lagging Cities — {year}",
        ],
        horizontal_spacing=0.18,
    )

    # Top 10 — green bars
    fig3.add_trace(go.Bar(
        y=top10["city"],
        x=top10["recovery_index"],
        orientation="h",
        marker_color="#27ae60",
        text=top10["recovery_index"].round(1).astype(str),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Recovery Index: %{x:.1f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # Bottom 10 — red bars
    fig3.add_trace(go.Bar(
        y=bottom10["city"],
        x=bottom10["recovery_index"],
        orientation="h",
        marker_color="#e74c3c",
        text=bottom10["recovery_index"].round(1).astype(str),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Recovery Index: %{x:.1f}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    # Baseline vlines on both
    for col_i in [1, 2]:
        fig3.add_vline(x=100, line_dash="dash", line_color="#333",
                       line_width=1.5, col=col_i, row=1)

    x_max_top = max(top10["recovery_index"].max() * 1.2, 130)
    x_max_bot = max(bottom10["recovery_index"].max() * 1.4, 60)

    fig3.update_xaxes(title_text="Recovery Index (2007=100)",
                      range=[0, x_max_top],
                      showgrid=True, gridcolor="rgba(0,0,0,0.06)", row=1, col=1)
    fig3.update_xaxes(title_text="Recovery Index (2007=100)",
                      range=[0, x_max_bot],
                      showgrid=True, gridcolor="rgba(0,0,0,0.06)", row=1, col=2)
    fig3.update_yaxes(tickfont=dict(size=9), row=1, col=1)
    fig3.update_yaxes(tickfont=dict(size=9), row=1, col=2)

    above_100 = (msa_all["recovery_index"] >= 100).sum()
    total_msa = len(msa_all)

    fig3.update_layout(
        height=420,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=50, b=40, l=10, r=80),
    )
    fig3 = significance_badge(fig3, sig, "MSA Recovery Gap (Top 10 vs Bottom 10)", x=0.01, y=0.97)
    st.plotly_chart(fig3, use_container_width=True)

    c3a, c3b, c3c = st.columns(3)
    below_100 = total_msa - above_100
    c3a.metric("Recovered (≥100)", str(above_100), f"{below_100} still below")
    c3b.metric("Fastest", top10.iloc[-1]["city"], f"{top10.iloc[-1]['recovery_index']:.0f}")
    c3c.metric("Slowest", bottom10.iloc[0]["city"], f"{bottom10.iloc[0]['recovery_index']:.0f}")

# ── Panel 4: Risk Appetite ────────────────────────────────────────────────────
with col4:
    st.subheader("4. Lender Risk Appetite")

    risk_yr   = risk_df[risk_df["year"] == year].iloc[0]
    risk_score = risk_yr["risk_score"]

    # Gauge
    fig4 = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        delta={"reference": risk_df[risk_df["year"]==2007].iloc[0]["risk_score"],
               "valueformat": ".1f",
               "suffix": " vs 2007"},
        title={"text": f"Risk Appetite Score — {year}<br>"
                       f"<span style='font-size:12px;color:gray'>"
                       f"Based on: Private lending share, LTI ratio, Approval rate</span>"},
        number={"suffix": " / 100"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#1a7abf"},
            "steps": [
                {"range": [0,  33], "color": "#d5e8f7"},
                {"range": [33, 66], "color": "#aed4f0"},
                {"range": [66, 100],"color": "#78b4e0"},
            ],
            "threshold": {
                "line": {"color": "#e74c3c", "width": 3},
                "thickness": 0.75,
                "value": risk_df[risk_df["year"]==2007].iloc[0]["risk_score"],
            },
        }
    ))
    fig4.update_layout(height=250, margin=dict(t=80, b=10, l=20, r=20),
                       paper_bgcolor="white")
    st.plotly_chart(fig4, use_container_width=True)

    # 3 component bars
    components = {
        "Private Lending Share": (risk_yr["conv_share"], 100, "%",  "#1a7abf"),
        "LTI Ratio (Leverage)":  (risk_yr["lti_median"], 3.0, "×",  "#e67e22"),
        "Approval Rate":         (risk_yr["approval_rate"], 100, "%", "#27ae60"),
    }
    fig4b = go.Figure()
    for i, (label, (val, max_val, suffix, color)) in enumerate(components.items()):
        fig4b.add_trace(go.Bar(
            x=[val], y=[label],
            orientation="h",
            marker_color=color,
            text=[f"{val:.1f}{suffix}"],
            textposition="outside",
            name=label, showlegend=False,
        ))
    fig4b.update_layout(
        height=170,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", range=[0, 110]),
        yaxis=dict(showgrid=False),
        margin=dict(t=10, b=30, l=10, r=60),
        barmode="group",
    )
    st.plotly_chart(fig4b, use_container_width=True)

    # Risk trend line (all years)
    fig4c = go.Figure()
    fig4c.add_trace(go.Scatter(
        x=risk_df["year"], y=risk_df["risk_score"],
        mode="lines+markers",
        line=dict(color="#1a7abf", width=2.5),
        marker=dict(size=[14 if y == year else 7 for y in risk_df["year"]],
                    color=["#e74c3c" if y == year else "#1a7abf" for y in risk_df["year"]]),
        hovertemplate="Year: %{x}<br>Risk Score: %{y:.1f}<extra></extra>",
    ))
    fig4c.add_hline(y=50, line_dash="dot", line_color="#aaa",
                    annotation_text="Neutral", annotation_position="right")
    fig4c.update_layout(
        height=140,
        xaxis=dict(tickmode="linear", dtick=1, showgrid=False, title=""),
        yaxis=dict(title="Score", range=[0, 100],
                   showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=10, b=30, l=10, r=10),
    )
    st.plotly_chart(fig4c, use_container_width=True)

st.markdown("---")

# ── Bottom summary ─────────────────────────────────────────────────────────────
st.subheader(f"📋 Year {year} Snapshot")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Applications",  f"{int(nat_row['total_loans']):,}")
c2.metric("Total Originated",    f"{int(nat_row['total_originated']):,}")
c3.metric("Govt Loan Share",     f"{nat_row['govt_share']:.1%}")
risk_row = risk_df[risk_df["year"]==year].iloc[0]
c4.metric("Risk Score",          f"{risk_row['risk_score']:.1f}/100")
c5.metric("Private Lending Share", f"{risk_row['conv_share']:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# FULL DASHBOARD — ALL PANELS (from viz_dashboard.py)
# ══════════════════════════════════════════════════════════════════════════════

import plotly.express as px
from plotly.subplots import make_subplots as _make_subplots

MORTGAGE_RATES = pd.DataFrame({
    "year":       list(range(2007, 2018)),
    "rate_30yr":  [6.34, 6.03, 5.04, 4.69, 4.45, 3.66, 3.98, 4.17, 3.85, 3.65, 3.99],
    "fed_funds":  [5.02, 1.93, 0.24, 0.18, 0.10, 0.14, 0.11, 0.09, 0.13, 0.40, 1.00],
})

BOTTOM_YEAR = int(nat.loc[nat["total_originated"].idxmin(), "year"])
INFLECTION_YEAR = int(
    nat[nat["year"] > BOTTOM_YEAR]
    .assign(yoy=lambda d: d["total_originated"].pct_change())
    .query("yoy > 0")
    .iloc[0]["year"]
)

st.markdown("---")
st.header("Full Dashboard — Deep Dive by Panel")
st.caption(f"Crash bottom: **{BOTTOM_YEAR}**  •  Recovery inflection: **{INFLECTION_YEAR}**")
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — Crash & Recovery National Trend
# ══════════════════════════════════════════════════════════════════════════════
st.header("1. National Crash & Recovery")

nat["approval_rate"] = ((nat["total_apps"] - nat["total_denied"]) / nat["total_loans"].clip(1)).round(4)
nat["orig_rate_all"]  = (nat["total_originated"] / nat["total_loans"].clip(1)).round(4)

fig1 = _make_subplots(specs=[[{"secondary_y": True}]])

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

fig1.add_vline(x=BOTTOM_YEAR, line_dash="dash", line_color="#e64d1f", line_width=1.5)
fig1.add_annotation(
    x=BOTTOM_YEAR, y=1, yref="paper",
    text=f"<b>Crash Bottom</b><br>{BOTTOM_YEAR}",
    showarrow=True, arrowhead=2, arrowcolor="#e64d1f",
    ax=40, ay=-60, font=dict(color="#e64d1f", size=12),
    bgcolor="rgba(255,255,255,0.85)",
)

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

# ── Interest Rate Chart ──────────────────────────────────────────────────────
st.subheader("30-Year Fixed Mortgage Rate (Freddie Mac PMMS)")

nat_rate = nat.merge(MORTGAGE_RATES, on="year")

fig1r = _make_subplots(specs=[[{"secondary_y": True}]])
fig1r.add_trace(go.Scatter(
    x=nat_rate["year"], y=nat_rate["rate_30yr"],
    name="30-Yr Fixed Rate (%)", mode="lines+markers+text",
    line=dict(color="#c0392b", width=3), marker=dict(size=10),
    text=nat_rate["rate_30yr"].astype(str) + "%",
    textposition="top center", textfont=dict(size=11, color="#c0392b"),
    fill="tozeroy", fillcolor="rgba(192,57,43,0.08)",
), secondary_y=False)

fig1r.add_trace(go.Bar(
    x=nat_rate["year"], y=nat_rate["total_originated"],
    name="Total Originated",
    marker_color="rgba(30,100,220,0.25)", marker_line_width=0,
), secondary_y=True)

fig1r.add_trace(go.Scatter(
    x=nat_rate["year"], y=nat_rate["fed_funds"],
    name="Fed Funds Rate (%)", mode="lines+markers+text",
    line=dict(color="#1a7abf", width=2.5, dash="dash"),
    marker=dict(size=8, symbol="square"),
    text=nat_rate["fed_funds"].astype(str) + "%",
    textposition="bottom center", textfont=dict(size=10, color="#1a7abf"),
), secondary_y=False)

RATE_EVENTS = [
    (2008, "Fed cuts rates<br>to near zero", "#e67e22", 30, -50),
    (2009, "Refi Boom #1<br>5.04%", "#27ae60", -50, -50),
    (2012, "Refi Boom #2<br>Record low 3.66%", "#27ae60", -60, -50),
    (2013, "Taper Tantrum<br>rates jump", "#c0392b", 40, -50),
    (2014, "Mortgage bottom<br>refi boom ends", "#c0392b", 40, -50),
]
for ev_year, ev_text, ev_color, ev_ax, ev_ay in RATE_EVENTS:
    ev_rate = MORTGAGE_RATES[MORTGAGE_RATES["year"] == ev_year]["rate_30yr"].values[0]
    fig1r.add_annotation(
        x=ev_year, y=ev_rate, yref="y",
        text=f"<b>{ev_text}</b>",
        showarrow=True, arrowhead=2, arrowcolor=ev_color,
        ax=ev_ax, ay=ev_ay,
        font=dict(color=ev_color, size=10),
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor=ev_color, borderwidth=1,
    )

fig1r.update_layout(
    height=400, legend=dict(orientation="h", y=1.1),
    plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(tickmode="linear", dtick=1, title="Year", showgrid=False),
    yaxis=dict(title="30-Yr Fixed Rate (%)", range=[2.5, 7.5],
               ticksuffix="%", showgrid=True, gridcolor="rgba(0,0,0,0.07)"),
    yaxis2=dict(title="Originations", tickformat=",", showgrid=False),
    margin=dict(t=40, b=40),
)
st.plotly_chart(fig1r, use_container_width=True)

rate_table = nat_rate[["year","fed_funds","rate_30yr","total_originated","total_loans"]].copy()
rate_table["orig_M"]    = (rate_table["total_originated"] / 1e6).round(2).astype(str) + "M"
rate_table["apps_M"]    = (rate_table["total_loans"]      / 1e6).round(2).astype(str) + "M"
rate_table["rate_30yr"] = rate_table["rate_30yr"].astype(str) + "%"
rate_table["fed_funds"] = rate_table["fed_funds"].astype(str) + "%"
rate_table["spread"]    = (nat_rate["rate_30yr"] - nat_rate["fed_funds"]).round(2).astype(str) + "%"
rate_table = rate_table[["year","fed_funds","rate_30yr","spread","apps_M","orig_M"]]
rate_table = rate_table.drop(columns=["total_loans"], errors="ignore")
rate_table.columns = ["Year","Fed Funds Rate","30-Yr Mortgage Rate","Spread","Applications","Originated"]
st.dataframe(rate_table.reset_index(drop=True), use_container_width=True, hide_index=True)

# ── Applications by Loan Purpose ─────────────────────────────────────────────
st.subheader("Applications by Loan Purpose Over the Years")
st.caption("Shows exactly what drove spikes and crashes in total application volume.")

lp_apps = lp.pivot_table(index="year", columns="loan_purpose_label", values="count", aggfunc="sum").fillna(0).reset_index()

COLORS_LP_D = {
    "Refinancing":      "#e74c3c",
    "Home Purchase":    "#2980b9",
    "Home Improvement": "#f39c12",
}

fig1p = go.Figure()
for lbl, clr in COLORS_LP_D.items():
    if lbl in lp_apps.columns:
        fig1p.add_trace(go.Bar(
            x=lp_apps["year"], y=lp_apps[lbl], name=lbl,
            marker_color=clr,
            hovertemplate=f"<b>{lbl}</b><br>Year: %{{x}}<br>Applications: %{{y:,.0f}}<extra></extra>",
        ))

fig1p.add_vline(x=2009, line_dash="dot", line_color="#27ae60", line_width=1.5)
fig1p.add_annotation(x=2009, y=1, yref="paper", text="<b>Refi Boom #1</b>",
    showarrow=False, font=dict(color="#27ae60", size=10), bgcolor="rgba(255,255,255,0.85)")
fig1p.add_vline(x=2012, line_dash="dot", line_color="#27ae60", line_width=1.5)
fig1p.add_annotation(x=2012, y=1, yref="paper", text="<b>Refi Boom #2</b>",
    showarrow=False, font=dict(color="#27ae60", size=10), bgcolor="rgba(255,255,255,0.85)")
fig1p.add_vline(x=2014, line_dash="dash", line_color="#e64d1f", line_width=1.5)
fig1p.add_annotation(x=2014, y=1, yref="paper", text="<b>Refi Collapses</b>",
    showarrow=False, font=dict(color="#e64d1f", size=10), bgcolor="rgba(255,255,255,0.85)")

fig1p.update_layout(
    barmode="stack", height=420,
    legend=dict(orientation="h", y=1.1),
    plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(tickmode="linear", dtick=1, title="Year", showgrid=False),
    yaxis=dict(title="Number of Applications", tickformat=",",
               showgrid=True, gridcolor="rgba(0,0,0,0.07)"),
    margin=dict(t=50, b=40),
)
st.plotly_chart(fig1p, use_container_width=True)

lp_tbl = lp_apps.copy()
for col in ["Home Purchase","Refinancing","Home Improvement"]:
    if col in lp_tbl.columns:
        lp_tbl[col] = lp_tbl[col].astype(int).map("{:,}".format)
lp_tbl["Total"] = lp.groupby("year")["count"].sum().reset_index()["count"].astype(int).map("{:,}".format)
st.dataframe(lp_tbl.reset_index(drop=True), use_container_width=True, hide_index=True)

# ── Conventional vs Govt ──────────────────────────────────────────────────────
st.subheader("Conventional vs Government-Backed: Unmasking Lender Tightening")
st.caption(
    "The aggregate approval rate masked two opposite trends. "
    "Conventional banks tightened dramatically — but only better applicants applied. "
    "Government loans absorbed the riskier borrowers conventional banks rejected."
)

lt["category_d"] = lt["loan_type"].map({
    1: "Conventional (Private)",
    2: "Govt-Backed (FHA/VA/FSA)",
    3: "Govt-Backed (FHA/VA/FSA)",
    4: "Govt-Backed (FHA/VA/FSA)",
})
cat_df = lt.groupby(["year","category_d"]).agg(count=("count","sum"), originated=("originated","sum")).reset_index()
cat_df["orig_rate"] = (cat_df["originated"] / cat_df["count"].clip(1)).round(4)

fig1c = _make_subplots(rows=1, cols=2,
    subplot_titles=["Origination Rate: Conventional vs Govt (%)", "Application Volume: Conventional vs Govt"],
    horizontal_spacing=0.10)

COLORS_CAT = {"Conventional (Private)": "#1a7abf", "Govt-Backed (FHA/VA/FSA)": "#e67e22"}

for cat, color in COLORS_CAT.items():
    subset_c = cat_df[cat_df["category_d"] == cat].sort_values("year")
    fig1c.add_trace(go.Scatter(
        x=subset_c["year"], y=(subset_c["orig_rate"]*100).round(1),
        name=cat, mode="lines+markers",
        line=dict(color=color, width=2.5), marker=dict(size=8),
        legendgroup=cat,
        hovertemplate=f"<b>{cat}</b><br>%{{y:.1f}}%<extra></extra>",
    ), row=1, col=1)
    fig1c.add_trace(go.Bar(
        x=subset_c["year"], y=subset_c["count"],
        name=cat, marker_color=color,
        legendgroup=cat, showlegend=False,
        hovertemplate=f"<b>{cat}</b><br>Apps: %{{y:,.0f}}<extra></extra>",
    ), row=1, col=2)

fig1c.add_annotation(
    x=2012, y=56, xref="x", yref="y",
    text="<b>Rising rate = only<br>qualified applicants left</b>",
    showarrow=True, arrowhead=2, arrowcolor="#1a7abf",
    ax=60, ay=-40, font=dict(color="#1a7abf", size=10),
    bgcolor="rgba(255,255,255,0.88)",
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

conv_2007 = cat_df[(cat_df["year"]==2007) & (cat_df["category_d"]=="Conventional (Private)")].iloc[0]
conv_2009 = cat_df[(cat_df["year"]==2009) & (cat_df["category_d"]=="Conventional (Private)")].iloc[0]
govt_2007 = cat_df[(cat_df["year"]==2007) & (cat_df["category_d"]=="Govt-Backed (FHA/VA/FSA)")].iloc[0]
govt_2009 = cat_df[(cat_df["year"]==2009) & (cat_df["category_d"]=="Govt-Backed (FHA/VA/FSA)")].iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("Conventional Apps 2007→2009",
          f"{int(conv_2009['count']):,}",
          f"{int(conv_2009['count']-conv_2007['count']):,} ({(conv_2009['count']/conv_2007['count']-1):.0%})")
c2.metric("Govt-Backed Apps 2007→2009",
          f"{int(govt_2009['count']):,}",
          f"+{int(govt_2009['count']-govt_2007['count']):,} (+{(govt_2009['count']/govt_2007['count']-1):.0%})")
c3.metric("Conventional Orig Rate 2007→2017",
          f"{conv_2009['orig_rate']:.1%} (2009)",
          f"Started at {conv_2007['orig_rate']:.1%} in 2007")

st.info(
    "**The paradox explained:** Conventional banks DID tighten dramatically after 2008 — "
    "but the riskiest borrowers had already stopped applying. The remaining applicant pool was "
    "so much stronger that the approval rate actually *rose*, masking the tightening. "
    "Meanwhile, government-backed loans absorbed 4× more applicants — the people conventional "
    "banks turned away."
)

# ── Approval Rate by Loan Purpose ─────────────────────────────────────────────
st.subheader("Approval & Origination Rate by Loan Purpose")
st.caption("Approval Rate = bank-approved loans / all applications (incl. withdrawals). Origination Rate = fully closed loans / all applications.")

lp["approval_rate"] = (lp["approved"]   / lp["count"].clip(1)).round(4)
lp["orig_rate"]     = (lp["originated"] / lp["count"].clip(1)).round(4)

COLORS_PURPOSE = {
    "Home Purchase":    ("#2980b9", "#aed6f1"),
    "Refinancing":      ("#e74c3c", "#f5b7b1"),
    "Home Improvement": ("#f39c12", "#fde8a1"),
}

fig1b = _make_subplots(
    rows=1, cols=2,
    subplot_titles=["Approval Rate by Loan Purpose (%)", "Origination Rate by Loan Purpose (%)"],
    horizontal_spacing=0.10,
)

for lbl, (solid, light) in COLORS_PURPOSE.items():
    subset_p = lp[lp["loan_purpose_label"] == lbl].sort_values("year")
    if subset_p.empty:
        continue
    fig1b.add_trace(go.Scatter(
        x=subset_p["year"], y=(subset_p["approval_rate"] * 100).round(1),
        name=lbl, mode="lines+markers",
        line=dict(color=solid, width=2.5), marker=dict(size=7),
        legendgroup=lbl,
        hovertemplate=f"<b>{lbl}</b><br>Approval Rate: %{{y:.1f}}%<extra></extra>",
    ), row=1, col=1)
    fig1b.add_trace(go.Scatter(
        x=subset_p["year"], y=(subset_p["orig_rate"] * 100).round(1),
        name=lbl, mode="lines+markers",
        line=dict(color=solid, width=2.5, dash="dot"), marker=dict(size=7, symbol="diamond"),
        legendgroup=lbl, showlegend=False,
        hovertemplate=f"<b>{lbl}</b><br>Orig Rate: %{{y:.1f}}%<extra></extra>",
    ), row=1, col=2)

for col_idx in [1, 2]:
    fig1b.add_vline(x=BOTTOM_YEAR, line_dash="dash", line_color="#e64d1f", line_width=1.2, col=col_idx, row=1)
    fig1b.add_vline(x=INFLECTION_YEAR, line_dash="dot", line_color="#2ab548", line_width=1.2, col=col_idx, row=1)

fig1b.update_layout(
    height=380, legend=dict(orientation="h", y=1.12),
    plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=50, b=40),
)
fig1b.update_xaxes(tickmode="linear", dtick=1, showgrid=False)
fig1b.update_yaxes(title_text="Rate (%)", ticksuffix="%", showgrid=True,
                   gridcolor="rgba(0,0,0,0.06)", range=[30, 100])
st.plotly_chart(fig1b, use_container_width=True)

lp_latest = lp[lp["year"] == 2017][["loan_purpose_label","count","originated","denied","approval_rate","orig_rate"]].copy()
lp_latest.columns = ["Loan Purpose","Applications","Originated","Denied","Approval Rate","Orig Rate"]
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
    colorbar=dict(title="Denial Rate (%)", ticksuffix="%", thickness=15, len=0.7),
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
fig2 = significance_badge(fig2, sig, "Denial Rate Differences Across States", x=0.01, y=0.97)
st.plotly_chart(fig2, use_container_width=True)

if selected_year > min(YEARS):
    prev_state_df = state[state["year"] == selected_year - 1][["state_abbr","denial_rate"]].rename(columns={"denial_rate":"prev_denial"})
    merged_st = state_year.merge(prev_state_df, on="state_abbr")
    merged_st["delta"] = ((merged_st["denial_rate"] - merged_st["prev_denial"]) * 100).round(2)
    top_increase = merged_st.nlargest(5,"delta")[["state_name","denial_pct","delta"]]
    top_decrease = merged_st.nsmallest(5,"delta")[["state_name","denial_pct","delta"]]
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Top 5 Denial Rate Increases vs {selected_year-1}")
        top_increase.columns = ["State","Denial Rate (%)","Change (pp)"]
        st.dataframe(top_increase.reset_index(drop=True), use_container_width=True, hide_index=True)
    with c2:
        st.subheader(f"Top 5 Denial Rate Decreases vs {selected_year-1}")
        top_decrease.columns = ["State","Denial Rate (%)","Change (pp)"]
        st.dataframe(top_decrease.reset_index(drop=True), use_container_width=True, hide_index=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3 — Government Backstop
# ══════════════════════════════════════════════════════════════════════════════
st.header("3. The Government Backstop: When Private Capital Fled")

lt_pivot_d = lt.pivot_table(index="year", columns="loan_type_label", values="count", aggfunc="sum").fillna(0)
total_by_year_d = lt_pivot_d.sum(axis=1)
lt_share = lt_pivot_d.div(total_by_year_d, axis=0) * 100

lp_pivot_d = lp.pivot_table(index="year", columns="loan_purpose_label", values="originated", aggfunc="sum").fillna(0)
lp_total_d = lp_pivot_d.sum(axis=1)
lp_share = lp_pivot_d.div(lp_total_d, axis=0) * 100

fig3 = _make_subplots(
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
for lbl, clr in COLORS_LT.items():
    if lbl in lt_share.columns:
        fig3.add_trace(go.Scatter(
            x=lt_share.index, y=lt_share[lbl].round(1),
            name=lbl, stackgroup="lt", mode="lines",
            line=dict(width=0.5, color=clr),
            hovertemplate="%{y:.1f}%<extra>" + lbl + "</extra>",
        ), row=1, col=1)

COLORS_LP3 = {
    "Home Purchase": "#2980b9",
    "Refinancing": "#e74c3c",
    "Home Improvement": "#f39c12",
}
for lbl, clr in COLORS_LP3.items():
    if lbl in lp_share.columns:
        fig3.add_trace(go.Scatter(
            x=lp_share.index, y=lp_share[lbl].round(1),
            name=lbl, stackgroup="lp", mode="lines",
            line=dict(width=0.5, color=clr),
            hovertemplate="%{y:.1f}%<extra>" + lbl + "</extra>",
        ), row=1, col=2)

for col_idx in [1, 2]:
    fig3.add_vline(x=2008, line_dash="dash", line_color="#e64d1f", line_width=1.2, col=col_idx, row=1)

fig3.update_layout(
    height=420,
    legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(t=50, b=80),
)
fig3.update_xaxes(tickmode="linear", dtick=1, showgrid=False)
fig3.update_yaxes(title_text="Share (%)", ticksuffix="%", showgrid=True,
                  gridcolor="rgba(0,0,0,0.06)", row=1, col=1)
fig3.update_yaxes(title_text="Share (%)", ticksuffix="%", showgrid=True,
                  gridcolor="rgba(0,0,0,0.06)", row=1, col=2)
st.plotly_chart(fig3, use_container_width=True)

govt_2007_share = nat[nat["year"] == 2007].iloc[0]["govt_share"]
govt_peak  = nat.loc[nat["govt_share"].idxmax()]
conv_2007_v = lt[(lt["year"] == 2007) & (lt["loan_type_label"] == "Conventional (Private)")]["count"].sum()
conv_2009_v = lt[(lt["year"] == 2009) & (lt["loan_type_label"] == "Conventional (Private)")]["count"].sum()

c1, c2, c3 = st.columns(3)
c1.metric("Govt Loan Share (2007)", f"{govt_2007_share:.1%}")
c2.metric(f"Govt Loan Share Peak ({int(govt_peak['year'])})", f"{govt_peak['govt_share']:.1%}",
          f"+{(govt_peak['govt_share']-govt_2007_share)*100:.1f} pp vs 2007")
c3.metric("Conventional Apps Drop (2007→2009)", f"{int(conv_2009_v - conv_2007_v):,}",
          f"{(conv_2009_v/conv_2007_v - 1):.1%}")

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
recovered_n = (msa_year["recovery_index"] >= 100).sum()

fig4a = go.Figure()
for lbl, clr, cond in [
    ("Recovered (≥100)", "#27ae60", msa_year["recovery_index"] >= 100),
    ("Partial (70–99)",  "#e67e22", (msa_year["recovery_index"] >= 70) & (msa_year["recovery_index"] < 100)),
    ("Lagging (<70)",    "#e74c3c", msa_year["recovery_index"] < 70),
]:
    sub_m = msa_year[cond]
    fig4a.add_trace(go.Scatter(
        x=sub_m["total_apps"], y=sub_m["recovery_index"],
        mode="markers", name=lbl,
        marker=dict(color=clr, size=7, opacity=0.7, line=dict(width=0.5, color="white")),
        hovertemplate="<b>%{customdata}</b><br>Recovery Index: %{y:.1f}<br>Apps: %{x:,}<extra></extra>",
        customdata=sub_m["msa_md_name"],
    ))

fig4a.add_hline(y=100, line_dash="dash", line_color="#555", line_width=1.5,
                annotation_text="2007 Baseline (100)", annotation_position="top right")
fig4a.update_layout(
    height=420,
    xaxis=dict(title="Total Applications (log scale)", type="log",
               showgrid=True, gridcolor="rgba(0,0,0,0.07)"),
    yaxis=dict(title="Recovery Index (2007=100)", showgrid=True, gridcolor="rgba(0,0,0,0.07)"),
    plot_bgcolor="white", paper_bgcolor="white",
    legend=dict(orientation="h", y=1.07),
    margin=dict(t=40, b=50),
    title=dict(text=f"MSA Recovery Index — {idx_year}  ({recovered_n}/{total_msas} MSAs at or above 2007 levels)",
               font=dict(size=14)),
)
st.plotly_chart(fig4a, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    top_m = msa_year.nlargest(top_n, "recovery_index")
    fig_top = go.Figure(go.Bar(
        y=top_m["msa_md_name"], x=top_m["recovery_index"],
        orientation="h",
        marker_color=top_m["recovery_index"].apply(lambda v: "#27ae60" if v >= 100 else "#e67e22"),
        text=top_m["recovery_index"].round(1), textposition="outside",
    ))
    fig_top.add_vline(x=100, line_dash="dash", line_color="#555")
    fig_top.update_layout(
        title=f"Top {top_n} Most Recovered MSAs ({idx_year})",
        height=max(400, top_n * 22),
        margin=dict(t=40, b=30, l=10, r=60),
        xaxis=dict(title="Recovery Index", range=[0, max(top_m["recovery_index"].max()*1.1, 120)]),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_top, use_container_width=True)

with c2:
    bottom_m = msa_year.nsmallest(top_n, "recovery_index")
    fig_bot = go.Figure(go.Bar(
        y=bottom_m["msa_md_name"], x=bottom_m["recovery_index"],
        orientation="h", marker_color="#e74c3c",
        text=bottom_m["recovery_index"].round(1), textposition="outside",
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

st.subheader("MSA Recovery Trend Over Time")
all_msa_names = sorted(msa_r["msa_md_name"].dropna().unique())
default_msas = ["San Jose - San Francisco - Oakland, CA"] if "San Jose - San Francisco - Oakland, CA" in all_msa_names else all_msa_names[:3]
selected_msas = st.multiselect("Select MSA(s) to compare", all_msa_names, default=default_msas[:3])

if selected_msas:
    trend_data = msa_r[msa_r["msa_md_name"].isin(selected_msas)]
    fig4b = go.Figure()
    for msa_name in selected_msas:
        sub_msa = trend_data[trend_data["msa_md_name"] == msa_name].sort_values("year")
        fig4b.add_trace(go.Scatter(
            x=sub_msa["year"], y=sub_msa["recovery_index"],
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
st.caption("Each denied application can cite up to 3 reasons. Counts reflect total reason citations, not unique applications.")

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

st.subheader("Denial Reasons Over the Years")
dr_pivot = dr_nat.pivot_table(
    index="year", columns="reason_label", values="count", aggfunc="sum"
).fillna(0).reset_index()

fig5a = go.Figure()
for reason, color in REASON_COLORS.items():
    if reason in dr_pivot.columns:
        fig5a.add_trace(go.Bar(
            x=dr_pivot["year"], y=dr_pivot[reason],
            name=reason, marker_color=color,
            hovertemplate=f"<b>{reason}</b><br>Year: %{{x}}<br>Count: %{{y:,.0f}}<extra></extra>",
        ))
fig5a.add_vline(x=2008, line_dash="dash", line_color="#c0392b", line_width=1.5)
fig5a.add_annotation(x=2008, y=1, yref="paper", text="<b>2008 Crash</b>",
    showarrow=False, font=dict(color="#c0392b", size=10), bgcolor="rgba(255,255,255,0.85)")
fig5a.update_layout(
    barmode="stack", height=450,
    legend=dict(orientation="h", y=1.12, font=dict(size=11)),
    plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(tickmode="linear", dtick=1, title="Year", showgrid=False),
    yaxis=dict(title="Denial Reason Citations", tickformat=",",
               showgrid=True, gridcolor="rgba(0,0,0,0.07)"),
    margin=dict(t=60, b=40),
)
fig5a = significance_badge(fig5a, sig, "Denial Reason Mix Shift Post-2008", x=0.01, y=0.97)
st.plotly_chart(fig5a, use_container_width=True)

st.subheader("How the Mix of Denial Reasons Shifted")
st.caption("Shows % share of each reason — reveals how bank priorities changed after the crash.")

dr_total = dr_pivot.drop(columns=["year"]).sum(axis=1)
dr_share_df = dr_pivot.copy()
for col in dr_pivot.columns:
    if col != "year":
        dr_share_df[col] = (dr_pivot[col] / dr_total * 100).round(2)

fig5b = go.Figure()
for reason, color in REASON_COLORS.items():
    if reason in dr_share_df.columns:
        fig5b.add_trace(go.Scatter(
            x=dr_share_df["year"], y=dr_share_df[reason],
            name=reason, stackgroup="one", mode="lines",
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

if purpose_year == "All Years":
    dr_filtered = dr_full.copy()
    year_label  = "2007–2017 (All Years)"
else:
    dr_filtered = dr_full[dr_full["year"] == purpose_year].copy()
    year_label  = str(purpose_year)

purpose_reason = (
    dr_filtered.groupby(["purpose_label","reason_label"])["count"]
    .sum().reset_index()
)
purpose_reason = purpose_reason[purpose_reason["purpose_label"].notna()]
purpose_total  = purpose_reason.groupby("purpose_label")["count"].transform("sum")
purpose_reason["share"] = (purpose_reason["count"] / purpose_total * 100).round(1)

purposes_list = ["Home Purchase","Refinancing","Home Improvement"]
fig5c = _make_subplots(rows=1, cols=3,
    subplot_titles=purposes_list, horizontal_spacing=0.08)

for i, purpose in enumerate(purposes_list, 1):
    sub_pr = purpose_reason[purpose_reason["purpose_label"] == purpose].sort_values(
        "count" if show_raw else "share", ascending=True
    )
    x_val  = sub_pr["count"] if show_raw else sub_pr["share"]
    x_text = sub_pr["count"].map("{:,}".format) if show_raw else (sub_pr["share"].astype(str) + "%")
    x_max  = x_val.max() * 1.3 if not sub_pr.empty else 10

    fig5c.add_trace(go.Bar(
        y=sub_pr["reason_label"], x=x_val,
        orientation="h",
        marker_color=[REASON_COLORS.get(r, "#aaa") for r in sub_pr["reason_label"]],
        text=x_text, textposition="outside",
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
x_title_d  = "Count" if show_raw else "Share (%)"
x_suffix_d = "" if show_raw else "%"
fig5c.update_xaxes(title_text=x_title_d, ticksuffix=x_suffix_d,
                   showgrid=True, gridcolor="rgba(0,0,0,0.06)")
fig5c.update_yaxes(showgrid=False)
st.plotly_chart(fig5c, use_container_width=True)

total_denials = dr_nat["count"].sum()
top_reason    = dr_nat.groupby("reason_label")["count"].sum().idxmax()
top_count_dr  = dr_nat.groupby("reason_label")["count"].sum().max()
peak_year_dtr = dr_nat[dr_nat["reason_label"]=="Debt-to-income ratio"].groupby("year")["count"].sum().idxmax()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Denial Citations (2007–17)", f"{total_denials:,.0f}")
c2.metric("Most Common Reason (all years)", top_reason, f"{top_count_dr:,.0f} citations")
c3.metric("DTI Denials Peak Year", str(peak_year_dtr))
c4.metric("Unique Reason Categories", "9")

st.subheader("Full Breakdown Table")
tbl5 = dr_nat.pivot_table(index="year", columns="reason_label", values="count", aggfunc="sum").fillna(0).astype(int)
tbl5["Total"] = tbl5.sum(axis=1)
tbl5 = tbl5.reset_index()
tbl5.columns.name = None
st.dataframe(tbl5, use_container_width=True, hide_index=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 6 — Income & Affordability Analysis
# ══════════════════════════════════════════════════════════════════════════════
st.header("6. Borrower Income & Affordability (2007–2017)")
st.caption(
    "Based on approved (originated) loans only. Income values in $000s. "
    "Outliers (HMDA cap code 9999) excluded. Loan-to-income ratio = loan amount ÷ annual income."
)

st.subheader("Mean & Median Income of Approved Borrowers")

fig6a = _make_subplots(specs=[[{"secondary_y": True}]])
fig6a.add_trace(go.Scatter(
    x=inc_nat["year"], y=inc_nat["approved_income_median"],
    name="Median Income ($K)", mode="lines+markers+text",
    line=dict(color="#2980b9", width=3), marker=dict(size=9),
    text=("$" + inc_nat["approved_income_median"].astype(int).astype(str) + "K"),
    textposition="top center", textfont=dict(size=10),
))
fig6a.add_trace(go.Scatter(
    x=inc_nat["year"], y=inc_nat["approved_income_mean"],
    name="Mean Income ($K)", mode="lines+markers+text",
    line=dict(color="#2980b9", width=2, dash="dot"), marker=dict(size=7, symbol="diamond"),
    text=("$" + inc_nat["approved_income_mean"].round(0).astype(int).astype(str) + "K"),
    textposition="bottom center", textfont=dict(size=10, color="#2980b9"),
))
fig6a.add_trace(go.Scatter(
    x=inc_nat["year"], y=inc_nat["denied_income_median"],
    name="Denied — Median Income ($K)", mode="lines+markers",
    line=dict(color="#e74c3c", width=2, dash="dash"), marker=dict(size=7),
))
fig6a.add_trace(go.Scatter(
    x=inc_nat["year"], y=inc_nat["approved_loan_median"],
    name="Median Loan Amount ($K)", mode="lines+markers+text",
    line=dict(color="#27ae60", width=2.5), marker=dict(size=8, symbol="square"),
    text=("$" + inc_nat["approved_loan_median"].astype(int).astype(str) + "K"),
    textposition="top center", textfont=dict(size=10, color="#27ae60"),
), secondary_y=True)

fig6a.add_vline(x=2008, line_dash="dash", line_color="#e64d1f", line_width=1.5)
fig6a.add_annotation(x=2008, y=1, yref="paper", text="<b>2008 Crash</b>",
    showarrow=False, font=dict(color="#e64d1f", size=10), bgcolor="rgba(255,255,255,0.85)")

fig6a.update_layout(
    height=420, legend=dict(orientation="h", y=1.12),
    plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(tickmode="linear", dtick=1, title="Year", showgrid=False),
    yaxis=dict(title="Income ($K)", showgrid=True, gridcolor="rgba(0,0,0,0.07)"),
    yaxis2=dict(title="Loan Amount ($K)", showgrid=False),
    margin=dict(t=50, b=40),
)
st.plotly_chart(fig6a, use_container_width=True)

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

st.subheader("Loan-to-Income Ratio & Income Required per $100K Loan")
st.caption(
    "LTI = loan amount ÷ annual income.  "
    "LTI of 2.0 means the loan is 2× your annual salary.  "
    "Higher LTI = less affordable.  Most lenders use a max LTI of 4–4.5×."
)

fig6b = _make_subplots(specs=[[{"secondary_y": True}]])
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
    line=dict(color="#8e44ad", width=2.5), marker=dict(size=8),
    text=("$" + inc_nat["income_per_100k_loan"].astype(str) + "K"),
    textposition="top center", textfont=dict(size=10, color="#8e44ad"),
), secondary_y=True)

fig6b.add_vline(x=2008, line_dash="dash", line_color="#e64d1f", line_width=1.5)
fig6b.update_layout(
    height=400, legend=dict(orientation="h", y=1.1),
    plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(tickmode="linear", dtick=1, title="Year", showgrid=False),
    yaxis=dict(title="LTI Ratio (×)", range=[0, 3.2],
               showgrid=True, gridcolor="rgba(0,0,0,0.07)"),
    yaxis2=dict(title="Income needed per $100K ($K)", range=[35, 60], showgrid=False),
    margin=dict(t=40, b=40),
)
fig6b = significance_badge(fig6b, sig, "LTI Ratio Increase (2007–2017)", x=0.99, y=0.97, xanchor="right")
st.plotly_chart(fig6b, use_container_width=True)

st.info(
    "**How to read this:** In 2007, the median approved borrower needed **$48.3K income to borrow $100K**. "
    "By 2017 that dropped to **$41.7K per $100K** — meaning loans grew faster than incomes. "
    "The LTI ratio rising from 2.07× to 2.40× confirms houses became less affordable relative to income "
    "even as the market 'recovered.' A rising LTI is an early warning sign of a new affordability bubble."
)

st.markdown("---")

st.subheader("Median Income & Loan by Loan Purpose")

fig6c = _make_subplots(rows=1, cols=2,
    subplot_titles=["Median Income of Approved Borrowers ($K)", "Median Loan Amount ($K)"],
    horizontal_spacing=0.10)

COLORS_PURPOSE2 = {"Home Purchase": "#2980b9", "Refinancing": "#e74c3c", "Home Improvement": "#f39c12"}
for lbl, clr in COLORS_PURPOSE2.items():
    sub6 = inc_lp[inc_lp["loan_purpose_label"] == lbl].sort_values("year")
    fig6c.add_trace(go.Scatter(
        x=sub6["year"], y=sub6["approved_income_median"],
        name=lbl, mode="lines+markers",
        line=dict(color=clr, width=2.5), marker=dict(size=7),
        legendgroup=lbl,
        hovertemplate=f"<b>{lbl}</b><br>Income: $%{{y:.0f}}K<extra></extra>",
    ), row=1, col=1)
    fig6c.add_trace(go.Scatter(
        x=sub6["year"], y=sub6["approved_loan_median"],
        name=lbl, mode="lines+markers",
        line=dict(color=clr, width=2.5, dash="dot"), marker=dict(size=7),
        legendgroup=lbl, showlegend=False,
        hovertemplate=f"<b>{lbl}</b><br>Loan: $%{{y:.0f}}K<extra></extra>",
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

st.subheader("Median Income of Approved Borrowers by State")
inc_yr_sel = st.select_slider("Select Year (Income Map)", options=YEARS, value=2017)
inc_st_yr = inc_st[inc_st["year"] == inc_yr_sel].copy()
inc_st_yr["hover"] = (
    "<b>" + inc_st_yr["state_abbr"] + "</b><br>"
    "Median Income: $" + inc_st_yr["approved_income_median"].round(0).astype(int).astype(str) + "K<br>"
    "Median Loan: $"   + inc_st_yr["approved_loan_median"].round(0).astype(int).astype(str) + "K<br>"
    "LTI Ratio: "      + inc_st_yr["lti_median"].round(2).astype(str) + "×"
)

c_inc, c_loan = st.columns(2)
map_metric = c_inc.radio("Map shows:", ["Median Income","Median Loan Amount","LTI Ratio"], horizontal=True)
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
    title=dict(text=f"{map_metric} of Approved Borrowers — {inc_yr_sel}", font=dict(size=14)),
    margin=dict(t=40, b=10, l=10, r=10),
    paper_bgcolor="white",
)
st.plotly_chart(fig6d, use_container_width=True)

st.subheader("Mean Income vs Mean Loan Amount by Year")
st.caption(
    "Mean is pulled up by high-income/high-loan outliers — always higher than median. "
    "The gap between mean and median reveals income inequality among approved borrowers."
)

fig6e = _make_subplots(rows=1, cols=2,
    subplot_titles=["Mean vs Median Income ($K) — Approved Borrowers",
                    "Mean vs Median Loan Amount ($K) — Originated Loans"],
    horizontal_spacing=0.10)

fig6e.add_trace(go.Scatter(
    x=inc_nat["year"], y=inc_nat["approved_income_mean"],
    name="Mean Income", mode="lines+markers+text",
    line=dict(color="#1a7abf", width=3), marker=dict(size=9),
    text=("$" + inc_nat["approved_income_mean"].round(0).astype(int).astype(str) + "K"),
    textposition="top center", textfont=dict(size=10),
), row=1, col=1)
fig6e.add_trace(go.Scatter(
    x=inc_nat["year"], y=inc_nat["approved_income_median"],
    name="Median Income", mode="lines+markers+text",
    line=dict(color="#1a7abf", width=2, dash="dot"), marker=dict(size=7),
    text=("$" + inc_nat["approved_income_median"].round(0).astype(int).astype(str) + "K"),
    textposition="bottom center", textfont=dict(size=10, color="#1a7abf"),
), row=1, col=1)

fig6e.add_trace(go.Scatter(
    x=inc_nat["year"], y=inc_nat["approved_loan_mean"],
    name="Mean Loan", mode="lines+markers+text",
    line=dict(color="#27ae60", width=3), marker=dict(size=9),
    text=("$" + inc_nat["approved_loan_mean"].round(0).astype(int).astype(str) + "K"),
    textposition="top center", textfont=dict(size=10, color="#27ae60"),
), row=1, col=2)
fig6e.add_trace(go.Scatter(
    x=inc_nat["year"], y=inc_nat["approved_loan_median"],
    name="Median Loan", mode="lines+markers+text",
    line=dict(color="#27ae60", width=2, dash="dot"), marker=dict(size=7),
    text=("$" + inc_nat["approved_loan_median"].round(0).astype(int).astype(str) + "K"),
    textposition="bottom center", textfont=dict(size=10, color="#27ae60"),
), row=1, col=2)

for col_idx in [1, 2]:
    fig6e.add_vline(x=2008, line_dash="dash", line_color="#e64d1f", line_width=1.2, col=col_idx, row=1)

fig6e.update_layout(
    height=400, legend=dict(orientation="h", y=1.12),
    plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=60, b=40),
)
fig6e.update_xaxes(tickmode="linear", dtick=1, showgrid=False)
fig6e.update_yaxes(tickprefix="$", ticksuffix="K", showgrid=True,
                   gridcolor="rgba(0,0,0,0.06)", row=1, col=1)
fig6e.update_yaxes(tickprefix="$", ticksuffix="K", showgrid=True,
                   gridcolor="rgba(0,0,0,0.06)", row=1, col=2)
st.plotly_chart(fig6e, use_container_width=True)

st.subheader("Full Income & Loan Summary Table (Approved Borrowers)")
tbl6 = inc_nat[[
    "year",
    "approved_income_mean","approved_income_median",
    "approved_loan_mean",  "approved_loan_median",
    "lti_mean",            "lti_median",
    "income_per_100k_loan",
]].copy()
tbl6.columns = [
    "Year","Mean Income ($K)","Median Income ($K)",
    "Mean Loan ($K)","Median Loan ($K)",
    "Mean LTI","Median LTI","Income/$100K Loan ($K)",
]
for col in tbl6.columns[1:]:
    tbl6[col] = tbl6[col].round(1)
st.dataframe(tbl6.reset_index(drop=True), use_container_width=True, hide_index=True)

st.markdown("---")

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
            "Finding":     row["finding"],
            "Test":        row["test_name"],
            "p-value":     f"{row['p_value']:.6f}",
            "Effect Size": f"{row['effect_size']:.4f} ({row['effect_size_label']})",
        })
    st.dataframe(pd.DataFrame(method_rows), use_container_width=True, hide_index=True)
    st.caption("Packages: scipy==1.13.0 · statsmodels==0.14.1 · pymannkendall==1.4.3")
