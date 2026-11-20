"""
app.py — FYP-Pulse: Final Year Project Early Warning System
FYP-Pulse | Team Star

One-command run: streamlit run app.py
Sections: Overview, Data Quality, Dashboard, Models, Predict & Explain,
          Fairness, Watchlist, What-If
"""
from __future__ import annotations

import os
import sys
import json
import glob
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# --- Path setup ---
ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

from src.cleaning import clean_data, LEAKAGE_COLS, VALID_CAMPUSES
from src.features import engineer_features, FEATURE_EXPLANATIONS, get_feature_columns
from src.modeling import (
    train_all_models, prepare_data, EXCLUDED_COLUMNS,
    NUMERIC_BASE, NUMERIC_ENGINEERED, CATEGORICAL_FEATURES
)
from src.explain import top_drivers, compute_feature_importances
from src.fairness import gender_fairness, campus_fairness
from src.watchlist import build_watchlist
from src.whatif import LEVERS, LOCKED_FEATURES, ASSOCIATION_NOTE, predict_with_levers, lever_sensitivity
from src.ui import (
    apply_custom_css, hero_header, kpi_row, risk_badge,
    section_header, insight_box, footer, empty_state, PLOTLY_PALETTE, TIER_COLORS
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FYP-Pulse | Team Star",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_custom_css()

# ---------------------------------------------------------------------------
# Data loading helper (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_clean(raw_csv_bytes: bytes) -> tuple:
    df_raw = pd.read_csv(pd.io.common.BytesIO(raw_csv_bytes))
    df_clean, log_df, summary = clean_data(df_raw)
    return df_raw, df_clean, log_df, summary


def auto_find_csv() -> str | None:
    target = os.path.join(ROOT, "data", "fyp_pulse_train.csv")
    if os.path.exists(target):
        return target
    data_dir = os.path.join(ROOT, "data")
    found = glob.glob(os.path.join(data_dir, "*.csv"))
    if found:
        return found[0]
    found_root = glob.glob(os.path.join(ROOT, "*.csv"))
    if found_root:
        return found_root[0]
    return None


# ---------------------------------------------------------------------------
# Session state keys with instant auto-init
# ---------------------------------------------------------------------------
if "train_results" not in st.session_state:
    st.session_state.train_results = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "log_df" not in st.session_state:
    st.session_state.log_df = None
if "clean_summary" not in st.session_state:
    st.session_state.clean_summary = None

# Auto-load on initial startup if data exists
if st.session_state.df_clean is None:
    default_csv = auto_find_csv()
    if default_csv and os.path.exists(default_csv):
        with open(default_csv, "rb") as f:
            raw_bytes = f.read()
        df_raw, df_clean, log_df, summary = load_and_clean(raw_bytes)
        st.session_state.df_raw = df_raw
        st.session_state.df_clean = df_clean
        st.session_state.log_df = log_df
        st.session_state.clean_summary = summary

# Auto-train models on initial start if not trained yet
if st.session_state.train_results is None and st.session_state.df_clean is not None:
    st.session_state.train_results = train_all_models(st.session_state.df_clean)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES = [
    "🏠 Overview",
    "🔬 Data Quality",
    "📊 Dashboard",
    "🤖 Models",
    "🎯 Predict & Explain",
    "⚖️ Fairness",
    "🚨 Watchlist",
    "🔀 What-If",
]

with st.sidebar:
    st.markdown("""
<div style="text-align:center; padding: 16px 0 8px 0;">
  <div style="font-size:2.5rem;">⭐</div>
  <div style="font-weight:700; color:#e2e8f0; font-size:1.1rem;">FYP-Pulse</div>
  <div style="color:#94a3b8; font-size:0.8rem;">Team Star</div>
</div>
<hr style="border-color:#334155; margin:8px 0 16px 0;">
""", unsafe_allow_html=True)

    page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    st.markdown("""
<hr style="border-color:#334155; margin:16px 0 8px 0;">
<div style="color:#64748b; font-size:0.75rem; text-align:center; padding-bottom:8px;">
  Built by <strong style="color:#94a3b8;">Team Star</strong> ⭐<br>
  Vibe Jam AI Sprint 2026
</div>
""", unsafe_allow_html=True)

    with st.expander("ℹ️ How to Use"):
        st.markdown("""
**Step 1:** Upload or auto-load the CSV on the **Overview** page.  
**Step 2:** Review cleaning fixes in **Data Quality**.  
**Step 3:** Explore trends in **Dashboard**.  
**Step 4:** Train models in **Models**.  
**Step 5:** Predict a group in **Predict & Explain**.  
**Step 6:** Check bias in **Fairness**.  
**Step 7:** See top-risk groups in **Watchlist**.  
**Step 8:** Simulate improvements in **What-If**.
""")


# ---------------------------------------------------------------------------
# Model training helper (cached by resource)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def run_training(clean_csv_hash: int) -> dict:
    df_clean = st.session_state.df_clean
    return train_all_models(df_clean)


# ---------------------------------------------------------------------------
# Plotly helper: consistent theme
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#f8fafc",
    font_family="Plus Jakarta Sans",
    title_font_size=16,
    title_font_color="#ffffff",
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
    xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", linecolor="rgba(255, 255, 255, 0.12)", tickfont=dict(color="#94a3b8")),
    yaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", linecolor="rgba(255, 255, 255, 0.12)", tickfont=dict(color="#94a3b8")),
    margin=dict(l=40, r=40, t=50, b=40),
)


def apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ===========================================================================
# PAGE: OVERVIEW
# ===========================================================================
if page == "🏠 Overview":
    hero_header("Final Year Project Early Warning System")

    st.markdown("""
<div style="color:#94a3b8; text-align:center; margin-bottom:24px;">
Predict at-risk FYP groups at week 8 — before final defense. No external APIs.
No data leakage. Fully explainable.
</div>
""", unsafe_allow_html=True)

    # --- Data Loading ---
    section_header("Load Dataset")
    csv_path = auto_find_csv()

    col1, col2 = st.columns([2, 1])
    with col1:
        if csv_path:
            st.success(f"✅ Auto-detected: `{os.path.basename(csv_path)}`")
        uploaded = st.file_uploader("Or upload a CSV", type=["csv"])

    with col2:
        load_btn = st.button("🔄 Load & Clean Data", use_container_width=True)

    if load_btn:
        raw_bytes = None
        if uploaded:
            raw_bytes = uploaded.read()
        elif csv_path and os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                raw_bytes = f.read()

        if raw_bytes is None:
            st.error("No CSV found. Place `fyp_pulse_train.csv` in the `data/` folder or upload.")
        else:
            with st.spinner("Cleaning data..."):
                df_raw, df_clean, log_df, summary = load_and_clean(raw_bytes)
                st.session_state.df_raw = df_raw
                st.session_state.df_clean = df_clean
                st.session_state.log_df = log_df
                st.session_state.clean_summary = summary
                st.session_state.train_results = train_all_models(df_clean)
            st.success(f"✅ Loaded {len(df_raw):,} rows → {len(df_clean):,} clean rows.")

    # --- Summary KPIs ---
    if st.session_state.df_clean is not None:
        s = st.session_state.clean_summary
        df = st.session_state.df_clean
        kpi_row([
            {"value": f"{s['initial_rows']:,}", "label": "Raw Rows"},
            {"value": f"{s['cleaned_rows']:,}", "label": "Cleaned Rows"},
            {"value": f"{s['total_poor_outcomes']:,}", "label": "Poor Outcomes", "color": "#d64545"},
            {"value": f"{s['poor_outcome_rate']*100:.1f}%", "label": "Poor Outcome Rate", "color": "#f2b705"},
            {"value": f"{s.get('total_range_errors', 0):,}", "label": "Range Errors Fixed", "color": "#06b6d4"},
        ])

        section_header("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

    # --- Leakage Exclusion Table ---
    section_header("⚠️ Leakage Exclusion Policy")
    st.markdown("""
> These columns are **never** used as model features. Using them would constitute data leakage.
""")
    excl_df = pd.DataFrame([
        {"Column": col, "Reason": reason}
        for col, reason in EXCLUDED_COLUMNS.items()
    ])
    st.dataframe(excl_df, use_container_width=True, hide_index=True)

    # --- Quick Start ---
    section_header("Quick Start")
    st.markdown("""
1. **Data Quality** → review every cleaning fix applied  
2. **Dashboard** → explore risk factors with interactive charts  
3. **Models** → train 4 ML models and select the best  
4. **Predict & Explain** → submit a group and get instant risk assessment  
5. **Fairness** → check for gender and campus bias  
6. **Watchlist** → see the highest-risk test groups  
7. **What-If** → simulate what happens if the group improves engagement  
""")

    footer()


# ===========================================================================
# PAGE: DATA QUALITY
# ===========================================================================
elif page == "🔬 Data Quality":
    hero_header("Data Quality Report")

    if st.session_state.df_clean is None:
        empty_state("Load data first from the Overview page.")
        st.stop()

    df_clean = st.session_state.df_clean
    df_raw = st.session_state.df_raw
    log_df = st.session_state.log_df
    s = st.session_state.clean_summary

    # --- Summary ---
    section_header("Cleaning Summary")
    kpi_row([
        {"value": f"{s['initial_rows']:,}", "label": "Raw Rows"},
        {"value": f"{s['cleaned_rows']:,}", "label": "Cleaned Rows"},
        {"value": f"{s.get('exact_duplicate_rows_removed', 0):,}", "label": "Exact Dupes Removed"},
        {"value": f"{s.get('rows_missing_target_dropped', 0):,}", "label": "Missing Target Dropped"},
        {"value": f"{s.get('total_range_errors', 0):,}", "label": "Out-of-Range Values Fixed"},
    ])

    kpi_row([
        {"value": f"{s.get('campus_corrections', 0):,}", "label": "Campus Corrections"},
        {"value": f"{s.get('gender_corrections', 0):,}", "label": "Gender Corrections"},
        {"value": f"{s.get('yes_no_corrections', 0):,}", "label": "Yes/No Corrections"},
        {"value": f"{s.get('cgpa_min_gt_avg_repaired', 0):,}", "label": "CGPA Min>Avg Repaired"},
        {"value": f"{s.get('cgpa_solo_rule_applied', 0):,}", "label": "Solo Rule Applied"},
    ])

    # --- Date ambiguity note ---
    section_header("Date Parsing Policy")
    st.info(s.get("date_ambiguity_note", ""))

    # --- Rule-by-rule fix table ---
    section_header("Fix Log (Rule-by-Rule)")
    if len(log_df) > 0:
        rule_summary = (
            log_df.groupby("rule_applied")
            .agg(fixes=("row_index", "count"), columns=("column", lambda x: ", ".join(x.unique())))
            .reset_index()
            .rename(columns={"rule_applied": "Rule", "fixes": "# Fixes", "columns": "Columns Affected"})
        )
        st.dataframe(rule_summary, use_container_width=True, hide_index=True)
        st.markdown(f"**Total fixes applied:** {len(log_df):,}")
    else:
        st.success("No fixes needed — data is already clean!")

    # --- Per-column range errors ---
    if s.get("range_errors_per_column"):
        section_header("Out-of-Range Errors by Column")
        st.markdown("> Out-of-range values are **errors**, not outliers. They are set to missing.")
        err_df = pd.DataFrame([
            {"Column": col, "Out-of-Range Count": cnt, "Action": "Set to NaN (missing)"}
            for col, cnt in s["range_errors_per_column"].items()
        ])
        st.dataframe(err_df, use_container_width=True, hide_index=True)

    # --- CGPA Rules ---
    section_header("CGPA Validation Rules")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Min > Avg Swapped", s.get("cgpa_min_gt_avg_repaired", 0))
        st.caption("When team_min_cgpa > team_avg_cgpa, the values are swapped.")
    with col2:
        st.metric("Solo Rule Applied", s.get("cgpa_solo_rule_applied", 0))
        st.caption("When group_size=1, team_min_cgpa is set equal to team_avg_cgpa.")

    # --- Before / After preview ---
    section_header("Before / After Preview")
    tab1, tab2 = st.tabs(["🔴 Raw (first 20 rows)", "✅ Cleaned (first 20 rows)"])
    with tab1:
        st.dataframe(df_raw.head(20), use_container_width=True)
    with tab2:
        st.dataframe(df_clean.head(20), use_container_width=True)

    # --- Downloads ---
    section_header("Downloads")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download Cleaned CSV",
            data=df_clean.to_csv(index=False),
            file_name="fyp_pulse_cleaned.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            "⬇️ Download Cleaning Log CSV",
            data=log_df.to_csv(index=False),
            file_name="fyp_pulse_cleaning_log.csv",
            mime="text/csv",
        )

    # --- Full log ---
    with st.expander("📋 Full Fix Log"):
        st.dataframe(log_df, use_container_width=True)

    footer()


# ===========================================================================
# PAGE: DASHBOARD
# ===========================================================================
elif page == "📊 Dashboard":
    hero_header("Exploratory Dashboard")

    if st.session_state.df_clean is None:
        empty_state("Load data first from the Overview page.")
        st.stop()

    df = st.session_state.df_clean.copy()
    df = engineer_features(df)
    df["poor_binary"] = (df["poor_outcome"] == "Yes").astype(int)
    poor_rate = df["poor_binary"].mean()

    # --- KPI Cards ---
    section_header("Key Metrics")
    kpi_row([
        {"value": f"{len(df):,}", "label": "Total Groups"},
        {"value": f"{df['poor_binary'].sum():,}", "label": "Poor Outcomes", "color": "#d64545"},
        {"value": f"{poor_rate*100:.1f}%", "label": "Poor Outcome Rate", "color": "#f2b705"},
        {"value": f"{df['team_avg_cgpa'].mean():.2f}", "label": "Mean Team CGPA", "color": "#2e9e5b"},
        {"value": f"{df['github_commits_8wk'].median():.0f}", "label": "Median Commits (8wk)", "color": "#6366f1"},
    ])

    # --- Chart 1: TEAM FACTOR — Poor outcome by group size ---
    section_header("Team Factor: Group Size vs Poor Outcome Rate")
    gs_rate = (
        df.groupby("group_size")["poor_binary"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "poor_rate", "count": "n_groups", "group_size": "Group Size"})
    )
    gs_rate["Poor Outcome Rate (%)"] = gs_rate["poor_rate"] * 100
    fig1 = px.bar(
        gs_rate, x="Group Size", y="Poor Outcome Rate (%)",
        color="Poor Outcome Rate (%)",
        color_continuous_scale=["#2e9e5b", "#f2b705", "#d64545"],
        text=gs_rate["Poor Outcome Rate (%)"].apply(lambda x: f"{x:.1f}%"),
        hover_data={"n_groups": True},
        labels={"n_groups": "# Groups"},
    )
    fig1.update_traces(textposition="outside")
    apply_theme(fig1)
    st.plotly_chart(fig1, use_container_width=True)
    # Find max risk size
    max_gs = gs_rate.loc[gs_rate["poor_rate"].idxmax()]
    insight_box(
        f"Groups of size {int(max_gs['Group Size'])} have the highest poor-outcome rate "
        f"({max_gs['Poor Outcome Rate (%)']:.1f}%) compared to the overall rate of {poor_rate*100:.1f}%."
    )

    # --- Chart 2: SUPERVISOR FACTOR — Experience vs poor outcome ---
    section_header("Supervisor Factor: Experience vs Poor Outcome Rate")
    df["exp_bin"] = pd.cut(
        df["supervisor_experience_years"].fillna(-1),
        bins=[-2, 0, 5, 10, 20, 46],
        labels=["Missing", "0-5 yrs", "6-10 yrs", "11-20 yrs", "21+ yrs"]
    )
    exp_rate = (
        df.groupby("exp_bin", observed=True)["poor_binary"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "poor_rate", "count": "n_groups", "exp_bin": "Supervisor Experience"})
    )
    exp_rate["Poor Outcome Rate (%)"] = exp_rate["poor_rate"] * 100
    fig2 = px.bar(
        exp_rate, x="Supervisor Experience", y="Poor Outcome Rate (%)",
        color="Poor Outcome Rate (%)",
        color_continuous_scale=["#2e9e5b", "#f2b705", "#d64545"],
        text=exp_rate["Poor Outcome Rate (%)"].apply(lambda x: f"{x:.1f}%"),
        hover_data={"n_groups": True},
    )
    fig2.update_traces(textposition="outside")
    apply_theme(fig2)
    st.plotly_chart(fig2, use_container_width=True)
    min_exp = exp_rate.loc[exp_rate["poor_rate"].idxmin()]
    insight_box(
        f"Supervisors with {min_exp['Supervisor Experience']} experience correlate with the "
        f"lowest poor-outcome rate ({min_exp['Poor Outcome Rate (%)']:.1f}%). "
        f"More experienced supervisors may provide better guidance."
    )

    # --- Chart 3: PROJECT CONTEXT — Project type vs poor outcome ---
    section_header("Project Context: Project Type vs Poor Outcome Rate")
    pt_rate = (
        df.dropna(subset=["project_type"])
        .groupby("project_type")["poor_binary"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "poor_rate", "count": "n_groups", "project_type": "Project Type"})
        .sort_values("poor_rate", ascending=False)
    )
    pt_rate["Poor Outcome Rate (%)"] = pt_rate["poor_rate"] * 100
    fig3 = px.bar(
        pt_rate, x="Project Type", y="Poor Outcome Rate (%)",
        color="Project Type",
        color_discrete_sequence=PLOTLY_PALETTE,
        text=pt_rate["Poor Outcome Rate (%)"].apply(lambda x: f"{x:.1f}%"),
        hover_data={"n_groups": True},
    )
    fig3.update_traces(textposition="outside")
    apply_theme(fig3)
    st.plotly_chart(fig3, use_container_width=True)
    top_pt = pt_rate.iloc[0]
    insight_box(
        f"**{top_pt['Project Type']}** projects have the highest poor-outcome rate "
        f"({top_pt['Poor Outcome Rate (%)']:.1f}%). Consider additional monitoring for this type."
    )

    # --- Chart 4: ENGINEERED FEATURE — Low Engagement Flag ---
    section_header("Engineered Feature: Low Engagement Flag vs Poor Outcome")
    st.markdown(f"""
<div class="insight-box" style="margin-bottom:12px;">
📐 <strong>low_engagement_flag</strong>: {FEATURE_EXPLANATIONS['low_engagement_flag']}
</div>
""", unsafe_allow_html=True)
    eng_rate = (
        df.groupby("low_engagement_flag")["poor_binary"]
        .agg(["mean", "count"])
        .reset_index()
    )
    eng_rate["Engagement"] = eng_rate["low_engagement_flag"].map({0: "Engaged", 1: "Low Engagement"})
    eng_rate["Poor Outcome Rate (%)"] = eng_rate["mean"] * 100
    fig4 = px.bar(
        eng_rate, x="Engagement", y="Poor Outcome Rate (%)",
        color="Engagement",
        color_discrete_map={"Engaged": "#2e9e5b", "Low Engagement": "#d64545"},
        text=eng_rate["Poor Outcome Rate (%)"].apply(lambda x: f"{x:.1f}%"),
        hover_data={"count": True},
    )
    fig4.update_traces(textposition="outside")
    apply_theme(fig4)
    st.plotly_chart(fig4, use_container_width=True)
    low_eng = eng_rate[eng_rate["low_engagement_flag"] == 1]
    hi_eng = eng_rate[eng_rate["low_engagement_flag"] == 0]
    if len(low_eng) > 0 and len(hi_eng) > 0:
        ratio = low_eng["mean"].values[0] / max(hi_eng["mean"].values[0], 0.001)
        insight_box(
            f"Low-engagement groups (few commits AND few meetings) have a poor-outcome rate "
            f"{ratio:.1f}× higher than engaged groups. This engineered flag is a strong early-warning signal."
        )

    # --- Chart 5: Campus ---
    section_header("Campus Distribution & Poor Outcome Rate")
    campus_rate = (
        df.dropna(subset=["campus"])
        .groupby("campus")["poor_binary"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "poor_rate", "count": "n_groups", "campus": "Campus"})
        .sort_values("poor_rate", ascending=False)
    )
    campus_rate["Poor Outcome Rate (%)"] = campus_rate["poor_rate"] * 100
    fig5 = px.bar(
        campus_rate, x="Campus", y="Poor Outcome Rate (%)",
        color="n_groups",
        color_continuous_scale="Blues",
        text=campus_rate["Poor Outcome Rate (%)"].apply(lambda x: f"{x:.1f}%"),
        hover_data={"n_groups": True},
        labels={"n_groups": "# Groups"},
    )
    fig5.update_traces(textposition="outside")
    apply_theme(fig5)
    st.plotly_chart(fig5, use_container_width=True)
    if len(campus_rate) > 0:
        top_c = campus_rate.iloc[0]
        insight_box(
            f"**{top_c['Campus']}** has the highest poor-outcome rate ({top_c['Poor Outcome Rate (%)']:.1f}%) "
            f"with {int(top_c['n_groups'])} groups."
        )

    # --- Chart 6: Program ---
    section_header("Program vs Poor Outcome Rate")
    prog_rate = (
        df.dropna(subset=["program"])
        .groupby("program")["poor_binary"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "poor_rate", "count": "n_groups", "program": "Program"})
        .sort_values("poor_rate", ascending=False)
    )
    prog_rate["Poor Outcome Rate (%)"] = prog_rate["poor_rate"] * 100
    fig6 = px.bar(
        prog_rate, x="Program", y="Poor Outcome Rate (%)",
        color="Program",
        color_discrete_sequence=PLOTLY_PALETTE,
        text=prog_rate["Poor Outcome Rate (%)"].apply(lambda x: f"{x:.1f}%"),
        hover_data={"n_groups": True},
    )
    fig6.update_traces(textposition="outside")
    apply_theme(fig6)
    st.plotly_chart(fig6, use_container_width=True)
    if len(prog_rate) > 0:
        top_prog = prog_rate.iloc[0]
        insight_box(
            f"**{top_prog['Program']}** students have the highest poor-outcome rate ({top_prog['Poor Outcome Rate (%)']:.1f}%)."
        )

    # --- Chart 7: Commits vs Mid-Eval Scatter ---
    section_header("Commits vs Mid-Eval Score (coloured by outcome)")
    df_scatter = df.dropna(subset=["github_commits_8wk", "mid_eval_score"])
    df_scatter["Outcome"] = df_scatter["poor_binary"].map({0: "Good", 1: "Poor"})
    fig7 = px.scatter(
        df_scatter,
        x="github_commits_8wk", y="mid_eval_score",
        color="Outcome",
        color_discrete_map={"Good": "#2e9e5b", "Poor": "#d64545"},
        opacity=0.6, size_max=8,
        labels={"github_commits_8wk": "GitHub Commits (8wk)", "mid_eval_score": "Mid-Eval Score"},
    )
    apply_theme(fig7)
    st.plotly_chart(fig7, use_container_width=True)
    poor_low_commits = df_scatter[(df_scatter["github_commits_8wk"] < 20) & (df_scatter["poor_binary"] == 1)]
    insight_box(
        f"{len(poor_low_commits):,} poor-outcome groups had fewer than 20 commits — "
        f"low commit activity is a strong early signal."
    )

    # --- Chart 8: Correlation heatmap ---
    section_header("Correlation Heatmap (Numeric Features)")
    num_cols = [c for c in NUMERIC_BASE + NUMERIC_ENGINEERED + ["poor_binary"] if c in df.columns]
    corr = df[num_cols].corr(numeric_only=True)
    fig8 = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        labels=dict(color="Correlation"),
    )
    apply_theme(fig8)
    fig8.update_layout(height=600)
    st.plotly_chart(fig8, use_container_width=True)
    top_corr = corr["poor_binary"].abs().drop("poor_binary").sort_values(ascending=False)
    insight_box(
        f"The numeric feature most correlated with poor_outcome is "
        f"**{top_corr.index[0].replace('_', ' ').title()}** (|r| = {top_corr.iloc[0]:.3f})."
    )

    footer()


# ===========================================================================
# PAGE: MODELS
# ===========================================================================
elif page == "🤖 Models":
    hero_header("Model Training & Selection")

    if st.session_state.df_clean is None:
        empty_state("Load data first from the Overview page.")
        st.stop()

    # --- Leakage columns ---
    section_header("Dropped Columns (Data Leakage Prevention)")
    excl_df = pd.DataFrame([
        {"Column": col, "Reason": reason}
        for col, reason in EXCLUDED_COLUMNS.items()
    ])
    st.dataframe(excl_df, use_container_width=True, hide_index=True)

    # --- Training button ---
    if st.button("🚀 Train All Models", use_container_width=True):
        with st.spinner("Training 4 models with 5-fold stratified CV... (~30 seconds)"):
            results = train_all_models(st.session_state.df_clean)
            st.session_state.train_results = results
        st.success(f"✅ Training complete! Best model: **{results['best_model_name']}**")

    res = st.session_state.train_results
    if res is None:
        st.info("👆 Click 'Train All Models' to start.")
        st.stop()

    # --- Comparison table ---
    section_header("Model Comparison (Test Set)")
    model_rows = []
    for name, m in res["models"].items():
        is_best = "⭐ " if name == res["best_model_name"] else ""
        model_rows.append({
            "Model": f"{is_best}{name}",
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1": f"{m['f1']:.4f}",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "PR-AUC": f"{m['pr_auc']:.4f}",
            "CV PR-AUC": f"{m['cv_pr_auc_mean']:.4f} ± {m['cv_pr_auc_std']:.4f}",
            "Threshold": f"{m['threshold']:.3f}",
        })
    cmp_df = pd.DataFrame(model_rows)
    st.dataframe(cmp_df, use_container_width=True, hide_index=True)

    # Naive baseline row
    y_test_arr = res["y_test"].values
    base_rate = y_test_arr.mean()
    st.markdown(f"""
<div class="insight-box">
📏 <strong>Naive "No Risk" baseline:</strong> Accuracy = {(1-base_rate)*100:.1f}% — 
but Recall = 0.0%, PR-AUC ≈ {base_rate:.3f}. This baseline is useless for early warning.
</div>
""", unsafe_allow_html=True)

    # --- Best model justification ---
    section_header(f"Selected Model: {res['best_model_name']}")
    best = res["models"][res["best_model_name"]]
    st.markdown(f"""
**Justification:** The model was selected based on highest CV PR-AUC 
({best['cv_pr_auc_mean']:.4f} ± {best['cv_pr_auc_std']:.4f}) across 5 stratified folds.
PR-AUC is the primary metric because it rewards models that correctly identify the minority 
(poor-outcome) class. The final threshold ({best['threshold']:.3f}) was chosen on CV predictions 
to maximise F1 while keeping recall ≥ 65%, favouring early detection over false-positive avoidance.

**Test-set recall = {best['recall']:.4f}** — the model catches {best['recall']*100:.1f}% of truly 
poor-outcome groups, far exceeding the 0% recall of the naive baseline.
""")

    # --- Per-model tabs ---
    section_header("Per-Model Details")
    model_names = list(res["models"].keys())
    tabs = st.tabs(model_names)
    for tab, name in zip(tabs, model_names):
        with tab:
            m = res["models"][name]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Recall", f"{m['recall']:.4f}")
                st.metric("F1", f"{m['f1']:.4f}")
            with col2:
                st.metric("PR-AUC", f"{m['pr_auc']:.4f}")
                st.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
            with col3:
                st.metric("CV PR-AUC", f"{m['cv_pr_auc_mean']:.4f}")
                st.metric("Threshold", f"{m['threshold']:.3f}")

            # Confusion matrix
            cm = np.array(m["confusion_matrix"])
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Good (0)", "Poor (1)"],
                y=["Good (0)", "Poor (1)"],
                color_continuous_scale="Blues",
                text_auto=True,
            )
            fig_cm.update_layout(title=f"Confusion Matrix — {name}", **PLOTLY_LAYOUT)
            st.plotly_chart(fig_cm, use_container_width=True)

    # --- ROC Curves ---
    section_header("ROC Curves (All Models)")
    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dash", color="#94a3b8"))
    for i, (name, m) in enumerate(res["models"].items()):
        roc = m["roc_curve"]
        fig_roc.add_trace(go.Scatter(
            x=roc["fpr"], y=roc["tpr"],
            name=f"{name} (AUC={m['roc_auc']:.3f})",
            line=dict(color=PLOTLY_PALETTE[i]),
        ))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # --- PR Curves ---
    section_header("Precision-Recall Curves (All Models)")
    fig_pr = go.Figure()
    fig_pr.add_hline(y=base_rate, line_dash="dash", line_color="#94a3b8",
                     annotation_text=f"Random baseline ({base_rate:.2f})")
    for i, (name, m) in enumerate(res["models"].items()):
        pr = m["pr_curve"]
        fig_pr.add_trace(go.Scatter(
            x=pr["recall"], y=pr["precision"],
            name=f"{name} (PR-AUC={m['pr_auc']:.3f})",
            line=dict(color=PLOTLY_PALETTE[i]),
        ))
    fig_pr.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_pr, use_container_width=True)

    # --- Calibration curve ---
    section_header("Calibration Curve (Best Model)")
    best_pipeline = res["final_pipeline"]
    X_test = res["X_test"]
    y_test = res["y_test"]
    y_prob_best = np.array(res["models"][res["best_model_name"]]["y_prob"])
    # Bin probabilities
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    fraction_pos = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob_best >= lo) & (y_prob_best < hi)
        if mask.sum() == 0:
            fraction_pos.append(np.nan)
        else:
            fraction_pos.append(y_test.values[mask].mean())
    cal_df = pd.DataFrame({"Mean Predicted Prob": bin_mids, "Fraction Positive": fraction_pos})
    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect Calibration",
                                 line=dict(dash="dash", color="#94a3b8")))
    fig_cal.add_trace(go.Scatter(x=cal_df["Mean Predicted Prob"],
                                 y=cal_df["Fraction Positive"].ffill().bfill(),
                                 name=res["best_model_name"],
                                 line=dict(color=PLOTLY_PALETTE[0]),
                                 mode="lines+markers"))
    fig_cal.update_layout(xaxis_title="Mean Predicted Probability",
                          yaxis_title="Fraction of Positives", **PLOTLY_LAYOUT)
    st.plotly_chart(fig_cal, use_container_width=True)

    # --- Tier monotonicity verification ---
    section_header("Tier Thresholds & Monotonicity Check")
    tiers = res["tier_thresholds"]
    y_prob_arr = np.array(y_prob_best)
    low_mask = y_prob_arr < tiers["low_max"]
    high_mask = y_prob_arr >= tiers["high_min"]
    mid_mask = ~low_mask & ~high_mask
    r_low = y_test.values[low_mask].mean() if low_mask.sum() > 0 else 0
    r_mid = y_test.values[mid_mask].mean() if mid_mask.sum() > 0 else 0
    r_hi = y_test.values[high_mask].mean() if high_mask.sum() > 0 else 0
    monotone_ok = r_low < r_mid < r_hi

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Low (<{:.2f}) Observed Rate".format(tiers["low_max"]),
                  f"{r_low*100:.1f}%", f"n={low_mask.sum()}")
    with col2:
        st.metric("Medium Observed Rate", f"{r_mid*100:.1f}%", f"n={mid_mask.sum()}")
    with col3:
        st.metric("High (≥{:.2f}) Observed Rate".format(tiers["high_min"]),
                  f"{r_hi*100:.1f}%", f"n={high_mask.sum()}")

    if monotone_ok:
        st.success(f"✅ Monotonicity verified: Low ({r_low*100:.1f}%) < Medium ({r_mid*100:.1f}%) < High ({r_hi*100:.1f}%)")
    else:
        st.warning(f"⚠️ Monotonicity not fully satisfied on test set. Low={r_low*100:.1f}%, Med={r_mid*100:.1f}%, High={r_hi*100:.1f}%")

    footer()


# ===========================================================================
# PAGE: PREDICT & EXPLAIN
# ===========================================================================
elif page == "🎯 Predict & Explain":
    hero_header("Predict & Explain")

    if st.session_state.train_results is None:
        empty_state("Train models first on the Models page.")
        st.stop()

    res = st.session_state.train_results
    pipeline = res["final_pipeline"]
    tiers = res["tier_thresholds"]
    feature_cols = res["feature_names"]

    section_header("Group Week-8 Details Form")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            campus = st.selectbox("Campus", VALID_CAMPUSES)
            program = st.selectbox("Program", ["BSCS", "BSSE", "BSAI", "BSIT", "BSDS"])
            leader_gender = st.selectbox("Leader Gender", ["Male", "Female"])
            group_size = st.slider("Group Size", 1, 5, 3)
            has_industry_sponsor = st.selectbox("Industry Sponsor", ["No", "Yes"])

        with col2:
            team_avg_cgpa = st.number_input("Team Avg CGPA", 0.0, 4.0, 3.0, step=0.01)
            team_min_cgpa = st.number_input("Team Min CGPA", 0.0, 4.0, 2.5, step=0.01)
            project_type = st.selectbox("Project Type",
                                        ["Web App", "Mobile App", "AI-ML", "IoT", "Research"])
            proposal_revisions = st.slider("Proposal Revisions", 0, 10, 2)
            weeks_to_proposal_approval = st.slider("Weeks to Approval", 0, 30, 4)

        with col3:
            supervisor_experience_years = st.slider("Supervisor Experience (yrs)", 0, 45, 8)
            supervisor_active_groups = st.slider("Supervisor Active Groups", 1, 15, 6)
            supervisor_meetings_8wk = st.slider("Supervisor Meetings (8wk)", 0, 30, 5)
            github_commits_8wk = st.slider("GitHub Commits (8wk)", 0, 1000, 60, step=5)
            mid_eval_score = st.slider("Mid-Eval Score", 0.0, 100.0, 70.0, step=0.5)
            proposal_date = st.date_input("Proposal Date")

        submitted = st.form_submit_button("🔮 Predict Risk", use_container_width=True)

    if submitted:
        # Validation
        errors = []
        if team_min_cgpa > team_avg_cgpa:
            errors.append("❌ Team Min CGPA cannot exceed Team Avg CGPA.")
        if group_size == 1 and abs(team_min_cgpa - team_avg_cgpa) > 0.001:
            errors.append("❌ For a solo group (size=1), Min CGPA must equal Avg CGPA.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            # Build input DataFrame
            row = {
                "campus": campus,
                "program": program,
                "leader_gender": leader_gender,
                "group_size": float(group_size),
                "team_avg_cgpa": team_avg_cgpa,
                "team_min_cgpa": team_min_cgpa,
                "project_type": project_type,
                "has_industry_sponsor": has_industry_sponsor,
                "supervisor_experience_years": float(supervisor_experience_years),
                "supervisor_active_groups": float(supervisor_active_groups),
                "proposal_revisions": float(proposal_revisions),
                "weeks_to_proposal_approval": float(weeks_to_proposal_approval),
                "supervisor_meetings_8wk": float(supervisor_meetings_8wk),
                "github_commits_8wk": float(github_commits_8wk),
                "mid_eval_score": float(mid_eval_score),
                "proposal_date": str(proposal_date),
            }
            df_row = pd.DataFrame([row])
            df_row = engineer_features(df_row)
            missing_cols = [c for c in feature_cols if c not in df_row.columns]
            for c in missing_cols:
                df_row[c] = np.nan
            df_row_model = df_row[feature_cols]

            prob = float(pipeline.predict_proba(df_row_model)[:, 1][0])
            tier = "Low" if prob < tiers["low_max"] else ("High" if prob >= tiers["high_min"] else "Medium")
            tier_color = TIER_COLORS[tier]

            # Result display
            st.markdown("---")
            section_header("Prediction Result")
            rcol1, rcol2, rcol3 = st.columns([1, 1, 2])
            with rcol1:
                st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-value" style="color:{tier_color};">{prob:.4f}</div>
  <div class="kpi-label">Poor Outcome Probability</div>
  <div class="kpi-delta" style="color:{tier_color};">{prob*100:.2f}%</div>
</div>""", unsafe_allow_html=True)
            with rcol2:
                st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-value" style="color:{tier_color};">
    {risk_badge(tier)}
  </div>
  <div class="kpi-label">Risk Tier</div>
</div>""", unsafe_allow_html=True)
            with rcol3:
                # Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Risk Score (%)", "font": {"color": "#e2e8f0"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#e2e8f0"},
                        "bar": {"color": tier_color},
                        "bgcolor": "#1e293b",
                        "steps": [
                            {"range": [0, tiers["low_max"] * 100], "color": "#2e9e5b22"},
                            {"range": [tiers["low_max"] * 100, tiers["high_min"] * 100], "color": "#f2b70522"},
                            {"range": [tiers["high_min"] * 100, 100], "color": "#d6454522"},
                        ],
                        "threshold": {"line": {"color": tier_color, "width": 4},
                                      "thickness": 0.75, "value": prob * 100},
                    },
                    number={"suffix": "%", "font": {"color": tier_color}},
                ))
                fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                        font_color="#e2e8f0", height=200, margin=dict(t=40, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Tier cut-offs and observed rates
            section_header("Tier Thresholds & Validation")
            y_prob_arr = np.array(res["models"][res["best_model_name"]]["y_prob"])
            y_test_arr = res["y_test"].values
            low_m = y_prob_arr < tiers["low_max"]
            high_m = y_prob_arr >= tiers["high_min"]
            mid_m = ~low_m & ~high_m
            r_low = y_test_arr[low_m].mean() if low_m.sum() > 0 else 0
            r_mid = y_test_arr[mid_m].mean() if mid_m.sum() > 0 else 0
            r_hi = y_test_arr[high_m].mean() if high_m.sum() > 0 else 0
            kpi_row([
                {"value": f"{r_low*100:.1f}%", "label": f"Low (<{tiers['low_max']:.2f}) Observed Rate",
                 "color": "#2e9e5b", "delta": f"n={low_m.sum()}"},
                {"value": f"{r_mid*100:.1f}%", "label": "Medium Observed Rate",
                 "color": "#f2b705", "delta": f"n={mid_m.sum()}"},
                {"value": f"{r_hi*100:.1f}%", "label": f"High (≥{tiers['high_min']:.2f}) Observed Rate",
                 "color": "#d64545", "delta": f"n={high_m.sum()}"},
            ])
            assert r_low < r_mid < r_hi, "Monotonicity check failed"
            st.success("✅ Monotonicity verified: Low < Medium < High observed rates on test set.")

            # Explainability
            section_header("Top 3 Risk Drivers")
            X_train = res["X_train"]
            drivers_df, plain_text, action = top_drivers(pipeline, df_row_model, X_train, n=3)

            for _, driver_row in drivers_df.iterrows():
                c = driver_row["contribution"]
                name_ = driver_row["readable_name"]
                arrow = "⬆️" if c > 0 else "⬇️"
                bar_len = min(int(abs(c) * 400), 300)
                bar_color = "#d64545" if c > 0 else "#2e9e5b"
                st.markdown(f"""
<div style="background:#1e293b; border-radius:8px; padding:12px; margin-bottom:8px; display:flex; align-items:center; gap:12px;">
  <div style="font-size:1.5rem;">{arrow}</div>
  <div style="flex:1;">
    <div style="font-weight:600; color:#e2e8f0;">{name_}</div>
    <div style="background:#334155; border-radius:4px; height:8px; margin-top:6px;">
      <div style="background:{bar_color}; width:{bar_len}px; height:8px; border-radius:4px;"></div>
    </div>
  </div>
  <div style="color:{bar_color}; font-weight:600;">{'+' if c>0 else ''}{c:.4f}</div>
</div>""", unsafe_allow_html=True)

            section_header("Plain-Language Explanation")
            st.markdown(plain_text)

            section_header("Recommended Action")
            st.markdown(f"""
<div style="background:linear-gradient(135deg,#1e293b,#0f172a); border:1px solid #334155;
border-left:4px solid #f2b705; border-radius:8px; padding:16px; color:#e2e8f0;">
{action}
</div>""", unsafe_allow_html=True)

    footer()


# ===========================================================================
# PAGE: FAIRNESS
# ===========================================================================
elif page == "⚖️ Fairness":
    hero_header("Fairness Analysis")

    if st.session_state.train_results is None:
        empty_state("Train models first on the Models page.")
        st.stop()

    res = st.session_state.train_results
    X_test = res["X_test"]
    y_test = res["y_test"]
    df_clean = st.session_state.df_clean

    best_m = res["models"][res["best_model_name"]]
    y_prob = np.array(best_m["y_prob"])
    threshold = best_m["threshold"]

    # Need original test rows for campus/gender
    from sklearn.model_selection import train_test_split as tts
    _, df_test_orig = tts(
        df_clean, test_size=0.2, random_state=42,
        stratify=(df_clean["poor_outcome"] == "Yes").astype(int)
    )
    df_test_orig = df_test_orig.reset_index(drop=True)

    # --- Gender fairness ---
    section_header("Gender Fairness: Male vs Female Leaders")
    gf = gender_fairness(df_test_orig, y_test.values, y_prob, threshold)

    if "groups" in gf:
        g_rows = []
        for gender, gdata in gf["groups"].items():
            g_rows.append({
                "Gender": gender,
                "# Groups": gdata["n"],
                "Actual Poor Outcomes": gdata.get("actual_poor_outcomes", "N/A"),
                "Predicted Positive": gdata.get("predicted_positive", "N/A"),
                "Predicted Poor Rate": f"{gdata['predicted_poor_rate']*100:.1f}%" if gdata["predicted_poor_rate"] is not None else "N/A",
                "Recall": f"{gdata['recall']*100:.1f}%" if gdata["recall"] is not None else "N/A",
                "Precision": f"{gdata['precision']*100:.1f}%" if gdata["precision"] is not None else "N/A",
                "95% CI": f"[{gdata['ci_low']*100:.1f}%, {gdata['ci_high']*100:.1f}%]" if gdata.get("ci_low") is not None else "N/A",
            })
        st.dataframe(pd.DataFrame(g_rows), use_container_width=True, hide_index=True)

        # Gap chart
        gender_names = [gf["groups"][g]["group"] for g in gf["groups"]]
        gender_rates = [gf["groups"][g]["predicted_poor_rate"] * 100 for g in gf["groups"]]
        fig_gen = go.Figure(go.Bar(
            x=gender_names, y=gender_rates,
            marker_color=["#6366f1", "#f97316"],
            text=[f"{v:.1f}%" for v in gender_rates],
            textposition="outside",
        ))
        apply_theme(fig_gen)
        fig_gen.update_layout(title="Predicted Poor Outcome Rate by Gender",
                              yaxis_title="Predicted Poor Rate (%)", yaxis_range=[0, max(gender_rates) * 1.3])
        st.plotly_chart(fig_gen, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rate Gap (pp)", f"{gf.get('rate_gap_pct', 0):.1f}")
        with col2:
            st.metric("Recall Gap (pp)", f"{gf.get('recall_gap_pct', 0):.1f}")

        if "stat_test" in gf:
            st.info(f"Statistical test: {gf['stat_test']['note']} (p={gf['stat_test'].get('p_value', 'N/A')})")

        st.markdown(f"**Takeaway:** {gf.get('takeaway', 'N/A')}")

    # --- Campus fairness ---
    section_header("Campus Fairness (Campuses with ≥30 Test Groups)")
    cf = campus_fairness(df_test_orig, y_test.values, y_prob, threshold, min_n=30)

    if "groups" in cf and cf["groups"]:
        c_rows = []
        for campus, cdata in cf["groups"].items():
            c_rows.append({
                "Campus": campus,
                "# Groups": cdata["n"],
                "Actual Poor": cdata.get("actual_poor_outcomes", "N/A"),
                "Pred. Poor Rate": f"{cdata['predicted_poor_rate']*100:.1f}%" if cdata["predicted_poor_rate"] else "N/A",
                "Recall": f"{cdata['recall']*100:.1f}%" if cdata["recall"] else "N/A",
                "Precision": f"{cdata['precision']*100:.1f}%" if cdata["precision"] else "N/A",
                "95% CI": f"[{cdata['ci_low']*100:.1f}%, {cdata['ci_high']*100:.1f}%]" if cdata.get("ci_low") is not None else "N/A",
            })
        st.dataframe(pd.DataFrame(c_rows), use_container_width=True, hide_index=True)

        # Gap chart for campuses
        campus_names = list(cf["groups"].keys())
        campus_rates = [cf["groups"][c]["predicted_poor_rate"] * 100 for c in campus_names]
        fig_camp = go.Figure(go.Bar(
            x=campus_names, y=campus_rates,
            marker_color=PLOTLY_PALETTE[:len(campus_names)],
            text=[f"{v:.1f}%" for v in campus_rates],
            textposition="outside",
        ))
        apply_theme(fig_camp)
        fig_camp.update_layout(title="Predicted Poor Outcome Rate by Campus (≥30 test groups)",
                               yaxis_title="Predicted Poor Rate (%)")
        st.plotly_chart(fig_camp, use_container_width=True)

        if "largest_rate_gap_pct" in cf:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Largest Rate Gap (pp)", f"{cf['largest_rate_gap_pct']:.1f}",
                          f"Between {cf['highest_risk_campus']} and {cf['lowest_risk_campus']}")
            with col2:
                st.metric("Largest Recall Gap (pp)", f"{cf.get('largest_recall_gap_pct', 0):.1f}")

        st.markdown(f"**Takeaway:** {cf.get('takeaway', 'N/A')}")

    else:
        st.warning(f"⚠️ {cf.get('takeaway', 'No campuses meet the ≥30 test-group threshold.')}")
        st.info(f"Eligible campuses found: {cf.get('eligible_campuses', [])}")
        # Show all campuses with smaller groups
        all_rows = []
        for campus in df_test_orig["campus"].dropna().unique():
            mask = df_test_orig["campus"] == campus
            n = mask.sum()
            g_y = y_test.values[mask.values]
            g_p = y_prob[mask.values]
            pred_rate = (g_p >= threshold).mean()
            all_rows.append({"Campus": campus, "# Groups": int(n),
                             "Pred Poor Rate": f"{pred_rate*100:.1f}%",
                             "Note": "✅ Eligible" if n >= 30 else f"⚠️ n<30"})
        st.dataframe(pd.DataFrame(all_rows).sort_values("# Groups", ascending=False),
                     use_container_width=True, hide_index=True)

    footer()


# ===========================================================================
# PAGE: WATCHLIST
# ===========================================================================
elif page == "🚨 Watchlist":
    hero_header("Weekly Watchlist — Highest Risk Groups")

    if st.session_state.train_results is None:
        empty_state("Train models first on the Models page.")
        st.stop()

    res = st.session_state.train_results
    best_m = res["models"][res["best_model_name"]]
    y_prob = np.array(best_m["y_prob"])
    y_test = res["y_test"]
    tiers = res["tier_thresholds"]
    threshold = best_m["threshold"]

    df_clean = st.session_state.df_clean
    from sklearn.model_selection import train_test_split as tts
    _, df_test_orig = tts(
        df_clean, test_size=0.2, random_state=42,
        stratify=(df_clean["poor_outcome"] == "Yes").astype(int)
    )
    df_test_orig = df_test_orig.reset_index(drop=True)

    section_header("Watchlist Controls")
    K = st.slider("Top-K Groups", min_value=10, max_value=min(100, len(y_prob)), value=20, step=5)

    watch = build_watchlist(
        df_test_orig, y_test.values, y_prob, tiers, threshold, K=K
    )
    kpi = watch["kpi"]
    comp = watch["comparison"]
    wdf = watch["watchlist_df"]

    # KPIs
    kpi_row([
        {"value": f"{K}", "label": "Groups in Watchlist"},
        {"value": f"{kpi['captured_poor']}", "label": "Poor Outcomes Captured",
         "color": "#d64545"},
        {"value": f"{kpi['precision_at_K']*100:.1f}%", "label": "Precision@K",
         "color": "#f2b705"},
        {"value": f"{kpi['recall_at_K']*100:.1f}%", "label": "Recall@K",
         "color": "#6366f1"},
        {"value": f"{kpi['lift']:.2f}×", "label": "Lift over Random",
         "color": "#2e9e5b"},
    ])

    # Random expectation comparison
    fig_comp = go.Figure(go.Bar(
        x=comp["labels"],
        y=comp["values"],
        marker_color=["#d64545", "#94a3b8"],
        text=[str(v) for v in comp["values"]],
        textposition="outside",
    ))
    apply_theme(fig_comp)
    fig_comp.update_layout(title="Captured Poor Outcomes: Model vs Random",
                           yaxis_title="# Poor Outcomes")
    st.plotly_chart(fig_comp, use_container_width=True)
    insight_box(
        f"The model captures {kpi['captured_poor']} poor-outcome groups in the top {K}, "
        f"vs {kpi['random_expectation']:.1f} expected by random selection "
        f"(base rate {kpi['base_rate']*100:.1f}%). "
        f"That's a {kpi['lift']:.2f}× lift."
    )

    # Watchlist table
    section_header("Top-K Groups Sorted by Risk")
    st.dataframe(wdf, use_container_width=True, hide_index=True)

    # CSV download
    st.download_button(
        "⬇️ Download Watchlist CSV",
        data=wdf.to_csv(index=False),
        file_name=f"fyp_watchlist_top{K}.csv",
        mime="text/csv",
    )

    footer()


# ===========================================================================
# PAGE: WHAT-IF
# ===========================================================================
elif page == "🔀 What-If":
    hero_header("What-If Simulator")

    if st.session_state.train_results is None:
        empty_state("Train models first on the Models page.")
        st.stop()

    st.markdown(ASSOCIATION_NOTE, unsafe_allow_html=True)
    st.markdown("")

    res = st.session_state.train_results
    pipeline = res["final_pipeline"]
    tiers = res["tier_thresholds"]
    feature_cols = res["feature_names"]
    df_clean = st.session_state.df_clean

    section_header("Step 1: Choose a Group (Locked features)")
    col1, col2, col3 = st.columns(3)
    with col1:
        campus = st.selectbox("Campus 🔒", VALID_CAMPUSES, key="wi_campus")
        program = st.selectbox("Program 🔒", ["BSCS", "BSSE", "BSAI", "BSIT", "BSDS"], key="wi_prog")
        leader_gender = st.selectbox("Leader Gender 🔒", ["Male", "Female"], key="wi_gender")
    with col2:
        team_avg_cgpa = st.number_input("Team Avg CGPA 🔒", 0.0, 4.0, 3.0, step=0.01, key="wi_avg")
        team_min_cgpa = st.number_input("Team Min CGPA 🔒", 0.0, 4.0, 2.5, step=0.01, key="wi_min")
        supervisor_experience_years = st.slider("Supervisor Experience (yrs) 🔒", 0, 45, 8, key="wi_exp")
    with col3:
        group_size = st.slider("Group Size", 1, 5, 3, key="wi_gs")
        project_type = st.selectbox("Project Type", ["Web App", "Mobile App", "AI-ML", "IoT", "Research"], key="wi_pt")
        mid_eval_score = st.slider("Mid-Eval Score", 0.0, 100.0, 65.0, step=0.5, key="wi_mes")
        proposal_revisions = st.slider("Proposal Revisions", 0, 10, 2, key="wi_pr")
        weeks_to_proposal_approval = st.slider("Weeks to Approval", 0, 30, 5, key="wi_wpa")

    section_header("Step 2: Adjust Levers")
    col1, col2 = st.columns(2)
    with col1:
        new_meetings = st.slider("Supervisor Meetings (8wk) 📅", 0, 30, 5, key="wi_meet")
        new_commits = st.slider("GitHub Commits (8wk) 💻", 0, 1000, 60, step=5, key="wi_comm")
    with col2:
        new_active_groups = st.slider("Supervisor Active Groups ⚠️", 1, 15, 6, key="wi_ag")
        new_sponsor = st.selectbox("Industry Sponsor 🏭", ["No", "Yes"], key="wi_sp")

    if st.button("⚡ Run What-If Analysis", use_container_width=True):
        # Build base row
        base = {
            "campus": campus, "program": program, "leader_gender": leader_gender,
            "group_size": float(group_size), "team_avg_cgpa": team_avg_cgpa,
            "team_min_cgpa": team_min_cgpa, "project_type": project_type,
            "has_industry_sponsor": "No",  # original locked sponsor
            "supervisor_experience_years": float(supervisor_experience_years),
            "supervisor_active_groups": float(new_active_groups),
            "proposal_revisions": float(proposal_revisions),
            "weeks_to_proposal_approval": float(weeks_to_proposal_approval),
            "supervisor_meetings_8wk": float(new_meetings),
            "github_commits_8wk": float(new_commits),
            "mid_eval_score": float(mid_eval_score),
            "proposal_date": "2024-09-01",
        }
        # Original state (use current lever values as "before" and defaults as "after")
        orig = base.copy()
        orig["supervisor_meetings_8wk"] = 2.0  # low meetings = risky original
        orig["github_commits_8wk"] = 10.0  # low commits
        orig["supervisor_active_groups"] = float(new_active_groups)
        orig["has_industry_sponsor"] = "No"

        df_orig = pd.DataFrame([orig])
        df_orig = engineer_features(df_orig)
        for c in [cc for cc in feature_cols if cc not in df_orig.columns]:
            df_orig[c] = np.nan
        df_orig_model = df_orig[feature_cols]

        df_new = pd.DataFrame([base])
        df_new["has_industry_sponsor"] = new_sponsor
        df_new = engineer_features(df_new)
        for c in [cc for cc in feature_cols if cc not in df_new.columns]:
            df_new[c] = np.nan
        df_new_model = df_new[feature_cols]

        prob_before = float(pipeline.predict_proba(df_orig_model)[:, 1][0])
        prob_after = float(pipeline.predict_proba(df_new_model)[:, 1][0])

        tier_before = "Low" if prob_before < tiers["low_max"] else ("High" if prob_before >= tiers["high_min"] else "Medium")
        tier_after = "Low" if prob_after < tiers["low_max"] else ("High" if prob_after >= tiers["high_min"] else "Medium")

        section_header("Before vs After")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-value" style="color:{TIER_COLORS[tier_before]};">{prob_before:.4f}</div>
  <div class="kpi-label">Before Probability</div>
  <div class="kpi-delta">{risk_badge(tier_before)}</div>
</div>""", unsafe_allow_html=True)
        with col2:
            delta = prob_after - prob_before
            delta_col = "#2e9e5b" if delta < 0 else "#d64545"
            st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-value" style="color:{delta_col};">{delta:+.4f}</div>
  <div class="kpi-label">Change in Probability</div>
  <div class="kpi-delta" style="color:{delta_col};">{delta*100:+.2f}%</div>
</div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-value" style="color:{TIER_COLORS[tier_after]};">{prob_after:.4f}</div>
  <div class="kpi-label">After Probability</div>
  <div class="kpi-delta">{risk_badge(tier_after)}</div>
</div>""", unsafe_allow_html=True)

        # Recommended action after
        if delta < 0:
            st.success(f"✅ Intervention reduced risk by {abs(delta)*100:.2f} percentage points.")
        elif delta > 0:
            st.error(f"⚠️ The changes increased risk by {delta*100:.2f} percentage points.")
        else:
            st.info("No significant change in risk.")

        # Lever ranking
        section_header("Lever Impact Ranking")
        st.info("Which lever moves this group's score the most? (sweep across full range)")
        with st.spinner("Sweeping levers..."):
            df_base = pd.DataFrame([base])
            sens_df = lever_sensitivity(pipeline, df_base, prob_before, feature_cols)
        fig_sens = go.Figure(go.Bar(
            x=sens_df["label"],
            y=sens_df["impact_range"],
            marker_color=PLOTLY_PALETTE[:len(sens_df)],
            text=sens_df["impact_range"].apply(lambda x: f"{x:.4f}"),
            textposition="outside",
        ))
        apply_theme(fig_sens)
        fig_sens.update_layout(
            title="Lever Sensitivity (Probability Range When Swept)",
            xaxis_title="Lever", yaxis_title="Max Prob Range",
        )
        st.plotly_chart(fig_sens, use_container_width=True)
        top_lever = sens_df.iloc[0]
        insight_box(
            f"**{top_lever['label']}** has the largest impact — varying it across its full range "
            f"changes the predicted probability by up to {top_lever['impact_range']:.4f} "
            f"({top_lever['impact_range']*100:.2f} pp)."
        )

        st.markdown(ASSOCIATION_NOTE)

    footer()
