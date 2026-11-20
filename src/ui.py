"""
src/ui.py — Shared UI Components for FYP-Pulse
Team Star | FYP-Pulse | Early Warning System
"""
from __future__ import annotations
import streamlit as st

# ---------------------------------------------------------------------------
# Theme / Palette
# ---------------------------------------------------------------------------
TIER_COLORS = {
    "Low": "#2e9e5b",
    "Medium": "#f2b705",
    "High": "#d64545",
}

PLOTLY_PALETTE = [
    "#6366f1", "#f2b705", "#2e9e5b", "#d64545", "#06b6d4",
    "#a855f7", "#f97316", "#ec4899", "#10b981", "#f59e0b",
]


def apply_custom_css():
    """Inject global custom CSS for the app."""
    st.markdown("""
<style>
/* ---- Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ---- Root variables ---- */
:root {
    --primary: #6366f1;
    --success: #2e9e5b;
    --warning: #f2b705;
    --danger: #d64545;
    --bg: #0f172a;
    --card-bg: #1e293b;
    --border: #334155;
    --text: #e2e8f0;
    --text-muted: #94a3b8;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1a1f3a 100%) !important;
    border-right: 1px solid #334155 !important;
}
[data-testid="stSidebar"] .stMarkdown { color: #e2e8f0 !important; }

/* ---- KPI Card ---- */
.kpi-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-card .kpi-value { font-size: 2.2rem; font-weight: 700; color: #6366f1; }
.kpi-card .kpi-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
.kpi-card .kpi-delta { font-size: 0.8rem; margin-top: 6px; }

/* ---- Hero Header ---- */
.hero-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #1a1f3a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    text-align: center;
}
.hero-header h1 { font-size: 2.4rem; font-weight: 700; color: #e2e8f0; margin: 0; }
.hero-header p { color: #94a3b8; margin-top: 8px; font-size: 1.05rem; }
.hero-logo { font-size: 3rem; }

/* ---- Risk Badge ---- */
.badge-low { background:#2e9e5b22; color:#2e9e5b; border:1px solid #2e9e5b; border-radius:20px; padding:4px 16px; font-weight:600; display:inline-block; }
.badge-medium { background:#f2b70522; color:#f2b705; border:1px solid #f2b705; border-radius:20px; padding:4px 16px; font-weight:600; display:inline-block; }
.badge-high { background:#d6454522; color:#d64545; border:1px solid #d64545; border-radius:20px; padding:4px 16px; font-weight:600; display:inline-block; }

/* ---- Section headers ---- */
.section-header {
    font-size: 1.3rem; font-weight: 600; color: #e2e8f0;
    border-left: 4px solid #6366f1; padding-left: 12px;
    margin: 24px 0 12px 0;
}

/* ---- Insight box ---- */
.insight-box {
    background: #1e293b;
    border-left: 3px solid #6366f1;
    border-radius: 8px;
    padding: 12px 16px;
    color: #94a3b8;
    font-size: 0.9rem;
    margin-top: 4px;
}

/* ---- Tooltip style ---- */
.tooltip-icon { cursor: help; color: #6366f1; font-size: 0.85rem; }

/* ---- Table improvements ---- */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ---- Button override ---- */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white; border: none; border-radius: 8px;
    padding: 8px 20px; font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5, #4338ca);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99,102,241,0.3);
}

/* ---- Footer ---- */
.footer-bar {
    background: #0f172a;
    border-top: 1px solid #334155;
    padding: 16px;
    text-align: center;
    color: #64748b;
    font-size: 0.8rem;
    margin-top: 40px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


def hero_header(subtitle: str = "Final Year Project Early Warning System"):
    st.markdown(f"""
<div class="hero-header">
  <div class="hero-logo">⭐</div>
  <h1>FYP-Pulse | Team Star</h1>
  <p>{subtitle}</p>
</div>
""", unsafe_allow_html=True)


def kpi_card(value, label: str, delta: str = "", color: str = "#6366f1"):
    return f"""
<div class="kpi-card">
  <div class="kpi-value" style="color:{color};">{value}</div>
  <div class="kpi-label">{label}</div>
  {'<div class="kpi-delta">' + delta + '</div>' if delta else ''}
</div>"""


def kpi_row(cards: list[dict]):
    """Display a row of KPI cards. Each dict: {value, label, delta?, color?}"""
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            st.markdown(
                kpi_card(
                    card["value"], card["label"],
                    card.get("delta", ""), card.get("color", "#6366f1")
                ),
                unsafe_allow_html=True
            )


def risk_badge(tier: str) -> str:
    css_class = f"badge-{tier.lower()}"
    return f'<span class="{css_class}">{tier} Risk</span>'


def section_header(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def insight_box(text: str):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


def footer():
    st.markdown("""
<div class="footer-bar">
  Built by <strong>Team Star</strong> ⭐ | FYP-Pulse Early Warning System | Vibe Jam AI Sprint 2026
</div>
""", unsafe_allow_html=True)


def empty_state(message: str = "No data available yet. Train the model first."):
    st.info(f"🔍 {message}")
