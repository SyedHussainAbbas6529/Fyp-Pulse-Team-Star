"""
src/ui.py — Shared UI Components for FYP-Pulse
Team Star | FYP-Pulse | Early Warning System
Ultra-Premium Modern UI with Cosmic Glassmorphism, Dynamic Backgrounds, and Fluid Micro-Animations.
"""
from __future__ import annotations
import streamlit as st

# ---------------------------------------------------------------------------
# Theme / Palette
# ---------------------------------------------------------------------------
TIER_COLORS = {
    "Low": "#10b981",      # Emerald
    "Medium": "#f59e0b",   # Amber/Gold
    "High": "#ef4444",     # Crimson
}

PLOTLY_PALETTE = [
    "#38bdf8", "#818cf8", "#34d399", "#f87171", "#fbbf24",
    "#c084fc", "#fb923c", "#f472b6", "#2dd4bf", "#60a5fa",
]


def apply_custom_css():
    """Inject global ultra-premium CSS with fluid animations, glassmorphism, and cosmic effects."""
    st.markdown("""
<style>
/* ---- Google Fonts ---- */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

/* ---- Design Tokens ---- */
:root {
    --primary: #6366f1;
    --primary-glow: rgba(99, 102, 241, 0.35);
    --cyan: #38bdf8;
    --cyan-glow: rgba(56, 189, 248, 0.3);
    --purple: #a855f7;
    --emerald: #10b981;
    --amber: #f59e0b;
    --rose: #f43f5e;
    
    --bg-dark: #070b14;
    --bg-surface: rgba(15, 23, 42, 0.72);
    --bg-card: rgba(30, 41, 59, 0.65);
    --bg-glass: rgba(255, 255, 255, 0.035);
    
    --border-light: rgba(255, 255, 255, 0.08);
    --border-glow: rgba(99, 102, 241, 0.3);
    --border-cyan: rgba(56, 189, 248, 0.35);
    
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --text-subtle: #64748b;
}

/* Global Font & Smooth Antialiasing */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
    color: var(--text-main);
    -webkit-font-smoothing: antialiased;
}

/* ---- Ambient Cosmic Background ---- */
.stApp {
    background-color: var(--bg-dark) !important;
    background-image: 
        radial-gradient(circle at 12% 18%, rgba(99, 102, 241, 0.14) 0%, transparent 45%),
        radial-gradient(circle at 88% 22%, rgba(56, 189, 248, 0.12) 0%, transparent 42%),
        radial-gradient(circle at 50% 75%, rgba(168, 85, 247, 0.09) 0%, transparent 55%),
        radial-gradient(circle at 80% 88%, rgba(16, 185, 129, 0.07) 0%, transparent 40%) !important;
    background-attachment: fixed !important;
}

/* Subtle Animated Ambient Mesh */
@keyframes ambientGlowPulse {
    0% { opacity: 0.75; transform: scale(1); }
    50% { opacity: 1.0; transform: scale(1.02); }
    100% { opacity: 0.75; transform: scale(1); }
}

/* ---- Sidebar Glassmorphism ---- */
[data-testid="stSidebar"] {
    background: rgba(10, 15, 29, 0.82) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-right: 1px solid var(--border-light) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.35) !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: var(--text-main) !important;
}

/* Sidebar Radio Navigation Items */
[data-testid="stSidebar"] .stRadio > div {
    gap: 6px !important;
}
[data-testid="stSidebar"] .stRadio label {
    background: rgba(30, 41, 59, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 10px !important;
    padding: 8px 14px !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    margin-bottom: 3px !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(99, 102, 241, 0.15) !important;
    border-color: rgba(99, 102, 241, 0.4) !important;
    color: #ffffff !important;
    transform: translateX(4px) !important;
}

/* ---- Hero Header with Radiant Glow ---- */
.hero-header {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 50%, rgba(26, 31, 58, 0.7) 100%);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 16px 40px -8px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    animation: fadeInDown 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.12) 0%, transparent 60%);
    pointer-events: none;
    animation: ambientGlowPulse 8s infinite ease-in-out;
}

.hero-logo {
    font-size: 3.4rem;
    display: inline-block;
    filter: drop-shadow(0 0 16px rgba(251, 191, 36, 0.6));
    animation: starFloat 4s ease-in-out infinite;
}

@keyframes starFloat {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-6px) rotate(4deg); }
}

.hero-header h1 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.6rem;
    font-weight: 800;
    margin: 8px 0 6px 0;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #ffffff 30%, #38bdf8 65%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-header p {
    color: #94a3b8;
    margin-top: 4px;
    font-size: 1.1rem;
    font-weight: 400;
    letter-spacing: 0.01em;
}

/* ---- KPI Card with Floating Elevation & Top Illumination ---- */
.kpi-card {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.75), rgba(15, 23, 42, 0.85));
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.06);
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 20%;
    right: 20%;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--primary), transparent);
    opacity: 0.6;
    transition: opacity 0.3s;
}

.kpi-card:hover {
    transform: translateY(-6px) scale(1.02);
    border-color: rgba(99, 102, 241, 0.45);
    box-shadow: 0 16px 36px -6px rgba(99, 102, 241, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.12);
}

.kpi-card:hover::before {
    left: 10%;
    right: 10%;
    opacity: 1;
}

.kpi-card .kpi-value {
    font-family: 'Outfit', sans-serif;
    font-size: 2.35rem;
    font-weight: 700;
    line-height: 1.1;
    letter-spacing: -0.02em;
    filter: drop-shadow(0 2px 8px rgba(0, 0, 0, 0.4));
}

.kpi-card .kpi-label {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #94a3b8;
    margin-top: 6px;
    font-weight: 600;
}

.kpi-card .kpi-delta {
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 6px;
}

/* ---- Section Headers with Neon Accent Bar ---- */
.section-header {
    font-family: 'Outfit', sans-serif;
    font-size: 1.38rem;
    font-weight: 700;
    color: #f8fafc;
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 32px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    position: relative;
}

.section-header::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 0;
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
}

/* ---- Glowing Insight Box ---- */
.insight-box {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.6));
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-left: 3px solid #38bdf8;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 14px 18px;
    color: #cbd5e1;
    font-size: 0.92rem;
    line-height: 1.5;
    margin-top: 8px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s, border-color 0.2s;
}

.insight-box:hover {
    transform: translateX(3px);
    border-left-color: #818cf8;
}

/* ---- Modern Risk Badges with Pulsing Dot ---- */
.badge-low, .badge-medium, .badge-high {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 18px;
    border-radius: 24px;
    font-weight: 700;
    font-size: 0.88rem;
    letter-spacing: 0.03em;
    backdrop-filter: blur(8px);
    transition: all 0.25s ease;
}

.badge-low {
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
    box-shadow: 0 0 16px rgba(16, 185, 129, 0.15);
}

.badge-medium {
    background: rgba(245, 158, 11, 0.12);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.35);
    box-shadow: 0 0 16px rgba(245, 158, 11, 0.15);
}

.badge-high {
    background: rgba(239, 68, 68, 0.14);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
    box-shadow: 0 0 16px rgba(239, 68, 68, 0.22);
}

/* Pulsing dot inside badge */
.badge-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}
.badge-low .badge-dot { background: #34d399; box-shadow: 0 0 8px #34d399; animation: badgePulse 2s infinite; }
.badge-medium .badge-dot { background: #fbbf24; box-shadow: 0 0 8px #fbbf24; animation: badgePulse 1.6s infinite; }
.badge-high .badge-dot { background: #f87171; box-shadow: 0 0 8px #f87171; animation: badgePulse 1.2s infinite; }

@keyframes badgePulse {
    0%, 100% { opacity: 0.4; transform: scale(0.9); }
    50% { opacity: 1.0; transform: scale(1.2); }
}

/* ---- Futuristic Gradient Action Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 50%, #4338ca 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 18px rgba(99, 102, 241, 0.32) !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 26px rgba(99, 102, 241, 0.48) !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
}

.stButton > button:active {
    transform: translateY(1px) !important;
}

/* Download Buttons Secondary Styling */
.stDownloadButton > button {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9)) !important;
    border: 1px solid rgba(56, 189, 248, 0.35) !important;
    color: #38bdf8 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
    transition: all 0.25s ease !important;
}

.stDownloadButton > button:hover {
    border-color: #38bdf8 !important;
    color: #ffffff !important;
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(99, 102, 241, 0.2)) !important;
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.35) !important;
    transform: translateY(-2px) !important;
}

/* ---- Input Controls, Selectors & Sliders ---- */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div {
    background: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
    transition: all 0.2s ease !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.35) !important;
}

/* Slider Track Enhancement */
.stSlider [data-baseweb="slider"] {
    padding: 12px 0 !important;
}

/* ---- DataFrames and Tables ---- */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3) !important;
}

/* ---- Smooth FadeIn Animations ---- */
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-12px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ---- Glass Footer Bar ---- */
.footer-bar {
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    padding: 18px 24px;
    text-align: center;
    color: #64748b;
    font-size: 0.85rem;
    margin-top: 48px;
    border-radius: 14px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}

.footer-bar strong {
    color: #94a3b8;
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
  {'<div class="kpi-delta" style="color:' + color + ';">' + delta + '</div>' if delta else ''}
</div>"""


def kpi_row(cards: list[dict]):
    """Display a row of KPI cards with uniform responsive glass spacing."""
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
    tier_clean = tier.capitalize()
    css_class = f"badge-{tier_clean.lower()}"
    return f'<span class="{css_class}"><span class="badge-dot"></span>{tier_clean} Risk</span>'


def section_header(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def insight_box(text: str):
    st.markdown(f'<div class="insight-box">💡 <b>Dynamic Insight:</b> {text}</div>', unsafe_allow_html=True)


def footer():
    st.markdown("""
<div class="footer-bar">
  Built by <strong>Team Star</strong> ⭐ | FYP-Pulse Early Warning System | Vibe Jam AI Sprint 2026
</div>
""", unsafe_allow_html=True)


def empty_state(message: str = "No data available yet. Train the model first."):
    st.info(f"🔍 {message}")
