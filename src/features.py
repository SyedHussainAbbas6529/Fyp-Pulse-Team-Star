"""
src/features.py — Feature Engineering for FYP-Pulse
Team Star | FYP-Pulse | Early Warning System

All features are computed ONLY from week-8-safe columns.
Leakage columns (final_defense_score, extension_granted, group_id) are excluded.
Features are explained inline.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Feature definitions with human-readable explanations
# ---------------------------------------------------------------------------
FEATURE_EXPLANATIONS: dict[str, str] = {
    "commits_per_meeting": (
        "GitHub commits ÷ supervisor meetings. High = productive but possibly "
        "unsupervised; low = disengaged or only meeting without output."
    ),
    "commits_per_week": (
        "Average weekly commit rate over 8 weeks. Directly captures development momentum."
    ),
    "meetings_per_week": (
        "Average weekly supervisor meetings over 8 weeks. Reflects supervisor engagement."
    ),
    "supervisor_load_ratio": (
        "Supervisor's active groups ÷ experience years. High = overloaded supervisor "
        "relative to experience — a known risk factor."
    ),
    "cgpa_gap": (
        "Difference between average and minimum team CGPA. Large gap suggests "
        "uneven capability distribution within the team."
    ),
    "low_engagement_flag": (
        "Binary flag: 1 if commits_per_week < 5 AND meetings_per_week < 1. "
        "Captures the double-disengagement scenario (no code AND no meetings) "
        "which strongly predicts poor outcomes."
    ),
    "mid_eval_vs_cgpa_mismatch": (
        "Mid-eval score minus (avg_cgpa × 25). Positive = overperforming relative "
        "to CGPA; negative = underperforming. Large negative gaps are a warning sign."
    ),
    "approval_delay_flag": (
        "Binary flag: 1 if weeks_to_proposal_approval > 10. Late approvals cost "
        "working weeks and correlate with eventual delays."
    ),
    "proposal_month": (
        "Month of proposal date (1-12). Captures academic calendar effects — "
        "early-semester proposals may have more runway."
    ),
    "sponsor_x_type": (
        "Interaction: has_industry_sponsor=Yes AND project_type=AI-ML (1 else 0). "
        "Captures whether sponsored AI-ML projects have different risk profiles."
    ),
}

# Leakage columns — NEVER used as features
LEAKAGE_COLS = ["final_defense_score", "extension_granted", "group_id"]

# All feature columns used by the model (after encoding)
CATEGORICAL_FEATURES = ["campus", "program", "leader_gender", "project_type", "has_industry_sponsor"]
NUMERIC_BASE = [
    "group_size", "team_avg_cgpa", "team_min_cgpa",
    "supervisor_experience_years", "supervisor_active_groups",
    "proposal_revisions", "weeks_to_proposal_approval",
    "supervisor_meetings_8wk", "github_commits_8wk", "mid_eval_score",
]
NUMERIC_ENGINEERED = list(FEATURE_EXPLANATIONS.keys())

TARGET_COL = "poor_outcome"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the dataframe.
    Returns a new dataframe with additional columns.
    Does NOT remove any existing columns.
    """
    df = df.copy()

    # Safe division helper
    def safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
        return a / b.replace(0, np.nan).fillna(np.nan)

    # --- Commits per meeting ---
    df["commits_per_meeting"] = safe_div(
        df.get("github_commits_8wk", pd.Series(np.nan, index=df.index)),
        df.get("supervisor_meetings_8wk", pd.Series(np.nan, index=df.index)),
    ).fillna(0)

    # --- Commits per week ---
    df["commits_per_week"] = df.get(
        "github_commits_8wk", pd.Series(np.nan, index=df.index)
    ).fillna(0) / 8.0

    # --- Meetings per week ---
    df["meetings_per_week"] = df.get(
        "supervisor_meetings_8wk", pd.Series(np.nan, index=df.index)
    ).fillna(0) / 8.0

    # --- Supervisor load ratio ---
    df["supervisor_load_ratio"] = safe_div(
        df.get("supervisor_active_groups", pd.Series(np.nan, index=df.index)),
        df.get("supervisor_experience_years", pd.Series(np.nan, index=df.index)),
    ).fillna(0)

    # --- CGPA gap ---
    df["cgpa_gap"] = (
        df.get("team_avg_cgpa", pd.Series(np.nan, index=df.index)) -
        df.get("team_min_cgpa", pd.Series(np.nan, index=df.index))
    ).clip(lower=0)

    # --- Low engagement flag ---
    cpw = df["commits_per_week"]
    mpw = df["meetings_per_week"]
    df["low_engagement_flag"] = ((cpw < 5) & (mpw < 1)).astype(int)

    # --- Mid eval vs CGPA mismatch ---
    df["mid_eval_vs_cgpa_mismatch"] = (
        df.get("mid_eval_score", pd.Series(np.nan, index=df.index)) -
        (df.get("team_avg_cgpa", pd.Series(np.nan, index=df.index)) * 25)
    )

    # --- Approval delay flag ---
    wpa = df.get("weeks_to_proposal_approval", pd.Series(np.nan, index=df.index))
    df["approval_delay_flag"] = (wpa > 10).astype(int)

    # --- Proposal month ---
    if "proposal_date" in df.columns:
        df["proposal_month"] = pd.to_datetime(df["proposal_date"], errors="coerce").dt.month.fillna(0).astype(int)
    else:
        df["proposal_month"] = 0

    # --- Sponsor × type interaction ---
    sponsor = (df.get("has_industry_sponsor", pd.Series("No", index=df.index))
               .str.strip().str.lower().map({"yes": 1, "no": 0}).fillna(0))
    ptype = (df.get("project_type", pd.Series("", index=df.index))
             .str.strip().str.lower().map({"ai-ml": 1}).fillna(0))
    df["sponsor_x_type"] = (sponsor * ptype).astype(int)

    return df


def get_feature_columns() -> dict[str, list[str]]:
    """Return feature column groups."""
    return {
        "numeric_base": NUMERIC_BASE,
        "numeric_engineered": NUMERIC_ENGINEERED,
        "categorical": CATEGORICAL_FEATURES,
        "leakage": LEAKAGE_COLS,
        "target": TARGET_COL,
        "explanations": FEATURE_EXPLANATIONS,
    }
