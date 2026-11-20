"""
src/explain.py — Explainability for FYP-Pulse
Team Star | FYP-Pulse | Early Warning System

Custom baseline-replacement explainer (SHAP-inspired but no SHAP dependency).
Works by measuring each feature's contribution: replace feature with its training
median/mode and measure predicted-probability drop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# Human-readable feature name map
READABLE_NAMES: dict[str, str] = {
    "group_size": "Team Size",
    "team_avg_cgpa": "Team Avg CGPA",
    "team_min_cgpa": "Team Min CGPA",
    "supervisor_experience_years": "Supervisor Experience (yrs)",
    "supervisor_active_groups": "Supervisor Active Groups",
    "proposal_revisions": "Proposal Revisions",
    "weeks_to_proposal_approval": "Weeks to Approval",
    "supervisor_meetings_8wk": "Supervisor Meetings (8wk)",
    "github_commits_8wk": "GitHub Commits (8wk)",
    "mid_eval_score": "Mid-Eval Score",
    "commits_per_meeting": "Commits per Meeting",
    "commits_per_week": "Commits per Week",
    "meetings_per_week": "Meetings per Week",
    "supervisor_load_ratio": "Supervisor Load Ratio",
    "cgpa_gap": "CGPA Gap (Avg−Min)",
    "low_engagement_flag": "Low Engagement Flag",
    "mid_eval_vs_cgpa_mismatch": "Mid-Eval vs CGPA Mismatch",
    "approval_delay_flag": "Approval Delay Flag",
    "proposal_month": "Proposal Month",
    "sponsor_x_type": "Sponsor × AI-ML Type",
    "campus": "Campus",
    "program": "Program",
    "leader_gender": "Leader Gender",
    "project_type": "Project Type",
    "has_industry_sponsor": "Industry Sponsor",
}

# Action map: biggest risk driver → concrete support step
ACTION_MAP: dict[str, str] = {
    "low_engagement_flag": (
        "📅 Schedule a weekly check-in: both supervisor meetings AND a code-push "
        "are below the engagement threshold. Coordinator should intervene this week."
    ),
    "supervisor_meetings_8wk": (
        "📅 Schedule more supervisor meetings — the group has had too few check-ins "
        "over the past 8 weeks. Contact the supervisor to set a weekly slot."
    ),
    "meetings_per_week": (
        "📅 Increase meeting frequency — this group is meeting less than once per week. "
        "Coordinator should request a weekly supervisor slot."
    ),
    "github_commits_8wk": (
        "💻 Encourage code activity — very few commits signal stalled development. "
        "Suggest a 24-hour coding sprint or a progress review."
    ),
    "commits_per_week": (
        "💻 Encourage more frequent commits — the group averages fewer than 5 commits/week. "
        "Set a minimum weekly code-push target."
    ),
    "commits_per_meeting": (
        "⚡ Output is low relative to meetings — the team may be discussing without building. "
        "Ask for a working prototype demo at the next meeting."
    ),
    "mid_eval_score": (
        "🎯 Mid-eval score is below average — schedule a focused improvement session and "
        "review the mid-eval rubric with the supervisor."
    ),
    "mid_eval_vs_cgpa_mismatch": (
        "🎯 Mid-eval score is below what the team CGPA would predict — this is a warning "
        "sign. Arrange a progress review meeting this week."
    ),
    "supervisor_active_groups": (
        "⚠️ Supervisor may be overloaded — consider assigning a co-supervisor or reducing "
        "the supervisor's group count."
    ),
    "supervisor_load_ratio": (
        "⚠️ Supervisor load is high relative to experience — recommend pairing with a "
        "more experienced co-supervisor."
    ),
    "approval_delay_flag": (
        "⏰ Proposal approval was significantly delayed — lost weeks compress the timeline. "
        "Accelerate progress milestones to compensate."
    ),
    "weeks_to_proposal_approval": (
        "⏰ Proposal took longer than usual to approve — ensure remaining milestones "
        "are on an accelerated schedule."
    ),
    "proposal_revisions": (
        "📝 Many proposal revisions indicate unclear project scope. Hold a scope-clarification "
        "workshop with the supervisor."
    ),
    "team_avg_cgpa": (
        "📚 Low team CGPA may indicate academic difficulty — connect the group with "
        "academic support and tutoring resources."
    ),
    "team_min_cgpa": (
        "📚 The weakest team member has a low CGPA — consider pairing them with "
        "a stronger peer for mentoring."
    ),
    "cgpa_gap": (
        "🤝 Large CGPA gap within the team risks uneven workload — redistribute tasks "
        "and strengthen peer collaboration."
    ),
    "group_size": (
        "👥 Small group size increases individual workload risk — monitor task distribution "
        "and consider whether scope is achievable."
    ),
}

DEFAULT_ACTION = (
    "📋 Schedule a comprehensive progress review meeting with the supervisor "
    "and ensure all milestones are on track."
)


def compute_feature_importances(
    pipeline,
    X_row: pd.DataFrame,
    X_train: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Baseline-replacement explainer:
    For each feature, replace its value with the training median/mode,
    predict, and measure probability drop. Positive delta = feature raised risk;
    negative = feature lowered risk.

    Returns DataFrame with columns: feature, readable_name, contribution, direction.
    """
    baseline_prob = float(pipeline.predict_proba(X_row)[:, 1][0])

    contribs = []
    feature_cols = [c for c in X_row.columns if c in X_train.columns]

    for col in feature_cols:
        X_mod = X_row.copy()
        # Replace with training median or mode
        if X_train[col].dtype in [np.float64, np.float32, float, int, np.int64]:
            baseline_val = X_train[col].median()
        else:
            mode_vals = X_train[col].mode()
            baseline_val = mode_vals.iloc[0] if len(mode_vals) > 0 else X_row[col].iloc[0]

        X_mod[col] = baseline_val
        mod_prob = float(pipeline.predict_proba(X_mod)[:, 1][0])
        contrib = baseline_prob - mod_prob  # positive = this feature raises risk above baseline
        contribs.append({
            "feature": col,
            "readable_name": READABLE_NAMES.get(col, col.replace("_", " ").title()),
            "contribution": contrib,
            "current_value": X_row[col].iloc[0],
            "baseline_value": baseline_val,
        })

    df = pd.DataFrame(contribs)
    df["abs_contrib"] = df["contribution"].abs()
    df = df.sort_values("abs_contrib", ascending=False).head(top_n).reset_index(drop=True)
    df["direction"] = df["contribution"].apply(lambda x: "risk_up" if x > 0 else "risk_down")
    return df


def top_drivers(
    pipeline,
    X_row: pd.DataFrame,
    X_train: pd.DataFrame,
    n: int = 3,
) -> tuple[pd.DataFrame, str, str]:
    """
    Returns:
    - top_df: top n risk drivers (DataFrame)
    - plain_text: human-readable explanation
    - action: concrete support step for the biggest driver
    """
    df = compute_feature_importances(pipeline, X_row, X_train, top_n=n)
    prob = float(pipeline.predict_proba(X_row)[:, 1][0])

    # Plain language explanation
    lines = []
    for _, row in df.iterrows():
        name = row["readable_name"]
        c = row["contribution"]
        val = row["current_value"]
        if abs(c) < 0.001:
            continue
        direction = "increases" if c > 0 else "decreases"
        impact = "significantly" if abs(c) > 0.05 else "moderately"
        val_str = f"{val:.2f}" if isinstance(val, (int, float, np.number)) else str(val)
        lines.append(
            f"• **{name}** (value: {val_str}) {impact} {direction} the risk."
        )

    if not lines:
        plain = "No strong individual feature effects detected for this group."
    else:
        plain = "\n".join(lines)

    # Action: use biggest positive contributor (biggest risk driver)
    top_risk_drivers = df[df["direction"] == "risk_up"]
    if len(top_risk_drivers) > 0:
        biggest = top_risk_drivers.iloc[0]["feature"]
    elif len(df) > 0:
        biggest = df.iloc[0]["feature"]
    else:
        biggest = ""

    action = ACTION_MAP.get(biggest, DEFAULT_ACTION)
    return df, plain, action
