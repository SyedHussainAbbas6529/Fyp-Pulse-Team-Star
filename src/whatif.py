"""
src/whatif.py — What-If Simulator for FYP-Pulse
Team Star | FYP-Pulse | Early Warning System

Editable levers: supervisor_meetings_8wk, github_commits_8wk,
supervisor_active_groups, has_industry_sponsor.

Locked: CGPA, campus, program, leader_gender, supervisor_experience_years.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from src.features import engineer_features

# Levers and their valid ranges
LEVERS: dict[str, dict] = {
    "supervisor_meetings_8wk": {
        "label": "Supervisor Meetings (8wk)",
        "min": 0, "max": 30, "step": 1, "type": "int",
        "help": "Number of supervisor meetings held in the past 8 weeks (0-30).",
    },
    "github_commits_8wk": {
        "label": "GitHub Commits (8wk)",
        "min": 0, "max": 1000, "step": 10, "type": "int",
        "help": "Total GitHub commits in the past 8 weeks (0-1000).",
    },
    "supervisor_active_groups": {
        "label": "Supervisor Active Groups",
        "min": 1, "max": 15, "step": 1, "type": "int",
        "help": "Total number of FYP groups currently supervised (1-15).",
    },
    "has_industry_sponsor": {
        "label": "Industry Sponsor",
        "options": ["Yes", "No"], "type": "select",
        "help": "Whether the project has an industry sponsor.",
    },
}

LOCKED_FEATURES = [
    "team_avg_cgpa", "team_min_cgpa", "campus", "program",
    "leader_gender", "supervisor_experience_years",
]

ASSOCIATION_NOTE = (
    "⚠️ **Note:** The model shows *association*, not a guaranteed causal effect. "
    "Changing these values may not automatically lead to the predicted outcome change."
)


def predict_with_levers(
    pipeline,
    base_row: pd.DataFrame,
    lever_values: dict,
    feature_cols: list[str],
) -> float:
    """Predict probability after applying lever overrides."""
    modified = base_row.copy()
    for col, val in lever_values.items():
        if col in modified.columns:
            modified[col] = val
    # Re-engineer features
    modified = engineer_features(modified)
    # Keep only feature cols
    missing_cols = [c for c in feature_cols if c not in modified.columns]
    for c in missing_cols:
        modified[c] = np.nan
    modified = modified[feature_cols]
    prob = float(pipeline.predict_proba(modified)[:, 1][0])
    return prob


def lever_sensitivity(
    pipeline,
    base_row: pd.DataFrame,
    base_prob: float,
    feature_cols: list[str],
    n_steps: int = 10,
) -> pd.DataFrame:
    """
    Sweep each lever across its valid range and measure probability change.
    Returns DataFrame sorted by impact (descending).
    """
    results = []
    for lever, cfg in LEVERS.items():
        if cfg["type"] == "select":
            # Binary sweep
            vals = ["Yes", "No"]
            probs = []
            for v in vals:
                p = predict_with_levers(pipeline, base_row, {lever: v}, feature_cols)
                probs.append(p)
            impact = max(probs) - min(probs)
            best_val = vals[np.argmin(probs)]  # value that lowers risk the most
            delta = min(probs) - base_prob
        else:
            lo = cfg["min"]
            hi = cfg["max"]
            sweep_vals = np.linspace(lo, hi, n_steps + 1)
            probs = [
                predict_with_levers(pipeline, base_row, {lever: v}, feature_cols)
                for v in sweep_vals
            ]
            impact = max(probs) - min(probs)
            best_val = sweep_vals[np.argmin(probs)]
            delta = min(probs) - base_prob

        results.append({
            "lever": lever,
            "label": cfg["label"],
            "impact_range": round(impact, 4),
            "best_value": best_val,
            "prob_at_best": round(min(probs), 4),
            "delta_from_base": round(delta, 4),
        })

    df = pd.DataFrame(results).sort_values("impact_range", ascending=False).reset_index(drop=True)
    return df
