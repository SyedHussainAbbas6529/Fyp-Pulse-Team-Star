"""
src/watchlist.py — Weekly Watchlist for FYP-Pulse
Team Star | FYP-Pulse | Early Warning System

Generates top-K highest-risk TEST groups with metrics: precision@K,
recall@K, lift, random expectation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from src.explain import READABLE_NAMES


def build_watchlist(
    df_test_original: pd.DataFrame,
    y_test: np.ndarray,
    y_prob: np.ndarray,
    tier_thresholds: dict,
    threshold: float,
    K: int = 20,
    top_drivers_fn=None,
) -> dict:
    """
    Build the weekly watchlist for K highest-risk test groups.

    Parameters
    ----------
    df_test_original : DataFrame with original columns (including group_id if available)
    y_test : true binary labels
    y_prob : predicted probabilities
    tier_thresholds : {low_max, high_min}
    threshold : decision threshold
    K : number of groups to show
    top_drivers_fn : optional callable(X_row) → str (top driver name)

    Returns
    -------
    dict with keys: watchlist_df, kpi, comparison
    """
    n_total = len(y_prob)
    base_rate = y_test.mean()

    # Sort by probability descending
    order = np.argsort(y_prob)[::-1]
    top_k_idx = order[:K]

    y_top = y_test[top_k_idx]
    p_top = y_prob[top_k_idx]

    # Tier assignment
    def assign_tier(p):
        if p >= tier_thresholds.get("high_min", 0.6):
            return "🔴 High"
        elif p >= tier_thresholds.get("low_max", 0.35):
            return "🟡 Medium"
        return "🟢 Low"

    tiers = [assign_tier(p) for p in p_top]

    # Reconstruct group IDs
    df_top = df_test_original.iloc[top_k_idx].reset_index(drop=True) if df_test_original is not None else pd.DataFrame()

    rows = []
    for rank, (orig_idx, prob, tier, actual) in enumerate(
        zip(top_k_idx, p_top, tiers, y_top), start=1
    ):
        gid = df_top.iloc[rank - 1].get("group_id", f"Group #{orig_idx}") if len(df_top) > 0 else f"Group #{orig_idx}"
        rows.append({
            "Rank": rank,
            "Group ID": gid,
            "Risk Probability": round(float(prob), 4),
            "Risk %": f"{prob*100:.1f}%",
            "Tier": tier,
            "Actual Poor Outcome": "✅ Yes" if actual == 1 else "No",
        })

    watchlist_df = pd.DataFrame(rows)

    # KPI metrics
    captured = int(y_top.sum())   # truly poor outcomes in top-K
    total_poor = int(y_test.sum())
    precision_k = captured / K if K > 0 else 0.0
    recall_k = captured / total_poor if total_poor > 0 else 0.0
    random_exp = K * base_rate
    lift = precision_k / base_rate if base_rate > 0 else 0.0

    kpi = {
        "K": K,
        "total_test": n_total,
        "total_poor": total_poor,
        "base_rate": round(float(base_rate), 4),
        "captured_poor": captured,
        "precision_at_K": round(precision_k, 4),
        "recall_at_K": round(recall_k, 4),
        "random_expectation": round(float(random_exp), 2),
        "lift": round(lift, 2),
    }

    # Comparison data for bar chart
    comparison = {
        "labels": ["Top-K Captured", "Random Expectation"],
        "values": [captured, round(float(random_exp), 2)],
    }

    return {
        "watchlist_df": watchlist_df,
        "kpi": kpi,
        "comparison": comparison,
        "top_k_indices": top_k_idx.tolist(),
    }
