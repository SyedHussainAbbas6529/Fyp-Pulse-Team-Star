"""
src/fairness.py — Fairness Analysis for FYP-Pulse
Team Star | FYP-Pulse | Early Warning System

Compares predicted poor-outcome rate and recall across:
- Gender (Male vs Female leaders)
- Campus (campuses with ≥30 test groups)

Uses Wilson confidence intervals and chi-square significance test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _group_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    label: str,
) -> dict:
    """Compute fairness metrics for one sub-group."""
    n = len(y_true)
    if n == 0:
        return {"group": label, "n": 0, "predicted_poor_rate": None,
                "recall": None, "precision": None, "ci_low": None, "ci_high": None}
    y_pred = (y_prob >= threshold).astype(int)
    pred_rate = y_pred.mean()
    actual_poor = int(y_true.sum())
    pred_poor = int(y_pred.sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / pred_poor if pred_poor > 0 else 0.0
    ci = _wilson_ci(int(pred_rate * n), n)
    return {
        "group": label,
        "n": n,
        "actual_poor_outcomes": actual_poor,
        "predicted_positive": pred_poor,
        "predicted_poor_rate": round(pred_rate, 4),
        "recall": round(recall, 4),
        "precision": round(prec, 4),
        "ci_low": round(ci[0], 4),
        "ci_high": round(ci[1], 4),
    }


def _chi2_test(group_a: dict, group_b: dict) -> dict:
    """
    Chi-square test of independence for predicted positive rates between two groups.
    """
    n_a, n_b = group_a["n"], group_b["n"]
    if n_a < 5 or n_b < 5:
        return {"p_value": None, "significant": None, "note": "Too few samples for significance test"}
    pos_a = round(group_a["predicted_poor_rate"] * n_a)
    pos_b = round(group_b["predicted_poor_rate"] * n_b)
    neg_a = n_a - pos_a
    neg_b = n_b - pos_b
    table = np.array([[pos_a, neg_a], [pos_b, neg_b]])
    if table.min() < 1:
        return {"p_value": None, "significant": None, "note": "Expected counts too low"}
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    return {
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p), 4),
        "dof": dof,
        "significant": bool(p < 0.05),
        "note": "Significant disparity (p<0.05)" if p < 0.05 else "No significant disparity (p≥0.05)",
    }


def gender_fairness(
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    """
    Compare Male vs Female leaders on the test set.
    Returns a dict with per-group metrics, gap, and statistical test.
    """
    result = {}
    groups = {}
    for gender in ["Male", "Female"]:
        mask = df_test["leader_gender"] == gender
        g_y = y_test[mask.values]
        g_p = y_prob[mask.values]
        groups[gender] = _group_metrics(g_y, g_p, threshold, gender)

    result["groups"] = groups
    rows = list(groups.values())
    if len(rows) == 2 and all(r["n"] > 0 for r in rows):
        gap_rate = abs(rows[0]["predicted_poor_rate"] - rows[1]["predicted_poor_rate"])
        gap_recall = abs(rows[0]["recall"] - rows[1]["recall"])
        result["rate_gap_pct"] = round(gap_rate * 100, 2)
        result["recall_gap_pct"] = round(gap_recall * 100, 2)
        result["stat_test"] = _chi2_test(rows[0], rows[1])

        # Takeaway
        bigger = "Male" if rows[0]["predicted_poor_rate"] > rows[1]["predicted_poor_rate"] else "Female"
        sig = result["stat_test"].get("significant")
        confidence = "likely a real disparity" if sig else "may be due to chance (small sample)"
        result["takeaway"] = (
            f"The model predicts higher risk for {bigger}-led groups by "
            f"{result['rate_gap_pct']:.1f} percentage points. "
            f"Given the group sizes, this {confidence}."
        )
    return result


def campus_fairness(
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    min_n: int = 30,
) -> dict:
    """
    Compare campuses with ≥ min_n test groups.
    Returns metrics per campus, largest gap, and takeaway.
    """
    result = {}
    campus_col = "campus" if "campus" in df_test.columns else None
    if campus_col is None:
        return {"error": "campus column not found"}

    campus_groups = {}
    for campus in df_test[campus_col].dropna().unique():
        mask = df_test[campus_col] == campus
        n = mask.sum()
        if n < min_n:
            continue
        g_y = y_test[mask.values]
        g_p = y_prob[mask.values]
        campus_groups[campus] = _group_metrics(g_y, g_p, threshold, campus)

    result["groups"] = campus_groups
    result["min_n_threshold"] = min_n
    result["eligible_campuses"] = list(campus_groups.keys())

    if len(campus_groups) >= 2:
        rates = {k: v["predicted_poor_rate"] for k, v in campus_groups.items() if v["n"] >= min_n}
        recalls = {k: v["recall"] for k, v in campus_groups.items() if v["n"] >= min_n}
        if rates:
            max_k = max(rates, key=rates.get)
            min_k = min(rates, key=rates.get)
            gap = round((rates[max_k] - rates[min_k]) * 100, 2)
            result["largest_rate_gap_pct"] = gap
            result["highest_risk_campus"] = max_k
            result["lowest_risk_campus"] = min_k
            # Recall gap
            rec_max_k = max(recalls, key=recalls.get)
            rec_min_k = min(recalls, key=recalls.get)
            result["largest_recall_gap_pct"] = round((recalls[rec_max_k] - recalls[rec_min_k]) * 100, 2)
            result["highest_recall_campus"] = rec_max_k
            result["lowest_recall_campus"] = rec_min_k

        # Takeaway
        n_eligible = len(campus_groups)
        smallest_n = min(v["n"] for v in campus_groups.values())
        confidence_note = (
            "with reasonable confidence (all groups ≥30)" if smallest_n >= 30
            else "with limited confidence (some groups near the 30-group threshold)"
        )
        if gap > 10:
            takeaway = (
                f"A {gap:.1f} percentage-point disparity in predicted risk exists between "
                f"{max_k} (highest) and {min_k} (lowest) campuses, {confidence_note}. "
                f"This may reflect real differences in support infrastructure."
            )
        else:
            takeaway = (
                f"The largest campus gap is {gap:.1f} percentage points, "
                f"which is relatively small across {n_eligible} eligible campuses. "
                f"No major campus-level disparity detected {confidence_note}."
            )
        result["takeaway"] = takeaway
    elif len(campus_groups) == 1:
        result["takeaway"] = (
            f"Only one campus ({list(campus_groups.keys())[0]}) has ≥{min_n} test groups. "
            f"Cannot compare campuses with the current test-set size."
        )
    else:
        result["takeaway"] = (
            f"No campuses meet the minimum {min_n}-group threshold in the test set. "
            f"Fairness analysis is inconclusive — consider a larger test set."
        )
    return result
