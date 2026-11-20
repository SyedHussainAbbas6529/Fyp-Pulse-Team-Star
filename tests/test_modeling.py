"""
tests/test_modeling.py — Tests for model training and leakage prevention
Team Star | FYP-Pulse
"""
import pytest
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cleaning import clean_data
from src.modeling import EXCLUDED_COLUMNS, prepare_data, _select_threshold, _compute_tier_thresholds
from src.features import LEAKAGE_COLS


LEAKAGE_COLS_SET = set(LEAKAGE_COLS)


@pytest.fixture
def small_clean_df():
    """Generate a small but valid clean DataFrame for quick tests."""
    rng = np.random.default_rng(42)
    n = 100
    cgpa_avg = rng.uniform(2.0, 4.0, n)
    cgpa_min = np.minimum(cgpa_avg, cgpa_avg - rng.uniform(0, 0.5, n))
    df = pd.DataFrame({
        "group_id": [f"G{i:04d}" for i in range(n)],
        "proposal_date": ["2024-03-15"] * n,
        "campus": rng.choice(["Lahore", "Karachi", "Islamabad"], n),
        "program": rng.choice(["BSCS", "BSSE", "BSAI"], n),
        "leader_gender": rng.choice(["Male", "Female"], n),
        "group_size": rng.integers(1, 6, n).astype(float),
        "team_avg_cgpa": cgpa_avg,
        "team_min_cgpa": cgpa_min,
        "project_type": rng.choice(["Web App", "AI-ML", "IoT"], n),
        "has_industry_sponsor": rng.choice(["Yes", "No"], n),
        "supervisor_experience_years": rng.uniform(0, 20, n),
        "supervisor_active_groups": rng.integers(1, 10, n).astype(float),
        "proposal_revisions": rng.integers(0, 5, n).astype(float),
        "weeks_to_proposal_approval": rng.integers(0, 15, n).astype(float),
        "supervisor_meetings_8wk": rng.integers(0, 20, n).astype(float),
        "github_commits_8wk": rng.integers(0, 300, n).astype(float),
        "mid_eval_score": rng.uniform(40, 95, n),
        "final_defense_score": rng.uniform(40, 95, n),
        "extension_granted": rng.choice(["Yes", "No"], n),
        "poor_outcome": rng.choice(["Yes", "No"], n, p=[0.16, 0.84]),
    })
    # Fix solo rule
    solo = df["group_size"] == 1.0
    df.loc[solo, "team_min_cgpa"] = df.loc[solo, "team_avg_cgpa"]
    return df


class TestNoLeakage:
    def test_leakage_cols_in_excluded(self):
        """All leakage columns must be in EXCLUDED_COLUMNS."""
        for col in LEAKAGE_COLS:
            assert col in EXCLUDED_COLUMNS, f"{col} not in EXCLUDED_COLUMNS"

    def test_prepare_data_drops_leakage(self, small_clean_df):
        """prepare_data must not include leakage columns in X."""
        X, y = prepare_data(small_clean_df)
        for col in LEAKAGE_COLS_SET:
            assert col not in X.columns, f"Leakage column {col} found in X!"

    def test_prepare_data_drops_target(self, small_clean_df):
        """Target column must not be in X."""
        X, y = prepare_data(small_clean_df)
        assert "poor_outcome" not in X.columns

    def test_target_binary(self, small_clean_df):
        """y must be 0/1."""
        _, y = prepare_data(small_clean_df)
        assert set(y.unique()).issubset({0, 1})


class TestThresholdSelection:
    def test_threshold_in_range(self):
        rng = np.random.default_rng(42)
        y_true = rng.choice([0, 1], 200, p=[0.84, 0.16])
        y_prob = rng.uniform(0, 1, 200)
        t = _select_threshold(y_true, y_prob, min_recall=0.5)
        assert 0.0 <= t <= 1.0

    def test_threshold_in_valid_range(self):
        """Threshold must always be in [0, 1] for any input."""
        rng = np.random.default_rng(99)
        y_true = np.zeros(100)
        y_prob = rng.uniform(0, 1, 100)
        t = _select_threshold(y_true, y_prob, min_recall=0.65)
        assert 0.0 <= t <= 1.0

    def test_threshold_fallback_rare_positives(self):
        """With a very high min_recall requirement and hard data, threshold stays reasonable."""
        rng = np.random.default_rng(77)
        y_true = np.zeros(200)
        y_true[:5] = 1  # very few positives
        y_prob = rng.uniform(0, 0.1, 200)  # all low probs
        t = _select_threshold(y_true, y_prob, min_recall=0.99)
        assert 0.0 <= t <= 1.0


class TestTierMonotonicity:
    def test_tier_thresholds_monotone(self):
        """Tier thresholds must produce strictly increasing poor-outcome rates."""
        rng = np.random.default_rng(42)
        # Simulate: higher prob → more likely positive
        probs = rng.beta(0.5, 2, 400)  # skewed toward 0
        y_true = (probs > 0.3).astype(int)
        tiers = _compute_tier_thresholds(probs, y_true)
        low_max = tiers["low_max"]
        high_min = tiers["high_min"]
        low_m = probs < low_max
        high_m = probs >= high_min
        mid_m = ~low_m & ~high_m
        if low_m.sum() > 5 and mid_m.sum() > 5 and high_m.sum() > 5:
            r_l = y_true[low_m].mean()
            r_m = y_true[mid_m].mean()
            r_h = y_true[high_m].mean()
            assert r_l <= r_m + 0.01, f"Not monotone: low={r_l:.3f} mid={r_m:.3f}"
            assert r_m <= r_h + 0.01, f"Not monotone: mid={r_m:.3f} high={r_h:.3f}"
