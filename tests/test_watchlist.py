"""
tests/test_watchlist.py — Tests for watchlist correctness
Team Star | FYP-Pulse
"""
import pytest
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.watchlist import build_watchlist


@pytest.fixture
def mock_data():
    rng = np.random.default_rng(42)
    n = 200
    y_test = rng.choice([0, 1], n, p=[0.84, 0.16])
    y_prob = rng.uniform(0, 1, n)
    df = pd.DataFrame({"group_id": [f"G{i:04d}" for i in range(n)]})
    tiers = {"low_max": 0.35, "high_min": 0.60}
    return y_test, y_prob, df, tiers


class TestWatchlist:
    def test_k_rows(self, mock_data):
        y_test, y_prob, df, tiers = mock_data
        for K in [10, 20, 50, 100]:
            result = build_watchlist(df, y_test, y_prob, tiers, 0.3, K=K)
            assert len(result["watchlist_df"]) == K

    def test_sorted_by_risk_desc(self, mock_data):
        y_test, y_prob, df, tiers = mock_data
        result = build_watchlist(df, y_test, y_prob, tiers, 0.3, K=20)
        probs = result["watchlist_df"]["Risk Probability"].values
        assert all(probs[i] >= probs[i+1] for i in range(len(probs)-1)), "Not sorted descending"

    def test_k_range_10_to_100(self, mock_data):
        y_test, y_prob, df, tiers = mock_data
        for K in range(10, 101, 10):
            result = build_watchlist(df, y_test, y_prob, tiers, 0.3, K=K)
            assert result["kpi"]["K"] == K

    def test_precision_at_k_valid(self, mock_data):
        y_test, y_prob, df, tiers = mock_data
        result = build_watchlist(df, y_test, y_prob, tiers, 0.3, K=20)
        prec = result["kpi"]["precision_at_K"]
        assert 0.0 <= prec <= 1.0

    def test_recall_at_k_valid(self, mock_data):
        y_test, y_prob, df, tiers = mock_data
        result = build_watchlist(df, y_test, y_prob, tiers, 0.3, K=20)
        rec = result["kpi"]["recall_at_K"]
        assert 0.0 <= rec <= 1.0

    def test_lift_positive(self, mock_data):
        y_test, y_prob, df, tiers = mock_data
        # Make probs correlate with y_test so lift should be > 1
        y_test_biased = np.zeros(200, dtype=int)
        y_test_biased[:20] = 1
        y_prob_biased = np.concatenate([
            np.random.uniform(0.7, 1.0, 20),
            np.random.uniform(0.0, 0.4, 180)
        ])
        result = build_watchlist(df, y_test_biased, y_prob_biased, tiers, 0.3, K=20)
        assert result["kpi"]["lift"] >= 1.0

    def test_random_expectation_formula(self, mock_data):
        y_test, y_prob, df, tiers = mock_data
        K = 25
        result = build_watchlist(df, y_test, y_prob, tiers, 0.3, K=K)
        kpi = result["kpi"]
        expected = round(K * kpi["base_rate"], 2)
        assert abs(kpi["random_expectation"] - expected) < 0.01
