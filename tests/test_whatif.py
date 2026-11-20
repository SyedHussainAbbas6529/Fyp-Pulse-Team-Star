"""
tests/test_whatif.py — Tests for What-If simulator bounds and behaviour
Team Star | FYP-Pulse
"""
import pytest
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.whatif import LEVERS, LOCKED_FEATURES, predict_with_levers, lever_sensitivity


class TestLeverConfig:
    def test_lever_ranges_valid(self):
        """All levers with min/max should have min < max."""
        for name, cfg in LEVERS.items():
            if cfg["type"] in ("int", "float"):
                assert cfg["min"] < cfg["max"], f"Lever {name}: min >= max"

    def test_locked_features_not_levers(self):
        """Locked features must not appear as levers."""
        lever_names = set(LEVERS.keys())
        locked_names = set(LOCKED_FEATURES)
        overlap = lever_names & locked_names
        assert len(overlap) == 0, f"Levers overlap with locked features: {overlap}"

    def test_all_levers_have_labels(self):
        for name, cfg in LEVERS.items():
            assert "label" in cfg, f"Lever {name} has no label"


class TestWhatIfBounds:
    """Tests that what-if outputs stay in [0, 1] probability range."""

    def _make_mock_pipeline(self):
        """Create a minimal sklearn pipeline for testing."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        import numpy as np

        # Train on tiny dummy data
        rng = np.random.default_rng(42)
        n = 50
        X = pd.DataFrame({
            "group_size": rng.uniform(1, 5, n),
            "team_avg_cgpa": rng.uniform(2, 4, n),
            "team_min_cgpa": rng.uniform(1.5, 3.5, n),
            "supervisor_experience_years": rng.uniform(0, 20, n),
            "supervisor_active_groups": rng.uniform(1, 10, n),
            "proposal_revisions": rng.uniform(0, 5, n),
            "weeks_to_proposal_approval": rng.uniform(0, 15, n),
            "supervisor_meetings_8wk": rng.uniform(0, 20, n),
            "github_commits_8wk": rng.uniform(0, 300, n),
            "mid_eval_score": rng.uniform(40, 95, n),
            "commits_per_meeting": rng.uniform(0, 50, n),
            "commits_per_week": rng.uniform(0, 40, n),
            "meetings_per_week": rng.uniform(0, 4, n),
            "supervisor_load_ratio": rng.uniform(0, 5, n),
            "cgpa_gap": rng.uniform(0, 1, n),
            "low_engagement_flag": rng.choice([0, 1], n).astype(float),
            "mid_eval_vs_cgpa_mismatch": rng.uniform(-30, 30, n),
            "approval_delay_flag": rng.choice([0, 1], n).astype(float),
            "proposal_month": rng.integers(1, 13, n).astype(float),
            "sponsor_x_type": rng.choice([0, 1], n).astype(float),
        })
        y = rng.choice([0, 1], n, p=[0.84, 0.16])

        num_cols = list(X.columns)
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())]), num_cols)
        ])
        pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=200))])
        pipe.fit(X, y)
        return pipe, num_cols

    def test_predict_in_unit_interval(self):
        pipe, feature_cols = self._make_mock_pipeline()
        base = pd.DataFrame([{c: 0.5 for c in feature_cols}])
        for lever_val in [0, 10, 30]:
            prob = predict_with_levers(pipe, base, {"supervisor_meetings_8wk": lever_val}, feature_cols)
            assert 0.0 <= prob <= 1.0, f"Prob {prob} out of [0,1]"

    def test_sensitivity_returns_sorted_df(self):
        pipe, feature_cols = self._make_mock_pipeline()
        base = pd.DataFrame([{c: 0.5 for c in feature_cols}])
        base_prob = float(pipe.predict_proba(base)[:, 1][0])
        df = lever_sensitivity(pipe, base, base_prob, feature_cols, n_steps=5)
        # Should have one row per lever
        assert len(df) == len(LEVERS)
        # Sorted descending by impact
        impacts = df["impact_range"].values
        assert all(impacts[i] >= impacts[i+1] - 1e-9 for i in range(len(impacts)-1))
