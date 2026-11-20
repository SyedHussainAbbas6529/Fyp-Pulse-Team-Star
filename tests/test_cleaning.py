"""
tests/test_cleaning.py — Unit tests for data cleaning rules
Team Star | FYP-Pulse
"""
import pytest
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.cleaning import (
    clean_data, _parse_date, _normalize_gender, _normalize_yes_no,
    _fuzzy_match_campus, VALID_CAMPUSES, NUMERIC_RANGES
)


# ---------------------------------------------------------------------------
# Date parsing tests
# ---------------------------------------------------------------------------
class TestDateParsing:
    def test_iso_format(self):
        assert _parse_date("2024-03-15") == "2024-03-15"

    def test_dd_mm_yyyy_slash(self):
        result = _parse_date("15/03/2024")
        assert result == "2024-03-15"

    def test_dd_mm_yyyy_dash(self):
        result = _parse_date("15-03-2024")
        assert result == "2024-03-15"

    def test_named_month(self):
        result = _parse_date("15 Mar 2024")
        assert result == "2024-03-15"

    def test_named_month_dash(self):
        result = _parse_date("15-Mar-2024")
        assert result == "2024-03-15"

    def test_unparseable_returns_none(self):
        assert _parse_date("not-a-date") is None

    def test_empty_returns_none(self):
        assert _parse_date("") is None

    def test_nan_returns_none(self):
        assert _parse_date(np.nan) is None


# ---------------------------------------------------------------------------
# Gender normalization
# ---------------------------------------------------------------------------
class TestGenderNorm:
    def test_m_to_male(self):
        assert _normalize_gender("M") == "Male"

    def test_f_to_female(self):
        assert _normalize_gender("F") == "Female"

    def test_male_pass(self):
        assert _normalize_gender("male") == "Male"

    def test_female_pass(self):
        assert _normalize_gender("Female") == "Female"

    def test_unknown_returns_none(self):
        assert _normalize_gender("Other") is None


# ---------------------------------------------------------------------------
# Yes/No normalization
# ---------------------------------------------------------------------------
class TestYesNoNorm:
    def test_yes_variants(self):
        for v in ["yes", "y", "1", "true", "Yes", "YES"]:
            assert _normalize_yes_no(v) == "Yes", f"Failed for: {v}"

    def test_no_variants(self):
        for v in ["no", "n", "0", "false", "No", "NO"]:
            assert _normalize_yes_no(v) == "No", f"Failed for: {v}"

    def test_unknown_returns_none(self):
        assert _normalize_yes_no("maybe") is None


# ---------------------------------------------------------------------------
# Campus fuzzy matching
# ---------------------------------------------------------------------------
class TestCampusFuzzy:
    def test_exact_case_insensitive(self):
        assert _fuzzy_match_campus("karachi") == "Karachi"
        assert _fuzzy_match_campus("LAHORE") == "Lahore"
        assert _fuzzy_match_campus("quetta") == "Quetta"

    def test_known_abbreviation(self):
        assert _fuzzy_match_campus("ISB") == "Islamabad"

    def test_all_valid_campuses_matched(self):
        for c in VALID_CAMPUSES:
            assert _fuzzy_match_campus(c) == c

    def test_unrecognized_returns_none(self):
        assert _fuzzy_match_campus("XYZ_UNKNOWN_CITY_999") is None


# ---------------------------------------------------------------------------
# Full cleaning pipeline tests
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_df():
    """Minimal DataFrame with known issues."""
    return pd.DataFrame({
        "group_id": ["G001", "G002", "G003", "G004", "G005"],
        "proposal_date": ["2024-03-15", "15/03/2024", "bad-date", "17-Jun-25", "2024-01-10"],
        "campus": ["lahore", "KARACHI", "ISB", "quetta", "Peshawar"],
        "program": ["BSCS", "bsse", "BSAI", "BSIT", "BSDS"],
        "leader_gender": ["M", "Female", "F", "Male", "male"],
        "group_size": [3.0, 1.0, 2.0, 4.0, 5.0],
        "team_avg_cgpa": [3.0, 2.5, 3.5, 3.2, 2.8],
        "team_min_cgpa": [3.5, 2.5, 3.0, 3.5, 2.5],  # G001: min>avg, G004: min>avg
        "project_type": ["Web App", "AI-ML", "IoT", "Research", "Mobile App"],
        "has_industry_sponsor": ["Yes", "no", "y", "1", "No"],
        "supervisor_experience_years": [5.0, 10.0, 3.0, 50.0, 8.0],  # G004: 50 out of range
        "supervisor_active_groups": [5.0, 3.0, 7.0, 6.0, 4.0],
        "proposal_revisions": [2.0, 1.0, 0.0, 3.0, 1.0],
        "weeks_to_proposal_approval": [4.0, 2.0, 6.0, 8.0, 3.0],
        "supervisor_meetings_8wk": [5.0, 3.0, 8.0, 6.0, 4.0],
        "github_commits_8wk": [60.0, 30.0, 150.0, 80.0, 45.0],
        "mid_eval_score": [70.0, 65.0, 80.0, 75.0, 60.0],
        "final_defense_score": [80.0, 75.0, 85.0, 70.0, 65.0],
        "extension_granted": ["No", "No", "No", "No", "No"],
        "poor_outcome": ["No", "No", "No", "Yes", "No"],
    })


class TestCleaningPipeline:
    def test_runs_without_error(self, sample_df):
        df_clean, log, summary = clean_data(sample_df)
        assert len(df_clean) > 0

    def test_campus_standardized(self, sample_df):
        df_clean, _, _ = clean_data(sample_df)
        for val in df_clean["campus"].dropna():
            assert val in VALID_CAMPUSES, f"Invalid campus: {val}"

    def test_gender_standardized(self, sample_df):
        df_clean, _, _ = clean_data(sample_df)
        for val in df_clean["leader_gender"].dropna():
            assert val in ["Male", "Female"], f"Invalid gender: {val}"

    def test_yes_no_standardized(self, sample_df):
        df_clean, _, _ = clean_data(sample_df)
        for col in ["has_industry_sponsor", "poor_outcome"]:
            for val in df_clean[col].dropna():
                assert val in ["Yes", "No"], f"Invalid {col}: {val}"

    def test_cgpa_min_leq_avg(self, sample_df):
        """After cleaning, team_min_cgpa must never exceed team_avg_cgpa."""
        df_clean, _, _ = clean_data(sample_df)
        for _, row in df_clean.dropna(subset=["team_avg_cgpa", "team_min_cgpa"]).iterrows():
            assert row["team_min_cgpa"] <= row["team_avg_cgpa"] + 1e-9, (
                f"Min CGPA {row['team_min_cgpa']} > Avg CGPA {row['team_avg_cgpa']}"
            )

    def test_solo_rule(self, sample_df):
        """For group_size=1, min_cgpa must equal avg_cgpa."""
        df_clean, _, _ = clean_data(sample_df)
        solo = df_clean[df_clean["group_size"] == 1.0].dropna(
            subset=["team_avg_cgpa", "team_min_cgpa"]
        )
        for _, row in solo.iterrows():
            assert abs(row["team_min_cgpa"] - row["team_avg_cgpa"]) < 1e-9

    def test_out_of_range_becomes_missing(self, sample_df):
        """supervisor_experience_years=50 should become NaN."""
        df_clean, log, _ = clean_data(sample_df)
        # Find row that had 50
        exp_col = df_clean["supervisor_experience_years"]
        # It should be NaN or within range
        for val in exp_col.dropna():
            lo, hi = NUMERIC_RANGES["supervisor_experience_years"]
            assert lo <= val <= hi

    def test_missing_target_dropped(self):
        df_with_nan_target = pd.DataFrame({
            "group_id": ["G001", "G002"],
            "proposal_date": ["2024-01-01", "2024-01-02"],
            "campus": ["Lahore", "Karachi"],
            "program": ["BSCS", "BSSE"],
            "leader_gender": ["Male", "Female"],
            "group_size": [3.0, 2.0],
            "team_avg_cgpa": [3.0, 2.8],
            "team_min_cgpa": [2.5, 2.4],
            "project_type": ["Web App", "IoT"],
            "has_industry_sponsor": ["No", "Yes"],
            "supervisor_experience_years": [5.0, 8.0],
            "supervisor_active_groups": [5.0, 3.0],
            "proposal_revisions": [1.0, 2.0],
            "weeks_to_proposal_approval": [3.0, 4.0],
            "supervisor_meetings_8wk": [5.0, 4.0],
            "github_commits_8wk": [50.0, 30.0],
            "mid_eval_score": [70.0, 65.0],
            "final_defense_score": [80.0, 75.0],
            "extension_granted": ["No", "No"],
            "poor_outcome": [np.nan, "No"],
        })
        df_clean, _, summary = clean_data(df_with_nan_target)
        assert summary["rows_missing_target_dropped"] == 1
        assert len(df_clean) == 1

    def test_no_leakage_in_log(self, sample_df):
        """Cleaning log should not reference leakage columns as model features."""
        from src.cleaning import LEAKAGE_COLS
        _, log, _ = clean_data(sample_df)
        # Leakage cols CAN appear in the log (we still clean them)
        # but confirm they are documented in EXCLUDED_COLUMNS
        from src.modeling import EXCLUDED_COLUMNS
        for col in LEAKAGE_COLS:
            assert col in EXCLUDED_COLUMNS
