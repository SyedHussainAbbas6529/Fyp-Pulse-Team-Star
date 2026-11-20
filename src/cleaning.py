"""
src/cleaning.py — FYP-Pulse Data Cleaning Pipeline
Team Star | FYP-Pulse | Early Warning System

Rule-based cleaner with full audit log. Operates on raw CSV, returns
(cleaned_df, log_df, summary_dict). All fixes are deterministic (no randomness).
Statistical imputation is NOT done here — it lives inside the sklearn Pipeline
fitted only on training data to prevent data leakage.
"""
from __future__ import annotations

import re
import difflib
import logging
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid domain values
# ---------------------------------------------------------------------------
VALID_CAMPUSES = [
    "Lahore", "Karachi", "Islamabad", "Faisalabad",
    "Multan", "Rawalpindi", "Peshawar", "Quetta"
]
VALID_PROGRAMS = ["BSCS", "BSSE", "BSAI", "BSIT", "BSDS"]
VALID_PROJECT_TYPES = ["Web App", "Mobile App", "AI-ML", "IoT", "Research"]
VALID_GENDERS = ["Male", "Female"]
VALID_YES_NO = ["Yes", "No"]

# Numeric column valid ranges: (min_inclusive, max_inclusive)
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "group_size": (1, 5),
    "team_avg_cgpa": (0.0, 4.0),
    "team_min_cgpa": (0.0, 4.0),
    "supervisor_experience_years": (0, 45),
    "supervisor_active_groups": (1, 15),
    "proposal_revisions": (0, 10),
    "weeks_to_proposal_approval": (0, 30),
    "supervisor_meetings_8wk": (0, 30),
    "github_commits_8wk": (0, 1000),
    "mid_eval_score": (0, 100),
    "final_defense_score": (0, 100),
}

# Columns that must be integer-valued
INTEGER_COLS = [
    "group_size", "supervisor_active_groups", "proposal_revisions",
    "weeks_to_proposal_approval", "supervisor_meetings_8wk", "github_commits_8wk",
]

# Leakage columns (never use as features)
LEAKAGE_COLS = ["final_defense_score", "extension_granted", "group_id"]

# Ambiguity note for date parsing
DATE_AMBIGUITY_NOTE = (
    "Dates with ambiguous day/month (e.g. '05/06/2024') are interpreted as "
    "DD/MM/YYYY first. If that fails (day>12), MM/DD/YYYY is attempted. "
    "Completely unparseable dates become NaT (missing)."
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log(records: list, row_idx, col: str, original, corrected, rule: str) -> None:
    records.append({
        "row_index": row_idx,
        "column": col,
        "original_value": original,
        "corrected_value": corrected,
        "rule_applied": rule,
    })


def _standardize_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip whitespace, replace spaces with underscores."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Rename extension_grant → extension_granted if needed
    if "extension_grant" in df.columns and "extension_granted" not in df.columns:
        df = df.rename(columns={"extension_grant": "extension_granted"})
    return df


def _fuzzy_match_campus(raw: str) -> Optional[str]:
    """Return best matching campus name or None if score < 0.6."""
    raw_stripped = raw.strip()
    # Direct case-insensitive match
    for v in VALID_CAMPUSES:
        if v.lower() == raw_stripped.lower():
            return v
    # Known abbreviations
    abbrevs = {"ISB": "Islamabad", "LHR": "Lahore", "KHI": "Karachi",
                "PEW": "Peshawar", "RWP": "Rawalpindi", "MUL": "Multan",
                "FSD": "Faisalabad", "QTA": "Quetta"}
    if raw_stripped.upper() in abbrevs:
        return abbrevs[raw_stripped.upper()]
    # difflib
    matches = difflib.get_close_matches(raw_stripped, VALID_CAMPUSES, n=1, cutoff=0.6)
    return matches[0] if matches else None


def _parse_date(val: str) -> Optional[str]:
    """
    Try multiple date formats, return ISO YYYY-MM-DD or None.
    Ambiguity rule: DD/MM/YYYY preferred; fall back to MM/DD/YYYY if day>12.
    """
    if pd.isna(val) or str(val).strip() == "":
        return None
    s = str(val).strip()

    # ISO 8601
    try:
        return pd.to_datetime(s, format="%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception:
        pass

    # DD/MM/YYYY or MM/DD/YYYY
    for sep in ["/", "-"]:
        parts = s.split(sep)
        if len(parts) == 3 and not parts[0].isalpha():
            try:
                # Try DD/MM/YYYY first
                d = pd.to_datetime(s, format=f"%d{sep}%m{sep}%Y")
                return d.strftime("%Y-%m-%d")
            except Exception:
                try:
                    d = pd.to_datetime(s, format=f"%m{sep}%d{sep}%Y")
                    return d.strftime("%Y-%m-%d")
                except Exception:
                    pass

    # Named month formats: "12 Mar 2024", "Mar 12 2024"
    for fmt in ["%d %b %Y", "%d-%b-%Y", "%b %d %Y", "%d %B %Y", "%d-%B-%Y"]:
        try:
            return pd.to_datetime(s, format=fmt).strftime("%Y-%m-%d")
        except Exception:
            pass

    # pandas flexible parse as last resort
    try:
        return pd.to_datetime(s, dayfirst=True).strftime("%Y-%m-%d")
    except Exception:
        return None


def _normalize_yes_no(val) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in ("yes", "y", "1", "true"):
        return "Yes"
    if s in ("no", "n", "0", "false"):
        return "No"
    return None


def _normalize_gender(val) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in ("male", "m"):
        return "Male"
    if s in ("female", "f"):
        return "Female"
    return None


def _safe_numeric(val):
    """Strip non-numeric chars (commas, units) and convert to float."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return float(s)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------

def clean_data(
    df_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Clean the raw FYP-Pulse dataframe.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned dataframe (no imputation — that happens in the ML pipeline).
    log_df : pd.DataFrame
        One row per fix applied.
    summary : dict
        High-level counts for the Data Quality page.
    """
    log_records: list[dict] = []
    df = _standardize_col_names(df_raw.copy())

    initial_rows = len(df)
    summary: dict = {
        "initial_rows": initial_rows,
        "date_ambiguity_note": DATE_AMBIGUITY_NOTE,
        "leakage_columns": LEAKAGE_COLS,
    }

    # --- 0. String whitespace trim on all object columns ---
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].str.strip()

    # === 1. Exact duplicate rows ===
    n_exact_dups = df.duplicated().sum()
    df = df.drop_duplicates()
    summary["exact_duplicate_rows_removed"] = int(n_exact_dups)

    # === 2. Duplicate group_id — keep most complete (fewest NaN) ===
    n_gid_dups = 0
    if "group_id" in df.columns:
        dup_mask = df.duplicated(subset=["group_id"], keep=False)
        dup_groups = df[dup_mask]["group_id"].unique()
        keep_indices = []
        drop_indices = []
        for gid in dup_groups:
            rows = df[df["group_id"] == gid]
            best = rows.isnull().sum(axis=1).idxmin()
            drop_indices.extend([i for i in rows.index if i != best])
            n_gid_dups += len(rows) - 1
        df = df.drop(index=drop_indices)
    summary["group_id_duplicate_rows_removed"] = int(n_gid_dups)

    df = df.reset_index(drop=True)

    # === 3. Campus standardization ===
    campus_corrections = 0
    if "campus" in df.columns:
        for i, val in df["campus"].items():
            if pd.isna(val):
                continue
            matched = _fuzzy_match_campus(str(val))
            if matched is None:
                _log(log_records, i, "campus", val, None,
                     "Campus unrecognized — set to missing")
                df.at[i, "campus"] = np.nan
                campus_corrections += 1
            elif matched != str(val).strip():
                _log(log_records, i, "campus", val, matched,
                     "Campus standardized via fuzzy match / case correction")
                df.at[i, "campus"] = matched
                campus_corrections += 1
    summary["campus_corrections"] = campus_corrections

    # === 4. Program standardization ===
    program_corrections = 0
    if "program" in df.columns:
        prog_map = {p.lower(): p for p in VALID_PROGRAMS}
        for i, val in df["program"].items():
            if pd.isna(val):
                continue
            norm = str(val).strip().upper()
            if norm in VALID_PROGRAMS:
                if norm != str(val).strip():
                    _log(log_records, i, "program", val, norm, "Program case normalized")
                    df.at[i, "program"] = norm
                    program_corrections += 1
            else:
                _log(log_records, i, "program", val, None, "Program unrecognized — set to missing")
                df.at[i, "program"] = np.nan
                program_corrections += 1
    summary["program_corrections"] = program_corrections

    # === 5. Project type standardization ===
    proj_corrections = 0
    if "project_type" in df.columns:
        proj_map = {p.lower().replace(" ", "").replace("-", ""): p for p in VALID_PROJECT_TYPES}
        for i, val in df["project_type"].items():
            if pd.isna(val):
                continue
            key = str(val).strip().lower().replace(" ", "").replace("-", "")
            if key in proj_map:
                canonical = proj_map[key]
                if canonical != str(val).strip():
                    _log(log_records, i, "project_type", val, canonical,
                         "project_type standardized")
                    df.at[i, "project_type"] = canonical
                    proj_corrections += 1
            else:
                _log(log_records, i, "project_type", val, None,
                     "project_type unrecognized — set to missing")
                df.at[i, "project_type"] = np.nan
                proj_corrections += 1
    summary["project_type_corrections"] = proj_corrections

    # === 6. Gender normalization ===
    gender_corrections = 0
    if "leader_gender" in df.columns:
        for i, val in df["leader_gender"].items():
            norm = _normalize_gender(val)
            if norm is None and not pd.isna(val):
                _log(log_records, i, "leader_gender", val, None,
                     "leader_gender unrecognized — set to missing")
                df.at[i, "leader_gender"] = np.nan
                gender_corrections += 1
            elif norm is not None and norm != str(val).strip():
                _log(log_records, i, "leader_gender", val, norm,
                     "leader_gender standardized (M/F → Male/Female)")
                df.at[i, "leader_gender"] = norm
                gender_corrections += 1
    summary["gender_corrections"] = gender_corrections

    # === 7. Yes/No columns ===
    yn_cols = ["has_industry_sponsor", "poor_outcome"]
    # extension_granted is leakage but still clean it
    if "extension_granted" in df.columns:
        yn_cols.append("extension_granted")
    yn_corrections = 0
    for col in yn_cols:
        if col not in df.columns:
            continue
        for i, val in df[col].items():
            norm = _normalize_yes_no(val)
            if norm is None and not pd.isna(val):
                _log(log_records, i, col, val, None,
                     f"{col}: unrecognized value — set to missing")
                df.at[i, col] = np.nan
                yn_corrections += 1
            elif norm is not None and norm != str(val).strip():
                _log(log_records, i, col, val, norm,
                     f"{col}: Yes/No normalized")
                df.at[i, col] = norm
                yn_corrections += 1
    summary["yes_no_corrections"] = yn_corrections

    # === 8. Date parsing ===
    date_errors = 0
    if "proposal_date" in df.columns:
        for i, val in df["proposal_date"].items():
            iso = _parse_date(val)
            if iso != str(val).strip():
                if iso is None:
                    _log(log_records, i, "proposal_date", val, None,
                         "proposal_date unparseable — set to missing")
                    df.at[i, "proposal_date"] = np.nan
                    date_errors += 1
                else:
                    _log(log_records, i, "proposal_date", val, iso,
                         "proposal_date standardized to ISO")
                    df.at[i, "proposal_date"] = iso
    summary["date_parse_errors"] = date_errors

    # === 9. Numeric conversion + range validation ===
    range_errors_per_col: dict[str, int] = {}
    for col, (lo, hi) in NUMERIC_RANGES.items():
        if col not in df.columns:
            continue
        errs = 0
        # Safe conversion
        df[col] = df[col].apply(_safe_numeric)
        for i, val in df[col].items():
            if pd.isna(val):
                continue
            if val < lo or val > hi:
                _log(log_records, i, col, val, None,
                     f"{col} out of range [{lo},{hi}] — set to missing (error, not outlier)")
                df.at[i, col] = np.nan
                errs += 1
        if errs:
            range_errors_per_col[col] = errs
    summary["range_errors_per_column"] = range_errors_per_col
    summary["total_range_errors"] = sum(range_errors_per_col.values())

    # === 10. Integer rounding for count columns ===
    for col in INTEGER_COLS:
        if col not in df.columns:
            continue
        for i, val in df[col].items():
            if pd.isna(val):
                continue
            rounded = int(round(float(val)))
            if rounded != val:
                _log(log_records, i, col, val, rounded,
                     f"{col}: non-integer count rounded")
                df.at[i, col] = float(rounded)

    # === 11. CGPA rules ===
    cgpa_mingtavg = 0
    cgpa_solo = 0
    if "team_avg_cgpa" in df.columns and "team_min_cgpa" in df.columns:
        for i, row in df.iterrows():
            avg = row.get("team_avg_cgpa")
            mn = row.get("team_min_cgpa")
            gs = row.get("group_size")

            if pd.isna(avg) or pd.isna(mn):
                continue

            # Solo rule: group_size == 1 → min must equal avg
            if not pd.isna(gs) and int(round(float(gs))) == 1:
                if mn != avg:
                    _log(log_records, i, "team_min_cgpa", mn, avg,
                         "Solo group (size=1): team_min_cgpa set equal to team_avg_cgpa")
                    df.at[i, "team_min_cgpa"] = avg
                    cgpa_solo += 1
                    mn = avg

            # min > avg is an error — swap them
            if mn > avg:
                _log(log_records, i, "team_min_cgpa", mn, avg,
                     "CGPA error: team_min_cgpa > team_avg_cgpa — values swapped")
                _log(log_records, i, "team_avg_cgpa", avg, mn,
                     "CGPA error: team_avg_cgpa adjusted (swapped with min)")
                df.at[i, "team_min_cgpa"] = avg
                df.at[i, "team_avg_cgpa"] = mn
                cgpa_mingtavg += 1

    summary["cgpa_min_gt_avg_repaired"] = cgpa_mingtavg
    summary["cgpa_solo_rule_applied"] = cgpa_solo

    # === 12. Drop rows with missing target ===
    if "poor_outcome" in df.columns:
        missing_target = df["poor_outcome"].isna().sum()
        df = df[df["poor_outcome"].notna()].reset_index(drop=True)
        summary["rows_missing_target_dropped"] = int(missing_target)
    else:
        summary["rows_missing_target_dropped"] = 0

    cleaned_rows = len(df)
    summary["cleaned_rows"] = cleaned_rows
    summary["rows_removed_total"] = initial_rows - cleaned_rows

    # Poor outcome stats
    if "poor_outcome" in df.columns:
        poor_count = (df["poor_outcome"] == "Yes").sum()
        summary["total_poor_outcomes"] = int(poor_count)
        summary["poor_outcome_rate"] = round(poor_count / cleaned_rows, 4) if cleaned_rows else 0
    else:
        summary["total_poor_outcomes"] = 0
        summary["poor_outcome_rate"] = 0

    log_df = pd.DataFrame(log_records, columns=[
        "row_index", "column", "original_value", "corrected_value", "rule_applied"
    ])

    return df, log_df, summary
