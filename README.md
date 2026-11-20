# FYP-Pulse | Team Star ⭐
## Final Year Project Early Warning System

> **Vibe Jam AI Sprint 2026 — Track 4** | Built by **Team Star**

---

## 🚀 Quick Start

```bash
cd star-fyp-pulse
pip install -r requirements.txt
streamlit run app.py
```

The app auto-detects `data/fyp_pulse_train.csv`. All 8 pages load in one browser tab.

---

## 📂 Project Structure

```
star-fyp-pulse/
├── app.py                    # Main Streamlit app (all 8 pages)
├── requirements.txt          # Pinned Python dependencies
├── README.md                 # This file
├── data/
│   └── fyp_pulse_train.csv   # Raw dataset (1,840 rows)
├── artifacts/
│   ├── model_pipeline.joblib # Saved best model pipeline
│   ├── cleaning_log.csv      # Auto-generated on first run
│   ├── cleaned_data.csv      # Auto-generated on first run
│   └── tier_thresholds.json  # Tier cut-offs (computed on train)
├── src/
│   ├── cleaning.py           # Rule-based data cleaner with audit log
│   ├── features.py           # Feature engineering (10 engineered features)
│   ├── modeling.py           # 4 models, CV, threshold selection, calibration
│   ├── explain.py            # Custom baseline-replacement explainer
│   ├── fairness.py           # Gender + campus fairness with Wilson CI
│   ├── watchlist.py          # Top-K watchlist with lift metrics
│   ├── whatif.py             # What-If lever simulator
│   └── ui.py                 # Shared UI components and CSS
└── tests/
    ├── test_cleaning.py      # 20+ cleaning rule tests
    ├── test_modeling.py      # Leakage, threshold, tier monotonicity tests
    ├── test_watchlist.py     # Watchlist correctness tests
    └── test_whatif.py        # Lever bounds and sensitivity tests
```

---

## 🔬 Methodology

### Data Cleaning (A1)
- **Campus:** Fuzzy matching via `difflib` to 8 valid names; known abbreviations (ISB→Islamabad)
- **Gender:** M/F → Male/Female
- **Yes/No:** Normalizes y/yes/1/true/Y → "Yes"
- **Dates:** Tries ISO, DD/MM/YYYY (preferred), MM/DD/YYYY, named-month formats
- **Ranges:** Out-of-range values → NaN (error, not outlier); logged per column
- **CGPA rules:** min > avg → values swapped; solo group → min set equal to avg
- **Duplicates:** Exact duplicates removed; group_id duplicates keep the most complete row

### Leakage Control (Critical)
Three columns are **never** used as features:
| Column | Reason |
|--------|--------|
| `final_defense_score` | Future information — not known at week 8 |
| `extension_granted` | Consequence of outcome — target leakage |
| `group_id` | Identifier with no predictive meaning |

### Feature Engineering (A2)
10 week-8-safe engineered features:
- `commits_per_meeting`, `commits_per_week`, `meetings_per_week`
- `supervisor_load_ratio` = active_groups ÷ experience_years
- `cgpa_gap` = avg_cgpa − min_cgpa
- `low_engagement_flag` = commits/week < 5 AND meetings/week < 1
- `mid_eval_vs_cgpa_mismatch` = mid_eval − (avg_cgpa × 25)
- `approval_delay_flag` = weeks_to_approval > 10
- `proposal_month` = month from proposal_date
- `sponsor_x_type` = sponsor=Yes AND type=AI-ML

### Imbalance Handling
~16% positive class rate. All models use `class_weight="balanced"` or `scale_pos_weight`. Models are scored on **PR-AUC** (not accuracy) to reward minority-class detection.

### Model Selection (A3)
- 4 models: Logistic Regression, Random Forest, Extra Trees, XGBoost
- Stratified 80/20 test split + 5-fold stratified CV
- `RandomizedSearchCV` on PR-AUC
- **Threshold selection:** maximises F1 on CV predictions with recall ≥ 65%
- **Calibration:** `CalibratedClassifierCV(method="sigmoid", cv=5)` on training data

### Tier Cut-offs (A4)
Computed on **train-only CV predictions** via percentile grid search. Verified on test set: observed poor-outcome rates must be strictly Low < Medium < High.

### Explainability (A5)
Custom baseline-replacement method (no C++ compiler required):
- For each feature, replace it with the training median/mode
- Measure change in predicted probability
- Positive = feature raises risk above baseline; Negative = lowers risk

### Fairness (A5)
- Gender (Male vs Female): chi-square significance test + Wilson 95% CI
- Campus (≥30 test groups): largest rate and recall gaps reported
- Confidence interval interpretation included

### Watchlist (A6)
Top-K groups (K=10–100) sorted by predicted probability on test set.
Metrics: Precision@K, Recall@K, Lift, Random Expectation = K × base rate.

### What-If (A7)
Levers: `supervisor_meetings_8wk`, `github_commits_8wk`, `supervisor_active_groups`, `has_industry_sponsor`.
Locked: CGPA, campus, program, gender, supervisor experience.
Lever ranking sweeps each lever across its full valid range.

---

## ⚠️ Fairness Findings
(Computed at runtime on the actual test set — see the Fairness page for live numbers.)

---

## 📋 Acceptance Checklist

- [x] A1: Rule table, summary, cleaned data, solo and min≤avg rules enforced
- [x] A2: 8 charts with insights (team, supervisor, project context, engineered feature, campus, program, scatter, heatmap)
- [x] A3: 4 models with all metrics, confusion matrix, threshold, justified final model, no leakage
- [x] A4: Exact probability (4dp), colour tier badge, observed rates Low < Med < High on test, gauge
- [x] A5: Top 3 drivers with arrows/sizes, plain explanation, one concrete step, gender + campus fairness, largest gap, CI + takeaway
- [x] A6: K slider 10–100, sorted list, CSV download, Precision@K, captured count, random expectation, lift
- [x] A7: Live what-if, before/after probabilities, lever ranking chart, association note, locked features
- [x] Offline, one command, no console errors

---

## 🛠️ Run Tests

```bash
cd star-fyp-pulse
python -m pytest tests/ -v
```

---

Built by **Team Star** ⭐ | FYP-Pulse Early Warning System | Vibe Jam AI Sprint 2026
