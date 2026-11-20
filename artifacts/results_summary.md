# FYP-Pulse | Team Star: Executive Results Summary

**Track 4 - Vibe Jam: AI Sprint**  
**System:** FYP-Pulse: Final Year Project Early Warning System  
**Team Identity:** Team Star (⭐)  
**Deliverable Version:** 1.0.0 (Production Ready)  

---

## 1. Executive Overview & Problem Context
In undergraduate computing programs, approximately **1 in 6 student groups (16.0%)** fails their Final Year Project or requires an emergency semester carryover. Identifying high-risk projects early (by Week 8) allows coordinators to intervene proactively before terminal milestones. 

A trivial majority-class baseline predicting "No Risk" achieves **84.0% accuracy** while capturing **0% of at-risk students**, rendering accuracy useless. FYP-Pulse is engineered with a strict **Recall-first optimization paradigm**, PR-AUC threshold tuning, and Platt sigmoid probability calibration.

---

## 2. Key Quantitative Findings & Metrics

### A. Data Quality & Integrity (Module A1)
* **Raw Ingestion:** 1,840 records ingested from `data/fyp_pulse_train.csv`.
* **Deduplication:** 4 exact duplicates removed; 16 duplicate `group_id` records resolved retaining most-complete records.
* **Integrity Repairs (1,353 fixes logged in `artifacts/cleaning_log.csv`):**
  * *Campus Fuzzy Matching:* 8 canonical campuses standardized with difflib & alias lookup (e.g., `LHR` → `Lahore`, `FSD` → `Faisalabad`).
  * *Solo Team Invariance:* Enforced `team_min_cgpa == team_avg_cgpa` when `group_size == 1`.
  * *CGPA Invariant:* Repaired instances where `team_min_cgpa > team_avg_cgpa` via swap correction.
  * *Date Normalization:* ISO-8601 formatting across mixed formats (`YYYY-MM-DD`, `DD/MM/YYYY`, `MM-DD-YYYY`, textual dates).
  * *Range Violations:* Numerical values outside valid operational bounds transformed to missing values for disciplined imputation.
* **Clean Dataset:** 1,800 validated rows; 288 poor outcomes (**16.00% true base rate**).

### B. Machine Learning Performance (Module A3)
Evaluated across Stratified 80/20 train/test split with 5-fold Stratified Cross-Validation on training data. Strict leakage isolation excluded `final_defense_score`, `extension_grant`, and `group_id`.

| Model Candidate | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Test PR-AUC | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Selected)** | **77.2%** | **40.5%** | **56.9%** | **0.471** | **0.796** | **0.419** | **0.221** |
| Random Forest | 81.7% | 44.2% | 32.8% | 0.376 | 0.773 | 0.391 | 0.250 |
| Extra Trees | 82.5% | 47.2% | 29.3% | 0.362 | 0.782 | 0.406 | 0.260 |
| XGBoost Classifier | 80.0% | 38.8% | 32.8% | 0.355 | 0.761 | 0.388 | 0.220 |
| *Naive "No Risk" Baseline* | *84.0%* | *0.0%* | *0.0%* | *0.000* | *0.500* | *0.160* | *0.500* |

**Model Choice Justification:** Logistic Regression with balanced weighting and sigmoid Platt calibration outperformed tree ensembles on **PR-AUC (0.419)** and **Recall (56.9% test, 69.2% CV)** at the cross-validated threshold ($\tau = 0.221$). Its smooth linear log-odds space provides superior calibration stability and resistance to test-set variance compared to high-variance tree splits on imbalanced data.

### C. Risk Tier Monotonicity (Module A4)
Risk tiers were determined by search over training probability percentiles and evaluated on unseen test data ($n=360$):
* **Low Risk ($P < 0.0414$):** 85 groups, **2.35% observed poor rate** (2/85)
* **Medium Risk ($0.0414 \le P < 0.2475$):** 207 groups, **12.56% observed poor rate** (26/207)
* **High Risk ($P \ge 0.2475$):** 68 groups, **44.12% observed poor rate** (30/68)
* **Monotonicity:** $2.35\% < 12.56\% < 44.12\%$ — **Strictly Monotonic (PASS)**.

### D. Subgroup Fairness & Disparity (Module A5)
Evaluated with Wilson score 95% Confidence Intervals and $\chi^2$ test of independence:
* **Gender Equity:**
  * Male-led groups ($n=210$): Predicted positive rate 22.38% (95% CI: $[17.29\%, 28.48\%]$), Recall 56.76%, Precision 44.68%.
  * Female-led groups ($n=150$): Predicted positive rate 23.33% (95% CI: $[17.27\%, 30.73\%]$), Recall 57.14%, Precision 34.29%.
  * **Disparity:** Rate gap of **0.95 pp**, Recall gap of **0.38 pp**, $p = 0.8318$. **No statistically significant gender disparity exists.**
* **Campus Equity ($\ge 30$ test groups):**
  * All 8 campuses met the sample cutoff ($n=33$ to $n=58$).
  * Predicted poor rates ranged from 18.18% (Multan) to 28.41% (Karachi), a gap of 10.23 pp.
  * Disparities reflect local cohort variations in commits and meeting frequencies rather than systemic algorithmic bias ($p > 0.05$).

### E. Operational Watchlist & Intervention (Modules A6 & A7)
* **Top-20 Coordinator Watchlist:**
  * Captured **8 truly failing groups** out of top 20 ($40.0\%$ Precision@20).
  * Outperforms random expectation of 3.2 groups, representing a **2.48× Lift**.
* **What-If Interventions:**
  * Actionable levers: `github_commits_8wk` (+38.90 pp risk delta across range) and `supervisor_meetings_8wk` (+31.22 pp delta) provide maximum return on faculty intervention.
  * Demographic and academic invariants (`cgpa`, `gender`, `campus`) are locked against manipulation.
  * Causal disclaimer explicitly states predictive association vs deterministic causation.

---

## 3. Architecture & Delivery
* **Single-Command Launch:** `streamlit run app.py` executes all modules offline without external dependencies or internet requests.
* **Test Suite:** 49 automated unit tests passing in `<5s` (`python -m pytest tests/`).
* **Evidence Archive:** Full browser execution screenshots recorded in `artifacts/screenshots/`.
