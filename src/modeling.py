"""
src/modeling.py — Model Training and Selection for FYP-Pulse
Team Star | FYP-Pulse | Early Warning System

Trains 4 models with stratified CV, selects best on PR-AUC/recall,
calibrates probabilities, and saves the pipeline artifact.

NO DATA LEAKAGE: final_defense_score, extension_granted, group_id are excluded.
Imputation is fitted on TRAIN data only inside the Pipeline.
Threshold is chosen on TRAIN/CV predictions only.
"""
from __future__ import annotations

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from typing import Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    StratifiedKFold, train_test_split, RandomizedSearchCV, cross_val_predict
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from src.features import (
    engineer_features, NUMERIC_BASE, NUMERIC_ENGINEERED,
    CATEGORICAL_FEATURES, TARGET_COL, LEAKAGE_COLS
)

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, "model_pipeline.joblib")
THRESHOLDS_PATH = os.path.join(ARTIFACTS_DIR, "tier_thresholds.json")

# ---------------------------------------------------------------------------
# Leakage exclusion documentation
# ---------------------------------------------------------------------------
EXCLUDED_COLUMNS = {
    "final_defense_score": (
        "Known ONLY after the final defense (future information at week 8). "
        "Using it would be data leakage."
    ),
    "extension_granted": (
        "Decided as a consequence of the outcome — it IS part of the outcome, "
        "not a predictor. Classic target leakage."
    ),
    "group_id": (
        "Identifier with no predictive meaning. Models must generalize to new groups."
    ),
}


def _build_preprocessor(numeric_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """
    Build sklearn ColumnTransformer:
    - Numeric: SimpleImputer(median) + StandardScaler + missing-indicator flags
    - Categorical: SimpleImputer(most_frequent) + OrdinalEncoder(unknown='use_encoded_value')
    Fitted on TRAIN data only.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, cat_cols),
    ], remainder="drop")


def prepare_data(df_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Engineer features and return X, y. Drops leakage cols."""
    df = engineer_features(df_clean)
    # Drop leakage
    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns], errors="ignore")
    # Target
    y = (df[TARGET_COL] == "Yes").astype(int)
    df = df.drop(columns=[TARGET_COL], errors="ignore")
    # Also drop proposal_date (non-numeric, replaced by proposal_month)
    df = df.drop(columns=["proposal_date"], errors="ignore")
    return df, y


def _get_model_definitions(pos_ratio: float) -> dict:
    """Return model name → (estimator, param_grid) mapping."""
    scale_pos = max(1.0, (1 - pos_ratio) / pos_ratio) if pos_ratio > 0 else 1.0
    cw = "balanced"

    models = {
        "Logistic Regression": (
            LogisticRegression(
                class_weight=cw, random_state=RANDOM_SEED,
                max_iter=1000, solver="lbfgs"
            ),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(
                class_weight=cw, random_state=RANDOM_SEED, n_jobs=-1
            ),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 6, 10],
                "clf__min_samples_leaf": [1, 2, 4],
            },
        ),
        "Extra Trees": (
            ExtraTreesClassifier(
                class_weight=cw, random_state=RANDOM_SEED, n_jobs=-1
            ),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 8, 12],
                "clf__min_samples_leaf": [1, 2],
            },
        ),
    }

    # XGBoost preferred, else LightGBM, else HistGradientBoosting
    if HAS_XGB:
        models["XGBoost"] = (
            xgb.XGBClassifier(
                scale_pos_weight=scale_pos, random_state=RANDOM_SEED,
                eval_metric="logloss", verbosity=0,
            ),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 5, 7],
                "clf__learning_rate": [0.05, 0.1, 0.2],
                "clf__subsample": [0.7, 1.0],
                "clf__colsample_bytree": [0.7, 1.0],
            },
        )
    elif HAS_LGB:
        models["LightGBM"] = (
            lgb.LGBMClassifier(
                class_weight=cw, random_state=RANDOM_SEED, verbose=-1
            ),
            {
                "clf__n_estimators": [100, 200],
                "clf__num_leaves": [31, 63],
                "clf__learning_rate": [0.05, 0.1, 0.2],
            },
        )
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        models["Gradient Boosting"] = (
            HistGradientBoostingClassifier(
                class_weight=cw, random_state=RANDOM_SEED
            ),
            {
                "clf__max_iter": [100, 200],
                "clf__max_depth": [3, 5, None],
                "clf__learning_rate": [0.05, 0.1, 0.2],
            },
        )

    return models


def _select_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                      min_recall: float = 0.65) -> float:
    """
    Choose threshold on CV predictions to maximize F1 while keeping recall >= min_recall.
    Falls back to 0.3 if no threshold meets the recall constraint.
    """
    prec, rec, thresh = precision_recall_curve(y_true, y_prob)
    # thresh has len = len(prec) - 1
    best_f1 = -1
    best_t = 0.3
    for p, r, t in zip(prec[:-1], rec[:-1], thresh):
        if r >= min_recall:
            f1 = 2 * p * r / (p + r + 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
    return float(best_t)


def _evaluate(name: str, pipeline: Pipeline, X_test: pd.DataFrame,
              y_test: np.ndarray, threshold: float) -> dict:
    """Compute all test-set metrics for one model."""
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    return {
        "model_name": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "y_prob": y_prob.tolist(),
        "y_pred": y_pred.tolist(),
        "pr_curve": {"precision": prec_arr.tolist(), "recall": rec_arr.tolist()},
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }


def train_all_models(
    df_clean: pd.DataFrame,
) -> dict:
    """
    Full training pipeline. Returns a results dict with:
    - 'models': per-model metrics
    - 'best_model_name': chosen model name
    - 'final_pipeline': fitted sklearn Pipeline
    - 'X_test', 'y_test': for downstream pages
    - 'tier_thresholds': {low_max, high_min}
    - 'feature_names': list of feature names fed to the model
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    X, y = prepare_data(df_clean)

    # Feature columns actually present
    num_cols = [c for c in NUMERIC_BASE + NUMERIC_ENGINEERED if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    pos_ratio = y.mean()

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    model_defs = _get_model_definitions(pos_ratio)

    results: dict = {
        "models": {},
        "feature_names": num_cols + cat_cols,
        "pos_ratio": float(pos_ratio),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "excluded_columns": EXCLUDED_COLUMNS,
    }

    best_score = -1
    best_name = None
    best_pipeline = None
    best_threshold = 0.3

    for name, (estimator, param_grid) in model_defs.items():
        preprocessor = _build_preprocessor(num_cols, cat_cols)
        pipe = Pipeline([("pre", preprocessor), ("clf", estimator)])

        # RandomizedSearchCV on PR-AUC
        search = RandomizedSearchCV(
            pipe, param_grid,
            n_iter=min(20, len(param_grid) * 4),
            scoring="average_precision",
            cv=cv, random_state=RANDOM_SEED, n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_

        # Calibrate using sigmoid (Platt scaling) cross-validation on training data
        calibrated = CalibratedClassifierCV(best_pipe, method="sigmoid", cv=5)
        calibrated.fit(X_train, y_train)

        # Calibrated training probabilities for threshold selection
        train_probs = calibrated.predict_proba(X_train)[:, 1]
        threshold = _select_threshold(y_train.values, train_probs, min_recall=0.65)

        train_preds = (train_probs >= threshold).astype(int)
        cv_metrics = {
            "cv_pr_auc_mean": float(search.best_score_),
            "cv_pr_auc_std": float(
                search.cv_results_["std_test_score"][search.best_index_]
            ),
            "cv_recall": float(recall_score(y_train, train_preds, zero_division=0)),
            "cv_f1": float(f1_score(y_train, train_preds, zero_division=0)),
        }

        # Test metrics evaluated on calibrated model using train-selected threshold
        test_metrics = _evaluate(name, calibrated, X_test, y_test, threshold)
        test_metrics.update(cv_metrics)
        test_metrics["best_params"] = search.best_params_

        results["models"][name] = test_metrics

        # Selection criterion: best CV PR-AUC
        sel_score = search.best_score_
        if sel_score > best_score:
            best_score = sel_score
            best_name = name
            best_pipeline = calibrated
            best_threshold = threshold

    results["best_model_name"] = best_name

    # Compute tier thresholds on TRAIN predictions from best pipeline
    train_probs = best_pipeline.predict_proba(X_train)[:, 1]
    tiers = _compute_tier_thresholds(train_probs, y_train.values)
    results["tier_thresholds"] = tiers

    # Store test data for downstream pages
    results["X_test"] = X_test
    results["y_test"] = y_test
    results["X_train"] = X_train
    results["y_train"] = y_train
    results["final_pipeline"] = best_pipeline
    results["best_threshold"] = best_threshold

    # Save artifacts
    joblib.dump(best_pipeline, PIPELINE_PATH)
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump({"tier_thresholds": tiers, "threshold": best_threshold}, f)

    return results


def _compute_tier_thresholds(
    train_probs: np.ndarray, y_train: np.ndarray
) -> dict[str, float]:
    """
    Choose tier cut-offs on TRAIN predictions so that:
    Low < Medium < High observed poor-outcome rates (monotone).
    Uses percentile-based splits refined to satisfy monotonicity.
    Returns {low_max, high_min}.
    """
    # Start with tertile splits
    p33 = float(np.percentile(train_probs, 33))
    p67 = float(np.percentile(train_probs, 67))

    # Check monotonicity; if not satisfied, nudge thresholds
    def rates(lo, hi):
        low_m = train_probs < lo
        high_m = train_probs >= hi
        mid_m = ~low_m & ~high_m
        r_low = y_train[low_m].mean() if low_m.sum() > 0 else 0.0
        r_mid = y_train[mid_m].mean() if mid_m.sum() > 0 else 0.0
        r_hi = y_train[high_m].mean() if high_m.sum() > 0 else 0.0
        return r_low, r_mid, r_hi, low_m.sum(), mid_m.sum(), high_m.sum()

    # Grid search to find best thresholds satisfying monotonicity
    best_low, best_high = p33, p67
    best_spread = -1
    for lo_p in range(20, 50, 5):
        for hi_p in range(55, 85, 5):
            lo = float(np.percentile(train_probs, lo_p))
            hi = float(np.percentile(train_probs, hi_p))
            if lo >= hi:
                continue
            r_l, r_m, r_h, n_l, n_m, n_h = rates(lo, hi)
            if r_l < r_m < r_h and min(n_l, n_m, n_h) >= 5:
                spread = r_h - r_l
                if spread > best_spread:
                    best_spread = spread
                    best_low, best_high = lo, hi

    return {"low_max": round(best_low, 4), "high_min": round(best_high, 4)}


def load_pipeline() -> Optional[Pipeline]:
    """Load saved pipeline if it exists."""
    if os.path.exists(PIPELINE_PATH):
        return joblib.load(PIPELINE_PATH)
    return None


def predict_group(pipeline, row_df: pd.DataFrame) -> tuple[float, str]:
    """
    Predict for a single group row DataFrame.
    Returns (probability, tier).
    """
    prob = float(pipeline.predict_proba(row_df)[:, 1][0])
    if os.path.exists(THRESHOLDS_PATH):
        with open(THRESHOLDS_PATH) as f:
            meta = json.load(f)
        tiers = meta.get("tier_thresholds", {})
    else:
        tiers = {"low_max": 0.35, "high_min": 0.60}
    tier = "Low" if prob < tiers["low_max"] else ("High" if prob >= tiers["high_min"] else "Medium")
    return prob, tier
