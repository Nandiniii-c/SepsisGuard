"""
train.py
--------
Trains 3 ML models × 3 prediction horizons = 9 models total.

Models:
  1. Logistic Regression  (interpretable baseline, balanced class weights)
  2. Random Forest        (non-linear tree baseline, balanced class weights)
  3. XGBoost              (primary high-performance model, scale_pos_weight)

For each horizon (3h, 6h, 12h):
  - Feature engineering is applied to train/val data
  - Models are trained on training set only
  - Threshold is selected on validation set (maximises F1)
  - Model + scaler + threshold + metadata are saved to models/

Run:
    python src/train.py
    python src/train.py --horizon 6   # train only 6h models
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import engineer_features, load_feature_list
from utils import (
    HORIZONS,
    LR_DIR,
    METRICS_DIR,
    MODEL_DIR,
    PREDS_DIR,
    PROCESSED_DIR,
    RF_DIR,
    XGB_DIR,
    RANDOM_STATE,
    ensure_dir,
    logger,
    save_json,
    set_seeds,
)

set_seeds(RANDOM_STATE)

# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                        metric: str = "f1") -> tuple[float, float]:
    """
    Search over [0.01, 0.99] in steps of 0.01 to find the threshold that
    maximises the chosen metric on the validation set.

    Returns
    -------
    best_threshold : float
    best_score     : float
    """
    thresholds = np.arange(0.01, 0.99, 0.01)
    best_thresh, best_score = 0.5, 0.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        if score > best_score:
            best_score = score
            best_thresh = t
    return float(best_thresh), float(best_score)


def quick_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                  threshold: float) -> dict:
    """Compute AUROC, AUPRC and F1 for a quick progress log."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auroc":  round(float(roc_auc_score(y_true, y_prob)), 4),
        "auprc":  round(float(average_precision_score(y_true, y_prob)), 4),
        "f1":     round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "threshold": round(threshold, 3),
    }


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_logistic_regression(scale_pos: float) -> Pipeline:
    """
    LogisticRegression inside a StandardScaler pipeline.
    class_weight='balanced' handles severe class imbalance.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            C=0.1,
            random_state=RANDOM_STATE,
        )),
    ])


def build_random_forest(scale_pos: float) -> RandomForestClassifier:
    """
    Random Forest with balanced class weights.
    n_jobs=-1 uses all available cores.
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def build_xgboost(scale_pos: float) -> XGBClassifier:
    """
    XGBoost with scale_pos_weight to handle class imbalance.
    scale_pos_weight = n_negative / n_positive.
    """
    return XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_horizon(horizon: int, train_df: pd.DataFrame, val_df: pd.DataFrame,
                  feature_cols: list) -> None:
    """
    Train all three models for a single prediction horizon.

    Parameters
    ----------
    horizon     : int — 3, 6, or 12
    train_df    : DataFrame with engineered features (training set)
    val_df      : DataFrame with engineered features (validation set)
    feature_cols: list of feature column names
    """
    label_col = f"label_{horizon}h"
    logger.info(f"\n{'='*60}")
    logger.info(f"Horizon: {horizon}h  —  target: {label_col}")
    logger.info(f"{'='*60}")

    X_train = train_df[feature_cols].values
    y_train = train_df[label_col].values
    X_val   = val_df[feature_cols].values
    y_val   = val_df[label_col].values

    n_neg   = int((y_train == 0).sum())
    n_pos   = int((y_train == 1).sum())
    scale_pos = n_neg / max(n_pos, 1)

    logger.info(f"  Train: {len(y_train):,} rows | "
                f"pos={n_pos:,} ({n_pos/len(y_train)*100:.3f}%) | "
                f"scale_pos_weight={scale_pos:.1f}")

    # Models to train in order
    models_cfg = [
        ("logistic_regression", build_logistic_regression(scale_pos), LR_DIR),
        ("random_forest",       build_random_forest(scale_pos),        RF_DIR),
        ("xgboost",             build_xgboost(scale_pos),              XGB_DIR),
    ]

    for model_name, model, save_dir in models_cfg:
        logger.info(f"\n  Training {model_name} ({horizon}h)...")
        t0 = time.time()

        # --- Train ---
        if model_name == "xgboost":
            # XGBoost supports early stopping via eval_set
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)

        elapsed = time.time() - t0
        logger.info(f"    Training time: {elapsed:.1f}s")

        # --- Validate ---
        if model_name == "logistic_regression":
            y_val_prob = model.predict_proba(X_val)[:, 1]
        elif model_name == "random_forest":
            y_val_prob = model.predict_proba(X_val)[:, 1]
        else:
            y_val_prob = model.predict_proba(X_val)[:, 1]

        threshold, val_f1 = find_best_threshold(y_val, y_val_prob, metric="f1")
        metrics = quick_metrics(y_val, y_val_prob, threshold)
        logger.info(f"    Val AUROC={metrics['auroc']} | AUPRC={metrics['auprc']} | "
                    f"F1={metrics['f1']} @ thresh={threshold:.3f}")

        # --- Save model ---
        ensure_dir(save_dir)
        model_path = save_dir / f"{model_name}_{horizon}h.pkl"
        joblib.dump(model, model_path)

        # --- Save metadata ---
        meta = {
            "model_name":        model_name,
            "horizon_hours":     horizon,
            "label_col":         label_col,
            "n_features":        len(feature_cols),
            "threshold":         threshold,
            "scale_pos_weight":  scale_pos if model_name == "xgboost" else None,
            "train_n_pos":       n_pos,
            "train_n_neg":       n_neg,
            "val_metrics":       metrics,
            "training_time_s":   round(elapsed, 2),
        }
        meta_path = save_dir / f"metadata_{horizon}h.json"
        save_json(meta, meta_path)

        logger.info(f"    Saved → {model_path}")


def main(horizons_to_train: list = None) -> None:
    if horizons_to_train is None:
        horizons_to_train = HORIZONS

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    logger.info("Loading processed data...")
    train_raw = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val_raw   = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    logger.info(f"  Train: {train_raw.shape}  |  Val: {val_raw.shape}")

    # -----------------------------------------------------------------------
    # 2. Feature engineering (fit=True saves feature_list.json)
    # -----------------------------------------------------------------------
    logger.info("Applying feature engineering to training data (fit=True)...")
    train_fe = engineer_features(train_raw, fit=True)
    logger.info("Applying feature engineering to validation data...")
    val_fe   = engineer_features(val_raw,   fit=False)

    feature_cols = load_feature_list()
    logger.info(f"Using {len(feature_cols)} features.")

    # Leakage check: make sure no label/ID columns crept in
    from utils import LABEL_COLS, ID_COLS
    forbidden = set(LABEL_COLS + ["Patient_ID", "SepsisLabel"])
    leaked = [c for c in feature_cols if c in forbidden]
    assert not leaked, f"LEAKAGE DETECTED: {leaked}"
    logger.info("✓ Leakage check passed.")

    # -----------------------------------------------------------------------
    # 3. Train each horizon
    # -----------------------------------------------------------------------
    for h in horizons_to_train:
        train_horizon(h, train_fe, val_fe, feature_cols)

    logger.info("\n" + "="*60)
    logger.info("All models trained successfully.")
    logger.info("Run  python src/evaluate.py  to evaluate on the test set.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sepsis prediction models.")
    parser.add_argument(
        "--horizon", type=int, default=None,
        choices=[3, 6, 12],
        help="Train only a specific horizon (default: all 3, 6, 12)"
    )
    args = parser.parse_args()

    horizons = [args.horizon] if args.horizon else HORIZONS
    main(horizons_to_train=horizons)
