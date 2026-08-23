"""
explain.py
----------
Generates SHAP explanations for the XGBoost models (all 3 horizons).

Outputs:
  - results/shap/summary_{h}h.png      — SHAP beeswarm summary plot
  - results/shap/importance_{h}h.png   — SHAP bar chart (mean |SHAP|)
  - results/shap/importance_{h}h.csv   — Ranked feature importance table
  - results/shap/patient_{id}_{h}h.png — Waterfall plot for selected patients

Run:
    python src/explain.py
    python src/explain.py --horizon 6          # single horizon
    python src/explain.py --n_samples 500       # fewer samples for speed
"""

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_engineering import engineer_features, load_feature_list
from utils import (
    HORIZONS,
    PROCESSED_DIR,
    RANDOM_STATE,
    SHAP_DIR,
    XGB_DIR,
    ensure_dir,
    load_json,
    logger,
    save_json,
    set_seeds,
)

set_seeds(RANDOM_STATE)

# ---------------------------------------------------------------------------
# Global SHAP analysis
# ---------------------------------------------------------------------------

def explain_global(horizon: int, test_fe: pd.DataFrame,
                   feature_cols: list, n_samples: int = 2000) -> None:
    """
    Compute and save global SHAP explanations for the XGBoost model at a
    given horizon.

    Uses TreeExplainer (fast and exact for tree-based models).
    Samples up to n_samples rows for speed; this is representative.
    """
    model_path = XGB_DIR / f"xgboost_{horizon}h.pkl"
    if not model_path.exists():
        logger.warning(f"XGBoost {horizon}h model not found — skipping SHAP.")
        return

    logger.info(f"\nComputing SHAP for {horizon}h model...")
    model = joblib.load(model_path)

    # Sample for speed (stratified: keep all positives, sample negatives)
    label_col = f"label_{horizon}h"
    pos_rows   = test_fe[test_fe[label_col] == 1]
    neg_rows   = test_fe[test_fe[label_col] == 0]

    rng = np.random.default_rng(RANDOM_STATE)
    n_neg_sample = min(n_samples - len(pos_rows), len(neg_rows))
    if n_neg_sample < 0:
        n_neg_sample = n_samples

    neg_idx = rng.choice(len(neg_rows), size=n_neg_sample, replace=False)
    sample  = pd.concat([pos_rows, neg_rows.iloc[neg_idx]], axis=0)

    X_sample = sample[feature_cols].values
    logger.info(f"  Using {len(X_sample):,} samples "
                f"(pos={len(pos_rows):,}, neg={n_neg_sample:,})")

    # XGBoost TreeExplainer
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    ensure_dir(SHAP_DIR)

    # --- Beeswarm Summary Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_cols,
        show=False,
        max_display=20,
        plot_size=None,
    )
    plt.title(f"SHAP Summary — XGBoost {horizon}h Prediction Horizon",
              fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / f"summary_{horizon}h.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Beeswarm plot → {SHAP_DIR / f'summary_{horizon}h.png'}")

    # --- Bar Plot (mean |SHAP|) ---
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":    feature_cols,
        "mean_shap":  mean_abs_shap,
    }).sort_values("mean_shap", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    top20 = importance_df.head(20)
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(top20)))
    ax.barh(top20["feature"][::-1], top20["mean_shap"][::-1],
            color=colors)
    ax.set_xlabel("Mean |SHAP value| (impact on model output)")
    ax.set_title(f"Feature Importance via SHAP — XGBoost {horizon}h Horizon")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / f"importance_{horizon}h.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Bar plot → {SHAP_DIR / f'importance_{horizon}h.png'}")

    # --- Save importance table ---
    importance_df.to_csv(SHAP_DIR / f"importance_{horizon}h.csv", index=False)
    logger.info(f"  Importance CSV → {SHAP_DIR / f'importance_{horizon}h.csv'}")

    # --- Save SHAP values for dashboard use ---
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df.to_parquet(SHAP_DIR / f"shap_values_{horizon}h.parquet", index=False)

    # Save feature importance as JSON for dashboard
    save_json(
        importance_df.head(30).to_dict(orient="records"),
        SHAP_DIR / f"top_features_{horizon}h.json",
    )


# ---------------------------------------------------------------------------
# Patient-level SHAP
# ---------------------------------------------------------------------------

def explain_patient(horizon: int, test_fe: pd.DataFrame,
                    feature_cols: list, patient_ids: list) -> None:
    """
    Generate patient-level SHAP waterfall plots for the specified patients.
    """
    model_path = XGB_DIR / f"xgboost_{horizon}h.pkl"
    if not model_path.exists():
        return

    model     = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)

    for pid in patient_ids:
        patient_rows = test_fe[test_fe["Patient_ID"] == pid]
        if len(patient_rows) == 0:
            continue

        # Use the row closest to onset (highest risk)
        label_col = f"label_{horizon}h"
        pos_rows  = patient_rows[patient_rows[label_col] == 1]
        if len(pos_rows) > 0:
            row_idx = pos_rows.index[0]  # first positive row
        else:
            row_idx = patient_rows.index[-1]  # last row for non-septic

        X_row     = test_fe.loc[[row_idx], feature_cols].values
        shap_vals = explainer.shap_values(X_row)[0]

        # Waterfall plot (top-15 features)
        shap_importance = np.abs(shap_vals)
        top_idx         = np.argsort(shap_importance)[::-1][:15]
        top_features    = [feature_cols[i] for i in top_idx]
        top_shap        = shap_vals[top_idx]
        top_values      = X_row[0][top_idx]

        colors = ["#e74c3c" if v > 0 else "#3498db" for v in top_shap]

        fig, ax = plt.subplots(figsize=(9, 6))
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_shap, color=colors, edgecolor="gray", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [f"{f} = {v:.2f}" for f, v in zip(top_features, top_values)],
            fontsize=9,
        )
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("SHAP Value (positive = increases sepsis risk)")
        ax.set_title(
            f"Patient {pid} — Feature Contributions\n"
            f"({horizon}h Prediction | "
            f"Red = Risk ↑, Blue = Risk ↓)",
            fontsize=11,
        )
        plt.tight_layout()
        save_path = SHAP_DIR / f"patient_{pid}_{horizon}h.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Patient SHAP → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(horizons_to_explain: list = None, n_samples: int = 2000) -> None:
    if horizons_to_explain is None:
        horizons_to_explain = HORIZONS

    logger.info("Loading test data for SHAP analysis...")
    test_raw = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    logger.info(f"  Test: {test_raw.shape}")

    logger.info("Applying feature engineering...")
    test_fe = engineer_features(test_raw, fit=False)

    feature_cols = load_feature_list()

    # Select a few septic patients for patient-level explanations
    septic_pids = (
        test_fe[test_fe["label_12h"] == 1]["Patient_ID"]
        .unique()
    )
    rng = np.random.default_rng(RANDOM_STATE)
    selected_pids = rng.choice(
        septic_pids,
        size=min(5, len(septic_pids)),
        replace=False,
    ).tolist()
    logger.info(f"Patient-level SHAP for patients: {selected_pids}")

    # Save selected patient IDs for dashboard
    ensure_dir(SHAP_DIR)
    save_json({"patient_ids": [int(p) for p in selected_pids]},
              SHAP_DIR / "selected_patients.json")

    for h in horizons_to_explain:
        explain_global(h, test_fe, feature_cols, n_samples=n_samples)
        explain_patient(h, test_fe, feature_cols, selected_pids)

    logger.info("\nSHAP analysis complete. Results saved under results/shap/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP explainability for sepsis models.")
    parser.add_argument("--horizon", type=int, default=None, choices=[3, 6, 12])
    parser.add_argument("--n_samples", type=int, default=2000,
                        help="Max samples for global SHAP (default=2000)")
    args = parser.parse_args()

    horizons = [args.horizon] if args.horizon else HORIZONS
    main(horizons_to_explain=horizons, n_samples=args.n_samples)
