"""
evaluate.py
-----------
Evaluates all trained models on the held-out test set and generates:
  - Per-model, per-horizon classification metrics (JSON + CSV)
  - ROC curves, Precision-Recall curves
  - Confusion matrices
  - Horizon comparison plots
  - Patient risk progression plots

Run:
    python src/evaluate.py
"""

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/script use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_engineering import engineer_features, load_feature_list
from utils import (
    HORIZONS,
    LABEL_COLS,
    LR_DIR,
    METRICS_DIR,
    MODEL_DIR,
    PLOTS_DIR,
    PREDS_DIR,
    PROCESSED_DIR,
    RF_DIR,
    XGB_DIR,
    RANDOM_STATE,
    ensure_dir,
    load_json,
    logger,
    save_json,
    set_seeds,
)

set_seeds(RANDOM_STATE)

# Colour scheme for consistency across plots
MODEL_COLORS = {
    "logistic_regression": "#3498db",  # blue
    "random_forest":       "#2ecc71",  # green
    "xgboost":             "#e74c3c",  # red
}
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
}
HORIZON_COLORS = {3: "#e74c3c", 6: "#f39c12", 12: "#3498db"}

MODEL_DIRS = {
    "logistic_regression": LR_DIR,
    "random_forest":       RF_DIR,
    "xgboost":             XGB_DIR,
}


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float) -> dict:
    """
    Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true    : true binary labels
    y_prob    : predicted probabilities for the positive class
    threshold : decision threshold chosen on validation set

    Returns
    -------
    dict with all metrics
    """
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    auroc  = roc_auc_score(y_true, y_prob)
    auprc  = average_precision_score(y_true, y_prob)
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr         = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr         = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "auroc":       round(float(auroc), 4),
        "auprc":       round(float(auprc), 4),
        "precision":   round(float(prec),  4),
        "recall":      round(float(rec),   4),
        "f1":          round(float(f1),    4),
        "specificity": round(float(specificity), 4),
        "fpr":         round(float(fpr),   4),
        "fnr":         round(float(fnr),   4),
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
        "threshold":   round(float(threshold), 4),
        "n_positive":  int(y_true.sum()),
        "n_negative":  int((y_true == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_roc_pr(y_true, y_prob, label, color, ax_roc, ax_pr) -> None:
    """Add one model's ROC and PR curves to the provided axes."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    pre, rec, _ = precision_recall_curve(y_true, y_prob)

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    ax_roc.plot(fpr, tpr, color=color, lw=2,
                label=f"{label} (AUROC={auroc:.3f})")
    ax_pr.plot(rec, pre, color=color, lw=2,
               label=f"{label} (AUPRC={auprc:.3f})")


def plot_confusion_matrix(y_true, y_pred, title, save_path) -> None:
    """Save a normalised confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=["No Sepsis", "Sepsis"],
        yticklabels=["No Sepsis", "Sepsis"],
        ax=ax, cbar=True,
    )
    # Also annotate raw counts
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j + 0.5, i + 0.7, f"n={val:,}", ha="center", va="center",
                fontsize=8, color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Horizon-level evaluation
# ---------------------------------------------------------------------------

def evaluate_horizon(horizon: int, test_fe: pd.DataFrame,
                     feature_cols: list) -> dict:
    """
    Evaluate all three models for a single prediction horizon on the test set.

    Returns a nested dict: {model_name: metrics_dict}
    """
    label_col = f"label_{horizon}h"
    y_true    = test_fe[label_col].values
    X_test    = test_fe[feature_cols].values

    horizon_results = {}
    roc_pr_data     = {}   # store curves for combined plot

    for model_name, mdir in MODEL_DIRS.items():
        model_path = mdir / f"{model_name}_{horizon}h.pkl"
        meta_path  = mdir / f"metadata_{horizon}h.json"

        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}  — skipping.")
            continue

        model     = joblib.load(model_path)
        meta      = load_json(meta_path)
        threshold = meta["threshold"]

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        metrics = compute_metrics(y_true, y_prob, threshold)
        horizon_results[model_name] = metrics
        roc_pr_data[model_name]     = (y_prob,)

        logger.info(f"  [{model_name}] {horizon}h — "
                    f"AUROC={metrics['auroc']} | AUPRC={metrics['auprc']} | "
                    f"F1={metrics['f1']} | Recall={metrics['recall']}")

        # Save per-model metrics
        ensure_dir(METRICS_DIR)
        save_json(
            {"model": model_name, "horizon": horizon, **metrics},
            METRICS_DIR / f"{model_name}_{horizon}h_metrics.json",
        )

        # Confusion matrix
        ensure_dir(PLOTS_DIR)
        plot_confusion_matrix(
            y_true, y_pred,
            title=f"{MODEL_LABELS[model_name]} — {horizon}h Confusion Matrix",
            save_path=PLOTS_DIR / f"cm_{model_name}_{horizon}h.png",
        )

        # Save test-set predictions
        ensure_dir(PREDS_DIR)
        pred_df = test_fe[["Patient_ID", "ICULOS", label_col]].copy()
        pred_df["prob"] = y_prob
        pred_df["pred"] = y_pred
        pred_df.to_csv(
            PREDS_DIR / f"preds_{model_name}_{horizon}h.csv", index=False
        )

    # --- Combined ROC + PR curve for this horizon ---
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))

    for model_name, (y_prob,) in roc_pr_data.items():
        plot_roc_pr(
            y_true, y_prob,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLORS[model_name],
            ax_roc=ax_roc,
            ax_pr=ax_pr,
        )

    # ROC formatting
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC Curve — {horizon}h Prediction Horizon")
    ax_roc.legend(loc="lower right")
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1)

    # PR formatting
    pos_rate = y_true.mean()
    ax_pr.axhline(y=pos_rate, color="gray", linestyle="--", lw=1,
                  label=f"No-skill baseline ({pos_rate:.4f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Precision-Recall Curve — {horizon}h Horizon")
    ax_pr.legend(loc="upper right")
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1)

    plt.suptitle(f"Model Comparison — {horizon}-Hour Early Warning",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"roc_pr_{horizon}h.png", dpi=150, bbox_inches="tight")
    plt.close()

    return horizon_results


# ---------------------------------------------------------------------------
# Cross-horizon comparison plots
# ---------------------------------------------------------------------------

def plot_horizon_comparison(all_results: dict) -> None:
    """
    Create a multi-panel figure comparing metrics across horizons.

    all_results: {horizon: {model_name: metrics_dict}}
    """
    metrics_to_plot = ["auroc", "auprc", "recall", "precision", "f1"]
    model_names     = list(MODEL_DIRS.keys())

    fig, axes = plt.subplots(1, len(metrics_to_plot),
                             figsize=(5 * len(metrics_to_plot), 5))

    for ax, metric in zip(axes, metrics_to_plot):
        for model_name in model_names:
            vals  = [all_results[h].get(model_name, {}).get(metric, np.nan)
                     for h in HORIZONS]
            ax.plot(HORIZONS, vals, "o-",
                    color=MODEL_COLORS[model_name],
                    label=MODEL_LABELS[model_name],
                    lw=2, markersize=8)

        ax.set_title(metric.upper())
        ax.set_xlabel("Prediction Horizon (hours before onset)")
        ax.set_xticks(HORIZONS)
        ax.set_xticklabels([f"{h}h" for h in HORIZONS])
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Performance vs. Prediction Horizon",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "horizon_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved horizon comparison → {PLOTS_DIR / 'horizon_comparison.png'}")


def plot_model_comparison(all_results: dict) -> None:
    """Bar chart comparing AUROC and AUPRC across models per horizon."""
    for horizon in HORIZONS:
        models = list(MODEL_DIRS.keys())
        aurocs = [all_results[horizon].get(m, {}).get("auroc", 0) for m in models]
        auprcs = [all_results[horizon].get(m, {}).get("auprc", 0) for m in models]

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        bars1 = ax.bar(x - width/2, aurocs, width, label="AUROC",
                       color=[MODEL_COLORS[m] for m in models], alpha=0.8)
        bars2 = ax.bar(x + width/2, auprcs, width, label="AUPRC",
                       color=[MODEL_COLORS[m] for m in models], alpha=0.5,
                       hatch="//")

        for bar in bars1 + bars2:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(f"Model Comparison — {horizon}h Prediction Horizon")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"model_comparison_{horizon}h.png",
                    dpi=150, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Patient risk progression
# ---------------------------------------------------------------------------

def plot_patient_risk_progression(test_fe: pd.DataFrame,
                                  feature_cols: list,
                                  n_patients: int = 10) -> None:
    """
    For a selection of septic test patients, plot the predicted sepsis
    probability over time for all three horizons.

    Shows how risk rises as clinical onset approaches.
    """
    # Find septic patients (positive in any label) in test set
    septic_patients = (
        test_fe[test_fe["label_12h"] == 1]["Patient_ID"]
        .unique()
    )
    if len(septic_patients) == 0:
        logger.warning("No septic patients found in test set.")
        return

    rng = np.random.default_rng(RANDOM_STATE)
    selected = rng.choice(
        septic_patients,
        size=min(n_patients, len(septic_patients)),
        replace=False,
    )

    # Load XGBoost models (best-performing)
    xgb_models = {}
    xgb_thresholds = {}
    for h in HORIZONS:
        model_path = XGB_DIR / f"xgboost_{h}h.pkl"
        meta_path  = XGB_DIR / f"metadata_{h}h.json"
        if model_path.exists():
            xgb_models[h]     = joblib.load(model_path)
            xgb_thresholds[h] = load_json(meta_path)["threshold"]

    X_test = test_fe[feature_cols].values

    for pid in selected:
        patient_mask = test_fe["Patient_ID"] == pid
        patient_rows = test_fe[patient_mask]

        # Find the onset hour (where label_3h first becomes 1 or end of window)
        onset_approx = None
        for h in HORIZONS:
            pos_rows = patient_rows[patient_rows[f"label_{h}h"] == 1]
            if len(pos_rows) > 0 and onset_approx is None:
                onset_approx = pos_rows["ICULOS"].min() + h

        fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                                  gridspec_kw={"height_ratios": [3, 1]})
        ax_risk = axes[0]
        ax_vitals = axes[1]

        for h in HORIZONS:
            if h not in xgb_models:
                continue
            probs = xgb_models[h].predict_proba(
                test_fe.loc[patient_mask, feature_cols].values
            )[:, 1]
            ax_risk.plot(patient_rows["ICULOS"].values, probs,
                         color=HORIZON_COLORS[h], lw=2.5,
                         label=f"{h}h model",
                         marker="o", markersize=3)
            # Threshold line
            ax_risk.axhline(xgb_thresholds[h],
                            color=HORIZON_COLORS[h],
                            linestyle="--", lw=1, alpha=0.5)

        if onset_approx:
            ax_risk.axvline(onset_approx, color="black", lw=2,
                            linestyle=":", label="Approx. onset")

        ax_risk.set_ylabel("Predicted Sepsis Probability")
        ax_risk.set_title(f"Patient {pid} — Sepsis Risk Trajectory")
        ax_risk.legend(loc="upper left")
        ax_risk.set_ylim(0, 1)
        ax_risk.grid(True, alpha=0.3)

        # Overlay HR and Resp as simple vitals reference
        ax_vitals.plot(patient_rows["ICULOS"].values,
                       patient_rows["HR"].values,
                       color="#e74c3c", lw=1.5, label="HR")
        ax_vitals.plot(patient_rows["ICULOS"].values,
                       patient_rows["Resp"].values,
                       color="#3498db", lw=1.5, label="Resp Rate")
        ax_vitals.set_xlabel("ICU Hours")
        ax_vitals.set_ylabel("Vital Signs")
        ax_vitals.legend(loc="upper left", fontsize=8)
        ax_vitals.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = PLOTS_DIR / f"patient_risk_{pid}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved patient risk plot → {save_path}")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def save_comparison_table(all_results: dict) -> None:
    """Save a multi-index CSV comparison table of all model × horizon results."""
    rows = []
    for horizon in HORIZONS:
        for model_name, metrics in all_results.get(horizon, {}).items():
            rows.append({
                "horizon_h":   horizon,
                "model":       model_name,
                "auroc":       metrics.get("auroc"),
                "auprc":       metrics.get("auprc"),
                "precision":   metrics.get("precision"),
                "recall":      metrics.get("recall"),
                "f1":          metrics.get("f1"),
                "specificity": metrics.get("specificity"),
                "fpr":         metrics.get("fpr"),
                "fnr":         metrics.get("fnr"),
                "threshold":   metrics.get("threshold"),
            })
    df = pd.DataFrame(rows)
    out_path = METRICS_DIR / "comparison_table.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Comparison table saved → {out_path}")
    # Pretty print
    print("\n" + "="*80)
    print("TEST-SET RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Loading test data...")
    test_raw = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    logger.info(f"  Test: {test_raw.shape}")

    logger.info("Applying feature engineering to test data...")
    test_fe = engineer_features(test_raw, fit=False)

    feature_cols = load_feature_list()
    logger.info(f"Using {len(feature_cols)} features.")

    all_results = {}

    for h in HORIZONS:
        logger.info(f"\nEvaluating {h}h horizon...")
        results = evaluate_horizon(h, test_fe, feature_cols)
        all_results[h] = results

    logger.info("\nGenerating cross-horizon comparison plots...")
    plot_horizon_comparison(all_results)
    plot_model_comparison(all_results)

    logger.info("Generating patient risk progression plots...")
    plot_patient_risk_progression(test_fe, feature_cols, n_patients=10)

    save_comparison_table(all_results)

    logger.info("Evaluation complete. Results saved under results/")


if __name__ == "__main__":
    main()
