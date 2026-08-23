"""
predict.py
----------
Standalone inference script. Loads trained models and generates sepsis
risk scores for new patient data.

Usage:
    python src/predict.py --patient_id 12345
    python src/predict.py --input_file data/new_patient.csv
    python src/predict.py --list_patients          # list septic test patients
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_engineering import engineer_features, load_feature_list
from utils import (
    HORIZONS,
    MODEL_DIR,
    PREDS_DIR,
    PROCESSED_DIR,
    XGB_DIR,
    LR_DIR,
    RF_DIR,
    load_json,
    logger,
    set_seeds,
    RANDOM_STATE,
)

set_seeds(RANDOM_STATE)

MODEL_DIRS = {
    "logistic_regression": LR_DIR,
    "random_forest":       RF_DIR,
    "xgboost":             XGB_DIR,
}


def classify_risk(prob: float, threshold: float) -> str:
    """
    Map a predicted probability to a clinical risk category.

    Thresholds are based on the decision threshold + margins:
      - Low:      prob < threshold * 0.5
      - Moderate: prob < threshold * 1.5
      - High:     prob >= threshold * 1.5
    """
    if prob < threshold * 0.5:
        return "Low Risk"
    elif prob < threshold * 1.5:
        return "Moderate Risk"
    else:
        return "High Risk"


def predict_patient(patient_data: pd.DataFrame, model_name: str = "xgboost") -> pd.DataFrame:
    """
    Run inference for a single patient across all three horizons.

    Parameters
    ----------
    patient_data : DataFrame with one patient's ICU time-series
                   (must have the same columns as the processed parquet files)
    model_name   : one of 'logistic_regression', 'random_forest', 'xgboost'

    Returns
    -------
    DataFrame with columns: ICULOS, prob_3h, prob_6h, prob_12h,
                            risk_3h, risk_6h, risk_12h
    """
    mdir = MODEL_DIRS.get(model_name)
    if mdir is None:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Choose from: {list(MODEL_DIRS.keys())}")

    # Feature engineering (single patient, no fitting)
    fe_data      = engineer_features(patient_data, fit=False)
    feature_cols = load_feature_list()
    X            = fe_data[feature_cols].values

    results = fe_data[["Patient_ID", "ICULOS"]].copy().reset_index(drop=True)

    for h in HORIZONS:
        model_path = mdir / f"{model_name}_{h}h.pkl"
        meta_path  = mdir / f"metadata_{h}h.json"

        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            results[f"prob_{h}h"] = np.nan
            results[f"risk_{h}h"] = "Unknown"
            continue

        model     = joblib.load(model_path)
        threshold = load_json(meta_path)["threshold"]
        prob      = model.predict_proba(X)[:, 1]

        results[f"prob_{h}h"] = prob
        results[f"risk_{h}h"] = [classify_risk(p, threshold) for p in prob]

    return results


def print_risk_summary(results: pd.DataFrame) -> None:
    """Print a formatted risk summary table to stdout."""
    print("\n" + "="*70)
    print(f"SEPSIS RISK PREDICTION — Patient {results['Patient_ID'].iloc[0]}")
    print("="*70)
    print(f"{'Hour':>6}  {'3h Prob':>9} {'3h Risk':>14}  "
          f"{'6h Prob':>9} {'6h Risk':>14}  "
          f"{'12h Prob':>9} {'12h Risk':>14}")
    print("-"*70)
    for _, row in results.iterrows():
        print(
            f"{row['ICULOS']:>6}  "
            f"{row.get('prob_3h', float('nan')):>9.4f} {row.get('risk_3h', 'N/A'):>14}  "
            f"{row.get('prob_6h', float('nan')):>9.4f} {row.get('risk_6h', 'N/A'):>14}  "
            f"{row.get('prob_12h', float('nan')):>9.4f} {row.get('risk_12h', 'N/A'):>14}"
        )
    print("="*70)
    print("\n⚠️  DISCLAIMER: These predictions are for research purposes only.")
    print("   Do NOT use for clinical decision-making without proper validation.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Predict sepsis risk for a patient."
    )
    parser.add_argument("--patient_id", type=int, default=None,
                        help="Patient ID from the test set")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Path to a CSV file with patient data")
    parser.add_argument("--model", type=str, default="xgboost",
                        choices=list(MODEL_DIRS.keys()),
                        help="Model to use for prediction (default: xgboost)")
    parser.add_argument("--list_patients", action="store_true",
                        help="List available patient IDs from the test set")
    args = parser.parse_args()

    # --- List patients mode ---
    if args.list_patients:
        test_raw = pd.read_parquet(PROCESSED_DIR / "test.parquet")
        septic   = test_raw[test_raw["label_12h"] == 1]["Patient_ID"].unique()
        non_sep  = test_raw[test_raw["label_12h"] == 0]["Patient_ID"].unique()
        print(f"\nTest set: {test_raw['Patient_ID'].nunique():,} patients")
        print(f"  Septic (label_12h=1):     {len(septic):,} patients")
        print(f"  Non-septic (label_12h=0): {len(non_sep):,} patients")
        print(f"\nSample septic patient IDs: {list(septic[:10])}")
        print(f"Sample non-septic IDs:     {list(non_sep[:10])}")
        return

    # --- Patient ID mode (load from test set) ---
    if args.patient_id is not None:
        logger.info(f"Loading test set for patient {args.patient_id}...")
        test_raw = pd.read_parquet(PROCESSED_DIR / "test.parquet")
        patient_data = test_raw[test_raw["Patient_ID"] == args.patient_id]
        if len(patient_data) == 0:
            logger.error(f"Patient {args.patient_id} not found in test set.")
            return
        logger.info(f"  Found {len(patient_data)} rows.")

    # --- CSV file mode ---
    elif args.input_file:
        logger.info(f"Loading patient data from {args.input_file}...")
        patient_data = pd.read_csv(args.input_file)

    else:
        parser.print_help()
        return

    results = predict_patient(patient_data, model_name=args.model)
    print_risk_summary(results)

    # Save to file
    out_path = PREDS_DIR / f"patient_{results['Patient_ID'].iloc[0]}_{args.model}.csv"
    results.to_csv(out_path, index=False)
    logger.info(f"Predictions saved → {out_path}")


if __name__ == "__main__":
    main()
