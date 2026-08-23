"""
feature_engineering.py
-----------------------
Constructs temporal ICU features for early sepsis detection.

Uses pd.concat() to build all features at once, avoiding DataFrame
fragmentation. All rolling features use shift(1) to prevent look-ahead bias.

Feature list is saved to models/feature_list.json after fitting on train data.

Run:
    python src/feature_engineering.py   (self-test)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    BASE_MEASUREMENT_COLS,
    LABEL_COLS,
    MODEL_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    ensure_dir,
    logger,
    save_json,
    set_seeds,
)

set_seeds(RANDOM_STATE)

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------

# Key columns: get full rolling stats (mean, min, max, std at 3h + mean at 6h)
ROLLING_COLS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "Resp",   # vital signs
    "Lactate", "WBC", "Creatinine", "BUN",           # key sepsis labs
    "pH", "HCO3", "FiO2", "PaCO2",                  # acid-base / oxygenation
    "Glucose", "Potassium",                          # metabolic
    "Hgb", "Platelets", "PTT",                       # haematology / coag
]

# All base columns: get lag + delta only (fast)
ALL_COLS = BASE_MEASUREMENT_COLS  # 34 columns

ROLL_SHORT = 3  # 3-hour window
ROLL_LONG  = 6  # 6-hour window


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    """
    Build temporal and composite features for the sepsis prediction model.

    All new columns are accumulated in a dict and joined once at the end
    using pd.concat() — this avoids DataFrame fragmentation and is fast.

    Leakage prevention
    ------------------
    - Rolling windows are applied to the .shift(1) series (no current-row info).
    - groupby("Patient_ID") ensures no cross-patient contamination.
    - Data is sorted by (Patient_ID, ICULOS) before any operation.
    """
    t0 = time.time()
    df = df.sort_values(["Patient_ID", "ICULOS"]).copy().reset_index(drop=True)
    n_patients = df["Patient_ID"].nunique()
    logger.info(f"Feature engineering: {n_patients:,} patients, {len(df):,} rows")

    # Accumulate all new feature series here, then concat once at the end
    new_features: dict = {}

    # ------------------------------------------------------------------
    # 1. Lag + delta for ALL base measurement columns  (fast)
    # ------------------------------------------------------------------
    logger.info("  Step 1/4: Lag + delta for all 34 measurement columns...")
    for col in ALL_COLS:
        prev  = df.groupby("Patient_ID", sort=False)[col].shift(1)
        delta = df[col] - prev
        new_features[f"{col}_prev"]  = prev.fillna(df[col])   # first-row → current value
        new_features[f"{col}_delta"] = delta.fillna(0)        # first-row → 0

    # ------------------------------------------------------------------
    # 2. Rolling stats for key columns using groupby().rolling()
    #    Applied on the SHIFTED series to prevent look-ahead.
    # ------------------------------------------------------------------
    logger.info(f"  Step 2/4: {ROLL_SHORT}h rolling (mean/min/max/std) "
                f"for {len(ROLLING_COLS)} key columns...")
    for col in ROLLING_COLS:
        # Shifted series: value at t-1
        shifted = df.groupby("Patient_ID", sort=False)[col].shift(1).fillna(df[col])

        # Temporarily attach shifted col for groupby().rolling()
        df["__s__"] = shifted
        roll3 = df.groupby("Patient_ID", sort=False)["__s__"].rolling(
            ROLL_SHORT, min_periods=1
        )
        new_features[f"{col}_roll{ROLL_SHORT}_mean"] = (
            roll3.mean().reset_index(level=0, drop=True)
        )
        new_features[f"{col}_roll{ROLL_SHORT}_min"] = (
            roll3.min().reset_index(level=0, drop=True)
        )
        new_features[f"{col}_roll{ROLL_SHORT}_max"] = (
            roll3.max().reset_index(level=0, drop=True)
        )
        new_features[f"{col}_roll{ROLL_SHORT}_std"] = (
            roll3.std().reset_index(level=0, drop=True).fillna(0)
        )
        df.drop(columns=["__s__"], inplace=True)

    logger.info(f"  Step 3/4: {ROLL_LONG}h rolling mean for "
                f"{len(ROLLING_COLS)} key columns...")
    for col in ROLLING_COLS:
        shifted = df.groupby("Patient_ID", sort=False)[col].shift(1).fillna(df[col])
        df["__s__"] = shifted
        roll6 = df.groupby("Patient_ID", sort=False)["__s__"].rolling(
            ROLL_LONG, min_periods=1
        )
        new_features[f"{col}_roll{ROLL_LONG}_mean"] = (
            roll6.mean().reset_index(level=0, drop=True)
        )
        df.drop(columns=["__s__"], inplace=True)

    # ------------------------------------------------------------------
    # 3. Time-based features
    # ------------------------------------------------------------------
    logger.info("  Step 4/4: Time + composite features...")
    new_features["hours_in_icu"] = df["ICULOS"]
    new_features["time_since_prev"] = (
        df.groupby("Patient_ID", sort=False)["ICULOS"].diff().fillna(0)
    )

    # ------------------------------------------------------------------
    # 4. Composite clinical features
    # ------------------------------------------------------------------
    eps = 1e-6
    new_features["shock_index"]    = df["HR"]      / (df["SBP"] + eps)
    new_features["pulse_pressure"] = df["SBP"]     - df["DBP"]
    new_features["bun_creatinine"] = df["BUN"]     / (df["Creatinine"] + eps)
    new_features["spo2_fio2"]      = df["O2Sat"]   / (df["FiO2"] + eps)
    new_features["lactate_wbc"]    = df["Lactate"] * df["WBC"]

    # SIRS score (0–4): Temp abnormal + tachycardia + tachypnoea + WBC abnormal
    new_features["sirs_score"] = (
        ((df["Temp"] > 38) | (df["Temp"] < 36)).astype(int)
        + (df["HR"]  > 90).astype(int)
        + (df["Resp"] > 20).astype(int)
        + ((df["WBC"] > 12) | (df["WBC"] < 4)).astype(int)
    )

    # ------------------------------------------------------------------
    # 5. Join all new features at once (no fragmentation)
    # ------------------------------------------------------------------
    new_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, new_df], axis=1)

    # ------------------------------------------------------------------
    # 6. Build and validate feature list
    # ------------------------------------------------------------------
    non_feature_cols = {"Patient_ID", "ICULOS", "Hour", "SepsisLabel"} | set(LABEL_COLS)
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    for lc in LABEL_COLS:
        assert lc not in feature_cols, f"LEAKAGE: label {lc} in features!"

    # Fill any residual NaNs (shouldn't exist, but safety net)
    null_total = df[feature_cols].isnull().sum().sum()
    if null_total > 0:
        logger.warning(f"  {null_total} NaNs found — filling with 0")
        df[feature_cols] = df[feature_cols].fillna(0)

    elapsed = time.time() - t0
    logger.info(f"  Done in {elapsed:.1f}s. Features: {len(feature_cols)}. "
                f"Shape: {df.shape}")

    if fit:
        ensure_dir(MODEL_DIR)
        fp = MODEL_DIR / "feature_list.json"
        save_json({"features": feature_cols, "n_features": len(feature_cols)}, fp)
        logger.info(f"  Feature list saved → {fp}")

    return df


def load_feature_list() -> list:
    """Load saved feature list from models/feature_list.json."""
    path = MODEL_DIR / "feature_list.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature list not found at {path}. Run train.py first."
        )
    return json.loads(path.read_text())["features"]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Self-test: feature engineering on training data...")
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    logger.info(f"Loaded train: {train.shape}")
    train_fe = engineer_features(train, fit=True)
    feature_cols = load_feature_list()
    logger.info(f"Feature list: {len(feature_cols)} features")
    null_check = train_fe[feature_cols].isnull().sum().sum()
    logger.info(f"NaN check: {null_check} nulls (expected 0)")
