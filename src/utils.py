"""
utils.py
--------
Shared constants, path helpers, and logging utilities used across all
modules in the early-sepsis-detection pipeline.

Import this at the top of every script:
    from utils import PROJECT_ROOT, DATA_DIR, MODEL_DIR, RESULTS_DIR, logger
"""

import json
import logging
import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project root — resolves correctly regardless of where the script is invoked
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------
DATA_DIR       = PROJECT_ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"

MODEL_DIR      = PROJECT_ROOT / "models"
LR_DIR         = MODEL_DIR / "logistic_regression"
RF_DIR         = MODEL_DIR / "random_forest"
XGB_DIR        = MODEL_DIR / "xgboost"

RESULTS_DIR    = PROJECT_ROOT / "results"
METRICS_DIR    = RESULTS_DIR / "metrics"
PLOTS_DIR      = RESULTS_DIR / "plots"
SHAP_DIR       = RESULTS_DIR / "shap"
PREDS_DIR      = RESULTS_DIR / "predictions"

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
HORIZONS     = [3, 6, 12]   # hours before clinical onset

# ---------------------------------------------------------------------------
# Column definitions — derived from actual dataset schema
# ---------------------------------------------------------------------------

# Columns that identify a row — never used as model features
ID_COLS = ["Patient_ID", "ICULOS", "Hour", "SepsisLabel"]

# Label columns (targets for each prediction horizon)
LABEL_COLS = ["label_3h", "label_6h", "label_12h"]

# Demographic / static features
DEMO_COLS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]

# Missingness indicators added by preprocessing
MISS_COLS = ["Lactate_measured", "WBC_measured", "Creatinine_measured"]

# Raw physiological and laboratory measurement columns
VITAL_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]

LAB_COLS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]

# Base measurement columns (vitals + labs)
BASE_MEASUREMENT_COLS = VITAL_COLS + LAB_COLS

# Model directory mapping
MODEL_DIR_MAP = {
    "logistic_regression": LR_DIR,
    "random_forest":       RF_DIR,
    "xgboost":             XGB_DIR,
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int = RANDOM_STATE) -> None:
    """Set random seeds for reproducibility across numpy, Python random, and OS."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str = "sepsis") -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# Convenience singleton logger for use as `from utils import logger`
logger = get_logger("sepsis")


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns the path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: Path) -> None:
    """Save a dictionary to a JSON file."""
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> dict:
    """Load a JSON file and return as a dictionary."""
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Leakage check utility
# ---------------------------------------------------------------------------

def check_no_patient_overlap(train_df, val_df, test_df) -> None:
    """
    Assert that no patient appears in more than one split.
    Raises AssertionError with details if leakage is detected.
    """
    tr = set(train_df["Patient_ID"].unique())
    va = set(val_df["Patient_ID"].unique())
    te = set(test_df["Patient_ID"].unique())

    tr_va = tr & va
    tr_te = tr & te
    va_te = va & te

    assert not tr_va, f"Patient leakage: train∩val = {len(tr_va)} patients"
    assert not tr_te, f"Patient leakage: train∩test = {len(tr_te)} patients"
    assert not va_te, f"Patient leakage: val∩test = {len(va_te)} patients"
    logger.info("✓ No patient leakage across train/val/test splits.")


def check_no_target_in_features(feature_cols: list) -> None:
    """Assert that no label or identifier columns are present in feature list."""
    forbidden = set(ID_COLS + LABEL_COLS)
    leaked = [c for c in feature_cols if c in forbidden]
    assert not leaked, f"Target/ID columns found in features: {leaked}"
    logger.info("✓ No target or ID columns in feature set.")
