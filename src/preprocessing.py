"""
preprocessing.py
-----------------
Loads the raw PhysioNet 2019 sepsis dataset (Dataset.csv), reconstructs
early-warning labels at multiple prediction horizons (3h / 6h / 12h before
clinical onset), handles missing values, and writes patient-level
train/val/test splits to data/processed/.

Run:
    python src/preprocessing.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/Dataset.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [3, 6, 12]  # hours before onset to flag as "positive"
RANDOM_STATE = 42

# Columns that are identifiers / time index, never used as model features
ID_COLS = ["Patient_ID", "Hour", "ICULOS", "SepsisLabel"]

# Vitals + labs that get forward-filled per patient (carry last observation
# forward, since ICU measurements are sparse and a stale value is still
# clinically informative until the next reading)
FFILL_COLS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # drop the stray unnamed pandas index column saved in the CSV
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


def reconstruct_onset_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each patient, find the first hour SepsisLabel==1 (t_onset).
    Truncate the patient's timeline at t_onset (rows after onset are
    dropped -- once sepsis has occurred there is nothing left to predict).
    For each horizon h, label_h = 1 if the row falls within h hours of
    t_onset, else 0. Non-septic patients are labeled 0 at every hour.
    """
    df = df.sort_values(["Patient_ID", "ICULOS"]).reset_index(drop=True)

    onset = (
        df.loc[df["SepsisLabel"] == 1]
        .groupby("Patient_ID")["ICULOS"]
        .min()
        .rename("t_onset")
    )
    df = df.merge(onset, on="Patient_ID", how="left")

    # drop rows at/after onset hour is not what we want -- keep the onset
    # hour itself out too, since by definition the horizon window is
    # STRICTLY before onset. Rows after onset carry no predictive value.
    before_onset_mask = df["t_onset"].isna() | (df["ICULOS"] < df["t_onset"])
    df = df[before_onset_mask].copy()

    for h in HORIZONS:
        df[f"label_{h}h"] = (
            df["t_onset"].notna() & ((df["t_onset"] - df["ICULOS"]) <= h)
        ).astype(int)

    df = df.drop(columns=["t_onset"])
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-stage imputation, patient-aware:
      1. Forward-fill within each patient (carries the last real reading
         forward -- appropriate for sparse, irregularly sampled vitals/labs).
      2. Any values still missing (e.g. a patient's very first hours before
         their first reading) are filled with the population median.
    Also adds a missingness indicator for a few labs that are clinically
    meaningful specifically when they're MISSING vs measured (e.g. Lactate
    is only drawn when sepsis is suspected, so "was it measured at all"
    carries signal).
    """
    df = df.sort_values(["Patient_ID", "ICULOS"]).copy()

    # missingness indicators (before imputation) for key sepsis-relevant labs
    for col in ["Lactate", "WBC", "Creatinine"]:
        df[f"{col}_measured"] = df[col].notna().astype(int)

    # forward-fill within patient
    df[FFILL_COLS] = df.groupby("Patient_ID")[FFILL_COLS].ffill()

    # population median for whatever's left (mostly leading NaNs)
    medians = df[FFILL_COLS].median()
    df[FFILL_COLS] = df[FFILL_COLS].fillna(medians)

    # Unit1/Unit2 (ICU type flags) and HospAdmTime: fill with 0 / median
    if "Unit1" in df.columns:
        df["Unit1"] = df["Unit1"].fillna(0)
    if "Unit2" in df.columns:
        df["Unit2"] = df["Unit2"].fillna(0)
    if "HospAdmTime" in df.columns:
        df["HospAdmTime"] = df["HospAdmTime"].fillna(df["HospAdmTime"].median())

    return df


def patient_level_split(df: pd.DataFrame, train=0.7, val=0.15, seed=RANDOM_STATE):
    """
    Split by Patient_ID (never by row) so no patient's timeline leaks
    across train/val/test. Stratify on ever-septic status so the rare
    positive class is represented proportionally in each split.
    """
    rng = np.random.default_rng(seed)
    patient_status = df.groupby("Patient_ID")["label_12h"].max()  # widest horizon = "ever positive"

    pos_ids = patient_status[patient_status == 1].index.to_numpy().copy()
    neg_ids = patient_status[patient_status == 0].index.to_numpy().copy()
    rng.shuffle(pos_ids)
    rng.shuffle(neg_ids)

    def split_ids(ids):
        n = len(ids)
        n_train = int(n * train)
        n_val = int(n * val)
        return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]

    pos_tr, pos_va, pos_te = split_ids(pos_ids)
    neg_tr, neg_va, neg_te = split_ids(neg_ids)

    train_ids = set(np.concatenate([pos_tr, neg_tr]))
    val_ids = set(np.concatenate([pos_va, neg_va]))
    test_ids = set(np.concatenate([pos_te, neg_te]))

    train_df = df[df["Patient_ID"].isin(train_ids)]
    val_df = df[df["Patient_ID"].isin(val_ids)]
    test_df = df[df["Patient_ID"].isin(test_ids)]
    return train_df, val_df, test_df


def main():
    print("Loading raw data...")
    df = load_raw(RAW_PATH)
    print(f"  raw shape: {df.shape}, patients: {df['Patient_ID'].nunique()}")

    print("Reconstructing onset labels (3h/6h/12h horizons)...")
    df = reconstruct_onset_labels(df)
    for h in HORIZONS:
        rate = df[f"label_{h}h"].mean()
        print(f"  label_{h}h positive rate: {rate:.4%}")

    print("Handling missing values...")
    df = handle_missing_values(df)
    remaining_na = df[FFILL_COLS].isna().sum().sum()
    print(f"  remaining NaNs after imputation: {remaining_na}")

    print("Splitting by patient (70/15/15, stratified on ever-positive)...")
    train_df, val_df, test_df = patient_level_split(df)
    print(f"  train: {train_df.shape}, patients={train_df['Patient_ID'].nunique()}")
    print(f"  val:   {val_df.shape}, patients={val_df['Patient_ID'].nunique()}")
    print(f"  test:  {test_df.shape}, patients={test_df['Patient_ID'].nunique()}")

    # sanity check: no patient overlap
    tr_ids = set(train_df["Patient_ID"])
    va_ids = set(val_df["Patient_ID"])
    te_ids = set(test_df["Patient_ID"])
    assert not (tr_ids & va_ids) and not (tr_ids & te_ids) and not (va_ids & te_ids), \
        "Patient leakage across splits!"
    print("  no patient leakage confirmed.")

    print("Saving processed splits...")
    try:
        train_df.to_parquet(OUT_DIR / "train.parquet", index=False)
        val_df.to_parquet(OUT_DIR / "val.parquet", index=False)
        test_df.to_parquet(OUT_DIR / "test.parquet", index=False)
        print(f"  saved parquet to {OUT_DIR}/")
    except ImportError:
        # pyarrow not installed -- fall back to CSV so the pipeline still runs
        train_df.to_csv(OUT_DIR / "train.csv", index=False)
        val_df.to_csv(OUT_DIR / "val.csv", index=False)
        test_df.to_csv(OUT_DIR / "test.csv", index=False)
        print(f"  pyarrow not found -- saved CSV to {OUT_DIR}/ instead")


if __name__ == "__main__":
    main()