# Early Sepsis Detection in ICU Patients

A research-oriented, multi-horizon machine-learning early-warning system for sepsis detection in ICU patients. Predicts sepsis risk **3, 6, and 12 hours** before clinical onset using the PhysioNet/CinC 2019 dataset, with full explainability via SHAP and an interactive Streamlit dashboard.

---

## ⚠️ Medical Disclaimer

> **This project is for academic and research purposes only.**  
> It must NOT be used for clinical decision-making, patient triage, or diagnosis without rigorous prospective validation. Sepsis prediction models require external validation before any clinical deployment.

---

## 📋 Project Objective

Standard sepsis prediction models answer: *"Will this patient develop sepsis?"*

This system answers: *"How early can we reliably detect sepsis risk, and how does that risk evolve over time?"*

Three prediction horizons are modelled independently:

| Horizon | Target Label | Clinical Meaning |
|---------|-------------|-----------------|
| **3h** | `label_3h` | Flag 3 hours before clinical sepsis onset |
| **6h** | `label_6h` | Flag 6 hours before onset |
| **12h**| `label_12h`| Flag 12 hours before onset |

---

## 📦 Dataset

**PhysioNet/CinC Challenge 2019 Sepsis Dataset**

| Split | Rows | Patients |
|-------|------|----------|
| Train | 1,067,055 | 27,936 |
| Validation | 226,503 | 5,985 |
| Test | 230,736 | 5,989 |
| **Total** | **1,524,294** | **~40,000** |

Positive rates (label = 1):

| Label | Positive Rate |
|-------|-------------|
| label_3h  | ~0.479% |
| label_6h  | ~0.924% |
| label_12h | ~1.727% |

---

## 🏗️ Project Architecture

```
sepsis-prediction/
│
├── data/
│   ├── raw/Dataset.csv               ← Raw PhysioNet dataset
│   └── processed/
│       ├── train.parquet
│       ├── val.parquet
│       └── test.parquet
│
├── src/
│   ├── preprocessing.py              ← Load raw data, generate labels, split
│   ├── utils.py                      ← Constants, paths, logging helpers
│   ├── feature_engineering.py        ← Temporal features (rolling, delta, composite)
│   ├── train.py                      ← Train 3 models × 3 horizons = 9 models
│   ├── evaluate.py                   ← Full evaluation metrics + all plots
│   ├── explain.py                    ← SHAP global + patient-level explanations
│   └── predict.py                    ← Standalone inference for new patients
│
├── models/
│   ├── feature_list.json             ← Saved feature column names
│   ├── logistic_regression/          ← lr_{3,6,12}h.pkl + metadata
│   ├── random_forest/                ← rf_{3,6,12}h.pkl + metadata
│   └── xgboost/                      ← xgboost_{3,6,12}h.pkl + metadata
│
├── results/
│   ├── metrics/                      ← JSON metrics + comparison_table.csv
│   ├── plots/                        ← ROC, PR, confusion, horizon comparison
│   ├── shap/                         ← SHAP summary + importance plots + CSVs
│   └── predictions/                  ← Per-model test-set probability CSVs
│
├── dashboard/
│   └── app.py                        ← 4-page Streamlit dashboard
│
├── notebooks/                        ← Jupyter notebooks (exploratory)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- Windows / Linux / macOS

### 1. Clone / navigate to the project
```bash
cd sepsis-prediction
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📂 Dataset Setup

Place the raw dataset at:
```
data/raw/Dataset.csv
```

The raw file should be the combined PhysioNet 2019 Sepsis Challenge CSV (all patients in a single file with a `Patient_ID` column).

---

## 🚀 Running the Pipeline

### Step 1 — Preprocessing (skip if `data/processed/` parquet files exist)
```bash
python src/preprocessing.py
```
Generates `train.parquet`, `val.parquet`, `test.parquet` in `data/processed/`.

### Step 2 — Training
```bash
# Train all 9 models (3 types × 3 horizons)
python src/train.py

# Train a specific horizon only
python src/train.py --horizon 6
```
Saves models to `models/` and training metadata to each model directory.

### Step 3 — Evaluation
```bash
python src/evaluate.py
```
Evaluates all models on the test set. Saves:
- Metrics → `results/metrics/`
- ROC / PR curves → `results/plots/`
- Confusion matrices → `results/plots/`
- Patient risk plots → `results/plots/`
- Test predictions → `results/predictions/`

### Step 4 — SHAP Explainability
```bash
python src/explain.py

# Faster (fewer samples):
python src/explain.py --n_samples 500
```
Saves SHAP plots and importance tables → `results/shap/`

### Step 5 — Dashboard
```bash
streamlit run dashboard/app.py
```
Opens the interactive dashboard at `http://localhost:8501`.

### Step 6 — Predict on a specific patient
```bash
# List available patients
python src/predict.py --list_patients

# Predict for patient 12345
python src/predict.py --patient_id 12345

# Use a different model
python src/predict.py --patient_id 12345 --model random_forest
```

---

## 🤖 Models

### Logistic Regression (Baseline)
- **Purpose**: Interpretable linear baseline
- **Imbalance handling**: `class_weight='balanced'`
- **Regularisation**: L2 (C=0.1)
- **Preprocessing**: StandardScaler pipeline

### Random Forest
- **Purpose**: Non-linear tree-based baseline
- **Imbalance handling**: `class_weight='balanced'`
- **Configuration**: 200 estimators, max_depth=12, min_samples_leaf=20

### XGBoost (Primary Model)
- **Purpose**: High-performance gradient boosting
- **Imbalance handling**: `scale_pos_weight = n_negative / n_positive`
- **Configuration**: 500 estimators, learning_rate=0.05, max_depth=6
- **Regularisation**: subsample=0.8, colsample_bytree=0.8

---

## 📊 Evaluation Metrics

Because the positive class is **extremely rare** (~0.5%–1.7%), accuracy is not meaningful.

| Metric | Why it matters |
|--------|---------------|
| **AUROC** | Overall discrimination — probability model ranks a random positive above a random negative |
| **AUPRC** | Area under precision-recall curve — most important for rare positive class |
| **Recall / Sensitivity** | Fraction of actual sepsis cases caught — missing sepsis is dangerous |
| **Precision** | Fraction of alerts that are true positives |
| **F1 Score** | Harmonic mean of precision and recall |
| **Specificity** | Fraction of non-septic patients correctly identified |
| **FPR / FNR** | False positive / false negative rates |

**Threshold selection**: The decision threshold is chosen on the **validation set** (never the test set) by maximising F1 score.

---

## 🔍 Explainability

SHAP (SHapley Additive exPlanations) is used to explain the XGBoost model:

- **Global**: Mean |SHAP| ranks features by overall importance
- **Beeswarm**: Shows the distribution of SHAP values and their direction
- **Patient-level**: Waterfall chart showing top contributing features for each prediction

Key sepsis-related features expected to dominate:
- Lactate (tissue hypoperfusion)
- HR / Resp rate (SIRS criteria)
- WBC (infection marker)
- MAP / shock_index (circulatory failure)
- ICULOS (time in ICU)

---

## 🧪 Feature Engineering

From 49 base columns, the pipeline generates ~300+ engineered features:

| Feature Type | Example | Purpose |
|-------------|---------|---------|
| Previous value | `HR_prev` | Track trend direction |
| Delta | `HR_delta` | Rate of change |
| % change | `HR_pct_change` | Normalised trend |
| 3h rolling mean/min/max/std | `HR_roll3_mean` | Short-term trend |
| 6h rolling mean | `HR_roll6_mean` | Medium-term trend |
| Time features | `hours_in_icu`, `time_since_prev` | Temporal context |
| Composite | `shock_index`, `sirs_score`, `lactate_wbc` | Clinical composites |

**Anti-leakage design**: All rolling operations use `.shift(1)` before windowing, so the current row's value never contributes to its own features.

---

## ⚡ Data Leakage Prevention

| Check | How it's done |
|-------|--------------|
| No patient overlap between splits | Patient-level split in `preprocessing.py` |
| No future data in features | `.shift(1)` before all rolling operations |
| No target in feature list | Explicit assertion in `train.py` |
| Threshold selection on val, not test | Threshold found on val in `train.py` |
| Imputation does not use future | Forward-fill only (past→present) |

---

## ⚠️ Limitations

1. **Single dataset**: No external validation on MIMIC-III or eICU
2. **Label definition**: Based on clinician-assigned `SepsisLabel`, which itself has limitations
3. **Hourly resolution**: Missing intermediate measurements may affect temporal features
4. **Class imbalance**: Even with `scale_pos_weight`, rare-class prediction is inherently challenging
5. **No physiological trends before ICU admission**: Pre-ICU data is unavailable

---

## 🔮 Future Work

- External validation on MIMIC-III
- Deep learning models (LSTM, Transformer) for sequence modelling
- Uncertainty quantification (conformal prediction)
- Real-time streaming integration
- Prospective clinical pilot study

---

## 📚 References

1. Reyna et al. (2019). Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019. *Critical Care Medicine*.
2. Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
3. Singer et al. (2016). The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*.

---

## 📄 License

This project uses the PhysioNet dataset which is subject to its own licensing terms. The code is provided for academic research purposes only.
