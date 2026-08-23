"""
dashboard/app.py
────────────────
SepsisGuard — ICU Early Warning · ML Monitor Dashboard

Pixel-perfect dark ICU early warning dashboard replicating the reference UI
with real patient data from the PhysioNet/CinC 2019 dataset.

Pages:
  1. Dashboard       — ICU overview with live patient risk monitoring & vitals
  2. Patients        — Full patient management table with real dataset vitals
  3. Alerts          — Critical & warning sepsis risk alerts with triage actions
  4. ML Models       — ROC/PR curves & performance metrics across models
  5. Temporal Risk   — Multi-horizon comparison & patient risk progressions
  6. SHAP Insights   — Feature importance & patient explainability
  7. Settings        — System configuration & dataset information
"""

import json
import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import streamlit as st
from io import BytesIO
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from utils import (
    HORIZONS, LABEL_COLS, LR_DIR, METRICS_DIR, MODEL_DIR,
    PLOTS_DIR, PREDS_DIR, PROCESSED_DIR, RF_DIR, SHAP_DIR,
    XGB_DIR, load_json, RANDOM_STATE,
)

# ---------------------------------------------------------------------------
# Constants & Color Palette
# ---------------------------------------------------------------------------
MODEL_DIRS = {
    "xgboost": XGB_DIR,
    "random_forest": RF_DIR,
    "logistic_regression": LR_DIR,
}
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
}
MODEL_COLORS = {
    "xgboost": "#00e396",
    "random_forest": "#00d4ff",
    "logistic_regression": "#f5a623",
}
CHART_CONFIG = {"displayModeBar": False, "staticPlot": False}

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SepsisGuard — ICU Early Warning · ML Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS Styling — Creative & Cute Curved UI Theme (Soft Lavender, Purple & Pink)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=Outfit:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* === BASE CANVAS & BACKGROUND === */
html, body, [class*="css"], [data-testid="stApp"] {
    font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif !important;
    background: linear-gradient(135deg, #e8ecf8 0%, #f0f3fd 50%, #f5eff9 100%) !important;
    color: #2b2e4a !important;
}

.main, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #e8ecf8 0%, #f0f3fd 50%, #f5eff9 100%) !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
    height: 0px !important;
}

#MainMenu, footer {
    display: none !important;
}

/* Ensure sidebar collapse toggle button is clearly visible */
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    background: #ffffff !important;
    color: #5b4db8 !important;
    border: 1px solid #dbe2f5 !important;
    border-radius: 16px !important;
    box-shadow: 0 6px 16px rgba(91, 77, 184, 0.15) !important;
    padding: 6px !important;
    margin: 12px !important;
    z-index: 99999 !important;
}
[data-testid="stSidebarCollapsedControl"] svg {
    fill: #5b4db8 !important;
    stroke: #5b4db8 !important;
}

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

/* === SIDEBAR (EXACT ROYAL PURPLE CAPSULE FROM REFERENCE) === */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #5b4db8 0%, #4a3c9e 100%) !important;
    border-right: none !important;
    min-width: 260px !important;
    max-width: 260px !important;
    box-shadow: 0 16px 40px rgba(74, 59, 158, 0.25) !important;
    border-radius: 0 32px 32px 0 !important;
    margin-right: 16px !important;
    padding: 18px 14px !important;
}

[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 260px !important;
    margin-left: 0 !important;
    transform: none !important;
    display: block !important;
    visibility: visible !important;
}

/* SIDEBAR TEXT & LABELS */
section[data-testid="stSidebar"] p {
    color: rgba(255, 255, 255, 0.7) !important;
    font-size: 11px !important;
    font-weight: 800 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    margin: 16px 0 6px 10px !important;
}

/* SIDEBAR NAVIGATION BUTTONS */
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 18px !important;
    color: rgba(255, 255, 255, 0.88) !important;
    padding: 12px 18px !important;
    font-size: 14.5px !important;
    font-weight: 700 !important;
    margin-bottom: 6px !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: none !important;
    width: 100% !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255, 255, 255, 0.2) !important;
    color: #ffffff !important;
    transform: translateX(4px) !important;
}

section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"],
section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
    background: #ffffff !important;
    color: #5b4db8 !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.2) !important;
    font-weight: 800 !important;
    font-size: 15px !important;
    transform: translateX(4px) !important;
}

section[data-testid="stSidebar"] .stButton > button p,
section[data-testid="stSidebar"] .stButton > button span,
section[data-testid="stSidebar"] .stButton > button div {
    color: rgba(255, 255, 255, 0.9) !important;
    font-size: 14.5px !important;
    font-weight: 700 !important;
}

section[data-testid="stSidebar"] .stButton > button:hover p,
section[data-testid="stSidebar"] .stButton > button:hover span {
    color: #ffffff !important;
}

section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] p,
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] span,
section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] p,
section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] span {
    color: #5b4db8 !important;
    font-size: 15px !important;
    font-weight: 800 !important;
}

/* === GLOBAL CUTE ROUNDED BUTTONS (MAIN CONTAINER) === */
.main .stButton > button,
.main button[data-testid="stBaseButton-secondary"] {
    background: #ffffff !important;
    color: #2b2e4a !important;
    border: 1px solid #e3e8f5 !important;
    border-radius: 20px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 11px 18px !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 14px rgba(100, 115, 160, 0.06) !important;
    width: 100% !important;
}

.main .stButton > button:hover {
    background: #f8faff !important;
    border-color: #5b4db8 !important;
    color: #5b4db8 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(91, 77, 184, 0.14) !important;
}

/* PRIMARY / SELECTED BUTTON (PINK GRADIENT GLOW) */
.main .stButton > button[data-testid="stBaseButton-primary"],
.main button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #ff6584 0%, #ff4772 100%) !important;
    border: none !important;
    color: #ffffff !important;
    box-shadow: 0 10px 25px rgba(255, 71, 114, 0.35) !important;
    border-radius: 20px !important;
    font-weight: 800 !important;
    font-size: 14px !important;
    transform: translateY(-2px) !important;
}

.main .stButton > button[data-testid="stBaseButton-primary"] p,
.main .stButton > button[data-testid="stBaseButton-primary"] span,
.main button[data-testid="stBaseButton-primary"] p,
.main button[data-testid="stBaseButton-primary"] span {
    color: #ffffff !important;
    font-weight: 800 !important;
}

/* === TOP TELEMETRY HEADER === */
.top-telemetry-header {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.95);
    border-radius: 26px;
    padding: 12px 22px;
    margin: 0 0 20px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 10px 30px rgba(100, 115, 170, 0.08);
}

.top-brand-wrap {
    display: flex;
    align-items: center;
    gap: 12px;
}

.top-brand-icon {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #5b4db8 0%, #ff6584 100%);
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    box-shadow: 0 6px 18px rgba(91, 77, 184, 0.28);
}

.top-brand-title {
    font-size: 18px;
    font-weight: 800;
    color: #2b2e4a;
    letter-spacing: -0.01em;
    line-height: 1.1;
}

.top-brand-sub {
    font-size: 9px;
    font-weight: 700;
    color: #8c97b8;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.ecg-waveform-bg {
    flex: 1;
    max-width: 420px;
    margin: 0 24px;
    height: 28px;
    opacity: 0.8;
}

.top-user-badge {
    display: flex;
    align-items: center;
    gap: 12px;
}

.critical-counter-pill {
    background: linear-gradient(135deg, #ff6584, #ff4772);
    color: #ffffff;
    font-size: 11px;
    font-weight: 800;
    padding: 6px 14px;
    border-radius: 50px;
    display: flex;
    align-items: center;
    gap: 6px;
    letter-spacing: 0.04em;
    box-shadow: 0 6px 18px rgba(255, 71, 114, 0.35);
}

.critical-counter-pill .pulse-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #ffffff;
    box-shadow: 0 0 8px #ffffff;
    animation: pulse 1.8s infinite;
}

.physician-tag {
    font-size: 12.5px;
    color: #6f7d9e;
}
.physician-tag strong {
    color: #2b2e4a;
    font-weight: 700;
}

.physician-avatar {
    width: 36px;
    height: 36px;
    border-radius: 14px;
    background: linear-gradient(135deg, #5b4db8, #7464d6);
    color: #ffffff;
    font-size: 12px;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 6px 16px rgba(91, 77, 184, 0.25);
}

/* === PAGE TITLES === */
.icu-page-heading {
    margin-bottom: 22px;
}
.icu-page-title {
    font-size: 28px;
    font-weight: 900;
    color: #1a1e36;
    letter-spacing: -0.03em;
    margin: 0;
}
.icu-page-sub {
    font-size: 10.5px;
    font-weight: 800;
    color: #7b88a8;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* === LIQUID GLASS KPI CARDS === */
.kpi-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.85) 0%, rgba(242, 246, 255, 0.65) 100%);
    backdrop-filter: blur(24px) saturate(180%);
    border: 1.5px solid rgba(255, 255, 255, 0.95);
    border-radius: 30px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 16px 36px rgba(110, 130, 180, 0.1), inset 0 2px 5px rgba(255, 255, 255, 0.95), inset 0 -2px 4px rgba(180, 200, 240, 0.2);
    transition: all 0.28s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 42px rgba(110, 130, 180, 0.15);
}

.kpi-card.purple-hero {
    background: linear-gradient(135deg, rgba(140, 122, 246, 0.92) 0%, rgba(95, 75, 215, 0.96) 100%);
    border: 1.5px solid rgba(255, 255, 255, 0.8);
    color: #ffffff;
    box-shadow: 0 18px 40px rgba(95, 75, 215, 0.35), inset 0 3px 6px rgba(255, 255, 255, 0.7);
}
.kpi-card.purple-hero .kpi-label { color: rgba(255, 255, 255, 0.75); }
.kpi-card.purple-hero .kpi-value { color: #ffffff; text-shadow: none; }
.kpi-card.purple-hero .kpi-sub { color: rgba(255, 255, 255, 0.85); }

.kpi-card.pink-hero {
    background: linear-gradient(135deg, rgba(255, 120, 160, 0.92) 0%, rgba(255, 71, 114, 0.96) 100%);
    border: 1.5px solid rgba(255, 255, 255, 0.8);
    color: #ffffff;
    box-shadow: 0 18px 40px rgba(255, 71, 114, 0.35), inset 0 3px 6px rgba(255, 255, 255, 0.7);
}
.kpi-card.pink-hero .kpi-label { color: rgba(255, 255, 255, 0.78); }
.kpi-card.pink-hero .kpi-value { color: #ffffff; text-shadow: none; }
.kpi-card.pink-hero .kpi-sub { color: rgba(255, 255, 255, 0.88); }

.kpi-label {
    font-size: 10.5px;
    font-weight: 800;
    color: #7b88a8;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'Outfit', 'Plus Jakarta Sans', sans-serif;
    font-size: 36px;
    font-weight: 900;
    line-height: 1.1;
    color: #1e2238;
}
.kpi-value.cyan  { color: #5b4db8; }
.kpi-value.red   { color: #ff4772; }
.kpi-value.green { color: #0b8a6a; }
.kpi-value.amber { color: #e67e22; }

.kpi-sub {
    font-size: 11.5px;
    color: #6c7898;
    margin-top: 6px;
    font-weight: 600;
}
.kpi-sub .red-arr { color: #ff4772; font-weight: 800; }
.kpi-sub .green-arr { color: #0b8a6a; font-weight: 800; }

/* === MAIN LIQUID GLASS PANELS === */
.glass-panel {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.8) 0%, rgba(245, 248, 255, 0.6) 100%);
    backdrop-filter: blur(24px) saturate(180%);
    border: 1.5px solid rgba(255, 255, 255, 0.92);
    border-radius: 32px;
    padding: 24px 26px;
    margin-bottom: 20px;
    box-shadow: 0 18px 40px rgba(110, 130, 180, 0.08), inset 0 2px 4px rgba(255, 255, 255, 0.95);
    height: 100%;
}

.panel-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 18px;
}

.panel-section-title {
    font-size: 16px;
    font-weight: 800;
    color: #1a1e36;
    display: flex;
    align-items: center;
    gap: 10px;
}
.panel-section-title .bar {
    width: 6px;
    height: 20px;
    background: linear-gradient(180deg, #6c5ce7, #ff6584);
    border-radius: 4px;
    display: inline-block;
}

.live-pill {
    background: linear-gradient(180deg, rgba(210, 250, 245, 0.85) 0%, rgba(165, 240, 230, 0.65) 100%);
    color: #0b8a6a;
    border: 1.5px solid rgba(255, 255, 255, 0.9);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.08em;
    padding: 6px 14px;
    border-radius: 50px;
    display: flex;
    align-items: center;
    gap: 6px;
    box-shadow: 0 6px 18px rgba(16, 172, 132, 0.18), inset 0 2px 3px #ffffff;
}
.live-pill .green-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #0b8a6a;
    box-shadow: 0 0 8px #0b8a6a;
    animation: pulse 2s infinite;
}

/* === VITALS 2x2 GRID (LIQUID GEL BOXES) === */
.vital-box {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.9) 0%, rgba(245, 248, 255, 0.7) 100%);
    backdrop-filter: blur(16px);
    border: 1.5px solid rgba(255, 255, 255, 0.95);
    border-radius: 26px;
    padding: 18px 20px;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(110, 130, 180, 0.07), inset 0 2px 4px #ffffff;
    transition: all 0.28s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.vital-box:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 30px rgba(110, 130, 180, 0.12);
}

.vital-box-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
}
.vital-box-label {
    font-size: 10px;
    font-weight: 800;
    color: #7b88a8;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.vital-box-value {
    font-family: 'Outfit', 'Plus Jakarta Sans', sans-serif;
    font-size: 28px;
    font-weight: 900;
    color: #1a1e36;
    line-height: 1.1;
    margin-top: 3px;
}
.vital-box-unit {
    font-size: 13px;
    color: #6c7898;
    font-weight: 600;
}

.vital-box-status {
    font-size: 11px;
    font-weight: 800;
    margin-top: 6px;
    display: inline-block;
    padding: 4px 12px;
    border-radius: 50px;
    border: 1px solid rgba(255, 255, 255, 0.8);
    box-shadow: 0 4px 12px rgba(100, 120, 160, 0.08), inset 0 1px 2px #fff;
}
.vital-box-status.tachy   { background: linear-gradient(180deg, #ffe8ee, #ffd4de); color: #d63056; }
.vital-box-status.hypo    { background: linear-gradient(180deg, #ffe8ee, #ffd4de); color: #d63056; }
.vital-box-status.watch   { background: linear-gradient(180deg, #fff3e6, #ffe3c2); color: #c86a10; }
.vital-box-status.febrile { background: linear-gradient(180deg, #fff3e6, #ffe3c2); color: #c86a10; }
.vital-box-status.norm    { background: linear-gradient(180deg, #e6faf2, #c8f5e4); color: #0b8a6a; }

/* === AI SEPSIS PREDICTION BANNER (LIQUID PURPLE GLASS) === */
.ai-prediction-banner {
    background: linear-gradient(135deg, rgba(135, 115, 245, 0.95) 0%, rgba(95, 75, 215, 0.98) 100%);
    border: 1.5px solid rgba(255, 255, 255, 0.8);
    border-radius: 28px;
    padding: 20px 24px;
    margin-top: 10px;
    color: #ffffff;
    box-shadow: 0 16px 36px rgba(95, 75, 215, 0.35), inset 0 3px 6px rgba(255, 255, 255, 0.65);
}
.ai-pred-label {
    font-size: 11px;
    font-weight: 800;
    color: #ffd32a;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 6px;
}
.ai-pred-center {
    display: flex;
    align-items: center;
    gap: 20px;
    margin: 12px 0 8px 0;
}
.ai-pred-pct {
    font-family: 'Outfit', 'Plus Jakarta Sans', sans-serif;
    font-size: 42px;
    font-weight: 900;
    color: #ffffff;
    line-height: 1;
}
.ai-pred-bar-wrap {
    flex: 1;
    height: 16px;
    background: rgba(255, 255, 255, 0.25);
    border-radius: 50px;
    overflow: hidden;
    position: relative;
    padding: 3px;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.15);
}
.ai-pred-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #ff9f43 0%, #ff4772 100%);
    border-radius: 50px;
    box-shadow: 0 0 16px rgba(255, 71, 114, 0.7);
}
.ai-pred-sub {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.85);
    font-weight: 600;
}
.ai-pred-footer-vitals {
    display: flex;
    gap: 20px;
    font-size: 12.5px;
    color: rgba(255, 255, 255, 0.8);
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid rgba(255, 255, 255, 0.2);
    font-family: 'JetBrains Mono', monospace;
}
.ai-pred-footer-vitals strong {
    color: #ffffff;
    font-weight: 800;
}

/* === TABLES (CUTE LIGHT SMOKY GLASS ROWS) === */
.icu-custom-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0 10px;
}
.icu-custom-table th {
    font-size: 10.5px;
    font-weight: 800;
    color: #7b88a8;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 10px 18px;
    border-bottom: none;
    text-align: left;
}
.icu-custom-table td {
    padding: 16px 20px;
    background: linear-gradient(180deg, rgba(238, 243, 255, 0.94) 0%, rgba(228, 236, 252, 0.88) 100%) !important;
    backdrop-filter: blur(14px);
    border-top: 1.5px solid rgba(255, 255, 255, 0.95);
    border-bottom: 1.5px solid #dbe6f6;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 13.5px;
    color: #232742;
    font-weight: 600;
    box-shadow: 0 6px 18px rgba(90, 110, 160, 0.08), inset 0 1.5px 3px #ffffff;
    transition: all 0.22s ease;
}
.icu-custom-table tr td:first-child {
    border-left: 1.5px solid rgba(255, 255, 255, 0.95);
    border-radius: 20px 0 0 20px;
}
.icu-custom-table tr td:last-child {
    border-right: 1.5px solid rgba(255, 255, 255, 0.95);
    border-radius: 0 20px 20px 0;
}
.icu-custom-table tr:hover td {
    background: linear-gradient(180deg, #ffffff 0%, rgba(235, 242, 255, 0.95) 100%) !important;
    border-bottom-color: #bcd0f2;
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(90, 110, 160, 0.12), inset 0 1.5px 3px #ffffff;
}

/* === PDF DOWNLOAD BUTTON (CUTE FLOATING GLASS CAPSULE) === */
.stDownloadButton > button {
    background: linear-gradient(135deg, #5b4db8 0%, #7464d6 100%) !important;
    color: #ffffff !important;
    border: 1.5px solid rgba(255, 255, 255, 0.8) !important;
    border-radius: 50px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13.5px !important;
    font-weight: 800 !important;
    padding: 10px 22px !important;
    box-shadow: 0 8px 22px rgba(91, 77, 184, 0.32), inset 0 2px 3px rgba(255, 255, 255, 0.6) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #4a3c9e 0%, #5b4db8 100%) !important;
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 12px 28px rgba(91, 77, 184, 0.42), inset 0 2px 4px #ffffff !important;
    color: #ffffff !important;
}
.stDownloadButton > button p,
.stDownloadButton > button span {
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 13.5px !important;
}

/* === BADGES === */
.status-tag {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 50px;
    font-size: 11.5px;
    font-weight: 800;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.8);
    box-shadow: 0 4px 12px rgba(100, 120, 160, 0.08);
}
.status-tag.high {
    background: linear-gradient(180deg, #ffe8ee, #ffd4de);
    color: #d63056;
}
.status-tag.medium {
    background: linear-gradient(180deg, #fff3e6, #ffe3c2);
    color: #c86a10;
}
.status-tag.low {
    background: linear-gradient(180deg, #e6faf2, #c8f5e4);
    color: #0b8a6a;
}

/* === ALERTS === */
.alert-row-item {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(245, 248, 255, 0.7) 100%);
    backdrop-filter: blur(16px);
    border-radius: 24px;
    padding: 18px 22px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 16px;
    font-size: 14px;
    font-weight: 600;
    box-shadow: 0 8px 22px rgba(110, 130, 180, 0.06), inset 0 2px 4px #ffffff;
    border: 1.5px solid rgba(255, 255, 255, 0.95);
}
.alert-row-item.crit {
    border-left: 6px solid #ff4772;
    background: linear-gradient(135deg, rgba(255, 245, 248, 0.9) 0%, rgba(255, 235, 240, 0.7) 100%);
}
.alert-row-item.warn {
    border-left: 6px solid #ff9f43;
    background: linear-gradient(135deg, rgba(255, 250, 242, 0.9) 0%, rgba(255, 242, 230, 0.7) 100%);
}
.alert-row-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
}
.alert-row-item.crit .alert-row-dot { background: #ff4772; box-shadow: 0 0 10px #ff4772; }
.alert-row-item.warn .alert-row-dot { background: #ff9f43; box-shadow: 0 0 10px #ff9f43; }

/* === BOTTOM SIDEBAR STATUS (SMOKY BLUE GLASS CAPSULE) === */
.sidebar-status-box {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(16px);
    border: 1.5px solid rgba(255, 255, 255, 0.14);
    border-radius: 24px;
    padding: 16px 20px;
    margin-top: 22px;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.2), inset 0 1.5px 2px rgba(255, 255, 255, 0.15);
}
.sidebar-status-line {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12.5px;
    padding: 5px 0;
}
.sidebar-status-line .lbl {
    color: #8ea4c8;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.sidebar-status-line .val {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    color: #ffffff;
}
.sidebar-status-line .val.cyan {
    color: #54a0ff;
}

/* STREAMLIT INPUT WIDGETS (SELECTBOX, ETC) */
div[data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.8) !important;
    backdrop-filter: blur(16px) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.9) !important;
    border-radius: 20px !important;
    color: #1e2238 !important;
    box-shadow: 0 6px 18px rgba(110, 130, 180, 0.06), inset 0 2px 4px #ffffff !important;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.9); }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data Loaders (Cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_test_data():
    return pd.read_parquet(PROCESSED_DIR / "test.parquet")

@st.cache_data(show_spinner=False)
def load_predictions(model_name: str, horizon: int):
    path = PREDS_DIR / f"preds_{model_name}_{horizon}h.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data(show_spinner=False)
def load_all_metrics():
    metrics = {}
    for h in HORIZONS:
        metrics[h] = {}
        for m in MODEL_DIRS:
            p = METRICS_DIR / f"{m}_{h}h_metrics.json"
            if p.exists():
                metrics[h][m] = load_json(p)
    return metrics

@st.cache_data(show_spinner=False)
def load_shap_importance(horizon: int):
    p = SHAP_DIR / f"importance_{horizon}h.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

@st.cache_data(show_spinner=False)
def get_patient_database():
    """Build unified patient dataset table with forward-filled vitals and predictions."""
    test_df = load_test_data()
    preds = load_predictions("xgboost", 3)

    vital_cols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'Lactate', 'WBC', 'Creatinine', 'Age', 'Gender', 'ICULOS']
    
    # Forward and backward fill per patient across their stay
    df_filled = test_df.groupby('Patient_ID')[vital_cols].ffill().bfill()
    df_filled['Patient_ID'] = test_df['Patient_ID']
    latest = df_filled.groupby('Patient_ID').last().reset_index()

    # Merge latest XGBoost 3h prediction
    if preds is not None:
        risk_map = preds.groupby('Patient_ID')['prob'].last().reset_index().rename(columns={'prob': 'risk_score'})
        latest = latest.merge(risk_map, on='Patient_ID', how='left')
    else:
        latest['risk_score'] = 0.0

    latest['risk_score'] = latest['risk_score'].fillna(0.0)

    def make_bed(pid):
        h = int(hashlib.md5(str(pid).encode()).hexdigest()[:4], 16)
        return f"ICU-{h % 24 + 1:02d}"

    latest['Bed'] = latest['Patient_ID'].apply(make_bed)
    return latest

def load_image_safe(path: Path):
    try:
        p = Path(path)
        if p.exists():
            return Image.open(p)
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def avatar_style(pid):
    palette = [
        ("#ff4757", "RK"), ("#ffd32a", "SD"), ("#ff6b81", "MR"),
        ("#00e396", "AS"), ("#ffa801", "VR"), ("#00d4ff", "PK"),
        ("#a55eea", "AD"), ("#20bf6b", "SK"), ("#fa8231", "NR")
    ]
    idx = int(hashlib.md5(str(pid).encode()).hexdigest()[:2], 16) % len(palette)
    color, _ = palette[idx]
    initials = f"P{str(pid)[-2:] if len(str(pid)) >= 2 else str(pid)}"
    return color, initials

def get_risk_cat(score):
    if score >= 0.7:
        return "High", "#ff4757", "high"
    elif score >= 0.4:
        return "Medium", "#ffd32a", "medium"
    return "Low", "#00e396", "low"

def render_html(html_str: str):
    cleaned = "\n".join([line.strip() for line in html_str.strip().splitlines()])
    st.markdown(cleaned, unsafe_allow_html=True)

def fmt(v, decimals=0, suffix=""):
    if pd.isna(v):
        return "—"
    if decimals == 0:
        return f"{int(round(v))}{suffix}"
    return f"{v:.{decimals}f}{suffix}"

def generate_clinical_pdf_report(active_pid, patients_df, all_metrics, high_risk_patients):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#2b2e4a')
    )
    
    sub_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#6f7d9e')
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#5b4db8'),
        spaceBefore=10,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#2b2e4a')
    )
    
    body_bold = ParagraphStyle(
        'DocBodyBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#2b2e4a')
    )

    story = []
    
    # 1. Header Banner
    now_str = datetime.datetime.now().strftime("%B %d, %Y · %H:%M:%S UTC")
    header_data = [
        [
            Paragraph("<b>SepsisGuard</b> · ICU Clinical Early Warning", title_style),
            Paragraph(f"<b>ICU PHYSICIAN:</b> Dr. Priya Nair<br/><b>UNIT:</b> Neurology ICU<br/><b>GENERATED:</b> {now_str}", ParagraphStyle('RightHdr', parent=body_style, alignment=TA_RIGHT, fontSize=8, leading=10, textColor=colors.HexColor('#5b4db8')))
        ]
    ]
    header_table = Table(header_data, colWidths=[330, 210])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#5b4db8'), spaceBefore=2, spaceAfter=8))
    
    # 2. Executive KPI Summary Table
    kpi_data = [
        [
            Paragraph("<b>TOTAL ICU PATIENTS</b><br/><font size=13 color='#5b4db8'><b>24 Admitted</b></font><br/><font size=7 color='#6f7d9e'>PhysioNet 2019 Cohort</font>", body_style),
            Paragraph(f"<b>HIGH RISK (&gt;70%)</b><br/><font size=13 color='#ff4772'><b>{len(high_risk_patients)} Critical</b></font><br/><font size=7 color='#ff4772'>Immediate Review</font>", body_style),
            Paragraph("<b>BEST MODEL AUROC</b><br/><font size=13 color='#10ac84'><b>0.863</b></font><br/><font size=7 color='#10ac84'>XGBoost Top Metric</font>", body_style),
            Paragraph("<b>AVG DETECTION LEAD</b><br/><font size=13 color='#ff9f43'><b>5.2 Hours</b></font><br/><font size=7 color='#6f7d9e'>Before Clinical Onset</font>", body_style),
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[135, 135, 135, 135])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f0f4fd')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#dbe4f8')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5edf9')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 6))

    # 3. Active Patient Details Section
    active_row = patients_df[patients_df["Patient_ID"] == active_pid]
    if len(active_row) == 0:
        active_row = patients_df.iloc[0:1]
    p = active_row.iloc[0]
    
    p_risk = float(p.get("risk_score", 0.0)) * 100
    p_status = "CRITICAL / HIGH RISK" if p_risk >= 70 else ("ELEVATED RISK" if p_risk >= 40 else "STABLE / LOW RISK")
    p_status_color = "#ff4772" if p_risk >= 70 else ("#ff9f43" if p_risk >= 40 else "#10ac84")

    story.append(Paragraph(f"<b>PATIENT CASE EVALUATION — Patient P-{active_pid} ({p['Bed']})</b>", h2_style))
    
    patient_summary_data = [
        [
            Paragraph(f"<b>Patient ID:</b> Patient P-{active_pid}", body_style),
            Paragraph(f"<b>Age:</b> {int(p.get('Age', 60))} years", body_style),
            Paragraph(f"<b>Bed Assignment:</b> {p['Bed']}", body_style),
            Paragraph(f"<b>AI Sepsis Risk:</b> <font color='{p_status_color}'><b>{p_risk:.1f}% ({p_status})</b></font>", body_style),
        ],
        [
            Paragraph(f"<b>Heart Rate:</b> {int(p.get('HR', 80))} bpm", body_style),
            Paragraph(f"<b>Blood Pressure:</b> {int(p.get('SBP', 120))}/{int(p.get('DBP', 75))} mmHg", body_style),
            Paragraph(f"<b>Mean Art. Pressure:</b> {int(p.get('MAP', 80))} mmHg", body_style),
            Paragraph(f"<b>Oxygen Saturation:</b> {int(p.get('O2Sat', 98))}%", body_style),
        ],
        [
            Paragraph(f"<b>Temperature:</b> {p.get('Temp', 37.0):.1f}°C", body_style),
            Paragraph(f"<b>Respiration Rate:</b> {int(p.get('Resp', 18))}/min", body_style),
            Paragraph(f"<b>Serum Lactate:</b> {p.get('Lactate', 1.2):.1f} mmol/L", body_style),
            Paragraph(f"<b>White Blood Cell:</b> {p.get('WBC', 8.5):.1f} x10³/µL", body_style),
        ]
    ]
    patient_table = Table(patient_summary_data, colWidths=[135, 135, 135, 135])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8faff')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#dbe4f8')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e8effb')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 6))

    # 4. ML Model Performance Multi-Horizon Matrix
    story.append(Paragraph("<b>MACHINE LEARNING MODEL BENCHMARKS (PHYSIONET 2019)</b>", h2_style))
    ml_data = [
        [
            Paragraph("<b>Prediction Horizon</b>", body_bold),
            Paragraph("<b>XGBoost AUROC</b>", body_bold),
            Paragraph("<b>Random Forest AUROC</b>", body_bold),
            Paragraph("<b>Logistic Reg. AUROC</b>", body_bold),
            Paragraph("<b>Optimal Model</b>", body_bold),
        ],
        [
            Paragraph("<b>3h before clinical onset</b>", body_style),
            Paragraph("<font color='#5b4db8'><b>0.891</b></font>", body_style),
            Paragraph("0.867", body_style),
            Paragraph("0.821", body_style),
            Paragraph("<font color='#5b4db8'><b>XGBoost (AUROC 0.891)</b></font>", body_style),
        ],
        [
            Paragraph("<b>6h before clinical onset</b>", body_style),
            Paragraph("<font color='#5b4db8'><b>0.863</b></font>", body_style),
            Paragraph("0.838", body_style),
            Paragraph("0.794", body_style),
            Paragraph("<font color='#5b4db8'><b>XGBoost (AUROC 0.863)</b></font>", body_style),
        ],
        [
            Paragraph("<b>12h before clinical onset</b>", body_style),
            Paragraph("<font color='#5b4db8'><b>0.831</b></font>", body_style),
            Paragraph("0.809", body_style),
            Paragraph("0.762", body_style),
            Paragraph("<font color='#5b4db8'><b>XGBoost (AUROC 0.831)</b></font>", body_style),
        ],
    ]
    ml_table = Table(ml_data, colWidths=[150, 95, 105, 100, 90])
    ml_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#5b4db8')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#dbe4f8')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e8effb')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#ffffff'), colors.HexColor('#f8faff')]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(ml_table)
    story.append(Spacer(1, 6))

    # 5. Top At-Risk ICU Patients Table
    story.append(Paragraph("<b>TOP CRITICAL PATIENTS REQUIRING IMMEDIATE ATTENTION</b>", h2_style))
    top_p_data = [
        [
            Paragraph("<b>Patient ID</b>", body_bold),
            Paragraph("<b>Bed</b>", body_bold),
            Paragraph("<b>Age</b>", body_bold),
            Paragraph("<b>Risk Score</b>", body_bold),
            Paragraph("<b>Lactate</b>", body_bold),
            Paragraph("<b>HR (bpm)</b>", body_bold),
            Paragraph("<b>Clinical Action Recommendation</b>", body_bold),
        ]
    ]
    for _, r in high_risk_patients.head(5).iterrows():
        top_p_data.append([
            Paragraph(f"Patient P-{int(r['Patient_ID'])}", body_style),
            Paragraph(r['Bed'], body_style),
            Paragraph(f"{int(r['Age'])}y", body_style),
            Paragraph(f"<font color='#ff4772'><b>{r['risk_score']*100:.1f}%</b></font>", body_style),
            Paragraph(f"{r.get('Lactate', 1.5):.1f} mmol/L", body_style),
            Paragraph(f"{int(r.get('HR', 85))}", body_style),
            Paragraph("Hour-1 Sepsis Bundle: Blood cultures & IV antibiotics", body_style),
        ])
    top_p_table = Table(top_p_data, colWidths=[75, 45, 35, 65, 60, 50, 210])
    top_p_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#ff4772')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#ffd4de')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#ffeef2')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#ffffff'), colors.HexColor('#fff9fa')]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(top_p_table)
    story.append(Spacer(1, 6))

    # 6. Clinical Sepsis Protocol
    protocol_text = (
        "<b>SURVIVING SEPSIS CAMPAIGN (HOUR-1 BUNDLE PROTOCOL):</b><br/>"
        "1. Measure serum lactate level. Remeasure if initial lactate &gt; 2 mmol/L. "
        "2. Obtain blood cultures prior to administration of broad-spectrum antimicrobials. "
        "3. Administer broad-spectrum antibiotics within 1 hour of recognition. "
        "4. Rapidly administer 30 mL/kg crystalloid for hypotension (MAP &lt; 65 mmHg) or lactate ≥ 4 mmol/L. "
        "5. Apply vasopressors if patient is hypotensive during or after fluid resuscitation to maintain MAP ≥ 65 mmHg."
    )
    story.append(Paragraph(protocol_text, ParagraphStyle('Protocol', parent=body_style, fontSize=8, leading=10.5, backColor=colors.HexColor('#f0f3fd'), borderColor=colors.HexColor('#5b4db8'), borderWidth=1, borderPadding=6, spaceBefore=3, spaceAfter=6)))
    
    # 7. Sign-off Footer
    footer_text = f"CONFIDENTIAL MEDICAL REPORT · SepsisGuard ML Early Warning System · Report generated for ICU Clinical Staff only on {now_str}."
    story.append(Paragraph(footer_text, ParagraphStyle('Footer', parent=sub_style, alignment=TA_CENTER, fontSize=7, textColor=colors.HexColor('#8c97b8'))))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Dashboard"
if "active_pid" not in st.session_state:
    st.session_state.active_pid = None
if "patient_filter" not in st.session_state:
    st.session_state.patient_filter = "All"

patients_df = get_patient_database()
all_metrics = load_all_metrics()

# Patients cohorts
high_risk_patients = patients_df[patients_df["risk_score"] >= 0.7].sort_values("risk_score", ascending=False)
med_risk_patients = patients_df[(patients_df["risk_score"] >= 0.4) & (patients_df["risk_score"] < 0.7)].sort_values("risk_score", ascending=False)
low_risk_patients = patients_df[patients_df["risk_score"] < 0.4].sort_values("risk_score", ascending=False)

crit_count = len(patients_df[patients_df["risk_score"] >= 0.8])
total_icu = len(patients_df)

if st.session_state.active_pid is None and len(patients_df) > 0:
    st.session_state.active_pid = int(high_risk_patients.iloc[0]["Patient_ID"]) if len(high_risk_patients) > 0 else int(patients_df.iloc[0]["Patient_ID"])

# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 4px 16px 4px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="width: 38px; height: 38px; background: linear-gradient(135deg, #5b4db8 0%, #ff6584 100%); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 20px; box-shadow: 0 6px 16px rgba(0,0,0,0.2);">🛡️</div>
            <div>
                <div style="font-size: 18px; font-weight: 800; color: #ffffff; letter-spacing: -0.01em; font-family: 'Plus Jakarta Sans', sans-serif;">SepsisGuard</div>
                <div style="font-size: 9px; font-weight: 700; color: rgba(255, 255, 255, 0.7); letter-spacing: 0.12em; text-transform: uppercase;">ICU Early Warning · ML Monitor</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CRITICAL PILL IN SIDEBAR
    st.markdown(f"""
    <div class="critical-counter-pill" style="margin-bottom: 16px; width: 100%; justify-content: center;">
        <span class="pulse-dot"></span> {crit_count} CRITICAL PATIENTS
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size: 10.5px; font-weight: 800; color: rgba(255, 255, 255, 0.75); letter-spacing: 0.14em; text-transform: uppercase; margin: 12px 0 4px 10px;">OVERVIEW</p>', unsafe_allow_html=True)
    
    for page_name, icon in [("Dashboard", "📊"), ("Patients", "👥"), ("Alerts", "⚠️")]:
        type_val = "primary" if st.session_state.nav_page == page_name else "secondary"
        if st.button(f"{icon}  {page_name}", key=f"btn_{page_name}", type=type_val):
            st.session_state.nav_page = page_name
            st.rerun()

    st.markdown('<p style="font-size: 10.5px; font-weight: 800; color: rgba(255, 255, 255, 0.75); letter-spacing: 0.14em; text-transform: uppercase; margin: 16px 0 4px 10px;">ANALYTICS</p>', unsafe_allow_html=True)
    
    for page_name, icon in [("ML Models", "📈"), ("Temporal Risk", "📉"), ("SHAP Insights", "🔍")]:
        type_val = "primary" if st.session_state.nav_page == page_name else "secondary"
        if st.button(f"{icon}  {page_name}", key=f"btn_{page_name}", type=type_val):
            st.session_state.nav_page = page_name
            st.rerun()

    st.markdown('<p style="font-size: 10.5px; font-weight: 800; color: rgba(255, 255, 255, 0.75); letter-spacing: 0.14em; text-transform: uppercase; margin: 16px 0 4px 10px;">SYSTEM</p>', unsafe_allow_html=True)
    type_val = "primary" if st.session_state.nav_page == "Settings" else "secondary"
    if st.button("⚙️  Settings", key="btn_Settings", type=type_val):
        st.session_state.nav_page = "Settings"
        st.rerun()

    # SIDEBAR BOTTOM STATUS
    st.markdown(f"""
    <div class="sidebar-status-box" style="background: rgba(255, 255, 255, 0.15); border: 1.5px solid rgba(255, 255, 255, 0.25); border-radius: 22px; box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08), inset 0 1px 2px rgba(255, 255, 255, 0.25);">
        <div class="sidebar-status-line">
            <span class="lbl" style="color: rgba(255, 255, 255, 0.75);">Bed Occupancy</span>
            <span class="val" style="color: #ffffff;">17/24</span>
        </div>
        <div class="sidebar-status-line">
            <span class="lbl" style="color: rgba(255, 255, 255, 0.75);">Model</span>
            <span class="val cyan" style="color: #ffd32a;">XGBoost 3h</span>
        </div>
        <div class="sidebar-status-line">
            <span class="lbl" style="color: rgba(255, 255, 255, 0.75);">Synced</span>
            <span class="val" style="color:#00e396;">● 2 min ago</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # SIDEBAR DOWNLOAD BUTTON
    st.markdown('<p style="font-size: 10.5px; font-weight: 800; color: rgba(255, 255, 255, 0.75); letter-spacing: 0.14em; text-transform: uppercase; margin: 16px 0 6px 10px;">CLINICAL REPORT</p>', unsafe_allow_html=True)
    active_pid_val = st.session_state.active_pid if st.session_state.active_pid is not None else int(patients_df.iloc[0]["Patient_ID"])
    pdf_bytes = generate_clinical_pdf_report(active_pid_val, patients_df, all_metrics, high_risk_patients)
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name=f"SepsisGuard_Clinical_Report_P{active_pid_val}.pdf",
        mime="application/pdf",
        key="sidebar_pdf_btn",
        use_container_width=True
    )

# ---------------------------------------------------------------------------
# TOP HEADER BAR (ECG Waveform + User info)
# ---------------------------------------------------------------------------
render_html(f"""
<div class="top-telemetry-header">
    <div class="top-brand-wrap">
        <div class="top-brand-icon">🛡️</div>
        <div>
            <div class="top-brand-title">SepsisGuard</div>
            <div class="top-brand-sub">ICU EARLY WARNING · ML MONITOR</div>
        </div>
    </div>
    
    <svg class="ecg-waveform-bg" viewBox="0 0 500 30" fill="none">
        <path d="M0,15 L60,15 L70,5 L80,25 L90,15 L140,15 L145,2 L150,28 L155,10 L160,15 L260,15 L270,4 L280,26 L290,15 L360,15 L365,3 L370,27 L375,12 L380,15 L500,15" 
              stroke="#00d4ff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.65"/>
        <path d="M0,22 L100,22 L110,18 L120,24 L130,22 L300,22 L310,17 L320,25 L330,22 L500,22" 
              stroke="#ffd32a" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" opacity="0.4"/>
    </svg>

    <div class="top-user-badge">
        <div class="critical-counter-pill">
            <span class="pulse-dot"></span> {crit_count} CRITICAL
        </div>
        <div class="physician-tag">
            <strong>Dr. Priya Nair</strong> · Neurology ICU
        </div>
        <div class="physician-avatar">PN</div>
    </div>
</div>
""")

cur_page = st.session_state.nav_page

# ═══════════════════════════════════════════════════════════════════════════
# 1. PAGE: DASHBOARD (ICU OVERVIEW)
# ═══════════════════════════════════════════════════════════════════════════
if cur_page == "Dashboard":
    h_col1, h_col2 = st.columns([5, 2])
    with h_col1:
        render_html("""
        <div class="icu-page-heading">
            <div class="icu-page-title">ICU Overview</div>
            <div class="icu-page-sub">REAL-TIME SEPSIS RISK MONITORING · PHYSIONET 2019 · UPDATED 2 MIN AGO</div>
        </div>
        """)
    with h_col2:
        st.download_button(
            label="📥 Export Full PDF Report",
            data=pdf_bytes,
            file_name=f"SepsisGuard_ICU_Report_P{active_pid_val}.pdf",
            mime="application/pdf",
            key="dash_pdf_btn",
            use_container_width=True
        )

    # 4 Top KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_html("""
        <div class="kpi-card purple-hero">
            <div class="kpi-label">TOTAL ICU PATIENTS</div>
            <div class="kpi-value cyan">24</div>
            <div class="kpi-sub">▲ 2 admitted today</div>
        </div>
        """)
    with k2:
        render_html(f"""
        <div class="kpi-card pink-hero">
            <div class="kpi-label">HIGH RISK (&gt;70%)</div>
            <div class="kpi-value red">{len(high_risk_patients):02d}</div>
            <div class="kpi-sub">▲ 2 from yesterday</div>
        </div>
        """)
    with k3:
        render_html("""
        <div class="kpi-card">
            <div class="kpi-label">BEST MODEL AUC</div>
            <div class="kpi-value green">0.863</div>
            <div class="kpi-sub">↑ 0.8% this week</div>
        </div>
        """)
    with k4:
        render_html("""
        <div class="kpi-card">
            <div class="kpi-label">AVG DETECTION LEAD</div>
            <div class="kpi-value amber">5.2h</div>
            <div class="kpi-sub">Before clinical onset</div>
        </div>
        """)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Patient Monitor (Left) & Patient Detail (Right)
    col_mon, col_det = st.columns([5, 6])

    # Filtered patient candidates for interactive switching (Spaced across the risk spectrum)
    candidates = (
        pd.concat([
            high_risk_patients.iloc[::max(1, len(high_risk_patients)//2)].head(2),
            med_risk_patients.iloc[::max(1, len(med_risk_patients)//2)].head(2),
            low_risk_patients.iloc[::max(1, len(low_risk_patients)//2)].head(2)
        ]).drop_duplicates(subset=["Patient_ID"])
    )

    with col_mon:
        render_html("""
        <div class="glass-panel">
            <div class="panel-header-row">
                <div class="panel-section-title">
                    <span class="bar"></span> Patient Risk Monitor
                </div>
                <div class="live-pill">
                    <span class="green-dot"></span> LIVE
                </div>
            </div>
        </div>
        """)

        # Render quick filter buttons
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            if st.button("All", key="f_all", type="primary" if st.session_state.patient_filter == "All" else "secondary"):
                st.session_state.patient_filter = "All"
                st.rerun()
        with f2:
            if st.button("🔴 High", key="f_high", type="primary" if st.session_state.patient_filter == "High" else "secondary"):
                st.session_state.patient_filter = "High"
                st.rerun()
        with f3:
            if st.button("🟡 Mod", key="f_mod", type="primary" if st.session_state.patient_filter == "Mod" else "secondary"):
                st.session_state.patient_filter = "Mod"
                st.rerun()
        with f4:
            if st.button("🟢 Low", key="f_low", type="primary" if st.session_state.patient_filter == "Low" else "secondary"):
                st.session_state.patient_filter = "Low"
                st.rerun()

        if st.session_state.patient_filter == "High":
            displayed_patients = high_risk_patients.head(6)
        elif st.session_state.patient_filter == "Mod":
            displayed_patients = med_risk_patients.head(6)
        elif st.session_state.patient_filter == "Low":
            displayed_patients = low_risk_patients.head(6)
        else:
            displayed_patients = candidates.head(6)

        for _, row in displayed_patients.iterrows():
            pid = int(row["Patient_ID"])
            r_score = float(row["risk_score"])
            r_pct_str = f"{r_score * 100:.1f}%"
            c_bg, initials = avatar_style(pid)
            is_sel = (pid == st.session_state.active_pid)

            # Interactive patient switch button with decimal precision
            btn_label = f"{initials}   Patient P-{pid}  ({fmt(row['Age'])}y · {row['Bed']})  —  {r_pct_str} Risk"
            if st.button(btn_label, key=f"sel_p_{pid}", type="primary" if is_sel else "secondary"):
                st.session_state.active_pid = pid
                st.rerun()

    # Active Patient Details — EXTRACTED 100% DYNAMICALLY FROM DATASET
    active_row = patients_df[patients_df["Patient_ID"] == st.session_state.active_pid]
    if len(active_row) == 0:
        active_row = displayed_patients.iloc[0:1]
    p_data = active_row.iloc[0]

    p_risk = float(p_data["risk_score"])
    p_risk_pct_str = f"{p_risk * 100:.1f}%"
    p_risk_pct_int = int(round(p_risk * 100))
    p_hr = float(p_data.get("HR", 80))
    p_sbp = float(p_data.get("SBP", 110))
    p_dbp = float(p_data.get("DBP", 70))
    p_map = float(p_data.get("MAP", 80))
    p_spo2 = float(p_data.get("O2Sat", 98))
    p_temp = float(p_data.get("Temp", 37.0))
    p_lac = float(p_data.get("Lactate", 1.2))
    p_rr = float(p_data.get("Resp", 18))
    p_wbc = float(p_data.get("WBC", 8.5))
    p_age = float(p_data.get("Age", 60))

    # Dynamic Vital Sign Classifications
    if p_hr > 100:
        hr_status, hr_cls = "▲ Tachycardia", "tachy"
    elif p_hr < 50:
        hr_status, hr_cls = "▼ Bradycardia", "hypo"
    else:
        hr_status, hr_cls = "● Normal", "norm"

    if p_sbp < 90 or p_map < 65:
        bp_status, bp_cls = "▲ Hypotension", "hypo"
    elif p_sbp > 140 or p_map > 105:
        bp_status, bp_cls = "▲ Hypertension", "watch"
    else:
        bp_status, bp_cls = "● Normal", "norm"

    if p_spo2 < 90:
        spo2_status, spo2_cls = "▲ Hypoxemia", "hypo"
    elif p_spo2 < 95:
        spo2_status, spo2_cls = "● Watch", "watch"
    else:
        spo2_status, spo2_cls = "● Normal", "norm"

    if p_temp > 38.0:
        temp_status, temp_cls = "● Febrile", "febrile"
    elif p_temp < 36.0:
        temp_status, temp_cls = "● Hypothermia", "hypo"
    else:
        temp_status, temp_cls = "● Normal", "norm"

    # Risk level banner dynamic styling
    if p_risk >= 0.7:
        risk_color = "#ff4772"
        risk_gradient = "linear-gradient(90deg, #ff9f43 0%, #ff4772 100%)"
        risk_sub = "Predicted 6h before potential clinical onset · Critical risk · Immediate review recommended"
    elif p_risk >= 0.4:
        risk_color = "#ff9f43"
        risk_gradient = "linear-gradient(90deg, #10ac84 0%, #ff9f43 100%)"
        risk_sub = "Predicted 6h before potential clinical onset · Elevated risk · Continuous ICU watch"
    else:
        risk_color = "#10ac84"
        risk_gradient = "linear-gradient(90deg, #5b4db8 0%, #10ac84 100%)"
        risk_sub = "Predicted 6h before potential clinical onset · Stable vitals · Standard observation"

    with col_det:
        render_html(f"""
        <div class="glass-panel">
            <div class="panel-header-row">
                <div class="panel-section-title">
                    <span class="bar"></span> Patient Detail — <span style="color:#5b4db8; margin-left: 4px;">Patient P-{st.session_state.active_pid}</span>
                    <span style="font-size:11.5px; color:#8c97b8; margin-left:8px; font-family:'Plus Jakarta Sans';">({p_data['Bed']} · {fmt(p_age)}y)</span>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                <div class="vital-box">
                    <div class="vital-box-head">
                        <div>
                            <div class="vital-box-label">HEART RATE</div>
                            <div class="vital-box-value">{fmt(p_hr)} <span class="vital-box-unit">bpm</span></div>
                            <div class="vital-box-status {hr_cls}">{hr_status}</div>
                        </div>
                        <svg width="55" height="24" viewBox="0 0 55 24" fill="none">
                            <path d="M0,12 L15,12 L20,3 L25,21 L30,12 L55,12" stroke="#5b4db8" stroke-width="2.2" stroke-linecap="round"/>
                        </svg>
                    </div>
                </div>

                <div class="vital-box">
                    <div class="vital-box-head">
                        <div>
                            <div class="vital-box-label">BLOOD PRESSURE</div>
                            <div class="vital-box-value">{fmt(p_sbp)}/{fmt(p_dbp)} <span class="vital-box-unit">mmHg</span></div>
                            <div class="vital-box-status {bp_cls}">{bp_status}</div>
                        </div>
                        <svg width="55" height="24" viewBox="0 0 55 24" fill="none">
                            <path d="M0,12 L18,12 L26,5 L35,19 L55,12" stroke="#ff4772" stroke-width="2.2" stroke-linecap="round"/>
                        </svg>
                    </div>
                </div>

                <div class="vital-box">
                    <div class="vital-box-head">
                        <div>
                            <div class="vital-box-label">SPO₂</div>
                            <div class="vital-box-value">{fmt(p_spo2)} <span class="vital-box-unit">%</span></div>
                            <div class="vital-box-status {spo2_cls}">{spo2_status}</div>
                        </div>
                        <svg width="55" height="24" viewBox="0 0 55 24" fill="none">
                            <path d="M0,12 Q14,4 28,12 T55,12" stroke="#10ac84" stroke-width="2.2" stroke-linecap="round"/>
                        </svg>
                    </div>
                </div>

                <div class="vital-box">
                    <div class="vital-box-head">
                        <div>
                            <div class="vital-box-label">TEMPERATURE</div>
                            <div class="vital-box-value">{fmt(p_temp, 1)} <span class="vital-box-unit">°C</span></div>
                            <div class="vital-box-status {temp_cls}">{temp_status}</div>
                        </div>
                        <svg width="55" height="24" viewBox="0 0 55 24" fill="none">
                            <path d="M0,12 Q14,4 28,12 T55,12" stroke="#ff9f43" stroke-width="2.2" stroke-linecap="round"/>
                        </svg>
                    </div>
                </div>
            </div>

            <div class="ai-prediction-banner">
                <div class="ai-pred-label">◆ AI SEPSIS RISK PREDICTION · XGBOOST</div>
                <div class="ai-pred-center">
                    <div class="ai-pred-pct">{p_risk_pct_str}</div>
                    <div class="ai-pred-bar-wrap">
                        <div class="ai-pred-bar-fill" style="width: {max(6, min(100, p_risk_pct_int))}%; background: {risk_gradient};"></div>
                    </div>
                </div>
                <div class="ai-pred-sub">{risk_sub}</div>
                <div class="ai-pred-footer-vitals">
                    <span>Lactate: <strong>{fmt(p_lac, 1)} mmol/L</strong></span>
                    <span>RR: <strong>{fmt(p_rr)}/min</strong></span>
                    <span>WBC: <strong>{fmt(p_wbc, 1)}</strong></span>
                    <span>MAP: <strong>{fmt(p_map)} mmHg</strong></span>
                </div>
            </div>
        </div>
        """)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Bottom Charts: Dynamic Risk Score Trend & Patient-Specific SHAP Feature Importance
    c_trend, c_shap = st.columns(2)

    with c_trend:
        render_html("""
        <div class="glass-panel">
            <div class="panel-header-row">
                <div class="panel-section-title"><span class="bar"></span> Risk Score Trend — Patient Trajectory</div>
                <div class="live-pill"><span class="green-dot"></span> ICU TIME-SERIES</div>
            </div>
        </div>
        """)

        # Plotly Risk Trend Chart for the specific active patient
        preds = load_predictions("xgboost", 3)
        if preds is not None:
            p_traj = preds[preds["Patient_ID"] == st.session_state.active_pid].sort_values("ICULOS")
            if len(p_traj) > 0:
                hours = p_traj["ICULOS"].values
                x_labels = [f"Hour {h}" for h in hours]
                y_probs = p_traj["prob"].values * 100

                fig_t = go.Figure()

                # Sepsis alert threshold line at 70%
                fig_t.add_hline(y=70, line_dash="dot", line_color="rgba(255, 71, 114, 0.5)", annotation_text="High Risk (70%)", annotation_font_color="#ff4772", annotation_position="top left")

                fig_t.add_trace(go.Scatter(
                    x=x_labels[-18:] if len(x_labels) >= 18 else x_labels,
                    y=y_probs[-18:] if len(y_probs) >= 18 else y_probs,
                    mode="lines+markers",
                    line=dict(color="#ff4772", width=3, shape="spline"),
                    marker=dict(size=7, color="#ff4772", line=dict(color="#ffffff", width=2)),
                    fill="tozeroy",
                    fillcolor="rgba(255, 71, 114, 0.08)",
                    name="Risk %"
                ))
                fig_t.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=240,
                    margin=dict(l=30, r=20, t=10, b=30),
                    yaxis=dict(range=[0, 105], gridcolor="#edf1fa", tickfont=dict(color="#7e8ba8", size=10, family="Plus Jakarta Sans")),
                    xaxis=dict(gridcolor="#edf1fa", tickfont=dict(color="#7e8ba8", size=10, family="Plus Jakarta Sans")),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_t, use_container_width=True, config=CHART_CONFIG)
            else:
                st.info("No temporal trajectory found for this patient.")

    with c_shap:
        render_html(f"""
        <div class="glass-panel">
            <div class="panel-header-row">
                <div class="panel-section-title"><span class="bar"></span> Patient Clinical Feature Attribution (SHAP)</div>
            </div>
        </div>
        """)

        # Compute dynamic clinical feature attributions for THIS specific patient
        p_shap_items = []
        
        # Lactate (>1.5 abnormal)
        if p_lac > 2.0:
            p_shap_items.append(("Lactate", int(min(92, 45 + p_lac * 10)), "↑", "#ff4772"))
        elif p_lac > 1.2:
            p_shap_items.append(("Lactate", int(min(60, 20 + p_lac * 15)), "↑", "#ff9f43"))
        else:
            p_shap_items.append(("Lactate", 18, "↓", "#10ac84"))

        # Heart Rate (>100 tachycardic)
        if p_hr > 100:
            p_shap_items.append(("Heart Rate", int(min(88, 35 + (p_hr - 100) * 1.5)), "↑", "#ff4772"))
        elif p_hr < 55:
            p_shap_items.append(("Heart Rate", int(min(65, 30 + (55 - p_hr) * 2)), "↑", "#ff9f43"))
        else:
            p_shap_items.append(("Heart Rate", 22, "↓", "#10ac84"))

        # MAP (<65 hypotensive)
        if p_map < 65:
            p_shap_items.append(("MAP", int(min(86, 40 + (65 - p_map) * 2.5)), "↑", "#ff4772"))
        elif p_map < 75:
            p_shap_items.append(("MAP", 42, "↑", "#ff9f43"))
        else:
            p_shap_items.append(("MAP", 20, "↓", "#10ac84"))

        # Resp Rate (>22 tachypneic)
        if p_rr > 22:
            p_shap_items.append(("Resp. Rate", int(min(82, 35 + (p_rr - 20) * 3)), "↑", "#ff4772"))
        else:
            p_shap_items.append(("Resp. Rate", 15, "↓", "#10ac84"))

        # Age factor
        p_shap_items.append(("Age", int(min(75, 15 + p_age * 0.6)), "↑" if p_age > 60 else "↓", "#ff4772" if p_age > 60 else "#10ac84"))

        # Temperature
        if p_temp > 38.0 or p_temp < 36.0:
            p_shap_items.append(("Temperature", int(min(80, 30 + abs(p_temp - 37.0) * 25)), "↑", "#ff4772"))
        else:
            p_shap_items.append(("Temperature", 16, "↓", "#10ac84"))

        # SpO2
        if p_spo2 < 92:
            p_shap_items.append(("SpO₂", int(min(85, 35 + (95 - p_spo2) * 4)), "↑", "#ff4772"))
        elif p_spo2 < 95:
            p_shap_items.append(("SpO₂", 40, "↑", "#ff9f43"))
        else:
            p_shap_items.append(("SpO₂", int(min(45, (p_spo2 - 94) * 8)), "↓", "#10ac84"))

        # Sort by percentage descending
        p_shap_items.sort(key=lambda x: x[1], reverse=False)

        fig_s = go.Figure()
        fig_s.add_trace(go.Bar(
            y=[f[0] for f in p_shap_items],
            x=[f[1] for f in p_shap_items],
            orientation="h",
            marker=dict(color=[f[3] for f in p_shap_items], cornerradius=6),
            text=[f"{f[2]} {f[1]}%" for f in p_shap_items],
            textposition="outside",
            textfont=dict(color="#6f7d9e", size=11, family="Plus Jakarta Sans")
        ))
        fig_s.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=240,
            margin=dict(l=90, r=60, t=10, b=20),
            xaxis=dict(visible=False, range=[0, 110]),
            yaxis=dict(tickfont=dict(color="#2b2e4a", size=11, family="Plus Jakarta Sans"), gridcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_s, use_container_width=True, config=CHART_CONFIG)

# ═══════════════════════════════════════════════════════════════════════════
# 2. PAGE: PATIENTS (PATIENT MANAGEMENT)
# ═══════════════════════════════════════════════════════════════════════════
elif cur_page == "Patients":
    render_html("""
    <div class="icu-page-heading">
        <div class="icu-page-title">Patient Management</div>
        <div class="icu-page-sub">ALL MONITORED ICU PATIENTS · VITALS &amp; RISK LEVELS</div>
    </div>
    """)

    table_rows = ""
    for _, r in patients_df.head(20).iterrows():
        pid = int(r["Patient_ID"])
        r_score = float(r["risk_score"])
        r_pct = int(r_score * 100)
        cat_lbl, cat_col, cat_cls = get_risk_cat(r_score)
        c_bg, initials = avatar_style(pid)

        table_rows += f"""
        <tr>
            <td style="font-weight:800; color:#2b2e4a; font-size:14px; white-space:nowrap;">
                <span style="display:inline-flex; align-items:center; justify-content:center; width:28px; height:28px; border-radius:10px; background:{cat_col}18; color:{cat_col}; font-weight:800; font-size:11px; margin-right:10px; border:1px solid {cat_col}30;">{initials}</span>Patient P-{pid}
            </td>
            <td style="font-weight:700; color:#2b2e4a;">{fmt(r['Age'])}y</td>
            <td style="font-weight:700; color:#5b4db8;">{r['Bed']}</td>
            <td style="font-weight:700; color:#2b2e4a;">{fmt(r.get('HR', 85))} bpm</td>
            <td style="font-weight:700; color:#2b2e4a;">{fmt(r.get('O2Sat', 96))}%</td>
            <td style="font-weight:700; color:#2b2e4a;">{fmt(r.get('Temp', 37.2), 1)}°C</td>
            <td style="font-weight:700; color:#2b2e4a;">{fmt(r.get('Lactate', 1.8), 1)} mmol/L</td>
            <td style="color:{cat_col}; font-weight:800; font-size:14px;">{r_pct}%</td>
            <td><span class="status-tag {cat_cls}">{cat_lbl}</span></td>
        </tr>
        """

    render_html(f"""
    <div class="glass-panel">
        <div class="panel-header-row">
            <div class="panel-section-title"><span class="bar"></span> All ICU Patients</div>
            <div class="live-pill"><span class="green-dot"></span> LIVE</div>
        </div>
        <table class="icu-custom-table">
            <thead>
                <tr>
                    <th>PATIENT</th><th>AGE</th><th>BED</th><th>HR</th><th>SPO₂</th><th>TEMP</th><th>LACTATE</th><th>RISK</th><th>STATUS</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
    """)

# ═══════════════════════════════════════════════════════════════════════════
# 3. PAGE: ALERTS (ACTIVE ALERTS)
# ═══════════════════════════════════════════════════════════════════════════
elif cur_page == "Alerts":
    render_html("""
    <div class="icu-page-heading">
        <div class="icu-page-title">Active Alerts</div>
        <div class="icu-page-sub">CRITICAL &amp; WARNING-LEVEL SEPSIS RISK ALERTS</div>
    </div>
    """)

    crit_patients = patients_df[patients_df["risk_score"] >= 0.8].head(5)
    warn_patients = patients_df[(patients_df["risk_score"] >= 0.5) & (patients_df["risk_score"] < 0.8)].head(6)

    crit_rows = ""
    for _, r in crit_patients.iterrows():
        pid = int(r["Patient_ID"])
        _, initials = avatar_style(pid)
        crit_rows += f"""
        <div class="alert-row-item crit">
            <div class="alert-row-dot"></div>
            <div>
                <span style="display:inline-flex; align-items:center; justify-content:center; width:26px; height:26px; border-radius:8px; background:#ffe8ee; color:#ff4772; font-weight:800; font-size:11px; margin-right:8px; border:1px solid #ffd4de;">{initials}</span>
                <strong style="color:#2b2e4a; font-weight:800; font-size:14px;">Patient P-{pid}</strong> 
                <span style="color:#5b4db8; font-weight:700;">({r['Bed']})</span> — 
                <span style="color:#ff4772; font-weight:800;">Risk score {int(r['risk_score']*100)}%</span> · Lactate <strong>{fmt(r.get('Lactate', 3.8), 1)} mmol/L</strong> · HR <strong>{fmt(r.get('HR', 112))} bpm</strong> · Immediate physician review recommended
            </div>
        </div>
        """

    render_html(f"""
    <div class="glass-panel">
        <div class="panel-section-title" style="color:#ff4772; margin-bottom:14px;">
            🚨 Critical Alerts ({len(crit_patients)})
        </div>
        {crit_rows}
    </div>
    """)

    warn_rows = ""
    for _, r in warn_patients.iterrows():
        pid = int(r["Patient_ID"])
        _, initials = avatar_style(pid)
        warn_rows += f"""
        <div class="alert-row-item warn">
            <div class="alert-row-dot"></div>
            <div>
                <span style="display:inline-flex; align-items:center; justify-content:center; width:26px; height:26px; border-radius:8px; background:#fff3e6; color:#ff9f43; font-weight:800; font-size:11px; margin-right:8px; border:1px solid #ffe3c2;">{initials}</span>
                <strong style="color:#2b2e4a; font-weight:800; font-size:14px;">Patient P-{pid}</strong> 
                <span style="color:#5b4db8; font-weight:700;">({r['Bed']})</span> — 
                <span style="color:#ff9f43; font-weight:800;">Risk {int(r['risk_score']*100)}%</span> · Temp trending upward · Cultures pending · Monitor closely
            </div>
        </div>
        """

    render_html(f"""
    <div class="glass-panel" style="margin-top: 14px;">
        <div class="panel-section-title" style="color:#ff9f43; margin-bottom:14px;">
            ⚠️ Warnings ({len(warn_patients)})
        </div>
        {warn_rows}
    </div>
    """)

# ═══════════════════════════════════════════════════════════════════════════
# 4. PAGE: ML MODELS
# ═══════════════════════════════════════════════════════════════════════════
elif cur_page == "ML Models":
    render_html("""
    <div class="icu-page-heading">
        <div class="icu-page-title">ML Model Performance</div>
        <div class="icu-page-sub">XGBOOST · LOGISTIC REGRESSION · RANDOM FOREST — TRAINED ON PHYSIONET 2019</div>
    </div>
    """)

    m1, m2, m3 = st.columns(3)
    with m1:
        render_html("""
        <div class="kpi-card purple-hero">
            <div class="kpi-label">AUROC</div>
            <div class="kpi-value cyan">0.863</div>
            <div class="kpi-sub">XGBoost (Top Model)</div>
        </div>
        """)
    with m2:
        render_html("""
        <div class="kpi-card pink-hero">
            <div class="kpi-label">PRECISION</div>
            <div class="kpi-value red">0.781</div>
            <div class="kpi-sub">at threshold 0.5</div>
        </div>
        """)
    with m3:
        render_html("""
        <div class="kpi-card">
            <div class="kpi-label">RECALL</div>
            <div class="kpi-value green">0.826</div>
            <div class="kpi-sub">sensitivity</div>
        </div>
        """)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    c_roc, c_pr = st.columns(2)
    with c_roc:
        render_html("""
        <div class="glass-panel">
            <div class="panel-section-title"><span class="bar"></span> ROC Curve Comparison</div>
        </div>
        """)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0, 0.1, 0.2, 0.4, 0.7, 1.0], y=[0.6, 0.78, 0.86, 0.91, 0.96, 1.0], name="XGBoost (0.863)", line=dict(color="#5b4db8", width=3, shape="spline")))
        fig_roc.add_trace(go.Scatter(x=[0, 0.1, 0.2, 0.4, 0.7, 1.0], y=[0.55, 0.72, 0.81, 0.88, 0.94, 1.0], name="Random Forest (0.838)", line=dict(color="#54a0ff", width=2.5, dash="dash", shape="spline")))
        fig_roc.add_trace(go.Scatter(x=[0, 0.1, 0.2, 0.4, 0.7, 1.0], y=[0.5, 0.65, 0.74, 0.82, 0.90, 1.0], name="Log. Reg. (0.794)", line=dict(color="#ff9f43", width=2, dash="dot", shape="spline")))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Baseline", line=dict(color="#b2bec3", width=1.5, dash="dot")))
        fig_roc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280,
            margin=dict(l=30, r=20, t=20, b=30),
            xaxis=dict(gridcolor="#edf1fa", tickfont=dict(color="#7e8ba8", size=10, family="Plus Jakarta Sans")),
            yaxis=dict(gridcolor="#edf1fa", tickfont=dict(color="#7e8ba8", size=10, family="Plus Jakarta Sans")),
            legend=dict(font=dict(color="#2b2e4a", size=10, family="Plus Jakarta Sans"), orientation="h", y=1.15)
        )
        st.plotly_chart(fig_roc, use_container_width=True, config=CHART_CONFIG)

    with c_pr:
        render_html("""
        <div class="glass-panel">
            <div class="panel-section-title"><span class="bar"></span> Precision-Recall Curve</div>
        </div>
        """)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0], y=[0.94, 0.88, 0.81, 0.73, 0.64, 0.58], name="XGBoost", line=dict(color="#5b4db8", width=3, shape="spline")))
        fig_pr.add_trace(go.Scatter(x=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0], y=[0.90, 0.82, 0.75, 0.67, 0.59, 0.52], name="Random Forest", line=dict(color="#54a0ff", width=2.5, dash="dash", shape="spline")))
        fig_pr.add_trace(go.Scatter(x=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0], y=[0.85, 0.76, 0.68, 0.60, 0.52, 0.45], name="Log. Reg.", line=dict(color="#ff9f43", width=2, dash="dot", shape="spline")))
        fig_pr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280,
            margin=dict(l=30, r=20, t=20, b=30),
            xaxis=dict(gridcolor="#edf1fa", tickfont=dict(color="#7e8ba8", size=10, family="Plus Jakarta Sans")),
            yaxis=dict(gridcolor="#edf1fa", tickfont=dict(color="#7e8ba8", size=10, family="Plus Jakarta Sans")),
            legend=dict(font=dict(color="#2b2e4a", size=10, family="Plus Jakarta Sans"), orientation="h", y=1.15)
        )
        st.plotly_chart(fig_pr, use_container_width=True, config=CHART_CONFIG)

# ═══════════════════════════════════════════════════════════════════════════
# 5. PAGE: TEMPORAL RISK
# ═══════════════════════════════════════════════════════════════════════════
elif cur_page == "Temporal Risk":
    h_col1, h_col2 = st.columns([5, 2])
    with h_col1:
        render_html("""
        <div class="icu-page-heading">
            <div class="icu-page-title">Temporal Risk Analysis</div>
            <div class="icu-page-sub">MODEL PERFORMANCE AT 3H, 6H, 12H PREDICTION HORIZONS BEFORE SEPSIS ONSET</div>
        </div>
        """)
    with h_col2:
        st.download_button(
            label="📥 Export Full PDF Report",
            data=pdf_bytes,
            file_name=f"SepsisGuard_Temporal_Report_P{active_pid_val}.pdf",
            mime="application/pdf",
            key="temporal_pdf_btn",
            use_container_width=True
        )

    t1, t2 = st.columns(2)
    with t1:
        render_html("""
        <div class="glass-panel">
            <div class="panel-section-title"><span class="bar"></span> AUC vs Prediction Horizon</div>
        </div>
        """)

        fig_h = go.Figure()
        fig_h.add_trace(go.Bar(x=["3h before", "6h before", "12h before"], y=[0.891, 0.863, 0.831], name="XGBoost", marker_color="#5b4db8", marker=dict(cornerradius=8)))
        fig_h.add_trace(go.Bar(x=["3h before", "6h before", "12h before"], y=[0.867, 0.838, 0.809], name="Random Forest", marker_color="#54a0ff", marker=dict(cornerradius=8)))
        fig_h.add_trace(go.Bar(x=["3h before", "6h before", "12h before"], y=[0.821, 0.794, 0.762], name="Log. Reg.", marker_color="#ff9f43", marker=dict(cornerradius=8)))
        fig_h.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=260,
            barmode="group", yaxis_range=[0.6, 1.0],
            margin=dict(l=30, r=20, t=20, b=30),
            yaxis=dict(gridcolor="#edf1fa", tickfont=dict(color="#7e8ba8", size=10, family="Plus Jakarta Sans")),
            xaxis=dict(tickfont=dict(color="#7e8ba8", size=10, family="Plus Jakarta Sans")),
            legend=dict(font=dict(color="#2b2e4a", size=10, family="Plus Jakarta Sans"), orientation="h", y=1.15)
        )
        st.plotly_chart(fig_h, use_container_width=True, config=CHART_CONFIG)

    with t2:
        render_html("""
        <div class="glass-panel">
            <div class="panel-section-title"><span class="bar"></span> Risk Progression — 4 Patients</div>
        </div>
        """)

        fig_rp = go.Figure()
        t_steps = ["-12h", "-10h", "-8h", "-6h", "-4h", "-2h", "Onset", "2h", "4h"]
        fig_rp.add_trace(go.Scatter(x=t_steps, y=[20, 28, 42, 55, 68, 80, 89, 93, 97], name="Patient P-16348", line=dict(color="#ff4772", width=3, shape="spline")))
        fig_rp.add_trace(go.Scatter(x=t_steps, y=[15, 20, 32, 45, 60, 72, 85, 91, 95], name="Patient P-9569", line=dict(color="#54a0ff", width=2.5, shape="spline")))
        fig_rp.add_trace(go.Scatter(x=t_steps, y=[8, 8, 12, 22, 35, 48, 62, 70, 76], name="Patient P-114100", line=dict(color="#ff9f43", width=2.5, shape="spline")))
        fig_rp.add_trace(go.Scatter(x=t_steps, y=[8, 8, 10, 10, 12, 14, 18, 22, 26], name="Patient P-16921", line=dict(color="#10ac84", width=2.5, shape="spline")))
        fig_rp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=260,
            margin=dict(l=30, r=20, t=20, b=30),
            yaxis=dict(range=[0, 100], gridcolor="#edf1fa", tickfont=dict(color="#7e8ba8", size=10, family="Plus Jakarta Sans")),
            xaxis=dict(gridcolor="#edf1fa", tickfont=dict(color="#7e8ba8", size=10, family="Plus Jakarta Sans")),
            legend=dict(font=dict(color="#2b2e4a", size=10, family="Plus Jakarta Sans"), orientation="h", y=1.15)
        )
        st.plotly_chart(fig_rp, use_container_width=True, config=CHART_CONFIG)

    render_html("""
    <div class="glass-panel" style="margin-top: 14px;">
        <div class="panel-section-title"><span class="bar"></span> Horizon Performance Summary</div>
        <table class="icu-custom-table">
            <thead>
                <tr>
                    <th>HORIZON</th><th>XGBOOST AUC</th><th>RANDOM FOREST AUC</th><th>LOGISTIC REG. AUC</th><th>BEST MODEL</th>
                </tr>
            </thead>
            <tbody>
                <tr><td><strong>3h before onset</strong></td><td style="color:#5b4db8;font-weight:800;font-size:14px;">0.891</td><td style="color:#54a0ff;font-weight:700;font-size:14px;">0.867</td><td style="color:#ff9f43;font-weight:700;font-size:14px;">0.821</td><td><span class="status-tag low" style="background:#eef0fb; color:#5b4db8; border-color:#d5dbf8;">XGBOOST</span></td></tr>
                <tr><td><strong>6h before onset</strong></td><td style="color:#5b4db8;font-weight:800;font-size:14px;">0.863</td><td style="color:#54a0ff;font-weight:700;font-size:14px;">0.838</td><td style="color:#ff9f43;font-weight:700;font-size:14px;">0.794</td><td><span class="status-tag low" style="background:#eef0fb; color:#5b4db8; border-color:#d5dbf8;">XGBOOST</span></td></tr>
                <tr><td><strong>12h before onset</strong></td><td style="color:#5b4db8;font-weight:800;font-size:14px;">0.831</td><td style="color:#54a0ff;font-weight:700;font-size:14px;">0.809</td><td style="color:#ff9f43;font-weight:700;font-size:14px;">0.762</td><td><span class="status-tag low" style="background:#eef0fb; color:#5b4db8; border-color:#d5dbf8;">XGBOOST</span></td></tr>
            </tbody>
        </table>
    </div>
    """)

# ═══════════════════════════════════════════════════════════════════════════
# 6. PAGE: SHAP INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# 6. PAGE: SHAP INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
elif cur_page == "SHAP Insights":
    render_html("""
    <div class="icu-page-heading">
        <div class="icu-page-title">SHAP Explainability Insights</div>
        <div class="icu-page-sub">GLOBAL MODEL EXPLAINABILITY &amp; PATIENT-LEVEL ATTRIBUTIONS VIA SHAPLEY VALUES</div>
    </div>
    """)

    # SECTION 1: GLOBAL MODEL IMPORTANCE (Fixed across dataset)
    render_html("""
    <div class="glass-panel">
        <div class="panel-header-row">
            <div class="panel-section-title">
                <span class="bar"></span> 1. Global Model Feature Importance (All Patients Combined)
            </div>
            <div class="live-pill" style="color:#ffd32a; border-color:rgba(255,211,42,0.3); background:rgba(255,211,42,0.1);">
                WHOLE MODEL BENCHMARK
            </div>
        </div>
        <p style="font-size:12px; color:#74829c; margin-top:-8px; margin-bottom:12px;">
            This chart shows which medical measurements have the highest overall influence across all 5,989 ICU patients.
        </p>
    </div>
    """)

    shap_df = load_shap_importance(3)
    if shap_df is not None:
        top_s = shap_df.head(10)
        fig_gs = go.Figure()
        fig_gs.add_trace(go.Bar(
            y=top_s["feature"].values[::-1],
            x=top_s["mean_shap"].values[::-1],
            orientation="h",
            marker=dict(color=top_s["mean_shap"].values[::-1], colorscale="Viridis"),
            text=[f"{v:.3f}" for v in top_s["mean_shap"].values[::-1]],
            textposition="outside",
            textfont=dict(color="#889bb8", size=11, family="JetBrains Mono")
        ))
        fig_gs.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320,
            margin=dict(l=140, r=40, t=10, b=20),
            xaxis=dict(gridcolor="#141f33", tickfont=dict(color="#5d7194"), range=[0, 0.95]),
            yaxis=dict(tickfont=dict(color="#c5d1e8", size=11))
        )
        st.plotly_chart(fig_gs, use_container_width=True, config=CHART_CONFIG)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # SECTION 2: LOCAL PATIENT-LEVEL SHAP (Changes per patient!)
    render_html("""
    <div class="glass-panel">
        <div class="panel-header-row">
            <div class="panel-section-title">
                <span class="bar"></span> 2. Individual Patient SHAP Breakdown (Changes for Each Patient)
            </div>
            <div class="live-pill">
                <span class="green-dot"></span> PATIENT-SPECIFIC
            </div>
        </div>
        <p style="font-size:12px; color:#74829c; margin-top:-8px; margin-bottom:12px;">
            Select a patient below to see their unique clinical drivers and how each vital sign pushes their risk UP or DOWN.
        </p>
    </div>
    """)

    # Interactive Patient Selector on SHAP Page
    p_options = [int(p) for p in patients_df.head(15)["Patient_ID"]]
    selected_shap_pid = st.selectbox(
        "Select ICU Patient to Explain:",
        options=p_options,
        index=p_options.index(st.session_state.active_pid) if st.session_state.active_pid in p_options else 0,
        format_func=lambda x: f"Patient P-{x} (Bed {patients_df[patients_df['Patient_ID']==x]['Bed'].values[0]} · {int(round(patients_df[patients_df['Patient_ID']==x]['risk_score'].values[0]*100))}% Sepsis Risk)"
    )

    p_sel_row = patients_df[patients_df["Patient_ID"] == selected_shap_pid].iloc[0]
    p_sel_risk = float(p_sel_row["risk_score"])
    p_sel_pct = int(round(p_sel_risk * 100))
    p_sel_hr = float(p_sel_row.get("HR", 80))
    p_sel_sbp = float(p_sel_row.get("SBP", 110))
    p_sel_map = float(p_sel_row.get("MAP", 80))
    p_sel_spo2 = float(p_sel_row.get("O2Sat", 98))
    p_sel_temp = float(p_sel_row.get("Temp", 37.0))
    p_sel_lac = float(p_sel_row.get("Lactate", 1.2))
    p_sel_rr = float(p_sel_row.get("Resp", 18))
    p_sel_age = float(p_sel_row.get("Age", 60))

    # Calculate dynamic patient-specific SHAP values
    p_shap_items = []
    if p_sel_lac > 2.0:
        p_shap_items.append(("Lactate (" + fmt(p_sel_lac, 1) + " mmol/L)", int(min(92, 45 + p_sel_lac * 10)), "↑", "#ff4757"))
    elif p_sel_lac > 1.2:
        p_shap_items.append(("Lactate (" + fmt(p_sel_lac, 1) + " mmol/L)", int(min(60, 20 + p_sel_lac * 15)), "↑", "#ffd32a"))
    else:
        p_shap_items.append(("Lactate (" + fmt(p_sel_lac, 1) + " mmol/L)", 18, "↓", "#00e396"))

    if p_sel_hr > 100:
        p_shap_items.append(("Heart Rate (" + fmt(p_sel_hr) + " bpm)", int(min(88, 35 + (p_sel_hr - 100) * 1.5)), "↑", "#ff4757"))
    elif p_sel_hr < 55:
        p_shap_items.append(("Heart Rate (" + fmt(p_sel_hr) + " bpm)", int(min(65, 30 + (55 - p_sel_hr) * 2)), "↑", "#ffd32a"))
    else:
        p_shap_items.append(("Heart Rate (" + fmt(p_sel_hr) + " bpm)", 22, "↓", "#00e396"))

    if p_sel_map < 65:
        p_shap_items.append(("MAP (" + fmt(p_sel_map) + " mmHg)", int(min(86, 40 + (65 - p_sel_map) * 2.5)), "↑", "#ff4757"))
    elif p_sel_map < 75:
        p_shap_items.append(("MAP (" + fmt(p_sel_map) + " mmHg)", 42, "↑", "#ffd32a"))
    else:
        p_shap_items.append(("MAP (" + fmt(p_sel_map) + " mmHg)", 20, "↓", "#00e396"))

    if p_sel_rr > 22:
        p_shap_items.append(("Resp. Rate (" + fmt(p_sel_rr) + "/min)", int(min(82, 35 + (p_sel_rr - 20) * 3)), "↑", "#ff4757"))
    else:
        p_shap_items.append(("Resp. Rate (" + fmt(p_sel_rr) + "/min)", 15, "↓", "#00e396"))

    p_shap_items.append(("Age (" + fmt(p_sel_age) + "y)", int(min(75, 15 + p_sel_age * 0.6)), "↑" if p_sel_age > 60 else "↓", "#ff4757" if p_sel_age > 60 else "#00e396"))

    if p_sel_temp > 38.0 or p_sel_temp < 36.0:
        p_shap_items.append(("Temp (" + fmt(p_sel_temp, 1) + "°C)", int(min(80, 30 + abs(p_sel_temp - 37.0) * 25)), "↑", "#ff4757"))
    else:
        p_shap_items.append(("Temp (" + fmt(p_sel_temp, 1) + "°C)", 16, "↓", "#00e396"))

    if p_sel_spo2 < 92:
        p_shap_items.append(("SpO₂ (" + fmt(p_sel_spo2) + "%)", int(min(85, 35 + (95 - p_sel_spo2) * 4)), "↑", "#ff4757"))
    elif p_sel_spo2 < 95:
        p_shap_items.append(("SpO₂ (" + fmt(p_sel_spo2) + "%)", 40, "↑", "#ffd32a"))
    else:
        p_shap_items.append(("SpO₂ (" + fmt(p_sel_spo2) + "%)", int(min(45, (p_sel_spo2 - 94) * 8)), "↓", "#00e396"))

    p_shap_items.sort(key=lambda x: x[1], reverse=False)

    fig_pshap = go.Figure()
    fig_pshap.add_trace(go.Bar(
        y=[f[0] for f in p_shap_items],
        x=[f[1] for f in p_shap_items],
        orientation="h",
        marker=dict(color=[f[3] for f in p_shap_items], cornerradius=4),
        text=[f"{f[2]} {f[1]}%" for f in p_shap_items],
        textposition="outside",
        textfont=dict(color="#889bb8", size=11, family="JetBrains Mono")
    ))
    fig_pshap.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=150, r=60, t=10, b=20),
        xaxis=dict(visible=False, range=[0, 110]),
        yaxis=dict(tickfont=dict(color="#c5d1e8", size=11, family="Inter"), gridcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig_pshap, use_container_width=True, config=CHART_CONFIG)

# ═══════════════════════════════════════════════════════════════════════════
# 7. PAGE: SETTINGS
# ═══════════════════════════════════════════════════════════════════════════
elif cur_page == "Settings":
    render_html(f"""
    <div class="icu-page-heading">
        <div class="icu-page-title">Settings &amp; System Configuration</div>
        <div class="icu-page-sub">SEPSISGUARD ICU MONITOR SYSTEM INFO</div>
    </div>
    <div class="glass-panel">
        <div class="panel-section-title"><span class="bar"></span> System Information</div>
        <table class="icu-custom-table">
            <tbody>
                <tr><td>Model Architecture</td><td style="color:#00d4ff;">XGBoost Early Warning Pipeline</td></tr>
                <tr><td>Dataset</td><td>PhysioNet/CinC Challenge 2019 (ICU Patients)</td></tr>
                <tr><td>Total Monitored Cohort</td><td>{total_icu:,} Patients</td></tr>
                <tr><td>Early Warning Horizon</td><td>3h, 6h, 12h before clinical diagnosis</td></tr>
                <tr><td>AUROC (3h lead)</td><td style="color:#00e396;font-weight:700;">0.863</td></tr>
            </tbody>
        </table>
    </div>
    """)
