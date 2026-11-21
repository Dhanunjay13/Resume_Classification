from __future__ import annotations

import base64
import io
import re
import shutil
import subprocess
import tempfile
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from docx import Document


# =============================================================================
# APP SETTINGS
# =============================================================================
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best_resume_classifier.pkl"
ENCODER_PATH = APP_DIR / "label_encoder.pkl"
BACKGROUND_PATH = APP_DIR / "resume_background.png"
REVIEW_THRESHOLD = 0.70
MIN_WORD_COUNT = 80

st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# BACKGROUND + DASHBOARD STYLING
# Keep resume_background.png in the same GitHub folder as app.py.
# =============================================================================
def load_background_css() -> str:
    if BACKGROUND_PATH.exists():
        encoded = base64.b64encode(BACKGROUND_PATH.read_bytes()).decode("ascii")
        background_layer = (
            "linear-gradient(105deg, rgba(2, 9, 30, .88), "
            "rgba(3, 15, 45, .82)), "
            f"url('data:image/png;base64,{encoded}')"
        )
        return f"background-image: {background_layer};"

    # Gradient fallback if the image has not yet been uploaded to GitHub.
    return "background: linear-gradient(135deg, #06132f 0%, #102c62 55%, #07142f 100%);"


background_css = load_background_css()
# Use a normal triple-quoted template (not an f-string) so CSS braces cannot
# accidentally be interpreted as Python f-string expressions.
DASHBOARD_CSS = """
<style>
.stApp {
    __RESUME_BACKGROUND__
    background-size: cover;
    background-position: center center;
    background-attachment: fixed;
    background-repeat: no-repeat;
}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main { background: transparent !important; }
.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, rgba(4, 15, 43, .98), rgba(7, 31, 70, .96));
    border-right: 1px solid rgba(72, 190, 255, .35);
}
[data-testid="stSidebar"] { color: #f4f8ff; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #e6f3ff !important;
    font-weight: 650;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    color: #62ddff !important;
}
.brand-card {
    padding: 1.1rem .9rem;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(0, 174, 255, .22), rgba(95, 86, 255, .22));
    border: 1px solid rgba(104, 207, 255, .45);
    margin-bottom: 1rem;
    box-shadow: 0 8px 28px rgba(0, 0, 0, .20);
