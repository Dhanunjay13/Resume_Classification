from __future__ import annotations

import base64
import io
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from docx import Document


# =============================================================================
# APPLICATION SETTINGS
# =============================================================================
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best_resume_classifier.pkl"
ENCODER_PATH = APP_DIR / "label_encoder.pkl"
BACKGROUND_PATH = APP_DIR / "resume_background.png"
PIKACHU_PATH = APP_DIR / "pikachu_processing.webp"
REVIEW_THRESHOLD = 0.70
MIN_WORD_COUNT = 80

SUPPORTED_UPLOADS = ["pdf", "docx", "doc", "txt", "csv", "xlsx", "xls"]
DOCUMENT_MIME_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "txt": "text/plain",
    "csv": "text/csv",
}

st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True
if "nav_section" not in st.session_state:
    st.session_state.nav_section = "Resume Classification - Prediction"
if st.session_state.nav_section not in {"Resume Classification - Prediction", "About the Model"}:
    st.session_state.nav_section = "Resume Classification - Prediction"


# =============================================================================
# DASHBOARD BACKGROUND AND VISUAL STYLE
# =============================================================================
def build_dashboard_css() -> str:
    if BACKGROUND_PATH.exists():
        encoded_bg = base64.b64encode(BACKGROUND_PATH.read_bytes()).decode("ascii")
        background_css = (
            "background-image: linear-gradient(110deg, rgba(3,12,34,.88), "
            "rgba(5,20,51,.78)), "
            f"url('data:image/png;base64,{encoded_bg}');"
        )
    else:
        background_css = "background: linear-gradient(135deg,#06132f,#102c62 55%,#07142f);"

    css_lines = [
        "<style>",
        ".stApp { " + background_css + " background-size: cover; background-position: center; background-attachment: fixed; background-repeat: no-repeat; }",
        '[data-testid="stAppViewContainer"], [data-testid="stMain"], .main {background:transparent !important;}',
        ".block-container {max-width:1500px; padding-top:1.35rem; padding-bottom:3rem;}",
        '[data-testid="stSidebar"] > div:first-child {background:linear-gradient(180deg,rgba(3,13,39,.99),rgba(8,33,77,.97)); border-right:1px solid rgba(90,205,255,.36);}',
        '[data-testid="stSidebar"] {color:#f5f9ff;}',
        '[data-testid="stSidebar"] [data-testid="stRadio"] label {color:#e8f5ff !important; font-weight:700; padding:.35rem .45rem; border-radius:10px;}',
        '[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {background:rgba(36,166,255,.15); color:#74e6ff !important;}',
        '[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radio"] {accent-color:#17c8ff;}',
        '[data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"] {display:none !important;}',
        ".brand-card {padding:1.05rem .95rem; border-radius:18px; background:linear-gradient(135deg,rgba(0,174,255,.25),rgba(100,77,255,.24)); border:1px solid rgba(104,207,255,.48); margin-bottom:.8rem; box-shadow:0 9px 30px rgba(0,0,0,.23);}",
        ".brand-title {font-size:1.55rem; line-height:1.2; font-weight:900; color:#fff; letter-spacing:.2px; text-shadow:0 2px 18px rgba(58,192,255,.35);}",
        ".brand-subtitle {color:#b9eaff; font-size:.82rem; margin-top:.35rem;}",
        ".hero-kicker {color:#7ee8ff; font-size:.76rem; font-weight:850; letter-spacing:.17em; text-transform:uppercase;}",
        ".hero-title {color:#fff; font-size:clamp(2.5rem,5vw,4rem); line-height:1.05; font-weight:950; margin:.25rem 0 .85rem 0; text-shadow:0 3px 26px rgba(0,0,0,.42);}",
        ".hero-subtitle {color:#d9edff; font-size:1rem; margin-bottom:1rem;}",
        ".glass-card {background:rgba(5,20,52,.82); border:1px solid rgba(110,203,255,.34); border-radius:20px; padding:1.15rem 1.25rem; box-shadow:0 14px 40px rgba(0,0,0,.24); backdrop-filter:blur(10px); margin-bottom:.9rem;}",
        ".section-title {color:#fff; font-weight:850; font-size:1.2rem;}",
        ".muted-copy {color:#c9ddf4;}",
        'div[data-testid="stMetric"] {background:rgba(5,22,56,.84); border:1px solid rgba(90,194,255,.36); padding:.8rem 1rem; border-radius:15px; box-shadow:0 8px 25px rgba(0,0,0,.14);}',
        'div[data-testid="stMetric"] label {color:#b9d8f8 !important;}',
        'div[data-testid="stMetric"] [data-testid="stMetricValue"] {color:#fff !important;}',
        ".stButton > button, .stDownloadButton > button {border:0 !important; border-radius:13px !important; color:#fff !important; font-weight:850 !important; background:linear-gradient(100deg,#0878e8,#635bfa) !important; box-shadow:0 7px 22px rgba(15,116,237,.28); min-height:2.8rem;}",
        ".stButton > button:hover, .stDownloadButton > button:hover {border:1px solid #8beaff !important; filter:brightness(1.12);}",
        '[data-testid="stFileUploader"] section {background:rgba(6,25,60,.83); border:1.5px dashed rgba(94,207,255,.75); border-radius:17px;}',
        '[data-testid="stFileUploader"] label, [data-testid="stFileUploader"] small {color:#e9f7ff !important;}',
        ".stProgress > div > div > div > div {background:linear-gradient(90deg,#04c7ff,#7867ff);}",
        '[data-testid="stDataFrame"] {background:rgba(4,18,46,.86); border:1px solid rgba(111,197,255,.3); border-radius:14px;}',
        "hr {border-color:rgba(141,205,255,.25) !important;}",
        "h1,h2,h3,p,label,.stMarkdown {text-shadow:0 1px 9px rgba(0,0,0,.16);}",
        '[data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"] {display:none !important;}',
        "" if st.session_state.sidebar_open else '[data-testid="stSidebar"] {display:none !important;} [data-testid="stMain"] {margin-left:0 !important; width:100% !important;}',
        ".pika-shell {position:relative;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:195px;margin:.6rem auto 1rem auto;padding:1rem;border-radius:22px;background:radial-gradient(circle at 50% 45%,rgba(27,166,255,.20),rgba(4,15,41,.78) 68%);border:1px solid rgba(111,220,255,.36);overflow:hidden;}",
        ".pika-shell img {width:min(300px,80vw);max-height:175px;object-fit:contain;position:relative;z-index:2;border-radius:16px;}",
        ".pika-working img {animation:pikaCharge 1s ease-in-out infinite;filter:drop-shadow(0 0 12px rgba(40,200,255,.9));}",
        ".pika-complete img {animation:pikaBlast 1.45s cubic-bezier(.16,.88,.28,1) both;filter:drop-shadow(0 0 24px #ffe45e);}",
        ".pika-caption {position:relative;z-index:3;color:#e7f7ff;font-weight:850;letter-spacing:.04em;margin-top:.45rem;text-align:center;}",
        ".pika-bolt {position:absolute;color:#ffe45e;font-size:2.5rem;opacity:.8;animation:boltFlash .55s ease-in-out infinite alternate;}",
        ".pika-bolt.left {left:12%;top:18%;}.pika-bolt.right {right:12%;bottom:18%;animation-delay:.22s;}",
        ".pika-complete:after {content:'';position:absolute;inset:12%;border:3px solid rgba(255,232,95,.85);border-radius:50%;animation:burstRing 1.15s ease-out both;}",
        "@keyframes pikaCharge {0%,100%{transform:translateY(2px) scale(.96) rotate(-1deg)}50%{transform:translateY(-8px) scale(1.04) rotate(1deg)}}",
        "@keyframes pikaBlast {0%{transform:scale(.48);opacity:.25}55%{transform:scale(1.22);opacity:1}78%{transform:scale(.96)}100%{transform:scale(1.05);opacity:1}}",
        "@keyframes burstRing {0%{transform:scale(.2);opacity:1}100%{transform:scale(1.7);opacity:0}}",
        "@keyframes boltFlash {from{opacity:.25;transform:scale(.75) rotate(-8deg)}to{opacity:1;transform:scale(1.18) rotate(8deg)}}",
        "</style>",
    ]
    return "\n".join(css_lines)


st.markdown(build_dashboard_css(), unsafe_allow_html=True)


def pikachu_animation_html(state: str) -> str:
    """Show the supplied Pikachu image with a CSS processing / completion effect."""
    if not PIKACHU_PATH.exists():
        return "<div class='glass-card'>⚡ Classification in progress...</div>"
    image_data = base64.b64encode(PIKACHU_PATH.read_bytes()).decode("ascii")
    if state == "complete":
        wrapper_class = "pika-shell pika-complete"
        caption = "⚡ CLASSIFICATION COMPLETE! ⚡"
    else:
        wrapper_class = "pika-shell pika-working"
        caption = "⚡ PIKACHU IS ANALYZING RESUMES... ⚡"
    return (
        f"<div class='{wrapper_class}'>"
        "<span class='pika-bolt left'>ϟ</span><span class='pika-bolt right'>ϟ</span>"
        f"<img src='data:image/webp;base64,{image_data}' alt='Pikachu classification assistant'>"
        f"<div class='pika-caption'>{caption}</div></div>"
    )


# =============================================================================
# TEXT PREPROCESSING — keep identical to the training preprocessing
# =============================================================================
def clean_resume_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(
        r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b",
        " EMAIL ",
        text,
        flags=re.I,
    )
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text, flags=re.I)
    phone_pattern = (
        r"\b(?:\+?\d{1,3}[\s.-]?)?"
        r"(?:\(?\d{2,4}\)?[\s.-]?)?"
        r"\d{3,4}[\s.-]?\d{4}\b"
    )
    text = re.sub(phone_pattern, " PHONE ", text)
    text = re.sub(r"[^\w+#./@ -]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


# =============================================================================
# DOCUMENT TEXT EXTRACTION
# =============================================================================
def extract_docx(file_bytes: bytes) -> str:
    document = Document(io.BytesIO(file_bytes))
    chunks = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            chunks.append(" | ".join(cell.text for cell in row.cells))
    for section in document.sections:
        chunks.extend(paragraph.text for paragraph in section.header.paragraphs)
        chunks.extend(paragraph.text for paragraph in section.footer.paragraphs)
    return "\n".join(chunks)


def extract_doc(file_bytes: bytes) -> str:
    if not shutil.which("antiword"):
        raise RuntimeError("The .doc reader is unavailable. Keep packages.txt with 'antiword' in the repository.")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name
        result = subprocess.run(
            ["antiword", temp_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=45,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(result.stderr.strip() or "The .doc file could not be read.")
        return result.stdout
    finally:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink(missing_ok=True)


def extract_document_text(filename: str, file_bytes: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
            return "\n".join(page.get_text("text") for page in pdf)
    if suffix == ".docx":
        return extract_docx(file_bytes)
    if suffix == ".doc":
        return extract_doc(file_bytes)
    if suffix == ".txt":
        return file_bytes.decode("utf-8", errors="replace")
    raise ValueError("Unsupported document type.")


# =============================================================================
# CSV / EXCEL INPUT
# A row is treated as one resume; the selected text column must contain resume text.
# =============================================================================
def read_tabular_file(filename: str, file_bytes: bytes) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    buffer = io.BytesIO(file_bytes)
    if suffix == ".csv":
        last_error = None
        for encoding in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                buffer.seek(0)
                return pd.read_csv(buffer, encoding=encoding, sep=None, engine="python")
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Could not read the CSV file: {last_error}")
    if suffix in {".xlsx", ".xls"}:
        # The first worksheet is used. Each non-empty row represents one resume.
        return pd.read_excel(buffer, sheet_name=0)
    raise ValueError("This is not a CSV or Excel file.")


def suggest_text_column(dataframe: pd.DataFrame) -> str:
    if dataframe.empty or len(dataframe.columns) == 0:
        raise ValueError("The spreadsheet does not contain any data columns.")
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(column).lower()): column
        for column in dataframe.columns
    }
    preferred_names = [
        "resumetext", "cvtext", "rawtext", "resume", "resumes", "cv",
        "content", "text", "description", "profile", "details",
    ]
    for name in preferred_names:
        if name in normalized:
            return normalized[name]

    # Otherwise choose the text-like column with the largest average text length.
    text_columns = [
        column for column in dataframe.columns
        if pd.api.types.is_object_dtype(dataframe[column])
        or pd.api.types.is_string_dtype(dataframe[column])
    ]
    if not text_columns:
        raise ValueError("No text column was found. Add a column containing the resume text.")
    return max(
        text_columns,
        key=lambda column: dataframe[column].fillna("").astype(str).str.len().mean(),
    )


def cell_to_text(value: Any) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    return str(value)


# =============================================================================
# MODEL LOADING AND PREDICTION
# =============================================================================
@st.cache_resource(show_spinner="Loading the trained resume classifier...")
def load_artifacts():
    missing = [path.name for path in (MODEL_PATH, ENCODER_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing file(s): " + ", ".join(missing)
            + ". Add both model files beside app.py in GitHub."
        )
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise TypeError("The saved Pipeline must support predict() and predict_proba().")
    if not hasattr(label_encoder, "inverse_transform"):
        raise TypeError("label_encoder.pkl is not a fitted LabelEncoder-compatible object.")
    return model, label_encoder


def get_class_names(model: Any, label_encoder: Any) -> list[str]:
    classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else None
    encoded_classes = getattr(classifier, "classes_", getattr(model, "classes_", None))
    if encoded_classes is None:
        raise ValueError("Could not read class labels from the saved classifier.")
    decoded = label_encoder.inverse_transform(np.asarray(encoded_classes).astype(int))
    return [str(label) for label in decoded]


def classify_text_record(
    record_name: str,
    raw_text: str,
    model: Any,
    label_encoder: Any,
    *,
    source_kind: str,
    original_bytes: bytes | None = None,
    source_file: str | None = None,
    source_row: dict[str, Any] | None = None,
    row_number: int | None = None,
) -> dict:
    cleaned_text = clean_resume_text(raw_text)
    word_count = len(cleaned_text.split())
    if not cleaned_text or word_count < MIN_WORD_COUNT:
        raise ValueError(
            f"Only {word_count} usable words were found. Check that the selected column contains full resume text."
        )

    predicted_encoded = int(np.asarray(model.predict([cleaned_text])).reshape(-1)[0])
    predicted_category = str(label_encoder.inverse_transform([predicted_encoded])[0])
    probabilities = np.asarray(model.predict_proba([cleaned_text]))[0]
    class_names = get_class_names(model, label_encoder)
    ranked = sorted(
        [(name, float(score)) for name, score in zip(class_names, probabilities)],
        key=lambda item: item[1],
        reverse=True,
    )
    top_probability = ranked[0][1]

    return {
        "file_name": record_name,
        "source_kind": source_kind,
        "file_bytes": original_bytes,
        "source_file": source_file,
        "source_row": source_row,
        "row_number": row_number,
        "predicted_category": predicted_category,
        "top_probability": top_probability,
        "word_count": word_count,
        "review_status": "Review recommended" if top_probability < REVIEW_THRESHOLD else "Suggested category",
        "category_scores": {name: score for name, score in ranked},
    }


def classify_document(filename: str, file_bytes: bytes, model: Any, label_encoder: Any) -> dict:
    raw_text = extract_document_text(filename, file_bytes)
    return classify_text_record(
        filename,
        raw_text,
        model,
        label_encoder,
        source_kind="document",
        original_bytes=file_bytes,
    )


# =============================================================================
# DOWNLOAD AND RESULT HELPERS
# =============================================================================
def safe_path_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", str(value)).strip(" ._")
    return cleaned[:100] or "Unsorted"


def spreadsheet_row_csv(item: dict) -> bytes:
    row_data = dict(item.get("source_row") or {})
    row_data["Predicted Category"] = item["predicted_category"]
    row_data["Model Score"] = round(float(item["top_probability"]), 4)
    row_data["Review Status"] = item["review_status"]
    return pd.DataFrame([row_data]).to_csv(index=False).encode("utf-8-sig")


def item_download(item: dict) -> tuple[bytes, str, str]:
    if item.get("source_kind") == "spreadsheet_row":
        content = spreadsheet_row_csv(item)
        stem = safe_path_part(Path(item["source_file"]).stem)
        filename = f"{safe_path_part(item['predicted_category'])}_{stem}_row_{item['row_number']}.csv"
        return content, filename, "text/csv"

    original_name = Path(item["file_name"]).name
    extension = Path(original_name).suffix.lower().lstrip(".")
    filename = f"{safe_path_part(item['predicted_category'])}_{safe_path_part(original_name)}"
    mime = DOCUMENT_MIME_TYPES.get(extension, "application/octet-stream")
    return item["file_bytes"], filename, mime


def make_classified_zip(results: list[dict]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, item in enumerate(results, start=1):
            category_folder = safe_path_part(item["predicted_category"])
            content, download_name, _ = item_download(item)
            archive_path = f"{category_folder}/{index:03d}_{safe_path_part(download_name)}"
            archive.writestr(archive_path, content)
    output.seek(0)
    return output.getvalue()


def build_results_dataframe(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Resume / record": item["file_name"],
                "Predicted category": item["predicted_category"],
                "Top class probability": round(float(item["top_probability"]), 4),
                "Extracted words": item["word_count"],
                "Status": item["review_status"],
            }
            for item in results
        ]
    )


# =============================================================================
# LOAD SAVED MODEL AND SESSION STATE
# =============================================================================
try:
    model, label_encoder = load_artifacts()
    available_categories = get_class_names(model, label_encoder)
except Exception as exc:
    st.error(f"Unable to load the trained classifier: {exc}")
    st.info("Keep best_resume_classifier.pkl and label_encoder.pkl beside app.py in the GitHub repository.")
    st.stop()

if "classification_results" not in st.session_state:
    st.session_state.classification_results = []
if "last_run_time" not in st.session_state:
    st.session_state.last_run_time = None

# The branded Resume Classifier button is the single custom navigation toggle.
# Native Streamlit >> / << controls remain hidden.
brand_left, brand_right = st.columns([2.2, 5])
with brand_left:
    if st.button("📄  Resume Classifier", key="resume_classifier_sidebar_toggle", use_container_width=True):
        st.session_state.sidebar_open = not st.session_state.sidebar_open
        st.rerun()
with brand_right:
    st.markdown(
        '<div class="hero-kicker" style="padding:.8rem 0">AI-POWERED RESUME ANALYSIS</div>',
        unsafe_allow_html=True,
    )

selected_section = st.session_state.nav_section
if st.session_state.sidebar_open:
    with st.sidebar:
        st.markdown("<div class='hero-kicker'>WORKSPACE</div>", unsafe_allow_html=True)
        selected_section = st.radio(
            "Dashboard navigation",
            ["Resume Classification - Prediction", "About the Model"],
            key="nav_section",
            label_visibility="collapsed",
        )

# =============================================================================
# MAIN SCREEN — UPLOAD, CLASSIFY, VIEW AND DOWNLOAD IN ONE PLACE
# =============================================================================
if selected_section == "Resume Classification - Prediction":
    st.markdown('<div class="hero-title">Resume Classification &amp; Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Upload resumes, run classification and download the results from this screen.</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card"><div class="section-title">Upload resume files or a resume dataset</div><div class="muted-copy">Select individual resumes, or a CSV / Excel table where each row contains one resume in a text column.</div></div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Choose resume files or CSV / Excel data",
        type=SUPPORTED_UPLOADS,
        accept_multiple_files=True,
        key="resume_file_uploader_v2",
        help="Supported: PDF, DOCX, DOC, TXT, CSV, XLSX and XLS. For a spreadsheet, each non-empty row in the selected text column is treated as one resume.",
    )

    spreadsheet_frames: dict[int, pd.DataFrame] = {}
    spreadsheet_columns: dict[int, str] = {}
    spreadsheet_load_errors: dict[int, str] = {}

    if uploaded_files:
        for file_index, uploaded_file in enumerate(uploaded_files):
            suffix = Path(uploaded_file.name).suffix.lower()
            if suffix in {".csv", ".xlsx", ".xls"}:
                try:
                    table = read_tabular_file(uploaded_file.name, uploaded_file.getvalue())
                    if table.empty:
                        raise ValueError("The spreadsheet has no data rows.")
                    suggested_column = suggest_text_column(table)
                    spreadsheet_frames[file_index] = table
                    column_lookup = {str(column): column for column in table.columns}
                    column_options = list(column_lookup.keys())
                    default_index = column_options.index(str(suggested_column))
                    st.markdown(f"**Text column for `{uploaded_file.name}`**")
                    chosen_column = st.selectbox(
                        "Select the column containing full resume text",
                        options=column_options,
                        index=default_index,
                        key=f"text_column_{file_index}",
                    )
                    spreadsheet_columns[file_index] = column_lookup[chosen_column]
                    st.caption(f"{len(table):,} rows detected · each non-empty row will be classified separately")
                except Exception as exc:
                    spreadsheet_load_errors[file_index] = str(exc)
                    st.error(f"Could not prepare `{uploaded_file.name}`: {exc}")

        count_col1, count_col2, count_col3 = st.columns(3)
        count_col1.metric("Files selected", len(uploaded_files))
        count_col2.metric("Learned categories", len(available_categories))
        table_rows = sum(len(frame) for frame in spreadsheet_frames.values())
        count_col3.metric("Spreadsheet rows", table_rows)

    classify_disabled = not uploaded_files or (len(spreadsheet_load_errors) > 0)
    classify_clicked = st.button(
        "🚀  ANALYZE & CLASSIFY",
        type="primary",
        use_container_width=True,
        disabled=classify_disabled,
        key="classify_uploads_button",
    )

    if classify_clicked and uploaded_files:
        new_results: list[dict] = []
        errors: list[tuple[str, str]] = []
        total_records = 0

        # Count individual documents plus spreadsheet rows before showing progress.
        for file_index, uploaded_file in enumerate(uploaded_files):
            suffix = Path(uploaded_file.name).suffix.lower()
            if suffix in {".csv", ".xlsx", ".xls"}:
                total_records += len(spreadsheet_frames.get(file_index, []))
            else:
                total_records += 1
        total_records = max(total_records, 1)

        anime_area = st.empty()
        anime_started = time.monotonic()
        anime_area.markdown(pikachu_animation_html("working"), unsafe_allow_html=True)
        progress = st.progress(0, text="Preparing classification...")
        status_area = st.empty()
        completed = 0

        with st.spinner("Analyzing resume text and generating category predictions..."):
            for file_index, uploaded_file in enumerate(uploaded_files):
                suffix = Path(uploaded_file.name).suffix.lower()
                file_bytes = uploaded_file.getvalue()

                if suffix in {".csv", ".xlsx", ".xls"}:
                    table = spreadsheet_frames.get(file_index)
                    text_column = spreadsheet_columns.get(file_index)
                    if table is None or text_column is None:
                        errors.append((uploaded_file.name, "Could not identify a resume-text column."))
                        continue

                    for row_position, (_, row) in enumerate(table.iterrows(), start=1):
                        status_area.info(f"🔎 Reading {uploaded_file.name} · resume row {row_position}")
                        try:
                            raw_resume_text = cell_to_text(row[text_column])
                            if not raw_resume_text.strip():
                                raise ValueError("The selected resume-text cell is empty.")
                            result = classify_text_record(
                                f"{uploaded_file.name} · row {row_position}",
                                raw_resume_text,
                                model,
                                label_encoder,
                                source_kind="spreadsheet_row",
                                source_file=uploaded_file.name,
                                source_row=row.to_dict(),
                                row_number=row_position,
                            )
                            new_results.append(result)
                        except Exception as exc:
                            errors.append((f"{uploaded_file.name} · row {row_position}", str(exc)))
                        completed += 1
                        progress.progress(completed / total_records, text=f"Classified {completed} of {total_records} resume record(s)")
                else:
                    status_area.info(f"🔎 Analyzing {uploaded_file.name}")
                    try:
                        result = classify_document(uploaded_file.name, file_bytes, model, label_encoder)
                        new_results.append(result)
                    except Exception as exc:
                        errors.append((uploaded_file.name, str(exc)))
                    completed += 1
                    progress.progress(completed / total_records, text=f"Classified {completed} of {total_records} resume record(s)")

        st.session_state.classification_results = new_results
        st.session_state.last_run_time = pd.Timestamp.now().strftime("%d %b %Y, %I:%M:%S %p")
        status_area.empty()
        progress.empty()

        # Keep the uploaded Pikachu visible during very fast single-resume predictions.
        remaining_animation = 1.2 - (time.monotonic() - anime_started)
        if remaining_animation > 0:
            time.sleep(remaining_animation)

        if new_results:
            anime_area.markdown(pikachu_animation_html("complete"), unsafe_allow_html=True)
            # Let the completion burst finish before the results/download area is rendered.
            time.sleep(1.5)
            st.success(f"Classification completed: {len(new_results)} resume record(s) processed.")
            st.balloons()
        else:
            anime_area.empty()
        if errors:
            with st.expander(f"Items needing attention ({len(errors)})", expanded=True):
                for filename, message in errors:
                    st.warning(f"{filename}: {message}")

    # Results and downloads remain directly below the upload controls.
    results = st.session_state.classification_results
    if results:
        st.markdown("## Classification results")
        summary_df = build_results_dataframe(results)
        display_df = summary_df.copy()
        display_df["Top class probability"] = display_df["Top class probability"].map(lambda score: f"{score:.1%}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("### Classified files")
        for result_index, item in enumerate(results, start=1):
            with st.container(border=True):
                col_file, col_category, col_score, col_download = st.columns([2.2, 1.5, 1, 1.4])
                col_file.markdown(f"**{result_index}. {item['file_name']}**")
                col_category.write(item["predicted_category"])
                col_score.metric("Model score", f"{item['top_probability']:.1%}")
                download_bytes, download_name, download_mime = item_download(item)
                col_download.download_button(
                    "⬇️ Download classified",
                    data=download_bytes,
                    file_name=download_name,
                    mime=download_mime,
                    key=f"main_download_{result_index}_{safe_path_part(item['file_name'])}",
                    use_container_width=True,
                )
                if item["top_probability"] < REVIEW_THRESHOLD:
                    st.caption("Manual review recommended for this prediction.")

        st.markdown("### Download center")
        zip_data = make_classified_zip(results)
        download_col1, download_col2 = st.columns([1.4, 1])
        with download_col1:
            st.download_button(
                "📦  DOWNLOAD ALL — CATEGORY FOLDERS (ZIP)",
                data=zip_data,
                file_name="classified_resumes_by_category.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
                key="download_classified_zip_main",
            )
        with download_col2:
            st.download_button(
                "📊  DOWNLOAD REPORT (CSV)",
                data=summary_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="resume_classification_report.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_classification_report_main",
            )

        st.markdown("### Resume classification analytics")
        category_counts = (
            summary_df["Predicted category"]
            .value_counts()
            .rename_axis("Category")
            .reset_index(name="Resumes")
        )
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.markdown("**Category-wise resume count**")
            bar_figure = px.bar(
                category_counts,
                x="Category",
                y="Resumes",
                color="Category",
                text="Resumes",
                color_discrete_sequence=["#19C6F5", "#6574F7", "#35D0A0", "#F6B84B", "#BE8CFF"],
            )
            bar_figure.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                margin=dict(l=10, r=10, t=20, b=10),
                yaxis_title="Number of resumes", xaxis_title="Predicted category",
            )
            st.plotly_chart(bar_figure, use_container_width=True)
        with chart_right:
            st.markdown("**Share of classified resumes**")
            pie_figure = px.pie(
                category_counts, names="Category", values="Resumes", hole=0.48,
                color_discrete_sequence=["#19C6F5", "#6574F7", "#35D0A0", "#F6B84B", "#BE8CFF"],
            )
            pie_figure.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=20, b=10),
                legend_title_text="Category",
            )
            pie_figure.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(pie_figure, use_container_width=True)


# =============================================================================
# SECTION 2 — ABOUT THE MODEL
# =============================================================================
elif selected_section == "About the Model":
    st.markdown('<div class="hero-kicker">MODEL OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">About the Model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">How resume text is transformed into a category suggestion.</div>',
        unsafe_allow_html=True,
    )

    overview1, overview2, overview3 = st.columns(3)
    overview1.metric("Algorithm", "Logistic Regression")
    overview2.metric("Text representation", "TF-IDF")
    overview3.metric("Learned categories", len(available_categories))

    st.markdown("### How the classifier works")
    info_cols = st.columns(3)
    with info_cols[0]:
        st.markdown(
            '<div class="glass-card"><div class="section-title">01 · Text preparation</div>'
            '<div class="muted-copy">Resume text is extracted from supported files, normalized, and contact details are masked before feature extraction.</div></div>',
            unsafe_allow_html=True,
        )
    with info_cols[1]:
        st.markdown(
            '<div class="glass-card"><div class="section-title">02 · TF-IDF features</div>'
            '<div class="muted-copy">Words and two-word phrases are converted into weighted numerical features that represent the document content.</div></div>',
            unsafe_allow_html=True,
        )
    with info_cols[2]:
        st.markdown(
            '<div class="glass-card"><div class="section-title">03 · Category prediction</div>'
            '<div class="muted-copy">Logistic Regression compares the feature pattern with learned classes and returns a suggested category with class-probability estimates.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Why Logistic Regression?")
    st.write(
        "Logistic Regression is a supervised classification algorithm suited to sparse, high-dimensional TF-IDF text features. "
        "It learns a decision boundary between the labelled resume categories and produces class-probability estimates "
        "that can be displayed for review."
    )

    st.markdown("### Model evaluation")
    eval1, eval2, eval3 = st.columns(3)
    eval1.metric("5-fold cross-validation Macro F1", "98.57%")
    eval2.metric("Holdout test Macro F1", "100%")
    eval3.metric("Holdout test accuracy", "100%")
    st.caption(
        "The holdout test contained 16 resumes. These project results are encouraging, but the small test set does not guarantee performance on all future resumes."
    )

    st.markdown("### Categories learned during training")
    category_columns = st.columns(2)
    for category_index, category in enumerate(available_categories):
        with category_columns[category_index % 2]:
            st.markdown(
                f'<div class="glass-card"><span style="color:#7ee8ff;font-weight:900">●</span> '
                f'<span style="color:#ffffff;font-weight:750">{category}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Model behavior and scope")
    st.write(
        "The classifier returns the closest category among the labels it learned during training. "
        "It can process resumes for different job roles, but it cannot create a new job-category label by itself. "
        "To recognize additional roles, add representative labelled resumes for each new category and retrain the model."
    )
    st.write(
        "Use the prediction to support resume organization and document routing. Review unfamiliar or low-confidence "
        "predictions manually, and do not use the classifier to rank candidates or make automated hiring decisions."
    )


# =============================================================================
# FOOTER
# =============================================================================
st.markdown(
    "<div style='text-align:center;color:#c5dcf5;font-size:.85rem;padding:1rem 0 .2rem'>"
    "📄 Resume Classifier · Resume analysis and category insights"
    "</div>",
    unsafe_allow_html=True,
)
