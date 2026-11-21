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
# Build CSS as a list of ordinary strings. This avoids multiline quote blocks,
# which can become unterminated if only part of the file is pasted into GitHub.
DASHBOARD_CSS = "\n".join([
    "<style>",
    ".stApp {",
    "    __RESUME_BACKGROUND__",
    "    background-size: cover;",
    "    background-position: center center;",
    "    background-attachment: fixed;",
    "    background-repeat: no-repeat;",
    "}",
    '[data-testid="stAppViewContainer"],',
    '[data-testid="stMain"],',
    ".main { background: transparent !important; }",
    ".block-container { max-width: 1450px; padding-top: 1.5rem; padding-bottom: 3rem; }",
    '[data-testid="stSidebar"] > div:first-child {',
    "    background: linear-gradient(180deg, rgba(4, 15, 43, .98), rgba(7, 31, 70, .96));",
    "    border-right: 1px solid rgba(72, 190, 255, .35);",
    "}",
    '[data-testid="stSidebar"] { color: #f4f8ff; }',
    '[data-testid="stSidebar"] [data-testid="stRadio"] label { color: #e6f3ff !important; font-weight: 650; }',
    '[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { color: #62ddff !important; }',
    ".brand-card { padding: 1.1rem .9rem; border-radius: 18px; background: linear-gradient(135deg, rgba(0,174,255,.22), rgba(95,86,255,.22)); border: 1px solid rgba(104,207,255,.45); margin-bottom: 1rem; box-shadow: 0 8px 28px rgba(0,0,0,.20); }",
    ".brand-title { font-size: 1.45rem; font-weight: 850; color: #ffffff; letter-spacing: .2px; }",
    ".brand-subtitle { color: #b9eaff; font-size: .82rem; margin-top: .25rem; }",
    ".hero-kicker { color: #7ee8ff; font-size: .78rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }",
    ".hero-title { color: #ffffff; font-size: clamp(2.2rem, 4vw, 3.35rem); line-height: 1.08; font-weight: 900; margin: .25rem 0 .55rem 0; text-shadow: 0 3px 24px rgba(0,0,0,.32); }",
    ".hero-copy { color: #d3eaff; font-size: 1.02rem; max-width: 850px; }",
    ".glass-card { background: rgba(5,20,52,.78); border: 1px solid rgba(110,203,255,.30); border-radius: 19px; padding: 1.15rem 1.25rem; box-shadow: 0 12px 36px rgba(0,0,0,.20); backdrop-filter: blur(9px); margin-bottom: .9rem; }",
    ".section-title { color: #ffffff; font-weight: 800; font-size: 1.22rem; }",
    ".muted-copy { color: #c4d9f0; }",
    'div[data-testid="stMetric"] { background: rgba(5,22,56,.80); border: 1px solid rgba(90,194,255,.32); padding: .8rem 1rem; border-radius: 15px; box-shadow: 0 8px 25px rgba(0,0,0,.12); }',
    'div[data-testid="stMetric"] label { color: #b9d8f8 !important; }',
    'div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #ffffff !important; }',
    ".stButton > button, .stDownloadButton > button { border: 0 !important; border-radius: 12px !important; color: #ffffff !important; font-weight: 800 !important; background: linear-gradient(100deg, #0878e8, #635bfa) !important; box-shadow: 0 7px 22px rgba(15,116,237,.25); min-height: 2.8rem; }",
    ".stButton > button:hover, .stDownloadButton > button:hover { border: 1px solid #8beaff !important; filter: brightness(1.12); }",
    '[data-testid="stFileUploader"] section { background: rgba(6,25,60,.78); border: 1.5px dashed rgba(94,207,255,.68); border-radius: 16px; }',
    '[data-testid="stFileUploader"] label, [data-testid="stFileUploader"] small { color: #e5f5ff !important; }',
    ".stProgress > div > div > div > div { background: linear-gradient(90deg, #04c7ff, #7867ff); }",
    '[data-testid="stDataFrame"] { background: rgba(4,18,46,.82); border: 1px solid rgba(111,197,255,.25); border-radius: 14px; }',
    "hr { border-color: rgba(141,205,255,.25) !important; }",
    "h1, h2, h3, p, label, .stMarkdown { text-shadow: 0 1px 9px rgba(0,0,0,.12); }",
    "</style>",
])
st.markdown(
    DASHBOARD_CSS.replace("__RESUME_BACKGROUND__", background_css),
    unsafe_allow_html=True,
)


# =============================================================================
# PREPROCESSING — must match the preprocessing used before TF-IDF in training
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
        raise RuntimeError(
            "The .doc reader is unavailable. Keep packages.txt with 'antiword' in the repository."
        )

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
    raise ValueError("Unsupported file type. Upload PDF, DOCX, DOC, or TXT.")


# =============================================================================
# LOAD SAVED MODEL + LABEL ENCODER
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


def classify_document(filename: str, file_bytes: bytes, model: Any, label_encoder: Any) -> dict:
    raw_text = extract_document_text(filename, file_bytes)
    cleaned_text = clean_resume_text(raw_text)
    word_count = len(cleaned_text.split())

    if not cleaned_text or word_count < MIN_WORD_COUNT:
        raise ValueError(
            f"Only {word_count} usable words were extracted. The file may be scanned, empty, or too short."
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
    low_confidence = top_probability < REVIEW_THRESHOLD

    return {
        "file_name": filename,
        "file_bytes": file_bytes,
        "predicted_category": predicted_category,
        "top_probability": top_probability,
        "word_count": word_count,
        "review_status": "Review required" if low_confidence else "Suggested category",
        "category_scores": {name: score for name, score in ranked},
    }


def safe_path_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", str(value)).strip(" ._")
    return cleaned[:100] or "Unsorted"


def make_classified_zip(results: list[dict]) -> bytes:
    """Package each original document under its predicted category folder."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, item in enumerate(results, start=1):
            category_folder = safe_path_part(item["predicted_category"])
            original_name = Path(item["file_name"]).name
            original_name = safe_path_part(original_name)
            archive_path = f"{category_folder}/{index:03d}_{original_name}"
            archive.writestr(archive_path, item["file_bytes"])
    output.seek(0)
    return output.getvalue()


def build_results_dataframe(results: list[dict]) -> pd.DataFrame:
    rows = []
    for item in results:
        rows.append(
            {
                "Resume": item["file_name"],
                "Predicted category": item["predicted_category"],
                "Top class probability": round(item["top_probability"], 4),
                "Extracted words": item["word_count"],
                "Status": item["review_status"],
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# LOAD MODEL + SESSION STATE
# =============================================================================
try:
    model, label_encoder = load_artifacts()
    available_categories = get_class_names(model, label_encoder)
except Exception as exc:
    st.error(f"Unable to load the trained classifier: {exc}")
    st.info(
        "Place best_resume_classifier.pkl and label_encoder.pkl in the same GitHub folder as app.py."
    )
    st.stop()

if "classification_results" not in st.session_state:
    st.session_state.classification_results = []
if "last_run_time" not in st.session_state:
    st.session_state.last_run_time = None


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand-card">
            <div style="font-size:2rem">📄</div>
            <div class="brand-title">Resume Classifier</div>
            <div class="brand-subtitle">Smart document intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Workspace")
    selected_section = st.radio(
        "Navigate",
        ["Resume Classification", "Prediction Results", "About the Model"],
        index=0,
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**AI ENGINE**")
    st.markdown("<span style='color:#83e8ff'>●</span> Logistic Regression + TF-IDF", unsafe_allow_html=True)
    st.markdown("**DOCUMENT INPUT**")
    st.caption("PDF  ·  DOCX  ·  DOC  ·  TXT")
    st.markdown("**WORKFLOW**")
    st.caption("Upload  →  Analyze  →  Classify  →  Download")
    st.divider()
    st.caption("Resume sorting workspace")


# =============================================================================
# SECTION 1 — UPLOAD AND PREDICT IN THE SAME PLACE
# =============================================================================
if selected_section == "Resume Classification":
    st.markdown('<div class="hero-kicker">AI-POWERED DOCUMENT SORTING</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">📄 Resume Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-copy">Upload resumes, identify their closest learned category, and download the classified documents in organized folders.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload your resume documents</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted-copy">Add one resume or a batch of resumes. Supported formats: PDF, DOCX, DOC and TXT.</div>',
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        "Drag and drop resumes here",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=True,
        key="resume_file_uploader",
        help="Text-based documents are supported. Scanned image-only PDFs need OCR before classification.",
    )
    metric_a, metric_b = st.columns(2)
    metric_a.metric("Files selected", len(uploaded_files) if uploaded_files else 0)
    metric_b.metric("Learned categories", len(available_categories))
    st.markdown('</div>', unsafe_allow_html=True)

    if len(available_categories) < 2:
        st.error("The saved model contains fewer than two classes. Please check the model and label encoder files.")

    classify_clicked = st.button(
        "🚀  Analyze & Classify Resumes",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    if classify_clicked and uploaded_files:
        new_results: list[dict] = []
        errors: list[tuple[str, str]] = []
        progress = st.progress(0, text="Preparing resume analysis...")
        status_area = st.empty()
        total_files = len(uploaded_files)

        # A clear processing state is shown while each uploaded document is handled.
        with st.spinner("Analyzing resume content and generating predictions..."):
            for index, uploaded_file in enumerate(uploaded_files, start=1):
                status_area.info(f"🔎 Analyzing: {uploaded_file.name}")
                try:
                    result = classify_document(
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        model,
                        label_encoder,
                    )
                    new_results.append(result)
                except Exception as exc:
                    errors.append((uploaded_file.name, str(exc)))

                progress.progress(
                    index / total_files,
                    text=f"Analyzed {index} of {total_files} document(s)",
                )

        st.session_state.classification_results = new_results
        st.session_state.last_run_time = pd.Timestamp.now().strftime("%d %b %Y, %I:%M:%S %p")
        status_area.empty()
        progress.empty()

        if new_results:
            st.success(f"✅ Classification complete — {len(new_results)} document(s) processed successfully.")
            st.balloons()
        if errors:
            st.warning("Some documents could not be classified:")
            for filename, message in errors:
                st.write(f"**{filename}:** {message}")

    # Prediction output is deliberately shown immediately below upload on this page.
    results = st.session_state.classification_results
    if results:
        st.markdown("## Classification Output")
        summary_df = build_results_dataframe(results)
        summary_display = summary_df.copy()
        summary_display["Top class probability"] = summary_display["Top class probability"].map(lambda x: f"{x:.1%}")
        st.dataframe(summary_display, use_container_width=True, hide_index=True)

        for index, item in enumerate(results, start=1):
            with st.container(border=True):
                left, middle, right = st.columns([2.3, 1.2, 1.2])
                left.markdown(f"**{index}. {item['file_name']}**")
                left.caption(f"Predicted category: {item['predicted_category']}")
                middle.metric("Model score", f"{item['top_probability']:.1%}")
                right.metric("Status", item["review_status"])
                if item["top_probability"] < REVIEW_THRESHOLD:
                    st.warning("Low model score. Verify this category manually before using the result.")
                file_ext = Path(item["file_name"]).suffix.lower().lstrip(".") or "txt"
                st.download_button(
                    "⬇️ Download this classified resume",
                    data=item["file_bytes"],
                    file_name=f"{safe_path_part(item['predicted_category'])}_{safe_path_part(Path(item['file_name']).name)}",
                    mime={"pdf": "application/pdf", "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "doc": "application/msword", "txt": "text/plain"}.get(file_ext, "application/octet-stream"),
                    key=f"download_classified_{index}_{safe_path_part(item['file_name'])}",
                    use_container_width=True,
                )

        st.markdown("### Download classified resumes")
        st.markdown(
            '<div class="muted-copy">Download all successfully classified original files. The ZIP is organized into folders using each predicted category.</div>',
            unsafe_allow_html=True,
        )
        zip_data = make_classified_zip(results)
        dl_col1, dl_col2 = st.columns([1.35, 1])
        with dl_col1:
            st.download_button(
                "📦  Download all classified resumes (ZIP)",
                data=zip_data,
                file_name="classified_resumes_by_category.zip",
                mime="application/zip",
                key="download_all_classified_zip_main",
                type="primary",
                use_container_width=True,
            )
        with dl_col2:
            csv_data = summary_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📊  Download classification report (CSV)",
                data=csv_data,
                file_name="resume_classification_report.csv",
                mime="text/csv",
                key="download_classification_csv_main",
                use_container_width=True,
            )

    st.markdown("## How it works")
    step1, step2, step3 = st.columns(3)
    with step1:
        st.markdown('<div class="glass-card"><div class="section-title">1. Upload</div><p class="muted-copy">Choose PDF, Word, or text resumes.</p></div>', unsafe_allow_html=True)
    with step2:
        st.markdown('<div class="glass-card"><div class="section-title">2. Analyze</div><p class="muted-copy">Extract resume text and apply matching preprocessing.</p></div>', unsafe_allow_html=True)
    with step3:
        st.markdown('<div class="glass-card"><div class="section-title">3. Classify & export</div><p class="muted-copy">View the suggested category and download organized files.</p></div>', unsafe_allow_html=True)

    st.info(
        "The current model was trained on four categories. It can accept different resume formats, "
        "but it cannot learn new job categories unless you add labelled examples and retrain it. "
        "Resumes outside the learned classes may be assigned to the closest known class, so verify predictions."
    )


# =============================================================================
# SECTION 2 — PREDICTION RESULTS
# =============================================================================
elif selected_section == "Prediction Results":
    st.markdown('<div class="hero-kicker">YOUR CLASSIFICATION HISTORY</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Prediction Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-copy">Review predictions and download the classified resumes.</div>', unsafe_allow_html=True)

    results = st.session_state.classification_results
    if not results:
        st.info("No results yet. Open Resume Classification to upload and classify documents.")
    else:
        summary_df = build_results_dataframe(results)
        total_files = len(results)
        review_count = sum(item["top_probability"] < REVIEW_THRESHOLD for item in results)
        avg_score = float(np.mean([item["top_probability"] for item in results]))

        m1, m2, m3 = st.columns(3)
        m1.metric("Classified documents", total_files)
        m2.metric("Flagged for review", review_count)
        m3.metric("Average top class score", f"{avg_score:.1%}")

        if st.session_state.last_run_time:
            st.caption(f"Latest run: {st.session_state.last_run_time}")

        display_df = summary_df.copy()
        display_df["Top class probability"] = display_df["Top class probability"].map(lambda x: f"{x:.1%}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("### Download center")
        zip_data = make_classified_zip(results)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "📦 Download classified resumes (ZIP)",
                data=zip_data,
                file_name="classified_resumes_by_category.zip",
                mime="application/zip",
                key="download_all_classified_zip_results",
                type="primary",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "📊 Download result report (CSV)",
                data=summary_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="resume_classification_report.csv",
                mime="text/csv",
                key="download_report_results",
                use_container_width=True,
            )

        st.markdown("### Category distribution")
        category_counts = summary_df["Predicted category"].value_counts().rename_axis("Category").reset_index(name="Documents")
        st.bar_chart(category_counts.set_index("Category"))

        if st.button("Clear current results", key="clear_results_button"):
            st.session_state.classification_results = []
            st.session_state.last_run_time = None
            st.rerun()


# =============================================================================
# SECTION 3 — ABOUT THE MODEL
# =============================================================================
elif selected_section == "About the Model":
    st.markdown('<div class="hero-kicker">MODEL INFORMATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">About the Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-copy">The text-classification engine used to organize uploaded resumes.</div>', unsafe_allow_html=True)

    info1, info2, info3 = st.columns(3)
    info1.metric("Algorithm", "Logistic Regression")
    info2.metric("Text features", "TF-IDF")
    info3.metric("Learned classes", len(available_categories))

    st.markdown("### Classification workflow")
    st.code(
        "Resume upload\n"
        "    ↓\n"
        "Text extraction (PDF / DOCX / DOC / TXT)\n"
        "    ↓\n"
        "Text normalization and contact-detail masking\n"
        "    ↓\n"
        "Saved TF-IDF vectorizer\n"
        "    ↓\n"
        "Logistic Regression classifier\n"
        "    ↓\n"
        "Predicted learned category + downloadable file",
        language="text",
    )

    st.markdown("### Model configuration")
    st.json(
        {
            "C": 0.1,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 3000,
            "selection_metric": "Macro F1",
        }
    )

    st.markdown("### Recorded model evaluation")
    e1, e2, e3 = st.columns(3)
    e1.metric("5-fold CV Macro F1", "98.57%")
    e2.metric("Holdout test Macro F1", "100%")
    e3.metric("Holdout test accuracy", "100%")
    st.caption("The holdout set contained 16 documents. These results do not guarantee performance on unseen resumes.")

    st.markdown("### Categories learned by this model")
    category_cols = st.columns(2)
    for index, category in enumerate(available_categories):
        with category_cols[index % 2]:
            st.markdown(
                f'<div class="glass-card"><span style="color:#7ee8ff;font-weight:800">●</span> '
                f'<span style="color:#ffffff;font-weight:700">{category}</span></div>',
                unsafe_allow_html=True,
            )

    st.warning(
        "This saved model currently recognizes only the categories listed above. "
        "It accepts resumes for many roles and formats, but it cannot reliably assign a new job category "
        "that was not included in training. To support more resume types, add labelled examples for each "
        "new category, retrain the model, and replace both saved model files."
    )
    st.markdown("### Responsible use")
    st.write(
        "Treat every output as a suggested document category. Verify low-scoring or unfamiliar resumes manually. "
        "This tool is for document organization, not candidate ranking or automated hiring decisions. "
        "Scanned PDFs require OCR before text extraction."
    )


# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.markdown(
    "<div style='text-align:center;color:#b8d5f1;font-size:.85rem;padding:.4rem'>"
    "📄 Resume Classifier &nbsp;•&nbsp; AI-assisted resume organization"
    "</div>",
    unsafe_allow_html=True,
)
