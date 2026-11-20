from __future__ import annotations

import io
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from docx import Document


# -----------------------------------------------------------------------------
# Application configuration
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best_resume_classifier.pkl"
ENCODER_PATH = APP_DIR / "label_encoder.pkl"
REVIEW_THRESHOLD = 0.70
MIN_WORD_COUNT = 80

st.set_page_config(
    page_title="Resume Classification | Project 691",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A small amount of CSS to give all three sections one consistent dashboard look.
st.markdown(
    """
    <style>
        .block-container {padding-top: 1.8rem; padding-bottom: 2.5rem;}
        .main-title {font-size: 2.15rem; font-weight: 750; margin-bottom: .15rem;}
        .sub-title {color: #8792a2; margin-bottom: 1.2rem;}
        .section-card {
            padding: 1rem 1.15rem;
            border: 1px solid rgba(128,128,128,.25);
            border-radius: 14px;
            margin-bottom: .8rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,.20);
            padding: .8rem;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Text preprocessing
# IMPORTANT: Keep this function identical to the one used before TF-IDF
# during model training. It applies Unicode normalization, lowercasing, and
# masks email addresses, URLs, and phone numbers.
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Document text extraction
# Supported: PDF, DOCX, DOC (requires antiword), and TXT.
# Scanned/image-only PDFs are not OCR processed by this app.
# -----------------------------------------------------------------------------
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
            "Legacy .doc support is not installed. Add a packages.txt file "
            "containing 'antiword' to the repository root, then redeploy."
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
            message = result.stderr.strip() or "The .doc file could not be read."
            raise RuntimeError(message)

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

    raise ValueError("Unsupported format. Upload PDF, DOCX, DOC, or TXT files.")


# -----------------------------------------------------------------------------
# Load the saved model and encoder once per Streamlit session.
# The saved model is the full TF-IDF + Logistic Regression Pipeline.
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the trained resume classifier...")
def load_artifacts():
    missing_files = [
        path.name for path in (MODEL_PATH, ENCODER_PATH) if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Required model file(s) missing beside app.py: " + ", ".join(missing_files)
        )

    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    if not hasattr(model, "predict"):
        raise TypeError("The saved model does not provide a predict() method.")
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The saved model does not provide predict_proba(). "
            "This dashboard expects your trained Logistic Regression Pipeline."
        )

    return model, label_encoder


def decode_model_classes(model, label_encoder):
    """Map the integer class IDs used by the classifier to readable labels."""
    classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else None
    encoded_classes = getattr(classifier, "classes_", None)

    if encoded_classes is None:
        encoded_classes = getattr(model, "classes_", None)

    if encoded_classes is None:
        raise ValueError("Could not read the trained model's class labels.")

    encoded_classes = np.asarray(encoded_classes).astype(int)
    return label_encoder.inverse_transform(encoded_classes)


def classify_document(filename: str, file_bytes: bytes, model, label_encoder) -> dict:
    raw_text = extract_document_text(filename, file_bytes)
    cleaned_text = clean_resume_text(raw_text)
    word_count = len(cleaned_text.split())

    if not cleaned_text or word_count < MIN_WORD_COUNT:
        raise ValueError(
            f"Only {word_count} usable words were extracted. The document may be "
            "empty, scanned, or too short. No prediction was made."
        )

    # The Pipeline applies its saved TF-IDF transformation and classifier.
    predicted_encoded = int(np.asarray(model.predict([cleaned_text])).reshape(-1)[0])
    predicted_label = str(
        label_encoder.inverse_transform([predicted_encoded])[0]
    )

    probabilities = np.asarray(model.predict_proba([cleaned_text]))[0]
    class_names = decode_model_classes(model, label_encoder)

    ranked = sorted(
        [(str(name), float(score)) for name, score in zip(class_names, probabilities)],
        key=lambda item: item[1],
        reverse=True,
    )

    top_label, top_probability = ranked[0]

    return {
        "file_name": filename,
        "predicted_category": predicted_label,
        "top_probability": float(top_probability),
        "word_count": int(word_count),
        "review_status": (
            "Manual review recommended"
            if top_probability < REVIEW_THRESHOLD
            else "Suggested category — verify"
        ),
        "category_scores": {name: score for name, score in ranked},
    }


# -----------------------------------------------------------------------------
# Load artifacts and set up session state
# -----------------------------------------------------------------------------
try:
    model, label_encoder = load_artifacts()
    available_categories = [str(label) for label in label_encoder.classes_]
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.info(
        "Place best_resume_classifier.pkl and label_encoder.pkl in the same "
        "GitHub repository folder as app.py."
    )
    st.stop()

if "classification_results" not in st.session_state:
    st.session_state.classification_results = []

if "last_run_time" not in st.session_state:
    st.session_state.last_run_time = None


# -----------------------------------------------------------------------------
# Sidebar navigation — all three requested sections in one app.py
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📄 Resume Classifier")
    st.caption("Project 691 | HR document classification")
    selected_section = st.radio(
        "Dashboard Sections",
        ["Resume Classification", "Prediction Results", "About the Model"],
        index=0,
    )
    st.divider()
    st.markdown("**Selected model**")
    st.write("Logistic Regression")
    st.markdown("**Supported categories**")
    for category in available_categories:
        st.write(f"• {category}")
    st.divider()
    st.caption(
        "Uploaded documents are processed in this app session. This application "
        "does not save resume files to a database."
    )


# =============================================================================
# SECTION 1 — RESUME CLASSIFICATION
# =============================================================================
if selected_section == "Resume Classification":
    st.markdown('<div class="main-title">Resume Classification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Upload resume documents and receive a suggested professional category.</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Upload text-based resumes in PDF, DOCX, DOC, or TXT format. "
        "Scanned PDFs that contain only images need OCR before classification."
    )

    uploaded_files = st.file_uploader(
        "Choose resume file(s)",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=True,
        help="You can upload one resume or multiple resumes in one batch.",
        key="resume_file_uploader",
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.metric("Selected files", len(uploaded_files) if uploaded_files else 0)
    with col_b:
        st.metric("Available categories", len(available_categories))

    classify_clicked = st.button(
        "Classify Resumes",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    if classify_clicked and uploaded_files:
        new_results = []
        errors = []

        progress = st.progress(0, text="Starting classification...")
        with st.spinner("Extracting text and classifying uploaded resumes..."):
            total_files = len(uploaded_files)
            for index, uploaded_file in enumerate(uploaded_files, start=1):
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
                    text=f"Processed {index} of {total_files} file(s)",
                )

        # Keep the latest batch in the dashboard results section.
        st.session_state.classification_results = new_results
        st.session_state.last_run_time = pd.Timestamp.now().strftime("%d %b %Y, %I:%M:%S %p")
        progress.empty()

        if new_results:
            st.success(
                f"Classification completed for {len(new_results)} resume(s). "
                "Open Prediction Results to review the output."
            )
        if errors:
            st.warning("Some files could not be processed:")
            for filename, message in errors:
                st.write(f"**{filename}:** {message}")

    st.markdown("---")
    st.markdown("### How it works")
    step1, step2, step3 = st.columns(3)
    step1.markdown(
        "**1. Upload**\n\nSelect PDF, Word, or text resume documents."
    )
    step2.markdown(
        "**2. Extract and preprocess**\n\nText is extracted and cleaned before entering the saved TF-IDF pipeline."
    )
    step3.markdown(
        "**3. Predict**\n\nThe Logistic Regression model suggests one of its learned categories."
    )


# =============================================================================
# SECTION 2 — PREDICTION RESULTS
# =============================================================================
elif selected_section == "Prediction Results":
    st.markdown('<div class="main-title">Prediction Results</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Review predicted categories, model scores, and files that need manual checking.</div>',
        unsafe_allow_html=True,
    )

    results = st.session_state.classification_results

    if not results:
        st.info("No predictions are available yet. Go to Resume Classification and upload one or more files.")
    else:
        results_df = pd.DataFrame(
            [
                {
                    "File Name": item["file_name"],
                    "Predicted Category": item["predicted_category"],
                    "Top Class Probability": item["top_probability"],
                    "Extracted Words": item["word_count"],
                    "Review Status": item["review_status"],
                }
                for item in results
            ]
        )

        total_files = len(results)
        review_count = sum(
            item["top_probability"] < REVIEW_THRESHOLD for item in results
        )
        average_score = float(np.mean([item["top_probability"] for item in results]))

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Classified resumes", total_files)
        metric2.metric("Manual review suggested", review_count)
        metric3.metric("Average top-class probability", f"{average_score:.1%}")

        if st.session_state.last_run_time:
            st.caption(f"Latest classification run: {st.session_state.last_run_time}")

        st.markdown("### Classification summary")
        display_df = results_df.copy()
        display_df["Top Class Probability"] = display_df["Top Class Probability"].map(
            lambda value: f"{value:.1%}"
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("### Individual prediction details")
        selected_file = st.selectbox(
            "Choose a resume to inspect",
            options=[item["file_name"] for item in results],
        )
        selected_result = next(
            item for item in results if item["file_name"] == selected_file
        )

        detail1, detail2, detail3 = st.columns(3)
        detail1.metric("Suggested category", selected_result["predicted_category"])
        detail2.metric(
            "Top class probability",
            f"{selected_result['top_probability']:.1%}",
        )
        detail3.metric("Extracted words", f"{selected_result['word_count']:,}")

        if selected_result["top_probability"] < REVIEW_THRESHOLD:
            st.warning(
                "The top class probability is below the review threshold. "
                "Please check the document manually."
            )
        else:
            st.success(
                "A suggested category is available. Please verify the result before routing the document."
            )

        st.markdown("**Scores for each trained category**")
        for category, score in selected_result["category_scores"].items():
            st.write(f"{category}: {score:.1%}")
            st.progress(float(score))

        csv_bytes = results_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download prediction report (CSV)",
            data=csv_bytes,
            file_name="resume_classification_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if st.button("Clear prediction results", type="secondary"):
            st.session_state.classification_results = []
            st.session_state.last_run_time = None
            st.rerun()

        st.caption(
            "The displayed probability is the model's estimated class probability; it is not a guarantee that the prediction is correct."
        )


# =============================================================================
# SECTION 3 — ABOUT THE MODEL
# =============================================================================
elif selected_section == "About the Model":
    st.markdown('<div class="main-title">About the Model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Project 691 — Machine Learning-based Resume Classification</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Project objective")
    st.write(
        "The system uses Natural Language Processing and machine learning to suggest "
        "a category for an uploaded resume. It is intended to support document "
        "organization and HR document routing, with human verification."
    )

    st.markdown("### Selected machine learning model")
    info1, info2, info3 = st.columns(3)
    info1.metric("Algorithm", "Logistic Regression")
    info2.metric("Text representation", "TF-IDF")
    info3.metric("Trained categories", len(available_categories))

    st.markdown("### Model pipeline")
    st.code(
        "Uploaded resume\n"
        "      ↓\n"
        "Text extraction (PDF / DOCX / DOC / TXT)\n"
        "      ↓\n"
        "Text cleaning and normalization\n"
        "      ↓\n"
        "Saved TF-IDF vectorizer\n"
        "      ↓\n"
        "Trained Logistic Regression classifier\n"
        "      ↓\n"
        "Predicted resume category",
        language="text",
    )

    st.markdown("### Selected model hyperparameters")
    st.json(
        {
            "C": 0.1,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 3000,
        }
    )

    st.markdown("### Recorded evaluation results")
    eval1, eval2, eval3 = st.columns(3)
    eval1.metric("5-fold CV Macro F1", "98.57%")
    eval2.metric("Holdout Test Macro F1", "100%")
    eval3.metric("Holdout Test Accuracy", "100%")
    st.caption(
        "These are the results recorded in the project notebook. The holdout set contained only 16 documents, "
        "so these scores should not be interpreted as a guarantee of performance on future resumes."
    )

    st.markdown("### Categories available in the saved model")
    for category in available_categories:
        st.write(f"• {category}")

    st.markdown("### Important limitations and responsible use")
    st.warning(
        "The model can only predict categories represented in its training data. "
        "A financial document or an unrelated document may still be assigned to one "
        "of the known resume categories because no separate financial-document class "
        "is available in this model."
    )
    st.markdown(
        "- Treat every prediction as a suggestion that requires verification.\n"
        "- Review low-probability predictions and documents with poor text extraction.\n"
        "- Do not use this classifier to rank candidates, assess candidate quality, or automatically reject applicants.\n"
        "- Add representative labelled examples and retrain before introducing new document categories.\n"
        "- Scanned PDFs need OCR before they can be classified by this text-based pipeline."
    )

    st.markdown("### Privacy note")
    st.write(
        "This app processes uploaded files in memory for the current session and does not "
        "write the original resume files to a database. Deploy the app only in an environment "
        "that meets your organization's privacy and document-handling requirements."
    )


# -----------------------------------------------------------------------------
# Footer shared by all sections
# -----------------------------------------------------------------------------
st.divider()
st.caption(
    "Project 691 | Resume Classification System • Machine-learning predictions are decision support only and require human review."
)
