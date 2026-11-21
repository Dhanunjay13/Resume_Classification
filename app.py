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
