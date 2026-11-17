"""Run with: streamlit run app/main.py"""
from pathlib import Path

import pandas as pd
import streamlit as st

from app.core import MODEL_DIR, split_paragraphs
from app.inference import HRPolicyClassifier

st.set_page_config(page_title="HR Policy Classifier", page_icon="📄", layout="wide")
st.title("HR Policy Document Classifier")
st.caption("Fine-tuned DistilGPT-2 classifies HR policy paragraphs using generated label tokens.")

with st.sidebar:
    st.header("Model status")
    st.code(str(MODEL_DIR), language="text")
    st.info("Train the model once before classifying documents.")
    st.markdown("**Training command**")
    st.code("python -m app.train", language="bash")

uploaded = st.file_uploader("Upload an HR policy text file", type=["txt"])
if uploaded is not None:
    try:
        document = uploaded.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError:
        st.error("Please upload a UTF-8 encoded .txt file.")
        st.stop()

    paragraphs = split_paragraphs(document)
    st.write(f"Found **{len(paragraphs)}** non-empty section(s). Separate sections with a blank line.")
    if not paragraphs:
        st.warning("The uploaded file does not contain any text paragraphs.")
        st.stop()

    if st.button("Classify sections", type="primary"):
        try:
            classifier = HRPolicyClassifier()
        except (FileNotFoundError, OSError, ValueError) as exc:
            st.error(str(exc))
            st.stop()

        rows = []
        with st.spinner("Classifying document sections..."):
            for index, paragraph in enumerate(paragraphs, start=1):
                prediction = classifier.classify_paragraph(paragraph)
                rows.append({
                    "Section": index,
                    "Category": prediction["category"],
                    "Department": prediction["department"],
                    "Confidence": f"{prediction['confidence']:.1%}",
                    "Text": paragraph,
                })
        st.success(f"Classified {len(rows)} section(s). Review the predictions before routing.")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.download_button(
            "Download classification results (CSV)",
            data=pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
            file_name="hr_policy_classifications.csv",
            mime="text/csv",
        )
else:
    st.write("Upload a `.txt` document. Use a blank line between policy paragraphs or sections.")

st.divider()
st.caption("Prototype only: confidence is a model score, not a guarantee. HR staff should verify classifications before acting on them.")
