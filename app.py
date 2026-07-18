"""
Streamlit UI for the spam email detector.

Run with:
    streamlit run app.py
"""

import streamlit as st
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from preprocess import clean_text, extract_features
from email_utils import parse_email_file, combined_text_for_prediction, authentication_summary

st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("model/spam_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer


def predict_email(text: str, model, vectorizer) -> dict:
    cleaned = clean_text(text)
    tfidf = vectorizer.transform([cleaned])
    engineered = pd.DataFrame([extract_features(text)]).values
    features = hstack([tfidf, csr_matrix(engineered)])

    pred = model.predict(features)[0]
    label = "SPAM" if pred == 1 else "HAM"

    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(features)[0][1])

    return {"label": label, "spam_probability": proba}


def render_result(result: dict):
    if result["label"] == "SPAM":
        st.error(f"🚨 This looks like **SPAM**")
    else:
        st.success(f"✅ This looks like **HAM** (not spam)")

    if result["spam_probability"] is not None:
        st.metric("Spam probability", f"{result['spam_probability']*100:.1f}%")
        st.progress(result["spam_probability"])
    else:
        st.caption(
            "Note: this model type doesn't output a probability score, "
            "only a hard label."
        )


def render_headers(parsed: dict):
    headers = parsed["headers"]
    st.subheader("📋 Email headers")

    simple_fields = ["From", "To", "Subject", "Date", "Message-ID"]
    for field in simple_fields:
        value = headers.get(field)
        st.markdown(f"**{field}:** {value if value else '_not present_'}")

    auth_status = authentication_summary(parsed)
    auth_icon = {"pass": "✅", "fail": "🚨", "mixed": "⚠️", "unknown": "❔"}[auth_status]
    st.markdown(f"**Authentication-Results:** {auth_icon} `{auth_status}`")
    auth_values = headers.get("Authentication-Results") or []
    if auth_values:
        with st.expander("View raw Authentication-Results header(s)"):
            for v in auth_values:
                st.code(v, language=None)
    else:
        st.caption("_No Authentication-Results header present_")

    received_values = headers.get("Received") or []
    with st.expander(f"View Received header(s) — {len(received_values)} hop(s)"):
        if received_values:
            for i, v in enumerate(received_values, 1):
                st.code(f"[{i}] {v}", language=None)
        else:
            st.caption("_No Received headers present_")


# --- Load model (with a friendly error if training hasn't run yet) ---
try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error(
        "No trained model found. Run `python train.py` first to create "
        "`model/spam_model.pkl` and `model/vectorizer.pkl`, then reload this page."
    )
    st.stop()

# --- Header ---
st.title("📧 AI Spam Email Detector")
st.caption("Paste email text, or upload a raw .eml file, and check whether it looks like spam.")

tab_paste, tab_upload = st.tabs(["✏️ Paste text", "📎 Upload email file (.eml)"])

# ---------------- Tab 1: paste text ----------------
with tab_paste:
    example_spam = "CONGRATULATIONS! You've WON a $5000 prize! Click here now: bit.ly/claim-now"
    example_ham = "Hey, can we move our meeting to 4pm tomorrow? Let me know if that works."

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Try a spam example"):
            st.session_state["email_text"] = example_spam
    with col2:
        if st.button("Try a ham example"):
            st.session_state["email_text"] = example_ham

    email_text = st.text_area(
        "Email text",
        value=st.session_state.get("email_text", ""),
        height=180,
        placeholder="Paste email subject + body here...",
    )

    if st.button("Check for spam", type="primary", key="check_paste"):
        if not email_text.strip():
            st.warning("Please enter some email text first.")
        else:
            result = predict_email(email_text, model, vectorizer)
            render_result(result)

# ---------------- Tab 2: upload .eml file ----------------
with tab_upload:
    st.caption(
        "Upload a raw email file (`.eml`) — most email clients let you export "
        "or 'Save As' a raw message. Headers (From, Subject, Date, Message-ID, "
        "Received, Authentication-Results) will be extracted and shown alongside "
        "the spam prediction."
    )
    uploaded_file = st.file_uploader("Choose a .eml file", type=["eml", "txt", "msg"])

    if uploaded_file is not None:
        try:
            parsed = parse_email_file(uploaded_file.read())
        except Exception as e:
            st.error(f"Couldn't parse this file as an email: {e}")
            parsed = None

        if parsed is not None:
            render_headers(parsed)
            st.divider()

            prediction_text = combined_text_for_prediction(parsed)
            if not prediction_text.strip():
                st.warning("No subject or body text could be extracted to classify.")
            else:
                result = predict_email(prediction_text, model, vectorizer)
                st.subheader("🔍 Spam prediction")
                render_result(result)

                auth_status = authentication_summary(parsed)
                if auth_status == "fail":
                    st.warning(
                        "⚠️ Note: SPF/DKIM/DMARC authentication failed for this "
                        "message — that's an independent red flag regardless of "
                        "the model's text-based prediction."
                    )

                with st.expander("View extracted text used for prediction"):
                    st.text(prediction_text)

st.divider()
st.caption(
    "⚠️ This model is trained on a small demo dataset for learning purposes — "
    "see the README for how to swap in a real dataset."
)
