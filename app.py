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
st.caption("Paste an email's subject/body below and check whether it looks like spam.")

# --- Input ---
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

# --- Predict ---
if st.button("Check for spam", type="primary"):
    if not email_text.strip():
        st.warning("Please enter some email text first.")
    else:
        result = predict_email(email_text, model, vectorizer)

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

st.divider()
st.caption(
    "⚠️ This model is trained on a small demo dataset for learning purposes — "
    "see the README for how to swap in a real dataset."
)