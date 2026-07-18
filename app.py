"""
Streamlit UI for the spam email detector.

Run with:
    streamlit run app.py
"""

import streamlit as st
import random
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from preprocess import clean_text, extract_features
from email_utils import parse_email_file, combined_text_for_prediction, authentication_summary
from url_utils import extract_urls, check_urls_safe_browsing

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


def render_link_check(text: str):
    """Extract URLs from text and check them against Google Safe Browsing in real time."""
    urls = extract_urls(text)
    if not urls:
        return

    api_key = st.secrets.get("SAFE_BROWSING_API_KEY", "")

    st.subheader("🔗 Real-time link check")
    st.caption(f"Found {len(urls)} link(s) in this message.")

    if not api_key:
        st.info(
            "Add a free Google Safe Browsing API key as a Streamlit secret "
            "(`SAFE_BROWSING_API_KEY`) to enable live link reputation checks. "
            "See the README for setup steps."
        )
        with st.expander("Links found (not yet checked)"):
            for u in urls:
                st.code(u, language=None)
        return

    with st.spinner("Checking links against Google Safe Browsing..."):
        result = check_urls_safe_browsing(urls, api_key)

    if result["error"]:
        st.warning(f"Link check unavailable: {result['error']}")
        return

    if result["flagged_urls"]:
        st.error(f"🚨 {len(result['flagged_urls'])} link(s) flagged as unsafe!")
        for url, threats in result["flagged_urls"].items():
            st.markdown(f"- `{url}` — **{', '.join(threats)}**")
    else:
        st.success("✅ No links matched Google's known threat lists.")

    if result["clean_urls"]:
        with st.expander(f"All checked links ({len(urls)})"):
            for u in urls:
                status = "🚨 flagged" if u in result["flagged_urls"] else "✅ clean"
                st.markdown(f"- `{u}` — {status}")


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
    SPAM_EXAMPLES = [
        "CONGRATULATIONS! You've WON a $5000 prize! Click here now: bit.ly/claim-now",
        "URGENT: Your account will be suspended in 24 hours. Verify your details immediately: secure-verify-now.com",
        "You are pre-approved for a $10,000 loan with 0% interest! Apply now, no credit check: clck.it/loan-offer",
        "FREE iPhone 16 for the first 100 people who click this link!! Claim yours before it's gone: win-big-today.net",
        "Make $500 a day working from home, no experience needed! Start today: bit.ly/easy-money",
        "Your PayPal account has been LIMITED. Confirm your identity now or lose access permanently: paypal-verify-account.net",
        "HOT SINGLES in your area want to meet you tonight! Sign up free, no credit card required: bit.ly/meet-now",
    ]

    HAM_EXAMPLES = [
        "Hey, can we move our meeting to 4pm tomorrow? Let me know if that works.",
        "Please find attached the quarterly report you asked for. Let me know if you have questions.",
        "Reminder: your dentist appointment is scheduled for 10am next Tuesday.",
        "Thanks for your help with the onboarding docs, really appreciate it!",
        "Here's the recipe you asked for — hope you enjoy making it this weekend.",
        "Just checking in to see how the project plans are coming along.",
        "Happy birthday! Hope you have a wonderful day, let's grab coffee soon.",
    ]

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Try a spam example"):
            st.session_state["email_text"] = random.choice(SPAM_EXAMPLES)
    with col2:
        if st.button("Try a ham example"):
            st.session_state["email_text"] = random.choice(HAM_EXAMPLES)

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
            st.divider()
            render_link_check(email_text)

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
                render_link_check(prediction_text)

st.divider()
st.caption(
    "⚠️ This model is trained on a small demo dataset for learning purposes — "
    "see the README for how to swap in a real dataset."
)
