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

st.set_page_config(page_title="Spam Email Detector", page_icon="📮", layout="centered")

# --------------------------------------------------------------------------
# Design goal: clarity first. Plain type, a calm neutral background, one
# unmistakable verdict banner, and a plain-language "why" so the result
# never feels like a black box. Namespaced CSS (sd-*) to avoid clashing
# with Streamlit internals.
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
.stApp { background-color: #F6F3EC; }
[data-testid="stHeader"] { background: transparent; }

.sd-title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 700;
    font-size: 2.1rem;
    color: #1B2A4A;
    margin-bottom: 0.2rem;
}
.sd-subtitle {
    color: #57604f;
    font-size: 1rem;
    max-width: 560px;
    margin-bottom: 1.4rem;
}

/* Verdict banner: the one thing that must be instantly readable */
.sd-verdict {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    padding: 1.1rem 1.3rem;
    border-radius: 10px;
    margin: 1.2rem 0 0.6rem 0;
    border: 1px solid;
}
.sd-verdict-spam { background: #FCEBEA; border-color: #E4A29B; }
.sd-verdict-ham  { background: #E9F5EE; border-color: #A7D6BC; }
.sd-verdict-icon { font-size: 2.1rem; line-height: 1; }
.sd-verdict-text-main {
    font-size: 1.25rem;
    font-weight: 700;
}
.sd-verdict-spam .sd-verdict-text-main { color: #A63A2E; }
.sd-verdict-ham .sd-verdict-text-main { color: #2F6B4F; }
.sd-verdict-text-sub {
    font-size: 0.88rem;
    color: #4a4636;
    margin-top: 0.15rem;
}

.sd-risk-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #4a4636;
    margin: 0.4rem 0 1.2rem 0;
}
.sd-risk-chip {
    font-weight: 700;
    padding: 0.1rem 0.5rem;
    border-radius: 4px;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
}
.sd-risk-low { background: #E9F5EE; color: #2F6B4F; }
.sd-risk-medium { background: #FCF3E0; color: #92660E; }
.sd-risk-high { background: #FCEBEA; color: #A63A2E; }

/* Plain card used for "why", headers, and link results */
.sd-card {
    background: #FFFFFF;
    border: 1px solid #E4DFCF;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.sd-card-title {
    font-weight: 700;
    font-size: 0.98rem;
    color: #1B2A4A;
    margin-bottom: 0.5rem;
}
.sd-reason {
    display: flex;
    gap: 0.5rem;
    padding: 0.3rem 0;
    font-size: 0.92rem;
    color: #33301f;
}
.sd-reason-icon { flex-shrink: 0; }

.sd-row {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #2b2a22;
    padding: 0.3rem 0;
    border-bottom: 1px dashed #E4DFCF;
}
.sd-row:last-child { border-bottom: none; }
.sd-row b { color: #1B2A4A; font-family: 'IBM Plex Sans', sans-serif; }

.sd-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-weight: 600;
}
.sd-badge-pass { background: #E9F5EE; color: #2F6B4F; }
.sd-badge-fail { background: #FCEBEA; color: #A63A2E; }
.sd-badge-mixed { background: #FCF3E0; color: #92660E; }
.sd-badge-unknown { background: #EFECE0; color: #57604f; }

.sd-footer {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #8a8367;
    text-align: center;
    margin-top: 2rem;
}

div.stButton > button[kind="primary"] {
    background-color: #1B2A4A;
    border-color: #1B2A4A;
    font-weight: 600;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #142038;
    border-color: #142038;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load("model/spam_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer


def predict_email(text: str, model, vectorizer) -> dict:
    cleaned = clean_text(text)
    tfidf = vectorizer.transform([cleaned])
    features = extract_features(text)
    engineered = pd.DataFrame([features]).values
    combined = hstack([tfidf, csr_matrix(engineered)])

    pred = model.predict(combined)[0]
    label = "SPAM" if pred == 1 else "HAM"

    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(combined)[0][1])

    return {"label": label, "spam_probability": proba, "features": features}


def risk_level(prob: float) -> tuple[str, str]:
    """Turn a raw probability into a plain-language risk tier + CSS class."""
    if prob < 0.34:
        return "Low risk", "sd-risk-low"
    if prob < 0.67:
        return "Medium risk", "sd-risk-medium"
    return "High risk", "sd-risk-high"


def explain_signals(text: str, features: dict, is_spam: bool) -> list[str]:
    """Plain-language reasons behind the verdict, based on the same signals the model sees."""
    reasons = []
    if features["num_exclamations"] >= 2:
        reasons.append(("❗", f"Uses urgent or excited punctuation ({features['num_exclamations']} \"!\" marks)"))
    if features["num_links"] >= 1:
        reasons.append(("🔗", f"Contains {features['num_links']} link(s)"))
    if features["caps_ratio"] > 0.15:
        reasons.append(("🔠", "Written largely in CAPITAL LETTERS"))
    if features["num_currency"] >= 1:
        reasons.append(("💰", "Mentions money, prizes, or currency symbols"))
    if features["length"] < 60:
        reasons.append(("✂️", "Very short message"))

    if not reasons:
        if is_spam:
            reasons.append(("🤖", "Flagged mainly on word choice and phrasing, not a specific red flag above"))
        else:
            reasons.append(("🙂", "No urgent language, suspicious links, or money-related phrases detected"))

    return reasons


def render_result(result: dict):
    """The one thing on this page that must be instantly, unambiguously readable."""
    is_spam = result["label"] == "SPAM"
    verdict_class = "sd-verdict-spam" if is_spam else "sd-verdict-ham"
    icon = "🚨" if is_spam else "✅"
    headline = "This looks like SPAM" if is_spam else "This looks like a normal email"
    subline = "Treat links and requests in this message with caution." if is_spam else "No major spam signals found."

    st.markdown(
        f'<div class="sd-verdict {verdict_class}">'
        f'<div class="sd-verdict-icon">{icon}</div>'
        f'<div><div class="sd-verdict-text-main">{headline}</div>'
        f'<div class="sd-verdict-text-sub">{subline}</div></div></div>',
        unsafe_allow_html=True,
    )

    if result["spam_probability"] is not None:
        pct = result["spam_probability"] * 100
        tier, tier_class = risk_level(result["spam_probability"])
        st.markdown(
            f'<div class="sd-risk-row">Spam likelihood: <b>{pct:.0f}%</b>'
            f'<span class="sd-risk-chip {tier_class}">{tier}</span></div>',
            unsafe_allow_html=True,
        )
        st.progress(result["spam_probability"])
    else:
        st.caption("This model type reports a label only, no confidence score.")

    reasons = explain_signals(st.session_state.get("_last_text", ""), result["features"], is_spam)
    reasons_html = "".join(
        f'<div class="sd-reason"><span class="sd-reason-icon">{icon}</span><span>{text}</span></div>'
        for icon, text in reasons
    )
    st.markdown(
        f'<div class="sd-card"><div class="sd-card-title">Why we think this</div>{reasons_html}</div>',
        unsafe_allow_html=True,
    )


def render_headers(parsed: dict):
    headers = parsed["headers"]

    rows_html = ""
    simple_fields = ["From", "To", "Subject", "Date", "Message-ID"]
    for field in simple_fields:
        value = headers.get(field) or "<i>not present</i>"
        rows_html += f'<div class="sd-row"><b>{field}:</b> {value}</div>'

    auth_status = authentication_summary(parsed)
    badge_class = {
        "pass": "sd-badge-pass", "fail": "sd-badge-fail",
        "mixed": "sd-badge-mixed", "unknown": "sd-badge-unknown",
    }[auth_status]
    rows_html += (
        f'<div class="sd-row"><b>Authentication-Results:</b> '
        f'<span class="sd-badge {badge_class}">{auth_status}</span></div>'
    )

    st.markdown(
        f'<div class="sd-card"><div class="sd-card-title">📋 Email headers</div>{rows_html}</div>',
        unsafe_allow_html=True,
    )

    auth_values = headers.get("Authentication-Results") or []
    if auth_values:
        with st.expander("View raw Authentication-Results header(s)"):
            for v in auth_values:
                st.code(v, language=None)

    received_values = headers.get("Received") or []
    with st.expander(f"View Received header(s) — {len(received_values)} hop(s)"):
        if received_values:
            for i, v in enumerate(received_values, 1):
                st.code(f"[{i}] {v}", language=None)
        else:
            st.caption("_No Received headers present_")


def render_link_check(text: str):
    urls = extract_urls(text)
    if not urls:
        return

    api_key = st.secrets.get("SAFE_BROWSING_API_KEY", "")

    if not api_key:
        st.markdown(
            f'<div class="sd-card"><div class="sd-card-title">🔗 Links found ({len(urls)})</div>'
            f'<div class="sd-row">Live safety checks are off — add a free Google Safe Browsing '
            f'API key as a Streamlit secret to enable this. See the README.</div></div>',
            unsafe_allow_html=True,
        )
        with st.expander("View links (not yet checked)"):
            for u in urls:
                st.code(u, language=None)
        return

    with st.spinner("Checking links against Google Safe Browsing..."):
        result = check_urls_safe_browsing(urls, api_key)

    if result["error"]:
        st.warning(f"Link check unavailable: {result['error']}")
        return

    rows_html = ""
    for u in urls:
        flagged = u in result["flagged_urls"]
        badge_class = "sd-badge-fail" if flagged else "sd-badge-pass"
        status = "flagged" if flagged else "clean"
        rows_html += f'<div class="sd-row">{u} <span class="sd-badge {badge_class}">{status}</span></div>'

    title = "🔗 Links — threat found" if result["flagged_urls"] else "🔗 Links — all clear"
    st.markdown(
        f'<div class="sd-card"><div class="sd-card-title">{title}</div>{rows_html}</div>',
        unsafe_allow_html=True,
    )
    if result["flagged_urls"]:
        for url, threats in result["flagged_urls"].items():
            st.caption(f"`{url}` — {', '.join(threats)}")


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
st.markdown('<div class="sd-title">🛡️ Spam Email Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sd-subtitle">Paste an email or upload a raw file. You\'ll get a clear verdict, '
    'a risk level, and the plain-language reasons behind it.</div>',
    unsafe_allow_html=True,
)

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
        if st.button("Try a normal example"):
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
            st.session_state["_last_text"] = email_text
            result = predict_email(email_text, model, vectorizer)
            render_result(result)
            render_link_check(email_text)

# ---------------- Tab 2: upload .eml file ----------------
with tab_upload:
    st.caption(
        "Upload a raw email file (`.eml`) — most email clients let you export "
        "or 'Save As' a raw message. Headers and any links will be checked too."
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

            prediction_text = combined_text_for_prediction(parsed)
            if not prediction_text.strip():
                st.warning("No subject or body text could be extracted to classify.")
            else:
                st.session_state["_last_text"] = prediction_text
                result = predict_email(prediction_text, model, vectorizer)
                render_result(result)

                auth_status = authentication_summary(parsed)
                if auth_status == "fail":
                    st.warning(
                        "⚠️ SPF/DKIM/DMARC authentication failed for this message — "
                        "an independent red flag, regardless of the verdict above."
                    )

                with st.expander("View extracted text used for the check"):
                    st.text(prediction_text)

                render_link_check(prediction_text)

st.markdown(
    '<div class="sd-footer">Trained on a small demo dataset for learning purposes — '
    'see the README to swap in a real dataset.</div>',
    unsafe_allow_html=True,
)
