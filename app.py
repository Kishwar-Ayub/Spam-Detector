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

st.set_page_config(page_title="The Mail Room — Spam Inspector", page_icon="📮", layout="centered")

# --------------------------------------------------------------------------
# Visual identity: a case-file / mail-inspector's desk aesthetic.
# Ink navy + kraft paper + a brass rule, with hand-stamped verdicts as the
# signature element. Namespaced CSS classes (mr-*) to avoid clashing with
# Streamlit's own internals.
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Special+Elite&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.mr-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-size: 0.72rem;
    color: #A63A2E;
    font-weight: 600;
    margin-bottom: 0.25rem;
}

.mr-title {
    font-family: 'Special Elite', monospace;
    font-size: 2.4rem;
    color: #1B2A4A;
    line-height: 1.15;
    margin: 0 0 0.35rem 0;
}

.mr-subtitle {
    font-family: 'IBM Plex Sans', sans-serif;
    color: #4a4636;
    font-size: 1rem;
    max-width: 560px;
    margin-bottom: 0.5rem;
}

.mr-rule {
    border: none;
    border-top: 3px double #B08D57;
    margin: 1.1rem 0 1.6rem 0;
}

/* Ink-stamp verdict badge — the signature element */
.mr-stamp-wrap {
    display: flex;
    justify-content: center;
    margin: 1.4rem 0;
}
.mr-stamp {
    font-family: 'Special Elite', monospace;
    display: inline-block;
    padding: 0.85rem 1.8rem;
    border: 4px double;
    border-radius: 4px;
    transform: rotate(-3deg);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 1.3rem;
    text-align: center;
    opacity: 0.92;
    box-shadow: 0 0 0 2px rgba(0,0,0,0.02);
}
.mr-stamp-spam {
    color: #A63A2E;
    border-color: #A63A2E;
    background: rgba(166, 58, 46, 0.06);
}
.mr-stamp-ham {
    color: #2F6B4F;
    border-color: #2F6B4F;
    background: rgba(47, 107, 79, 0.06);
}
.mr-stamp-sub {
    font-family: 'IBM Plex Mono', monospace;
    text-align: center;
    font-size: 0.8rem;
    color: #4a4636;
    margin-top: -0.6rem;
    margin-bottom: 1rem;
}

/* Ledger-style card for headers / link results */
.mr-card {
    background: #FAF7EE;
    border: 1px solid #cfc6a8;
    border-left: 4px solid #B08D57;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.mr-card-title {
    font-family: 'Special Elite', monospace;
    font-size: 1.05rem;
    color: #1B2A4A;
    margin-bottom: 0.6rem;
}
.mr-row {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.87rem;
    color: #2b2a22;
    padding: 0.25rem 0;
    border-bottom: 1px dashed #ded4b2;
}
.mr-row:last-child { border-bottom: none; }
.mr-row b {
    color: #1B2A4A;
    font-family: 'IBM Plex Mono', monospace;
}

.mr-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    font-weight: 600;
}
.mr-badge-pass { background: rgba(47,107,79,0.12); color: #2F6B4F; }
.mr-badge-fail { background: rgba(166,58,46,0.12); color: #A63A2E; }
.mr-badge-mixed { background: rgba(176,141,87,0.18); color: #8a6a2e; }
.mr-badge-unknown { background: rgba(75,70,54,0.1); color: #4a4636; }

.mr-footer {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #8a8367;
    text-align: center;
    margin-top: 2rem;
}

/* Buttons — wax-seal red primary, quieter secondary */
div.stButton > button[kind="primary"] {
    background-color: #A63A2E;
    border-color: #A63A2E;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    letter-spacing: 0.02em;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #8c2f25;
    border-color: #8c2f25;
}
div.stButton > button:not([kind="primary"]) {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
}

/* Tabs styled like folder tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    background-color: #e2dabf;
    border-radius: 6px 6px 0 0;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background-color: #FAF7EE !important;
    border-bottom: 3px solid #A63A2E !important;
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
    engineered = pd.DataFrame([extract_features(text)]).values
    features = hstack([tfidf, csr_matrix(engineered)])

    pred = model.predict(features)[0]
    label = "SPAM" if pred == 1 else "HAM"

    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(features)[0][1])

    return {"label": label, "spam_probability": proba}


def render_result(result: dict):
    """Renders the verdict as a hand-stamped badge — the page's signature element."""
    is_spam = result["label"] == "SPAM"
    stamp_class = "mr-stamp-spam" if is_spam else "mr-stamp-ham"
    stamp_text = "🚫 Flagged: Spam" if is_spam else "✅ Cleared: Ham"

    st.markdown(
        f'<div class="mr-stamp-wrap"><div class="mr-stamp {stamp_class}">{stamp_text}</div></div>',
        unsafe_allow_html=True,
    )

    if result["spam_probability"] is not None:
        pct = result["spam_probability"] * 100
        st.markdown(
            f'<div class="mr-stamp-sub">confidence: spam probability {pct:.1f}%</div>',
            unsafe_allow_html=True,
        )
        st.progress(result["spam_probability"])
    else:
        st.markdown(
            '<div class="mr-stamp-sub">this model type reports a label only, no probability score</div>',
            unsafe_allow_html=True,
        )


def render_headers(parsed: dict):
    headers = parsed["headers"]

    rows_html = ""
    simple_fields = ["From", "To", "Subject", "Date", "Message-ID"]
    for field in simple_fields:
        value = headers.get(field) or "<i>not present</i>"
        rows_html += f'<div class="mr-row"><b>{field}:</b> {value}</div>'

    auth_status = authentication_summary(parsed)
    badge_class = {
        "pass": "mr-badge-pass",
        "fail": "mr-badge-fail",
        "mixed": "mr-badge-mixed",
        "unknown": "mr-badge-unknown",
    }[auth_status]
    rows_html += (
        f'<div class="mr-row"><b>Authentication-Results:</b> '
        f'<span class="mr-badge {badge_class}">{auth_status}</span></div>'
    )

    st.markdown(
        f'<div class="mr-card"><div class="mr-card-title">📋 Case file — header record</div>{rows_html}</div>',
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
    """Extract URLs from text and check them against Google Safe Browsing in real time."""
    urls = extract_urls(text)
    if not urls:
        return

    api_key = st.secrets.get("SAFE_BROWSING_API_KEY", "")

    if not api_key:
        st.markdown(
            f'<div class="mr-card"><div class="mr-card-title">🔗 Link watch list ({len(urls)} found)</div>'
            f'<div class="mr-row">Live checks are off — add a free Safe Browsing API key as a Streamlit '
            f'secret to enable this. See the README.</div></div>',
            unsafe_allow_html=True,
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

    rows_html = ""
    for u in urls:
        flagged = u in result["flagged_urls"]
        badge_class = "mr-badge-fail" if flagged else "mr-badge-pass"
        status = "flagged" if flagged else "clean"
        rows_html += f'<div class="mr-row">{u} <span class="mr-badge {badge_class}">{status}</span></div>'

    title = "🔗 Link watch list — threat found" if result["flagged_urls"] else "🔗 Link watch list — all clear"
    st.markdown(
        f'<div class="mr-card"><div class="mr-card-title">{title}</div>{rows_html}</div>',
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

# --- Hero ---
st.markdown('<div class="mr-eyebrow">Case File // Inbox Division</div>', unsafe_allow_html=True)
st.markdown('<h1 class="mr-title">The Mail Room</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="mr-subtitle">Submit a message for inspection. We check the language, '
    'the header trail, and any links it carries — then stamp a verdict.</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="mr-rule">', unsafe_allow_html=True)

tab_paste, tab_upload = st.tabs(["📝 New submission", "📁 Upload case file (.eml)"])

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
        if st.button("🎲 Pull a spam sample"):
            st.session_state["email_text"] = random.choice(SPAM_EXAMPLES)
    with col2:
        if st.button("🎲 Pull a clean sample"):
            st.session_state["email_text"] = random.choice(HAM_EXAMPLES)

    email_text = st.text_area(
        "Message text",
        value=st.session_state.get("email_text", ""),
        height=180,
        placeholder="Paste the email subject + body here...",
        label_visibility="collapsed",
    )

    if st.button("Inspect this message", type="primary", key="check_paste"):
        if not email_text.strip():
            st.warning("Please enter some email text first.")
        else:
            result = predict_email(email_text, model, vectorizer)
            render_result(result)
            render_link_check(email_text)

# ---------------- Tab 2: upload .eml file ----------------
with tab_upload:
    st.caption(
        "Upload a raw email file (`.eml`) — most email clients let you export "
        "or 'Save As' a raw message. We'll pull the header trail (From, Subject, "
        "Date, Message-ID, Received, Authentication-Results) and stamp a verdict."
    )
    uploaded_file = st.file_uploader("Choose a .eml file", type=["eml", "txt", "msg"], label_visibility="collapsed")

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
                result = predict_email(prediction_text, model, vectorizer)
                render_result(result)

                auth_status = authentication_summary(parsed)
                if auth_status == "fail":
                    st.warning(
                        "⚠️ SPF/DKIM/DMARC authentication failed for this message — "
                        "an independent red flag, regardless of the text verdict above."
                    )

                with st.expander("View extracted text used for inspection"):
                    st.text(prediction_text)

                render_link_check(prediction_text)

st.markdown(
    '<div class="mr-footer">Trained on a small demo dataset for learning purposes — '
    'see the README to swap in a real dataset.</div>',
    unsafe_allow_html=True,
)
