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
from threat_intel import check_url_virustotal, check_domain_urlscan, check_domain_blacklist_mxtoolbox

FEATURE_ORDER = ["num_exclamations", "num_links", "num_currency", "caps_ratio"]

st.set_page_config(page_title="Spam Email Detector", page_icon="🛡️", layout="centered")

# --------------------------------------------------------------------------
# Design: a clean "scan report" look. Soft neutral background, one accent
# color for interactive elements, and a circular risk gauge as the single
# focal visual — it doubles as the clearest possible way to read the result
# (a number + color + position on a ring) rather than being decorative.
# Namespaced CSS (sd-*) to avoid clashing with Streamlit internals.
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(circle at 20% 0%, #161B2C 0%, #0B0E17 55%); }
[data-testid="stHeader"] { background: transparent; }

.sd-header {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.2rem;
}
.sd-header-icon {
    width: 42px; height: 42px;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #6366F1, #A855F7);
    border-radius: 12px;
    font-size: 1.3rem;
    box-shadow: 0 4px 16px rgba(99,102,241,0.35);
}
.sd-title { font-weight: 800; font-size: 1.65rem; color: #F1F5F9; }
.sd-subtitle { color: #8B93A7; font-size: 0.95rem; max-width: 560px; margin: 0.3rem 0 1.4rem 0; }

/* Report shell wrapping the result */
.sd-report {
    background: #141928;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 1.6rem 1.6rem 1.3rem 1.6rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.45);
    margin: 1.2rem 0 1.2rem 0;
}
.sd-report-top {
    display: flex;
    align-items: center;
    gap: 1.4rem;
    flex-wrap: wrap;
}
.sd-gauge-wrap { flex-shrink: 0; }
.sd-verdict-block { flex: 1; min-width: 180px; }
.sd-verdict-headline { font-weight: 800; font-size: 1.35rem; margin-bottom: 0.2rem; }
.sd-verdict-spam .sd-verdict-headline { color: #F87171; }
.sd-verdict-ham .sd-verdict-headline { color: #4ADE80; }
.sd-verdict-sub { color: #8B93A7; font-size: 0.9rem; }
.sd-risk-chip {
    display: inline-block;
    margin-top: 0.6rem;
    font-weight: 700;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
}
.sd-risk-low { background: rgba(74,222,128,0.14); color: #4ADE80; }
.sd-risk-medium { background: rgba(251,191,36,0.14); color: #FBBF24; }
.sd-risk-high { background: rgba(248,113,113,0.14); color: #F87171; }

/* Reasons panel */
.sd-card {
    background: #141928;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
    box-shadow: 0 6px 20px rgba(0,0,0,0.35);
}
.sd-card-title { font-weight: 700; font-size: 0.95rem; color: #F1F5F9; margin-bottom: 0.6rem; }
.sd-reason { display: flex; gap: 0.6rem; padding: 0.32rem 0; font-size: 0.92rem; color: #C6CCDB; }

.sd-row {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.83rem;
    color: #C6CCDB;
    padding: 0.35rem 0;
    border-bottom: 1px dashed rgba(255,255,255,0.08);
}
.sd-row:last-child { border-bottom: none; }
.sd-row b { color: #F1F5F9; font-family: 'Inter', sans-serif; }

.sd-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 0.14rem 0.5rem;
    border-radius: 6px;
    font-weight: 600;
    margin-right: 0.3rem;
}
.sd-badge-pass { background: rgba(74,222,128,0.14); color: #4ADE80; }
.sd-badge-fail { background: rgba(248,113,113,0.14); color: #F87171; }
.sd-badge-mixed { background: rgba(251,191,36,0.14); color: #FBBF24; }
.sd-badge-unknown { background: rgba(148,163,184,0.14); color: #94A3B8; }

.sd-footer {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #5B6478;
    text-align: center;
    margin-top: 2rem;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1, #A855F7);
    border: none;
    font-weight: 600;
    box-shadow: 0 4px 16px rgba(99,102,241,0.3);
}
div.stButton > button[kind="primary"]:hover { filter: brightness(1.12); }
div.stButton > button:not([kind="primary"]) {
    background-color: #1B2233;
    color: #E5E7EB;
    border: 1px solid rgba(255,255,255,0.08);
}

.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    font-weight: 600; font-size: 0.9rem;
    color: #8B93A7;
    background-color: #10141F;
    border-radius: 10px 10px 0 0;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background-color: #141928 !important;
    color: #A855F7 !important;
    border-bottom: 3px solid #A855F7 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background-color: #141928;
    border: 1px solid rgba(255,255,255,0.06);
    border-top: none;
    padding: 1.3rem 1.4rem 1.5rem 1.4rem;
    border-radius: 0 14px 14px 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

.stTextArea textarea {
    background-color: #10141F;
    color: #E5E7EB;
    border: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stFileUploaderDropzone"] {
    background-color: #10141F;
    border: 1px dashed rgba(255,255,255,0.15);
}
.stCaption, [data-testid="stCaptionContainer"] { color: #8B93A7 !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def svg_gauge(percent: float, size: int = 118) -> str:
    """A circular risk gauge — green/amber/red arc + big centered percentage."""
    percent = max(0, min(100, percent))
    if percent < 34:
        color = "#4ADE80"
    elif percent < 67:
        color = "#FBBF24"
    else:
        color = "#F87171"

    r = (size / 2) - 12
    circumference = 2 * 3.14159265 * r
    offset = circumference * (1 - percent / 100)
    cx = cy = size / 2

    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#232B3F" stroke-width="12"/>
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="12"
            stroke-dasharray="{circumference:.1f}" stroke-dashoffset="{offset:.1f}"
            stroke-linecap="round" transform="rotate(-90 {cx} {cy})"/>
        <text x="{cx}" y="{cy + 6}" text-anchor="middle" font-family="Inter, sans-serif"
            font-size="22" font-weight="800" fill="{color}">{percent:.0f}%</text>
    </svg>
    """


@st.cache_resource
def load_artifacts():
    model = joblib.load("model/spam_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, vectorizer, scaler


def predict_email(text: str, model, vectorizer, scaler) -> dict:
    cleaned = clean_text(text)
    tfidf = vectorizer.transform([cleaned])
    features = extract_features(text)
    raw_engineered = pd.DataFrame([features])[FEATURE_ORDER].values
    scaled_engineered = scaler.transform(raw_engineered)
    combined = hstack([tfidf, csr_matrix(scaled_engineered)])

    pred = model.predict(combined)[0]
    label = "SPAM" if pred == 1 else "HAM"

    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(combined)[0][1])

    return {"label": label, "spam_probability": proba, "features": features}


def risk_level(prob: float) -> tuple[str, str]:
    if prob < 0.34:
        return "Low risk", "sd-risk-low"
    if prob < 0.67:
        return "Medium risk", "sd-risk-medium"
    return "High risk", "sd-risk-high"


def explain_signals(features: dict, is_spam: bool) -> list[str]:
    reasons = []
    if features["num_exclamations"] >= 2:
        reasons.append(("❗", f"Uses urgent or excited punctuation ({features['num_exclamations']} \"!\" marks)"))
    if features["num_links"] >= 1:
        reasons.append(("🔗", f"Contains {features['num_links']} link(s)"))
    if features["caps_ratio"] > 0.15:
        reasons.append(("🔠", "Written largely in CAPITAL LETTERS"))
    if features["num_currency"] >= 1:
        reasons.append(("💰", "Mentions money, prizes, or currency symbols"))

    if not reasons:
        if is_spam:
            reasons.append(("🤖", "Flagged mainly on word choice and phrasing, not a specific red flag above"))
        else:
            reasons.append(("🙂", "No urgent language, suspicious links, or money-related phrases detected"))
    return reasons


def render_result(result: dict):
    is_spam = result["label"] == "SPAM"
    verdict_class = "sd-verdict-spam" if is_spam else "sd-verdict-ham"
    headline = "Flagged as spam" if is_spam else "Looks legitimate"
    subline = "Treat links and requests in this message with caution." if is_spam else "No major spam signals found."

    pct = (result["spam_probability"] or 0) * 100
    gauge_svg = svg_gauge(pct)
    tier, tier_class = risk_level(result["spam_probability"] or 0)

    st.markdown(
        f'<div class="sd-report {verdict_class}"><div class="sd-report-top">'
        f'<div class="sd-gauge-wrap">{gauge_svg}</div>'
        f'<div class="sd-verdict-block">'
        f'<div class="sd-verdict-headline">{headline}</div>'
        f'<div class="sd-verdict-sub">{subline}</div>'
        f'<span class="sd-risk-chip {tier_class}">{tier}</span>'
        f'</div></div></div>',
        unsafe_allow_html=True,
    )

    reasons = explain_signals(result["features"], is_spam)
    reasons_html = "".join(
        f'<div class="sd-reason"><span>{icon}</span><span>{text}</span></div>' for icon, text in reasons
    )
    st.markdown(
        f'<div class="sd-card"><div class="sd-card-title">Why we think this</div>{reasons_html}</div>',
        unsafe_allow_html=True,
    )


def render_headers(parsed: dict):
    headers = parsed["headers"]
    rows_html = ""
    for field in ["From", "To", "Subject", "Date", "Message-ID"]:
        value = headers.get(field) or "<i>not present</i>"
        rows_html += f'<div class="sd-row"><b>{field}:</b> {value}</div>'

    auth_status = authentication_summary(parsed)
    badge_class = {"pass": "sd-badge-pass", "fail": "sd-badge-fail",
                   "mixed": "sd-badge-mixed", "unknown": "sd-badge-unknown"}[auth_status]
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

    sb_key = st.secrets.get("SAFE_BROWSING_API_KEY", "")
    vt_key = st.secrets.get("VIRUSTOTAL_API_KEY", "")
    us_key = st.secrets.get("URLSCAN_API_KEY", "")
    active = [n for n, k in [("Google Safe Browsing", sb_key), ("VirusTotal", vt_key), ("urlscan.io", us_key)] if k]

    if not active:
        st.markdown(
            f'<div class="sd-card"><div class="sd-card-title">🔗 Links found ({len(urls)})</div>'
            f'<div class="sd-row">Live checks are off — add a Safe Browsing, VirusTotal, or '
            f'urlscan.io API key as a Streamlit secret to enable them. See the README.</div></div>',
            unsafe_allow_html=True,
        )
        with st.expander("View links (not yet checked)"):
            for u in urls:
                st.code(u, language=None)
        return

    with st.spinner("Checking links..."):
        rows_html = ""
        any_flagged = False
        for u in urls:
            badges = []
            if sb_key:
                sb = check_urls_safe_browsing([u], sb_key)
                if sb["error"]:
                    badges.append('<span class="sd-badge sd-badge-unknown">Safe Browsing: error</span>')
                elif u in sb["flagged_urls"]:
                    badges.append('<span class="sd-badge sd-badge-fail">Safe Browsing: flagged</span>')
                    any_flagged = True
                else:
                    badges.append('<span class="sd-badge sd-badge-pass">Safe Browsing: clean</span>')
            if vt_key:
                vt = check_url_virustotal(u, vt_key)
                if vt["error"]:
                    badges.append('<span class="sd-badge sd-badge-unknown">VirusTotal: error</span>')
                elif not vt["found"]:
                    badges.append('<span class="sd-badge sd-badge-unknown">VirusTotal: unseen</span>')
                elif vt["malicious"] > 0:
                    badges.append(f'<span class="sd-badge sd-badge-fail">VirusTotal: {vt["malicious"]}/{vt["total_engines"]} flagged</span>')
                    any_flagged = True
                else:
                    badges.append(f'<span class="sd-badge sd-badge-pass">VirusTotal: 0/{vt["total_engines"]} flagged</span>')
            if us_key:
                us = check_domain_urlscan(u, us_key)
                if us["error"]:
                    badges.append('<span class="sd-badge sd-badge-unknown">urlscan.io: error</span>')
                elif us["malicious_scan_count"] > 0:
                    badges.append(f'<span class="sd-badge sd-badge-fail">urlscan.io: {us["malicious_scan_count"]} malicious</span>')
                    any_flagged = True
                elif us["scan_count"] > 0:
                    badges.append(f'<span class="sd-badge sd-badge-pass">urlscan.io: {us["scan_count"]} clean</span>')
                else:
                    badges.append('<span class="sd-badge sd-badge-unknown">urlscan.io: no history</span>')
            rows_html += f'<div class="sd-row">{u}<br>{" ".join(badges)}</div>'

    title = "🔗 Links — threat found" if any_flagged else "🔗 Links — all sources clear"
    st.markdown(f'<div class="sd-card"><div class="sd-card-title">{title}</div>{rows_html}</div>', unsafe_allow_html=True)


def render_sender_domain_check(parsed: dict):
    mx_key = st.secrets.get("MXTOOLBOX_API_KEY", "")
    if not mx_key:
        return
    from_header = parsed["headers"].get("From") or ""
    if "@" not in from_header:
        return
    domain = from_header.split("@")[-1].strip().rstrip(">").strip()
    if not domain:
        return

    with st.spinner(f"Checking {domain} against email blacklists..."):
        result = check_domain_blacklist_mxtoolbox(domain, mx_key)

    if result["error"]:
        st.caption(f"Sender domain blacklist check unavailable: {result['error']}")
        return

    if result["listed_count"] > 0:
        st.markdown(
            f'<div class="sd-card"><div class="sd-card-title">🚩 Sender domain flagged</div>'
            f'<div class="sd-row"><b>{domain}</b> appears on {result["listed_count"]} blacklist(s): '
            f'{", ".join(result["listed_on"])}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="sd-card"><div class="sd-card-title">✅ Sender domain reputation</div>'
            f'<div class="sd-row"><b>{domain}</b> is not on any blacklists MXToolbox checked.</div></div>',
            unsafe_allow_html=True,
        )


# --- Load model ---
try:
    model, vectorizer, scaler = load_artifacts()
except FileNotFoundError:
    st.error(
        "No trained model found. Run `python train.py` first to create "
        "`model/spam_model.pkl`, `model/vectorizer.pkl`, and `model/scaler.pkl`, then reload this page."
    )
    st.stop()

# --- Header ---
st.markdown(
    '<div class="sd-header"><div class="sd-header-icon">🛡️</div>'
    '<div class="sd-title">Spam Email Detector</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sd-subtitle">Paste an email or upload a raw file for a scan report: '
    'a risk score, the reasons behind it, and live link/domain checks.</div>',
    unsafe_allow_html=True,
)

tab_paste, tab_upload = st.tabs(["✏️ Paste text", "📎 Upload email file (.eml)"])

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
        "Here's the doc we discussed: https://docs.google.com/document/d/1a2b3c — take a look when you can.",
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

    if st.button("Scan this message", type="primary", key="check_paste"):
        if not email_text.strip():
            st.warning("Please enter some email text first.")
        else:
            result = predict_email(email_text, model, vectorizer, scaler)
            render_result(result)
            render_link_check(email_text)

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
            render_sender_domain_check(parsed)

            prediction_text = combined_text_for_prediction(parsed)
            if not prediction_text.strip():
                st.warning("No subject or body text could be extracted to classify.")
            else:
                result = predict_email(prediction_text, model, vectorizer, scaler)
                render_result(result)

                auth_status = authentication_summary(parsed)
                if auth_status == "fail":
                    st.warning(
                        "⚠️ SPF/DKIM/DMARC authentication failed for this message — "
                        "an independent red flag, regardless of the verdict above."
                    )

                with st.expander("View extracted text used for the scan"):
                    st.text(prediction_text)

                render_link_check(prediction_text)

st.markdown(
    '<div class="sd-footer">Trained on a small demo dataset for learning purposes — '
    'see the README to swap in a real dataset.</div>',
    unsafe_allow_html=True,
)
