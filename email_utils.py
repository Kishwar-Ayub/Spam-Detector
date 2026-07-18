"""
Parse raw email files (.eml / .msg-as-text / raw RFC 822 text) into the
headers and body text the app needs.

Uses Python's built-in `email` module — no extra dependencies required.
"""

from email import message_from_bytes, message_from_string
from email.policy import default as default_policy


HEADERS_OF_INTEREST = [
    "From",
    "To",
    "Subject",
    "Date",
    "Message-ID",
    "Authentication-Results",
    "Received",
]


def parse_email_file(file_bytes: bytes) -> dict:
    """
    Parse raw email bytes (as read from an uploaded .eml file) into a dict:
        {
            "headers": {header_name: value_or_list_of_values, ...},
            "body": "plain text body used for spam prediction",
            "raw_subject": "...",
        }
    Received and Authentication-Results can legitimately appear multiple
    times (one per hop), so those are returned as a list of strings;
    everything else is a single string (or None if absent).
    """
    try:
        msg = message_from_bytes(file_bytes, policy=default_policy)
    except Exception:
        # Fall back to text parsing if it's not clean bytes
        msg = message_from_string(file_bytes.decode("utf-8", errors="replace"), policy=default_policy)

    headers = {}
    for name in HEADERS_OF_INTEREST:
        values = msg.get_all(name)
        if values is None:
            headers[name] = None
        elif name in ("Received", "Authentication-Results"):
            headers[name] = [str(v) for v in values]
        else:
            headers[name] = str(values[0])

    body = _extract_body(msg)

    return {
        "headers": headers,
        "body": body,
        "raw_subject": headers.get("Subject") or "",
    }


def _extract_body(msg) -> str:
    """
    Pull the best plain-text body out of a (possibly multipart) email.
    Falls back to stripping HTML tags if only an HTML part is present.
    """
    if msg.is_multipart():
        # Prefer a text/plain part
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and not part.is_attachment():
                try:
                    return part.get_content().strip()
                except Exception:
                    continue
        # Fall back to text/html, stripped of tags
        for part in msg.walk():
            if part.get_content_type() == "text/html" and not part.is_attachment():
                try:
                    import re
                    html = part.get_content()
                    return re.sub(r"<[^>]+>", " ", html).strip()
                except Exception:
                    continue
        return ""
    else:
        try:
            return msg.get_content().strip()
        except Exception:
            return str(msg.get_payload())


def combined_text_for_prediction(parsed: dict) -> str:
    """
    Build the text string fed into the spam classifier: subject + body,
    since both carry spam signal.
    """
    subject = parsed["raw_subject"] or ""
    body = parsed["body"] or ""
    return f"{subject}\n{body}".strip()


def authentication_summary(parsed: dict) -> str:
    """
    Quick human-readable read on SPF/DKIM/DMARC results, if present.
    Returns one of: 'pass', 'fail', 'mixed', 'unknown'.
    """
    auth_headers = parsed["headers"].get("Authentication-Results") or []
    if not auth_headers:
        return "unknown"

    joined = " ".join(auth_headers).lower()
    has_fail = "fail" in joined
    has_pass = "pass" in joined

    if has_fail and has_pass:
        return "mixed"
    if has_fail:
        return "fail"
    if has_pass:
        return "pass"
    return "unknown"
