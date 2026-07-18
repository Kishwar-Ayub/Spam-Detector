"""
Extracts URLs from email text and checks them against Google Safe Browsing
in real time, so the app can flag known phishing/malware links even if the
text itself doesn't look obviously spammy.

Requires a free Google Safe Browsing API key:
  1. Go to https://console.cloud.google.com/
  2. Create a project (or use an existing one)
  3. Enable the "Safe Browsing API"
  4. Create an API key under APIs & Services > Credentials
  5. Store it as a Streamlit secret (see README) — never hardcode it here

Safe Browsing API terms: non-commercial use only. For commercial use,
Google's paid Web Risk API is the equivalent product.
"""

import re
import requests

URL_PATTERN = re.compile(
    r"(https?://[^\s<>\")]+"
    r"|www\.[^\s<>\")]+"
    r"|\b(?:[a-zA-Z0-9-]+\.)+(?:com|net|org|io|co|info|biz|ly|xyz|top|click|link|app)\b(?:/[^\s<>\")]*)?)",
    re.IGNORECASE,
)

SAFE_BROWSING_ENDPOINT = "https://safebrowsing.googleapis.com/v4/threatMatches:find"


def extract_urls(text: str) -> list[str]:
    """Find candidate URLs in a block of text. De-duplicated, order preserved."""
    if not text:
        return []
    found = URL_PATTERN.findall(text)
    seen = []
    for url in found:
        if url not in seen:
            seen.append(url)
    return seen


def check_urls_safe_browsing(urls: list[str], api_key: str, timeout: int = 8) -> dict:
    """
    Check a list of URLs against Google Safe Browsing.

    Returns:
        {
            "checked": True/False,          # whether the API call succeeded
            "error": "..." or None,
            "flagged_urls": {url: [threat_types]},   # URLs found unsafe
            "clean_urls": [url, ...],                 # URLs checked and found clean
        }
    """
    result = {"checked": False, "error": None, "flagged_urls": {}, "clean_urls": []}

    if not urls:
        result["checked"] = True
        return result

    if not api_key:
        result["error"] = "No Safe Browsing API key configured."
        return result

    body = {
        "client": {"clientId": "spam-detector-demo", "clientVersion": "1.0.0"},
        "threatInfo": {
            "threatTypes": [
                "MALWARE",
                "SOCIAL_ENGINEERING",
                "UNWANTED_SOFTWARE",
                "POTENTIALLY_HARMFUL_APPLICATION",
            ],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": u} for u in urls],
        },
    }

    try:
        resp = requests.post(
            SAFE_BROWSING_ENDPOINT,
            params={"key": api_key},
            json=body,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        result["error"] = f"Request to Safe Browsing failed: {e}"
        return result

    matches = data.get("matches", [])
    flagged = {}
    for m in matches:
        url = m.get("threat", {}).get("url")
        threat_type = m.get("threatType")
        if url:
            flagged.setdefault(url, []).append(threat_type)

    result["checked"] = True
    result["flagged_urls"] = flagged
    result["clean_urls"] = [u for u in urls if u not in flagged]
    return result
