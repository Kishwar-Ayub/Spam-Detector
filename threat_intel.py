"""
Cross-checks URLs and sender domains against established threat-intel
platforms: VirusTotal, urlscan.io, and MXToolbox.

All three are optional — each function degrades gracefully (returns a
"not configured" result) if its API key isn't set. Keys are read from
Streamlit secrets by the caller and passed in here; nothing is hardcoded.

Rate limits (free tiers, subject to change — verify on each provider's site):
  - VirusTotal: ~4 requests/minute, non-commercial use only
  - urlscan.io: search API is generous; submitting *new* scans is separate
    and much more limited, so this module only searches existing scans
  - MXToolbox: free tier request count varies by plan; check your account
"""

import base64
import requests

VT_BASE = "https://www.virustotal.com/api/v3"
URLSCAN_SEARCH_URL = "https://urlscan.io/api/v1/search/"
MXTOOLBOX_BASE = "https://api.mxtoolbox.com/api/v1"


def _domain_from_url(url: str) -> str:
    """Best-effort domain extraction without a full URL parser dependency."""
    stripped = url.split("://", 1)[-1]
    stripped = stripped.split("/", 1)[0]
    stripped = stripped.split("@")[-1]  # strip any userinfo@ prefix
    return stripped.strip().lower()


# --------------------------------------------------------------------------
# VirusTotal — multi-engine URL reputation
# --------------------------------------------------------------------------
def check_url_virustotal(url: str, api_key: str, timeout: int = 12) -> dict:
    """
    Looks up an existing VirusTotal report for a URL (does not submit a new
    scan, to stay fast and within free-tier limits).

    Returns:
        {"checked": bool, "error": str|None, "malicious": int, "suspicious": int,
         "harmless": int, "total_engines": int, "found": bool}
    """
    result = {
        "checked": False, "error": None, "malicious": 0, "suspicious": 0,
        "harmless": 0, "total_engines": 0, "found": False,
    }

    if not api_key:
        result["error"] = "No VirusTotal API key configured."
        return result

    url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")

    try:
        resp = requests.get(
            f"{VT_BASE}/urls/{url_id}",
            headers={"x-apikey": api_key},
            timeout=timeout,
        )
    except requests.exceptions.RequestException as e:
        result["error"] = f"VirusTotal request failed: {e}"
        return result

    if resp.status_code == 404:
        # VirusTotal has never seen this URL — not an error, just no data
        result["checked"] = True
        result["found"] = False
        return result

    if resp.status_code != 200:
        result["error"] = f"VirusTotal returned status {resp.status_code}"
        return result

    stats = resp.json().get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
    result["checked"] = True
    result["found"] = True
    result["malicious"] = stats.get("malicious", 0)
    result["suspicious"] = stats.get("suspicious", 0)
    result["harmless"] = stats.get("harmless", 0)
    result["total_engines"] = sum(stats.values()) if stats else 0
    return result


# --------------------------------------------------------------------------
# urlscan.io — prior scan history for a domain (fast, no new-scan wait)
# --------------------------------------------------------------------------
def check_domain_urlscan(url_or_domain: str, api_key: str, timeout: int = 12) -> dict:
    """
    Searches urlscan.io's existing scan index for a domain. Does NOT submit
    a new scan (those take 30+ seconds and would block the UI).

    Returns:
        {"checked": bool, "error": str|None, "scan_count": int,
         "malicious_scan_count": int, "example_result_url": str|None}
    """
    result = {
        "checked": False, "error": None, "scan_count": 0,
        "malicious_scan_count": 0, "example_result_url": None,
    }

    if not api_key:
        result["error"] = "No urlscan.io API key configured."
        return result

    domain = _domain_from_url(url_or_domain)

    try:
        resp = requests.get(
            URLSCAN_SEARCH_URL,
            headers={"API-Key": api_key},
            params={"q": f"domain:{domain}", "size": 10},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        result["error"] = f"urlscan.io request failed: {e}"
        return result

    hits = data.get("results", [])
    result["checked"] = True
    result["scan_count"] = len(hits)

    malicious = 0
    example_url = None
    for hit in hits:
        verdicts = hit.get("verdicts", {}).get("overall", {})
        if verdicts.get("malicious"):
            malicious += 1
            if example_url is None:
                example_url = hit.get("result")

    result["malicious_scan_count"] = malicious
    result["example_result_url"] = example_url
    return result


# --------------------------------------------------------------------------
# MXToolbox — sender domain blacklist status
# --------------------------------------------------------------------------
def check_domain_blacklist_mxtoolbox(domain: str, api_key: str, timeout: int = 12) -> dict:
    """
    Checks whether a domain appears on any email blacklists via MXToolbox.

    Returns:
        {"checked": bool, "error": str|None, "listed_count": int,
         "listed_on": [names...]}
    """
    result = {"checked": False, "error": None, "listed_count": 0, "listed_on": []}

    if not api_key:
        result["error"] = "No MXToolbox API key configured."
        return result

    try:
        resp = requests.get(
            f"{MXTOOLBOX_BASE}/lookup/blacklist/{domain}",
            headers={"Authorization": api_key},
            timeout=timeout,
        )
    except requests.exceptions.RequestException as e:
        result["error"] = f"MXToolbox request failed: {e}"
        return result

    if resp.status_code == 401:
        result["error"] = "MXToolbox rejected the API key (check it's valid / plan includes API access)."
        return result
    if resp.status_code != 200:
        result["error"] = f"MXToolbox returned status {resp.status_code}"
        return result

    data = resp.json()
    failed = [item for item in data.get("Failed", [])]
    result["checked"] = True
    result["listed_count"] = len(failed)
    result["listed_on"] = [item.get("Name", "unknown") for item in failed]
    return result
