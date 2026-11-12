"""
Web T.R.A.C.E — Two-Stage Hybrid Inference Pipeline (two_stage.py).

Stage 1: Instant URL-only evaluation (<15ms, zero scraping, no network egress).
Stage 2: Conditional HTML deep inspection triggered ONLY for ambiguous URLs
         (0.30 <= P_phishing <= 0.70) with strict SSRF protection and a 3-second timeout.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

HTML_STAGE2_FEATURE_COLS: list[str] = [
    "LineOfCode",
    "HasTitle",
    "DomainTitleMatchScore",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "NoOfURLRedirect",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfExternalRef",
]

# SSRF Blocklist (IPv4 & IPv6 private/reserved ranges)
PRIVATE_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local / Cloud metadata
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def is_ssrf_safe(hostname: str) -> tuple[bool, str]:
    """
    Validates that the target hostname does not resolve to private,
    loopback, or cloud-metadata IP addresses.
    """
    clean_host = hostname.split(":")[0].strip().lower()
    if clean_host in ("localhost", "127.0.0.1", "::1"):
        return False, "Target is localhost loopback"

    try:
        _, _, ip_list = socket.gethostbyname_ex(clean_host)
    except Exception as e:
        return False, f"DNS resolution failed: {e}"

    for ip_str in ip_list:
        try:
            ip_obj = ipaddress.ip_address(ip_str)
            for net in PRIVATE_NETWORKS:
                if ip_obj in net:
                    return False, f"Resolved IP {ip_str} falls within blocked network {net}"
        except ValueError:
            return False, f"Invalid IP address format: {ip_str}"

    return True, "Safe public IP"


def fetch_page_secure(url: str, timeout: float = 3.0, max_bytes: int = 1024 * 1024) -> Optional[str]:
    """
    Safely fetches target HTML with SSRF protection, size caps, and strict timeouts.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if not hostname:
        return None

    safe, reason = is_ssrf_safe(hostname)
    if not safe:
        print(f"[SSRF Protection] Blocked fetch to {url}: {reason}")
        return None

    try:
        session = requests.Session()
        session.max_redirects = 3
        headers = {"User-Agent": "WebTRACE-Security-Audit/2.0"}

        with session.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return None

            raw_bytes = bytearray()
            for chunk in resp.iter_content(chunk_size=4096):
                raw_bytes.extend(chunk)
                if len(raw_bytes) > max_bytes:
                    break
            return raw_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        print(f"[Stage 2 Fetch] Error fetching {url}: {e}")
        return None


def extract_html_features_from_source(html: str, target_url: str) -> dict[str, float]:
    """
    Extracts Stage 2 HTML features matching PhiUSIIL columns from fetched page source.
    """
    if not html:
        return {col: 0.0 for col in HTML_STAGE2_FEATURE_COLS}

    parsed = urlparse(target_url)
    domain = parsed.hostname or ""
    domain_clean = re.sub(r"^www\d*\.", "", domain.lower())

    lines = html.splitlines()
    loc = float(len(lines))

    # Title extraction
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else ""
    has_title = 1.0 if len(title) > 0 else 0.0

    # Title domain match score
    domain_title_match = 0.0
    if title and domain_clean:
        title_lower = title.lower()
        dom_parts = [p for p in domain_clean.split(".") if len(p) > 2]
        match_count = sum(1 for p in dom_parts if p in title_lower)
        domain_title_match = match_count / max(1, len(dom_parts))

    # Favicon
    has_favicon = 1.0 if re.search(r"<link[^>]+rel=[\"'][^\"']*icon[^\"']*[\"']", html, re.IGNORECASE) else 0.0

    # Password field
    has_password = 1.0 if re.search(r"<input[^>]+type=[\"']password[\"']", html, re.IGNORECASE) else 0.0

    # External form submit
    has_ext_form = 0.0
    for form in re.finditer(r"<form[^>]+action=[\"']([^\"']+)[\"']", html, re.IGNORECASE):
        action = form.group(1).strip()
        if action.startswith(("http://", "https://")):
            act_host = urlparse(action).hostname or ""
            if act_host and domain_clean not in act_host.lower():
                has_ext_form = 1.0
                break

    # Hidden fields & Iframes
    has_hidden = 1.0 if re.search(r"<input[^>]+type=[\"']hidden[\"']", html, re.IGNORECASE) else 0.0
    iframes = len(re.findall(r"<iframe", html, re.IGNORECASE))

    # External references
    all_hrefs = re.findall(r"href=[\"']([^\"']+)[\"']", html, re.IGNORECASE)
    ext_refs = 0
    for href in all_hrefs:
        if href.startswith(("http://", "https://")):
            h_host = urlparse(href).hostname or ""
            if h_host and domain_clean not in h_host.lower():
                ext_refs += 1

    # Financial keywords
    lower_html = html.lower()
    has_bank = 1.0 if "bank" in lower_html else 0.0
    has_pay = 1.0 if "pay" in lower_html or "paypal" in lower_html else 0.0
    has_crypto = 1.0 if any(k in lower_html for k in ("crypto", "bitcoin", "ethereum", "wallet")) else 0.0

    # Resource counts
    no_of_images = float(len(re.findall(r"<img", html, re.IGNORECASE)))
    no_of_css = float(len(re.findall(r"<link[^>]+rel=[\"']stylesheet[\"']", html, re.IGNORECASE)))
    no_of_js = float(len(re.findall(r"<script", html, re.IGNORECASE)))

    return {
        "LineOfCode": loc,
        "HasTitle": has_title,
        "DomainTitleMatchScore": domain_title_match,
        "HasFavicon": has_favicon,
        "Robots": 0.0,
        "IsResponsive": 1.0 if "viewport" in lower_html else 0.0,
        "NoOfURLRedirect": 0.0,
        "NoOfiFrame": float(iframes),
        "HasExternalFormSubmit": has_ext_form,
        "HasHiddenFields": has_hidden,
        "HasPasswordField": has_password,
        "Bank": has_bank,
        "Pay": has_pay,
        "Crypto": has_crypto,
        "NoOfImage": no_of_images,
        "NoOfCSS": no_of_css,
        "NoOfJS": no_of_js,
        "NoOfExternalRef": float(ext_refs),
    }
