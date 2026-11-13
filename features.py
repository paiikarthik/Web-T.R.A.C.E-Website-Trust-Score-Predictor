"""
Web T.R.A.C.E — Unified Feature Engineering Engine (features.py).

Single shared feature extraction module used by BOTH model training and inference.
Guarantees zero train/serve skew and strictly enforces honest domain evaluation:
- Preserves audited base domain features.
- Computes host-based features strictly after stripping 'www.' (no shortcut on 'www' or 'https').
- Computes Public Suffix List (PSL) private domain & tenant-label features via tldextract.
- Computes brand-impersonation, homoglyphs, and Levenshtein edit distance.
- Strictly segregates path/query features for the free-host slice model only.
"""

from __future__ import annotations

import functools
import math
import re
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import tldextract

# ---------------------------------------------------------------------------
# Offline TLDExtract Configuration
# ---------------------------------------------------------------------------
_tld_extractor = tldextract.TLDExtract(
    suffix_list_urls=None,
    fallback_to_snapshot=True,
    cache_dir=None,
    include_psl_private_domains=True,
)

# ---------------------------------------------------------------------------
# Target Brands & Suspicious Keywords
# ---------------------------------------------------------------------------
TARGET_BRANDS: tuple[str, ...] = (
    "paypal", "google", "apple", "microsoft", "amazon", "netflix",
    "facebook", "instagram", "whatsapp", "wellsfargo", "chase",
    "bankofamerica", "binance", "coinbase", "ebay", "twitter", "linkedin",
    "yahoo", "outlook", "adobe"
)

HOMOGLYPH_MAP = str.maketrans({
    "0": "o",
    "1": "l",
    "3": "e",
    "4": "a",
    "5": "s",
    "@": "a",
    "$": "s",
    "8": "b",
})

SUSPICIOUS_KEYWORDS: tuple[str, ...] = (
    "login", "signin", "verify", "verification", "account", "update",
    "banking", "secure", "security", "webscr", "confirm", "password",
    "wallet", "admin", "service", "support", "billing", "auth",
    "ebayisapi"
)

SUSPICIOUS_TLDS: set[str] = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "work", "icu", "buzz", "cn", "ru"
}

# Free-hosting platforms & cloud tenant suffixes
FREE_HOST_PLATFORMS: dict[str, int] = {
    # 1: Developer Cloud / Code Hosting
    "github.io": 1,
    "gitlab.io": 1,
    "pages.dev": 1,
    "vercel.app": 1,
    "netlify.app": 1,
    "surge.sh": 1,
    # 2: App Hosting / PaaS
    "firebaseapp.com": 2,
    "web.app": 2,
    "glitch.me": 2,
    "render.com": 2,
    "onrender.com": 2,
    "herokuapp.com": 2,
    "appspot.com": 2,
    "azurewebsites.net": 2,
    "cloudfunctions.net": 2,
    "workers.dev": 2,
    # 3: Site Builders & Free Blogs
    "weeblysite.com": 3,
    "weebly.com": 3,
    "wixsite.com": 3,
    "wordpress.com": 3,
    "blogspot.com": 3,
    "000webhostapp.com": 3,
    # 4: Cloud Storage Buckets
    "s3.amazonaws.com": 4,
    "blob.core.windows.net": 4,
    "storage.googleapis.com": 4,
}

# ---------------------------------------------------------------------------
# General Model Feature List (Strictly NO path/query, NO https, NO www)
# ---------------------------------------------------------------------------
GENERAL_FEATURE_NAMES: list[str] = [
    # 1. Base Domain Features (14)
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "netloc_entropy",
    "hyphen_domain",
    "dot_domain",
    "kw_count",
    "has_kw",
    "is_susp_tld",
    "digits_domain",
    "digit_ratio_domain",
    # 2. Host-Based Features (stripped 'www.') (10)
    "host_clean_len",
    "host_clean_entropy",
    "host_clean_hyphens",
    "host_clean_dots",
    "host_clean_digits",
    "host_clean_digit_ratio",
    "host_clean_vowel_ratio",
    "host_clean_longest_consonant_run",
    "host_clean_token_count",
    "host_clean_avg_token_len",
    # 3. PSL Tenant-Label Features (9)
    "is_free_host",
    "platform_category",
    "tenant_len",
    "tenant_entropy",
    "tenant_digits",
    "tenant_hyphens",
    "tenant_keywords",
    "tenant_vowel_ratio",
    "tenant_longest_consonant_run",
    # 4. Brand-Impersonation Features (4)
    "brand_in_subdomain",
    "brand_in_path_diff_domain",
    "brand_edit_distance_min",
    "brand_homoglyph_typosquat",
]

# Separate Path & Query Features (Used ONLY for Free-Host Slice Model)
PATH_QUERY_FEATURE_NAMES: list[str] = [
    "path_len",
    "path_depth",
    "path_entropy",
    "query_len",
    "query_param_count",
    "has_query",
    "brand_in_path",
]


# ---------------------------------------------------------------------------
# Mathematical & String Utilities
# ---------------------------------------------------------------------------
def shannon_entropy(s: str) -> float:
    """Calculates Shannon entropy in bits for string s."""
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def longest_consonant_run(s: str) -> int:
    """Finds the maximum length of consecutive consonants in string s."""
    if not s:
        return 0
    consonants = set("bcdfghjklmnpqrstvwxyz")
    max_run = 0
    curr_run = 0
    for char in s.lower():
        if char in consonants:
            curr_run += 1
            if curr_run > max_run:
                max_run = curr_run
        else:
            curr_run = 0
    return max_run


def levenshtein_dist(s1: str, s2: str) -> int:
    """Computes the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_dist(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (c1 != c2)
            curr.append(min(insertions, deletions, substitutions))
        prev = curr
    return prev[-1]


@functools.lru_cache(maxsize=150000)
def min_brand_edit_distance_norm(domain_label: str) -> float:
    """Normalized minimum Levenshtein distance against known target brands."""
    label = domain_label.lower().strip()
    if not label:
        return 1.0
    min_norm = 1.0
    for brand in TARGET_BRANDS:
        d = levenshtein_dist(label, brand)
        norm = d / max(len(label), len(brand))
        if norm < min_norm:
            min_norm = norm
    return round(min_norm, 4)


@functools.lru_cache(maxsize=150000)
def detect_brand_homoglyph(domain_label: str) -> int:
    """Detects digit-for-letter substitution impersonating known brands."""
    label = domain_label.lower().strip()
    if not label:
        return 0
    # If the label is an authentic brand already, it's not a typosquat
    if label in TARGET_BRANDS:
        return 0
    subbed = label.translate(HOMOGLYPH_MAP)
    for brand in TARGET_BRANDS:
        if brand in subbed and brand not in label:
            return 1
    return 0


# ---------------------------------------------------------------------------
# URL Parsing & Component Extraction
# ---------------------------------------------------------------------------
def parse_url_components(raw_url: str):
    """
    Parses a URL string into netloc, path, query, host_clean, and PSL extraction.
    """
    url_str = str(raw_url).strip()
    if not url_str.startswith(("http://", "https://")):
        url_str = "http://" + url_str

    parsed = urlparse(url_str)
    netloc = parsed.netloc or parsed.path.split("/")[0]
    if "@" in netloc:
        netloc = netloc.split("@")[-1]
    if ":" in netloc:
        netloc = netloc.split(":")[0]
    netloc_lower = netloc.lower().strip()

    # Host clean: strip leading www. or www[0-9]*.
    host_clean = re.sub(r"^www\d*\.", "", netloc_lower)

    path = parsed.path or ""
    query = parsed.query or ""

    # PSL extraction
    ext = _tld_extractor(url_str)
    subdomain = ext.subdomain.lower() if ext.subdomain else ""
    domain = ext.domain.lower() if ext.domain else ""
    suffix = ext.suffix.lower() if ext.suffix else ""
    
    # Registered domain
    try:
        reg_domain = ext.top_domain_under_public_suffix.lower() if ext.top_domain_under_public_suffix else f"{domain}.{suffix}"
    except AttributeError:
        reg_domain = f"{domain}.{suffix}" if domain and suffix else host_clean

    # Free hosting platform identification
    is_free = False
    platform_cat = 0
    tenant_label = ""
    platform_name = ""

    if suffix in FREE_HOST_PLATFORMS:
        is_free = True
        platform_name = suffix
        platform_cat = FREE_HOST_PLATFORMS[suffix]
        tenant_label = domain
    elif reg_domain in FREE_HOST_PLATFORMS:
        is_free = True
        platform_name = reg_domain
        platform_cat = FREE_HOST_PLATFORMS[reg_domain]
        # Tenant is lowest subdomain
        tenant_label = subdomain.split(".")[-1] if subdomain else ""

    return {
        "url_str": url_str,
        "netloc": netloc_lower,
        "host_clean": host_clean,
        "path": path,
        "query": query,
        "subdomain": subdomain,
        "domain": domain,
        "suffix": suffix,
        "reg_domain": reg_domain,
        "is_free": is_free,
        "platform_cat": platform_cat,
        "tenant_label": tenant_label,
        "platform_name": platform_name,
    }


def get_domain_group(url: str) -> str:
    """
    Returns the domain group for domain-disjoint splitting.
    For free-host URLs, groups by tenant (e.g. 'tenant.firebaseapp.com').
    For standard URLs, groups by registered domain (e.g. 'google.com').
    """
    info = parse_url_components(url)
    if info["is_free"] and info["tenant_label"]:
        return f"{info['tenant_label']}.{info['platform_name']}".rstrip(".")
    grp = info["reg_domain"] or info["host_clean"]
    return grp.rstrip(".")


# ---------------------------------------------------------------------------
# Feature Extraction Functions
# ---------------------------------------------------------------------------
def extract_features(raw_url: str, include_path_query: bool = False) -> dict[str, Any]:
    """
    Extracts features for a single URL string.
    
    Args:
        raw_url: Raw URL string.
        include_path_query: When False (default), extracts strictly general model
            features (no path/query shortcuts). When True, includes separate
            path and query features for the free-host slice model.
            
    Returns:
        Dictionary mapping feature name to numeric value.
    """
    info = parse_url_components(raw_url)
    url_str = info["url_str"]
    netloc = info["netloc"]
    host_clean = info["host_clean"]
    path = info["path"]
    query = info["query"]
    subdomain = info["subdomain"]
    domain = info["domain"]
    suffix = info["suffix"]
    is_free = info["is_free"]
    platform_cat = info["platform_cat"]
    tenant_label = info["tenant_label"]

    # 1. Base Domain Features (14)
    is_ip = 1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", netloc) else 0
    subdomain_parts = [p for p in subdomain.split(".") if p]
    no_of_subdomains = len(subdomain_parts)
    
    # HasObfuscation & NoOfObfuscatedChar: percent-encoded hex sequences (%XX) matching PhiUSIIL
    hex_escapes = re.findall(r"%[0-9a-fA-F]{2}", url_str)
    no_of_obfuscated_char = len(hex_escapes) * 3
    has_obf = 1 if len(hex_escapes) > 0 else 0
    
    netloc_ent = shannon_entropy(netloc)
    hyphen_dom = netloc.count("-")
    dot_dom = netloc.count(".")
    
    kw_cnt = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in host_clean)
    has_kw_val = 1 if kw_cnt > 0 else 0
    is_susp_tld_val = 1 if suffix in SUSPICIOUS_TLDS else 0
    
    digits_dom = sum(c.isdigit() for c in netloc)
    digit_ratio_dom = digits_dom / max(1, len(netloc))

    # 2. Host-Based Features (stripped 'www.') (10)
    host_clean_len = len(host_clean)
    host_clean_ent = shannon_entropy(host_clean)
    host_clean_hyphens = host_clean.count("-")
    host_clean_dots = host_clean.count(".")
    host_clean_digits = sum(c.isdigit() for c in host_clean)
    host_clean_digit_ratio = host_clean_digits / max(1, host_clean_len)
    host_clean_vowels = sum(c in "aeiou" for c in host_clean)
    host_clean_vowel_ratio = host_clean_vowels / max(1, host_clean_len)
    host_clean_longest_consonants = longest_consonant_run(host_clean)
    tokens = [t for t in re.split(r"[.\-_]", host_clean) if t]
    host_clean_token_cnt = len(tokens)
    host_clean_avg_tok_len = sum(len(t) for t in tokens) / max(1, host_clean_token_cnt)

    # 3. PSL Tenant-Label Features (9)
    if is_free and tenant_label:
        tenant_len = len(tenant_label)
        tenant_ent = shannon_entropy(tenant_label)
        tenant_digits = sum(c.isdigit() for c in tenant_label)
        tenant_hyphens = tenant_label.count("-")
        tenant_keywords = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in tenant_label)
        tenant_vowels = sum(c in "aeiou" for c in tenant_label)
        tenant_vowel_ratio = tenant_vowels / max(1, tenant_len)
        tenant_longest_consonants = longest_consonant_run(tenant_label)
    else:
        tenant_len = 0
        tenant_ent = 0.0
        tenant_digits = 0
        tenant_hyphens = 0
        tenant_keywords = 0
        tenant_vowel_ratio = 0.0
        tenant_longest_consonants = 0

    # 4. Brand-Impersonation Features (4)
    brand_in_sub = 1 if any(b in subdomain for b in TARGET_BRANDS) else 0
    brand_in_path_diff = 1 if (any(b in path.lower() for b in TARGET_BRANDS) and domain not in TARGET_BRANDS) else 0
    effective_target_label = tenant_label if (is_free and tenant_label) else domain
    brand_edit_dist = min_brand_edit_distance_norm(effective_target_label)
    brand_homoglyph = detect_brand_homoglyph(effective_target_label)

    features: dict[str, Any] = {
        # Base Domain
        "DomainLength": len(netloc),
        "IsDomainIP": is_ip,
        "TLDLength": len(suffix),
        "NoOfSubDomain": no_of_subdomains,
        "HasObfuscation": has_obf,
        "NoOfObfuscatedChar": no_of_obfuscated_char,
        "netloc_entropy": round(netloc_ent, 4),
        "hyphen_domain": hyphen_dom,
        "dot_domain": dot_dom,
        "kw_count": kw_cnt,
        "has_kw": has_kw_val,
        "is_susp_tld": is_susp_tld_val,
        "digits_domain": digits_dom,
        "digit_ratio_domain": round(digit_ratio_dom, 4),
        # Host-Based ('www.' stripped)
        "host_clean_len": host_clean_len,
        "host_clean_entropy": round(host_clean_ent, 4),
        "host_clean_hyphens": host_clean_hyphens,
        "host_clean_dots": host_clean_dots,
        "host_clean_digits": host_clean_digits,
        "host_clean_digit_ratio": round(host_clean_digit_ratio, 4),
        "host_clean_vowel_ratio": round(host_clean_vowel_ratio, 4),
        "host_clean_longest_consonant_run": host_clean_longest_consonants,
        "host_clean_token_count": host_clean_token_cnt,
        "host_clean_avg_token_len": round(host_clean_avg_tok_len, 4),
        # PSL Tenant
        "is_free_host": int(is_free),
        "platform_category": platform_cat,
        "tenant_len": tenant_len,
        "tenant_entropy": round(tenant_ent, 4),
        "tenant_digits": tenant_digits,
        "tenant_hyphens": tenant_hyphens,
        "tenant_keywords": tenant_keywords,
        "tenant_vowel_ratio": round(tenant_vowel_ratio, 4),
        "tenant_longest_consonant_run": tenant_longest_consonants,
        # Brand Impersonation
        "brand_in_subdomain": brand_in_sub,
        "brand_in_path_diff_domain": brand_in_path_diff,
        "brand_edit_distance_min": brand_edit_dist,
        "brand_homoglyph_typosquat": brand_homoglyph,
    }

    if include_path_query:
        path_depth = len([s for s in path.split("/") if s])
        brand_in_path_val = 1 if any(b in path.lower() for b in TARGET_BRANDS) else 0
        features.update({
            "path_len": len(path),
            "path_depth": path_depth,
            "path_entropy": round(shannon_entropy(path), 4),
            "query_len": len(query),
            "query_param_count": len(query.split("&")) if query else 0,
            "has_query": int(len(query) > 0),
            "brand_in_path": brand_in_path_val,
        })

    return features


def extract_features_df(urls: pd.Series | list[str], include_path_query: bool = False) -> pd.DataFrame:
    """
    Extracts features for a collection of URLs into a pandas DataFrame.
    """
    records = [extract_features(u, include_path_query=include_path_query) for u in urls]
    return pd.DataFrame(records)


def extract_host_clean(raw_url: str) -> str:
    """Returns host with 'www.' stripped for n-gram feature generation."""
    info = parse_url_components(raw_url)
    return info["host_clean"]
