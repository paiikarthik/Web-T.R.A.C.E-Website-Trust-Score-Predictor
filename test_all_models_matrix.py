"""
Web T.R.A.C.E — Comprehensive Multi-Model Stress-Test & Evaluation Matrix.

Tests all 11 models across every major URL category:
- Legitimate Reputable Apex Domains
- Legitimate Educational & Non-Profit Sites
- Brand Homoglyphs & Typosquats (paypa1, micros0ft, g00gle)
- Brand-in-Subdomain Impersonation
- Direct IP Address Harvesters
- Suspicious Disposable TLDs (.tk, .xyz, .top)
- Free-Hosting Cloud Platforms (github.io, firebaseapp.com, web.app)
- Credential-Lure Keyword Packaging (login, verify, secure)
- Obfuscated & Delimiter-packed URLs
- Machine-Generated DGA-like Random Strings

Calculates accuracy, false positives, false negatives, average trust scores,
latency, and ranks which model delivers the best operational results.
"""

import time
import pandas as pd
import numpy as np
from app import app
import json

# Comprehensive test URLs with ground truth (0 = Legitimate, 1 = Phishing)
TEST_SUITE = [
    # 1. Clean Reputable Legitimate Domains
    {"url": "https://www.wikipedia.org", "expected": 0, "category": "Legitimate Apex"},
    {"url": "https://www.python.org", "expected": 0, "category": "Legitimate Apex"},
    {"url": "https://www.nature.com", "expected": 0, "category": "Legitimate Publishing"},
    {"url": "https://www.mit.edu", "expected": 0, "category": "Legitimate Educational"},
    {"url": "https://www.kernel.org", "expected": 0, "category": "Legitimate Infrastructure"},
    {"url": "https://www.bbc.com", "expected": 0, "category": "Legitimate Media"},
    {"url": "https://www.apache.org", "expected": 0, "category": "Legitimate Open Source"},
    {"url": "https://www.mozilla.org", "expected": 0, "category": "Legitimate Tech"},
    {"url": "https://www.stackoverflow.com", "expected": 0, "category": "Legitimate Tech"},
    {"url": "https://www.archive.org", "expected": 0, "category": "Legitimate Non-Profit"},

    # 2. Homoglyph Typosquatting
    {"url": "http://paypa1-account-update.tk/signin", "expected": 1, "category": "Homoglyph Typosquat"},
    {"url": "http://micros0ft-security-auth.com/login", "expected": 1, "category": "Homoglyph Typosquat"},
    {"url": "http://g00gle-drive-verification.xyz/auth", "expected": 1, "category": "Homoglyph Typosquat"},
    {"url": "http://app1e-id-recovery.cf/verify", "expected": 1, "category": "Homoglyph Typosquat"},
    {"url": "http://netf1ix-billing-update.top/account", "expected": 1, "category": "Homoglyph Typosquat"},

    # 3. Brand-in-Subdomain Spoofing
    {"url": "http://paypal.com.security-check.ru/webscr", "expected": 1, "category": "Subdomain Spoof"},
    {"url": "http://chase.online-banking-auth.com/signin", "expected": 1, "category": "Subdomain Spoof"},
    {"url": "http://wellsfargo.verify-customer.info/portal", "expected": 1, "category": "Subdomain Spoof"},
    {"url": "http://binance.wallet-unlock.xyz/session", "expected": 1, "category": "Subdomain Spoof"},
    {"url": "http://amazon.orders-verification.com/ap/signin", "expected": 1, "category": "Subdomain Spoof"},

    # 4. Direct Numerical IP Hostnames
    {"url": "http://192.168.1.1/admin/login", "expected": 1, "category": "Raw IP Host"},
    {"url": "http://45.33.32.156/secure/portal", "expected": 1, "category": "Raw IP Host"},
    {"url": "http://185.220.101.5/bank/verify", "expected": 1, "category": "Raw IP Host"},

    # 5. Suspicious Disposable TLDs
    {"url": "http://urgent-security-check.tk", "expected": 1, "category": "Suspicious TLD"},
    {"url": "http://account-suspended-action-required.xyz", "expected": 1, "category": "Suspicious TLD"},
    {"url": "http://banking-alert-notice.top", "expected": 1, "category": "Suspicious TLD"},

    # 6. Free-Hosting Platforms (with Cloud Disclaimers)
    {"url": "https://legit-looking-service.firebaseapp.com/login", "expected": 1, "category": "Free-Host Platform"},
    {"url": "http://secure-credential-check.web.app", "expected": 1, "category": "Free-Host Platform"},
    {"url": "http://billing-update-form.weeblysite.com", "expected": 1, "category": "Free-Host Platform"},
    {"url": "http://support-ticket-verification.glitch.me", "expected": 1, "category": "Free-Host Platform"},

    # 7. Credential-Harvesting Keywords
    {"url": "http://login-verify-account-security-update.com", "expected": 1, "category": "Keyword Stacking"},
    {"url": "http://secure-banking-verification-portal.com", "expected": 1, "category": "Keyword Stacking"},

    # 8. Obfuscated / Special Char Delimiters
    {"url": "http://login%40paypal.com%3fupdate@badsite.com", "expected": 1, "category": "Obfuscation"},

    # 9. DGA / Algorithmic Random Hostnames
    {"url": "http://x7q9b2m1k0p8w4z.xyz/auth", "expected": 1, "category": "DGA Randomness"},
    {"url": "http://a98f72c3d1e0b5.top/gate", "expected": 1, "category": "DGA Randomness"},
]

def run_stress_test():
    client = app.test_client()
    
    # 1. Fetch available models
    res_models = client.get("/api/models")
    models_meta = res_models.get_json()
    model_ids = [m["id"] for m in models_meta]
    
    print("=" * 80)
    print(f"WEB T.R.A.C.E — EXHAUSTIVE MULTI-MODEL STRESS TEST MATRIX")
    print(f"Testing {len(TEST_SUITE)} diverse URLs across {len(model_ids)} distinct models")
    print("=" * 80)
    
    # Performance trackers
    results_by_model = {m: {
        "correct": 0, "total": len(TEST_SUITE),
        "fp": 0, "fn": 0, "tp": 0, "tn": 0,
        "latencies": [],
        "predictions": []
    } for m in model_ids}

    # Execute all combinations
    for i, item in enumerate(TEST_SUITE):
        u = item["url"]
        exp = item["expected"]
        cat = item["category"]
        
        print(f"[{i+1:02d}/{len(TEST_SUITE)}] Testing {cat:22} | URL: {u[:45]}...")
        
        for m in model_ids:
            t0 = time.perf_counter()
            resp = client.post("/api/scan", json={"url": u, "model": m})
            latency_ms = (time.perf_counter() - t0) * 1000.0
            
            if resp.status_code != 200:
                print(f"  [!] Model {m} failed with status {resp.status_code}: {resp.data}")
                continue
                
            data = resp.get_json()
            is_phish = int(data["is_phishing"])
            prob = data["phishing_probability"]
            trust = data["trust_score"]
            tier = data["risk_tier"]
            
            # Record metrics
            results_by_model[m]["latencies"].append(latency_ms)
            
            if is_phish == exp:
                results_by_model[m]["correct"] += 1
                if exp == 1:
                    results_by_model[m]["tp"] += 1
                else:
                    results_by_model[m]["tn"] += 1
            else:
                if exp == 0 and is_phish == 1:
                    results_by_model[m]["fp"] += 1
                else:
                    results_by_model[m]["fn"] += 1
                    
            results_by_model[m]["predictions"].append({
                "url": u, "expected": exp, "predicted": is_phish,
                "prob": prob, "trust": trust, "tier": tier, "category": cat
            })

    # Summary table
    summary_rows = []
    for m in model_ids:
        r = results_by_model[m]
        acc = (r["correct"] / r["total"]) * 100.0
        prec = (r["tp"] / (r["tp"] + r["fp"])) * 100.0 if (r["tp"] + r["fp"]) > 0 else 0.0
        rec = (r["tp"] / (r["tp"] + r["fn"])) * 100.0 if (r["tp"] + r["fn"]) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        med_lat = np.median(r["latencies"])
        p95_lat = np.percentile(r["latencies"], 95)
        
        summary_rows.append({
            "Model ID": m,
            "Accuracy (%)": round(acc, 2),
            "Precision (%)": round(prec, 2),
            "Recall (%)": round(rec, 2),
            "F1-Score (%)": round(f1, 2),
            "False Positives (Clean Marked Phish)": r["fp"],
            "False Negatives (Phish Missed)": r["fn"],
            "Median Latency (ms)": round(med_lat, 2),
            "p95 Latency (ms)": round(p95_lat, 2),
        })

    df_summary = pd.DataFrame(summary_rows).sort_values(by=["F1-Score (%)", "Accuracy (%)"], ascending=False).reset_index(drop=True)
    df_summary.index += 1
    
    print("\n" + "=" * 100)
    print("STRESS TEST LEADERBOARD (RANKED BY F1-SCORE & ACCURACY):")
    print("=" * 100)
    print(df_summary.to_string())
    print("=" * 100)
    
    # Save results to outputs/reports
    df_summary.to_csv("outputs/reports/stress_test_leaderboard.csv", index=True)
    
    # Check specific categories for best model
    best_model = df_summary.iloc[0]["Model ID"]
    print(f"\n[WINNER]: Best performing model across all edge categories is: {best_model}")
    print(f"Accuracy: {df_summary.iloc[0]['Accuracy (%)']}% | F1: {df_summary.iloc[0]['F1-Score (%)']}% | FP: {df_summary.iloc[0]['False Positives (Clean Marked Phish)']}")
    
    return df_summary

if __name__ == "__main__":
    run_stress_test()
