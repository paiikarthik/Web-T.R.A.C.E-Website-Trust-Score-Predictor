# Data Quality and Data Leakage Audit Report

**Audit Date:** 2026-10-27  
**Analyzed Dataset:** [`PhiUSIIL_Phishing_URL_Dataset.csv`](file:///C:/Users/Karthik%20Pai/Documents/MajorProject/PhiUSIIL_Phishing_URL_Dataset.csv)  
**Sample Size:** 235,795 raw rows, 56 attributes  

---

## 1. Executive Summary

A comprehensive data quality and leakage audit was conducted to investigate why baseline models in the literature easily report $\ge 99\%$ accuracy on the PhiUSIIL corpus, and to distinguish **statistically legitimate signals** from **artificial collection shortcuts and target leakage**.

The audit identified three major categories of features that must be removed for a defensible, real-world, zero-scraping URL phishing detector:
1. **Confirmed Target Leakage:** Derived corpus metrics that directly encode ground-truth label distributions (`URLSimilarityIndex`, `TLDLegitimateProb`).
2. **Collection-Source Artifacts:** Structural peculiarities of how legitimate URLs were assembled versus phishing links (`URLLength - DomainLength == 7`, `IsHTTPS == 1` for 100% of benign sites).
3. **Inappropriate Webpage Scraped Predictors:** DOM and HTML features (`LineOfCode`, `HasSocialNet`, `NoOfJS`) that are unavailable in real-time URL inspection and vulnerable to anti-bot cloaking.

---

## 2. Detailed Findings

### A. Duplicate URLs and Label Consistency
* **Exact Duplicate URLs:** 425 duplicate rows were detected ($0.18\%$ of the dataset).
* **Label Ambiguity:** 0 URLs exhibited conflicting labels. Every duplicate record had consistent ground truth.
* **Resolution:** Dropped duplicates on `subset=["URL"]`, retaining 235,370 clean, unique records.

---

### B. Domain Identifier Analysis (Hostname vs. Registrable Domain)
* **Unique Values in `Domain` Column:** 220,086 distinct strings across 235,795 URLs.
* **Multi-URL Domains:** 5,526 domains contain multiple distinct URL paths.
  * Top repeated domains: `ipfs.io` (1,197 URLs), `docs.google.com` (526 URLs), `mail.deliverylifesupport.com` (370 URLs), `cloudflare-ipfs.com` (359 URLs), `storageapi.fleek.co` (318 URLs).
* **Granularity Limitation:**
  * 74.89% (176,596) of domain strings start with `www.`.
  * The column represents the **full fully qualified hostname (FQDN)** rather than the effective Second-Level Domain (eSLD / registrable apex domain).
  * *Operational Risk:* Subdomains hosted on shared cloud providers (e.g., `user1.firebaseapp.com` and `user2.firebaseapp.com`) are treated as distinct entities by the `Domain` column. While `docs.google.com` URLs are kept together in group splits, disposable subdomains on third-party SaaS platforms may still span training and testing.

---

### C. Confirmed Target Leakage & Derived Corpus Metrics

| Feature Name | Correlation with Phishing ($y=1$) | Legitimate ($y=0$) Distribution | Phishing ($y=1$) Distribution | Evidence & Leakage Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **`URLSimilarityIndex`** | **$-0.8604$** | **$\mu = 100.0, \sigma = 0.0$**<br>(Min: 100.0, Max: 100.0) | $\mu = 49.62, \sigma = 22.57$<br>(Min: 0.15, Max: 100.0) | **Confirmed Target Leakage.** Literally 100% of legitimate URLs have an exact index of 100.0. A single decision stump `URLSimilarityIndex < 100` trivially separates almost all phishing. |
| **`CharContinuationRate`** | $-0.3211$ | Derived statistical metric | Derived statistical metric | **Derived Statistical Leakage.** Computed using corpus-level n-gram probabilities. |
| **`TLDLegitimateProb`** | $-0.2984$ | Derived empirical prior | Derived empirical prior | **Corpus Prior Leakage.** Encodes the empirical probability of legitimacy for the TLD across the dataset. |
| **`URLCharProb`** | $-0.2415$ | Derived character distribution | Derived character distribution | **Corpus Prior Leakage.** Encodes vocabulary frequencies from the target set. |

* **Remediation in Code:** All 4 derived columns are explicitly purged via `DROP_DERIVED_FEATURES = True`.

---

### D. Collection-Source Artifacts (The "Shortcut Trap")

#### 1. The Root-Domain Homepage Algebraic Shortcut
* **Observation:** In the legitimate sample collection (harvested from top rankings), every URL was stored as a bare homepage (e.g., `https://example.com` or `https://www.example.com`).
* **The Mathematical Signature:**
  $$\text{URLLength} - \text{DomainLength} == 7$$
  * For **100% of Legitimate URLs (134,850 / 134,850)**, this difference is exactly 7 (representing the protocol string `https://` without a trailing slash).
  * For Phishing URLs, the difference varies widely (6, 8, 9, 16, 25, 40+ characters) because phishing kits are hosted in deep subdirectories.
* **The Vulnerability:** If a tree or neural network receives `URLLength` and `DomainLength`, it can learn this exact subtraction rule and achieve near $100\%$ accuracy without evaluating phishing syntax.
* **Remediation in Code:** `URLLength` and `NoOfLettersInURL` are dropped via `DROP_SHORTCUT_FEATURES = True`. Valid normalized character ratios (`LetterRatioInURL`, `DegitRatioInURL`, etc.) are retained.

#### 2. The HTTPS Binary Shortcut
* **Observation:** In the PhiUSIIL dataset, `IsHTTPS == 1` for **100% of legitimate sites** (134,850 / 134,850). Zero legitimate sites are HTTP.
* **The Vulnerability:** In real-world cybercrime (APWG 2024–2026), over $85\%$ of active phishing websites use valid, free HTTPS certificates. A classifier trained on this dataset with `IsHTTPS` learns that `IsHTTPS == 0 \implies \text{Phishing}$`, which fails against modern HTTPS phishing.
* **Remediation in Code:** `IsHTTPS` is dropped via `DROP_IS_HTTPS = True`.

---

### E. Webpage Scraped Predictors (HTML & DOM Features)

The dataset contains 28 features extracted from the HTML source code of the homepage:
* **`LineOfCode`:** 
  * Legitimate: Mean = 1,947 lines, Median = 1,105 lines, Min = 102 lines.
  * Phishing: Mean = 65 lines, Median = 12 lines, 25% have only 2 lines!
* **`HasSocialNet` (Correlation: $-0.784$):** Legitimate corporate portals embed functional social media links; ephemeral phishing kits do not.
* **`HasCopyrightInfo` (Correlation: $-0.743$):** Benign sites include formal copyright notices in footers.

* **Why They Are Inappropriate for URL Detection:**
  1. **Latency:** Downloading and parsing HTML takes 2,000–5,000 ms per URL.
  2. **Security:** Fetching untrusted HTML exposes the scanner to browser exploits and drive-by malware.
  3. **Evasion:** Attackers detect automated crawlers and return blank pages (cloaking).
* **Remediation in Code:** All 28 HTML features are removed via `URL_ONLY_FEATURES = True`, ensuring a lightweight, zero-scraping, sub-millisecond edge architecture.

---

## 3. Summary of Audit Classifications

| Feature Category | Features Included | Audit Finding | Action Taken in Pipeline |
| :--- | :--- | :--- | :--- |
| **Identifiers** | `FILENAME`, `URL`, `Domain`, `TLD`, `Title` | Non-generalizable identifiers | Dropped from feature matrix; `Domain` used for group splitting. |
| **Ground Truth** | `label`, `y` | Target labels | Separated as ground truth vector `y`. |
| **Target Leakage** | `URLSimilarityIndex`, `TLDLegitimateProb`, etc. | Directly encode ground truth | Dropped (`DROP_DERIVED_FEATURES = True`). |
| **Collection Shortcuts** | `URLLength`, `NoOfLettersInURL`, `IsHTTPS` | Artificial dataset collection artifacts | Dropped (`DROP_SHORTCUT_FEATURES = True`, `DROP_IS_HTTPS = True`). |
| **Scraped HTML Features** | 28 DOM/HTML features (`LineOfCode`, `NoOfJS`, etc.) | Require live web crawling; latency/cloaking bottleneck | Dropped (`URL_ONLY_FEATURES = True`). |
| **Legitimate URL Features** | 15 dataset counters/ratios + 18 engineered URL features | Valid, zero-scraping syntactic & lexical signals | **Retained & Modeled across 5 ML + 5 DL algorithms.** |
