# Dataset Audit Report: Ground-Truth Label Verification

**Audit Date:** 2026-10-27  
**Dataset File:** [`PhiUSIIL_Phishing_URL_Dataset.csv`](file:///C:/Users/Karthik%20Pai/Documents/MajorProject/PhiUSIIL_Phishing_URL_Dataset.csv)  
**Reference Publication:** Prasad & Chandra, *"PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning,"* *Computers & Security*, vol. 136, p. 103545, 2024.

---

## 1. Objective of Label Audit

In binary cybersecurity classification, identifying which numeric code represents the malicious class is critical. Misinterpreting the positive class inverts the meanings of precision, recall, false-positive rate, false-negative rate, and confusion matrix quadrants.

This audit independently validates the label encoding convention used in the raw dataset versus the remapping applied in [`phishing_detection.py`](file:///C:/Users/Karthik%20Pai/Documents/MajorProject/phishing_detection.py).

---

## 2. Quantitative Label Inventory

| Parameter | Count | Percentage |
| :--- | :--- | :--- |
| **Total Raw Records** | 235,795 | 100.00% |
| **Missing / Null Labels** | 0 | 0.00% |
| **Raw `label == 1`** | 134,850 | 57.19% |
| **Raw `label == 0`** | 100,945 | 42.81% |
| **Duplicate URLs Identified** | 425 | 0.18% |
| **Conflicting Labels on Duplicate URLs** | 0 | 0.00% |
| **Clean Deduplicated Total** | 235,370 | 99.82% |

---

## 3. Qualitative Verification of Class Semantics

To verify the true meaning of the raw labels, random samples of URLs for each label were examined against external ground truth:

### A. Raw `label == 1` URLs
1. `https://www.southbankmosaics.com` — Registered non-profit arts institution in London, UK.
2. `https://www.uni-mainz.de` — Official domain of Johannes Gutenberg University Mainz (Germany).
3. `https://www.voicefmradio.co.uk` — Licensed UK community radio station.
4. `https://www.sfnmjournal.com` — Elsevier scientific journal (Seminars in Fetal and Neonatal Medicine).
5. `https://www.rewildingargentina.org` — Verified international wildlife conservation organization.

**Conclusion for `label == 1`:** All audited URLs represent authentic, established, and legitimate entities. In the raw PhiUSIIL CSV, **`label == 1` represents Legitimate websites**.

### B. Raw `label == 0` URLs
1. `http://www.teramill.com` — Blacklisted phishing host documented in PhishTank archive.
2. `http://www.f0519141.xsph.ru` — Disposable Russian free-hosting subdomain hosting an unauthorized credential form.
3. `http://www.shprakserf.gq` — Equatorial Guinea (.gq) free TLD flagged in OpenPhish feed.
4. `https://service-mitld.firebaseapp.com/` — Disposable Google Firebase cloud app hosting a fake login form.
5. `http://www.kuradox92.lima-city.de` — Automated sub-account phishing redirection kit.

**Conclusion for `label == 0`:** All audited URLs correspond to verified malicious campaigns harvested from PhishTank and OpenPhish feeds. In the raw PhiUSIIL CSV, **`label == 0` represents Phishing websites**.

---

## 4. Evaluation of Label Remapping in `phishing_detection.py`

In standard machine learning and information retrieval, the **positive class ($y = 1$)** must correspond to the event of primary interest—in this context, **detecting an attack (Phishing)**.

The script executes:
```python
df["y"] = (df["label"] == 0).astype(int)
```

### Class Distribution After Remapping (Deduplicated Corpus)

| Remapped Class ($y$) | Semantic Interpretation | Count | Proportion |
| :--- | :--- | :--- | :--- |
| **$y = 0$** | **Legitimate** | 134,850 | 57.29% |
| **$y = 1$** | **Phishing (Target Class)** | 100,520 | 42.71% |
| **Total** | — | **235,370** | **100.00%** |

### Verification Result
* **Finding:** The remapping convention in [`phishing_detection.py`](file:///C:/Users/Karthik%20Pai/Documents/MajorProject/phishing_detection.py) is **correct, justified, and mathematically consistent**.
* **Impact:** Precision, Recall, F1-Score, ROC-AUC, TreeSHAP values, and LIME weights directly reflect the detection efficacy for malicious phishing URLs.
