# Split Validation Report: Random vs. Unseen-Domain Holdout

**Validation Date:** 2026-10-27  
**Dataset:** Cleaned PhiUSIIL Phishing URL Dataset (235,370 unique URLs)  
**Evaluator:** [`phishing_detection.py`](file:///C:/Users/Karthik%20Pai/Documents/MajorProject/phishing_detection.py)  

---

## 1. Executive Summary

A critical flaw in historical phishing detection literature is evaluating models exclusively via standard random train/test splits. When random splitting is applied to web corpora containing multiple URLs from the same domain, identical domain names appear in both training and test partitions. This allows classifiers to memorize domain-specific artifacts rather than learning generalized phishing syntax.

To address this, two distinct evaluation protocols were implemented and empirically validated:
* **Experiment A (Random Stratified Holdout):** Represents the conventional machine learning baseline.
* **Experiment B (Unseen-Domain Holdout):** Implements strict domain-grouped partitioning via `GroupShuffleSplit(groups=Domain)` to ensure zero domain overlap between training and testing.

---

## 2. Quantitative Comparison of Evaluation Protocols

| Parameter | Experiment A: Random Stratified Split | Experiment B: Unseen-Domain Split | Validation Status |
| :--- | :--- | :--- | :--- |
| **Splitting Strategy** | `train_test_split(stratify=y)` | `GroupShuffleSplit(groups=Domain)` | Distinctly Implemented |
| **Random Seed** | 42 | 42 | Identical |
| **Feature Count** | 33 features | 33 features | Identical |
| **Training Sample Count** | 188,296 | 188,738 | Balanced (~80%) |
| **Test Sample Count** | 47,074 | 46,632 | Balanced (~20%) |
| **Training Class Balance** | 57.29% Legit (0), 42.71% Phish (1) | 57.18% Legit (0), 42.82% Phish (1) | Preserved Class Prior |
| **Test Class Balance** | 57.29% Legit (0), 42.71% Phish (1) | 57.74% Legit (0), 42.26% Phish (1) | Preserved Class Prior |
| **Unique Training Domains** | 176,979 | 176,068 | Audited |
| **Unique Testing Domains** | 45,186 | 44,018 | Audited |
| **Overlapping Domains** | **2,079 domains** (Domain Leakage) | **0 domains (Strictly Zero Overlap)** | **CONFIRMED 0 OVERLAP** |

---

## 3. Key Observations & Inherent Limitations

### A. The 2,079 Leaked Domains in Standard Random Splitting
In Experiment A, exactly **2,079 domain names** were shared across both the training set and the test set. For URLs under these shared domains, classifiers with high memorization capacity (such as unconstrained decision trees or nearest-neighbor classifiers) can achieve artificially inflated accuracy simply by recognizing familiar domain strings.

### B. Verification of Zero Overlap in Experiment B
In Experiment B, computing the set intersection of domain names between training and testing yielded:
$$\text{Domains}_{\text{train}} \cap \text{Domains}_{\text{test}} = \emptyset \quad (\text{Count} = 0)$$
This confirms that the test partition consists strictly of **unseen domains**, providing an objective measure of zero-day out-of-distribution generalization.

### C. Granularity Limitation: Hostname vs. Registrable Apex Domain
While zero domain overlap is mathematically guaranteed with respect to the `Domain` column, a qualitative limitation must be noted:
* In the PhiUSIIL schema, `Domain` represents the **fully qualified domain name (FQDN)**, which includes subdomains.
* For third-party hosting services (e.g., `ipfs.io`, `firebaseapp.com`, `fleek.co`), multiple distinct subdomains or hashes can exist (e.g., `hash1.ipfs.io` vs. `hash2.ipfs.io`).
* While `docs.google.com` (526 URLs) and `ipfs.io` (1,197 URLs) are kept intact in their respective partitions because their string matches exactly, distinct subdomains hosted on shared platforms may theoretically exist in both sets. 
* However, because all raw URL strings and domain text are purged from the feature matrix before model training (only derived numerical features are passed), models cannot memorize text tokens of subdomains across the split.
