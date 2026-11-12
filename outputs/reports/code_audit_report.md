# Code Audit Report: `phishing_detection.py`

**Audit Date:** 2026-10-27  
**Audited Artifact:** [`phishing_detection.py`](file:///C:/Users/Karthik%20Pai/Documents/MajorProject/phishing_detection.py)  
**Dataset Reference:** `PhiUSIIL_Phishing_URL_Dataset.csv`  

---

## 1. Executive Summary

This report documents the systematic technical audit performed on the codebase [`phishing_detection.py`](file:///C:/Users/Karthik%20Pai/Documents/MajorProject/phishing_detection.py). The audit evaluated the script against data science standards, cybersecurity operational constraints, and machine learning integrity rules.

Overall, the pipeline exhibits high engineering maturity: it uses isolated scikit-learn `Pipeline` objects, ensures preprocessing fitting occurs only on training data, implements both random and domain-grouped holdout evaluations, and integrates modern explainability tools (TreeSHAP and LIME). However, specific implementation limitations, architectural mismatches, and domain grouping caveats were identified and addressed.

---

## 2. Detailed Findings and Issues

### Issue 1: Scikit-Learn Estimator Interface Omission in PyTorch Wrapper
* **Problem Identified:** In `cross_validate()` from `sklearn.model_selection`, evaluating the `roc_auc` scoring metric on PyTorch models returned `NaN ± NaN`.
* **Relevant Code Location:** Lines 430–480, `class PyTorchClassifierWrapper(BaseEstimator, ClassifierMixin)`.
* **Why It Matters:** Scikit-learn's internal scoring dispatcher checks `getattr(estimator, "_estimator_type", None) == "classifier"` to determine whether to query `predict_proba` for ROC-AUC. Without this attribute explicitly declared on the class, the cross-validation scoring pipeline defaulted to treating the wrapper as an unclassified estimator, producing missing ROC-AUC values.
* **Classification:** **Confirmed Bug.**
* **Recommended Fix:** Add `_estimator_type = "classifier"` as an explicit class-level attribute in `PyTorchClassifierWrapper`.
* **Status:** **Implemented, Tested, and Verified.** Cross-validation metrics now compute cleanly.

---

### Issue 2: Tabular Features Treated as Spatial/Sequential Sequences in 1D-CNN and LSTM
* **Problem Identified:** The deep learning models `DL_CNN1D`, `DL_LSTM`, and `DL_Hybrid_CNNLSTM` reshape tabular feature vectors of shape `(batch, 33)` into pseudo-sequences `(batch, 1, 33)` or `(batch, 33, 1)`.
* **Relevant Code Location:** Lines 340–415, `CNN1DNet`, `LSTMNet`, and `CNNLSTMNet`.
* **Why It Matters:** 
  * Convolutional layers assume translation invariance across adjacent spatial inputs.
  * LSTMs assume temporal dependencies from step $t$ to step $t+1$.
  * In a tabular feature matrix, the column order (e.g., `DomainLength`, followed by `IsDomainIP`, followed by `TLDLength`) is arbitrary. While convolutions and recurrent cells still learn non-linear combinations of adjacent columns, their inductive biases do not physically match tabular column geometry as they would for raw character strings or time-series.
* **Classification:** **Research Limitation / Architectural Mismatch.**
* **Recommended Fix:** 
  1. Acknowledge this limitation clearly in research documentation.
  2. Treat `DL_DeepMLP` (Multi-Layer Perceptron) and `DL_SelfAttention` (Multi-Head Self-Attention, which calculates permutation-invariant pairwise attention between all features) as the primary, methodologically sound deep learning models for tabular representations.
  3. Retain `DL_CNN1D`, `DL_LSTM`, and `DL_Hybrid_CNNLSTM` as exploratory comparative baselines aligned with related literature (e.g., Sarumathy et al. 2026, Vulfin et al. 2026).
* **Status:** **Documented and Benchmarked.**

---

### Issue 3: Domain Grouping Granularity (Hostname vs. Registrable Domain)
* **Problem Identified:** In `build_feature_matrix()`, the domain grouping identifier is assigned directly from the dataset's `Domain` column (`groups = df["Domain"].astype(str)`).
* **Relevant Code Location:** Lines 244–248, `build_feature_matrix()` and Lines 265–278, `make_splits()`.
* **Why It Matters:** In the PhiUSIIL dataset, the `Domain` column contains the full hostname, including subdomains and `www.` prefixes (e.g., `service1.firebaseapp.com`, `service2.firebaseapp.com`, `www.google.com`). Grouping strictly by this column prevents exact hostname overlap, but allows two distinct URLs sharing the same registrable apex domain (e.g., `subA.target.com` and `subB.target.com`) to be split across train and test sets.
* **Classification:** **Potential Risk / Domain Grouping Limitation.**
* **Recommended Fix:** Audit the exact domain overlap and distribution. Strip `www.` prefixes during analysis, and evaluate whether eSLD (effective Second-Level Domain) extraction introduces severe class imbalances or unrepresented test classes.
* **Status:** **Audited and Documented in `split_validation_report.md`.**

---

### Issue 4: Binary Decision Hardcoding (0.5 Cutoff) without Calibrated Operational Risk Tiers
* **Problem Identified:** Model inference previously relied on a single hardcoded threshold ($P \ge 0.5 \implies \text{Phishing}$), ignoring prediction confidence.
* **Relevant Code Location:** Lines 346–352, `predict_scores()` and `predict()`.
* **Why It Matters:** In enterprise security environments (SOCs, email gateways), binary decisions cause severe alert fatigue. A borderline prediction ($P = 0.51$) requires secondary verification or step-up authentication, whereas an unequivocal malicious prediction ($P = 0.99$) warrants immediate automated firewall quarantine.
* **Classification:** **Operational Limitation.**
* **Recommended Fix:** Integrate a post-processing operational risk engine that categorizes probabilities into actionable tiers:
  * **Low Risk ($< 0.35$):** Safe / Allow.
  * **Medium Risk ($0.35\text{--}0.70$):** Suspicious / Warning & Step-Up Auth.
  * **High Risk ($\ge 0.70$):** Critical / Immediate Block & Quarantine.
* **Status:** **Implemented, Tested, and Verified.** Separate risk-tier distribution reports are now generated for each split and model.

---

### Issue 5: Artificial Accuracy Inflation via Collection-Artifact Leakage
* **Problem Identified:** Standard training on raw PhiUSIIL columns permits models to exploit dataset collection shortcuts:
  * `URLSimilarityIndex == 100.0` for 100% of legitimate URLs.
  * `URLLength - DomainLength == 7` for 100% of legitimate URLs (since all legitimate samples are homepage roots without paths).
  * `IsHTTPS == 1` for 100% of legitimate URLs (creating an artificial non-HTTPS penalty).
* **Relevant Code Location:** Lines 75–95 and 222–237, `build_feature_matrix()`.
* **Why It Matters:** If these features are retained, simple linear models and shallow trees achieve $\approx 99.9\%$ accuracy trivially, without learning actual phishing characteristics.
* **Classification:** **Confirmed Collection Leakage.**
* **Recommended Fix:** Enforce `DROP_SHORTCUT_FEATURES = True`, `DROP_IS_HTTPS = True`, and `DROP_DERIVED_FEATURES = True`, while engineering legitimate URL-native lexical features (entropy, path depth, keyword presence).
* **Status:** **Implemented, Tested, and Verified.**

---

## 3. Summary of Code Audit Status

| Component | Audit Status | Resolution / Validation |
| :--- | :--- | :--- |
| **Data Ingestion** | Passed | Verified integrity across all 235,795 rows. |
| **Missing Values** | Passed | 0 missing values; median imputer preserved in pipeline. |
| **Label Remapping** | Passed | Verified: 0 = Phishing, 1 = Legitimate in raw CSV; inverted to 1 = Phishing. |
| **Preprocessing Isolation** | Passed | Scaler and Imputer fitted exclusively inside `Pipeline` on training data. |
| **Model Architectures** | Passed with Enhancements | 5 ML + 5 DL models implemented and benchmarked. PyTorch wrapper patched. |
| **Risk Stratification** | Enhanced | Operational tiers implemented and saved to reports. |
| **Explainability (XAI)** | Passed | TreeSHAP and LIME verified on holdout samples. |
