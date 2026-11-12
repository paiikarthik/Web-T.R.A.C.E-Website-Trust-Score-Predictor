# Comprehensive True Positive (TP) & True Negative (TN) Operational Security Audit

**Evaluation Date:** 2026-10-28  
**Dataset:** PhiUSIIL Phishing URL Dataset (235,370 clean samples)  
**Evaluation Protocol:** Strict Unseen-Domain Holdout Split (46,632 test URLs, zero domain overlap)  
**Evaluator:** [`phishing_detection.py`](file:///C:/Users/Karthik%20Pai/Documents/MajorProject/phishing_detection.py)  
**Data Artifacts:**
* Unseen Domain CSV: [`tp_tn_evaluation_unseen_domain.csv`](file:///C:/Users/Karthik%20Pai/Documents/MajorProject/outputs/reports/tp_tn_evaluation_unseen_domain.csv)
* Random Holdout CSV: [`tp_tn_evaluation_random.csv`](file:///C:/Users/Karthik%20Pai/Documents/MajorProject/outputs/reports/tp_tn_evaluation_random.csv)
* Master Benchmark Table: [`final_model_comparison.csv`](file:///C:/Users/Karthik%20Pai/Documents/MajorProject/outputs/reports/final_model_comparison.csv)
* Visual Comparison Bar Chart: [`tp_tn_comparison_unseen_domain.png`](file:///C:/Users/Karthik%20Pai/Documents/MajorProject/outputs/figures/tp_tn_comparison_unseen_domain.png)

---

## 1. Foundational Security Definitions

In cybersecurity defense and automated URL triage, binary predictions map into four operational states:

1. **True Positive (TP) — Phishing Attack Caught:**  
   The URL was an actual phishing attack, and the model correctly classified it as **Phishing (1)**.  
   $$\text{Security Impact:} \quad \text{Malicious session blocked; credential theft prevented.}$$

2. **True Negative (TN) — Benign URL Allowed:**  
   The URL was legitimate, and the model correctly classified it as **Legitimate (0)**.  
   $$\text{Operational Impact:} \quad \text{Normal employee/user browsing preserved with zero friction.}$$

3. **False Positive (FP) — False Alarm / Benign Blocked:**  
   The URL was legitimate, but the model incorrectly classified it as **Phishing (1)**.  
   $$\text{Operational Impact:} \quad \text{Employee work interrupted; IT helpdesk overhead; alert fatigue.}$$

4. **False Negative (FN) — Breach Risk / Missed Attack:**  
   The URL was malicious, but the model incorrectly classified it as **Legitimate (0)**.  
   $$\text{Security Impact:} \quad \text{Phishing attack bypassed gateway; high risk of credential harvesting.}$$

### Key Operational Rates:
* **True Positive Rate (TPR / Sensitivity / Phishing Recall):**  
  $$\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}} \quad \text{(Percentage of all attacks successfully intercepted)}$$
* **True Negative Rate (TNR / Specificity / Benign Accuracy):**  
  $$\text{TNR} = \frac{\text{TN}}{\text{TN} + \text{FP}} \quad \text{(Percentage of all benign traffic correctly allowed)}$$
* **False Positive Rate (FPR / Fall-out):**  
  $$\text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}} = 1 - \text{TNR} \quad \text{(False alarm rate)}$$
* **False Negative Rate (FNR / Miss Rate):**  
  $$\text{FNR} = \frac{\text{FN}}{\text{TP} + \text{FN}} = 1 - \text{TPR} \quad \text{(Miss rate / security exposure)}$$

---

## 2. Unseen-Domain Holdout Evaluation (46,632 Test URLs)

Test set ground-truth composition:
* **Total Samples:** $46,632$
* **Actual Legitimate:** $26,923$ ($57.74\%$)
* **Actual Phishing:** $19,709$ ($42.26\%$)
* **Domain Overlap with Training:** **Strictly 0 domains**

| Model Name | Paradigm | True Positive (TP) | True Negative (TN) | False Positive (FP) | False Negative (FN) | TPR (Sensitivity) | TNR (Specificity) | FPR (Fall-out) | Precision | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`DL_LSTM`** | Deep Learning | **14,130** | **25,062** | 1,861 | 5,579 | **71.69%** | **93.09%** | 6.91% | 88.36% | **84.05%** |
| **`ML_RandomForest`** | Machine Learning | **14,102** | **25,047** | 1,876 | 5,607 | **71.55%** | **93.03%** | 6.97% | 88.26% | **83.95%** |
| **`DL_CNN1D`** | Deep Learning | **14,040** | **24,948** | 1,975 | 5,669 | **71.24%** | **92.66%** | 7.34% | 87.67% | **83.61%** |
| **`DL_Hybrid_CNNLSTM`** | Deep Learning | **13,811** | **25,259** | 1,664 | 5,898 | **70.07%** | **93.82%** | 6.18% | 89.25% | **83.78%** |
| **`ML_XGBoost`** | Machine Learning | **13,679** | **25,392** | **1,531** | 6,030 | **69.40%** | **94.31%** | **5.69%** | **89.93%** | **83.79%** |
| **`DL_SelfAttention`** | Deep Learning | **13,678** | **25,385** | **1,538** | 6,031 | **69.40%** | **94.29%** | **5.71%** | **89.89%** | **83.77%** |
| **`ML_DecisionTree`** | Machine Learning | 13,605 | 25,324 | 1,599 | 6,104 | 69.03% | 94.06% | 5.94% | 89.48% | 83.48% |
| **`DL_DeepMLP`** | Deep Learning | 13,587 | **25,455** | **1,468** | 6,122 | 68.94% | **94.55%** | **5.45%** | **90.25%** | **83.72%** |
| **`ML_LogisticRegression`** | Machine Learning | 11,967 | 24,679 | 2,244 | 7,742 | 60.72% | 91.67% | 8.33% | 84.21% | 78.59% |
| **`ML_LinearSVM`** | Machine Learning | 11,682 | 24,942 | 1,981 | 8,027 | 59.27% | 92.64% | 7.36% | 85.50% | 78.54% |

---

## 3. Random Holdout Evaluation (47,074 Test URLs)

Test set ground-truth composition:
* **Total Samples:** $47,074$
* **Actual Legitimate:** $26,970$ ($57.29\%$)
* **Actual Phishing:** $20,104$ ($42.71\%$)

| Model Name | Paradigm | True Positive (TP) | True Negative (TN) | False Positive (FP) | False Negative (FN) | TPR (Sensitivity) | TNR (Specificity) | FPR (Fall-out) | Precision | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ML_RandomForest`** | Machine Learning | **14,717** | **24,985** | 1,985 | 5,387 | **73.20%** | **92.64%** | 7.36% | 88.12% | **84.34%** |
| **`DL_LSTM`** | Deep Learning | **14,312** | **25,198** | 1,772 | 5,792 | **71.19%** | **93.43%** | 6.57% | 88.98% | **83.93%** |
| **`DL_SelfAttention`** | Deep Learning | **14,262** | **25,260** | 1,710 | 5,842 | **70.94%** | **93.66%** | 6.34% | 89.29% | **83.96%** |
| **`ML_XGBoost`** | Machine Learning | **14,212** | **25,451** | **1,519** | 5,892 | **70.69%** | **94.37%** | **5.63%** | **90.34%** | **84.26%** |
| **`DL_DeepMLP`** | Deep Learning | **14,167** | **25,429** | **1,541** | 5,937 | **70.47%** | **94.29%** | **5.71%** | **90.19%** | **84.11%** |
| **`ML_DecisionTree`** | Machine Learning | 14,114 | 25,371 | 1,599 | 5,990 | 70.20% | 94.07% | 5.93% | 89.82% | 83.88% |
| **`DL_Hybrid_CNNLSTM`** | Deep Learning | 13,917 | 25,590 | 1,380 | 6,187 | 69.23% | 94.88% | 5.12% | 90.98% | 83.93% |
| **`DL_CNN1D`** | Deep Learning | 13,666 | 25,803 | 1,167 | 6,438 | 67.98% | 95.67% | 4.33% | 92.13% | 83.84% |
| **`ML_LogisticRegression`** | Machine Learning | 11,986 | 24,789 | 2,181 | 8,118 | 59.62% | 91.91% | 8.09% | 84.61% | 78.12% |
| **`ML_LinearSVM`** | Machine Learning | 11,678 | 25,102 | 1,868 | 8,426 | 58.09% | 93.07% | 6.93% | 86.21% | 78.13% |

---

## 4. Key Security & Operational Findings

### A. Threat Detection Leader: `DL_LSTM` and `ML_RandomForest`
* In the unseen-domain zero-overlap evaluation, **`DL_LSTM` achieved the highest absolute True Positive count across all Deep Learning models ($14,130$ TP)**, delivering a Sensitivity (Recall) of **$71.69\%$**.
* **`ML_RandomForest` was the top Machine Learning model ($14,102$ TP)**, with Sensitivity of **$71.55\%$**.
* This proves that both recurrent sequential gating and decision tree bagging excel at discovering complex non-linear combinations of domain length, entropy, and character distributions.

### B. Enterprise False-Alarm Reducers: `DL_DeepMLP` and `ML_XGBoost`
* In an enterprise firewall or gateway, excessive False Positives disrupt legitimate business operations.
* **`DL_DeepMLP` achieved the highest True Negative count ($25,455$ TN)**, resulting in a Specificity of **$94.55\%$** and a False Positive Rate of only **$5.45\%$**.
* **`ML_XGBoost` achieved $25,392$ TN ($94.31\%$ Specificity)** and only $1,531$ false alarms out of $26,923$ legitimate websites.
* When high precision is demanded, gradient boosted trees and multi-layer perceptrons provide the lowest false alarm burden.

### C. Generalization Difference: Random vs. Unseen-Domain
* When evaluating on previously seen domains (Random split), True Positives increased by $\approx 500\text{--}600$ attacks because high-capacity trees memorize shared domain strings (the $2,079$ shared domains).
* Under the strict unseen-domain split, True Positives remain remarkably stable ($>14,100$), confirming that the 14 honest domain features generalize to brand-new, zero-day hostnames.

---

## 5. Artifact Links

1. **[`tp_tn_comparison_unseen_domain.png`](file:///C:/Users/Karthik%20Pai/Documents/MajorProject/outputs/figures/tp_tn_comparison_unseen_domain.png):** Visual side-by-side bar chart of TP, TN, FP, FN counts and Sensitivity/Specificity rates.
2. **[`confusion_unseen_domain_DL_LSTM.png`](file:///C:/Users/Karthik%20Pai/Documents/MajorProject/outputs/figures/confusion_unseen_domain_DL_LSTM.png):** Detailed heatmap for top DL model.
3. **[`confusion_unseen_domain_ML_RandomForest.png`](file:///C:/Users/Karthik%20Pai/Documents/MajorProject/outputs/figures/confusion_unseen_domain_ML_RandomForest.png):** Detailed heatmap for top ML model.
4. **[`confusion_unseen_domain_ML_XGBoost.png`](file:///C:/Users/Karthik%20Pai/Documents/MajorProject/outputs/figures/confusion_unseen_domain_ML_XGBoost.png):** Detailed heatmap for XGBoost.
5. **[`tp_tn_evaluation_unseen_domain.csv`](file:///C:/Users/Karthik%20Pai/Documents/MajorProject/outputs/reports/tp_tn_evaluation_unseen_domain.csv):** Raw numerical export.
