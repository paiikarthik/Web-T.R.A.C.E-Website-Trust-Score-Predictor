# Domain-Grouped Cross-Validation Analysis Report

**Experiment Date:** 2026-10-27  
**Validation Strategy:** 5-Fold `StratifiedGroupKFold` (Grouping Identifier: `Domain`)  
**Feature Matrix:** 33 Leakage-Hardened URL Features (Exp 1 Full Hardened Pipeline)  
**Detailed Metrics Table:** [`domain_grouped_cv.csv`](file:///C:/Users/Karthik%20Pai/Documents/MajorProject/outputs/reports/domain_grouped_cv.csv)  

---

## 1. Executive Summary

Standard k-fold cross-validation (`StratifiedKFold`) randomly shuffles rows across folds. When a dataset contains multiple URLs from the same domain (e.g. `ipfs.io`, `docs.google.com`), URLs from that domain are partitioned into both the training fold and the validation fold. This allows models to memorize domain-level patterns, inflating cross-validation performance.

To rigorously evaluate true generalization across unseen domains, a 5-fold **`StratifiedGroupKFold`** experiment was executed. In this setup, all URLs sharing the same `Domain` string are strictly assigned to either the training set or the validation set in each fold.

---

## 2. Quantitative Results: 5-Fold `StratifiedGroupKFold`

Every fold was validated to have **exactly 0 domain overlap** between training and validation sets:
$$\text{Domains}_{\text{train, fold } k} \cap \text{Domains}_{\text{val, fold } k} = \emptyset, \quad \forall k \in \{1, 2, 3, 4, 5\}$$

### Fold-by-Fold Performance Summary

| Model | Fold | Accuracy | Phishing Precision | Phishing Recall | Phishing F1-Score | ROC-AUC | Overlapping Domains |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ML_XGBoost`** | Fold 1 | 0.9885 | 0.9902 | 0.9868 | 0.9885 | 0.9979 | **0** |
| | Fold 2 | 0.9936 | 0.9950 | 0.9922 | 0.9936 | 0.9985 | **0** |
| | Fold 3 | 0.9908 | 0.9924 | 0.9892 | 0.9908 | 0.9982 | **0** |
| | Fold 4 | 0.9899 | 0.9918 | 0.9880 | 0.9899 | 0.9983 | **0** |
| | Fold 5 | 0.9924 | 0.9940 | 0.9908 | 0.9924 | 0.9977 | **0** |
| **`ML_XGBoost` Mean ± Std** | — | **0.9910 ± 0.0019** | **0.9927 ± 0.0017** | **0.9894 ± 0.0019** | **0.9910 ± 0.0019** | **0.9981 ± 0.0003** | **0** |
| **`ML_RandomForest`** | Fold 1 | 0.9561 | 0.9870 | 0.9246 | 0.9548 | 0.9947 | **0** |
| | Fold 2 | 0.9582 | 0.9892 | 0.9265 | 0.9568 | 0.9963 | **0** |
| | Fold 3 | 0.9584 | 0.9889 | 0.9272 | 0.9571 | 0.9960 | **0** |
| | Fold 4 | 0.9525 | 0.9854 | 0.9186 | 0.9508 | 0.9956 | **0** |
| | Fold 5 | 0.9546 | 0.9868 | 0.9214 | 0.9530 | 0.9954 | **0** |
| **`ML_RandomForest` Mean ± Std** | — | **0.9560 ± 0.0022** | **0.9875 ± 0.0014** | **0.9237 ± 0.0032** | **0.9545 ± 0.0023** | **0.9956 ± 0.0005** | **0** |

---

## 3. Comparison: Random Stratified CV vs. Domain-Grouped CV

| Evaluation Setup | Splitter Used | Domain Leakage Across Folds? | XGBoost Mean F1 | Random Forest Mean F1 |
| :--- | :--- | :---: | :---: | :---: |
| **Standard Random CV** | `StratifiedKFold` | **Yes** (Shared domains across folds) | $0.9916 \pm 0.0007$ | $0.9472 \pm 0.0013$ |
| **Domain-Grouped CV** | `StratifiedGroupKFold` | **No** (Strict zero-overlap across folds) | $0.9910 \pm 0.0019$ | $0.9545 \pm 0.0023$ |

---

## 4. Key Takeaways & Research Conclusions

1. **Defensible Generalization Confirmed:**
   * `ML_XGBoost` maintains a mean F1-score of **$0.9910$** and a standard deviation of only **$0.0019$** across all 5 unseen-domain folds.
   * `ML_RandomForest` maintains an F1-score of **$0.9545 \pm 0.0023$** and precision of **$98.75\%$**.
   * The negligible drop between Random CV ($0.9916$) and Domain-Grouped CV ($0.9910$) proves that the model **does not rely on memorizing domain identities**. It has learned generalizable syntactic and structural patterns of phishing URLs.
2. **Stability Across Folds:**
   * The low variance across folds ($\sigma < 0.0025$) indicates that the classification rules remain stable even when large clusters of subdomains are entirely hidden from training.
