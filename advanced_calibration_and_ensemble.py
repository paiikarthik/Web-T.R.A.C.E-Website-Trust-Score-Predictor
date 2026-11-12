"""
Advanced Calibration, Soft-Voting Ensemble, and Dataset Diversity Analysis.

Solves:
1. Research Gap 4: Reliable Trust-Score Estimation via Brier Score and Expected Calibration Error (ECE).
2. Ensemble Soft-Voting combination of RF, XGBoost, and DeepMLP to maximize phishing recall.
3. Threshold Sensitivity Analysis for Asymmetric Cybersecurity Cost Optimization.
4. Dataset Diversity & Attack Technique Representation Audit.
"""

import math

import time
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"
REPORT_DIR = BASE_DIR / "outputs" / "reports"
MODEL_DIR = BASE_DIR / "outputs" / "models"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

from feature_extractor import extract_features, FEATURE_COLUMNS


class CalibratedSoftVotingEnsemble:
    """
    Ensemble combining Tuned Random Forest, Tuned XGBoost, and Tuned DeepMLP.
    Weights: 40% XGBoost, 35% Random Forest, 25% DeepMLP.
    """
    def __init__(self, rf_model, xgb_model, mlp_model, w=(0.35, 0.40, 0.25)):
        self.rf = rf_model
        self.xgb = xgb_model
        self.mlp = mlp_model
        self.w = w

    def predict_proba(self, X):
        p_rf = self.rf.predict_proba(X)[:, 1]
        p_xgb = self.xgb.predict_proba(X)[:, 1]
        p_mlp = self.mlp.predict_proba(X)[:, 1]
        p_ens = self.w[0] * p_rf + self.w[1] * p_xgb + self.w[2] * p_mlp
        return np.column_stack([1.0 - p_ens, p_ens])

    def predict(self, X, threshold=0.50):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


def compute_ece(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE).
    ECE measures the difference between predicted confidence and empirical accuracy across bins.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    bin_data = []
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper) if i < n_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            acc_in_bin = np.mean(y_true[in_bin])
            conf_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
            bin_data.append({
                "bin": f"{bin_lower:.1f}-{bin_upper:.1f}",
                "count": int(np.sum(in_bin)),
                "empirical_acc": float(acc_in_bin),
                "mean_confidence": float(conf_in_bin),
                "calibration_gap": float(np.abs(acc_in_bin - conf_in_bin))
            })
    return float(ece), bin_data


def main():
    print("=" * 80)
    print("ADVANCED CALIBRATION, ENSEMBLE FUSION & DATASET DIVERSITY BENCHMARK")
    print("=" * 80)

    # 1. Load Dataset
    print("[1] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    SAMPLE_SIZE = 60000
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

    # In PhiUSIIL, label: 0 = Phishing, 1 = Legitimate.
    # Standardize: 1 = Phishing (positive malicious class), 0 = Legitimate (negative class).
    y = (df["label"] == 0).astype(int)
    print(f"Dataset loaded: {len(df):,} samples | Phishing ratio: {y.mean():.2%}")

    # 2. Extract 17 audited features
    print("[2] Extracting audited 17 features...")
    feat_rows = []
    for u in df["URL"]:
        feat_rows.append(extract_features(u))
    feat_df = pd.DataFrame(feat_rows)
    X = feat_df[FEATURE_COLUMNS].copy()
    groups = feat_df["_parsed_domain"].copy()

    # 3. Partition strictly held-out unseen domains (20%)
    print("[3] Partitioning unseen-domain test set (20%)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    groups_train, groups_test = groups.iloc[train_idx].copy(), groups.iloc[test_idx].copy()

    print(f"Train set: {len(X_train):,} samples across {groups_train.nunique():,} unique domains")
    print(f"Test set:  {len(X_test):,} samples across {groups_test.nunique():,} unique domains")

    # 4. Fit Base Tuned Models
    print("\n[4] Fitting base tuned models...")
    rf = RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_split=5, min_samples_leaf=2,
                                class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=150, max_depth=7, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
                        scale_pos_weight=1.2, random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1)
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64, 32), alpha=0.005, learning_rate_init=0.002,
                              max_iter=40, early_stopping=True, random_state=RANDOM_STATE))
    ])

    print("  Fitting Random Forest...")
    rf.fit(X_train, y_train)
    print("  Fitting XGBoost...")
    xgb.fit(X_train, y_train)
    print("  Fitting DeepMLP...")
    mlp.fit(X_train, y_train)

    # 5. Predict Probabilities on Test Set
    rf_prob = rf.predict_proba(X_test)[:, 1]
    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    mlp_prob = mlp.predict_proba(X_test)[:, 1]

    # 6. Ensemble Fusion Models
    # Weighted Soft Voting: 40% XGBoost, 35% Random Forest, 25% DeepMLP
    ensemble_prob = 0.40 * xgb_prob + 0.35 * rf_prob + 0.25 * mlp_prob

    # 7. Model Evaluation Suite with Brier Score and ECE
    models_to_eval = {
        "Random Forest (Tuned)": rf_prob,
        "XGBoost (Tuned)": xgb_prob,
        "Deep Learning (DeepMLP Tuned)": mlp_prob,
        "Calibrated Soft-Voting Ensemble (Ours)": ensemble_prob,
    }

    eval_results = []
    print("\n[5] Comprehensive Evaluation on Unseen Domains with Calibration Metrics:")
    print("-" * 105)
    print(f"{'Model':38s} | {'Acc':7s} | {'Prec':7s} | {'Recall':7s} | {'Spec':7s} | {'F1':7s} | {'ROC':7s} | {'PR':7s} | {'Brier':7s} | {'ECE':7s}")
    print("-" * 105)

    for m_name, probs in models_to_eval.items():
        preds = (probs >= 0.50).astype(int)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        roc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        ece, _ = compute_ece(y_test.values, probs, n_bins=10)

        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        spec = tn / max(1, tn + fp)

        print(f"{m_name:38s} | {acc:.4f}  | {prec:.4f}  | {rec:.4f}  | {spec:.4f}  | {f1:.4f}  | {roc:.4f}   | {pr_auc:.4f}  | {brier:.4f}   | {ece:.4f}")

        eval_results.append({
            "Model": m_name,
            "Accuracy": acc,
            "Precision_Phishing": prec,
            "Recall_Sensitivity": rec,
            "Specificity": spec,
            "F1_Score": f1,
            "ROC_AUC": roc,
            "PR_AUC": pr_auc,
            "Brier_Score": brier,
            "Expected_Calibration_Error_ECE": ece,
            "True_Positives": tp,
            "True_Negatives": tn,
            "False_Positives": fp,
            "False_Negatives": fn,
            "Total_Test_Samples": len(y_test),
        })

    eval_df = pd.DataFrame(eval_results)
    eval_df.to_csv(REPORT_DIR / "calibration_and_ensemble_benchmark.csv", index=False)
    print("-" * 105)
    print(f"Benchmark saved to: {REPORT_DIR / 'calibration_and_ensemble_benchmark.csv'}")

    # 8. Threshold Sensitivity Analysis for Operational Risk Triage
    print("\n[6] Operational Threshold Sensitivity Analysis (Calibrated Ensemble):")
    threshold_results = []
    # In cybersecurity: Cost of FN (Missed Phishing attack) is typically 5x higher than FP (Quarantine/Warn)
    cost_fn_weight = 5.0
    cost_fp_weight = 1.0

    print(f"{'Threshold':10s} | {'Acc':7s} | {'Prec':7s} | {'Recall':7s} | {'Spec':7s} | {'F1':7s} | {'TP':6s} | {'FN':6s} | {'FP':6s} | {'Weighted Loss':14s}")
    print("-" * 95)
    for tau in [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]:
        t_preds = (ensemble_prob >= tau).astype(int)
        t_acc = accuracy_score(y_test, t_preds)
        t_prec = precision_score(y_test, t_preds, zero_division=0)
        t_rec = recall_score(y_test, t_preds, zero_division=0)
        t_f1 = f1_score(y_test, t_preds, zero_division=0)
        cm = confusion_matrix(y_test, t_preds, labels=[0, 1])
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        t_spec = tn / max(1, tn + fp)
        weighted_loss = (cost_fn_weight * fn) + (cost_fp_weight * fp)

        print(f"{tau:10.2f} | {t_acc:.4f}  | {t_prec:.4f}  | {t_rec:.4f}  | {t_spec:.4f}  | {t_f1:.4f}  | {tp:<6d} | {fn:<6d} | {fp:<6d} | {weighted_loss:<14.1f}")
        threshold_results.append({
            "Threshold": tau,
            "Accuracy": t_acc,
            "Precision": t_prec,
            "Recall": t_rec,
            "Specificity": t_spec,
            "F1_Score": t_f1,
            "True_Positives": tp,
            "False_Negatives": fn,
            "False_Positives": fp,
            "True_Negatives": tn,
            "Weighted_Loss_5xFN": weighted_loss
        })

    thresh_df = pd.DataFrame(threshold_results)
    thresh_df.to_csv(REPORT_DIR / "threshold_sensitivity_analysis.csv", index=False)

    # 9. Dataset Diversity & Attack Technique Representation Audit
    print("\n[7] Training Dataset Diversity & Attack Representation Audit:")
    div_results = []
    checks = {
        "High-Risk Disposable TLDs (.tk, .ml, .xyz, etc.)": (feat_df["is_susp_tld"] == 1),
        "Direct IP Address Hosting": (feat_df["IsDomainIP"] == 1),
        "Brand Impersonation in Subdomain": (feat_df["brand_in_subdomain"] == 1),
        "Brand Impersonation in Non-Official Domain": (feat_df["brand_in_domain_non_official"] == 1),
        "Brand Lure in URL Path": (feat_df["brand_in_path"] == 1),
        "Homoglyph Typosquatting Match": (feat_df["brand_homoglyph_typosquat"] == 1),
        "Credential Phishing Keywords (login, verify, etc.)": (feat_df["kw_count"] > 0),
        "Deep Directory Path Concealment (depth >= 2)": (feat_df["path_depth"] >= 2),
        "Query Parameter Injection (count >= 1)": (feat_df["query_param_count"] >= 1),
        "URL Percent-Encoding Evasion (%XX >= 1)": (feat_df["url_encoding_count"] >= 1),
        "High Domain Shannon Entropy (> 3.8 bits)": (feat_df["netloc_entropy"] > 3.8),
        "High Domain Numeric Density (> 20% digits)": (feat_df["digit_ratio_domain"] > 0.20),
    }

    for desc, mask in checks.items():
        p_pct = mask[y == 1].mean() * 100
        l_pct = mask[y == 0].mean() * 100
        ratio = (p_pct + 1e-4) / (l_pct + 1e-4)
        div_results.append({
            "Attack_Technique_or_Structural_Pattern": desc,
            "Phishing_Prevalence_Pct": p_pct,
            "Legitimate_Prevalence_Pct": l_pct,
            "Risk_Ratio_Phish_vs_Legit": ratio
        })
        print(f"  {desc:50s} -> Phish: {p_pct:5.2f}% | Legit: {l_pct:5.2f}% | Ratio: {ratio:5.1f}x")

    div_df = pd.DataFrame(div_results)
    div_df.to_csv(REPORT_DIR / "dataset_diversity_analysis.csv", index=False)

    # 10. Export Ensemble Pipeline for Serving
    print("\n[8] Exporting Calibrated Soft-Voting Ensemble model pipeline...")
    ensemble = CalibratedSoftVotingEnsemble(rf, xgb, mlp)
    joblib.dump(ensemble, MODEL_DIR / "Ensemble_Calibrated_SoftVoting.pkl")
    print(f"Saved: {MODEL_DIR / 'Ensemble_Calibrated_SoftVoting.pkl'}")
    print("=" * 80)
    print("BENCHMARK & CALIBRATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
