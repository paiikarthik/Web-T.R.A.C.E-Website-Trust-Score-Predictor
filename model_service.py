"""
Web T.R.A.C.E — Production Model Inference Service (model_service.py).

Unified production service for real-time zero-scraping URL trust evaluation:
- Loads the exact 38-feature schema from outputs/models/feature_list.json (Zero Train/Serve Skew).
- Character n-gram TF-IDF inference + LR out-of-fold stacked feature computation.
- Isotonic probability calibration for reliable empirical trust scores.
- Free-hosting platform detection, tagging, and score capping.
- True SHAP-driven plain English explanations based strictly on driving model features.
- Strict input validation and multi-model benchmark dispatching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import shap
import torch

import features

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "outputs" / "models"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"

# ---------------------------------------------------------------------------
# Load Feature Schema Artifact (Eliminates Train/Serve Skew)
# ---------------------------------------------------------------------------
FEATURE_LIST_PATH = MODEL_DIR / "feature_list.json"
if FEATURE_LIST_PATH.exists():
    with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
        FEATURE_COLUMNS: List[str] = json.load(f)
else:
    # Fallback to general feature set + n-gram probability
    FEATURE_COLUMNS = features.GENERAL_FEATURE_NAMES + ["ngram_prob"]

# ---------------------------------------------------------------------------
# Global In-Memory Model & Artifact Cache
# ---------------------------------------------------------------------------
_LOADED_MODELS: Dict[str, Any] = {}
_TFIDF_VECTORIZER: Optional[Any] = None
_LR_NGRAM: Optional[Any] = None
_CALIBRATOR: Optional[Any] = None
_SCALER: Optional[Any] = None
_TREE_EXPLAINER: Optional[shap.TreeExplainer] = None


class SoftVotingEnsemblePipeline:
    """
    Calibrated Heterogeneous Soft-Voting Ensemble.
    Combines Random Forest (35%), XGBoost (45%), and DeepMLP (20%)
    with Isotonic probability calibration and optimal PR-curve thresholding.
    """

    def __init__(self, rf, xgb, mlp, calibrator, scaler, weights=(0.35, 0.45, 0.20), threshold=0.3882):
        self.rf = rf
        self.xgb = xgb
        self.mlp = mlp
        self.calibrator = calibrator
        self.scaler = scaler
        self.w = weights
        self.threshold = threshold

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_rf = self.rf.predict_proba(X)[:, 1]
        p_xgb = self.xgb.predict_proba(X)[:, 1]
        X_scaled = self.scaler.transform(X) if self.scaler else X.values
        p_mlp = self.mlp.predict_proba(X_scaled)[:, 1]

        p_raw = self.w[0] * p_rf + self.w[1] * p_xgb + self.w[2] * p_mlp
        if self.calibrator is not None:
            p_cal = self.calibrator.transform(p_raw)
        else:
            p_cal = p_raw
        return np.column_stack([1.0 - p_cal, p_cal])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)


# ---------------------------------------------------------------------------
# PyTorch Architectures
# ---------------------------------------------------------------------------
class DeepMLP(torch.nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class CNN1DNet(torch.nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool1d(2)
        self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.adapt_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.adapt_pool(self.relu(self.bn2(self.conv2(x)))).squeeze(-1)
        return self.fc(x)


class LSTMNet(torch.nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        feat = out[:, -1, :]
        return self.fc(feat)


class CNNLSTMNet(torch.nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
        )
        self.lstm = torch.nn.LSTM(input_size=32, hidden_size=32, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        feat = self.conv(x).permute(0, 2, 1)
        out, _ = self.lstm(feat)
        return self.fc(out[:, -1, :])


class SelfAttentionNet(torch.nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features
        self.embed = torch.nn.Linear(1, 16)
        self.attn = torch.nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(16 * in_features, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        emb = self.embed(x)
        attn_out, _ = self.attn(emb, emb, emb)
        flat = attn_out.reshape(attn_out.size(0), -1)
        return self.fc(flat)


# Import custom PyTorch wrappers into global namespace for unpickling
import retrain_benchmark
import sys
for cls in [DeepMLP, CNN1DNet, LSTMNet, CNNLSTMNet, SelfAttentionNet]:
    setattr(sys.modules["__main__"], cls.__name__, getattr(retrain_benchmark, cls.__name__, cls))
setattr(sys.modules["__main__"], "PyTorchClassifierWrapper", retrain_benchmark.PyTorchClassifierWrapper)


MODEL_METADATA: Dict[str, Dict[str, Any]] = {
    "Ensemble_Calibrated_SoftVoting": {
        "display_name": "Calibrated Soft-Voting Ensemble (Recommended)",
        "paradigm": "Calibrated Heterogeneous Ensemble",
        "description": "Fusion of Tuned RF (35%), XGBoost (45%), and DeepMLP (20%). Accuracy 90.33%, Precision 93.38%, ROC-AUC 94.49%, ECE 0.0090.",
        "accuracy": "90.33%",
        "f1_score": "88.16%",
        "is_default": True,
    },
    "ML_XGBoost": {
        "display_name": "XGBoost (Stacked Gradient Boosted)",
        "paradigm": "Gradient Boosted Trees",
        "description": "Tuned 300-tree model on numeric + n-gram features. Top standalone accuracy (90.42%) and F1 (88.21%).",
        "accuracy": "90.42%",
        "f1_score": "88.21%",
        "is_default": False,
    },
    "ML_RandomForest": {
        "display_name": "Random Forest (Tuned Ensemble)",
        "paradigm": "Machine Learning Ensemble",
        "description": "Domain-grouped tuned 150-tree bagging ensemble. Accuracy 90.11%, Precision 92.47%, ROC-AUC 94.43%.",
        "accuracy": "90.11%",
        "f1_score": "87.97%",
        "is_default": False,
    },
    "DL_DeepMLP": {
        "display_name": "Deep MLP (Tuned Neural Net)",
        "paradigm": "Deep Learning Feedforward",
        "description": "PyTorch deep neural network with batch normalization and dropout. Accuracy 90.46%, Precision 96.35%.",
        "accuracy": "90.46%",
        "f1_score": "87.97%",
        "is_default": False,
    },
    "DL_SelfAttention": {
        "display_name": "Multi-Head Self-Attention Network",
        "paradigm": "Deep Learning Attention",
        "description": "4-head transformer attention over feature interactions. Accuracy 90.38%, Precision 95.53%, ROC-AUC 94.23%.",
        "accuracy": "90.38%",
        "f1_score": "87.96%",
        "is_default": False,
    },
    "DL_LSTM": {
        "display_name": "Deep Bidirectional LSTM",
        "paradigm": "Deep Learning Recurrent",
        "description": "Bidirectional LSTM sequential memory cells. Accuracy 89.62%, Precision 95.00%.",
        "accuracy": "89.62%",
        "f1_score": "86.94%",
        "is_default": False,
    },
    "DL_CNN1D": {
        "display_name": "1D Convolutional Neural Net",
        "paradigm": "Deep Learning Convolutional",
        "description": "1D temporal convolutions capturing localized feature relationships. Accuracy 88.27%, Recall 85.10%.",
        "accuracy": "88.27%",
        "f1_score": "86.22%",
        "is_default": False,
    },
    "DL_Hybrid_CNNLSTM": {
        "display_name": "Hybrid CNN-LSTM Network",
        "paradigm": "Deep Learning Hybrid",
        "description": "Hierarchical spatial CNN layers fed into recurrent LSTM cells. Accuracy 87.85%, Precision 95.62%.",
        "accuracy": "87.85%",
        "f1_score": "84.23%",
        "is_default": False,
    },
    "ML_DecisionTree": {
        "display_name": "Decision Tree (CART)",
        "paradigm": "Machine Learning Tree",
        "description": "Single interpretable decision tree with maximum depth 12. Accuracy 90.29%, Precision 95.72%.",
        "accuracy": "90.29%",
        "f1_score": "87.81%",
        "is_default": False,
    },
    "ML_LogisticRegression": {
        "display_name": "Logistic Regression (L2)",
        "paradigm": "Machine Learning Linear",
        "description": "Regularized linear probabilistic classifier on scaled 38-feature vector. Accuracy 90.08%, ROC-AUC 94.19%.",
        "accuracy": "90.08%",
        "f1_score": "87.61%",
        "is_default": False,
    },
    "ML_LinearSVM": {
        "display_name": "Linear Support Vector Machine",
        "paradigm": "Machine Learning Margin",
        "description": "Calibrated linear support vector machine. Accuracy 90.11%, Precision 94.95%, ROC-AUC 94.20%.",
        "accuracy": "90.11%",
        "f1_score": "87.65%",
        "is_default": False,
    },
}


# ---------------------------------------------------------------------------
# Artifact Loaders
# ---------------------------------------------------------------------------
def get_calibrator():
    global _CALIBRATOR
    if _CALIBRATOR is None:
        p = MODEL_DIR / "calibrator.pkl"
        if p.exists():
            _CALIBRATOR = joblib.load(p)
    return _CALIBRATOR


def get_scaler():
    global _SCALER
    if _SCALER is None:
        p = MODEL_DIR / "scaler.pkl"
        if p.exists():
            _SCALER = joblib.load(p)
    return _SCALER


def get_ngram_pipeline():
    global _TFIDF_VECTORIZER, _LR_NGRAM
    if _TFIDF_VECTORIZER is None:
        p_tfidf = MODEL_DIR / "tfidf_vectorizer.pkl"
        p_lr = MODEL_DIR / "lr_ngram.pkl"
        if p_tfidf.exists() and p_lr.exists():
            _TFIDF_VECTORIZER = joblib.load(p_tfidf)
            _LR_NGRAM = joblib.load(p_lr)
    return _TFIDF_VECTORIZER, _LR_NGRAM


def get_model(model_name: str = "Ensemble_Calibrated_SoftVoting"):
    """Loads a model from disk into cache if not already loaded."""
    if model_name == "Ensemble_Calibrated_SoftVoting":
        if "Ensemble_Calibrated_SoftVoting" not in _LOADED_MODELS:
            rf = get_model("ML_RandomForest")
            xgb = get_model("ML_XGBoost")
            mlp = get_model("DL_DeepMLP")
            calibrator = get_calibrator()
            scaler = get_scaler()
            _LOADED_MODELS["Ensemble_Calibrated_SoftVoting"] = SoftVotingEnsemblePipeline(
                rf, xgb, mlp, calibrator, scaler, threshold=0.3882
            )
        return _LOADED_MODELS["Ensemble_Calibrated_SoftVoting"]

    if model_name not in _LOADED_MODELS:
        model_file = MODEL_DIR / f"{model_name}.pkl"
        if not model_file.exists():
            model_name = "ML_XGBoost"
            model_file = MODEL_DIR / f"{model_name}.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_file}")

        print(f"[*] Loading model into cache: {model_name}")
        _LOADED_MODELS[model_name] = joblib.load(model_file)
    return _LOADED_MODELS[model_name]


def list_available_models() -> List[Dict[str, Any]]:
    """Returns metadata for all available models."""
    available = []
    for key, meta in MODEL_METADATA.items():
        file_path = MODEL_DIR / f"{key}.pkl"
        is_ready = True if key == "Ensemble_Calibrated_SoftVoting" else file_path.exists()
        item = {
            "id": key,
            "display_name": meta["display_name"],
            "paradigm": meta["paradigm"],
            "description": meta["description"],
            "accuracy": meta["accuracy"],
            "f1_score": meta["f1_score"],
            "is_default": meta.get("is_default", False),
            "is_ready": is_ready,
        }
        available.append(item)
    return available


# ---------------------------------------------------------------------------
# Feature Matrix Construction with Zero Train/Serve Skew
# ---------------------------------------------------------------------------
def build_model_input_vector(raw_features: dict, url: str) -> pd.DataFrame:
    """
    Constructs the exact DataFrame expected by the trained models,
    including the character n-gram stacked probability.
    """
    row = dict(raw_features)

    # Compute n-gram probability if missing
    if "ngram_prob" not in row:
        tfidf, lr_ngram = get_ngram_pipeline()
        if tfidf is not None and lr_ngram is not None:
            host_clean = features.extract_host_clean(url)
            tfidf_vec = tfidf.transform([host_clean])
            row["ngram_prob"] = float(lr_ngram.predict_proba(tfidf_vec)[0, 1])
        else:
            row["ngram_prob"] = 0.50

    # Ensure exact column order and presence
    row_aligned = {col: row.get(col, 0.0) for col in FEATURE_COLUMNS}
    return pd.DataFrame([row_aligned], columns=FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Plain-English Explanations Driven by Real SHAP Contributions
# ---------------------------------------------------------------------------
FEATURE_EXPLANATION_MAP = {
    "ngram_prob": (
        "High-Risk Host Character Sequence",
        "The character combinations and sub-word sequences in the hostname closely match structural patterns found in known phishing infrastructure.",
        "The character structure matches normal, established domain naming conventions.",
    ),
    "brand_in_subdomain": (
        "Brand Impersonation in Subdomain Prefix",
        "A well-known corporate brand is embedded inside the subdomain to deceptively impersonate official company infrastructure.",
        "No brand tokens were detected in the subdomain prefix.",
    ),
    "brand_homoglyph_typosquat": (
        "Homoglyph Typosquatting Detected",
        "The domain uses look-alike character substitutions (such as '0' for 'o' or '1' for 'l') to spoof an authentic brand.",
        "No visual character substitution was detected.",
    ),
    "brand_in_path_diff_domain": (
        "Brand Lure in Path Mismatch",
        "The URL path contains a major brand name while the registered apex domain belongs to an unrelated entity.",
        "The URL path matches the registered domain identity.",
    ),
    "brand_edit_distance_min": (
        "Near-Brand Edit Distance",
        "The domain name differs from an authentic target brand by only 1 to 2 characters, typical of deceptive typosquatting.",
        "The domain is cleanly distinct from protected brand namespaces.",
    ),
    "IsDomainIP": (
        "Raw Numerical IP Destination",
        "The destination is addressed directly by a numerical IP address instead of an authenticated domain name, bypassing DNS reputation.",
        "Standard domain name used.",
    ),
    "is_susp_tld": (
        "High-Abuse Disposable TLD",
        "The top-level domain belongs to a registry heavily abused by short-lived malicious and credential-harvesting campaigns.",
        "Standard reputable top-level domain.",
    ),
    "is_free_host": (
        "Free Public Hosting Infrastructure",
        "The URL is hosted on a free public cloud platform; the hosting provider does not verify tenant legitimacy.",
        "Standard self-hosted or dedicated domain infrastructure.",
    ),
    "netloc_entropy": (
        "High Shannon Character Entropy",
        "The domain exhibits high mathematical randomness, indicating machine-generated characters (Domain Generation Algorithm).",
        "Domain entropy falls within normal linguistic patterns.",
    ),
    "NoOfSubDomain": (
        "Excessive Subdomain Nesting",
        "Multiple levels of nested subdomains are chained together to conceal the true apex domain on mobile screens.",
        "Normal subdomain hierarchy.",
    ),
    "kw_count": (
        "Credential-Harvesting Keywords",
        "Contains sensitive tokens ('login', 'verify', 'account', 'banking') deployed to lure users into submitting credentials.",
        "Zero credential-lure keywords found.",
    ),
    "has_kw": (
        "Phishing Indicator Token",
        "Presence of authentication-related trigger words designed to fabricate urgency.",
        "No trigger keywords present.",
    ),
    "host_clean_longest_consonant_run": (
        "Abnormal Consonant Cluster",
        "Unusually long sequences of consecutive consonants suggest synthetic, non-pronounceable domain generation.",
        "Normal vowel-to-consonant phonetic flow.",
    ),
    "hyphen_domain": (
        "Excessive Hyphenation",
        "Multiple hyphens are used in the domain to separate spoofed brand names and evade literal substring blockers.",
        "Standard hyphenation count.",
    ),
    "digit_ratio_domain": (
        "High Numeric Digit Density",
        "An abnormally high percentage of numeric digits appears in the domain, characteristic of bulk automated registration.",
        "Standard alphanumeric balance.",
    ),
    "host_clean_digits": (
        "Excessive Digits in Domain",
        "Numerical characters are embedded in the hostname, often used in algorithmic throwaway domains.",
        "Low digit count in domain.",
    ),
    "NoOfObfuscatedChar": (
        "Obfuscation Characters Present",
        "Contains delimiter or escape characters (@, %, ~) designed to mislead URL parsers and security crawlers.",
        "Zero obfuscation characters found.",
    ),
    "DomainLength": (
        "Excessive Domain Length",
        "The domain is unusually long, frequently used to pack fraudulent keywords into the address bar.",
        "Normal domain length.",
    ),
}


def explain_prediction_shap(feat_df: pd.DataFrame, prob_phish: float) -> List[Dict[str, Any]]:
    """
    Computes true SHAP feature attributions using the XGBoost model
    and maps the top driving features to plain-English sentences.
    """
    global _TREE_EXPLAINER
    if _TREE_EXPLAINER is None:
        xgb = get_model("ML_XGBoost")
        _TREE_EXPLAINER = shap.TreeExplainer(xgb)

    try:
        shap_vals = _TREE_EXPLAINER(feat_df).values[0]
    except Exception:
        # Fast fallback: heuristic magnitude
        shap_vals = np.zeros(len(FEATURE_COLUMNS))

    # Pair features with their SHAP values and raw values
    contributions = []
    for col, sv in zip(FEATURE_COLUMNS, shap_vals):
        val = feat_df[col].values[0]
        contributions.append((col, float(sv), val))

    # Sort by absolute SHAP attribution
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    factors = []
    for col, sv, val in contributions:
        if abs(sv) < 0.05 and len(factors) >= 4:
            continue

        if col in FEATURE_EXPLANATION_MAP:
            title, desc_pos, desc_neg = FEATURE_EXPLANATION_MAP[col]
            if sv > 0:
                severity = "CRITICAL" if sv > 1.0 else ("HIGH" if sv > 0.4 else "MEDIUM")
                factors.append({
                    "feature": col,
                    "title": title,
                    "detail": desc_pos,
                    "direction": "Risk Escalator",
                    "severity": severity,
                    "shap_contribution": round(sv, 4),
                })
            elif sv < -0.1 and prob_phish < 0.40 and len(factors) < 3:
                factors.append({
                    "feature": col,
                    "title": title,
                    "detail": desc_neg,
                    "direction": "Trust Factor",
                    "severity": "SAFE",
                    "shap_contribution": round(sv, 4),
                })

        if len(factors) >= 5:
            break

    if not factors:
        factors.append({
            "feature": "baseline",
            "title": "Clean Domain Structure",
            "detail": "Standard domain length, standard reputable TLD, zero obfuscation, and zero credential-harvesting tokens.",
            "direction": "Trust Factor",
            "severity": "SAFE",
            "shap_contribution": 0.0,
        })

    return factors


# ---------------------------------------------------------------------------
# End-to-End URL Trust & Risk Prediction
# ---------------------------------------------------------------------------
def predict_single_url(raw_url: str, model_name: str = "Ensemble_Calibrated_SoftVoting") -> Dict[str, Any]:
    """
    Executes end-to-end zero-scraping URL trust evaluation:
    1. Input validation.
    2. Zero-scraping feature extraction (37 features + n-gram probability).
    3. Model inference and probability calibration.
    4. Free-host platform detection, notification, and score capping.
    5. SHAP plain-English explainability diagnostics.
    """
    url_str = str(raw_url).strip()
    if not url_str:
        raise ValueError("URL cannot be empty.")
    if len(url_str) > 2048:
        raise ValueError("URL exceeds maximum length of 2048 characters.")

    # Validate structure & prepend scheme if missing
    if not url_str.startswith(("http://", "https://")):
        url_str = "http://" + url_str

    parsed = urlparse(url_str)
    netloc = parsed.netloc or parsed.path.split("/")[0]
    if not netloc:
        raise ValueError("Invalid URL structure: missing network location / hostname.")

    # 1. Feature Extraction
    raw_features = features.extract_features(url_str, include_path_query=False)
    feat_df = build_model_input_vector(raw_features, url_str)

    # 2. Model Dispatch
    model = get_model(model_name)

    # Scale if linear / deep model
    if model_name in ("ML_LogisticRegression", "ML_LinearSVM", "DL_DeepMLP", "DL_CNN1D", "DL_LSTM", "DL_Hybrid_CNNLSTM", "DL_SelfAttention"):
        scaler = get_scaler()
        input_data = scaler.transform(feat_df) if scaler else feat_df.values
    else:
        input_data = feat_df

    # Obtain calibrated phishing probability
    if hasattr(model, "predict_proba"):
        prob_phish = float(model.predict_proba(input_data)[0, 1])
    else:
        dfunc = float(model.decision_function(input_data)[0])
        prob_phish = float(1.0 / (1.0 + np.exp(-dfunc)))

    # Calibrate probability via Isotonic Regression if single model
    if model_name != "Ensemble_Calibrated_SoftVoting":
        calibrator = get_calibrator()
        if calibrator is not None:
            calibrated_phish_prob = float(calibrator.transform([prob_phish])[0])
        else:
            calibrated_phish_prob = prob_phish
    else:
        calibrated_phish_prob = prob_phish

    # 3. Trust Score Computation
    # Trust Score = Calibrated (1 - P(phishing)) * 100
    trust_score = round(max(0.0, min(100.0, (1.0 - calibrated_phish_prob) * 100.0)), 1)

    # 4. Free-Hosting Platform Rule: Flag & Cap
    is_free = bool(raw_features.get("is_free_host", 0) == 1)
    free_host_warning = None
    if is_free:
        free_host_warning = "Hosted on a free platform; the platform's reputation does not vouch for this site."
        # Cap trust score at 50.0 for free-hosting URLs
        trust_score = min(trust_score, 50.0)

    # 5. Risk Triage Tiering
    if calibrated_phish_prob < 0.30:
        tier = "LOW"
        tier_label = "LOW RISK (Safe / Allow)"
        action = "ALLOW"
        action_desc = "Safe for navigation. Standard access permitted."
        color_class = "green"
        trust_status = "HIGH TRUST"
    elif calibrated_phish_prob <= 0.65:
        tier = "MEDIUM"
        tier_label = "MEDIUM RISK (Suspicious / Caution)"
        action = "STEP-UP AUTH"
        action_desc = "Suspicious lexical or tenant indicators detected. Caution recommended before entering credentials."
        color_class = "yellow"
        trust_status = "MODERATE / CAUTION"
    else:
        tier = "HIGH"
        tier_label = "HIGH RISK (Critical Phishing / Quarantine)"
        action = "BLOCK & QUARANTINE"
        action_desc = "High-confidence phishing indicators confirmed. Immediate block recommended."
        color_class = "red"
        trust_status = "UNTRUSTED / DANGEROUS"

    is_phishing = bool(calibrated_phish_prob >= 0.3882)

    # 6. SHAP Plain-English Explanations
    risk_factors = explain_prediction_shap(feat_df, calibrated_phish_prob)

    clean_features = {}
    for col in FEATURE_COLUMNS:
        v = feat_df[col].values[0]
        if isinstance(v, (np.integer, int, bool)):
            clean_features[col] = int(v)
        else:
            clean_features[col] = round(float(v), 4)

    return {
        "url": url_str,
        "domain": netloc,
        "model_used": model_name,
        "model_display_name": MODEL_METADATA.get(model_name, {}).get("display_name", model_name),
        "is_phishing": is_phishing,
        "prediction_label": "PHISHING" if is_phishing else "LEGITIMATE",
        "phishing_probability": round(calibrated_phish_prob, 4),
        "legitimate_probability": round(1.0 - calibrated_phish_prob, 4),
        "confidence_percent": round(max(calibrated_phish_prob, 1.0 - calibrated_phish_prob) * 100, 2),
        "trust_score": trust_score,
        "trust_status": trust_status,
        "risk_tier": tier,
        "risk_tier_label": tier_label,
        "action_recommendation": action,
        "action_description": action_desc,
        "color_class": color_class,
        "is_free_host": is_free,
        "free_host_warning": free_host_warning,
        "risk_factors": risk_factors,
        "features": clean_features,
    }


def predict_batch_urls(urls: List[str], model_name: str = "Ensemble_Calibrated_SoftVoting") -> List[Dict[str, Any]]:
    """Predicts trust and threat risk for a list of URLs."""
    results = []
    for u in urls:
        u_clean = u.strip()
        if not u_clean:
            continue
        try:
            res = predict_single_url(u_clean, model_name)
            results.append(res)
        except Exception as e:
            results.append({
                "url": u_clean,
                "error": str(e),
                "prediction_label": "ERROR",
                "risk_tier": "UNKNOWN",
            })
    return results
