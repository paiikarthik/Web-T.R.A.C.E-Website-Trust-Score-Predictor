"""
Phishing Website Detection on the PhiUSIIL Phishing URL Dataset.

Comparative Benchmark across:
  - 5 Machine Learning Algorithms:
      1. Random Forest (ML_RandomForest)
      2. Extreme Gradient Boosting (ML_XGBoost)
      3. Decision Tree (ML_DecisionTree)
      4. Logistic Regression (ML_LogisticRegression)
      5. Support Vector Machine (ML_LinearSVM)
  - 5 Deep Learning Algorithms:
      1. Deep Multi-Layer Perceptron (DL_DeepMLP)
      2. 1D Convolutional Neural Network (DL_CNN1D) / Deep Pyramidal ANN
      3. Bidirectional Long Short-Term Memory RNN (DL_LSTM) / Wide & Deep ANN
      4. Hybrid Convolutional-Recurrent Network (DL_Hybrid_CNNLSTM) / Tapered Deep ANN
      5. Multi-Head Self-Attention Tabular Network (DL_SelfAttention) / Bottleneck Autoencoder ANN

Pipeline:
  load -> missing value check -> clean -> rich URL feature engineering ->
  train/test split (random & unseen-domain) -> scale on train only ->
  train 10 models -> evaluate (Acc, P, R, F1, AUC) -> operational risk tiering ->
  explainability (SHAP & LIME) -> 5-fold cross-validation -> save models & artifacts.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import warnings
from pathlib import Path
from urllib.parse import urlparse

# Set Matplotlib cache dir before importing matplotlib to avoid permission warnings on Windows
if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(__file__).resolve().parent / "outputs" / ".mpl_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

# ---------------------------------------------------------------------------
# Windows DLL Configuration for PyTorch in Spyder / Conda environments
# Resolves OSError: [WinError 127] shm.dll or one of its dependencies
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    try:
        candidate_paths = [
            Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib",
            Path(sys.executable).parent / "Lib" / "site-packages" / "torch" / "lib",
            Path(sys.prefix) / "Library" / "bin",
        ]
        for p in candidate_paths:
            if p.exists():
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(p))
                os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    warnings.warn(
        f"PyTorch DLL loading failed ({e}). "
        "Falling back to scikit-learn Deep Neural Network architectures."
    )
    TORCH_AVAILABLE = False

import joblib
import matplotlib

# Detect interactive IPython / Spyder environment to preserve Plots pane
IN_INTERACTIVE = any(k in sys.modules for k in ["IPython", "spyder", "spyder_kernels"]) or hasattr(sys, "ps1")
if not IN_INTERACTIVE and os.environ.get("SHOW_PLOTS", "1") == "0":
    matplotlib.use("Agg")  # Headless mode only when explicitly requested

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ndcg_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# SECTION 0 — Project Configuration
# ---------------------------------------------------------------------------
# Set to True to display figures in Spyder's "Plots" pane or popup windows.
# All figures are simultaneously saved to outputs/figures/.
SHOW_PLOTS = True

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_CV_FOLDS = 5

# Set True only while debugging. False uses full dataset.
QUICK_DEMO = False
QUICK_DEMO_ROWS = 25_000

# ---------------------------------------------------------------------------
# EVALUATION MODE:
# "honest_domain": (RECOMMENDED for scientific defense)
#   Strips all path-presence shortcuts (`path_len`, `slash_count`, `path_depth`,
#   `path_entropy`, etc.) to prevent models from cheating on the collection flaw
#   where 100% of legitimate URLs in PhiUSIIL have no paths.
#   Realistic, honest accuracy: ~84.1%, Precision: ~90.1%.
#
# "standard_url":
#   Full URL features including path & character ratios (~99.2% accuracy).
# ---------------------------------------------------------------------------
EVALUATION_MODE = "honest_domain"

# Zero-scraping, pure-URL configuration (no dangerous live downloads or 2-5s latency)
URL_ONLY_FEATURES = True
DROP_DERIVED_FEATURES = True
DROP_IS_HTTPS = True
DROP_SHORTCUT_FEATURES = True

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
EXPLAIN_DIR = OUTPUT_DIR / "explanations"
REPORT_DIR = OUTPUT_DIR / "reports"

IDENTIFIER_COLS = ["FILENAME", "URL", "Domain", "TLD", "Title"]
LEAKY_OR_DERIVED_COLS = [
    "URLSimilarityIndex",
    "CharContinuationRate",
    "TLDLegitimateProb",
    "URLCharProb",
]
SHORTCUT_AND_REDUNDANT_COLS = [
    "URLLength",
    "NoOfLettersInURL",
]
HTML_FEATURE_COLS = [
    "LineOfCode",
    "LargestLineLength",
    "HasTitle",
    "DomainTitleMatchScore",
    "URLTitleMatchScore",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
    "HasDescription",
    "NoOfPopup",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfExternalRef",
]

SUSPICIOUS_KEYWORDS = (
    "login", "signin", "verify", "verification", "account", "update",
    "banking", "secure", "security", "webscr", "confirm", "password",
    "wallet", "admin", "service", "support", "billing", "auth",
    "ebayisapi", "paypal", "apple", "microsoft", "google", "netflix"
)

SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "work", "icu", "buzz", "cn", "ru"
}


def ensure_output_dirs() -> None:
    for folder in (OUTPUT_DIR, FIGURE_DIR, MODEL_DIR, EXPLAIN_DIR, REPORT_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def save_json(data, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# SECTION 1 — Data Ingestion & Preprocessing
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place PhiUSIIL_Phishing_URL_Dataset.csv next to this script."
        )
    df = pd.read_csv(path)
    print(f"[1] Loaded {df.shape[0]:,} rows and {df.shape[1]} columns from {path.name}")
    return df


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing_count": df.isna().sum().values,
            "missing_percent": (100.0 * df.isna().mean().values).round(4),
            "n_unique": df.nunique(dropna=False).values,
        }
    )
    report.to_csv(REPORT_DIR / "missing_value_report.csv", index=False)
    missing_only = report[report["missing_count"] > 0]
    print(f"[2] Missing-value columns: {len(missing_only)} (full table saved)")
    return report


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    n_dup_url = int(df["URL"].duplicated().sum())
    df = df.drop_duplicates(subset=["URL"], keep="first").copy()
    print(f"[3] Dropped {n0 - len(df):,} duplicate URLs ({n_dup_url:,} extra copies)")

    if QUICK_DEMO and len(df) > QUICK_DEMO_ROWS:
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), QUICK_DEMO_ROWS // 2), random_state=RANDOM_STATE))
            .reset_index(drop=True)
        )
        print(f"    QUICK_DEMO=True -> using {len(df):,} stratified rows")

    # Positive class (1) = Phishing, Negative class (0) = Legitimate
    df["y"] = (df["label"] == 0).astype(int)
    print("    Class distribution (1 = phishing, 0 = legitimate):")
    print(df["y"].value_counts().rename({1: "phishing", 0: "legitimate"}).to_string())
    return df


# ---------------------------------------------------------------------------
# SECTION 2 — Unified Feature Extraction Engine (Zero Scraping)
# ---------------------------------------------------------------------------
from features import (
    shannon_entropy,
    extract_features,
    extract_features_df,
    extract_host_clean,
    get_domain_group,
    GENERAL_FEATURE_NAMES,
)

def extract_rich_url_features(urls: pd.Series) -> pd.DataFrame:
    """
    Extracts features via the unified features.py module (zero duplication).
    """
    return extract_features_df(urls, include_path_query=False)


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    drop_cols = set(IDENTIFIER_COLS + ["label", "y"])
    if DROP_DERIVED_FEATURES:
        drop_cols.update(LEAKY_OR_DERIVED_COLS)
        print("[4] Dropping derived/leaky columns:", ", ".join(LEAKY_OR_DERIVED_COLS))
    if URL_ONLY_FEATURES:
        drop_cols.update(HTML_FEATURE_COLS)
        print("[4] URL_ONLY_FEATURES=True -> HTML/page-source columns removed (zero-scraping)")
    if DROP_IS_HTTPS:
        drop_cols.add("IsHTTPS")
        print("[4] Dropping IsHTTPS (every legitimate row in this CSV is HTTPS)")
    if DROP_SHORTCUT_FEATURES:
        drop_cols.update(SHORTCUT_AND_REDUNDANT_COLS)
        print("[4] Dropping raw shortcut features (URLLength, NoOfLettersInURL) to prevent algebraic cheating")

    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = feature_df.select_dtypes(include=[np.number]).copy()

    print("[4] Extracting rich URL-native features (entropy, domain characteristics, keywords)...")
    rich_url_df = extract_rich_url_features(df["URL"])
    overlap_cols = [c for c in rich_url_df.columns if c in X.columns]
    if overlap_cols:
        X = X.drop(columns=overlap_cols)
    X = pd.concat([X.reset_index(drop=True), rich_url_df.reset_index(drop=True)], axis=1)
    X = X.loc[:, ~X.columns.duplicated()].copy()

    # Apply Honest Domain Filter if configured
    if EVALUATION_MODE == "honest_domain":
        print("\n" + "=" * 70)
        print("=== EVALUATION MODE: HONEST PATH-AGNOSTIC DOMAIN DETECTION ===")
        print("=== Stripping path-presence shortcuts to prevent artificial 100% accuracy ===")
        print("=" * 70)
        path_shortcuts = [
            "slash_count", "path_depth", "path_len", "path_entropy",
            "url_entropy", "kw_in_path", "hyphen_path", "has_double_slash_path",
            "has_query", "query_len", "NoOfEqualsInURL", "NoOfQMarkInURL",
            "NoOfAmpersandInURL", "NoOfOtherSpecialCharsInURL", "NoOfDegitsInURL",
            "SpacialCharRatioInURL", "DegitRatioInURL", "LetterRatioInURL", "ObfuscationRatio"
        ]
        drop_path_cols = [c for c in path_shortcuts if c in X.columns]
        X = X.drop(columns=drop_path_cols)
        print(f"[4] Retained honest domain features: {list(X.columns)}")

    y = df["y"].astype(int).reset_index(drop=True)
    groups = df["Domain"].astype(str).reset_index(drop=True)
    print(f"[4] Feature matrix X: {X.shape[0]:,} samples x {X.shape[1]} features")
    pd.Series(X.columns, name="feature").to_csv(REPORT_DIR / "feature_list.csv", index=False)
    return X, y, groups


# ---------------------------------------------------------------------------
# SECTION 3 — Deep Learning Architectures (PyTorch or Sklearn Fallback)
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class DeepMLP(nn.Module):
        def __init__(self, in_features: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)

    class CNN1DNet(nn.Module):
        def __init__(self, in_features: int):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(32)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(64)
            self.adapt_pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            x = x.unsqueeze(1)
            x = self.pool(self.relu(self.bn1(self.conv1(x))))
            x = self.adapt_pool(self.relu(self.bn2(self.conv2(x)))).squeeze(-1)
            return self.fc(x)

    class LSTMNet(nn.Module):
        def __init__(self, in_features: int):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
            self.fc = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            x = x.unsqueeze(-1)
            out, _ = self.lstm(x)
            feat = out[:, -1, :]
            return self.fc(feat)

    class CNNLSTMNet(nn.Module):
        def __init__(self, in_features: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )
            self.lstm = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            x = x.unsqueeze(1)
            feat = self.conv(x).permute(0, 2, 1)
            out, _ = self.lstm(feat)
            return self.fc(out[:, -1, :])

    class SelfAttentionNet(nn.Module):
        def __init__(self, in_features: int):
            super().__init__()
            self.in_features = in_features
            self.embed = nn.Linear(1, 16)
            self.attn = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(16 * in_features, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            x = x.unsqueeze(-1)
            emb = self.embed(x)
            attn_out, _ = self.attn(emb, emb, emb)
            flat = attn_out.reshape(attn_out.size(0), -1)
            return self.fc(flat)

    class PyTorchClassifierWrapper(BaseEstimator, ClassifierMixin):
        _estimator_type = "classifier"

        def __init__(self, model_class, epochs=8, batch_size=256, lr=0.002, random_state=42):
            self.model_class = model_class
            self.epochs = epochs
            self.batch_size = batch_size
            self.lr = lr
            self.random_state = random_state
            self.model_ = None

        def fit(self, X, y):
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            X_arr = np.asarray(X, dtype=np.float32)
            y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)

            self.model_ = self.model_class(X_arr.shape[1])
            self.model_.train()

            loader = DataLoader(
                TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y_arr)),
                batch_size=self.batch_size,
                shuffle=True,
            )
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

            self.loss_curve_ = []
            for _ in range(self.epochs):
                running_loss = 0.0
                total_samples = 0
                for bx, by in loader:
                    optimizer.zero_grad()
                    out = self.model_(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                    running_loss += float(loss.item()) * len(bx)
                    total_samples += len(bx)
                epoch_loss = float(running_loss / max(1, total_samples))
                self.loss_curve_.append(epoch_loss)

            self.model_.eval()
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, X):
            X_arr = np.asarray(X, dtype=np.float32)
            self.model_.eval()
            with torch.no_grad():
                p = self.model_(torch.from_numpy(X_arr)).numpy().flatten()
            p = np.clip(p, 1e-7, 1.0 - 1e-7)
            return np.column_stack([1.0 - p, p])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# SECTION 4 — Estimators Setup: 5 ML + 5 DL Models
# ---------------------------------------------------------------------------
def make_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def make_model_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            ("model", estimator),
        ]
    )


def get_all_estimators() -> dict:
    """Returns 5 Machine Learning algorithms and 5 Deep Learning algorithms."""
    estimators = {
        # --- 5 Machine Learning Algorithms ---
        "ML_RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "ML_XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.08,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "ML_DecisionTree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=12,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
        ),
        "ML_LogisticRegression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "ML_LinearSVM": LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual=False,
            max_iter=3000,
            random_state=RANDOM_STATE,
        ),
    }

    # --- 5 Deep Learning Algorithms ---
    if TORCH_AVAILABLE:
        estimators.update({
            "DL_DeepMLP": PyTorchClassifierWrapper(DeepMLP, epochs=8, batch_size=256, lr=0.002, random_state=RANDOM_STATE),
            "DL_CNN1D": PyTorchClassifierWrapper(CNN1DNet, epochs=8, batch_size=256, lr=0.002, random_state=RANDOM_STATE),
            "DL_LSTM": PyTorchClassifierWrapper(LSTMNet, epochs=8, batch_size=256, lr=0.002, random_state=RANDOM_STATE),
            "DL_Hybrid_CNNLSTM": PyTorchClassifierWrapper(CNNLSTMNet, epochs=8, batch_size=256, lr=0.002, random_state=RANDOM_STATE),
            "DL_SelfAttention": PyTorchClassifierWrapper(SelfAttentionNet, epochs=8, batch_size=256, lr=0.002, random_state=RANDOM_STATE),
        })
    else:
        # High-capacity Scikit-Learn Deep Neural Architectures (Zero-dependency fallback)
        estimators.update({
            "DL_DeepMLP": MLPClassifier(hidden_layer_sizes=(128, 64, 32, 16), activation="relu", max_iter=35, early_stopping=True, random_state=RANDOM_STATE),
            "DL_WideAndDeep": MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu", max_iter=35, early_stopping=True, random_state=RANDOM_STATE),
            "DL_PyramidalNet": MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu", max_iter=35, early_stopping=True, random_state=RANDOM_STATE),
            "DL_TaperedANN": MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation="relu", max_iter=35, early_stopping=True, random_state=RANDOM_STATE),
            "DL_BottleneckAutoencoder": MLPClassifier(hidden_layer_sizes=(64, 16, 64), activation="tanh", max_iter=35, early_stopping=True, random_state=RANDOM_STATE),
        })

    return estimators


def predict_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    model = pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(X)


def metric_dict(y_true, y_pred, y_score) -> dict:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)

    # Confusion matrix elements: TP, TN, FP, FN
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    # Rates: Sensitivity (TPR) and Specificity (TNR)
    tpr = float(tp / max(1, tp + fn))
    tnr = float(tn / max(1, tn + fp))
    fpr = float(fp / max(1, fp + tn))
    fnr = float(fn / max(1, fn + tp))

    acc = float(accuracy_score(y_true, y_pred))
    p = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    r = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_score))

    # 1. PR AUC (Precision-Recall Area Under Curve / Average Precision)
    pr_auc = float(average_precision_score(y_true, y_score))

    # 2. Spearman Rank Correlation (Monotonicity of predicted risk scores)
    spearman_rho, _ = stats.spearmanr(y_score_arr, y_true_arr)

    # 3. Kendall's Tau (Pairwise ranking concordance)
    kendall_tau, _ = stats.kendalltau(y_score_arr, y_true_arr)

    # 4. NDCG@k (Normalized Discounted Cumulative Gain at rank k)
    ndcg_10 = float(ndcg_score([y_true_arr], [y_score_arr], k=10))
    ndcg_50 = float(ndcg_score([y_true_arr], [y_score_arr], k=50))
    ndcg_100 = float(ndcg_score([y_true_arr], [y_score_arr], k=100))
    ndcg_500 = float(ndcg_score([y_true_arr], [y_score_arr], k=500))
    ndcg_1000 = float(ndcg_score([y_true_arr], [y_score_arr], k=1000))

    return {
        "accuracy": acc,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "tpr_sensitivity": tpr,
        "tnr_specificity": tnr,
        "fpr_fallout": fpr,
        "fnr_miss_rate": fnr,
        "precision_phishing": p,
        "recall_phishing": r,
        "f1_phishing": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "spearman_rho": float(spearman_rho),
        "kendall_tau": float(kendall_tau),
        "ndcg_10": ndcg_10,
        "ndcg_50": ndcg_50,
        "ndcg_100": ndcg_100,
        "ndcg_500": ndcg_500,
        "ndcg_1000": ndcg_1000,
    }


# ---------------------------------------------------------------------------
# SECTION 5 — Visualizations & Reports
# ---------------------------------------------------------------------------
def plot_confusion(cm: np.ndarray, title: str, path: Path) -> None:
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    total = np.sum(cm)
    acc = (tp + tn) / max(1, total)
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp)
    f1 = 2 * (prec * tpr) / max(1e-9, prec + tpr)

    labels = np.array([
        [f"True Negative (TN)\n{tn:,}\n({tn/total:.1%})", f"False Positive (FP)\n{fp:,}\n({fp/total:.1%})"],
        [f"False Negative (FN)\n{fn:,}\n({fn/total:.1%})", f"True Positive (TP)\n{tp:,}\n({tp/total:.1%})"],
    ])
  
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Predicted Legit (0)", "Predicted Phish (1)"],
        yticklabels=["Actual Legit (0)", "Actual Phish (1)"],
        ax=ax,
        annot_kws={"fontsize": 10.5, "fontweight": "semibold"},
    )
    ax.set_title(
        f"{title}\nTP={tp:,} | TN={tn:,} | FP={fp:,} | FN={fn:,}\n"
        f"Accuracy={acc:.2%} | F1={f1:.2%} | Sensitivity={tpr:.2%} | Specificity={tnr:.2%}",
        fontsize=10.5,
        pad=12,
        fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_metric_bars(results: pd.DataFrame, path: Path) -> None:
    plot_df = results.melt(id_vars=["model", "paradigm"], var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(15, 7))
    palette = sns.color_palette("tab10", n_colors=results["model"].nunique())
    sns.barplot(data=plot_df, x="metric", y="score", hue="model", ax=ax, palette=palette)
    ax.set_ylim(0, 1.18)
    ax.set_title(f"Benchmark: 5 ML & 5 DL Models (Mode: {EVALUATION_MODE})\n[Values Annotated Above Each Bar]", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_ylabel("Score (0.0 to 1.0)", fontweight="bold")
    ax.set_xlabel("Evaluation Metric", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=6.8, rotation=90, fontweight="bold")
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8.5, frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_roc(roc_data: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    for name, (fpr, tpr, auc) in roc_data.items():
        linestyle = "--" if name.startswith("DL_") else "-"
        ax.plot(fpr, tpr, linestyle=linestyle, linewidth=1.8, label=f"{name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle=":", color="gray", label="Random Chance (AUC = 0.5000)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontweight="bold")
    ax.set_title(f"ROC Curves: 5 ML vs 5 DL Models ({EVALUATION_MODE})", fontsize=12.5, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_pr(pr_data: dict, baseline_prior: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    for name, (recalls, precisions, pr_auc) in pr_data.items():
        linestyle = "--" if name.startswith("DL_") else "-"
        ax.plot(recalls, precisions, linestyle=linestyle, linewidth=1.8, label=f"{name} (PR-AUC = {pr_auc:.4f})")
    ax.axhline(y=baseline_prior, linestyle=":", color="red", label=f"No-Skill Prior ({baseline_prior:.4f})")
    ax.set_xlabel("Recall (Phishing)", fontweight="bold")
    ax.set_ylabel("Precision (Phishing)", fontweight="bold")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-0.02, 1.02)
    ax.set_title(f"Precision-Recall (PR) Curves: 5 ML vs 5 DL Models ({EVALUATION_MODE})", fontsize=12.5, fontweight="bold", pad=10)
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_loss_curves(loss_histories: dict, path: Path) -> None:
    if not loss_histories:
        return
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    colors = sns.color_palette("deep", n_colors=len(loss_histories))
    for idx, (name, losses) in enumerate(loss_histories.items()):
        epochs = list(range(1, len(losses) + 1))
        linestyle = "--" if "Attention" in name or "Autoencoder" in name else "-"
        init_loss = losses[0]
        final_loss = losses[-1]
        pct_drop = ((final_loss - init_loss) / init_loss) * 100
        ax.plot(
            epochs, losses, marker="o", markersize=5, linewidth=2.0, linestyle=linestyle, color=colors[idx],
            label=f"{name}: Start={init_loss:.4f} → End={final_loss:.4f} ({pct_drop:+.1f}%)"
        )
        ax.annotate(
            f"{final_loss:.4f}",
            (epochs[-1], final_loss),
            textcoords="offset points",
            xytext=(6, -2),
            fontsize=8.5,
            fontweight="bold",
            color=colors[idx]
        )
    ax.set_xlabel("Epoch / Iteration (Batch Size = 128, Adam)", fontweight="bold")
    ax.set_ylabel("Binary Cross-Entropy Loss (BCE)", fontweight="bold")
    ax.set_title(f"Deep Learning Training Loss Convergence ({EVALUATION_MODE})\n[Start & Final Loss Annotated]", fontsize=12.5, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_ranking_bars(results: pd.DataFrame, path: Path) -> None:
    ranking_cols = ["model", "paradigm", "pr_auc", "spearman_rho", "kendall_tau", "ndcg_10", "ndcg_100", "ndcg_1000"]
    available_cols = [c for c in ranking_cols if c in results.columns]
    plot_df = results[available_cols].melt(id_vars=["model", "paradigm"], var_name="ranking_metric", value_name="score")
    fig, ax = plt.subplots(figsize=(15, 7))
    palette = sns.color_palette("tab10", n_colors=results["model"].nunique())
    sns.barplot(data=plot_df, x="ranking_metric", y="score", hue="model", ax=ax, palette=palette)
    ax.set_ylim(0, 1.20)
    ax.set_title(f"Operational Ranking Metrics: PR AUC, Spearman Rho, Kendall Tau, NDCG@k ({EVALUATION_MODE})\n[Scores Annotated Above Each Bar]", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_ylabel("Metric Value", fontweight="bold")
    ax.set_xlabel("Ranking / Prioritization Metric", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=6.8, rotation=90, fontweight="bold")
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8.5, frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_tp_tn(tp_tn_df: pd.DataFrame, path: Path) -> None:
    df = tp_tn_df.sort_values("True_Positive_TP", ascending=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    x = np.arange(len(df))
    width = 0.38
    bars_tp = ax1.bar(x - width/2, df["True_Positive_TP"], width, label="True Positives (Phish Caught)", color="#2b5c8f", edgecolor="black", alpha=0.9)
    bars_tn = ax1.bar(x + width/2, df["True_Negative_TN"], width, label="True Negatives (Legit Allowed)", color="#469b77", edgecolor="black", alpha=0.9)

    ax1.set_title(f"Absolute Correct Classifications: TP vs TN ({EVALUATION_MODE})\n[Exact URL Counts Labeled on Bars]", fontweight="bold")
    ax1.set_ylabel("Number of URLs Correctly Identified", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Model"], rotation=45, ha="right", fontsize=9)
    ax1.set_ylim(0, max(df["True_Negative_TN"].max(), df["True_Positive_TP"].max()) * 1.18)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    for bar in bars_tp:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 300, f"{int(h):,}", ha="center", va="bottom", fontsize=8, rotation=90, fontweight="bold", color="#163250")

    for bar in bars_tn:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 300, f"{int(h):,}", ha="center", va="bottom", fontsize=8, rotation=90, fontweight="bold", color="#1b4d36")

    bars_tpr = ax2.bar(x - width/2, df["TPR_Sensitivity"] * 100, width, label="Sensitivity / TPR (% Threats Caught)", color="#1f77b4", edgecolor="black", alpha=0.9)
    bars_tnr = ax2.bar(x + width/2, df["TNR_Specificity"] * 100, width, label="Specificity / TNR (% Benign Kept)", color="#2ca02c", edgecolor="black", alpha=0.9)

    ax2.set_title(f"Operational Detection Rates: TPR vs TNR (%)\n[Percentage Rates Labeled on Bars]", fontweight="bold")
    ax2.set_ylabel("Rate Percentage (%)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["Model"], rotation=45, ha="right", fontsize=9)
    ax2.set_ylim(0, 115)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True)

    for bar in bars_tpr:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, rotation=90, fontweight="bold", color="#0e4468")

    for bar in bars_tnr:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, rotation=90, fontweight="bold", color="#115511")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# SECTION 6 — Train, Evaluate, and Risk Stratification
# ---------------------------------------------------------------------------
def make_splits(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[5] Random stratified split -> train {len(X_train):,} / test {len(X_test):,}")

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))
    domain_split = {
        "X_train": X.iloc[train_idx],
        "X_test": X.iloc[test_idx],
        "y_train": y.iloc[train_idx],
        "y_test": y.iloc[test_idx],
    }
    overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    print(f"[5] Unseen-domain split -> train {len(train_idx):,} / test {len(test_idx):,}; overlapping domains = {len(overlap)}")
    return (X_train, X_test, y_train, y_test), domain_split


def train_and_evaluate(X_train, X_test, y_train, y_test, tag: str) -> pd.DataFrame:
    X_train = X_train.loc[:, ~X_train.columns.duplicated()].copy()
    X_test = X_test.loc[:, ~X_test.columns.duplicated()].copy()
    estimators = get_all_estimators()
    rows = []
    roc_data = {}
    pr_data = {}
    loss_histories = {}
    classification_texts = []

    baseline_prior = float(np.mean(y_test))

    for name, estimator in estimators.items():
        paradigm = "Deep Learning" if name.startswith("DL_") else "Machine Learning"
        print(f"[6] Training {name} ({paradigm}) on split '{tag}' ...")
        pipe = make_model_pipeline(estimator)
        pipe.fit(X_train, y_train)

        # Extract loss history if available
        model_step = pipe.named_steps["model"]
        if hasattr(model_step, "loss_curve_"):
            loss_histories[name] = model_step.loss_curve_

        y_pred = pipe.predict(X_test)
        y_score = predict_scores(pipe, X_test)
        metrics = metric_dict(y_test, y_pred, y_score)
        metrics["model"] = name
        metrics["paradigm"] = paradigm
        metrics["split"] = tag
        rows.append(metrics)

        # Operational Risk Assessment Tiering (Low / Medium / High)
        if hasattr(model_step, "predict_proba"):
            probs = pipe.predict_proba(X_test)[:, 1]
            tiers = np.where(
                probs < 0.35,
                "LOW (Safe / Allow)",
                np.where(probs < 0.70, "MEDIUM (Suspicious / Step-Up Auth)", "HIGH (Critical / Quarantine)"),
            )
            tier_df = pd.Series(tiers, name="count").value_counts().to_frame()
            tier_df.to_csv(REPORT_DIR / f"risk_tier_{tag}_{name}.csv")

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        plot_confusion(cm, f"{name} confusion matrix ({tag})", FIGURE_DIR / f"confusion_{tag}_{name}.png")
        pd.DataFrame(cm, index=["actual_legit", "actual_phish"], columns=["pred_legit", "pred_phish"]).to_csv(
            REPORT_DIR / f"confusion_{tag}_{name}.csv"
        )

        report = classification_report(
            y_test, y_pred, labels=[0, 1], target_names=["legitimate", "phishing"], digits=4, zero_division=0
        )
        classification_texts.append(f"===== {name} ({tag}) =====\n" + str(report) + "\n")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_data[name] = (fpr, tpr, metrics["roc_auc"])

        # Precision-Recall Curve
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_score)
        pr_data[name] = (rec_curve, prec_curve, metrics["pr_auc"])

        if tag == "random":
            joblib.dump(pipe, MODEL_DIR / f"{name}.pkl")

        print(
            f"    acc={metrics['accuracy']:.4f}  "
            f"TP={metrics['true_positives']}  "
            f"TN={metrics['true_negatives']}  "
            f"FP={metrics['false_positives']}  "
            f"FN={metrics['false_negatives']}  "
            f"Sens(TPR)={metrics['tpr_sensitivity']:.4f}  "
            f"Spec(TNR)={metrics['tnr_specificity']:.4f}  "
            f"P={metrics['precision_phishing']:.4f}  "
            f"F1={metrics['f1_phishing']:.4f}  "
            f"AUC={metrics['roc_auc']:.4f}"
        )

    results = pd.DataFrame(rows)
    results.to_csv(REPORT_DIR / f"metrics_{tag}.csv", index=False)

    tp_tn_rows = []
    for r in rows:
        tp = r["true_positives"]
        tn = r["true_negatives"]
        fp = r["false_positives"]
        fn = r["false_negatives"]
        tot = tp + tn + fp + fn
        tp_tn_rows.append({
            "Model": r["model"],
            "Paradigm": r["paradigm"],
            "Split": tag,
            "True_Positive_TP": tp,
            "True_Negative_TN": tn,
            "False_Positive_FP": fp,
            "False_Negative_FN": fn,
            "Total_Samples": tot,
            "TPR_Sensitivity": r["tpr_sensitivity"],
            "TNR_Specificity": r["tnr_specificity"],
            "FPR_Fallout": r["fpr_fallout"],
            "FNR_MissRate": r["fnr_miss_rate"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision_phishing"],
            "F1_Score": r["f1_phishing"],
        })
    tp_tn_df = pd.DataFrame(tp_tn_rows)
    tp_tn_df.to_csv(REPORT_DIR / f"tp_tn_evaluation_{tag}.csv", index=False)
    (REPORT_DIR / f"classification_report_{tag}.txt").write_text("\n".join(classification_texts), encoding="utf-8")

    # Save loss history and plot loss graph
    if loss_histories:
        max_len = max(len(v) for v in loss_histories.values())
        loss_df = pd.DataFrame({"epoch": list(range(1, max_len + 1))})
        for m_name, losses in loss_histories.items():
            padded = losses + [np.nan] * (max_len - len(losses))
            loss_df[m_name] = padded
        loss_df.to_csv(REPORT_DIR / f"dl_training_loss_{tag}.csv", index=False)
        plot_loss_curves(loss_histories, FIGURE_DIR / f"loss_curves_{tag}.png")

    # Classification & Ranking Plots
    plot_metric_bars(
        results[["model", "paradigm", "accuracy", "precision_phishing", "recall_phishing", "f1_phishing", "roc_auc", "pr_auc"]],
        FIGURE_DIR / f"model_comparison_{tag}.png",
    )
    plot_tp_tn(tp_tn_df, FIGURE_DIR / f"tp_tn_comparison_{tag}.png")
    plot_ranking_bars(results, FIGURE_DIR / f"ranking_metrics_{tag}.png")
    plot_roc(roc_data, FIGURE_DIR / f"roc_curves_{tag}.png")
    plot_pr(pr_data, baseline_prior, FIGURE_DIR / f"pr_curves_{tag}.png")
    return results


# ---------------------------------------------------------------------------
# SECTION 7 — 5-Fold Stratified Cross-Validation
# ---------------------------------------------------------------------------
def run_cross_validation(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    print("\n[7] Five-fold stratified cross-validation on representative models...")
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    # Representative models for cross-validation
    cv_models = {
        "ML_RandomForest": RandomForestClassifier(n_estimators=100, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1),
        "ML_XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.08, random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1),
        "ML_DecisionTree": DecisionTreeClassifier(max_depth=12, random_state=RANDOM_STATE),
        "ML_LogisticRegression": LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
    }

    rows = []
    for name, estimator in cv_models.items():
        pipe = make_model_pipeline(estimator)
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1, return_train_score=False)
        row = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
            print(f"    {name:22s} {metric:10s} {row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}")
        rows.append(row)

    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(REPORT_DIR / "cv_five_fold.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    bars = ax.bar(cv_df["model"], cv_df["f1_mean"], yerr=cv_df["f1_std"], capsize=6, color="#2b5c8f", edgecolor="black", alpha=0.88, width=0.55)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 (Phishing) Mean ± Std Deviation", fontweight="bold")
    ax.set_xlabel("Cross-Validated Model", fontweight="bold")
    ax.set_title("5-Fold Stratified Cross-Validation F1-Score Stability\n[Exact Mean ± Std Annotated Above Bars]", fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, mean_val, std_val in zip(bars, cv_df["f1_mean"], cv_df["f1_std"]):
        ax.text(bar.get_x() + bar.get_width() / 2, mean_val + std_val + 0.025, f"{mean_val:.4f}\n±{std_val:.4f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#122d4a")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "cv_f1_comparison.png", dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return cv_df


# ---------------------------------------------------------------------------
# SECTION 8 — Explainability (SHAP & LIME)
# ---------------------------------------------------------------------------
def run_shap(random_forest_pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    print("\n[8] SHAP Explanations for Tree Ensemble (class 1 = phishing)...")
    pre = random_forest_pipeline.named_steps["preprocess"]
    rf = random_forest_pipeline.named_steps["model"]
    X_sm = pd.DataFrame(pre.transform(X_test.sample(n=min(300, len(X_test)), random_state=RANDOM_STATE)), columns=X_test.columns)

    explainer = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_sm.values)

    if isinstance(shap_values, list):
        sv_phish = shap_values[1]
    elif getattr(shap_values, "ndim", 0) == 3:
        sv_phish = shap_values[:, :, 1]
    else:
        sv_phish = shap_values

    plt.figure()
    shap.summary_plot(sv_phish, X_sm, show=False, max_display=min(20, X_sm.shape[1]))
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "shap_summary_random_forest.png", dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    mean_abs = np.abs(sv_phish).mean(axis=0)
    importance = pd.DataFrame({"feature": X_sm.columns, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    importance.to_csv(REPORT_DIR / "shap_mean_abs_random_forest.csv", index=False)
    print("    Saved SHAP summary plot and feature importance table.")


def run_lime(pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    print("[8] LIME Local Instance Explanations (5 test URLs)...")
    train_values = X_train.to_numpy(dtype=float)
    explainer = LimeTabularExplainer(
        train_values,
        feature_names=list(X_train.columns),
        class_names=["legitimate", "phishing"],
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )

    def predict_fn(instances):
        frame = pd.DataFrame(instances, columns=X_train.columns)
        return pipeline.predict_proba(frame)

    n_explain = min(5, len(X_test))
    sample = X_test.sample(n=n_explain, random_state=RANDOM_STATE)
    records = []
    for i, (idx, row) in enumerate(sample.iterrows(), start=1):
        exp = explainer.explain_instance(
            row.to_numpy(dtype=float),
            predict_fn,
            num_features=min(10, X_train.shape[1]),
            labels=(1,),
        )
        html_path = EXPLAIN_DIR / f"lime_instance_{i}.html"
        exp.save_to_file(str(html_path))
        records.append({
            "instance": i,
            "row_index": int(idx) if np.issubdtype(type(idx), np.integer) else str(idx),
            "true_label": int(y_test.loc[idx]),
            "true_name": "phishing" if int(y_test.loc[idx]) == 1 else "legitimate",
            "html_file": html_path.name,
            "top_features": [
                {"rule": feat, "weight_toward_phishing": float(weight)}
                for feat, weight in exp.as_list(label=1)
            ],
        })
    save_json(records, REPORT_DIR / "lime_explanations.json")
    print(f"    Wrote {n_explain} LIME HTML instance files.")


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def main() -> None:
    ensure_output_dirs()
    sns.set_theme(style="whitegrid")

    df_raw = load_dataset(DATA_PATH)
    missing_value_report(df_raw)
    df = clean_dataset(df_raw)
    X, y, groups = build_feature_matrix(df)

    (X_train, X_test, y_train, y_test), domain_split = make_splits(X, y, groups)

    print("\n==========================================================================")
    print("=== RANDOM STRATIFIED EVALUATION (10 Models: 5 ML + 5 DL) ===")
    print("==========================================================================")
    random_results = train_and_evaluate(X_train, X_test, y_train, y_test, tag="random")

    print("\n==========================================================================")
    print("=== UNSEEN-DOMAIN EVALUATION (Zero Domain Overlap: 5 ML + 5 DL) ===")
    print("==========================================================================")
    domain_results = train_and_evaluate(
        domain_split["X_train"],
        domain_split["X_test"],
        domain_split["y_train"],
        domain_split["y_test"],
        tag="unseen_domain",
    )

    combined = pd.concat([random_results, domain_results], ignore_index=True)
    combined.to_csv(REPORT_DIR / "metrics_all_splits.csv", index=False)

    # Automatically synchronize final_model_comparison.csv
    rand = random_results.set_index("model")
    unseen = domain_results.set_index("model")

    comparison = pd.DataFrame({
        "Model": rand.index,
        "Paradigm": rand["paradigm"].values,
        "Random_Accuracy": rand["accuracy"].round(4).values,
        "Random_F1": rand["f1_phishing"].round(4).values,
        "Random_AUC": rand["roc_auc"].round(4).values,
        "Random_PR_AUC": rand["pr_auc"].round(4).values,
        "UnseenDomain_Accuracy": unseen.loc[rand.index, "accuracy"].round(4).values,
        "UnseenDomain_Precision": unseen.loc[rand.index, "precision_phishing"].round(4).values,
        "UnseenDomain_Recall": unseen.loc[rand.index, "recall_phishing"].round(4).values,
        "UnseenDomain_F1": unseen.loc[rand.index, "f1_phishing"].round(4).values,
        "UnseenDomain_AUC": unseen.loc[rand.index, "roc_auc"].round(4).values,
        "UnseenDomain_PR_AUC": unseen.loc[rand.index, "pr_auc"].round(4).values,
        "UnseenDomain_Spearman": unseen.loc[rand.index, "spearman_rho"].round(4).values,
        "UnseenDomain_KendallTau": unseen.loc[rand.index, "kendall_tau"].round(4).values,
        "UnseenDomain_NDCG100": unseen.loc[rand.index, "ndcg_100"].round(4).values,
    }).sort_values(by="UnseenDomain_Accuracy", ascending=False)
    comparison.to_csv(REPORT_DIR / "final_model_comparison.csv", index=False)

    # Standalone Ranking Metrics Table
    ranking_table = pd.DataFrame({
        "Model": unseen.index,
        "Paradigm": unseen["paradigm"].values,
        "PR_AUC": unseen["pr_auc"].round(4).values,
        "ROC_AUC": unseen["roc_auc"].round(4).values,
        "Spearman_Rho": unseen["spearman_rho"].round(4).values,
        "Kendall_Tau": unseen["kendall_tau"].round(4).values,
        "NDCG@10": unseen["ndcg_10"].round(4).values,
        "NDCG@50": unseen["ndcg_50"].round(4).values,
        "NDCG@100": unseen["ndcg_100"].round(4).values,
        "NDCG@500": unseen["ndcg_500"].round(4).values,
        "NDCG@1000": unseen["ndcg_1000"].round(4).values,
    }).sort_values(by="NDCG@100", ascending=False)
    ranking_table.to_csv(REPORT_DIR / "ranking_metrics_evaluation.csv", index=False)

    run_cross_validation(X_train, y_train)

    rf_path = MODEL_DIR / "ML_RandomForest.pkl"
    if rf_path.exists():
        rf_pipe = joblib.load(rf_path)
        run_shap(rf_pipe, X_train, X_test)
        run_lime(rf_pipe, X_train, X_test, y_test)

    config = {
        "dataset": str(DATA_PATH.name),
        "evaluation_mode": EVALUATION_MODE,
        "n_rows_after_clean": int(len(X)),
        "n_features": int(X.shape[1]),
        "features": list(X.columns),
        "torch_available": TORCH_AVAILABLE,
        "positive_class": "phishing (y=1); original label 0",
        "drop_derived_features": DROP_DERIVED_FEATURES,
        "drop_is_https": DROP_IS_HTTPS,
        "drop_shortcut_features": DROP_SHORTCUT_FEATURES,
        "url_only_features": URL_ONLY_FEATURES,
        "quick_demo": QUICK_DEMO,
        "test_size": TEST_SIZE,
        "cv_folds": N_CV_FOLDS,
        "random_state": RANDOM_STATE,
    }
    save_json(config, REPORT_DIR / "run_config.json")
    print("\n[Done] All 10 models (5 ML + 5 DL) trained, evaluated, and saved in outputs/ folder.")


if __name__ == "__main__":
    main()
