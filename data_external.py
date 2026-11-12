"""
Web T.R.A.C.E — External Data Loader & Free-Host Slice Engine (data_external.py).

Addresses the critical dataset artifact where PhiUSIIL contains 0 legitimate
free-hosting platform URLs (e.g., github.io, web.app, firebaseapp.com, etc.),
causing any URL on these domains to be trivially misclassified as phishing.

This module provides:
1. Configurable ingestion of external phishing (OpenPhish, PhishTank) and
   legitimate URLs (Tranco list / benign cloud-hosted domains).
2. Graceful fallback with transparent warnings when external files are absent.
3. Free-host slice dataset generation and slice-specific model training (using
   path/query features legitimately when balanced external data is provided).
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

import features

# Default configurable paths
DEFAULT_PHISHING_PATH = Path(os.environ.get("EXTERNAL_PHISHING_PATH", "external_phishing.csv"))
DEFAULT_LEGITIMATE_PATH = Path(os.environ.get("EXTERNAL_LEGITIMATE_PATH", "external_legitimate.csv"))


def load_external_dataset(
    phishing_path: Path | str = DEFAULT_PHISHING_PATH,
    legitimate_path: Path | str = DEFAULT_LEGITIMATE_PATH,
    free_hosts_only: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Loads and merges external phishing and legitimate URL datasets.
    
    If the files do not exist, prints a clear scientific disclaimer and returns None.
    """
    p_path = Path(phishing_path)
    l_path = Path(legitimate_path)

    if not p_path.exists() or not l_path.exists():
        missing = []
        if not p_path.exists():
            missing.append(f"Phishing CSV ({p_path})")
        if not l_path.exists():
            missing.append(f"Legitimate CSV ({l_path})")
        warnings.warn(
            f"\n[RESEARCH WARNING - FREE-HOST FALSE TRUST GAP]\n"
            f"Missing external validation files: {', '.join(missing)}.\n"
            f"PhiUSIIL contains 100% phishing labels for all free-hosting platforms (github.io, web.app, etc.)\n"
            f"and 0 legitimate free-host URLs. Therefore, free-host metrics on pure PhiUSIIL reflect dataset\n"
            f"collection bias rather than true model discernment. Provide balanced external data to benchmark\n"
            f"the free-host slice accurately.\n",
            stacklevel=2,
        )
        return None

    # Load external files
    try:
        phish_df = pd.read_csv(p_path)
        legit_df = pd.read_csv(l_path)
    except Exception as e:
        print(f"[!] Error reading external datasets: {e}")
        return None

    # Find URL column name
    def get_url_col(d: pd.DataFrame) -> str:
        for c in ("url", "URL", "Url", "target_url", "domain"):
            if c in d.columns:
                return c
        return d.columns[0]

    p_col = get_url_col(phish_df)
    l_col = get_url_col(legit_df)

    p_urls = phish_df[p_col].dropna().astype(str).unique()
    l_urls = legit_df[l_col].dropna().astype(str).unique()

    records = []
    for u in p_urls:
        records.append({"URL": u, "y": 1})
    for u in l_urls:
        records.append({"URL": u, "y": 0})

    df = pd.DataFrame(records).drop_duplicates(subset=["URL"]).reset_index(drop=True)

    if free_hosts_only:
        # Filter strictly for free host URLs
        is_free_mask = [features.parse_url_components(u)["is_free"] for u in df["URL"]]
        df = df[is_free_mask].reset_index(drop=True)
        print(f"[data_external] Filtered to {len(df):,} free-host URLs ({df['y'].value_counts().to_dict()})")

    return df


def evaluate_free_host_slice(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
) -> dict[str, float]:
    """
    Evaluates model performance separately on the free-host slice vs. standard domains.
    """
    if "is_free_host" not in X_test.columns:
        return {}

    free_mask = (X_test["is_free_host"] == 1).values
    std_mask = ~free_mask

    results = {}
    for subset_name, mask in [("free_host", free_mask), ("standard_domain", std_mask), ("overall", slice(None))]:
        n_samples = np.sum(mask) if isinstance(mask, np.ndarray) else len(X_test)
        if n_samples == 0:
            continue

        X_sub = X_test.iloc[mask][feature_names] if isinstance(mask, np.ndarray) else X_test[feature_names]
        y_sub = y_test.iloc[mask] if isinstance(mask, np.ndarray) else y_test

        if len(y_sub.unique()) < 2:
            # Single class present in subset (e.g. 100% phishing in PhiUSIIL free hosts)
            preds = model.predict(X_sub)
            acc = accuracy_score(y_sub, preds)
            results[f"{subset_name}_n_samples"] = int(n_samples)
            results[f"{subset_name}_accuracy"] = round(float(acc), 4)
            results[f"{subset_name}_phish_prevalence"] = round(float(y_sub.mean()), 4)
            continue

        preds = model.predict(X_sub)
        proba = model.predict_proba(X_sub)[:, 1] if hasattr(model, "predict_proba") else None

        results[f"{subset_name}_n_samples"] = int(n_samples)
        results[f"{subset_name}_accuracy"] = round(float(accuracy_score(y_sub, preds)), 4)
        results[f"{subset_name}_precision"] = round(float(precision_score(y_sub, preds, zero_division=0)), 4)
        results[f"{subset_name}_recall"] = round(float(recall_score(y_sub, preds, zero_division=0)), 4)
        results[f"{subset_name}_f1"] = round(float(f1_score(y_sub, preds, zero_division=0)), 4)
        if proba is not None:
            results[f"{subset_name}_roc_auc"] = round(float(roc_auc_score(y_sub, proba)), 4)

    return results
