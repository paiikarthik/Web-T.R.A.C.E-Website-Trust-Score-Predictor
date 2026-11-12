"""
Display All Phishing Detection Benchmark Figures with Metric Annotations.

Run this script directly in Spyder (%runfile show_plots.py) or terminal
to view all generated loss graphs, PR curves, ROC curves, ranking metrics,
TP/TN comparisons, and SHAP explanations with full numerical annotations
without re-training the models.
"""

import os
import sys
from pathlib import Path

# Ensure Matplotlib cache directory is writable before importing matplotlib
if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(__file__).resolve().parent / "outputs" / ".mpl_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASE_DIR = Path(__file__).resolve().parent
FIGURE_DIR = BASE_DIR / "outputs" / "figures"

# Key figures to display in order of presentation (all annotated with metrics)
FIGURE_ORDER = [
    ("model_comparison_unseen_domain.png", "1. Performance Metric Benchmark (Unseen Domain) [Values Labeled on Bars]"),
    ("ranking_metrics_unseen_domain.png", "2. Operational Ranking Metrics (PR AUC, Spearman, Kendall, NDCG@k) [Labeled]"),
    ("tp_tn_comparison_unseen_domain.png", "3. True Positives (TP) & True Negatives (TN) [Counts & Rates Labeled]"),
    ("loss_curves_unseen_domain.png", "4. Deep Learning Loss Convergence Graph [Start & End BCE Losses Labeled]"),
    ("cv_f1_comparison.png", "5. 5-Fold Cross-Validation F1-Score Stability [Mean ± Std Labeled]"),
    ("pr_curves_unseen_domain.png", "6. Precision-Recall (PR) Curves (5 ML vs 5 DL Models) [PR-AUC in Legend]"),
    ("roc_curves_unseen_domain.png", "7. ROC Curves (5 ML vs 5 DL Models) [AUC in Legend]"),
    ("shap_bar_random_forest.png", "8. TreeSHAP Top-15 Feature Importance [Mean |SHAP| Values Labeled]"),
    ("shap_summary_random_forest.png", "9. TreeSHAP Beeswarm Distribution Summary Plot"),
    ("confusion_unseen_domain_DL_LSTM.png", "10. Top DL Model (DL_LSTM) Full Metric Confusion Matrix"),
    ("confusion_unseen_domain_ML_RandomForest.png", "11. Top ML Model (ML_RandomForest) Full Metric Confusion Matrix"),
    ("confusion_unseen_domain_ML_XGBoost.png", "12. Top Gradient Booster (ML_XGBoost) Full Metric Confusion Matrix"),
]

def main():
    if not FIGURE_DIR.exists():
        print(f"Figures directory not found: {FIGURE_DIR}")
        return

    print("=" * 75)
    print("DISPLAYING PHISHING DETECTION BENCHMARK FIGURES (ALL METRICS INCLUDED)")
    print("=" * 75)
    print(f"Loading figures from: {FIGURE_DIR}\n")

    displayed = 0
    for filename, title in FIGURE_ORDER:
        filepath = FIGURE_DIR / filename
        if filepath.exists():
            print(f"[*] Showing: {title} ({filename})")
            img = mpimg.imread(str(filepath))
            fig, ax = plt.subplots(figsize=(10.5, 6.5))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
            plt.tight_layout()
            plt.show()
            displayed += 1
        else:
            print(f"[-] Missing: {filename}")

    print(f"\n[Done] Successfully displayed {displayed} metric-annotated figures in Spyder / interactive windows.")
    print(f"All full-resolution PNG files remain saved in:\n  {FIGURE_DIR}")

if __name__ == "__main__":
    main()
