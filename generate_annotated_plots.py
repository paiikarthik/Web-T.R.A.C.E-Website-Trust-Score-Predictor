"""
Generate All Phishing Detection Figures with Comprehensive Metric Annotations.

This script loads the benchmark outputs and regenerates all figures in
outputs/figures/ ensuring every chart includes explicit, high-visibility
numerical metric annotations (bar values, percentages, convergence rates,
TP/TN counts, and statistical intervals).
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure Matplotlib cache directory is writable
if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(__file__).resolve().parent / "outputs" / ".mpl_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "outputs" / "reports"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
})


# ---------------------------------------------------------------------------
# 1. Model Comparison Bar Charts (with exact metric values on every bar)
# ---------------------------------------------------------------------------
def generate_model_comparison_plots():
    print("[1] Generating Model Comparison Bar Charts with Metric Labels...")
    for tag in ["unseen_domain", "random"]:
        csv_file = REPORT_DIR / f"metrics_{tag}.csv"
        if not csv_file.exists():
            continue
        df = pd.read_csv(csv_file)
        metrics_to_plot = [
            ("accuracy", "Accuracy"),
            ("precision_phishing", "Precision"),
            ("recall_phishing", "Recall (Sens.)"),
            ("f1_phishing", "F1-Score"),
            ("roc_auc", "ROC-AUC"),
            ("pr_auc", "PR-AUC"),
        ]
        
        plot_rows = []
        for _, row in df.iterrows():
            for col, label in metrics_to_plot:
                if col in row:
                    plot_rows.append({
                        "model": row["model"],
                        "paradigm": row.get("paradigm", "ML"),
                        "metric": label,
                        "score": row[col],
                    })
        plot_df = pd.DataFrame(plot_rows)

        fig, ax = plt.subplots(figsize=(15, 7))
        palette = sns.color_palette("tab10", n_colors=df["model"].nunique())
        sns.barplot(data=plot_df, x="metric", y="score", hue="model", ax=ax, palette=palette)
        
        ax.set_ylim(0, 1.18)
        ax.set_title(f"Model Performance Benchmark: 5 ML vs 5 DL Models (Split: {tag.replace('_', ' ').title()})\n[Values Annotated Above Each Bar]", fontweight="bold", pad=14)
        ax.set_ylabel("Metric Score (0.0 to 1.0)", fontweight="bold")
        ax.set_xlabel("Evaluation Metric", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Annotate every single bar with its numerical value
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=3, fontsize=6.8, rotation=90, fontweight="bold")

        ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
        fig.tight_layout()
        save_path = FIGURE_DIR / f"model_comparison_{tag}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"    Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# 2. Operational Ranking Metrics (PR-AUC, Spearman, Kendall, NDCG@k)
# ---------------------------------------------------------------------------
def generate_ranking_metrics_plots():
    print("[2] Generating Ranking Metrics Bar Charts with Metric Labels...")
    for tag in ["unseen_domain", "random"]:
        csv_file = REPORT_DIR / f"metrics_{tag}.csv"
        if not csv_file.exists():
            continue
        df = pd.read_csv(csv_file)
        ranking_cols = [
            ("pr_auc", "PR-AUC"),
            ("spearman_rho", "Spearman Rho"),
            ("kendall_tau", "Kendall Tau"),
            ("ndcg_10", "NDCG@10"),
            ("ndcg_100", "NDCG@100"),
            ("ndcg_1000", "NDCG@1000"),
        ]
        plot_rows = []
        for _, row in df.iterrows():
            for col, label in ranking_cols:
                if col in row:
                    plot_rows.append({
                        "model": row["model"],
                        "ranking_metric": label,
                        "score": row[col],
                    })
        plot_df = pd.DataFrame(plot_rows)

        fig, ax = plt.subplots(figsize=(15, 7))
        palette = sns.color_palette("tab10", n_colors=df["model"].nunique())
        sns.barplot(data=plot_df, x="ranking_metric", y="score", hue="model", ax=ax, palette=palette)

        ax.set_ylim(0, 1.20)
        ax.set_title(f"Operational Ranking Metrics: Prioritization Benchmark ({tag.replace('_', ' ').title()})\n[Scores Annotated on Every Bar]", fontweight="bold", pad=14)
        ax.set_ylabel("Metric Score", fontweight="bold")
        ax.set_xlabel("Ranking / Triage Prioritization Metric", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=6.8, rotation=90, fontweight="bold")

        ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
        fig.tight_layout()
        save_path = FIGURE_DIR / f"ranking_metrics_{tag}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"    Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# 3. TP & TN Comprehensive Audit (Counts and Percentages)
# ---------------------------------------------------------------------------
def generate_tp_tn_plots():
    print("[3] Generating True Positives (TP) & True Negatives (TN) Comparison...")
    for tag in ["unseen_domain", "random"]:
        csv_file = REPORT_DIR / f"tp_tn_evaluation_{tag}.csv"
        if not csv_file.exists():
            continue
        df = pd.read_csv(csv_file).sort_values("True_Positive_TP", ascending=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

        # Panel 1: Absolute Counts
        x = np.arange(len(df))
        width = 0.38
        bars_tp = ax1.bar(x - width/2, df["True_Positive_TP"], width, label="True Positives (Phish Caught)", color="#2b5c8f", edgecolor="black", alpha=0.9)
        bars_tn = ax1.bar(x + width/2, df["True_Negative_TN"], width, label="True Negatives (Legit Allowed)", color="#469b77", edgecolor="black", alpha=0.9)

        ax1.set_title(f"Absolute Correct Classifications: TP vs TN ({tag.replace('_', ' ').title()})\n[Exact URL Counts Labeled on Bars]", fontweight="bold")
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

        # Panel 2: Rates (%)
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
        save_path = FIGURE_DIR / f"tp_tn_comparison_{tag}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"    Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# 4. Deep Learning Loss Convergence Graphs (with Start/End Metrics)
# ---------------------------------------------------------------------------
def generate_loss_curves_plots():
    print("[4] Generating Loss Curves with Convergence Metrics...")
    for tag in ["unseen_domain", "random"]:
        csv_file = REPORT_DIR / f"dl_training_loss_{tag}.csv"
        if not csv_file.exists():
            continue
        df = pd.read_csv(csv_file)
        models = [c for c in df.columns if c != "epoch"]
        epochs = df["epoch"].values

        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        colors = sns.color_palette("deep", n_colors=len(models))

        for idx, model in enumerate(models):
            losses = df[model].dropna().values
            init_loss = losses[0]
            final_loss = losses[-1]
            pct_drop = ((final_loss - init_loss) / init_loss) * 100
            
            linestyle = "--" if "Attention" in model else "-"
            line = ax.plot(
                epochs[:len(losses)],
                losses,
                marker="o",
                markersize=5,
                linewidth=2.0,
                linestyle=linestyle,
                color=colors[idx],
                label=f"{model}: Start={init_loss:.4f} → End={final_loss:.4f} ({pct_drop:+.1f}%)"
            )

            # Annotate start and final loss points directly on the graph
            ax.annotate(
                f"{final_loss:.4f}",
                (epochs[len(losses)-1], final_loss),
                textcoords="offset points",
                xytext=(6, -2),
                fontsize=8.5,
                fontweight="bold",
                color=colors[idx]
            )

        ax.set_xlabel("Epoch Number (Batch Size = 128, Adam Optimizer)", fontweight="bold")
        ax.set_ylabel("Binary Cross-Entropy Loss (BCE)", fontweight="bold")
        ax.set_title(f"Deep Learning Training Loss Convergence ({tag.replace('_', ' ').title()})\n[Epoch 1 Start vs Epoch 8 End Loss Annotated]", fontweight="bold", pad=12)
        ax.legend(loc="upper right", frameon=True, fontsize=8.5)
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        save_path = FIGURE_DIR / f"loss_curves_{tag}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"    Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# 5. 5-Fold Stratified Cross-Validation F1-Score Stability
# ---------------------------------------------------------------------------
def generate_cv_plot():
    print("[5] Generating 5-Fold Cross-Validation Plot with Metrics...")
    csv_file = REPORT_DIR / "cv_five_fold.csv"
    if not csv_file.exists():
        return
    df = pd.read_csv(csv_file)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    bars = ax.bar(
        df["model"],
        df["f1_mean"],
        yerr=df["f1_std"],
        capsize=6,
        color="#2b5c8f",
        edgecolor="black",
        alpha=0.88,
        width=0.55
    )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 (Phishing) Mean ± Std Deviation", fontweight="bold")
    ax.set_xlabel("Cross-Validated Model", fontweight="bold")
    ax.set_title("5-Fold Stratified Cross-Validation F1-Score Stability\n[Exact Mean ± Std Annotated Above Bars]", fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, mean_val, std_val in zip(bars, df["f1_mean"], df["f1_std"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean_val + std_val + 0.025,
            f"{mean_val:.4f}\n±{std_val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#122d4a"
        )

    fig.tight_layout()
    save_path = FIGURE_DIR / "cv_f1_comparison.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# 6. SHAP Mean Absolute Feature Importance Bar Chart
# ---------------------------------------------------------------------------
def generate_shap_bar_plot():
    print("[6] Generating SHAP Feature Importance Bar Chart with Metrics...")
    csv_file = REPORT_DIR / "shap_mean_abs_random_forest.csv"
    if not csv_file.exists():
        return
    df = pd.read_csv(csv_file).head(15).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.barh(df["feature"], df["mean_abs_shap"], color="#3b72a8", edgecolor="black", alpha=0.9, height=0.65)

    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Model Phishing Prediction)", fontweight="bold")
    ax.set_ylabel("URL Syntactic & Structural Feature", fontweight="bold")
    ax.set_title("Top 15 Most Influential Features (Random Forest TreeSHAP)\n[Mean |SHAP| Annotated on Each Bar]", fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_xlim(0, df["mean_abs_shap"].max() * 1.22)

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.4f}",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color="#143454"
        )

    fig.tight_layout()
    save_path = FIGURE_DIR / "shap_bar_random_forest.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# 7. Re-annotate All Confusion Matrices (All 10 Models)
# ---------------------------------------------------------------------------
def generate_annotated_confusion_matrices():
    print("[7] Re-annotating All Confusion Matrices with Full Metrics...")
    for tag in ["unseen_domain", "random"]:
        cm_files = list(REPORT_DIR.glob(f"confusion_{tag}_*.csv"))
        for cm_file in cm_files:
            model_name = cm_file.stem.replace(f"confusion_{tag}_", "")
            cm_df = pd.read_csv(cm_file, index_col=0)
            cm = cm_df.values
            tn = int(cm[0, 0])
            fp = int(cm[0, 1])
            fn = int(cm[1, 0])
            tp = int(cm[1, 1])
            total = tn + fp + fn + tp
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
                f"{model_name} Confusion Matrix ({tag.replace('_', ' ').title()})\n"
                f"TP={tp:,} | TN={tn:,} | FP={fp:,} | FN={fn:,}\n"
                f"Accuracy={acc:.2%} | F1={f1:.2%} | Sensitivity={tpr:.2%} | Specificity={tnr:.2%}",
                fontsize=10.5,
                pad=12,
                fontweight="bold"
            )
            fig.tight_layout()
            save_path = FIGURE_DIR / f"confusion_{tag}_{model_name}.png"
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
    print("    Annotated all 20 confusion matrices successfully.")


# ---------------------------------------------------------------------------
# 8. Comparative Accuracy Grouped Bar Chart (Random vs Unseen-Domain)
# ---------------------------------------------------------------------------
def generate_comparative_accuracy_plots():
    print("[8] Generating Comparative Accuracy Grouped Bar Chart (All 10 Models)...")
    csv_path = REPORT_DIR / "final_model_comparison.csv"
    if not csv_path.exists():
        print(f"    Warning: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    model_name_map = {
        "ML_RandomForest": "Random Forest",
        "ML_XGBoost": "XGBoost",
        "DL_DeepMLP": "Deep MLP",
        "DL_LSTM": "Deep LSTM",
        "DL_SelfAttention": "Self-Attention",
        "DL_Hybrid_CNNLSTM": "Hybrid CNN-LSTM",
        "DL_CNN1D": "1D CNN",
        "ML_DecisionTree": "Decision Tree",
        "ML_LogisticRegression": "Logistic Regression",
        "ML_LinearSVM": "Linear SVM",
    }
    df["Clean_Model"] = df["Model"].map(lambda m: model_name_map.get(m, m))
    df = df.sort_values(by="UnseenDomain_Accuracy", ascending=False).reset_index(drop=True)

    models = df["Clean_Model"].tolist()
    rand_acc = (df["Random_Accuracy"] * 100).values
    unseen_acc = (df["UnseenDomain_Accuracy"] * 100).values

    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(15, 7.5))
    c_rand = "#2B5C8F"    # Deep Navy Slate
    c_unseen = "#2A9D8F"  # Vibrant Emerald Teal

    bars_rand = ax.bar(x - width/2, rand_acc, width, label="Random Stratified Accuracy",
                       color=c_rand, edgecolor="black", linewidth=0.8, alpha=0.92)
    bars_unseen = ax.bar(x + width/2, unseen_acc, width, label="Unseen-Domain Accuracy",
                         color=c_unseen, edgecolor="black", linewidth=0.8, alpha=0.92)

    for bar in bars_rand:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8.3, fontweight='bold', color="#163250")

    for bar in bars_unseen:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8.3, fontweight='bold', color="#114b43")

    ax.set_title("Comparative Model Accuracy Across Evaluation Regimes: Random Stratified vs. Unseen-Domain (10 Models)\n"
                 "[Zero-Scraping Lexical URL Phishing Detection Benchmark]",
                 fontsize=13, fontweight="bold", pad=16)
    ax.set_xlabel("Machine Learning & Deep Learning Algorithm", fontweight="bold", fontsize=11, labelpad=10)
    ax.set_ylabel("Classification Accuracy (%)", fontweight="bold", fontsize=11, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=10, fontweight="semibold")
    ax.set_ylim(0, 105)

    ax.axhline(90, linestyle=":", color="#e63946", alpha=0.7, linewidth=1.2, label="90% Benchmark Threshold")
    ax.axhline(80, linestyle="--", color="gray", alpha=0.4, linewidth=1.0)

    ax.legend(loc="upper right", frameon=True, framealpha=0.95, facecolor="white", edgecolor="#cccccc")
    ax.grid(axis="y", linestyle="--", alpha=0.45)

    fig.tight_layout()
    save_path = FIGURE_DIR / "comparative_accuracy_all_models.png"
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"    Saved: {save_path.name}")

    # Unified chart: Accuracy + TP/TN
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [1.1, 1.0]})
    bars_rand1 = ax1.bar(x - width/2, rand_acc, width, label="Random Stratified Accuracy",
                         color=c_rand, edgecolor="black", linewidth=0.8, alpha=0.92)
    bars_unseen1 = ax1.bar(x + width/2, unseen_acc, width, label="Unseen-Domain Accuracy",
                           color=c_unseen, edgecolor="black", linewidth=0.8, alpha=0.92)

    for bar in bars_rand1:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8.2, fontweight='bold', color="#163250")

    for bar in bars_unseen1:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8.2, fontweight='bold', color="#114b43")

    ax1.set_title("A. Comparative Model Accuracy: Random Stratified vs. Unseen-Domain Split (10 Algorithms)",
                  fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("Accuracy (%)", fontweight="bold", fontsize=10.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=20, ha="right", fontsize=9.5, fontweight="semibold")
    ax1.set_ylim(0, 105)
    ax1.axhline(90, linestyle=":", color="#e63946", alpha=0.7, linewidth=1.2, label="90% Benchmark Threshold")
    ax1.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.45)

    tp_tn_csv = REPORT_DIR / "tp_tn_evaluation_unseen_domain.csv"
    if tp_tn_csv.exists():
        tp_df = pd.read_csv(tp_tn_csv).set_index("Model")
        tp_vals = [tp_df.loc[m, "True_Positive_TP"] if m in tp_df.index else 0 for m in df["Model"]]
        tn_vals = [tp_df.loc[m, "True_Negative_TN"] if m in tp_df.index else 0 for m in df["Model"]]
        fp_vals = [tp_df.loc[m, "False_Positive_FP"] if m in tp_df.index else 0 for m in df["Model"]]
        fn_vals = [tp_df.loc[m, "False_Negative_FN"] if m in tp_df.index else 0 for m in df["Model"]]

        b_tp = ax2.bar(x - 1.5*0.2, tp_vals, 0.2, label="True Positives (Phish Caught)", color="#1F77B4", edgecolor="black", alpha=0.9)
        b_tn = ax2.bar(x - 0.5*0.2, tn_vals, 0.2, label="True Negatives (Legit Kept)", color="#2CA02C", edgecolor="black", alpha=0.9)
        b_fp = ax2.bar(x + 0.5*0.2, fp_vals, 0.2, label="False Positives (False Alarms)", color="#FF7F0E", edgecolor="black", alpha=0.9)
        b_fn = ax2.bar(x + 1.5*0.2, fn_vals, 0.2, label="False Negatives (Missed Phish)", color="#D62728", edgecolor="black", alpha=0.9)

        for bar in b_tp:
            h = bar.get_height()
            ax2.annotate(f"{int(h):,}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 2),
                         textcoords="offset points", ha='center', va='bottom', fontsize=7.2, rotation=90, fontweight="bold", color="#124068")
        for bar in b_tn:
            h = bar.get_height()
            ax2.annotate(f"{int(h):,}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 2),
                         textcoords="offset points", ha='center', va='bottom', fontsize=7.2, rotation=90, fontweight="bold", color="#135213")

        ax2.set_title("B. Confusion Matrix Classifications on Unseen Domains (TP, TN, FP, FN Counts)",
                      fontsize=12, fontweight="bold", pad=12)
        ax2.set_xlabel("Machine Learning & Deep Learning Algorithm", fontweight="bold", fontsize=10.5, labelpad=8)
        ax2.set_ylabel("Number of URLs", fontweight="bold", fontsize=10.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=20, ha="right", fontsize=9.5, fontweight="semibold")
        ax2.set_ylim(0, max(tn_vals) * 1.25)
        ax2.legend(loc="upper right", frameon=True, fontsize=9.5, ncol=2)
        ax2.grid(axis="y", linestyle="--", alpha=0.45)

    fig.tight_layout()
    save_path_uni = FIGURE_DIR / "comparative_accuracy_and_tp_tn_unified.png"
    fig.savefig(save_path_uni, dpi=160)
    plt.close(fig)
    print(f"    Saved: {save_path_uni.name}")


def main():
    print("=" * 70)
    print("REGENERATING ALL BENCHMARK FIGURES WITH METRIC ANNOTATIONS")
    print("=" * 70)
    generate_model_comparison_plots()
    generate_ranking_metrics_plots()
    generate_tp_tn_plots()
    generate_loss_curves_plots()
    generate_cv_plot()
    generate_shap_bar_plot()
    generate_annotated_confusion_matrices()
    generate_comparative_accuracy_plots()
    print("=" * 70)
    print("ALL FIGURES SUCCESSFULLY UPDATED WITH EMBEDDED METRICS!")
    print(f"Directory: {FIGURE_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
