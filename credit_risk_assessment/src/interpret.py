"""Model interpretability (Feature Importance + Decision Trees) and publication-ready charts."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.tree import plot_tree

from src.clean_data import get_feature_names
from src.config import BAYES_OPTIMAL_THRESHOLD, RANDOM_STATE
from src.metrics import expected_cost

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {
    "LogisticRegression": "#1f77b4",
    "SVM": "#ff7f0e",
    "GradientBoosting": "#2ca02c",
    "DecisionTree_Shallow": "#9467bd",
}


def compute_permutation_importance_raw(
    fitted_pipeline,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    n_repeats: int = 20,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compute model-agnostic permutation importance aggregated cleanly at the 20 raw feature level."""
    perm = permutation_importance(
        fitted_pipeline,
        X_eval,
        y_eval,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    df = pd.DataFrame(
        {
            "feature": list(X_eval.columns),
            "perm_roc_auc_drop": np.round(perm.importances_mean, 4),
            "perm_std": np.round(perm.importances_std, 4),
        }
    )
    return df.sort_values("perm_roc_auc_drop", ascending=False).reset_index(drop=True)


def plot_eda_overview(df_raw: pd.DataFrame, audit: dict, save_path: Path | None = None):
    """Plot 2x2 EDA overview: target split, checking/history bad rates, skewed amount, and top IV."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # 1. Target imbalance & 5:1 cost weight illustration
    ax = axes[0, 0]
    counts = df_raw["bad_risk"].value_counts().sort_index()
    bars = ax.bar(
        ["Good Credit (0)", "Bad Credit (1)"],
        [counts[0], counts[1]],
        color=["#2a9d8f", "#e76f51"],
        width=0.55,
    )
    for bar, val in zip(bars, [counts[0], counts[1]]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 15,
            f"{val} ({val / len(df_raw) * 100:.0f}%)",
            ha="center",
            fontweight="bold",
        )
    ax.set_title("Target Distribution (Cost Matrix: FN=5 vs FP=1)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Applicant Count")
    ax.set_ylim(0, 820)

    # 2. Bad rate by Checking Account (highlights A14 implicit 'no checking' anomaly)
    ax = axes[0, 1]
    chk_map = {"A11": "<0 DM", "A12": "0-200 DM", "A13": ">=200 DM", "A14": "No checking (A14)"}
    chk_df = (
        df_raw.groupby("checking_status")["bad_risk"]
        .agg(["count", "mean"])
        .reindex(["A11", "A12", "A13", "A14"])
    )
    labels = [chk_map[k] for k in chk_df.index]
    colors = ["#e76f51", "#f4a261", "#e9c46a", "#264653"]
    bars = ax.bar(labels, chk_df["mean"] * 100, color=colors, width=0.58)
    for bar, (_, r) in zip(bars, chk_df.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            r["mean"] * 100 + 1.5,
            f"{r['mean'] * 100:.1f}%\n(n={int(r['count'])})",
            ha="center",
            fontsize=9,
        )
    ax.axhline(30.0, color="gray", linestyle="--", linewidth=1, label="Dataset avg (30%)")
    ax.set_title("Bad Credit Rate by Checking Account Status", fontsize=12, fontweight="bold")
    ax.set_ylabel("Bad Rate (%)")
    ax.set_ylim(0, 62)
    ax.legend(loc="upper right")

    # 3. Skewed Credit Amount distribution by class
    ax = axes[1, 0]
    sns.histplot(
        df_raw.loc[df_raw["bad_risk"] == 0, "credit_amount"],
        bins=28,
        kde=True,
        color="#2a9d8f",
        alpha=0.45,
        label="Good (0)",
        ax=ax,
    )
    sns.histplot(
        df_raw.loc[df_raw["bad_risk"] == 1, "credit_amount"],
        bins=28,
        kde=True,
        color="#e76f51",
        alpha=0.55,
        label="Bad (1)",
        ax=ax,
    )
    ax.axvline(
        df_raw["credit_amount"].quantile(0.99),
        color="#264653",
        linestyle="--",
        linewidth=1.5,
        label="p99 clip (14,179 DM)",
    )
    ax.set_title("Credit Amount Distribution (Right-Skew = 1.95)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Credit Amount (DM)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")

    # 4. Top 12 Information Value (IV) ranking
    ax = axes[1, 1]
    top_iv = audit["feature_strength"].head(12).iloc[::-1]
    bar_colors = ["#264653" if v >= 0.1 else "#8d99ae" for v in top_iv["iv"]]
    ax.barh(top_iv["feature"], top_iv["iv"], color=bar_colors)
    ax.axvline(0.1, color="#e76f51", linestyle="--", linewidth=1, label="Medium/Strong IV >= 0.10")
    ax.axvline(0.02, color="#f4a261", linestyle=":", linewidth=1, label="Weak threshold (0.02)")
    ax.set_title("Top 12 Predictors by Information Value (IV)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Information Value (Smoothed)")
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout(pad=2.0)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparison(results_dict: dict, y_test: np.ndarray, save_path: Path | None = None):
    """Plot 2x2 benchmark: ROC curves, PR curves, Expected Cost vs threshold, and Calibration."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # 1. ROC Curves
    ax = axes[0, 0]
    for name, res in results_dict.items():
        fpr, tpr, _ = roc_curve(y_test, res["test_probs"])
        ax.plot(
            fpr,
            tpr,
            linewidth=2.0,
            label=f"{name} (AUC={res['test_metrics']['roc_auc']:.3f})",
            color=PALETTE.get(name, "#333333"),
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (0.500)")
    ax.set_title("ROC Curves on Untouched Test Set (n=200)", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall Bad)")
    ax.legend(loc="lower right", fontsize=9)

    # 2. Precision-Recall Curves
    ax = axes[0, 1]
    for name, res in results_dict.items():
        prec, rec, _ = precision_recall_curve(y_test, res["test_probs"])
        ax.plot(
            rec,
            prec,
            linewidth=2.0,
            label=f"{name} (AP={res['test_metrics']['pr_auc']:.3f})",
            color=PALETTE.get(name, "#333333"),
        )
    ax.axhline(y_test.mean(), color="gray", linestyle="--", linewidth=1, label=f"No-skill ({y_test.mean():.2f})")
    ax.set_title("Precision-Recall Curves (Positive = Bad Risk)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recall (Bad Risk)")
    ax.set_ylabel("Precision (Bad Risk)")
    ax.legend(loc="lower left", fontsize=9)

    # 3. Expected Cost across decision thresholds
    ax = axes[1, 0]
    t_grid = np.linspace(0.05, 0.90, 86)
    for name, res in results_dict.items():
        costs = [expected_cost(y_test, (res["test_probs"] >= t).astype(int)) for t in t_grid]
        ax.plot(t_grid, costs, label=f"{name} (t_oof={res['threshold']:.2f})", color=PALETTE.get(name))
        ax.scatter(
            [res["threshold"]],
            [res["test_metrics"]["expected_cost"]],
            color=PALETTE.get(name),
            s=45,
            zorder=5,
        )
    ax.axhline(0.70, color="#6c757d", linestyle="--", linewidth=1, label="Naive All-Bad (cost=0.70)")
    ax.axhline(1.50, color="#adb5bd", linestyle=":", linewidth=1, label="Naive All-Good (cost=1.50)")
    ax.axvline(
        BAYES_OPTIMAL_THRESHOLD,
        color="#d62828",
        linestyle="-.",
        linewidth=1,
        label=f"Bayes t*={BAYES_OPTIMAL_THRESHOLD:.3f} (unweighted)",
    )
    ax.set_title("Expected Cost (5*FN + 1*FP)/N vs Decision Threshold", fontsize=12, fontweight="bold")
    ax.set_xlabel("Probability Threshold t")
    ax.set_ylabel("Mean Cost per Applicant (Lower is Better)")
    ax.set_ylim(0.30, 1.65)
    ax.legend(loc="upper left", fontsize=8)

    # 4. Calibration Curves
    ax = axes[1, 1]
    for name, res in results_dict.items():
        prob_true, prob_pred = calibration_curve(y_test, res["test_probs"], n_bins=7, strategy="quantile")
        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            label=f"{name} (Brier={res['test_metrics']['brier_score']:.3f})",
            color=PALETTE.get(name),
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect Calibration")
    ax.set_title("Probability Calibration (Cost-Weighted Probabilities)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Fraction of Bad Risk")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout(pad=2.0)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrices(results_dict: dict, y_test: np.ndarray, save_path: Path | None = None):
    """Plot side-by-side confusion matrices annotated with total 5:1 cost."""
    n = len(results_dict)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.3))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results_dict.items()):
        preds = (res["test_probs"] >= res["threshold"]).astype(int)
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Good (0)", "Bad (1)"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        m = res["test_metrics"]
        ax.set_title(
            f"{name}\nCost={m['expected_cost']:.3f} (FN={m['fn']}, FP={m['fp']})",
            fontsize=10.5,
            fontweight="bold",
        )
    fig.tight_layout(pad=1.5)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(
    lr_odds_df: pd.DataFrame,
    perm_imp_by_model: dict[str, pd.DataFrame],
    gbdt_gain_df: pd.DataFrame,
    save_path: Path | None = None,
):
    """Plot 1x3 interpretability chart: LR Odds Ratios, Raw Permutation Importance, and GBDT Gain."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # 1. Logistic Regression top log-odds (signed direction!)
    ax = axes[0]
    top_lr = lr_odds_df.head(14).iloc[::-1]
    colors = ["#e76f51" if c >= 0 else "#2a9d8f" for c in top_lr["coef_log_odds"]]
    ax.barh(top_lr["feature"], top_lr["coef_log_odds"], color=colors)
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_title("Logistic Regression: Signed Log-Odds\n(Red = Higher Risk, Teal = Lower Risk)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Standardized Coefficient (beta)")

    # 2. Grouped Permutation Importance across LR, SVM, GBDT on raw 20 attributes
    ax = axes[1]
    merged = None
    for m_name, df_p in perm_imp_by_model.items():
        sub = df_p[["feature", "perm_roc_auc_drop"]].rename(columns={"perm_roc_auc_drop": m_name})
        merged = sub if merged is None else merged.merge(sub, on="feature", how="outer")
    merged["mean_drop"] = merged[list(perm_imp_by_model.keys())].mean(axis=1)
    top_perm = merged.sort_values("mean_drop", ascending=False).head(12).iloc[::-1]

    y_pos = np.arange(len(top_perm))
    width = 0.25
    model_names = list(perm_imp_by_model.keys())
    for i, m_name in enumerate(model_names):
        ax.barh(
            y_pos + (i - 1) * width,
            top_perm[m_name],
            height=width,
            label=m_name,
            color=PALETTE.get(m_name, "#555555"),
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_perm["feature"])
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title("Model-Agnostic Permutation Importance\n(ROC-AUC Drop on Held-Out Test Set)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean ROC-AUC Drop")
    ax.legend(loc="lower right", fontsize=9)

    # 3. Gradient Boosting Gain Importance on clean binned/flagged features
    ax = axes[2]
    top_gb = gbdt_gain_df.head(14).iloc[::-1]
    ax.barh(top_gb["feature"], top_gb["gain_importance"], color="#2ca02c")
    ax.set_title("Gradient Boosting: Top 14 Features\n(Impurity/Gain Importance)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Relative Gain Importance")

    fig.tight_layout(pad=2.0)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_decision_trees(
    standalone_dt_pipeline,
    surrogate_info: dict,
    best_model_name: str,
    save_path: Path | None = None,
):
    """Visualize both Standalone shallow Decision Tree and Global Surrogate Tree."""
    fig, axes = plt.subplots(2, 1, figsize=(20, 14))

    dt_names = get_feature_names(standalone_dt_pipeline.named_steps["preprocess"])
    dt_clf = standalone_dt_pipeline.named_steps["model"]
    plot_tree(
        dt_clf,
        feature_names=dt_names,
        class_names=["Good", "Bad"],
        filled=True,
        rounded=True,
        fontsize=9,
        max_depth=3,
        ax=axes[0],
    )
    axes[0].set_title(
        "Standalone Shallow Decision Tree (Entropy / Cost-Weighted 5:1, Top 3 Levels Shown)",
        fontsize=13,
        fontweight="bold",
    )

    plot_tree(
        surrogate_info["surrogate_tree"],
        feature_names=surrogate_info["feature_names"],
        class_names=["Good", "Bad"],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=axes[1],
    )
    axes[1].set_title(
        f"Global Surrogate Decision Tree Explaining Best Model ({best_model_name}) — Fidelity = {surrogate_info['fidelity_agreement'] * 100:.1f}%",
        fontsize=13,
        fontweight="bold",
    )

    fig.tight_layout(pad=2.0)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
