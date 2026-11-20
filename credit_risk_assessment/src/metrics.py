"""Pure-math credit risk cost metrics, Bayes threshold, bootstrap CIs, and fairness slices."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import BAYES_OPTIMAL_THRESHOLD, COST_FN, COST_FP, RANDOM_STATE


def total_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> float:
    """Compute total German Credit misclassification cost: 5*FN + 1*FP."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(cost_fn * fn + cost_fp * fp)


def expected_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> float:
    """Compute mean misclassification cost per applicant: (5*FN + 1*FP) / N."""
    return total_cost(y_true, y_pred, cost_fn=cost_fn, cost_fp=cost_fp) / len(y_true)


def bayes_optimal_threshold(
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> float:
    """Theoretical Bayes decision threshold for calibrated probabilities: C_FP / (C_FP + C_FN)."""
    return float(cost_fp / (cost_fp + cost_fn))


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
    grid_size: int = 181,
) -> tuple[float, float]:
    """Grid-search probability threshold on validation/OOF predictions to minimize expected cost."""
    thresholds = np.linspace(0.05, 0.95, grid_size)
    best_t = BAYES_OPTIMAL_THRESHOLD
    best_cost = float("inf")
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    for t in thresholds:
        preds = (y_prob_arr >= t).astype(int)
        c = expected_cost(y_true_arr, preds, cost_fn=cost_fn, cost_fp=cost_fp)
        if c < best_cost - 1e-12 or (
            abs(c - best_cost) <= 1e-12
            and abs(t - BAYES_OPTIMAL_THRESHOLD) < abs(best_t - BAYES_OPTIMAL_THRESHOLD)
        ):
            best_cost = c
            best_t = float(t)
    return round(best_t, 4), round(best_cost, 4)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> dict:
    """Compute complete classification and credit-cost metrics for Bad=1, Good=0."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "threshold": round(float(threshold), 4),
        "expected_cost": round(expected_cost(y_true_arr, y_pred, cost_fn, cost_fp), 4),
        "total_cost": round(total_cost(y_true_arr, y_pred, cost_fn, cost_fp), 1),
        "roc_auc": round(float(roc_auc_score(y_true_arr, y_prob_arr)), 4),
        "pr_auc": round(float(average_precision_score(y_true_arr, y_prob_arr)), 4),
        "f2_bad": round(float(fbeta_score(y_true_arr, y_pred, beta=2.0, zero_division=0)), 4),
        "f1_bad": round(float(f1_score(y_true_arr, y_pred, zero_division=0)), 4),
        "recall_bad": round(float(recall_score(y_true_arr, y_pred, zero_division=0)), 4),
        "specificity_good": round(specificity, 4),
        "precision_bad": round(float(precision_score(y_true_arr, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true_arr, y_pred)), 4),
        "brier_score": round(float(brier_score_loss(y_true_arr, y_prob_arr)), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def naive_baselines(y_true: np.ndarray) -> pd.DataFrame:
    """Compute trivial all-Good and all-Bad baselines under the 5:1 cost matrix."""
    y_arr = np.asarray(y_true).astype(int)
    all_good = np.zeros_like(y_arr)
    all_bad = np.ones_like(y_arr)
    rows = []
    for name, pred in [("Naive_All_Good (0)", all_good), ("Naive_All_Bad (1)", all_bad)]:
        tn, fp, fn, tp = confusion_matrix(y_arr, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "model": name,
                "expected_cost": round(expected_cost(y_arr, pred), 4),
                "total_cost": round(total_cost(y_arr, pred), 1),
                "accuracy": round(float(accuracy_score(y_arr, pred)), 4),
                "recall_bad": round(float(recall_score(y_arr, pred, zero_division=0)), 4),
                "specificity_good": round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0, 4),
                "fn": int(fn),
                "fp": int(fp),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_boot: int = 1000,
    seed: int = RANDOM_STATE,
) -> dict:
    """Compute 95% percentile bootstrap confidence intervals for key evaluation metrics."""
    rng = np.random.default_rng(seed)
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    n = len(y_true_arr)

    costs, aucs, pr_aucs, recalls = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true_arr[idx]
        yp = y_prob_arr[idx]
        if len(np.unique(yt)) < 2:
            continue
        pred = (yp >= threshold).astype(int)
        costs.append(expected_cost(yt, pred))
        aucs.append(roc_auc_score(yt, yp))
        pr_aucs.append(average_precision_score(yt, yp))
        recalls.append(recall_score(yt, pred, zero_division=0))

    def _ci(vals: list[float]) -> tuple[float, float]:
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return round(float(lo), 4), round(float(hi), 4)

    return {
        "expected_cost_ci95": _ci(costs),
        "roc_auc_ci95": _ci(aucs),
        "pr_auc_ci95": _ci(pr_aucs),
        "recall_bad_ci95": _ci(recalls),
    }


def slice_error_analysis(
    df_raw: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Slice-based error and fairness evaluation across protected and financial groups."""
    work = df_raw.copy().reset_index(drop=True)
    work["y_true"] = np.asarray(y_true).astype(int)
    work["y_prob"] = np.asarray(y_prob).astype(float)
    work["y_pred"] = (work["y_prob"] >= threshold).astype(int)
    work["age_group"] = pd.cut(
        work["age"], bins=[18, 25, 35, 50, 100], labels=["19_25", "26_35", "36_50", "51_75"]
    ).astype(str)

    slice_cols = ["personal_status_sex", "foreign_worker", "age_group", "checking_status"]
    rows = []
    for col in slice_cols:
        for grp, sub in work.groupby(col, observed=False):
            if len(sub) == 0:
                continue
            yt = sub["y_true"].to_numpy()
            yp = sub["y_pred"].to_numpy()
            prob = sub["y_prob"].to_numpy()
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            auc = float(roc_auc_score(yt, prob)) if len(np.unique(yt)) == 2 else np.nan
            rows.append(
                {
                    "slice_feature": col,
                    "group": str(grp),
                    "n": int(len(sub)),
                    "actual_bad_rate": round(float(yt.mean()), 4),
                    "predicted_bad_rate": round(float(yp.mean()), 4),
                    "expected_cost": round(expected_cost(yt, yp), 4),
                    "recall_bad": round(rec, 4),
                    "false_positive_rate": round(fpr, 4),
                    "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
                }
            )
    return pd.DataFrame(rows)

