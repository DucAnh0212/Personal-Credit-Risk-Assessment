"""Interpretability Models: Standalone shallow Decision Tree and Global Surrogate Tree."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from src.clean_data import build_preprocessor, get_feature_names
from src.config import RANDOM_STATE
from src.imbalance import resolve_class_weight


def build_decision_tree(
    criterion: str = "entropy",
    max_depth: int = 4,
    min_samples_leaf: int = 20,
    class_weight_mode: str = "cost_5_1",
    random_state: int = RANDOM_STATE,
) -> DecisionTreeClassifier:
    """Create shallow interpretable Decision Tree (entropy aligns with Information Gain slides)."""
    return DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=resolve_class_weight(class_weight_mode),
        random_state=random_state,
    )


def build_decision_tree_pipeline(
    feature_mode: str = "binned",
    class_weight_mode: str = "cost_5_1",
    criterion: str = "entropy",
    max_depth: int = 4,
    min_samples_leaf: int = 20,
) -> Pipeline:
    """Combine unscaled readable preprocessor with shallow Decision Tree."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(mode=feature_mode, scale_numeric=False)),
            (   
                "model",
                build_decision_tree(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight_mode=class_weight_mode,
                ),
            ),
        ]
    )


def summarize_tree_leaves(
    tree_clf: DecisionTreeClassifier,
    X_encoded: np.ndarray,
    y_actual: np.ndarray,
) -> pd.DataFrame:
    """Summarize each decision tree leaf with sample count, coverage, predicted label, and Bad rate."""
    leaf_ids = tree_clf.apply(X_encoded)
    preds = tree_clf.predict(X_encoded)
    n_total = len(y_actual)
    rows = []
    for lid in np.unique(leaf_ids):
        mask = leaf_ids == lid
        cnt = int(mask.sum())
        rows.append(
            {
                "leaf_id": int(lid),
                "n_samples": cnt,
                "coverage_pct": round(100.0 * cnt / n_total, 2),
                "predicted_class": "Bad (1)" if int(preds[mask][0]) == 1 else "Good (0)",
                "actual_bad_rate": round(float(np.mean(y_actual[mask])), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("actual_bad_rate", ascending=False).reset_index(drop=True)


def extract_standalone_tree_summary(
    dt_pipeline: Pipeline,
    X_raw: pd.DataFrame,
    y_actual: np.ndarray,
) -> dict:
    """Extract IF-THEN text rules and leaf coverage/bad-rate summary from standalone Decision Tree."""
    prep = dt_pipeline.named_steps["preprocess"]
    dt_clf = dt_pipeline.named_steps["model"]
    X_enc = prep.transform(X_raw)
    feature_names = get_feature_names(prep)
    rules = export_text(dt_clf, feature_names=feature_names, decimals=2)
    leaves = summarize_tree_leaves(dt_clf, X_enc, y_actual)
    return {
        "feature_names": feature_names,
        "rules_text": rules,
        "leaf_summary": leaves,
    }


def fit_global_surrogate_tree(
    teacher_pipeline: Pipeline,
    X_raw: pd.DataFrame,
    y_actual: np.ndarray,
    teacher_threshold: float,
    max_depth: int = 4,
    min_samples_leaf: int = 20,
    ccp_alpha: float = 0.005,
) -> dict:
    """Fit a shallow pruned Surrogate Tree imitating the best model's risk decisions.

    Uses an unscaled readable binned representation so split thresholds read directly as
    business rules (e.g., `checking_status_no_checking <= 0.50`).
    """
    teacher_probs = teacher_pipeline.predict_proba(X_raw)[:, 1]
    teacher_preds = (teacher_probs >= teacher_threshold).astype(int)

    prep = build_preprocessor(mode="binned", scale_numeric=False)
    X_enc = prep.fit_transform(X_raw)
    feature_names = get_feature_names(prep)

    surrogate = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        ccp_alpha=ccp_alpha,
        random_state=RANDOM_STATE,
    )
    surrogate.fit(X_enc, teacher_preds)
    surr_preds = surrogate.predict(X_enc)
    fidelity = round(float(accuracy_score(teacher_preds, surr_preds)), 4)

    rules = export_text(surrogate, feature_names=feature_names, decimals=2)
    leaves = summarize_tree_leaves(surrogate, X_enc, y_actual)

    return {
        "surrogate_tree": surrogate,
        "feature_names": feature_names,
        "fidelity_agreement": fidelity,
        "rules_text": rules,
        "leaf_summary": leaves,
    }
