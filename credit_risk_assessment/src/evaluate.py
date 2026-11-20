import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split

from src.config import BAYES_OPTIMAL_THRESHOLD, COLUMN_NAMES, N_SPLITS, RANDOM_STATE, TEST_SIZE
from src.imbalance import build_smotenc_pipeline
from src.metrics import (
    bootstrap_metric_ci,
    compute_classification_metrics,
    find_optimal_threshold,
    naive_baselines,
)
from src.models.decision_tree import build_decision_tree_pipeline
from src.models.gradient_boosting import (
    build_gbdt_pipeline,
    build_gradient_boosting,
    get_gbdt_param_grid,
)
from src.models.logistic_regression import (
    build_logistic_pipeline,
    build_logistic_regression,
    get_logistic_param_grid,
)
from src.models.svm_model import build_svm, build_svm_pipeline, get_svm_param_grid


def split_train_test(df_raw: pd.DataFrame):
    feature_cols = [c for c in COLUMN_NAMES if c != "raw_target"]
    X = df_raw[feature_cols].copy()
    y = df_raw["bad_risk"].to_numpy().astype(int)
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def tune_and_evaluate_model(
    name: str,
    base_pipeline,
    param_grid: dict | list[dict] | None,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    fixed_threshold: float | None = None,
    n_splits: int = N_SPLITS,
) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    t0 = time.perf_counter()

    if param_grid:
        search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="roc_auc",
            refit=True,
            n_jobs=1,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        cv_roc_auc = round(float(search.best_score_), 4)
    else:
        best_model = base_pipeline
        best_model.fit(X_train, y_train)
        best_params = {}
        cv_roc_auc = None

    fit_seconds = round(time.perf_counter() - t0, 3)

    # Leakage-free out-of-fold probabilities on train_dev solely for threshold calibration
    oof_probs = cross_val_predict(
        best_model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    if fixed_threshold is not None:
        chosen_t = float(fixed_threshold)
    else:
        chosen_t, _ = find_optimal_threshold(y_train, oof_probs)

    oof_metrics = compute_classification_metrics(y_train, oof_probs, chosen_t)

    # Single untouched test-set evaluation
    t_inf = time.perf_counter()
    test_probs = best_model.predict_proba(X_test)[:, 1]
    infer_ms = round((time.perf_counter() - t_inf) * 1000.0, 2)

    test_metrics = compute_classification_metrics(y_test, test_probs, chosen_t)
    test_ci95 = bootstrap_metric_ci(y_test, test_probs, chosen_t, n_boot=1000, seed=RANDOM_STATE)

    return {
        "name": name,
        "best_estimator": best_model,
        "best_params": best_params,
        "cv_best_roc_auc": cv_roc_auc if cv_roc_auc is not None else oof_metrics["roc_auc"],
        "threshold": chosen_t,
        "oof_probs": oof_probs,
        "oof_metrics": oof_metrics,
        "test_probs": test_probs,
        "test_metrics": test_metrics,
        "test_ci95": test_ci95,
        "fit_seconds": fit_seconds,
        "infer_ms_200rows": infer_ms,
    }


def run_primary_model_comparison(df_raw: pd.DataFrame) -> dict:
    X_train, X_test, y_train, y_test = split_train_test(df_raw)

    models_spec = [
        (
            "LogisticRegression",
            build_logistic_pipeline(feature_mode="binned", class_weight_mode="cost_5_1"),
            get_logistic_param_grid(),
        ),
        (
            "SVM",
            build_svm_pipeline(feature_mode="binned", class_weight_mode="cost_5_1"),
            get_svm_param_grid(),
        ),
        (
            "GradientBoosting",
            build_gbdt_pipeline(feature_mode="binned", class_weight_mode="cost_5_1"),
            get_gbdt_param_grid(),
        ),
        (
            "DecisionTree_Shallow",
            build_decision_tree_pipeline(
                feature_mode="binned", class_weight_mode="cost_5_1", criterion="entropy", max_depth=4
            ),
            {"model__max_depth": [3, 4], "model__min_samples_leaf": [15, 25]},
        ),
    ]

    results = {}
    summary_rows = []
    for name, pipe, grid in models_spec:
        res = tune_and_evaluate_model(
            name=name,
            base_pipeline=pipe,
            param_grid=grid,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        results[name] = res
        row = {
            "model": name,
            "oof_expected_cost": res["oof_metrics"]["expected_cost"],
            "oof_roc_auc": res["oof_metrics"]["roc_auc"],
            "oof_pr_auc": res["oof_metrics"]["pr_auc"],
            "test_expected_cost": res["test_metrics"]["expected_cost"],
            "test_total_cost": res["test_metrics"]["total_cost"],
            "test_roc_auc": res["test_metrics"]["roc_auc"],
            "test_pr_auc": res["test_metrics"]["pr_auc"],
            "test_f2_bad": res["test_metrics"]["f2_bad"],
            "test_recall_bad": res["test_metrics"]["recall_bad"],
            "test_specificity_good": res["test_metrics"]["specificity_good"],
            "test_precision_bad": res["test_metrics"]["precision_bad"],
            "test_accuracy": res["test_metrics"]["accuracy"],
            "test_brier": res["test_metrics"]["brier_score"],
            "threshold": res["threshold"],
            "fit_sec": res["fit_seconds"],
            "cost_ci95_lo": res["test_ci95"]["expected_cost_ci95"][0],
            "cost_ci95_hi": res["test_ci95"]["expected_cost_ci95"][1],
        }
        summary_rows.append(row)

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["oof_expected_cost", "oof_roc_auc"], ascending=[True, False])
        .reset_index(drop=True)
    )
    # Pick best among the 3 primary requested classifiers by OOF Expected Cost (no test peeking!)
    primary_only = summary_df[summary_df["model"].isin(["LogisticRegression", "SVM", "GradientBoosting"])]
    best_model_name = str(primary_only.iloc[0]["model"])

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "baselines_test": naive_baselines(y_test),
        "results": results,
        "summary_df": summary_df,
        "best_model_name": best_model_name,
    }


def run_imbalance_ablation(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> pd.DataFrame:
    configs = [
        (
            "LR | ClassWeight_5_1 + OOF_t",
            build_logistic_pipeline("binned", "cost_5_1", C=0.25, l1_ratio=0.0),
            None,
        ),
        (
            "LR | ClassWeight_Balanced + OOF_t",
            build_logistic_pipeline("binned", "balanced", C=0.25, l1_ratio=0.0),
            None,
        ),
        (
            "LR | Unweighted + Bayes_t (0.1667)",
            build_logistic_pipeline("binned", "none", C=0.25, l1_ratio=0.0),
            BAYES_OPTIMAL_THRESHOLD,
        ),
        (
            "LR | SMOTE_NC (in-fold) + OOF_t",
            build_smotenc_pipeline(
                build_logistic_regression(C=0.25, l1_ratio=0.0, class_weight_mode="none"),
                mode="binned",
            ),
            None,
        ),
        (
            "GBDT | ClassWeight_5_1 + OOF_t",
            build_gbdt_pipeline("binned", "cost_5_1", learning_rate=0.05, n_estimators=100, max_depth=2),
            None,
        ),
        (
            "GBDT | ClassWeight_Balanced + OOF_t",
            build_gbdt_pipeline("binned", "balanced", learning_rate=0.05, n_estimators=100, max_depth=2),
            None,
        ),
        (
            "GBDT | Unweighted + Bayes_t (0.1667)",
            build_gbdt_pipeline("binned", "none", learning_rate=0.05, n_estimators=100, max_depth=2),
            BAYES_OPTIMAL_THRESHOLD,
        ),
        (
            "GBDT | SMOTE_NC (in-fold) + OOF_t",
            build_smotenc_pipeline(
                build_gradient_boosting(
                    learning_rate=0.05, n_estimators=100, max_depth=2, class_weight_mode="none"
                ),
                mode="binned",
            ),
            None,
        ),
    ]

    rows = []
    for label, pipe, fixed_t in configs:
        res = tune_and_evaluate_model(
            name=label,
            base_pipeline=pipe,
            param_grid=None,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            fixed_threshold=fixed_t,
        )
        rows.append(
            {
                "strategy": label,
                "threshold": res["threshold"],
                "oof_expected_cost": res["oof_metrics"]["expected_cost"],
                "oof_roc_auc": res["oof_metrics"]["roc_auc"],
                "test_expected_cost": res["test_metrics"]["expected_cost"],
                "test_roc_auc": res["test_metrics"]["roc_auc"],
                "test_pr_auc": res["test_metrics"]["pr_auc"],
                "test_f2_bad": res["test_metrics"]["f2_bad"],
                "test_recall_bad": res["test_metrics"]["recall_bad"],
                "test_specificity_good": res["test_metrics"]["specificity_good"],
                "test_brier": res["test_metrics"]["brier_score"],
            }
        )
    return pd.DataFrame(rows)


def run_preprocessing_ablation(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for mode in ["binned", "continuous"]:
        specs = [
            ("LogisticRegression", build_logistic_pipeline(mode, "cost_5_1", C=0.25, l1_ratio=0.0)),
            ("SVM", build_svm_pipeline(mode, "cost_5_1", kernel="rbf", C=1.0, gamma="scale")),
            (
                "GradientBoosting",
                build_gbdt_pipeline(mode, "cost_5_1", learning_rate=0.05, n_estimators=100, max_depth=2),
            ),
        ]
        for m_name, pipe in specs:
            res = tune_and_evaluate_model(
                name=f"{m_name}_{mode}",
                base_pipeline=pipe,
                param_grid=None,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
            rows.append(
                {
                    "feature_mode": mode,
                    "model": m_name,
                    "oof_expected_cost": res["oof_metrics"]["expected_cost"],
                    "oof_roc_auc": res["oof_metrics"]["roc_auc"],
                    "test_expected_cost": res["test_metrics"]["expected_cost"],
                    "test_roc_auc": res["test_metrics"]["roc_auc"],
                    "test_pr_auc": res["test_metrics"]["pr_auc"],
                    "test_recall_bad": res["test_metrics"]["recall_bad"],
                }
            )
    return pd.DataFrame(rows)
