import numpy as np
import pandas as pd

from src.clean_data import (
    PercentileClipLogTransformer,
    QuantileFeatureBinner,
    build_preprocessor,
    clean_static_rules,
    get_feature_names,
)
from src.config import BAYES_OPTIMAL_THRESHOLD
from src.data_audit import audit_data_quality, load_raw_german_data, load_raw_german_numeric
from src.evaluate import split_train_test
from src.imbalance import build_smotenc_pipeline
from src.metrics import bayes_optimal_threshold, expected_cost, find_optimal_threshold
from src.models.decision_tree import build_decision_tree_pipeline, fit_global_surrogate_tree
from src.models.gradient_boosting import build_gbdt_pipeline, extract_gbdt_gain_importance
from src.models.logistic_regression import (
    build_logistic_pipeline,
    build_logistic_regression,
    extract_odds_ratios,
)
from src.models.svm_model import build_svm_pipeline


def test_raw_data_audit_integrity():
    df_raw = load_raw_german_data()
    df_num = load_raw_german_numeric()
    audit = audit_data_quality(df_raw, df_num)

    assert audit["n_rows"] == 1000
    assert audit["n_features"] == 20
    assert audit["explicit_missing"] == 0
    assert audit["duplicate_full_rows"] == 0
    assert audit["target_counts"] == {"good_0": 700, "bad_1": 300, "bad_rate": 0.3}
    assert audit["dead_doc_codes"] == {"purpose": ["A47"], "personal_status_sex": ["A95"]}
    assert audit["contradictions"]["age_lt_25_with_employment_ge_7yr"] == 9
    assert audit["contradictions"]["unemployed_A71_with_high_skill_job_A174"] == 33
    assert audit["numeric_file_diff"]["target_mismatches"] == 0
    assert audit["numeric_file_diff"]["credit_amount_round100_exact_matches"] == 1000


def test_pure_math_cost_and_bayes_threshold():
    # 700 Good (0), 300 Bad (1)
    y_true = np.array([0] * 700 + [1] * 300)
    assert np.isclose(expected_cost(y_true, np.zeros_like(y_true)), 1.50)
    assert np.isclose(expected_cost(y_true, np.ones_like(y_true)), 0.70)
    assert np.isclose(bayes_optimal_threshold(), 1.0 / 6.0)
    assert np.isclose(BAYES_OPTIMAL_THRESHOLD, 1.0 / 6.0)

    probs = np.where(y_true == 1, 0.85, 0.10)
    t_opt, best_c = find_optimal_threshold(y_true, probs)
    assert best_c == 0.0
    assert 0.10 < t_opt <= 0.85


def test_cleaning_preserves_rows_and_prevents_leakage():
    df_raw = load_raw_german_data()
    X_train, X_test, y_train, y_test = split_train_test(df_raw)

    cleaned = clean_static_rules(X_train)
    assert len(cleaned) == len(X_train) == 800
    assert cleaned["purpose_clean"].isin(["domestic_app", "repairs", "retraining", "others"]).sum() == 0
    assert (cleaned["purpose_clean"] == "rare_other").sum() > 0
    assert np.allclose(cleaned["monthly_burden"], X_train["credit_amount"] / X_train["duration"])

    # Fit binner strictly on train and transform extreme test values without NaN
    binner = QuantileFeatureBinner(cols=["credit_amount", "duration", "monthly_burden"], n_bins=4)
    binner.fit(cleaned)
    extreme_test = clean_static_rules(X_test)
    extreme_test.loc[extreme_test.index[0], "credit_amount"] = 999999
    binned_test = binner.transform(extreme_test)
    assert binned_test[["credit_amount_bin", "duration_bin", "monthly_burden_bin"]].isna().sum().sum() == 0

    # Fit clipper on train only
    clipper = PercentileClipLogTransformer(cols=["credit_amount"], apply_log1p=False)
    clipper.fit(cleaned)
    train_p99 = cleaned["credit_amount"].quantile(0.99)
    clipped_test = clipper.transform(extreme_test)
    assert np.isclose(clipped_test.loc[extreme_test.index[0], "credit_amount"], train_p99)

    for mode in ["binned", "continuous"]:
        prep = build_preprocessor(mode=mode)
        Xt_tr = prep.fit_transform(X_train)
        Xt_te = prep.transform(X_test)
        assert Xt_tr.shape[1] == Xt_te.shape[1] == len(get_feature_names(prep))
        assert not np.isnan(Xt_tr).any()
        assert not np.isnan(Xt_te).any()


def test_all_models_smotenc_and_interpretability_end_to_end():
    df_raw = load_raw_german_data()
    X_train, X_test, y_train, y_test = split_train_test(df_raw)

    lr = build_logistic_pipeline("binned", "cost_5_1")
    svm = build_svm_pipeline("binned", "cost_5_1")
    gbdt = build_gbdt_pipeline("binned", "cost_5_1", n_estimators=30)
    dt = build_decision_tree_pipeline("binned", "cost_5_1", max_depth=3)
    smote_lr = build_smotenc_pipeline(
        build_logistic_regression(class_weight_mode="none"), mode="binned"
    )

    for clf in [lr, svm, gbdt, dt, smote_lr]:
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        assert probs.shape == (len(X_test),)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

    odds_df = extract_odds_ratios(lr)
    gain_df = extract_gbdt_gain_importance(gbdt)
    surrogate = fit_global_surrogate_tree(gbdt, X_train, y_train, teacher_threshold=0.5, max_depth=3)

    assert len(odds_df) > 10 and (odds_df["odds_ratio"] > 0).all()
    assert len(gain_df) > 10 and np.isclose(gain_df["gain_importance"].sum(), 1.0, atol=1e-3)
    assert 0.5 <= surrogate["fidelity_agreement"] <= 1.0
    assert "class:" in surrogate["rules_text"]


def test_plotting_functions_render_without_error(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.interpret import plot_eda_overview, plot_model_comparison
    from src.metrics import compute_classification_metrics

    df_raw = load_raw_german_data()
    audit = audit_data_quality(df_raw)
    fig1 = plot_eda_overview(df_raw, audit, save_path=tmp_path / "eda.png")
    assert (tmp_path / "eda.png").exists()

    X_train, X_test, y_train, y_test = split_train_test(df_raw)
    lr = build_logistic_pipeline("binned", "cost_5_1").fit(X_train, y_train)
    probs = lr.predict_proba(X_test)[:, 1]
    mock_res = {
        "LogisticRegression": {
            "test_probs": probs,
            "threshold": 0.45,
            "test_metrics": compute_classification_metrics(y_test, probs, 0.45),
        }
    }
    fig2 = plot_model_comparison(mock_res, y_test, save_path=tmp_path / "cmp.png")
    assert (tmp_path / "cmp.png").exists()
    plt.close(fig1)
    plt.close(fig2)
