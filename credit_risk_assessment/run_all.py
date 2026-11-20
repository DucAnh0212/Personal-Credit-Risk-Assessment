"""Reproducible CLI entry point: audit -> clean -> CV benchmark -> interpretability -> outputs."""

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import BAYES_OPTIMAL_THRESHOLD, OUTPUTS_DIR
from src.data_audit import audit_data_quality, load_raw_german_data, load_raw_german_numeric
from src.evaluate import (
    run_imbalance_ablation,
    run_preprocessing_ablation,
    run_primary_model_comparison,
)
from src.interpret import (
    compute_permutation_importance_raw,
    plot_confusion_matrices,
    plot_decision_trees,
    plot_eda_overview,
    plot_feature_importance,
    plot_model_comparison,
)
from src.metrics import slice_error_analysis
from src.models.decision_tree import fit_global_surrogate_tree
from src.models.gradient_boosting import extract_gbdt_gain_importance
from src.models.logistic_regression import extract_odds_ratios


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Loading raw German Credit datasets (read-only) and running 6-dimension audit...")
    df_raw = load_raw_german_data()
    df_num = load_raw_german_numeric()
    audit = audit_data_quality(df_raw, df_num)
    print(
        f"   Loaded {audit['n_rows']} rows x {audit['n_features']} features | "
        f"Bad rate: {audit['target_counts']['bad_rate']*100:.1f}% | "
        f"Bayes t*: {BAYES_OPTIMAL_THRESHOLD:.4f}"
    )

    print("2. Running Stratified 5-Fold CV tuning & evaluation for LR, SVM, GBDT, and Decision Tree...")
    bench = run_primary_model_comparison(df_raw)
    print("\n--- Naive Baselines (Test Set, n=200) ---")
    print(bench["baselines_test"].to_string(index=False))
    print("\n--- Primary Model Comparison (Binned Clean Data, Cost-Weighted 5:1 + OOF Threshold) ---")
    print(bench["summary_df"].to_string(index=False))
    print(f"\nSelected best primary model (by OOF Expected Cost): {bench['best_model_name']}")

    print("\n3. Running Class Imbalance Ablation (Class Weighting 5:1 / Balanced / Bayes t* / SMOTE-NC)...")
    imb_df = run_imbalance_ablation(
        bench["X_train"], bench["y_train"], bench["X_test"], bench["y_test"]
    )
    print(imb_df.to_string(index=False))

    print("\n4. Running Discretization Ablation (Binned vs Clipped+Log Continuous)...")
    prep_df = run_preprocessing_ablation(
        bench["X_train"], bench["y_train"], bench["X_test"], bench["y_test"]
    )
    print(prep_df.to_string(index=False))

    print("\n5. Extracting Interpretability Artifacts (Odds Ratios, Permutation Importance, GBDT Gain, Decision Trees)...")
    lr_res = bench["results"]["LogisticRegression"]
    svm_res = bench["results"]["SVM"]
    gbdt_res = bench["results"]["GradientBoosting"]
    dt_res = bench["results"]["DecisionTree_Shallow"]
    best_res = bench["results"][bench["best_model_name"]]

    lr_odds = extract_odds_ratios(lr_res["best_estimator"])
    gbdt_gain = extract_gbdt_gain_importance(gbdt_res["best_estimator"])
    perm_dict = {
        "LogisticRegression": compute_permutation_importance_raw(
            lr_res["best_estimator"], bench["X_test"], bench["y_test"]
        ),
        "SVM": compute_permutation_importance_raw(
            svm_res["best_estimator"], bench["X_test"], bench["y_test"]
        ),
        "GradientBoosting": compute_permutation_importance_raw(
            gbdt_res["best_estimator"], bench["X_test"], bench["y_test"]
        ),
    }

    surrogate = fit_global_surrogate_tree(
        teacher_pipeline=best_res["best_estimator"],
        X_raw=bench["X_train"],
        y_actual=bench["y_train"],
        teacher_threshold=best_res["threshold"],
        max_depth=4,
        min_samples_leaf=20,
        ccp_alpha=0.005,
    )
    print(f"   Surrogate Tree Fidelity vs {bench['best_model_name']}: {surrogate['fidelity_agreement']*100:.1f}%")

    # Slice / fairness error analysis on test set
    test_raw_slice = df_raw.loc[bench["X_test"].index]
    fairness_df = slice_error_analysis(
        test_raw_slice, bench["y_test"], best_res["test_probs"], best_res["threshold"]
    )

    print("6. Saving charts and tabular summaries to outputs/...")
    plot_eda_overview(df_raw, audit, OUTPUTS_DIR / "01_eda_overview.png")
    plot_model_comparison(bench["results"], bench["y_test"], OUTPUTS_DIR / "02_model_comparison.png")
    plot_confusion_matrices(bench["results"], bench["y_test"], OUTPUTS_DIR / "03_confusion_matrices.png")
    plot_feature_importance(lr_odds, perm_dict, gbdt_gain, OUTPUTS_DIR / "04_feature_importance.png")
    plot_decision_trees(
        dt_res["best_estimator"], surrogate, bench["best_model_name"], OUTPUTS_DIR / "05_decision_trees.png"
    )
    plt.close("all")

    bench["summary_df"].to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)
    imb_df.to_csv(OUTPUTS_DIR / "imbalance_ablation.csv", index=False)
    prep_df.to_csv(OUTPUTS_DIR / "preprocessing_ablation.csv", index=False)
    lr_odds.to_csv(OUTPUTS_DIR / "lr_odds_ratios.csv", index=False)
    gbdt_gain.to_csv(OUTPUTS_DIR / "gbdt_gain_importance.csv", index=False)
    fairness_df.to_csv(OUTPUTS_DIR / "slice_fairness_analysis.csv", index=False)
    surrogate["leaf_summary"].to_csv(OUTPUTS_DIR / "surrogate_leaf_summary.csv", index=False)
    (OUTPUTS_DIR / "surrogate_rules.txt").write_text(surrogate["rules_text"], encoding="utf-8")

    report_payload = {
        "best_model_name": bench["best_model_name"],
        "bayes_optimal_threshold": round(BAYES_OPTIMAL_THRESHOLD, 4),
        "best_params": {k: v["best_params"] for k, v in bench["results"].items()},
        "test_metrics": {k: v["test_metrics"] for k, v in bench["results"].items()},
        "test_ci95": {k: v["test_ci95"] for k, v in bench["results"].items()},
        "surrogate_fidelity": surrogate["fidelity_agreement"],
        "audit_highlights": {
            "n_rows": audit["n_rows"],
            "explicit_missing": audit["explicit_missing"],
            "duplicate_full_rows": audit["duplicate_full_rows"],
            "dead_doc_codes": audit["dead_doc_codes"],
            "contradictions": audit["contradictions"],
            "implicit_unknowns": audit["implicit_unknowns"],
            "numeric_file_diff": audit["numeric_file_diff"],
        },
    }
    (OUTPUTS_DIR / "summary.json").write_text(
        json.dumps(report_payload, indent=2), encoding="utf-8"
    )
    print(f"Done. Outputs written to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
