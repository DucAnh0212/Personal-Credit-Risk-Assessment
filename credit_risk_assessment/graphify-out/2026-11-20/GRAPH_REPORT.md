# Graph Report - credit_risk_assessment  (2026-11-20)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 195 nodes · 529 edges · 11 communities (6 shown, 5 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c629b2db`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- evaluate.py
- build_preprocessor
- run_all.py
- numpy
- imbalance.py
- test_pipeline.py
- decision_tree.py
- ColumnSelector
- sample_weights_from_class_weight

## God Nodes (most connected - your core abstractions)
1. `main()` - 19 edges
2. `build_preprocessor()` - 17 edges
3. `run_primary_model_comparison()` - 16 edges
4. `get_feature_names()` - 14 edges
5. `PercentileClipLogTransformer` - 13 edges
6. `run_imbalance_ablation()` - 13 edges
7. `test_all_models_smotenc_and_interpretability_end_to_end()` - 13 edges
8. `test_cleaning_preserves_rows_and_prevents_leakage()` - 13 edges
9. `expected_cost()` - 13 edges
10. `QuantileFeatureBinner` - 12 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `run_imbalance_ablation()`  [EXTRACTED]
  run_all.py → src/evaluate.py
- `main()` --calls--> `run_preprocessing_ablation()`  [EXTRACTED]
  run_all.py → src/evaluate.py
- `main()` --calls--> `extract_gbdt_gain_importance()`  [EXTRACTED]
  run_all.py → src/models/gradient_boosting.py
- `main()` --calls--> `slice_error_analysis()`  [EXTRACTED]
  run_all.py → src/metrics.py
- `main()` --calls--> `load_raw_german_data()`  [EXTRACTED]
  run_all.py → src/data_audit.py

## Import Cycles
- None detected.

## Communities (11 total, 5 thin omitted)

### Community 0 - "evaluate.py"
Cohesion: 0.09
Nodes (16): run_imbalance_ablation(), run_preprocessing_ablation(), run_primary_model_comparison(), split_train_test(), tune_and_evaluate_model(), build_decision_tree_pipeline(), build_gbdt_pipeline(), build_gradient_boosting() (+8 more)

### Community 1 - "build_preprocessor"
Cohesion: 0.14
Nodes (7): build_preprocessor(), clean_static_rules(), PercentileClipLogTransformer, QuantileFeatureBinner, StaticRuleCleaner, build_smotenc_pipeline(), test_cleaning_preserves_rows_and_prevents_leakage()

### Community 2 - "run_all.py"
Cohesion: 0.19
Nodes (7): main(), compute_permutation_importance_raw(), plot_confusion_matrices(), plot_decision_trees(), plot_eda_overview(), plot_feature_importance(), plot_model_comparison()

### Community 3 - "numpy"
Cohesion: 0.18
Nodes (9): bayes_optimal_threshold(), bootstrap_metric_ci(), compute_classification_metrics(), expected_cost(), find_optimal_threshold(), naive_baselines(), slice_error_analysis(), total_cost() (+1 more)

### Community 5 - "test_pipeline.py"
Cohesion: 0.20
Nodes (7): audit_data_quality(), compute_information_value(), load_raw_german_data(), load_raw_german_numeric(), extract_odds_ratios(), test_plotting_functions_render_without_error(), test_raw_data_audit_integrity()

### Community 6 - "decision_tree.py"
Cohesion: 0.16
Nodes (5): get_feature_names(), build_decision_tree(), extract_standalone_tree_summary(), fit_global_surrogate_tree(), summarize_tree_leaves()

## Knowledge Gaps
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ColumnSelector` connect `ColumnSelector` to `build_preprocessor`, `imbalance.py`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Should `evaluate.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09059233449477352 - nodes in this community are weakly interconnected._
- **Why does `build_smotenc_pipeline()` connect `build_preprocessor` to `evaluate.py`, `imbalance.py`, `test_pipeline.py`, `ColumnSelector`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Should `build_preprocessor` be split into smaller, more focused modules?**
  _Cohesion score 0.13538461538461538 - nodes in this community are weakly interconnected._
- **Why does `PercentileClipLogTransformer` connect `build_preprocessor` to `imbalance.py`, `test_pipeline.py`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Should `imbalance.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1471861471861472 - nodes in this community are weakly interconnected._