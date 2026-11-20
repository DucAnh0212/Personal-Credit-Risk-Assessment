# Graph Report - credit_risk_assessment  (2026-11-20)

## Corpus Check
- 16 files · ~8,180 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 10 file(s) not represented in the graph (top: .csv 7, (none) 1, .ini 1)

## Summary
- 195 nodes · 526 edges · 14 communities (7 shown, 7 thin omitted)
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
5. `expected_cost()` - 13 edges
6. `PercentileClipLogTransformer` - 13 edges
7. `run_imbalance_ablation()` - 13 edges
8. `test_all_models_smotenc_and_interpretability_end_to_end()` - 13 edges
9. `test_cleaning_preserves_rows_and_prevents_leakage()` - 13 edges
10. `QuantileFeatureBinner` - 12 edges

## Surprising Connections (you probably didn't know these)
- `test_all_models_smotenc_and_interpretability_end_to_end()` --calls--> `build_smotenc_pipeline()`  [EXTRACTED]
  tests/test_pipeline.py → src/imbalance.py
- `test_plotting_functions_render_without_error()` --calls--> `plot_eda_overview()`  [EXTRACTED]
  tests/test_pipeline.py → src/interpret.py
- `test_plotting_functions_render_without_error()` --calls--> `plot_model_comparison()`  [EXTRACTED]
  tests/test_pipeline.py → src/interpret.py
- `test_plotting_functions_render_without_error()` --calls--> `compute_classification_metrics()`  [EXTRACTED]
  tests/test_pipeline.py → src/metrics.py
- `main()` --calls--> `slice_error_analysis()`  [EXTRACTED]
  run_all.py → src/metrics.py

## Import Cycles
- None detected.

## Communities (14 total, 7 thin omitted)

### Community 0 - "evaluate.py"
Cohesion: 0.11
Nodes (15): run_imbalance_ablation(), run_preprocessing_ablation(), run_primary_model_comparison(), split_train_test(), tune_and_evaluate_model(), build_decision_tree_pipeline(), build_gbdt_pipeline(), build_gradient_boosting() (+7 more)

### Community 1 - "build_preprocessor"
Cohesion: 0.14
Nodes (6): clean_static_rules(), PercentileClipLogTransformer, QuantileFeatureBinner, StaticRuleCleaner, build_smotenc_pipeline(), test_cleaning_preserves_rows_and_prevents_leakage()

### Community 2 - "run_all.py"
Cohesion: 0.18
Nodes (7): main(), compute_permutation_importance_raw(), plot_confusion_matrices(), plot_decision_trees(), plot_eda_overview(), plot_feature_importance(), plot_model_comparison()

### Community 3 - "numpy"
Cohesion: 0.18
Nodes (9): bayes_optimal_threshold(), bootstrap_metric_ci(), compute_classification_metrics(), expected_cost(), find_optimal_threshold(), naive_baselines(), slice_error_analysis(), total_cost() (+1 more)

### Community 4 - "imbalance.py"
Cohesion: 0.14
Nodes (4): build_preprocessor(), resolve_class_weight(), build_decision_tree(), build_svm()

### Community 5 - "test_pipeline.py"
Cohesion: 0.19
Nodes (7): audit_data_quality(), compute_information_value(), load_raw_german_data(), load_raw_german_numeric(), extract_gbdt_gain_importance(), test_plotting_functions_render_without_error(), test_raw_data_audit_integrity()

### Community 6 - "decision_tree.py"
Cohesion: 0.17
Nodes (5): get_feature_names(), extract_standalone_tree_summary(), fit_global_surrogate_tree(), summarize_tree_leaves(), extract_odds_ratios()

## Knowledge Gaps
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ColumnSelector` connect `ColumnSelector` to `build_preprocessor`, `imbalance.py`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Should `evaluate.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10634920634920635 - nodes in this community are weakly interconnected._
- **Why does `build_smotenc_pipeline()` connect `build_preprocessor` to `evaluate.py`, `imbalance.py`, `test_pipeline.py`, `ColumnSelector`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Should `build_preprocessor` be split into smaller, more focused modules?**
  _Cohesion score 0.14130434782608695 - nodes in this community are weakly interconnected._
- **Why does `PercentileClipLogTransformer` connect `build_preprocessor` to `imbalance.py`, `test_pipeline.py`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Should `imbalance.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1402116402116402 - nodes in this community are weakly interconnected._