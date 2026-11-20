# Personal Credit Risk Assessment (UCI Statlog German Credit Data)

Isolated, modular Data Mining project built strictly on `../statlog+german+credit+data/german.data` (read-only; raw data directory is never modified).

## Project Structure

- `src/config.py`: Dataset paths, schema, readable business maps, Cost Matrix (`FN=5`, `FP=1`), and Bayes threshold ($t^* = 1/6 \approx 0.1667$).
- `src/data_audit.py`: 6-dimension data quality audit (`Data.pptx` Slide 62), Information Value (IV), $\chi^2$ tests, logical contradiction checks, and side-by-side loss comparison against `german.data-numeric`.
- `src/clean_data.py`: Row-preserving semantic cleaning (contradiction flags, explicit `"unknown"` indicators, rare `purpose` grouping `<3%`, `monthly_burden = credit_amount / duration`, domain `age` bins, fold-fit quantile binning & `[p1, p99]` winsorizing + `log1p`).
- `src/imbalance.py`: Cost-sensitive Class Weighting (`5:1` / `balanced`) and in-fold `SMOTE-NC` resampling before One-Hot encoding.
- `src/metrics.py`: Pure-math `Expected Cost`, Bayes threshold, OOF threshold optimization, `ROC-AUC`, `PR-AUC`, `F2`, `Brier Score`, 95% bootstrap CIs, and fairness slice analysis.
- `src/models/`: Dedicated files for each model family:
  - `logistic_regression.py` (L1/L2, Odds Ratios $e^\beta$)
  - `svm_model.py` (RBF & Linear kernels with Platt calibration)
  - `gradient_boosting.py` (Cost-weighted shallow GBDT, gain importance)
  - `decision_tree.py` (Standalone shallow Entropy tree & Global Surrogate tree)
- `src/evaluate.py`: Fixed Stratified 80/20 split + Stratified 5-Fold CV tuning, out-of-fold threshold calibration, imbalance ablation, and preprocessing ablation.
- `src/interpret.py`: Raw-feature Permutation Importance across LR/SVM/GBDT, Odds Ratios, GBDT Gain, Decision Tree rule visualization, and all charts.
- `run_all.py`: End-to-end reproducible CLI runner exporting summaries and PNG figures to `outputs/`.
- `report.ipynb`: Executed Jupyter Notebook report importing directly from `src/` for lecturer submission.
- `tests/test_pipeline.py`: Automated integrity, math, anti-leakage, and end-to-end pipeline tests.

## Quick Start (Windows PowerShell)

```powershell
# Run automated verification tests
.\.venv\Scripts\python.exe -m pytest -v

# Run full pipeline and regenerate outputs/
.\.venv\Scripts\python.exe run_all.py

# Launch or re-execute the Jupyter report
.\.venv\Scripts\python.exe -m jupyter lab report.ipynb
```

