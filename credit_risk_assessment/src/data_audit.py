from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.config import (
    CATEGORICAL_COLS,
    COLUMN_NAMES,
    CONTINUOUS_COLS,
    DISCRETE_NUM_COLS,
    DOC_CODEBOOK,
    GERMAN_DATA_PATH,
    GERMAN_NUMERIC_PATH,
)


def load_raw_german_data(path: Path = GERMAN_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES, engine="python")
    df["bad_risk"] = (df["raw_target"] == 2).astype(int)
    return df


def load_raw_german_numeric(path: Path = GERMAN_NUMERIC_PATH) -> pd.DataFrame:
    cols = [f"num_col_{i}" for i in range(1, 25)] + ["raw_target"]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=cols, engine="python")
    df["bad_risk"] = (df["raw_target"] == 2).astype(int)
    return df


def compute_information_value(
    df: pd.DataFrame,
    feature: str,
    target: str = "bad_risk",
    n_bins: int = 5,
) -> float:
    series = df[feature]
    if feature in CONTINUOUS_COLS:
        binned = pd.qcut(series, q=n_bins, duplicates="drop")
    else:
        binned = series

    ct = pd.crosstab(binned, df[target])
    good = ct.get(0, pd.Series(0, index=ct.index)).astype(float)
    bad = ct.get(1, pd.Series(0, index=ct.index)).astype(float)

    k = len(ct)
    p_good = (good + 0.5) / (good.sum() + 0.5 * k)
    p_bad = (bad + 0.5) / (bad.sum() + 0.5 * k)
    iv = float(((p_good - p_bad) * np.log(p_good / p_bad)).sum())
    return round(iv, 4)


def audit_data_quality(
    df_raw: pd.DataFrame,
    df_num: pd.DataFrame | None = None,
) -> dict:
    feature_cols = [c for c in COLUMN_NAMES if c != "raw_target"]

    # 1. Completeness & duplicates
    explicit_missing = int(df_raw[feature_cols].isna().sum().sum())
    dup_full = int(df_raw[feature_cols + ["raw_target"]].duplicated().sum())
    dup_features = int(df_raw[feature_cols].duplicated().sum())

    implicit_unknowns = {
        "checking_A14_no_checking": {
            "count": int((df_raw["checking_status"] == "A14").sum()),
            "bad_rate": round(float(df_raw.loc[df_raw["checking_status"] == "A14", "bad_risk"].mean()), 4),
        },
        "savings_A65_unknown_or_none": {
            "count": int((df_raw["savings_status"] == "A65").sum()),
            "bad_rate": round(float(df_raw.loc[df_raw["savings_status"] == "A65", "bad_risk"].mean()), 4),
        },
        "property_A124_unknown_or_none": {
            "count": int((df_raw["property_magnitude"] == "A124").sum()),
            "bad_rate": round(float(df_raw.loc[df_raw["property_magnitude"] == "A124", "bad_risk"].mean()), 4),
        },
    }

    # 2. Codebook conformance vs german.doc
    dead_doc_codes = {}
    unexpected_codes = {}
    for col, expected in DOC_CODEBOOK.items():
        observed = set(df_raw[col].unique())
        missing = sorted(set(expected) - observed)
        extra = sorted(observed - set(expected))
        if missing:
            dead_doc_codes[col] = missing
        if extra:
            unexpected_codes[col] = extra

    # 3. Logical / domain consistency checks
    contradictions = {
        "age_lt_25_with_employment_ge_7yr": int(
            ((df_raw["age"] < 25) & (df_raw["employment"] == "A75")).sum()
        ),
        "unemployed_A71_with_high_skill_job_A174": int(
            ((df_raw["employment"] == "A71") & (df_raw["job"] == "A174")).sum()
        ),
        "own_housing_A152_with_no_property_A124": int(
            ((df_raw["housing"] == "A152") & (df_raw["property_magnitude"] == "A124")).sum()
        ),
        "rent_housing_A151_with_real_estate_A121": int(
            ((df_raw["housing"] == "A151") & (df_raw["property_magnitude"] == "A121")).sum()
        ),
        "credit_hist_A30_with_existing_credits_ge_2": int(
            ((df_raw["credit_history"] == "A30") & (df_raw["existing_credits"] >= 2)).sum()
        ),
    }

    # 4. Numerical distributions & IQR outliers
    num_summary_rows = []
    for col in CONTINUOUS_COLS + DISCRETE_NUM_COLS:
        s = df_raw[col].astype(float)
        q25, med, q75 = s.quantile([0.25, 0.50, 0.75])
        p99 = float(s.quantile(0.99))
        iqr = float(q75 - q25)
        outliers = int(((s < q25 - 1.5 * iqr) | (s > q75 + 1.5 * iqr)).sum())
        num_summary_rows.append(
            {
                "feature": col,
                "n_unique": int(s.nunique()),
                "min": float(s.min()),
                "q25": float(q25),
                "median": float(med),
                "q75": float(q75),
                "p99": round(p99, 2),
                "max": float(s.max()),
                "mean": round(float(s.mean()), 2),
                "std": round(float(s.std(ddof=0)), 2),
                "skew": round(float(s.skew()), 3),
                "iqr_outliers": outliers,
                "mean_good": round(float(df_raw.loc[df_raw["bad_risk"] == 0, col].mean()), 2),
                "mean_bad": round(float(df_raw.loc[df_raw["bad_risk"] == 1, col].mean()), 2),
            }
        )
    numeric_summary = pd.DataFrame(num_summary_rows)

    # 5. Univariate predictive strength: IV + Chi-square p-values for categoricals
    strength_rows = []
    for col in feature_cols:
        iv = compute_information_value(df_raw, col)
        p_val = None
        chi2_stat = None
        if col in CATEGORICAL_COLS + DISCRETE_NUM_COLS:
            ct = pd.crosstab(df_raw[col], df_raw["bad_risk"])
            chi2_stat, p_val, _, _ = chi2_contingency(ct)
            chi2_stat = round(float(chi2_stat), 3)
            p_val = round(float(p_val), 6)
        strength_rows.append(
            {
                "feature": col,
                "iv": iv,
                "chi2": chi2_stat,
                "chi2_pvalue": p_val,
            }
        )
    feature_strength = (
        pd.DataFrame(strength_rows)
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )

    # 6. Numeric-file loss comparison (german.data-numeric vs german.data)
    numeric_file_diff = {}
    if df_num is not None:
        amt_round_matches = int(
            (np.round(df_raw["credit_amount"] / 100.0) == df_num["num_col_4"]).sum()
        )
        numeric_file_diff = {
            "n_rows_numeric": int(len(df_num)),
            "target_mismatches": int((df_raw["raw_target"] != df_num["raw_target"]).sum()),
            "credit_amount_unique_raw": int(df_raw["credit_amount"].nunique()),
            "credit_amount_unique_numeric": int(df_num["num_col_4"].nunique()),
            "credit_amount_round100_exact_matches": amt_round_matches,
            "purpose_A40_flag_matches": int(
                ((df_raw["purpose"] == "A40").astype(int) == df_num["num_col_16"]).sum()
            ),
            "purpose_A41_flag_matches": int(
                ((df_raw["purpose"] == "A41").astype(int) == df_num["num_col_17"]).sum()
            ),
            "installment_rate_present_in_numeric": False,
        }

    return {
        "n_rows": int(len(df_raw)),
        "n_features": len(feature_cols),
        "target_counts": {
            "good_0": int((df_raw["bad_risk"] == 0).sum()),
            "bad_1": int((df_raw["bad_risk"] == 1).sum()),
            "bad_rate": round(float(df_raw["bad_risk"].mean()), 4),
        },
        "explicit_missing": explicit_missing,
        "duplicate_full_rows": dup_full,
        "duplicate_feature_rows": dup_features,
        "implicit_unknowns": implicit_unknowns,
        "dead_doc_codes": dead_doc_codes,
        "unexpected_codes": unexpected_codes,
        "contradictions": contradictions,
        "corr_duration_credit_amount": round(
            float(df_raw["duration"].corr(df_raw["credit_amount"])), 4
        ),
        "numeric_summary": numeric_summary,
        "feature_strength": feature_strength,
        "numeric_file_diff": numeric_file_diff,
    }

