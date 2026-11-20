import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    AGE_DOMAIN_EDGES,
    AGE_DOMAIN_LABELS,
    DISCRETE_NUM_COLS,
    EMPLOYMENT_ORDER,
    JOB_ORDER,
    RARE_PURPOSE_CODES,
    READABLE_MAPS,
)

# Columns after clean_static_rules (non-redundant domain contradiction flags;
# implicit unknowns A14/A65/A124 are already represented cleanly once by OneHotEncoder)
STATIC_FLAG_COLS = [
    "flag_age_emp_conflict",
    "flag_unemp_high_job",
    "flag_housing_prop_mismatch",
    "flag_a30_multi_credits",
]
ORDINAL_ENCODED_COLS = ["employment_ord", "job_ord"]
NOMINAL_CLEAN_COLS = [
    "checking_status",
    "credit_history",
    "purpose_clean",
    "savings_status",
    "personal_status_sex",
    "other_debtors",
    "property_magnitude",
    "other_installment_plans",
    "housing",
    "telephone",
    "foreign_worker",
]
SKEWED_CONTINUOUS_COLS = ["credit_amount", "duration", "monthly_burden"]
BINNED_CAT_COLS = ["age_bin", "credit_amount_bin", "duration_bin", "monthly_burden_bin"]


def clean_static_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 1. Domain contradiction flags (computed on raw codes before label translation)
    out["flag_age_emp_conflict"] = ((out["age"] < 25) & (out["employment"] == "A75")).astype(int)
    out["flag_unemp_high_job"] = ((out["employment"] == "A71") & (out["job"] == "A174")).astype(int)
    out["flag_housing_prop_mismatch"] = (
        ((out["housing"] == "A152") & (out["property_magnitude"] == "A124"))
        | ((out["housing"] == "A151") & (out["property_magnitude"] == "A121"))
    ).astype(int)
    out["flag_a30_multi_credits"] = (
        (out["credit_history"] == "A30") & (out["existing_credits"] >= 2)
    ).astype(int)

    # 2. Implicit missing / structural absence indicators
    out["is_no_checking"] = (out["checking_status"] == "A14").astype(int)
    out["is_unknown_savings"] = (out["savings_status"] == "A65").astype(int)
    out["is_no_property"] = (out["property_magnitude"] == "A124").astype(int)

    # 3. True monotonic ordinals (keeping checking/savings/property nominal so unknown isn't ordered)
    out["employment_ord"] = out["employment"].map(EMPLOYMENT_ORDER).astype(float)
    out["job_ord"] = out["job"].map(JOB_ORDER).astype(float)

    # 4. Group rare purpose categories (< 3% each: A44, A45, A48, A410 => 55 rows total)
    out["purpose_clean"] = out["purpose"].where(
        ~out["purpose"].isin(RARE_PURPOSE_CODES), "rare_other"
    )
    out["purpose_clean"] = out["purpose_clean"].replace(READABLE_MAPS["purpose"])

    # 5. Translate remaining nominal codes into readable labels for interpretability
    for col in NOMINAL_CLEAN_COLS:
        if col == "purpose_clean":
            continue
        out[col] = out[col].replace(READABLE_MAPS[col])

    # 6. Pure-math derived feature compensating for absent raw income
    out["monthly_burden"] = out["credit_amount"].astype(float) / out["duration"].astype(float)

    # 7. Domain-fixed discretization for age (zero data leakage)
    out["age_bin"] = pd.cut(
        out["age"],
        bins=AGE_DOMAIN_EDGES,
        labels=AGE_DOMAIN_LABELS,
        include_lowest=True,
    ).astype(str)

    return out


class StaticRuleCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return clean_static_rules(X)


class PercentileClipLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        cols: list[str] | None = None,
        lower_q: float = 0.01,
        upper_q: float = 0.99,
        apply_log1p: bool = True,
    ):
        self.cols = cols
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.apply_log1p = apply_log1p

    def fit(self, X: pd.DataFrame, y=None):
        target_cols = self.cols if self.cols is not None else list(X.columns)
        self.bounds_ = {}
        for col in target_cols:
            lo = float(np.quantile(X[col].astype(float), self.lower_q))
            hi = float(np.quantile(X[col].astype(float), self.upper_q))
            self.bounds_[col] = (lo, hi)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, (lo, hi) in self.bounds_.items():
            clipped = np.clip(out[col].astype(float), lo, hi)
            out[col] = np.log1p(clipped) if self.apply_log1p else clipped
        return out

    def get_feature_names_out(self, input_features=None):
        names = input_features if input_features is not None else list(self.bounds_.keys())
        return np.asarray(names, dtype=object)


class QuantileFeatureBinner(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        cols: list[str] | None = None,
        n_bins: int = 4,
    ):
        self.cols = cols
        self.n_bins = n_bins

    def fit(self, X: pd.DataFrame, y=None):
        target_cols = self.cols if self.cols is not None else SKEWED_CONTINUOUS_COLS
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
        self.edges_ = {}
        self.labels_ = {}
        base_names = ["Q1_low", "Q2_med_low", "Q3_med_high", "Q4_high", "Q5_very_high"]

        for col in target_cols:
            raw_edges = np.quantile(X[col].astype(float), quantiles)
            unique_edges = np.unique(raw_edges)
            # Extend outer bounds to -inf/+inf so unseen fold values never become NaN
            unique_edges[0] = -np.inf
            unique_edges[-1] = np.inf
            n_intervals = len(unique_edges) - 1
            labels = base_names[:n_intervals] if n_intervals <= len(base_names) else [
                f"Q{i + 1}" for i in range(n_intervals)
            ]
            self.edges_[col] = unique_edges
            self.labels_[col] = labels
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, edges in self.edges_.items():
            out[f"{col}_bin"] = pd.cut(
                out[col].astype(float),
                bins=edges,
                labels=self.labels_[col],
                include_lowest=True,
            ).astype(str)
        return out


def build_preprocessor(mode: str = "binned", scale_numeric: bool = True) -> Pipeline:
    """Build leakage-free preprocessing Pipeline.

    Modes:
      - `binned` (Primary, matches prompt): domain + fold-fit quantile discretization on
        continuous features (`age`, `credit_amount`, `duration`, `monthly_burden`) +
        readable One-Hot encoding + scaled discrete/ordinal/flag features.
      - `continuous` (Ablation): fold-fit [p1, p99] clipping + log1p on continuous features
        + One-Hot on nominal features + StandardScaler.
    """
    scaler = StandardScaler() if scale_numeric else "passthrough"
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    if mode == "binned":
        num_cols = DISCRETE_NUM_COLS + ORDINAL_ENCODED_COLS + STATIC_FLAG_COLS
        cat_cols = BINNED_CAT_COLS + NOMINAL_CLEAN_COLS
        col_tf = ColumnTransformer(
            transformers=[
                ("num", scaler, num_cols),
                ("cat", ohe, cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return Pipeline(
            steps=[
                ("static_clean", StaticRuleCleaner()),
                ("quantile_bin", QuantileFeatureBinner(cols=SKEWED_CONTINUOUS_COLS, n_bins=4)),
                ("encode_scale", col_tf),
            ]
        )

    if mode == "continuous":
        cont_cols = SKEWED_CONTINUOUS_COLS + ["age"]
        other_num_cols = DISCRETE_NUM_COLS + ORDINAL_ENCODED_COLS + STATIC_FLAG_COLS
        cont_pipe = Pipeline(
            steps=[
                (
                    "clip_log",
                    PercentileClipLogTransformer(
                        cols=cont_cols, lower_q=0.01, upper_q=0.99, apply_log1p=True
                    ),
                ),
                ("scale", scaler),
            ]
        )
        col_tf = ColumnTransformer(
            transformers=[
                ("cont", cont_pipe, cont_cols),
                ("num", scaler, other_num_cols),
                ("cat", ohe, NOMINAL_CLEAN_COLS),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return Pipeline(
            steps=[
                ("static_clean", StaticRuleCleaner()),
                ("encode_scale", col_tf),
            ]
        )

    raise ValueError(f"Unsupported preprocessing mode: {mode}")


def get_feature_names(fitted_preprocessor: Pipeline) -> list[str]:
    """Extract clean human-readable output feature names from a fitted preprocessor."""
    col_tf = fitted_preprocessor.named_steps["encode_scale"]
    return [str(name) for name in col_tf.get_feature_names_out()]
