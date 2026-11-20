import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.clean_data import (
    BINNED_CAT_COLS,
    NOMINAL_CLEAN_COLS,
    ORDINAL_ENCODED_COLS,
    SKEWED_CONTINUOUS_COLS,
    STATIC_FLAG_COLS,
    PercentileClipLogTransformer,
    QuantileFeatureBinner,
    StaticRuleCleaner,
)
from src.config import COST_FN, COST_FP, DISCRETE_NUM_COLS, RANDOM_STATE

CLASS_WEIGHT_MAP = {
    "cost_5_1": {0: COST_FP, 1: COST_FN},
    "balanced": "balanced",
    "none": None,
}

def resolve_class_weight(mode: str):
    if mode not in CLASS_WEIGHT_MAP:
        raise ValueError(f"Unknown class_weight mode '{mode}'. Choose from {list(CLASS_WEIGHT_MAP)}")
    return CLASS_WEIGHT_MAP[mode]


def sample_weights_from_class_weight(y: np.ndarray, class_weight) -> np.ndarray | None:
    if class_weight is None:
        return None
    return compute_sample_weight(class_weight=class_weight, y=np.asarray(y))

class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, num_cols: list[str], cat_cols: list[str]):
        self.num_cols = num_cols
        self.cat_cols = cat_cols

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[self.num_cols + self.cat_cols].copy()
        for col in self.num_cols:
            out[col] = out[col].astype(float)
        for col in self.cat_cols:
            out[col] = out[col].astype(str)
        return out


def build_smotenc_pipeline(
    estimator,
    mode: str = "binned",
    sampling_strategy: float | str = "auto",
    random_state: int = RANDOM_STATE,
) -> ImbPipeline:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    if mode == "binned":
        num_cols = DISCRETE_NUM_COLS + ORDINAL_ENCODED_COLS + STATIC_FLAG_COLS
        cat_cols = BINNED_CAT_COLS + NOMINAL_CLEAN_COLS
        cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
        encode_scale = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", ohe, cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return ImbPipeline(
            steps=[
                ("static_clean", StaticRuleCleaner()),
                ("quantile_bin", QuantileFeatureBinner(cols=SKEWED_CONTINUOUS_COLS, n_bins=4)),
                ("select_cols", ColumnSelector(num_cols=num_cols, cat_cols=cat_cols)),
                (
                    "smotenc",
                    SMOTENC(
                        categorical_features=cat_indices,
                        sampling_strategy=sampling_strategy,
                        random_state=random_state,
                    ),
                ),
                ("encode_scale", encode_scale),
                ("model", estimator),
            ]
        )

    if mode == "continuous":
        cont_cols = SKEWED_CONTINUOUS_COLS + ["age"]
        num_cols = cont_cols + DISCRETE_NUM_COLS + ORDINAL_ENCODED_COLS + STATIC_FLAG_COLS
        cat_cols = NOMINAL_CLEAN_COLS
        cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
        encode_scale = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", ohe, cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return ImbPipeline(
            steps=[
                ("static_clean", StaticRuleCleaner()),
                (
                    "clip_log",
                    PercentileClipLogTransformer(
                        cols=cont_cols, lower_q=0.01, upper_q=0.99, apply_log1p=True
                    ),
                ),
                ("select_cols", ColumnSelector(num_cols=num_cols, cat_cols=cat_cols)),
                (
                    "smotenc",
                    SMOTENC(
                        categorical_features=cat_indices,
                        sampling_strategy=sampling_strategy,
                        random_state=random_state,
                    ),
                ),
                ("encode_scale", encode_scale),
                ("model", estimator),
            ]
        )

    raise ValueError(f"Unsupported mode: {mode}")

