import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.clean_data import build_preprocessor, get_feature_names
from src.config import RANDOM_STATE
from src.imbalance import resolve_class_weight


def build_logistic_regression(
    C: float = 0.5,
    l1_ratio: float = 0.0,
    class_weight_mode: str = "cost_5_1",
    max_iter: int = 1000,
    random_state: int = RANDOM_STATE,
) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        l1_ratio=l1_ratio,
        solver="liblinear",
        class_weight=resolve_class_weight(class_weight_mode),
        max_iter=max_iter,
        random_state=random_state,
    )


def build_logistic_pipeline(
    feature_mode: str = "binned",
    class_weight_mode: str = "cost_5_1",
    C: float = 0.5,
    l1_ratio: float = 0.0,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(mode=feature_mode, scale_numeric=True)),
            (
                "model",
                build_logistic_regression(
                    C=C,
                    l1_ratio=l1_ratio,
                    class_weight_mode=class_weight_mode,
                ),
            ),
        ]
    )


def get_logistic_param_grid() -> dict:
    return {
        "model__C": [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
        "model__l1_ratio": [0.0, 1.0],
    }


def extract_odds_ratios(fitted_pipeline: Pipeline) -> pd.DataFrame:
    names = get_feature_names(fitted_pipeline.named_steps["preprocess"])
    coefs = fitted_pipeline.named_steps["model"].coef_.ravel()
    df = pd.DataFrame(
        {
            "feature": names,
            "coef_log_odds": np.round(coefs, 4),
            "odds_ratio": np.round(np.exp(coefs), 4),
            "abs_coef": np.round(np.abs(coefs), 4),
            "direction": np.where(coefs >= 0, "Increases Bad Risk (+)", "Decreases Bad Risk (-)"),
        }
    )
    return df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
