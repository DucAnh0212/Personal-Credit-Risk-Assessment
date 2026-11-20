import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.clean_data import build_preprocessor, get_feature_names
from src.config import RANDOM_STATE
from src.imbalance import resolve_class_weight, sample_weights_from_class_weight


class CostWeightedGradientBoosting(GradientBoostingClassifier):
    def __init__(
        self,
        *,
        loss: str = "log_loss",
        learning_rate: float = 0.05,
        n_estimators: int = 100,
        subsample: float = 0.8,
        max_depth: int = 2,
        min_samples_leaf: int = 15,
        class_weight=None,
        random_state: int = RANDOM_STATE,
    ):
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        self.class_weight = class_weight

    def fit(self, X, y, sample_weight=None, monitor=None):
        if sample_weight is None and self.class_weight is not None:
            sample_weight = sample_weights_from_class_weight(y, self.class_weight)
        return super().fit(X, y, sample_weight=sample_weight, monitor=monitor)


def build_gradient_boosting(
    learning_rate: float = 0.05,
    n_estimators: int = 100,
    max_depth: int = 2,
    subsample: float = 0.8,
    min_samples_leaf: int = 15,
    class_weight_mode: str = "cost_5_1",
    random_state: int = RANDOM_STATE,
) -> CostWeightedGradientBoosting:
    return CostWeightedGradientBoosting(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_leaf=min_samples_leaf,
        class_weight=resolve_class_weight(class_weight_mode),
        random_state=random_state,
    )


def build_gbdt_pipeline(
    feature_mode: str = "binned",
    class_weight_mode: str = "cost_5_1",
    learning_rate: float = 0.05,
    n_estimators: int = 100,
    max_depth: int = 2,
    subsample: float = 0.8,
    min_samples_leaf: int = 15,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(mode=feature_mode, scale_numeric=False)),
            (
                "model",
                build_gradient_boosting(
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    subsample=subsample,
                    min_samples_leaf=min_samples_leaf,
                    class_weight_mode=class_weight_mode,
                ),
            ),
        ]
    )


def get_gbdt_param_grid() -> dict:
    return {
        "model__n_estimators": [60, 100, 150],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3],
        "model__subsample": [0.8, 1.0],
        "model__min_samples_leaf": [10, 20],
    }


def extract_gbdt_gain_importance(fitted_pipeline: Pipeline) -> pd.DataFrame:
    names = get_feature_names(fitted_pipeline.named_steps["preprocess"])
    importances = fitted_pipeline.named_steps["model"].feature_importances_
    df = pd.DataFrame(
        {
            "feature": names,
            "gain_importance": np.round(importances, 5),
        }
    )
    return df.sort_values("gain_importance", ascending=False).reset_index(drop=True)

