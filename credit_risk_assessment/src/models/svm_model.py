from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.clean_data import build_preprocessor
from src.config import RANDOM_STATE
from src.imbalance import resolve_class_weight


def build_svm(
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str | float = "scale",
    class_weight_mode: str = "cost_5_1",
    random_state: int = RANDOM_STATE,
) -> CalibratedClassifierCV:
    base_svc = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight=resolve_class_weight(class_weight_mode),
        random_state=random_state,
    )
    return CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=5, ensemble=False)


def build_svm_pipeline(
    feature_mode: str = "binned",
    class_weight_mode: str = "cost_5_1",
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str | float = "scale",
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(mode=feature_mode, scale_numeric=True)),
            (
                "model",
                build_svm(
                    kernel=kernel,
                    C=C,
                    gamma=gamma,
                    class_weight_mode=class_weight_mode,
                ),
            ),
        ]
    )


def get_svm_param_grid() -> list[dict]:
    return [
        {
            "model__estimator__kernel": ["rbf"],
            "model__estimator__C": [0.25, 0.5, 1.0, 2.0, 5.0],
            "model__estimator__gamma": ["scale", 0.01, 0.03],
        },
        {
            "model__estimator__kernel": ["linear"],
            "model__estimator__C": [0.05, 0.1, 0.5, 1.0, 2.0],
        },
    ]
