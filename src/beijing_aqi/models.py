"""Training and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from beijing_aqi.features import classification_dataset, regression_dataset


@dataclass(frozen=True)
class TrainingConfig:
    test_size: float = 0.30
    random_state: int = 42
    logistic_max_iter: int = 1000


def train_all_models(
    frame: pd.DataFrame,
    output_dir: str | Path,
    config: TrainingConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Train regression and classification baselines and persist artifacts."""
    config = config or TrainingConfig()
    output_path = Path(output_dir)
    model_path = output_path / "models"
    model_path.mkdir(parents=True, exist_ok=True)

    metrics = {
        "linear_regression": train_linear_regression(frame, model_path, config),
        "logistic_regression": train_logistic_regression(frame, model_path, config),
        "naive_bayes": train_naive_bayes(frame, model_path, config),
    }
    return metrics


def train_linear_regression(
    frame: pd.DataFrame,
    model_dir: str | Path,
    config: TrainingConfig,
) -> dict[str, float]:
    X, y = regression_dataset(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    predictions = _fit_and_predict(model, X_train, y_train, X_test)

    model_dir = Path(model_dir)
    joblib.dump(model, model_dir / "linear_regression.joblib")

    mse = mean_squared_error(y_test, predictions)
    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_test, predictions)),
    }


def train_logistic_regression(
    frame: pd.DataFrame,
    model_dir: str | Path,
    config: TrainingConfig,
) -> dict[str, float]:
    X, y = classification_dataset(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=_stratify_if_possible(y),
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=config.logistic_max_iter,
                    random_state=config.random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )
    predictions = _fit_and_predict(model, X_train, y_train, X_test)

    model_dir = Path(model_dir)
    joblib.dump(model, model_dir / "logistic_regression.joblib")
    return _classification_metrics(y_test, predictions)


def train_naive_bayes(
    frame: pd.DataFrame,
    model_dir: str | Path,
    config: TrainingConfig,
) -> dict[str, float]:
    X, y = classification_dataset(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=_stratify_if_possible(y),
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", GaussianNB()),
        ]
    )
    predictions = _fit_and_predict(model, X_train, y_train, X_test)

    model_dir = Path(model_dir)
    joblib.dump(model, model_dir / "naive_bayes.joblib")
    return _classification_metrics(y_test, predictions)


def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _stratify_if_possible(y: pd.Series) -> pd.Series | None:
    class_counts = y.value_counts()
    if len(class_counts) > 1 and class_counts.min() >= 2:
        return y
    return None


def _fit_and_predict(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        model.fit(X_train, y_train)
        return model.predict(X_test)
