"""From-scratch training and evaluation helpers.

These implementations intentionally avoid scikit-learn estimators so the
production pipeline keeps the same learning-from-first-principles spirit as the
original notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd

from beijing_aqi.aqi import available_categories
from beijing_aqi.features import classification_dataset, regression_dataset


@dataclass(frozen=True)
class TrainingConfig:
    test_size: float = 0.30
    random_state: int = 42
    logistic_max_iter: int = 300
    logistic_learning_rate: float = 0.05
    logistic_batch_size: int = 10000
    regularization: float = 1e-4


def train_all_models(
    frame: pd.DataFrame,
    output_dir: str | Path,
    config: TrainingConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Train custom regression and classification models and persist artifacts."""
    config = config or TrainingConfig()
    output_path = Path(output_dir)
    model_path = output_path / "models"
    model_path.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
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
    X_train, X_test, y_train, y_test = _split_frame(X, y, config)
    X_train_scaled, X_test_scaled, mean, std = _standardize(X_train, X_test)

    X_train_design = _add_bias(X_train_scaled)
    X_test_design = _add_bias(X_test_scaled)

    weights = np.linalg.pinv(X_train_design.T @ X_train_design) @ X_train_design.T @ y_train
    predictions = X_test_design @ weights

    artifact = {
        "model_type": "custom_linear_regression_closed_form",
        "features": list(X.columns),
        "mean": mean,
        "std": std,
        "weights": weights,
    }
    joblib.dump(artifact, Path(model_dir) / "linear_regression.joblib")

    return _regression_metrics(y_test, predictions)


def train_logistic_regression(
    frame: pd.DataFrame,
    model_dir: str | Path,
    config: TrainingConfig,
) -> dict[str, float]:
    X, y = classification_dataset(frame)
    class_labels = _ordered_labels(y)
    class_to_index = {label: index for index, label in enumerate(class_labels)}
    y_encoded = y.map(class_to_index).to_numpy()

    X_train, X_test, y_train, y_test = _split_arrays(
        X.to_numpy(dtype=float),
        y_encoded,
        config,
        stratify=y_encoded,
    )
    X_train_scaled, X_test_scaled, mean, std = _standardize_arrays(X_train, X_test)
    X_train_design = _add_bias(X_train_scaled)
    X_test_design = _add_bias(X_test_scaled)

    weights = _fit_softmax_regression(
        X_train_design,
        y_train,
        class_count=len(class_labels),
        config=config,
    )
    probabilities = _softmax(X_test_design @ weights)
    predictions = probabilities.argmax(axis=1)

    artifact = {
        "model_type": "custom_softmax_logistic_regression",
        "features": list(X.columns),
        "classes": class_labels,
        "mean": mean,
        "std": std,
        "weights": weights,
    }
    joblib.dump(artifact, Path(model_dir) / "logistic_regression.joblib")

    return _classification_metrics(y_test, predictions, class_count=len(class_labels))


def train_naive_bayes(
    frame: pd.DataFrame,
    model_dir: str | Path,
    config: TrainingConfig,
) -> dict[str, float]:
    X, y = classification_dataset(frame)
    class_labels = _ordered_labels(y)
    class_to_index = {label: index for index, label in enumerate(class_labels)}
    y_encoded = y.map(class_to_index).to_numpy()

    X_train, X_test, y_train, y_test = _split_arrays(
        X.to_numpy(dtype=float),
        y_encoded,
        config,
        stratify=y_encoded,
    )
    X_train_scaled, X_test_scaled, mean, std = _standardize_arrays(X_train, X_test)

    class_means, class_variances, priors = _fit_gaussian_naive_bayes(
        X_train_scaled,
        y_train,
        class_count=len(class_labels),
    )
    predictions = _predict_gaussian_naive_bayes(
        X_test_scaled,
        class_means,
        class_variances,
        priors,
    )

    artifact = {
        "model_type": "custom_gaussian_naive_bayes",
        "features": list(X.columns),
        "classes": class_labels,
        "mean": mean,
        "std": std,
        "class_means": class_means,
        "class_variances": class_variances,
        "priors": priors,
    }
    joblib.dump(artifact, Path(model_dir) / "naive_bayes.joblib")

    return _classification_metrics(y_test, predictions, class_count=len(class_labels))


def _fit_softmax_regression(
    X: np.ndarray,
    y: np.ndarray,
    class_count: int,
    config: TrainingConfig,
) -> np.ndarray:
    rng = np.random.default_rng(config.random_state)
    weights = np.zeros((X.shape[1], class_count), dtype=float)
    y_one_hot = np.eye(class_count)[y]
    batch_size = min(config.logistic_batch_size, X.shape[0])

    for _ in range(config.logistic_max_iter):
        batch_indices = rng.choice(X.shape[0], size=batch_size, replace=False)
        X_batch = X[batch_indices]
        y_batch = y_one_hot[batch_indices]

        probabilities = _softmax(X_batch @ weights)
        gradient = X_batch.T @ (probabilities - y_batch) / batch_size
        gradient[1:] += config.regularization * weights[1:]
        weights -= config.logistic_learning_rate * gradient

    return weights


def _fit_gaussian_naive_bayes(
    X: np.ndarray,
    y: np.ndarray,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_count = X.shape[1]
    class_means = np.zeros((class_count, feature_count), dtype=float)
    class_variances = np.zeros((class_count, feature_count), dtype=float)
    priors = np.zeros(class_count, dtype=float)

    for class_index in range(class_count):
        class_rows = X[y == class_index]
        if len(class_rows) == 0:
            class_variances[class_index] = 1.0
            continue
        class_means[class_index] = class_rows.mean(axis=0)
        class_variances[class_index] = class_rows.var(axis=0) + 1e-9
        priors[class_index] = len(class_rows) / len(X)

    priors = np.where(priors == 0, 1e-12, priors)
    return class_means, class_variances, priors


def _predict_gaussian_naive_bayes(
    X: np.ndarray,
    class_means: np.ndarray,
    class_variances: np.ndarray,
    priors: np.ndarray,
) -> np.ndarray:
    log_priors = np.log(priors)
    log_likelihoods = []

    for class_index in range(len(priors)):
        variance = class_variances[class_index]
        mean = class_means[class_index]
        log_density = -0.5 * (
            np.log(2 * np.pi * variance) + ((X - mean) ** 2 / variance)
        )
        log_likelihoods.append(log_priors[class_index] + log_density.sum(axis=1))

    return np.vstack(log_likelihoods).T.argmax(axis=1)


def _split_frame(
    X: pd.DataFrame,
    y: pd.Series,
    config: TrainingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = _split_arrays(
        X.to_numpy(dtype=float),
        y.to_numpy(dtype=float),
        config,
    )
    return X_train, X_test, y_train, y_test


def _split_arrays(
    X: np.ndarray,
    y: np.ndarray,
    config: TrainingConfig,
    stratify: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.random_state)
    test_indices = _test_indices(y, config.test_size, rng, stratify)
    test_mask = np.zeros(len(y), dtype=bool)
    test_mask[test_indices] = True

    return X[~test_mask], X[test_mask], y[~test_mask], y[test_mask]


def _test_indices(
    y: np.ndarray,
    test_size: float,
    rng: np.random.Generator,
    stratify: np.ndarray | None,
) -> np.ndarray:
    if stratify is None:
        indices = rng.permutation(len(y))
        test_count = max(1, int(round(len(y) * test_size)))
        return indices[:test_count]

    test_indices = []
    for class_value in np.unique(stratify):
        class_indices = np.flatnonzero(stratify == class_value)
        shuffled = rng.permutation(class_indices)
        class_test_count = max(1, int(round(len(class_indices) * test_size)))
        test_indices.extend(shuffled[:class_test_count])
    return np.array(test_indices, dtype=int)


def _standardize(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _standardize_arrays(X_train, X_test)


def _standardize_arrays(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def _add_bias(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X])


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def _ordered_labels(y: pd.Series) -> list[str]:
    present = set(y.unique())
    ordered = [label for label in available_categories() if label in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_true - y_pred
    mse = np.mean(residuals**2)
    mae = np.mean(np.abs(residuals))
    total_sum_squares = np.sum((y_true - y_true.mean()) ** 2)
    residual_sum_squares = np.sum(residuals**2)
    r2 = 1 - (residual_sum_squares / total_sum_squares)
    return {
        "mae": float(mae),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2),
    }


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_count: int,
) -> dict[str, float]:
    accuracy = np.mean(y_true == y_pred)
    precisions = []
    recalls = []
    f1_scores = []

    for class_index in range(class_count):
        true_positive = np.sum((y_true == class_index) & (y_pred == class_index))
        false_positive = np.sum((y_true != class_index) & (y_pred == class_index))
        false_negative = np.sum((y_true == class_index) & (y_pred != class_index))

        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
