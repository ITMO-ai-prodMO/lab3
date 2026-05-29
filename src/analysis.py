from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def correlation_matrix(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Calculate a Pearson correlation matrix for selected columns."""
    return data[columns].corr(method="pearson")


def autocorrelation(values: ArrayLike, max_lag: int) -> NDArray[np.float64]:
    """Calculate normalized autocorrelation for lags from 0 to ``max_lag``."""
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")

    series = np.asarray(values, dtype=float)
    if series.ndim != 1:
        raise ValueError("values must be a one-dimensional array")
    if len(series) == 0:
        return np.array([], dtype=float)

    max_lag = min(max_lag, len(series) - 1)
    centered = series - np.mean(series)
    variance_sum = float(centered @ centered)
    if variance_sum == 0:
        return np.ones(max_lag + 1, dtype=float)

    coefficients = np.empty(max_lag + 1, dtype=float)
    for lag in range(max_lag + 1):
        coefficients[lag] = float(centered[: len(series) - lag] @ centered[lag:] / variance_sum)

    return coefficients


def dominant_period(values: ArrayLike, max_lag: int = 20) -> tuple[int, float]:
    """Find the strongest positive autocorrelation lag after lag zero."""
    coefficients = autocorrelation(values, max_lag=max_lag)
    if len(coefficients) <= 1:
        return 0, 0.0

    lag = int(np.argmax(coefficients[1:]) + 1)
    return lag, float(coefficients[lag])


def descriptive_statistics(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return transposed descriptive statistics for selected columns."""
    return data[columns].describe().T
