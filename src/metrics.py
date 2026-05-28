from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def rmse(original: ArrayLike, filtered: ArrayLike) -> float:
    """Calculate root mean squared difference between two equal-length series."""
    source = np.asarray(original, dtype=float)
    result = np.asarray(filtered, dtype=float)
    _validate_same_shape(source, result)
    return float(np.sqrt(np.mean((source - result) ** 2)))


def smoothness(values: ArrayLike) -> float:
    """Estimate smoothness as the standard deviation of first differences."""
    series = np.asarray(values, dtype=float)
    if series.ndim != 1:
        raise ValueError("values must be a one-dimensional array")
    if len(series) < 2:
        return 0.0
    return float(np.std(np.diff(series), ddof=0))


def filtering_summary(original: ArrayLike, filtered_series: dict[str, ArrayLike]) -> pd.DataFrame:
    """Build a compact comparison table for several filtered series."""
    rows = []
    original_smoothness = smoothness(original)

    for method_name, filtered in filtered_series.items():
        filtered_smoothness = smoothness(filtered)
        rows.append(
            {
                "method": method_name,
                "rmse_to_original": rmse(original, filtered),
                "smoothness": filtered_smoothness,
                "smoothness_ratio": _safe_ratio(filtered_smoothness, original_smoothness),
            }
        )

    return pd.DataFrame(rows)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _validate_same_shape(left: np.ndarray, right: np.ndarray) -> None:
    if left.shape != right.shape:
        raise ValueError("series must have the same shape")
