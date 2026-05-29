from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def moving_average(values: ArrayLike, window_size: int) -> NDArray[np.float64]:
    """Smooth a one-dimensional series with a centered moving average.

    Edge values are padded with the nearest observed value, so the output has
    the same length as the input.

    Args:
        values: Source numeric time series.
        window_size: Odd number of points in the averaging window.

    Returns:
        Smoothed series with the same length as ``values``.
    """
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd for centered smoothing")

    series = np.asarray(values, dtype=float)
    if series.ndim != 1:
        raise ValueError("values must be a one-dimensional array")
    if len(series) == 0:
        return series.copy()

    half_window = window_size // 2
    padded = np.pad(series, pad_width=half_window, mode="edge")
    kernel = np.ones(window_size, dtype=float) / window_size
    return np.convolve(padded, kernel, mode="valid")
