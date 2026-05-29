from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def savitzky_golay(
    values: ArrayLike,
    window_size: int,
    poly_order: int,
) -> NDArray[np.float64]:
    """Smooth a series with a manually implemented Savitzky-Golay filter.

    For each point, the filter approximates nearby values with a local
    polynomial and uses the polynomial value at the center of the window.

    Args:
        values: Source numeric time series.
        window_size: Odd number of points in the local window.
        poly_order: Degree of the approximating polynomial.

    Returns:
        Smoothed series with the same length as ``values``.
    """
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    if poly_order < 0:
        raise ValueError("poly_order must be non-negative")
    if poly_order >= window_size:
        raise ValueError("poly_order must be less than window_size")

    series = np.asarray(values, dtype=float)
    if series.ndim != 1:
        raise ValueError("values must be a one-dimensional array")
    if len(series) == 0:
        return series.copy()

    half_window = window_size // 2
    offsets = np.arange(-half_window, half_window + 1, dtype=float)
    design_matrix = np.vander(offsets, N=poly_order + 1, increasing=True)
    smoothing_coefficients = np.linalg.pinv(design_matrix)[0]

    padded = np.pad(series, pad_width=half_window, mode="edge")
    smoothed = np.empty_like(series, dtype=float)

    for index in range(len(series)):
        window = padded[index : index + window_size]
        smoothed[index] = float(smoothing_coefficients @ window)

    return smoothed
