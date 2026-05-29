from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def kalman_random_walk(
    measurements: ArrayLike,
    process_variance: float = 1.0,
    measurement_variance: float = 16.0,
    initial_estimate: float | None = None,
    initial_error: float = 1.0,
) -> NDArray[np.float64]:
    """Apply a scalar Kalman filter with a random-walk state model.

    The model assumes that the hidden true value changes gradually:
    ``x[k] = x[k - 1] + w[k]``, while observations are noisy measurements
    ``z[k] = x[k] + v[k]``.

    Args:
        measurements: Observed one-dimensional time series.
        process_variance: Variance of the state evolution noise ``w``.
        measurement_variance: Variance of the observation noise ``v``.
        initial_estimate: First state estimate. Defaults to the first observed
            value.
        initial_error: Initial estimation error variance.

    Returns:
        Filtered state estimates with the same length as ``measurements``.
    """
    if process_variance <= 0:
        raise ValueError("process_variance must be positive")
    if measurement_variance <= 0:
        raise ValueError("measurement_variance must be positive")
    if initial_error <= 0:
        raise ValueError("initial_error must be positive")

    observed = np.asarray(measurements, dtype=float)
    if observed.ndim != 1:
        raise ValueError("measurements must be a one-dimensional array")
    if len(observed) == 0:
        return observed.copy()

    estimate = float(observed[0] if initial_estimate is None else initial_estimate)
    error = float(initial_error)
    estimates = np.empty_like(observed, dtype=float)

    for index, measurement in enumerate(observed):
        predicted_estimate = estimate
        predicted_error = error + process_variance

        kalman_gain = predicted_error / (predicted_error + measurement_variance)
        estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
        error = (1.0 - kalman_gain) * predicted_error

        estimates[index] = estimate

    return estimates
