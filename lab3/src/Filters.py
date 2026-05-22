from __future__ import annotations

from collections.abc import Callable

import numpy as np


ArrayLike1D = np.ndarray | list[float] | tuple[float, ...]


def _as_signal(values: ArrayLike1D) -> np.ndarray:
    signal = np.asarray(values, dtype=float)
    if signal.ndim != 1:
        raise ValueError()
    if signal.size and not np.isfinite(signal).all():
        raise ValueError()
    return signal.copy()


def moving_average(values: ArrayLike1D, window: int = 5) -> np.ndarray:
    signal = _as_signal(values)
    if window < 1:
        raise ValueError()
    if signal.size == 0 or window == 1:
        return signal

    left = window // 2
    right = window - left - 1
    padded = np.pad(signal, (left, right), mode="edge")
    kernel = np.full(window, 1.0 / window)
    return np.convolve(padded, kernel, mode="valid")


def kalman_filter_1d(
    values: ArrayLike1D,
    process_variance: float | None = None,
    measurement_variance: float | None = None,
    initial_error_covariance: float = 1.0,
) -> np.ndarray:
    signal = _as_signal(values)
    if signal.size == 0:
        return signal

    first_diff_variance = float(np.var(np.diff(signal))) if signal.size > 1 else 1.0
    signal_variance = float(np.var(signal)) if signal.size > 1 else 1.0
    estimated_measurement_variance = max(first_diff_variance / 2.0, signal_variance * 1e-6, 1e-12)
    r = estimated_measurement_variance if measurement_variance is None else float(measurement_variance)
    q = max(r * 0.02, 1e-12) if process_variance is None else float(process_variance)
    if q <= 0 or r <= 0 or initial_error_covariance <= 0:
        raise ValueError()

    estimate = float(signal[0])
    covariance = float(initial_error_covariance)
    filtered = np.empty_like(signal)
    filtered[0] = estimate

    for index in range(1, signal.size):
        covariance += q
        gain = covariance / (covariance + r)
        estimate += gain * (signal[index] - estimate)
        covariance *= 1.0 - gain
        filtered[index] = estimate

    return filtered


def savitzky_golay_filter(values: ArrayLike1D, window: int = 7, polyorder: int = 2) -> np.ndarray:
    signal = _as_signal(values)
    if window < 3 or window % 2 == 0:
        raise ValueError()
    if polyorder < 0 or polyorder >= window:
        raise ValueError()
    if signal.size == 0:
        return signal

    half_window = window // 2
    offsets = np.arange(-half_window, half_window + 1, dtype=float)
    design = np.vander(offsets, N=polyorder + 1, increasing=True)
    weights = np.linalg.pinv(design)[0]
    padded = np.pad(signal, (half_window, half_window), mode="edge")
    return np.array([np.dot(weights, padded[i : i + window]) for i in range(signal.size)])


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def haar_wavelet_denoise(
    values: ArrayLike1D,
    level: int | None = None,
    threshold_scale: float = 1.0,
) -> np.ndarray:
    signal = _as_signal(values)
    if signal.size < 2:
        return signal
    if threshold_scale < 0:
        raise ValueError()

    padded_size = 1 << int(np.ceil(np.log2(signal.size)))
    approximation = np.pad(signal, (0, padded_size - signal.size), mode="edge")
    max_level = int(np.log2(padded_size))
    selected_level = max_level if level is None else min(int(level), max_level)
    if selected_level < 1:
        return signal

    details: list[np.ndarray] = []
    sqrt_two = np.sqrt(2.0)
    for _ in range(selected_level):
        even = approximation[0::2]
        odd = approximation[1::2]
        details.append((even - odd) / sqrt_two)
        approximation = (even + odd) / sqrt_two

    finest_detail = details[0]
    sigma = np.median(np.abs(finest_detail - np.median(finest_detail))) / 0.6745
    threshold = float(threshold_scale * sigma * np.sqrt(2.0 * np.log(padded_size)))
    denoised_details = [_soft_threshold(detail, threshold) for detail in details]

    reconstructed = approximation
    for detail in reversed(denoised_details):
        restored = np.empty(detail.size * 2)
        restored[0::2] = (reconstructed + detail) / sqrt_two
        restored[1::2] = (reconstructed - detail) / sqrt_two
        reconstructed = restored

    return reconstructed[: signal.size]


def lms_predictive_filter(
    values: ArrayLike1D,
    order: int = 4,
    step_size: float = 0.2,
    normalized: bool = True,
) -> np.ndarray:
    signal = _as_signal(values)
    if order < 1:
        raise ValueError()
    if step_size <= 0:
        raise ValueError()
    if signal.size <= order:
        return signal

    weights = np.full(order, 1.0 / order)
    filtered = signal.copy()
    eps = 1e-12

    for index in range(order, signal.size):
        regressors = signal[index - order : index][::-1]
        prediction = float(np.dot(weights, regressors))
        error = signal[index] - prediction
        denominator = float(np.dot(regressors, regressors)) + eps if normalized else 1.0
        weights += step_size * error * regressors / denominator
        filtered[index] = prediction

    return filtered


FilterFn = Callable[[ArrayLike1D], np.ndarray]

FILTERS: dict[str, FilterFn] = {
    "moving_average": moving_average,
    "kalman": kalman_filter_1d,
    "savitzky_golay": savitzky_golay_filter,
    "haar_wavelet": haar_wavelet_denoise,
    "lms": lms_predictive_filter,
}


def apply_filters(values: ArrayLike1D) -> dict[str, np.ndarray]:
    signal = _as_signal(values)
    return {"raw": signal, **{name: method(signal) for name, method in FILTERS.items()}}
