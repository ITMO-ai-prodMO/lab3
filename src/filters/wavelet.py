from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class HaarDecomposition:
    """Approximation and detail coefficients for a Haar wavelet decomposition."""

    approximation: NDArray[np.float64]
    details: list[NDArray[np.float64]]
    original_length: int
    padded_length: int


def haar_wavelet_denoise(
    values: ArrayLike,
    levels: int | None = None,
    threshold: float | None = None,
    threshold_mode: str = "soft",
) -> NDArray[np.float64]:
    """Denoise a one-dimensional series with a manually implemented Haar wavelet.

    The series is padded to the next power of two, decomposed into Haar
    approximation/detail coefficients, small detail coefficients are suppressed,
    and the signal is reconstructed back to the original length.

    Args:
        values: Source numeric time series.
        levels: Number of decomposition levels. Defaults to the maximum allowed.
        threshold: Detail coefficient threshold. Defaults to a robust universal
            threshold estimated from the finest detail level.
        threshold_mode: Either ``"soft"`` or ``"hard"`` thresholding.

    Returns:
        Denoised series with the same length as ``values``.
    """
    decomposition = haar_decompose(values, levels=levels)
    if not decomposition.details:
        return decomposition.approximation[: decomposition.original_length].copy()

    used_threshold = (
        _universal_threshold(decomposition.details[0], decomposition.original_length)
        if threshold is None
        else float(threshold)
    )
    filtered_details = [
        _threshold_coefficients(detail, used_threshold, threshold_mode)
        for detail in decomposition.details
    ]

    reconstructed = haar_reconstruct(
        HaarDecomposition(
            approximation=decomposition.approximation,
            details=filtered_details,
            original_length=decomposition.original_length,
            padded_length=decomposition.padded_length,
        )
    )
    return reconstructed[: decomposition.original_length]


def haar_decompose(values: ArrayLike, levels: int | None = None) -> HaarDecomposition:
    """Decompose a one-dimensional series into Haar wavelet coefficients."""
    series = np.asarray(values, dtype=float)
    if series.ndim != 1:
        raise ValueError("values must be a one-dimensional array")
    if len(series) == 0:
        return HaarDecomposition(series.copy(), [], 0, 0)

    padded = _pad_to_power_of_two(series)
    max_levels = int(math.log2(len(padded)))
    if levels is None:
        levels = max_levels
    if not 0 <= levels <= max_levels:
        raise ValueError(f"levels must be in range 0..{max_levels}")

    approximation = padded.copy()
    details: list[NDArray[np.float64]] = []

    sqrt_two = math.sqrt(2.0)
    for _ in range(levels):
        even = approximation[0::2]
        odd = approximation[1::2]
        next_approximation = (even + odd) / sqrt_two
        detail = (even - odd) / sqrt_two
        details.append(detail)
        approximation = next_approximation

    return HaarDecomposition(
        approximation=approximation,
        details=details,
        original_length=len(series),
        padded_length=len(padded),
    )


def haar_reconstruct(decomposition: HaarDecomposition) -> NDArray[np.float64]:
    """Reconstruct a series from Haar wavelet coefficients."""
    approximation = decomposition.approximation.copy()
    sqrt_two = math.sqrt(2.0)

    for detail in reversed(decomposition.details):
        if len(detail) != len(approximation):
            raise ValueError("detail and approximation lengths are inconsistent")

        restored = np.empty(len(approximation) * 2, dtype=float)
        restored[0::2] = (approximation + detail) / sqrt_two
        restored[1::2] = (approximation - detail) / sqrt_two
        approximation = restored

    return approximation[: decomposition.padded_length]


def _pad_to_power_of_two(series: NDArray[np.float64]) -> NDArray[np.float64]:
    target_length = 1 << (len(series) - 1).bit_length()
    if target_length == len(series):
        return series.copy()
    return np.pad(series, (0, target_length - len(series)), mode="edge")


def _universal_threshold(detail: NDArray[np.float64], length: int) -> float:
    median_abs_deviation = np.median(np.abs(detail - np.median(detail)))
    sigma = median_abs_deviation / 0.6745 if median_abs_deviation > 0 else np.std(detail)
    if sigma == 0:
        return 0.0
    return float(sigma * math.sqrt(2.0 * math.log(max(length, 2))))


def _threshold_coefficients(
    coefficients: NDArray[np.float64],
    threshold: float,
    mode: str,
) -> NDArray[np.float64]:
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if mode == "hard":
        return np.where(np.abs(coefficients) >= threshold, coefficients, 0.0)
    if mode == "soft":
        return np.sign(coefficients) * np.maximum(np.abs(coefficients) - threshold, 0.0)
    raise ValueError("threshold_mode must be either 'soft' or 'hard'")
