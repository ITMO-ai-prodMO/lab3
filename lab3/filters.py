from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Iterable, Literal

import numpy as np


def _as_1d_float_array(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float)
    return arr


def moving_average(x: Iterable[float], window: int) -> np.ndarray:
    x = _as_1d_float_array(x)
    if window == 1:
        return x.copy()

    pad_left = window // 2
    pad_right = window - 1 - pad_left
    xp = np.pad(x, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(xp, kernel, mode="valid")


def exponential_smoothing(x: Iterable[float], alpha: float) -> np.ndarray:
    x = _as_1d_float_array(x)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


@dataclass(frozen=True)
class KalmanParams:
    q: float = 1.0
    r: float = 10.0
    x0: float | None = None
    p0: float = 1.0


def kalman_filter_1d_random_walk(z: Iterable[float], params: KalmanParams) -> np.ndarray:
    z = _as_1d_float_array(z)
    if len(z) == 0:
        return z.copy()

    x = float(z[0] if params.x0 is None else params.x0)
    p = float(params.p0)
    out = np.empty_like(z)

    for i in range(len(z)):
        p = p + params.q
        k = p / (p + params.r)
        x = x + k * (float(z[i]) - x)
        p = (1.0 - k) * p
        out[i] = x

    return out


def savitzky_golay(
    x: Iterable[float],
    window_length: int,
    polyorder: int,
    *,
    deriv: int = 0,
    delta: float = 1.0,
) -> np.ndarray:
    x = _as_1d_float_array(x)
    if len(x) == 0:
        return x.copy()

    half = window_length // 2
    pos = np.arange(-half, half + 1, dtype=float)
    A = np.vander(pos, N=polyorder + 1, increasing=True)
    pinv = np.linalg.pinv(A)
    coeff = pinv[deriv] * factorial(deriv) / (delta ** deriv)

    xp = np.pad(x, (half, half), mode="reflect")
    y = np.convolve(xp, coeff[::-1], mode="valid")
    return y


def _soft_threshold(a: np.ndarray, t: float) -> np.ndarray:
    return np.sign(a) * np.maximum(np.abs(a) - t, 0.0)


def _hard_threshold(a: np.ndarray, t: float) -> np.ndarray:
    out = a.copy()
    out[np.abs(out) < t] = 0.0
    return out


def _haar_dwt_once(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    padded = False
    if len(x) % 2 == 1:
        x = np.append(x, x[-1])
        padded = True
    s2 = np.sqrt(2.0)
    a = (x[0::2] + x[1::2]) / s2
    d = (x[0::2] - x[1::2]) / s2
    return a, d, padded


def _haar_idwt_once(a: np.ndarray, d: np.ndarray, *, padded: bool) -> np.ndarray:
    s2 = np.sqrt(2.0)
    x0 = (a + d) / s2
    x1 = (a - d) / s2
    x = np.empty((len(a) * 2,), dtype=float)
    x[0::2] = x0
    x[1::2] = x1
    if padded:
        return x[:-1]
    return x


def wavelet_denoise_haar(
    x: Iterable[float],
    *,
    level: int | None = None,
    threshold: float | None = None,
    threshold_mode: Literal["soft", "hard"] = "soft",
) -> np.ndarray:
    x = _as_1d_float_array(x)
    n0 = len(x)
    if n0 == 0:
        return x.copy()
    if n0 == 1:
        return x.copy()

    max_level = int(np.floor(np.log2(n0)))
    if level is None:
        level = min(6, max(1, max_level))
    if level > max_level:
        level = max_level

    approx = x.copy()
    details: list[np.ndarray] = []
    padded_flags: list[bool] = []

    for _ in range(level):
        approx, d, padded = _haar_dwt_once(approx)
        details.append(d)
        padded_flags.append(padded)

    if threshold is None:
        finest = details[0]
        mad = np.median(np.abs(finest - np.median(finest)))
        sigma = mad / 0.6745 if mad > 0 else np.std(finest)
        threshold = float(sigma * np.sqrt(2.0 * np.log(len(x) + 1.0)))

    thr_fn = _soft_threshold if threshold_mode == "soft" else _hard_threshold
    details = [thr_fn(d, float(threshold)) for d in details]

    rec = approx
    for d, padded in zip(reversed(details), reversed(padded_flags), strict=True):
        rec = _haar_idwt_once(rec, d, padded=padded)

    return rec[:n0]
