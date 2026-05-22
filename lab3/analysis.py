from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SeriesStats:
    n: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    trend_slope_per_step: float


def series_stats(x: Iterable[float]) -> SeriesStats:
    x = np.asarray(list(x), dtype=float)
    n = int(x.size)
    if n == 0:
        return SeriesStats(
            n=0,
            mean=float("nan"),
            std=float("nan"),
            min=float("nan"),
            max=float("nan"),
            median=float("nan"),
            trend_slope_per_step=float("nan"),
        )

    idx = np.arange(n, dtype=float)
    a, _b = np.polyfit(idx, x, 1)
    return SeriesStats(
        n=n,
        mean=float(np.mean(x)),
        std=float(np.std(x, ddof=1)) if n > 1 else 0.0,
        min=float(np.min(x)),
        max=float(np.max(x)),
        median=float(np.median(x)),
        trend_slope_per_step=float(a),
    )


def autocorrelation(x: Iterable[float], max_lag: int) -> np.ndarray:
    x = np.asarray(list(x), dtype=float)
    n = int(x.size)
    if n == 0:
        return np.zeros((0,), dtype=float)
    if max_lag < 0:
        raise ValueError("max_lag must be >= 0")
    max_lag = min(int(max_lag), n - 1)
    x = x - np.mean(x)
    denom = float(np.dot(x, x))
    if denom == 0:
        out = np.zeros((max_lag + 1,), dtype=float)
        out[0] = 1.0
        return out
    acf = np.empty((max_lag + 1,), dtype=float)
    for lag in range(max_lag + 1):
        acf[lag] = float(np.dot(x[: n - lag], x[lag:]) / denom)
    return acf


def dominant_period_via_fft(x: Iterable[float], *, fs: float = 1.0) -> tuple[float | None, np.ndarray, np.ndarray]:
    x = np.asarray(list(x), dtype=float)
    n = int(x.size)
    if n < 4:
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        power = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
        return None, freqs, power

    x0 = x - np.mean(x)
    fft = np.fft.rfft(x0)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    if power.size <= 1:
        return None, freqs, power
    k = int(np.argmax(power[1:]) + 1)
    f = float(freqs[k])
    if f <= 0:
        return None, freqs, power
    period = fs / f
    return float(period), freqs, power


def correlation_matrix(series: dict[str, Iterable[float]]) -> tuple[list[str], np.ndarray]:
    keys = list(series.keys())
    mat = np.vstack([np.asarray(list(series[k]), dtype=float) for k in keys]).T  # (n, m)
    corr = np.corrcoef(mat, rowvar=False)
    return keys, corr

