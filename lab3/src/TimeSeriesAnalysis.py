from __future__ import annotations

import numpy as np
import pandas as pd


def autocorrelation(values: np.ndarray | pd.Series, max_lag: int | None = None) -> pd.Series:
    signal = np.asarray(values, dtype=float)
    signal = signal[np.isfinite(signal)]
    if signal.size < 2:
        return pd.Series(dtype=float, name="autocorrelation")

    centered = signal - signal.mean()
    denominator = float(np.dot(centered, centered))
    max_lag = min(signal.size - 1, max_lag or max(1, signal.size // 2))
    if denominator == 0:
        return pd.Series(np.zeros(max_lag + 1), index=np.arange(max_lag + 1), name="autocorrelation")

    values_by_lag = [
        float(np.dot(centered[: signal.size - lag], centered[lag:]) / denominator)
        for lag in range(max_lag + 1)
    ]
    return pd.Series(values_by_lag, index=np.arange(max_lag + 1), name="autocorrelation")


def dominant_period(values: np.ndarray | pd.Series, min_lag: int = 2, max_lag: int | None = None) -> float:
    correlations = autocorrelation(values, max_lag=max_lag)
    candidates = correlations[correlations.index >= min_lag]
    if candidates.empty:
        return float("nan")
    return float(candidates.idxmax())


def linear_trend_slope(values: np.ndarray | pd.Series) -> float:
    signal = np.asarray(values, dtype=float)
    mask = np.isfinite(signal)
    if mask.sum() < 2:
        return float("nan")
    positions = np.arange(signal.size, dtype=float)[mask]
    return float(np.polyfit(positions, signal[mask], deg=1)[0])


def summarize_frame(
    frame: pd.DataFrame,
    min_period_lag: int = 2,
    max_lag: int | None = None,
) -> pd.DataFrame:
    rows = []
    for column in frame.select_dtypes(include="number").columns:
        series = frame[column].dropna()
        rows.append(
            {
                "series": column,
                "count": int(series.size),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
                "trend_slope_per_step": linear_trend_slope(series),
                "lag1_autocorrelation": float(series.autocorr(lag=1)),
                "dominant_period_steps": dominant_period(series, min_lag=min_period_lag, max_lag=max_lag),
            }
        )
    return pd.DataFrame(rows)
