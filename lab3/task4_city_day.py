from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CityDayPairSeries:
    city: str
    dates: list[dt.date]
    a_name: str
    b_name: str
    a: np.ndarray
    b: np.ndarray


def _parse_float(s: str | None) -> float:
    if s is None:
        return float("nan")
    s = s.strip()
    if not s:
        return float("nan")
    return float(s)


def fill_nans_linear(x: Iterable[float]) -> np.ndarray:
    x = np.asarray(list(x), dtype=float)
    if x.size == 0:
        return x
    idx = np.arange(x.size, dtype=float)
    ok = ~np.isnan(x)
    if not np.any(ok):
        return x
    return np.interp(idx, idx[ok], x[ok]).astype(float, copy=False)


def load_city_day_pair(
    path: str | Path,
    *,
    city: str = "Delhi",
    a_col: str = "PM2.5",
    b_col: str = "PM10",
) -> CityDayPairSeries:
    path = Path(path)

    dates: list[dt.date] = []
    a: list[float] = []
    b: list[float] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("City") != city:
                continue
            d = dt.date.fromisoformat(row["Date"])
            dates.append(d)
            a.append(_parse_float(row.get(a_col)))
            b.append(_parse_float(row.get(b_col)))

    order = np.argsort(np.asarray([d.toordinal() for d in dates], dtype=int))
    dates = [dates[i] for i in order]
    a_arr = np.asarray([a[i] for i in order], dtype=float)
    b_arr = np.asarray([b[i] for i in order], dtype=float)

    ords = np.asarray([d.toordinal() for d in dates], dtype=int)

    return CityDayPairSeries(city=city, dates=dates, a_name=a_col, b_name=b_col, a=a_arr, b=b_arr)


def cross_correlation(x: Iterable[float], y: Iterable[float], max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    n = int(min(x.size, y.size))
    x = x[:n] - np.mean(x[:n])
    y = y[:n] - np.mean(y[:n])
    denom = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denom == 0:
        lags = np.arange(-max_lag, max_lag + 1, dtype=int)
        return lags, np.zeros_like(lags, dtype=float)

    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    cc = np.empty_like(lags, dtype=float)
    for i, lag in enumerate(lags):
        if lag < 0:
            cc[i] = float(np.dot(x[: n + lag], y[-lag:]) / denom)
        elif lag > 0:
            cc[i] = float(np.dot(x[lag:], y[: n - lag]) / denom)
        else:
            cc[i] = float(np.dot(x, y) / denom)
    return lags, cc
