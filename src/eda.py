"""Описательная статистика, тренд, стационарность и визуализация временных рядов."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.tsa.stattools import adfuller, kpss


# ---------------------------------------------------------------------------
# 1. Descriptive statistics
# ---------------------------------------------------------------------------

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Таблица mean/std/min/q25/median/q75/max/skewness/kurtosis(Fisher)/CV по всем колонкам."""
    result = pd.DataFrame(index=df.columns)
    result["mean"] = df.mean()
    result["std"] = df.std()
    result["min"] = df.min()
    result["q25"] = df.quantile(0.25)
    result["median"] = df.median()
    result["q75"] = df.quantile(0.75)
    result["max"] = df.max()
    result["skewness"] = df.skew()
    result["kurtosis"] = df.apply(lambda s: sp_stats.kurtosis(s, fisher=True))
    result["CV"] = result["std"] / result["mean"].abs()
    return result


# ---------------------------------------------------------------------------
# 2. Trend analysis
# ---------------------------------------------------------------------------

def trend_analysis(series: pd.Series, name: str | None = None) -> dict:
    """Линейный тренд (linregress) + тест Манна-Кендалла (kendalltau) для одного ряда."""
    s = series.dropna().values
    n = len(s)
    t = np.arange(n)

    slope, intercept, r, p_lr, se = sp_stats.linregress(t, s)

    tau, p_mk = sp_stats.kendalltau(t, s)

    return {
        "name": name or (series.name if series.name is not None else "series"),
        "n": n,
        # Linear regression
        "lr_slope": slope,
        "lr_intercept": intercept,
        "lr_r": r,
        "lr_r2": r ** 2,
        "lr_p": p_lr,
        "lr_se": se,
        "lr_significant": bool(p_lr < 0.05),
        # Mann-Kendall via kendalltau
        "mk_tau": tau,
        "mk_p": p_mk,
        "mk_significant": bool(p_mk < 0.05),
        "mk_direction": "increasing" if tau > 0 else ("decreasing" if tau < 0 else "none"),
    }


# ---------------------------------------------------------------------------
# 3. Stationarity tests
# ---------------------------------------------------------------------------

_ADF_KPSS_VERDICT = {
    # (adf_rejects, kpss_rejects)
    # ADF H0=unit-root; KPSS H0=stationary
    (True,  False): "stationary",
    (False, True):  "non-stationary",
    (True,  True):  "contradictory",
    (False, False): "inconclusive",
}


def stationarity_tests(series: pd.Series) -> dict:
    """ADF + KPSS тесты; совместный вердикт из 4-клеточной матрицы ADF×KPSS."""
    s = series.dropna()

    # ADF: H0 = unit root (non-stationary)
    adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, adf_icbest = adfuller(s, autolag="AIC")
    adf_rejects = bool(adf_p < 0.05)

    # KPSS: H0 = stationary
    kpss_warning = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(s, regression="c", nlags="auto")
        for w in caught:
            if issubclass(w.category, UserWarning):
                kpss_warning = str(w.message)
    kpss_rejects = bool(kpss_p < 0.05)

    verdict = _ADF_KPSS_VERDICT[(adf_rejects, kpss_rejects)]

    return {
        "adf_stat": adf_stat,
        "adf_p": adf_p,
        "adf_lags": adf_lags,
        "adf_crit": adf_crit,
        "adf_rejects_H0": adf_rejects,
        "kpss_stat": kpss_stat,
        "kpss_p": kpss_p,
        "kpss_lags": kpss_lags,
        "kpss_crit": kpss_crit,
        "kpss_rejects_H0": kpss_rejects,
        "kpss_warning": kpss_warning,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# 4. plot_overview
# ---------------------------------------------------------------------------

def plot_overview(df: pd.DataFrame, save_path: str | Path) -> None:
    """6 субплотов: серия + скользящее среднее (окно 7) + заштрихованная полоса rolling mean ± 2·rolling std."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cols = list(df.columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), tight_layout=True)
    fig.suptitle("Обзор: ряд и скользящее среднее с полосой +-2*ско (окно=7)", fontsize=13)

    for ax, col in zip(axes.flat, cols):
        s = df[col]
        roll = s.rolling(7)
        rm = roll.mean()
        rs = roll.std()
        ax.plot(s.index, s.values, alpha=0.6, label="сырой ряд", linewidth=1)
        ax.plot(rm.index, rm.values, color="tab:orange", linewidth=1.5, label="скольз. среднее")
        ax.fill_between(
            rm.index,
            (rm - 2 * rs).values,
            (rm + 2 * rs).values,
            alpha=0.25,
            color="tab:orange",
            label="+-2*ско",
        )
        ax.set_title(col)
        ax.set_xlabel("день")
        ax.legend(fontsize=7)

    fig.savefig(save_path, dpi=120)


# ---------------------------------------------------------------------------
# 5. plot_distributions
# ---------------------------------------------------------------------------

def plot_distributions(df: pd.DataFrame, save_path: str | Path) -> None:
    """Гистограммы + KDE (scipy) для каждой из 6 метрик."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cols = list(df.columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), tight_layout=True)
    fig.suptitle("Распределения: гистограмма и ядерная оценка плотности", fontsize=13)

    for ax, col in zip(axes.flat, cols):
        s = df[col].dropna().values
        ax.hist(s, bins=12, density=True, alpha=0.5, color="tab:blue", label="гистограмма")
        x = np.linspace(s.min(), s.max(), 200)
        kde = sp_stats.gaussian_kde(s)
        ax.plot(x, kde(x), color="tab:red", linewidth=2, label="плотность")
        ax.set_title(col)
        ax.legend(fontsize=7)

    fig.savefig(save_path, dpi=120)
