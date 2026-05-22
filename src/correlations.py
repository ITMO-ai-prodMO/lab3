"""Анализ корреляций и зависимостей для временных рядов продаж.

Модуль реализует:
- Корреляционные матрицы (Pearson, Spearman) с p-value
- Частные корреляции (partial correlations)
- Кросс-корреляционный анализ с лагами
- Взаимная информация (нелинейные зависимости)
- Кросс-вариантный анализ прибыли
- Визуализации (heatmaps, pairplot, xcorr, lag-scatter)
"""

from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from utils import ALL_METRIC_COLS, PROFIT_COL


# ---------------------------------------------------------------------------
# 1. Корреляционная матрица
# ---------------------------------------------------------------------------

def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Корреляционная матрица между всеми столбцами df.

    Parameters
    ----------
    df : DataFrame с числовыми столбцами
    method : 'pearson' или 'spearman'

    Returns
    -------
    DataFrame shape (n_cols × n_cols)
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")
    return df.corr(method=method)


# ---------------------------------------------------------------------------
# 2. Корреляционная матрица + p-value
# ---------------------------------------------------------------------------

def correlation_with_pvalues(
    df: pd.DataFrame, method: str = "pearson"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Корреляционная матрица и матрица p-value, вычисленные поэлементно.

    Для малых N некоторые scipy-функции могут вернуть NaN — это нормально,
    результат заполняется np.nan без исключений.

    Returns
    -------
    (corr_df, pval_df) — оба shape (n_cols × n_cols)
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")

    cols = list(df.columns)
    n = len(cols)
    corr_arr = np.full((n, n), np.nan)
    pval_arr = np.full((n, n), np.nan)

    stat_fn = stats.pearsonr if method == "pearson" else stats.spearmanr

    for i in range(n):
        for j in range(n):
            xi = df.iloc[:, i].values
            xj = df.iloc[:, j].values
            if i == j:
                corr_arr[i, j] = 1.0
                pval_arr[i, j] = 0.0
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    c, p = stat_fn(xi, xj)
                corr_arr[i, j] = float(c) if np.isfinite(c) else np.nan
                pval_arr[i, j] = float(p) if np.isfinite(p) else np.nan
            except Exception:
                corr_arr[i, j] = np.nan
                pval_arr[i, j] = np.nan

    corr_df = pd.DataFrame(corr_arr, index=cols, columns=cols)
    pval_df = pd.DataFrame(pval_arr, index=cols, columns=cols)
    return corr_df, pval_df


# ---------------------------------------------------------------------------
# 3. Частные корреляции (partial correlation matrix)
# ---------------------------------------------------------------------------

def partial_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Частные корреляции через инверсию ковариационной матрицы.

    Алгоритм:
        inv_C = (pseudo-)inverse of corr(X)
        p_ij  = -inv_C[i,j] / sqrt(inv_C[i,i] * inv_C[j,j])
        p_ii  = 1

    Использует pinv для устойчивости при мультиколлинеарности.
    """
    cols = list(df.columns)
    n = len(cols)
    C = df.corr(method="pearson").values

    cond = np.linalg.cond(C)
    if cond > 1e12:
        warnings.warn(
            f"Correlation matrix is near-singular (cond={cond:.2e}), using pinv.",
            RuntimeWarning,
        )
        inv_C = np.linalg.pinv(C)
    else:
        inv_C = np.linalg.inv(C)

    P = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                P[i, j] = 1.0
            else:
                denom = np.sqrt(inv_C[i, i] * inv_C[j, j])
                if denom > 1e-15:
                    P[i, j] = -inv_C[i, j] / denom
                else:
                    P[i, j] = np.nan

    return pd.DataFrame(P, index=cols, columns=cols)


# ---------------------------------------------------------------------------
# 4. Кросс-корреляция с лагами
# ---------------------------------------------------------------------------

def cross_correlation(
    x: pd.Series, y: pd.Series, max_lag: int = 15
) -> pd.DataFrame:
    """Корреляция (x, y) на лагах [-max_lag .. max_lag].

    Соглашение:
        lag > 0  →  y отстаёт от x: corr(x[:-lag], y[lag:])
        lag == 0 →  corr(x, y)
        lag < 0  →  y опережает x: corr(x[-lag:], y[:lag])  (lag отриц.)

    Returns
    -------
    DataFrame с колонками: lag, corr, p_value
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    records = []

    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            xi, yi = x_arr, y_arr
        elif lag > 0:
            xi, yi = x_arr[:-lag], y_arr[lag:]
        else:  # lag < 0
            xi, yi = x_arr[(-lag):], y_arr[:lag]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if len(xi) < 3:
                    c, p = np.nan, np.nan
                else:
                    c, p = stats.pearsonr(xi, yi)
                    c = float(c) if np.isfinite(c) else np.nan
                    p = float(p) if np.isfinite(p) else np.nan
        except Exception:
            c, p = np.nan, np.nan

        records.append({"lag": lag, "corr": c, "p_value": p})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5. Лучший лаг
# ---------------------------------------------------------------------------

def best_lag(x: pd.Series, y: pd.Series, max_lag: int = 15) -> dict:
    """Лаг с максимальной |corr|.

    Returns
    -------
    dict с ключами: lag, corr, p_value
    """
    xcorr = cross_correlation(x, y, max_lag=max_lag)
    xcorr["abs_corr"] = xcorr["corr"].abs()
    idx = xcorr["abs_corr"].idxmax()
    row = xcorr.loc[idx]
    return {
        "lag": int(row["lag"]),
        "corr": float(row["corr"]) if np.isfinite(row["corr"]) else None,
        "p_value": float(row["p_value"]) if np.isfinite(row["p_value"]) else None,
    }


# ---------------------------------------------------------------------------
# 6. Lead-lag таблица
# ---------------------------------------------------------------------------

def lead_lag_table(df: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
    """Таблица best_lag для всех уникальных пар колонок.

    Colums: pair_a, pair_b, best_lag, corr, p_value, abs_corr

    Интерпретация:
        best_lag > 0 → pair_a «ведёт» pair_b (pair_b реагирует с задержкой)
        best_lag < 0 → pair_b «ведёт» pair_a
        best_lag == 0 → синхронная зависимость
    """
    cols = list(df.columns)
    records = []
    for a, b in itertools.combinations(cols, 2):
        res = best_lag(df[a], df[b], max_lag=max_lag)
        records.append(
            {
                "pair_a": a,
                "pair_b": b,
                "best_lag": res["lag"],
                "corr": res["corr"],
                "p_value": res["p_value"],
                "abs_corr": abs(res["corr"]) if res["corr"] is not None else None,
            }
        )
    return pd.DataFrame(records).sort_values("abs_corr", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. Взаимная информация
# ---------------------------------------------------------------------------

def mutual_information_matrix(df: pd.DataFrame, bins: int = 8) -> pd.DataFrame:
    """Матрица взаимной информации через совместную гистограмму.

    MI(X, Y) = H(X) + H(Y) - H(X, Y)

    Нелинейные зависимости, которые Pearson пропускает.
    """
    cols = list(df.columns)
    n = len(cols)
    mi_arr = np.zeros((n, n))

    def entropy(arr: np.ndarray) -> float:
        counts, _ = np.histogram(arr, bins=bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    def joint_entropy(a: np.ndarray, b: np.ndarray) -> float:
        counts, _, _ = np.histogram2d(a, b, bins=bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    values = [df.iloc[:, i].values.astype(float) for i in range(n)]
    h = [entropy(v) for v in values]

    for i in range(n):
        for j in range(n):
            if i == j:
                mi_arr[i, j] = h[i]
            else:
                hxy = joint_entropy(values[i], values[j])
                mi_arr[i, j] = max(0.0, h[i] + h[j] - hxy)

    return pd.DataFrame(mi_arr, index=cols, columns=cols)


# ---------------------------------------------------------------------------
# 8. Кросс-вариантная корреляция прибыли
# ---------------------------------------------------------------------------

def cross_variant_profit_corr(
    all_variants: dict[int, pd.DataFrame], target_variant: int = 25
) -> pd.Series:
    """Корреляция Pearson прибыли варианта target_variant с прибылью каждого варианта.

    Возвращает Series с индексами 1..60, отсортированная по убыванию.
    """
    target_profit = all_variants[target_variant][PROFIT_COL].values
    corrs = {}
    for v, vdf in all_variants.items():
        other_profit = vdf[PROFIT_COL].values
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c, _ = stats.pearsonr(target_profit, other_profit)
            corrs[v] = float(c) if np.isfinite(c) else np.nan
        except Exception:
            corrs[v] = np.nan
    return pd.Series(corrs).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# 9. Heatmap корреляционной матрицы
# ---------------------------------------------------------------------------

def plot_corr_heatmap(
    corr: pd.DataFrame,
    save_path: str | Path,
    title: str | None = None,
    annot: bool = True,
) -> None:
    """Seaborn heatmap корреляционной матрицы, диапазон [-1, 1], cmap='RdBu_r'."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        annot=annot,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    if title:
        ax.set_title(title, fontsize=13)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)


# ---------------------------------------------------------------------------
# 10. Кросс-корреляционный stem/bar plot
# ---------------------------------------------------------------------------

def plot_xcorr(
    x: pd.Series,
    y: pd.Series,
    max_lag: int,
    save_path: str | Path,
    title: str | None = None,
) -> None:
    """Bar/stem plot кросс-корреляции с горизонтальной линией значимости.

    Линия значимости: ±1.96/sqrt(N_effective), где N_effective — размер
    при lag=0. При крайних лагах реальный N меньше, что консервативно.
    """
    xcorr = cross_correlation(x, y, max_lag=max_lag)
    lags = xcorr["lag"].values
    corrs = xcorr["corr"].values
    N = len(x)
    significance = 1.96 / np.sqrt(N)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(lags, corrs, color="steelblue", alpha=0.7, label="корреляция по лагу")
    ax.axhline(significance, color="red", linestyle="--", linewidth=1.0,
               label=f"граница значимости {significance:.3f}")
    ax.axhline(-significance, color="red", linestyle="--", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Лаг")
    ax.set_ylabel("Корреляция Pearson")
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)

