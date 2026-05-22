"""Анализ периодичности и спектральный анализ временных рядов — Лаборатория №3.

Содержит функции для ACF/PACF, FFT-спектра, STL-декомпозиции и
поиска доминирующего периода на коротких рядах (N=50).

ПРЕДУПРЕЖДЕНИЕ: при N=50 все оценки периодичности ненадёжны
для периодов > N//3 ≈ 16 дней; интерпретируй с осторожностью.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

_N = 50
_MAX_RELIABLE_LAG = _N // 2 - 1   # 24 — жёсткий потолок для ACF/PACF
_RELIABLE_PERIOD = _N // 3        # ≈16 — периоды выше этого ненадёжны
_MIN_CYCLES = 4                    # меньше 4 циклов → STL ненадёжен

plt.rcParams["font.family"] = "DejaVu Sans"


def _series_to_array(series: pd.Series | np.ndarray) -> np.ndarray:
    """Приводит вход к numpy float64 без NaN."""
    arr = np.asarray(series, dtype=float)
    if np.any(np.isnan(arr)):
        raise ValueError("Входной ряд содержит NaN — обработай пропуски до вызова.")
    return arr


# ------------------------------------------------------------------
# 1. ACF / PACF
# ------------------------------------------------------------------

def acf_pacf(series: pd.Series | np.ndarray, nlags: int = 20) -> dict:
    """Вычисляет ACF и PACF с 95% доверительными интервалами.

    Parameters
    ----------
    series : array-like, длина N=50
    nlags  : число лагов (не более N//2 - 1 = 24)

    Returns
    -------
    dict с ключами:
        acf_vals       : np.ndarray (nlags+1,)
        acf_ci         : np.ndarray (nlags+1, 2) — CI вокруг нуля
        pacf_vals      : np.ndarray (nlags+1,)
        pacf_ci        : np.ndarray (nlags+1, 2) — CI вокруг нуля
        significant_acf  : list[int]  — индексы лагов (>=1), вышедших за CI
        significant_pacf : list[int]
        nlags          : int
        warning        : str (только если ненадёжно)
    """
    arr = _series_to_array(series)
    n = len(arr)
    nlags = min(nlags, n // 2 - 1)

    result: dict = {}
    warn_parts = []

    if n < 30:
        warn_parts.append(f"N={n} очень мало для ACF/PACF.")
    if nlags > _RELIABLE_PERIOD:
        warn_parts.append(f"Лаги > {_RELIABLE_PERIOD} ненадёжны при N={n}.")

    # ACF — alpha=0.05 → confint центрирован на acf_vals
    acf_vals, acf_confint = acf(arr, nlags=nlags, alpha=0.05, fft=True)
    # confint[k] = [acf[k] + lower_delta, acf[k] + upper_delta]
    # Пересчитываем CI вокруг нуля:
    acf_ci_zero = acf_confint - acf_vals[:, None]  # shape (nlags+1, 2)

    # PACF — method='ywm' устойчив на коротких рядах
    pacf_vals, pacf_confint = pacf(arr, nlags=nlags, method="ywm", alpha=0.05)
    pacf_ci_zero = pacf_confint - pacf_vals[:, None]

    # Значимые лаги: 0 не в CI вокруг нуля, пропускаем lag=0
    def _significant(vals, ci_zero):
        sig = []
        for k in range(1, len(vals)):
            lo, hi = ci_zero[k]
            if vals[k] < lo or vals[k] > hi:
                sig.append(k)
        return sig

    sig_acf = _significant(acf_vals, acf_ci_zero)
    sig_pacf = _significant(pacf_vals, pacf_ci_zero)

    result["acf_vals"] = acf_vals
    result["acf_ci"] = acf_confint       # хранить оригинальный confint для графиков
    result["acf_ci_zero"] = acf_ci_zero
    result["pacf_vals"] = pacf_vals
    result["pacf_ci"] = pacf_confint
    result["pacf_ci_zero"] = pacf_ci_zero
    result["significant_acf"] = sig_acf
    result["significant_pacf"] = sig_pacf
    result["nlags"] = nlags

    if warn_parts:
        result["warning"] = " ".join(warn_parts)

    return result


# ------------------------------------------------------------------
# 2. FFT спектр
# ------------------------------------------------------------------

def fft_spectrum(series: pd.Series | np.ndarray, detrend: bool = True) -> dict:
    """FFT-спектр (односторонний) с топ-3 доминирующими периодами.

    Parameters
    ----------
    series  : array-like
    detrend : снять линейный тренд перед FFT

    Returns
    -------
    dict:
        freqs           : np.ndarray — частоты (1/день)
        power           : np.ndarray — |X|^2
        dominant_periods: list[dict]  — топ-3: {period, freq, power}
        warning         : str (если N мал)
    """
    from scipy.signal import detrend as sp_detrend

    arr = _series_to_array(series)
    n = len(arr)

    x = sp_detrend(arr, type="linear") if detrend else arr.copy()

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0)
    power = np.abs(X) ** 2

    # Исключаем DC (индекс 0) при поиске пиков
    power_no_dc = power.copy()
    power_no_dc[0] = 0.0

    top_k = 3
    top_idx = np.argsort(power_no_dc)[::-1][:top_k]
    dominant_periods = []
    for idx in top_idx:
        f = freqs[idx]
        if f > 0:
            dominant_periods.append({
                "period_days": float(1.0 / f),
                "freq": float(f),
                "power": float(power[idx]),
            })

    result: dict = {
        "freqs": freqs,
        "power": power,
        "dominant_periods": dominant_periods,
    }

    if n < 50:
        result["warning"] = f"N={n} мало для FFT — спектр шумный."
    else:
        result["warning"] = (
            "N=50 — FFT имеет низкое частотное разрешение (Δf=0.02/день). "
            "Результаты ориентировочные."
        )

    return result


# ------------------------------------------------------------------
# 3. Periodogram (scipy.signal.periodogram)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# 4. STL декомпозиция
# ------------------------------------------------------------------

def stl_decomposition(series: pd.Series | np.ndarray, period: int = 7) -> dict:
    """STL и классическая аддитивная декомпозиция.

    Возвращает компоненты и меры силы сезонности/тренда (Wang et al. 2006):
        Fs = max(0, 1 - var(resid) / var(seasonal + resid))
        Ft = max(0, 1 - var(resid) / var(trend   + resid))

    Parameters
    ----------
    series : array-like, N=50
    period : гипотетический период (7=неделя, 12=квартал-ish)

    Returns
    -------
    dict:
        stl_trend, stl_seasonal, stl_resid  : pd.Series
        sd_trend,  sd_seasonal,  sd_resid   : pd.Series (seasonal_decompose)
        Fs_stl, Ft_stl                      : float
        Fs_sd,  Ft_sd                       : float
        period                              : int
        n_cycles                            : float
        warning                             : str (если < 4 циклов или period > N//3)
    """
    arr = _series_to_array(series)
    n = len(arr)
    n_cycles = n / period

    # Собираем предупреждения
    warn_parts = []
    if n_cycles < _MIN_CYCLES:
        warn_parts.append(
            f"Только {n_cycles:.1f} циклов при period={period}: STL ненадёжен."
        )
    if period > _RELIABLE_PERIOD:
        warn_parts.append(
            f"period={period} > {_RELIABLE_PERIOD} (N//3): результат сомнителен."
        )

    # STL
    s = pd.Series(arr)
    stl_res = STL(s, period=period, robust=True).fit()

    # seasonal_decompose требует два полных цикла минимум
    sd_res = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sd_res = seasonal_decompose(s, model="additive", period=period, extrapolate_trend="freq")
    except Exception as e:
        warn_parts.append(f"seasonal_decompose ошибка: {e}")

    def _strength(resid, other):
        """Сила компоненты; дропаем NaN перед расчётом."""
        r = pd.Series(resid).dropna().values
        o = pd.Series(other).dropna().values
        min_len = min(len(r), len(o))
        r, o = r[:min_len], o[:min_len]
        var_r = float(np.var(r, ddof=1)) if len(r) > 1 else 0.0
        var_ro = float(np.var(r + o, ddof=1)) if len(r + o) > 1 else 0.0
        if var_ro == 0:
            return 0.0
        return max(0.0, 1.0 - var_r / var_ro)

    stl_trend = pd.Series(stl_res.trend, name="trend")
    stl_seasonal = pd.Series(stl_res.seasonal, name="seasonal")
    stl_resid = pd.Series(stl_res.resid, name="resid")

    Fs_stl = _strength(stl_resid, stl_seasonal)
    Ft_stl = _strength(stl_resid, stl_trend)

    result: dict = {
        "stl_trend": stl_trend,
        "stl_seasonal": stl_seasonal,
        "stl_resid": stl_resid,
        "Fs_stl": Fs_stl,
        "Ft_stl": Ft_stl,
        "period": period,
        "n_cycles": n_cycles,
    }

    if sd_res is not None:
        sd_trend = pd.Series(sd_res.trend, name="trend")
        sd_seasonal = pd.Series(sd_res.seasonal, name="seasonal")
        sd_resid = pd.Series(sd_res.resid, name="resid")
        Fs_sd = _strength(sd_resid, sd_seasonal)
        Ft_sd = _strength(sd_resid, sd_trend)
        result.update({
            "sd_trend": sd_trend,
            "sd_seasonal": sd_seasonal,
            "sd_resid": sd_resid,
            "Fs_sd": Fs_sd,
            "Ft_sd": Ft_sd,
        })

    if warn_parts:
        result["warning"] = " ".join(warn_parts)

    return result


# ------------------------------------------------------------------
# 5. Поиск доминирующего периода
# ------------------------------------------------------------------

def find_dominant_period(
    series: pd.Series | np.ndarray,
    candidate_periods: range = range(2, 26),
) -> dict:
    """Ищет доминирующий период перебором кандидатов.

    Для каждого периода p вычисляет:
    - автокорреляцию на лаге p (из statsmodels acf)
    - силу сезонности Fs через STL

    Выбирает лучший по Fs × |acf_at_lag_p|.

    Returns
    -------
    dict:
        scores          : list[dict] — все кандидаты с метриками
        best_period     : int
        best_Fs         : float
        best_acf_at_lag : float
        warning         : str
    """
    arr = _series_to_array(series)
    n = len(arr)

    max_lag = n // 2 - 1
    # Вычислим ACF один раз до max_lag
    acf_vals = acf(arr, nlags=max_lag, fft=True, alpha=None)

    scores = []
    for p in candidate_periods:
        if p >= n:
            continue

        # ACF на лаге p
        acf_at_p = float(acf_vals[p]) if p <= max_lag else float("nan")

        # Сила сезонности STL
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stl_d = stl_decomposition(arr, period=p)
            Fs = stl_d["Fs_stl"]
        except Exception:
            Fs = 0.0

        reliable = p <= _RELIABLE_PERIOD
        score = Fs * abs(acf_at_p) if not np.isnan(acf_at_p) else 0.0
        scores.append({
            "period": p,
            "acf_at_lag": acf_at_p,
            "Fs": Fs,
            "score": score,
            "reliable": reliable,
        })

    if not scores:
        return {"warning": "Нет подходящих кандидатов.", "best_period": None}

    best = max(scores, key=lambda x: x["score"])

    # Лучший НАДЁЖНЫЙ период (p <= N//3) — используй для отчёта
    reliable_scores = [s for s in scores if s["reliable"]]
    best_reliable = max(reliable_scores, key=lambda x: x["score"]) if reliable_scores else None

    result: dict = {
        "scores": scores,
        "best_period": best["period"],
        "best_Fs": best["Fs"],
        "best_acf_at_lag": best["acf_at_lag"],
        "best_reliable_period": best_reliable["period"] if best_reliable else None,
        "best_reliable_Fs": best_reliable["Fs"] if best_reliable else None,
    }
    result["warning"] = (
        f"N={n}: результаты для периодов > {_RELIABLE_PERIOD} ненадёжны. "
        f"Выбранный (глобальный) период={best['period']} "
        + ("(надёжный)" if best["reliable"] else "(НЕНАДЁЖНЫЙ — период > N//3, STL переобучается). "
           f"Рекомендуется использовать best_reliable_period={best_reliable['period'] if best_reliable else 'N/A'}.")
    )
    return result


# ------------------------------------------------------------------
# 6. График ACF / PACF
# ------------------------------------------------------------------

def plot_acf_pacf(
    series: pd.Series | np.ndarray,
    save_path: str | Path,
    nlags: int = 20,
    title: str = "",
) -> None:
    """2 субплота: ACF (верх) + PACF (низ). Сохраняет PNG."""
    arr = _series_to_array(series)
    n = len(arr)
    nlags = min(nlags, n // 2 - 1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle(title or "ACF / PACF", fontsize=13)

    plot_acf(arr, lags=nlags, ax=axes[0], alpha=0.05, title="ACF")
    plot_pacf(arr, lags=nlags, ax=axes[1], alpha=0.05, method="ywm", title="PACF")

    axes[0].set_xlabel("Лаг (дни)")
    axes[1].set_xlabel("Лаг (дни)")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")


# ------------------------------------------------------------------
# 7. График FFT
# ------------------------------------------------------------------

def plot_fft(
    series: pd.Series | np.ndarray,
    save_path: str | Path,
    top_k: int = 3,
    title: str = "",
) -> None:
    """Спектральная плотность с подсвеченными топ-k пиками."""
    spec = fft_spectrum(series, detrend=True)
    freqs = spec["freqs"]
    power = spec["power"]
    dominant = spec["dominant_periods"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs[1:], power[1:], lw=1.2, label="Мощность |X|^2")
    ax.set_xlabel("Частота (1/день)")
    ax.set_ylabel("|X|^2")
    ax.set_title(title or "FFT-спектр")

    colors = ["red", "orange", "green"]
    for i, dp in enumerate(dominant[:top_k]):
        f = dp["freq"]
        p_val = dp["period_days"]
        ax.axvline(f, color=colors[i % len(colors)], linestyle="--", alpha=0.8,
                   label=f"период {p_val:.1f} дн")
        ax.annotate(
            f"T≈{p_val:.1f}д",
            xy=(f, dp["power"]),
            xytext=(f + 0.005, dp["power"] * 0.85),
            fontsize=8,
            color=colors[i % len(colors)],
        )

    ax.legend(fontsize=8)
    if "warning" in spec:
        ax.text(0.01, 0.01, spec["warning"], transform=ax.transAxes,
                fontsize=6, color="gray", va="bottom")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")


# ------------------------------------------------------------------
# 8. График STL декомпозиции
# ------------------------------------------------------------------

def plot_decomposition(
    series: pd.Series | np.ndarray,
    period: int,
    save_path: str | Path,
    title: str = "",
) -> None:
    """4 субплота: observed / trend / seasonal / resid (STL)."""
    arr = _series_to_array(series)
    stl_d = stl_decomposition(arr, period=period)

    observed = arr
    trend = stl_d["stl_trend"].values
    seasonal = stl_d["stl_seasonal"].values
    resid = stl_d["stl_resid"].values

    n = len(arr)
    x = np.arange(1, n + 1)

    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(title or f"STL-декомпозиция (period={period})", fontsize=12)

    data_pairs = [
        (observed, "Наблюдения", "steelblue"),
        (trend, "Тренд", "darkorange"),
        (seasonal, "Сезонность", "green"),
        (resid, "Остаток", "red"),
    ]
    for ax, (data, label, color) in zip(axes, data_pairs):
        ax.plot(x, data, lw=1.2, color=color, label=label)
        ax.set_ylabel(label, fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("День")

    fs = stl_d.get("Fs_stl", float("nan"))
    ft = stl_d.get("Ft_stl", float("nan"))
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")


