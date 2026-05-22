"""Вейвлет-денойзинг (DWT + пороговая обработка) и LMS-адаптивный фильтр.

Hard-часть Задачи 1 (п. 1.4) Лаборатории №3.

Функции:
    dwt_denoise    — DWT-денойзинг с MAD-оценкой шума и универсальным порогом
    LMSFilter      — LMS-адаптивный фильтр (prediction setup)
    lms_denoise    — обёртка с авто-настройкой mu
    apply_all_hard — сводная таблица фильтраций
    filter_metrics — MSE/MAE/RMSE/variance/bias/smoothing_ratio
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pywt


# ---------------------------------------------------------------------------
# 1. DWT-денойзинг
# ---------------------------------------------------------------------------

def dwt_denoise(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int | None = None,
    threshold: str = "universal",
    mode: str = "soft",
) -> dict:
    """DWT-денойзинг с MAD-оценкой шума (Donoho) и порогом Donoho-Johnstone.

    Parameters
    ----------
    x        : 1-D массив сигнала
    wavelet  : имя вейвлета (pywt)
    level    : уровень декомпозиции; если None — автоопределение с кепом 4
    threshold: 'universal' (λ = σ·√(2·ln N)) или 'minimax' (half-universal)
    mode     : 'soft' или 'hard'

    Returns
    -------
    dict с ключами: denoised, coeffs_raw, coeffs_thresh, sigma_est,
                    threshold_lambda, level_used
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    wav = pywt.Wavelet(wavelet)

    # Определить уровень
    if level is None:
        max_lev = pywt.dwt_max_level(N, wav.dec_len)
        level = min(max_lev, 4)

    # Разложение
    coeffs = pywt.wavedec(x, wavelet, level=level, mode="periodization")

    # Оценка σ по MAD первого детального уровня (стандарт Donoho 1994)
    detail_1 = coeffs[-1]
    sigma_est = np.median(np.abs(detail_1)) / 0.6745

    # Универсальный порог Donoho-Johnstone
    if threshold == "universal":
        lambda_thr = sigma_est * np.sqrt(2.0 * np.log(N))
    else:  # minimax
        lambda_thr = sigma_est * np.sqrt(2.0 * np.log(N)) / 2.0

    # Применить порог только к детальным коэффициентам (не к аппроксимации)
    coeffs_thresh = [coeffs[0].copy()]  # аппроксимация без изменений
    for c in coeffs[1:]:
        coeffs_thresh.append(pywt.threshold(c, value=lambda_thr, mode=mode))

    # Реконструкция
    denoised = pywt.waverec(coeffs_thresh, wavelet, mode="periodization")
    denoised = denoised[: N]  # periodization может дать N+1 при нечётных уровнях

    return {
        "denoised": denoised,
        "coeffs_raw": coeffs,
        "coeffs_thresh": coeffs_thresh,
        "sigma_est": float(sigma_est),
        "threshold_lambda": float(lambda_thr),
        "level_used": int(level),
    }


# ---------------------------------------------------------------------------
# 2. LMS-адаптивный фильтр
# ---------------------------------------------------------------------------

class LMSFilter:
    """Адаптивный LMS-фильтр (prediction setup).

    Предсказывает d[t] по d[t-1 .. t-n_taps] (авторегрессионный режим).

    Parameters
    ----------
    n_taps : количество отводов (ширина окна)
    mu     : шаг обучения (step size / learning rate)
    """

    def __init__(self, n_taps: int = 5, mu: float = 0.01) -> None:
        self.n_taps = n_taps
        self.mu = mu
        self.weights_history: list[np.ndarray] = []

    def fit_predict(self, d: np.ndarray, w_init: np.ndarray | None = None) -> np.ndarray:
        """Адаптивная фильтрация в режиме предсказания.

        Parameters
        ----------
        d      : желаемый сигнал (шумный; prediction target)
        w_init : начальные веса (если None — нули); для продолжения обучения
                 между несколькими проходами передавать weights_final прошлого прохода

        Returns
        -------
        np.ndarray той же длины что и d:
            NaN для первых n_taps позиций, предсказание для остальных.
        """
        d = np.asarray(d, dtype=float)
        N = len(d)
        n = self.n_taps
        w = np.zeros(n, dtype=float) if w_init is None else w_init.copy()
        output = np.full(N, np.nan)
        self.weights_history = []

        for t in range(n, N):
            x_t = d[t - n: t][::-1]  # вектор признаков (t-1..t-n), от новых к старым
            y_t = float(np.dot(w, x_t))
            e_t = d[t] - y_t
            w = w + 2.0 * self.mu * e_t * x_t
            output[t] = y_t
            self.weights_history.append(w.copy())

        return output


# ---------------------------------------------------------------------------
# 6. lms_denoise — обёртка с авто-mu и несколькими проходами
# ---------------------------------------------------------------------------

def lms_denoise(
    x: np.ndarray,
    n_taps: int = 5,
    mu: float | str = "auto",
    n_passes: int = 3,
) -> dict:
    """Денойзинг через LMS в prediction-режиме.

    Сигнал нормируется (z-score) перед фильтрацией, выход обратно масштабируется.
    Это необходимо для устойчивости mu='auto' при больших значениях сигнала.

    Parameters
    ----------
    x        : входной (шумный) сигнал
    n_taps   : ширина окна LMS
    mu       : шаг обучения; 'auto' → 0.1 · 2/(n_taps · power(x_norm))
    n_passes : количество проходов (улучшает сходимость на коротких рядах)

    Returns
    -------
    dict: denoised (np.ndarray), weights_final, mu_used, error_curve
    """
    x = np.asarray(x, dtype=float)

    # Z-score нормировка
    x_mean = np.mean(x)
    x_std = np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else 1.0
    x_norm = (x - x_mean) / x_std

    # Авто-mu по нормированному сигналу
    power = float(np.mean(x_norm**2))
    if mu == "auto":
        mu_used = 0.1 * (2.0 / (n_taps * max(power, 1e-10)))
        # safety cap: не превышать 0.5 (нестабильность)
        mu_used = min(mu_used, 0.5)
    else:
        mu_used = float(mu)

    # Несколько проходов по x_norm с накоплением весов.
    # Каждый проход тренируется на оригинальном x_norm (не на предсказаниях!),
    # передавая финальные веса в следующий проход как горячий старт.
    # Это позволяет весам сойтись дальше за N=50 шагов.
    lms_obj = LMSFilter(n_taps=n_taps, mu=mu_used)
    all_error_curves: list[np.ndarray] = []
    w_current: np.ndarray | None = None
    pred: np.ndarray = np.full(len(x_norm), np.nan)

    for _ in range(n_passes):
        pred = lms_obj.fit_predict(x_norm, w_init=w_current)
        # Передать финальные веса следующему проходу
        if lms_obj.weights_history:
            w_current = lms_obj.weights_history[-1].copy()
        # Ошибка на обучении (только для позиций где есть предсказание)
        valid_mask = ~np.isnan(pred)
        error_abs = np.abs(x_norm[valid_mask] - pred[valid_mask])
        all_error_curves.append(error_abs)

    # Денормализация
    denoised_norm = pred.copy()
    denoised_norm[np.isnan(denoised_norm)] = x_norm[np.isnan(denoised_norm)]
    denoised = denoised_norm * x_std + x_mean

    error_curve = np.concatenate(all_error_curves)
    weights_final = w_current if w_current is not None else np.zeros(n_taps)

    return {
        "denoised": denoised,
        "weights_final": weights_final,
        "mu_used": float(mu_used),
        "error_curve": error_curve,
    }


# ---------------------------------------------------------------------------
# 7. Метрики фильтрации (LOCAL COPY — mirror of filters.py, Agent-4 interface)
# ---------------------------------------------------------------------------

def filter_metrics(original: np.ndarray, filtered: np.ndarray) -> dict:
    """Метрики качества фильтрации (без ground truth — сравнение с оригиналом).

    LOCAL COPY — зеркало интерфейса filters.py (Agent-4).
    При появлении filters.py импортировать оттуда, но не переписывать.

    Returns
    -------
    dict: mse, mae, rmse, residual_variance, bias, smoothing_ratio
    """
    original = np.asarray(original, dtype=float)
    filtered = np.asarray(filtered, dtype=float)

    # Маскируем NaN (начало LMS)
    mask = ~(np.isnan(original) | np.isnan(filtered))
    o = original[mask]
    f = filtered[mask]

    residuals = o - f
    mse = float(np.mean(residuals**2))
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(mse))
    residual_variance = float(np.var(residuals, ddof=1))
    bias = float(np.mean(residuals))
    var_orig = float(np.var(o, ddof=1))
    var_filt = float(np.var(f, ddof=1))
    smoothing_ratio = float(1.0 - var_filt / var_orig) if var_orig > 0 else 0.0

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "residual_variance": residual_variance,
        "bias": bias,
        "smoothing_ratio": smoothing_ratio,
    }


# ---------------------------------------------------------------------------
# 8. apply_all_hard — сводная таблица фильтраций
# ---------------------------------------------------------------------------

def apply_all_hard(series: pd.Series) -> pd.DataFrame:
    """Применить DWT(db4), DWT(sym4) и LMS(5,auto) к ряду.

    Parameters
    ----------
    series : pd.Series с временным рядом

    Returns
    -------
    DataFrame с колонками: original, DWT(db4), DWT(sym4), LMS(5,auto)
    Индекс совпадает с индексом series.
    """
    x = series.values.astype(float)

    dwt_db4 = dwt_denoise(x, wavelet="db4")["denoised"]
    dwt_sym4 = dwt_denoise(x, wavelet="sym4")["denoised"]
    lms_res = lms_denoise(x, n_taps=5, mu="auto", n_passes=3)["denoised"]

    return pd.DataFrame(
        {
            "original": x,
            "DWT(db4)": dwt_db4,
            "DWT(sym4)": dwt_sym4,
            "LMS(5,auto)": lms_res,
        },
        index=series.index,
    )

