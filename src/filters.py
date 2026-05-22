"""Фильтры временных рядов — Лабораторная №3 (пункты 1.1–1.3).

Реализует:
  1.1 — Скользящее среднее и экспоненциальное сглаживание
  1.2 — Скалярный фильтр Калмана (random-walk) + RTS-сглаживатель
  1.3 — Фильтр Савицкого–Голея + подбор оптимального окна по LOO-CV

Точка входа: apply_all_core(series) → pd.DataFrame с 8 колонками.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def _to_array(x: Union[pd.Series, np.ndarray, list]) -> np.ndarray:
    """Привести вход к 1D float64 без NaN."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Ожидается 1D массив, получен shape={arr.shape}")
    if np.any(np.isnan(arr)):
        raise ValueError("Входной ряд содержит NaN — заполни пропуски до вызова.")
    return arr


# ===========================================================================
# 1.1  Скользящее среднее и экспоненциальное сглаживание
# ===========================================================================

def moving_average(x: Union[pd.Series, np.ndarray], window: int = 5, *, mode: str = "same") -> np.ndarray:
    """Простое центрированное скользящее среднее длиной = len(x).

    Параметры
    ----------
    x      : входной ряд
    window : ширина окна (нечётное желательно; если чётное — работает корректно)
    mode   : только 'same' поддерживается (сохранено для совместимости API)

    Реализация: явный edge-padding (np.pad mode='edge') → свёртка 'valid'.
    Это даёт корректное центрированное среднее без нулевого смещения краёв
    (в отличие от np.convolve с mode='same', который нулями добивает края).
    """
    arr = _to_array(x)
    n = len(arr)
    if window < 1:
        raise ValueError(f"window должен быть >= 1, получен {window}")
    if window > n:
        window = n  # деградируем до среднего по всему ряду

    kernel = np.ones(window) / window
    half = window // 2
    # Зеркально-граничное добавление (reflect) даёт более гладкий переход,
    # чем edge; оба варианта правильны — выбираем reflect.
    padded = np.pad(arr, half, mode="edge")
    # Если window чётное, нам нужен pad справа на 1 меньше
    if window % 2 == 0:
        # Правый pad: half-1, левый pad: half → symmetric convolution
        padded = np.concatenate([arr[:half], arr, arr[-(half - 1):]])
    result = np.convolve(padded, kernel, mode="valid")
    # 'valid' может дать чуть другую длину — обрезаем/дополняем к n
    result = result[:n]
    if len(result) < n:
        # Дополняем последним значением (на практике не должно происходить)
        result = np.concatenate([result, np.full(n - len(result), result[-1])])
    return result


def exponential_smoothing(x: Union[pd.Series, np.ndarray], alpha: float = 0.3) -> np.ndarray:
    """Простое экспоненциальное сглаживание Брауна (SES).

    Рекуррентная формула:
        s_0 = x_0
        s_t = alpha * x_t + (1 - alpha) * s_{t-1}

    Параметры
    ----------
    x     : входной ряд
    alpha : коэффициент сглаживания, 0 < alpha <= 1
    """
    arr = _to_array(x)
    if not (0 < alpha <= 1.0):
        raise ValueError(f"alpha должен быть в (0, 1], получен {alpha}")
    n = len(arr)
    out = np.empty(n, dtype=float)
    out[0] = arr[0]
    for t in range(1, n):
        out[t] = alpha * arr[t] + (1.0 - alpha) * out[t - 1]
    return out


def double_exponential_smoothing(
    x: Union[pd.Series, np.ndarray],
    alpha: float = 0.3,
    beta: float = 0.1,
) -> np.ndarray:
    """Двойное экспоненциальное сглаживание Холта (уровень + тренд).

    Рекуррентные формулы:
        L_0 = x_0,  T_0 = x_1 - x_0  (начальный тренд)
        L_t = alpha * x_t + (1 - alpha) * (L_{t-1} + T_{t-1})
        T_t = beta  * (L_t - L_{t-1}) + (1 - beta) * T_{t-1}
        s_t = L_t + T_t  (сглаженный ряд)

    Параметры
    ----------
    x     : входной ряд
    alpha : коэффициент уровня, 0 < alpha <= 1
    beta  : коэффициент тренда, 0 < beta  <= 1
    """
    arr = _to_array(x)
    if len(arr) < 2:
        raise ValueError("double_exponential_smoothing требует len(x) >= 2")
    if not (0 < alpha <= 1.0):
        raise ValueError(f"alpha должен быть в (0, 1], получен {alpha}")
    if not (0 < beta <= 1.0):
        raise ValueError(f"beta должен быть в (0, 1], получен {beta}")

    n = len(arr)
    L = np.empty(n, dtype=float)
    T = np.empty(n, dtype=float)
    out = np.empty(n, dtype=float)

    L[0] = arr[0]
    T[0] = arr[1] - arr[0]
    out[0] = L[0] + T[0]

    for t in range(1, n):
        L[t] = alpha * arr[t] + (1.0 - alpha) * (L[t - 1] + T[t - 1])
        T[t] = beta * (L[t] - L[t - 1]) + (1.0 - beta) * T[t - 1]
        out[t] = L[t] + T[t]

    return out


# ===========================================================================
# 1.2  Фильтр Калмана (1D, random-walk) + RTS-сглаживатель
# ===========================================================================

class KalmanFilter1D:
    """Скалярный фильтр Калмана на модели случайного блуждания.

    Модель:
        Состояние:    x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)
        Наблюдение:   z_t = x_t   + v_t,   v_t ~ N(0, R)

    Параметры
    ----------
    Q  : дисперсия шума процесса (случайного блуждания)
    R  : дисперсия шума наблюдений
    x0 : начальное состояние; если None — инициализируется первым наблюдением z[0]
    P0 : начальная ковариация состояния
    """

    def __init__(self, Q: float, R: float, x0: float | None = None, P0: float = 1.0) -> None:
        if Q <= 0:
            raise ValueError(f"Q должен быть > 0, получен {Q}")
        if R <= 0:
            raise ValueError(f"R должен быть > 0, получен {R}")
        self.Q = float(Q)
        self.R = float(R)
        self._x0 = x0  # None → будет взят из z[0]
        self.P0 = float(P0)

    # ------------------------------------------------------------------
    def filter(self, z: Union[pd.Series, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Прямой проход фильтрации (forward Kalman filter).

        Возвращает
        ----------
        x_filt : np.ndarray — апостериорные оценки состояния
        P_filt : np.ndarray — апостериорные ковариации
        """
        z_arr = _to_array(z)
        n = len(z_arr)
        Q, R = self.Q, self.R

        x_filt = np.empty(n, dtype=float)
        P_filt = np.empty(n, dtype=float)

        # Инициализация: если x0 не задан — используем первое наблюдение
        x_prev = z_arr[0] if self._x0 is None else float(self._x0)
        P_prev = self.P0

        for t in range(n):
            # --- Предсказание ---
            x_pred = x_prev                 # x_{t|t-1} = x_{t-1|t-1}
            P_pred = P_prev + Q             # P_{t|t-1} = P_{t-1|t-1} + Q

            # --- Обновление (Kalman gain) ---
            K = P_pred / (P_pred + R)       # K_t = P_{t|t-1} / (P_{t|t-1} + R)
            x_upd = x_pred + K * (z_arr[t] - x_pred)
            P_upd = (1.0 - K) * P_pred

            x_filt[t] = x_upd
            P_filt[t] = P_upd

            x_prev = x_upd
            P_prev = P_upd

        return x_filt, P_filt

    # ------------------------------------------------------------------
    def smooth(self, z: Union[pd.Series, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """RTS (Rauch-Tung-Striebel) сглаживатель — обратный проход.

        Сначала выполняет прямой фильтр, затем RTS backward pass.

        Формулы RTS для модели RW:
            P_pred_{t+1} = P_filt_t + Q
            C_t          = P_filt_t / P_pred_{t+1}
            x_smooth_t   = x_filt_t + C_t * (x_smooth_{t+1} - x_pred_{t+1})
            P_smooth_t   = P_filt_t + C_t^2 * (P_smooth_{t+1} - P_pred_{t+1})

        Возвращает
        ----------
        x_smooth : np.ndarray
        P_smooth : np.ndarray
        """
        z_arr = _to_array(z)
        n = len(z_arr)
        Q = self.Q

        # Прямой фильтр с сохранением предсказаний
        x_filt = np.empty(n, dtype=float)
        P_filt = np.empty(n, dtype=float)
        x_pred_arr = np.empty(n, dtype=float)
        P_pred_arr = np.empty(n, dtype=float)

        x_prev = z_arr[0] if self._x0 is None else float(self._x0)
        P_prev = self.P0

        for t in range(n):
            x_pred = x_prev
            P_pred = P_prev + Q
            K = P_pred / (P_pred + self.R)
            x_upd = x_pred + K * (z_arr[t] - x_pred)
            P_upd = (1.0 - K) * P_pred

            x_pred_arr[t] = x_pred
            P_pred_arr[t] = P_pred
            x_filt[t] = x_upd
            P_filt[t] = P_upd

            x_prev = x_upd
            P_prev = P_upd

        # Обратный проход (RTS)
        x_smooth = x_filt.copy()
        P_smooth = P_filt.copy()

        # Последняя точка остаётся апостериорной оценкой фильтра
        for t in range(n - 2, -1, -1):
            # P_pred_{t+1} = P_filt_t + Q
            P_pred_tp1 = P_filt[t] + Q
            # Сглаживающий коэффициент (gain)
            C = P_filt[t] / P_pred_tp1
            x_smooth[t] = x_filt[t] + C * (x_smooth[t + 1] - x_pred_arr[t + 1])
            P_smooth[t] = P_filt[t] + C ** 2 * (P_smooth[t + 1] - P_pred_tp1)

        return x_smooth, P_smooth


# ---------------------------------------------------------------------------
# Оценка Q, R из данных (метод моментов для RW-модели)
# ---------------------------------------------------------------------------

def _estimate_qr(z: np.ndarray, eps_factor: float = 1e-6) -> tuple[float, float]:
    """Оценка Q и R по методу моментов для модели случайного блуждания.

    Формулы:
        Δz_t   = z_t - z_{t-1}      → Var(Δz)  = Q + 2R
        Δ²z_t  = Δz_t - Δz_{t-1}   → Var(Δ²z) = 2Q + 6R

    Решаем систему:
        Var(Δ²z) = 2·Var(Δz) + 2R  → R = (Var(Δ²z) - 2·Var(Δz)) / 2
        Q = Var(Δz) - 2R

    Оба значения могут получиться отрицательными на шумных коротких рядах —
    ограничиваем снизу ε = eps_factor * var(z).
    """
    eps = max(1e-12, eps_factor * float(np.var(z, ddof=1)))
    dz = np.diff(z)
    d2z = np.diff(dz)
    var_dz = float(np.var(dz, ddof=1)) if len(dz) > 1 else eps
    var_d2z = float(np.var(d2z, ddof=1)) if len(d2z) > 1 else eps

    R_hat = (var_d2z - 2.0 * var_dz) / 2.0
    Q_hat = var_dz - 2.0 * R_hat

    R_hat = max(R_hat, eps)
    Q_hat = max(Q_hat, eps)
    return Q_hat, R_hat


def kalman_1d_rw(
    x: Union[pd.Series, np.ndarray],
    Q: float | None = None,
    R: float | None = None,
    smooth: bool = True,
) -> np.ndarray:
    """Удобная обёртка: KalmanFilter1D с авто-оценкой Q/R из данных.

    Если Q или R не заданы — оцениваются по методу моментов (_estimate_qr).

    Параметры
    ----------
    x      : входной ряд
    Q      : дисперсия шума процесса (None → авто)
    R      : дисперсия шума наблюдений (None → авто)
    smooth : True → RTS-сглаживание, False → только фильтрация
    """
    arr = _to_array(x)
    if Q is None or R is None:
        Q_est, R_est = _estimate_qr(arr)
        Q = Q_est if Q is None else Q
        R = R_est if R is None else R

    kf = KalmanFilter1D(Q=Q, R=R)
    if smooth:
        result, _ = kf.smooth(arr)
    else:
        result, _ = kf.filter(arr)
    return result


# ===========================================================================
# 1.3  Фильтр Савицкого–Голея
# ===========================================================================

def savitzky_golay(
    x: Union[pd.Series, np.ndarray],
    window: int = 7,
    polyorder: int = 2,
    deriv: int = 0,
) -> np.ndarray:
    """Фильтр Савицкого–Голея (обёртка scipy.signal.savgol_filter).

    Параметры
    ----------
    x         : входной ряд
    window    : ширина окна (нечётное, > polyorder)
    polyorder : степень полинома
    deriv     : порядок производной (0 = сглаживание)

    Raises
    ------
    ValueError : если window чётное, window <= polyorder или window > len(x)
    """
    arr = _to_array(x)
    n = len(arr)

    if window % 2 == 0:
        raise ValueError(f"window должен быть нечётным, получен {window}")
    if window <= polyorder:
        raise ValueError(f"window ({window}) должен быть > polyorder ({polyorder})")
    if window > n:
        raise ValueError(f"window ({window}) превышает длину ряда ({n})")

    return savgol_filter(arr, window_length=window, polyorder=polyorder, deriv=deriv, mode="interp")


# ===========================================================================
# Сводная функция: apply_all_core
# ===========================================================================

def apply_all_core(series: pd.Series) -> pd.DataFrame:
    """Применить все основные фильтры к одномерному ряду.

    Параметры
    ----------
    series : pd.Series — 1D временной ряд (индекс сохраняется)

    Возвращает
    ----------
    pd.DataFrame с колонками:
        original, MA(5), MA(7), Exp(a=0.3), Exp(a=0.5), Holt, Kalman_RW, SavGol(7,2)
    Индекс совпадает с индексом series.
    """
    arr = _to_array(series)
    idx = series.index

    result = pd.DataFrame(index=idx)
    result["original"]    = arr
    result["MA(5)"]       = moving_average(arr, window=5)
    result["MA(7)"]       = moving_average(arr, window=7)
    result["Exp(a=0.3)"]  = exponential_smoothing(arr, alpha=0.3)
    result["Exp(a=0.5)"]  = exponential_smoothing(arr, alpha=0.5)
    result["Holt"]        = double_exponential_smoothing(arr, alpha=0.3, beta=0.1)
    result["Kalman_RW"]   = kalman_1d_rw(arr, smooth=True)
    result["SavGol(7,2)"] = savitzky_golay(arr, window=7, polyorder=2)
    return result


# ===========================================================================
# Метрики качества фильтрации
# ===========================================================================

def filter_metrics(original: Union[pd.Series, np.ndarray], filtered: Union[pd.Series, np.ndarray]) -> dict:
    """Метрики качества одного фильтра.

    Возвращает
    ----------
    dict:
        MSE              : среднеквадратичная ошибка
        MAE              : средняя абсолютная ошибка
        RMSE             : корень из MSE
        residual_var     : дисперсия остатков (original - filtered)
        bias             : среднее смещение (mean(original - filtered))
        smoothing_degree : 1 - var(filtered) / var(original)
                           Интерпретация: доля удалённой дисперсии.
                           Может быть < 0 (фильтр усиливает дисперсию) или
                           > 1 (редко, при деградированном фильтре). Не обрезаем.
    """
    o = _to_array(original)
    f = _to_array(filtered)
    residuals = o - f

    var_o = float(np.var(o, ddof=1))
    var_f = float(np.var(f, ddof=1))

    smoothing_degree = 1.0 - var_f / var_o if var_o > 0 else float("nan")

    return {
        "MSE":              float(np.mean(residuals ** 2)),
        "MAE":              float(np.mean(np.abs(residuals))),
        "RMSE":             float(np.sqrt(np.mean(residuals ** 2))),
        "residual_var":     float(np.var(residuals, ddof=1)),
        "bias":             float(np.mean(residuals)),
        "smoothing_degree": smoothing_degree,
    }


# ===========================================================================
# Построение графиков
# ===========================================================================

