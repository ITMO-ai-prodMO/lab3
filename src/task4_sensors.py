"""Задача 4: двойные датчики — загрузка, фьюжн, анализ, аномалии.

Источник данных: UCI Air Quality dataset (CO(GT) reference + PT08.S1(CO) sensor).
URL: https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip
Резервная стратегия: синтетические dual-sensor данные (seed=42).
"""

from __future__ import annotations

import json
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).parent
_LAB_DIR = _SRC_DIR.parent
DATA_DIR = _LAB_DIR / "data_external"
FIG_DIR = _LAB_DIR / "figures"
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
UCI_ZIP = DATA_DIR / "AirQualityUCI.zip"
UCI_CSV = DATA_DIR / "AirQualityUCI.csv"


# ---------------------------------------------------------------------------
# 1. load_dual_sensor
# ---------------------------------------------------------------------------

def _rolling_mean_np(arr: np.ndarray, window: int) -> np.ndarray:
    """Скользящее среднее через numpy; ближние значения — частичные окна."""
    result = np.empty_like(arr, dtype=float)
    for i in range(len(arr)):
        lo = max(0, i - window // 2)
        hi = min(len(arr), lo + window)
        result[i] = arr[lo:hi].mean()
    return result


def _load_uci() -> Optional[tuple[pd.DataFrame, dict]]:
    """Попытаться загрузить UCI Air Quality и вернуть (df, meta) или None."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if not UCI_CSV.exists():
            if not UCI_ZIP.exists():
                urllib.request.urlretrieve(UCI_URL, UCI_ZIP)
            with zipfile.ZipFile(UCI_ZIP) as zf:
                zf.extract("AirQualityUCI.csv", DATA_DIR)

        raw = pd.read_csv(UCI_CSV, sep=";", decimal=",", encoding="utf-8")
        # Удалить пустые хвостовые колонки / строки
        raw = raw.dropna(how="all").loc[:, ~raw.columns.str.startswith("Unnamed")]

        # Парсинг времени
        raw["datetime"] = pd.to_datetime(
            raw["Date"] + " " + raw["Time"],
            format="%d/%m/%Y %H.%M.%S",
            errors="coerce",
        )
        raw = raw.dropna(subset=["datetime"])

        # Два датчика CO: reference (GT) и metal-oxide sensor
        co_gt = raw["CO(GT)"].replace(-200, np.nan)
        pt08 = raw["PT08.S1(CO)"].replace(-200, np.nan)

        # Фильтруем строки, где оба действительны
        mask = co_gt.notna() & pt08.notna()
        t = raw["datetime"][mask].reset_index(drop=True)
        z1_raw = co_gt[mask].to_numpy(dtype=float)   # mg/m³
        z2_raw = pt08[mask].to_numpy(dtype=float)    # raw response

        # Аффинное выравнивание: z2_aligned = a*z2 + b, fit via lstsq
        A = np.column_stack([z2_raw, np.ones(len(z2_raw))])
        params, _, _, _ = np.linalg.lstsq(A, z1_raw, rcond=None)
        a_align, b_align = params
        z2_aligned = a_align * z2_raw + b_align

        df = pd.DataFrame({"t": t, "sensor_1": z1_raw, "sensor_2": z2_aligned})

        meta = {
            "source": UCI_URL,
            "sensor_1": "CO(GT) — reference electrochemical sensor [mg/m³]",
            "sensor_2": "PT08.S1(CO) — metal-oxide sensor (affine-aligned to mg/m³)",
            "align_params": {"a": float(a_align), "b": float(b_align)},
            "n_rows": len(df),
            "dataset": "UCI Air Quality Dataset (10/03/2004 – 04/04/2005)",
        }
        return df, meta
    except Exception as exc:  # noqa: BLE001
        print(f"[task4] UCI load failed: {exc}")
        return None


def _make_synthetic() -> tuple[pd.DataFrame, dict]:
    """Синтетические данные: объект с движением + два датчика."""
    rng = np.random.default_rng(42)
    N, dt = 500, 1.0
    sigma1, sigma2, bias = 0.5, 2.0, 0.3

    # Истинная траектория: случайное ускорение + slow drift
    a = rng.normal(0, 0.05, N)
    v = np.zeros(N)
    x = np.zeros(N)
    for i in range(1, N):
        v[i] = v[i - 1] + a[i - 1] * dt
        x[i] = x[i - 1] + v[i - 1] * dt + 0.5 * a[i - 1] * dt ** 2
    # Slow drift
    x += np.linspace(0, 3.0, N)

    z1 = x + rng.normal(0, sigma1, N)
    z2 = x + rng.normal(0, sigma2, N) + bias

    df = pd.DataFrame({
        "t": np.arange(N, dtype=float),
        "sensor_1": z1,
        "sensor_2": z2,
        "truth": x,
    })
    meta = {
        "source": "synthesized",
        "sensor_1": "z1 = truth + N(0, 0.5²) [точный датчик]",
        "sensor_2": "z2 = truth + N(0, 2.0²) + 0.3 [шумный датчик со смещением]",
        "sigma1": sigma1,
        "sigma2": sigma2,
        "bias": bias,
        "n_rows": N,
        "note": (
            "Реальные данные недоступны; синтетика используется для демонстрации "
            "корректности метода при известных параметрах шума."
        ),
    }
    return df, meta


def load_dual_sensor() -> tuple[pd.DataFrame, dict]:
    """Загрузить данные двойных датчиков.

    Returns
    -------
    df : DataFrame с колонками ``t, sensor_1, sensor_2`` (+ ``truth`` для синтетики).
    meta : словарь с описанием источника и параметров.
    """
    result = _load_uci()
    if result is not None:
        return result
    return _make_synthetic()


# ---------------------------------------------------------------------------
# 2. fuse_sensors_kalman
# ---------------------------------------------------------------------------

def fuse_sensors_kalman(
    z1: np.ndarray,
    z2: np.ndarray,
    Q: Optional[float] = None,
    R1: Optional[float] = None,
    R2: Optional[float] = None,
) -> dict:
    """2-сенсорный фильтр Калмана с одним скалярным состоянием.

    На каждом шаге: predict → update-z1 → update-z2 (sequential update).

    Parameters
    ----------
    z1, z2 : массивы измерений.
    Q : дисперсия процесса. None → эвристика из данных.
    R1, R2 : дисперсии шумов датчиков. None → MLE из локальных вариаций.

    Returns
    -------
    dict : fused, variance, K1 (gains sensor_1), K2 (gains sensor_2).
    """
    N = len(z1)

    # Эвристическая оценка параметров
    if R1 is None:
        rm1 = _rolling_mean_np(z1, 5)
        R1 = float(np.var(z1 - rm1)) or 1e-4
    if R2 is None:
        rm2 = _rolling_mean_np(z2, 5)
        R2 = float(np.var(z2 - rm2)) or 1e-4
    if Q is None:
        avg = (z1 + z2) / 2.0
        Q = float(np.var(np.diff(avg)) / 10.0) or 1e-6

    # Начальное состояние
    x = float((z1[0] + z2[0]) / 2.0)
    P = (R1 + R2) / 2.0

    fused = np.empty(N)
    variance = np.empty(N)
    k1_arr = np.empty(N)
    k2_arr = np.empty(N)

    for i in range(N):
        # --- predict ---
        x_pred = x
        P_pred = P + Q

        # --- update with z1 ---
        K1 = P_pred / (P_pred + R1)
        x_post1 = x_pred + K1 * (z1[i] - x_pred)
        P_post1 = (1.0 - K1) * P_pred

        # --- update with z2 ---
        K2 = P_post1 / (P_post1 + R2)
        x_post2 = x_post1 + K2 * (z2[i] - x_post1)
        P_post2 = (1.0 - K2) * P_post1

        fused[i] = x_post2
        variance[i] = P_post2
        k1_arr[i] = K1
        k2_arr[i] = K2

        x, P = x_post2, P_post2

    return {
        "fused": fused,
        "variance": variance,
        "K1": k1_arr,
        "K2": k2_arr,
        "R1": R1,
        "R2": R2,
        "Q": Q,
    }


# ---------------------------------------------------------------------------
# 4. analyze_residuals
# ---------------------------------------------------------------------------

def analyze_residuals(
    fused: np.ndarray,
    z1: np.ndarray,
    z2: np.ndarray,
) -> dict:
    """Анализ остатков: дисперсия, корреляция, смещение.

    Returns
    -------
    dict с полями: var_z1, var_z2, var_fused, snr_improvement_z1_pct,
    snr_improvement_z2_pct, crosscorr_residuals, pearson_z1_z2,
    bias_z2_vs_z1.
    """
    res1 = z1 - fused
    res2 = z2 - fused
    var_z1 = float(np.var(z1 - _rolling_mean_np(z1, 10)))
    var_z2 = float(np.var(z2 - _rolling_mean_np(z2, 10)))
    var_fused = float(np.var(fused - _rolling_mean_np(fused, 10)))

    # Взаимная корреляция остатков (lag=0)
    c1 = res1 - res1.mean()
    c2 = res2 - res2.mean()
    denom = (np.std(res1) * np.std(res2)) or 1e-12
    crosscorr = float(np.dot(c1, c2) / (len(res1) * denom))

    # Pearson между z1 и z2
    z1c, z2c = z1 - z1.mean(), z2 - z2.mean()
    pearson = float(np.dot(z1c, z2c) / (len(z1) * np.std(z1) * np.std(z2) + 1e-12))

    bias_z2_vs_z1 = float(np.mean(z2 - z1))

    snr_imp_z1 = float((1.0 - var_fused / var_z1) * 100.0) if var_z1 > 0 else 0.0
    snr_imp_z2 = float((1.0 - var_fused / var_z2) * 100.0) if var_z2 > 0 else 0.0

    return {
        "var_z1": var_z1,
        "var_z2": var_z2,
        "var_fused": var_fused,
        "snr_improvement_z1_pct": snr_imp_z1,
        "snr_improvement_z2_pct": snr_imp_z2,
        "crosscorr_residuals": crosscorr,
        "pearson_z1_z2": pearson,
        "bias_z2_vs_z1": bias_z2_vs_z1,
    }


# ---------------------------------------------------------------------------
# 5. detect_anomalies
# ---------------------------------------------------------------------------

def detect_anomalies(
    df: pd.DataFrame,
    fused: np.ndarray,
    z_window: int = 10,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Обнаружить аномалии: точки, где |z_i - fused| > threshold * rolling_std(z_i).

    Parameters
    ----------
    df : DataFrame с колонками t, sensor_1, sensor_2.
    fused : оценка фьюжн (N,).
    z_window : окно для скользящего СКО.
    threshold : множитель.

    Returns
    -------
    DataFrame: timestamp, sensor, value, fused_value, residual.
    """
    rows: list[dict] = []
    for sensor_name in ("sensor_1", "sensor_2"):
        z = df[sensor_name].to_numpy(dtype=float)
        s = pd.Series(z)
        roll_std = s.rolling(z_window, min_periods=1).std().to_numpy()
        roll_std = np.where(roll_std == 0, 1e-9, roll_std)
        residuals = np.abs(z - fused)
        flags = residuals > threshold * roll_std
        for idx in np.where(flags)[0]:
            rows.append({
                "timestamp": df["t"].iloc[idx],
                "sensor": sensor_name,
                "value": float(z[idx]),
                "fused_value": float(fused[idx]),
                "residual": float(z[idx] - fused[idx]),
            })
    return pd.DataFrame(rows, columns=["timestamp", "sensor", "value", "fused_value", "residual"])


# ---------------------------------------------------------------------------
# Построение графиков
# ---------------------------------------------------------------------------

def _ensure_fig_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_raw_sensors(df: pd.DataFrame, meta: dict) -> None:
    """График сырых сигналов двух датчиков."""
    _ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(12, 4))
    t = df["t"]
    ax.plot(t, df["sensor_1"], label="Датчик 1 (эталон)", alpha=0.7, linewidth=0.8)
    ax.plot(t, df["sensor_2"], label="Датчик 2 (шумный, выровненный)", alpha=0.7, linewidth=0.8)
    if "truth" in df.columns:
        ax.plot(t, df["truth"], label="Истинный сигнал", color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Время")
    ax.set_ylabel("Концентрация CO")
    ax.set_title("Сырые сигналы двух датчиков")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task4_raw_sensors.png", dpi=150)


def plot_fused_kalman(df: pd.DataFrame, kalman: dict) -> None:
    """Fused estimate с uncertainty band ±2σ."""
    _ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(12, 4))
    t = df["t"]
    fused = kalman["fused"]
    std = np.sqrt(np.maximum(kalman["variance"], 0))
    ax.plot(t, df["sensor_1"], label="Датчик 1", alpha=0.35, linewidth=0.7, color="steelblue")
    ax.plot(t, df["sensor_2"], label="Датчик 2", alpha=0.35, linewidth=0.7, color="tomato")
    ax.plot(t, fused, label="Калман-фьюжн", color="darkgreen", linewidth=1.2)
    ax.fill_between(t, fused - 2 * std, fused + 2 * std,
                    alpha=0.2, color="green", label="+-2*ско")
    if "truth" in df.columns:
        ax.plot(t, df["truth"], label="Истина", color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Калман-фьюжн двух датчиков с зоной неопределенности")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task4_fused_kalman.png", dpi=150)


def plot_residuals(df: pd.DataFrame, fused: np.ndarray) -> None:
    """Остатки z_i - fused и гистограммы."""
    _ensure_fig_dir()
    res1 = df["sensor_1"].to_numpy() - fused
    res2 = df["sensor_2"].to_numpy() - fused

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    t = np.arange(len(fused))
    axes[0, 0].plot(t, res1, linewidth=0.7, color="steelblue")
    axes[0, 0].axhline(0, color="red", linestyle="--")
    axes[0, 0].set_title("Остатки: Датчик 1 минус фьюжн")
    axes[0, 1].hist(res1, bins=50, color="steelblue", edgecolor="white")
    axes[0, 1].set_title("Распределение остатков Датчика 1")

    axes[1, 0].plot(t, res2, linewidth=0.7, color="tomato")
    axes[1, 0].axhline(0, color="red", linestyle="--")
    axes[1, 0].set_title("Остатки: Датчик 2 минус фьюжн")
    axes[1, 1].hist(res2, bins=50, color="tomato", edgecolor="white")
    axes[1, 1].set_title("Распределение остатков Датчика 2")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "task4_residuals_per_sensor.png", dpi=150)


def plot_anomalies(df: pd.DataFrame, fused: np.ndarray, anomalies: pd.DataFrame) -> None:
    """Fused + аномалии."""
    _ensure_fig_dir()
    fig, ax = plt.subplots(figsize=(12, 4))
    t_vals = np.arange(len(fused))
    ax.plot(t_vals, fused, label="Калман-фьюжн", linewidth=0.9, color="darkgreen", zorder=2)

    for sensor_name, color in [("sensor_1", "steelblue"), ("sensor_2", "tomato")]:
        sub = anomalies[anomalies["sensor"] == sensor_name]
        if len(sub) > 0:
            # Map timestamps to indices
            t_col = df["t"].reset_index(drop=True)
            idx = sub["timestamp"].apply(lambda ts: (t_col == ts).idxmax())
            ax.scatter(idx, sub["value"], color=color, s=20, zorder=5,
                       label=f"Аномалия на {sensor_name} (n={len(sub)})", alpha=0.8)

    ax.set_title(f"Обнаружение аномалий (всего {len(anomalies)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task4_anomalies.png", dpi=150)


def plot_snr_improvement(residuals_info: dict) -> None:
    """Bar chart дисперсий."""
    _ensure_fig_dir()
    labels = ["Sensor 1", "Sensor 2", "Fused (Kalman)"]
    values = [
        residuals_info["var_z1"],
        residuals_info["var_z2"],
        residuals_info["var_fused"],
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["steelblue", "tomato", "darkgreen"]
    bars = ax.bar(labels, values, color=colors)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Дисперсия (относительно скользящего среднего)")
    ax.set_title("Снижение шума за счет фьюжн двух датчиков")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task4_snr_improvement.png", dpi=150)

