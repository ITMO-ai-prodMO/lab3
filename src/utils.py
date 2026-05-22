"""Утилиты загрузки и подготовки данных лабораторной №3.

Источник: ``sell.csv`` — таблица продаж 60 вариантов товара
за 50 дней. На каждый вариант приходится 6 столбцов:
``мыло, порошок, средство, краска, пена`` (штуки) и
``прибыль`` (тыс. руб.).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

PRODUCT_COLS: tuple[str, ...] = ("мыло", "порошок", "средство", "краска", "пена")
PROFIT_COL: str = "прибыль"
ALL_METRIC_COLS: tuple[str, ...] = PRODUCT_COLS + (PROFIT_COL,)


def seed_everything(seed: int = 42) -> None:
    """Зафиксировать seed для воспроизводимости (см. инструкцию §3.1)."""
    random.seed(seed)
    np.random.seed(seed)


def compute_variant(isu_numbers: list[int]) -> int:
    """Вычислить номер варианта = сумма ИСУ mod 60.

    Если сумма даёт 0, возвращаем 60 (так как варианты пронумерованы с 1).
    """
    total = sum(isu_numbers)
    variant = total % 60
    return variant if variant != 0 else 60


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Прочитать sell.csv в исходном многоуровневом виде."""
    return pd.read_csv(
        csv_path,
        sep=";",
        encoding="cp1251",
        header=[0, 1, 2],
        decimal=",",
    )


def _slice_variant(raw: pd.DataFrame, variant: int) -> pd.DataFrame:
    """Извлечь 6 столбцов варианта по позиции (1-й столбец = day, далее по 6 на вариант)."""
    if not 1 <= variant <= 60:
        raise ValueError(f"variant must be in [1, 60], got {variant}")
    start = 1 + (variant - 1) * 6
    sub = raw.iloc[:, start : start + 6].copy()
    # 2-й уровень MultiIndex'а — осмысленные имена метрик
    sub.columns = [c[1] for c in sub.columns]
    sub = sub[list(ALL_METRIC_COLS)]
    sub.index = pd.RangeIndex(start=1, stop=len(sub) + 1, name="day")
    return sub.astype({c: float for c in ALL_METRIC_COLS})


def load_variant(csv_path: str | Path, variant: int) -> pd.DataFrame:
    """DataFrame длиной 50 дней с колонками ``мыло, порошок, средство, краска, пена, прибыль``."""
    raw = load_raw(csv_path)
    return _slice_variant(raw, variant)


def load_all_variants(csv_path: str | Path) -> dict[int, pd.DataFrame]:
    """Все 60 вариантов в виде словаря {variant_number: DataFrame}."""
    raw = load_raw(csv_path)
    return {v: _slice_variant(raw, v) for v in range(1, 61)}
