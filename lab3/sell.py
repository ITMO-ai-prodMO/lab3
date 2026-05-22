from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SellVariantData:
    days: np.ndarray
    series: dict[str, np.ndarray]


def _parse_ru_float(s: str) -> float:
    s = s.strip()
    if not s:
        return float("nan")
    return float(s.replace(",", "."))


def load_sell_csv_variant(path: str | Path, variant: int) -> SellVariantData:
    variant = int(variant)
    if not (1 <= variant <= 60):
        raise ValueError("variant must be in 1..60")

    path = Path(path)

    offset = 1 + (variant - 1) * 6
    names = ["мыло", "порошок", "средство", "краска", "пена", "прибыль"]

    days: list[int] = []
    cols: dict[str, list[float]] = {k: [] for k in names}

    def _read_with_encoding(enc: str) -> None:
        with path.open("r", encoding=enc, newline="") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader, None)
            next(reader, None)
            next(reader, None)

            for row in reader:
                if not row:
                    continue
                if len(row) < offset + 6:
                    continue
                day_s = row[0].strip()
                if not day_s:
                    continue
                days.append(int(day_s))
                vals = row[offset : offset + 6]
                for k, vs in zip(names, vals, strict=True):
                    cols[k].append(_parse_ru_float(vs))

    _read_with_encoding("cp1251")

    days_arr = np.asarray(days, dtype=int)
    series_arr = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
    return SellVariantData(days=days_arr, series=series_arr)
