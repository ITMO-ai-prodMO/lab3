from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SELL_COLUMNS = ("soap", "powder", "cleaner", "paint", "foam", "profit")
AIR_SENSOR_COLUMNS = {
    "co_sensor": "PT08.S1(CO)",
    "ozone_sensor": "PT08.S5(O3)",
}


def isu_variant(isu: int | str, variants_count: int = 60) -> int:
    digits = [int(symbol) for symbol in str(isu) if symbol.isdigit()]
    if not digits:
        raise ValueError()
    remainder = sum(digits) % variants_count
    return remainder or variants_count


def load_sell_variant(path: str | Path, *, variant: int | None = None, isu: int | str | None = None) -> pd.DataFrame:
    if variant is None:
        if isu is None:
            raise ValueError()
        variant = isu_variant(isu)
    if not 1 <= variant <= 60:
        raise ValueError()

    raw = pd.read_csv(path, sep=";", header=None, skiprows=3, dtype=str, encoding="cp1251")
    variant_start = 1 + (variant - 1) * len(SELL_COLUMNS)
    selected_positions = [0, *range(variant_start, variant_start + len(SELL_COLUMNS))]
    selected = raw.iloc[:, selected_positions].copy()
    selected.columns = ("day", *SELL_COLUMNS)

    for column in selected.columns:
        text = selected[column].astype(str).str.replace(",", ".", regex=False)
        selected[column] = pd.to_numeric(text, errors="coerce")

    selected = selected.dropna(subset=["day"]).reset_index(drop=True)
    selected["day"] = selected["day"].astype(int)
    selected.attrs["variant"] = variant
    return selected


def load_air_quality_sensors(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";", decimal=",")
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    frame = frame.replace(-200, np.nan)
    timestamp = pd.to_datetime(
        frame["Date"].astype(str) + " " + frame["Time"].astype(str),
        format="%d/%m/%Y %H.%M.%S",
        errors="coerce",
    )
    frame = frame.assign(timestamp=timestamp).dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    sensors = frame[list(AIR_SENSOR_COLUMNS.values())].rename(
        columns={source: name for name, source in AIR_SENSOR_COLUMNS.items()}
    )
    sensors = sensors.apply(pd.to_numeric, errors="coerce").interpolate(method="time", limit_direction="both")
    return sensors.dropna()
