from __future__ import annotations

from pathlib import Path

ISU: tuple[int, int] = (466264, 408607)
ISU_SUM_MOD_60: int = 11

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELL_CSV_PATH = PROJECT_ROOT / "sell.csv"
CITY_DAY_PATH = PROJECT_ROOT / "data" / "city_day.csv"

CITY: str = "Delhi"
CITY_A_COL: str = "PM2.5"
CITY_B_COL: str = "PM10"
