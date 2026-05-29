from __future__ import annotations

from pathlib import Path

import pandas as pd


PRODUCT_COLUMNS = ["soap", "powder", "cleaner", "paint", "foam", "profit"]
PRODUCT_LABELS = {
    "soap": "Мыло",
    "powder": "Порошок",
    "cleaner": "Средство",
    "paint": "Краска",
    "foam": "Пена",
    "profit": "Прибыль",
}


def isu_to_variant(isu_number: int, variants_count: int = 60) -> int:
    """Convert an ISU number to a lab variant number.

    The assignment uses ``ISU mod 60``. If the remainder is zero, the last
    variant is used.
    """
    if isu_number <= 0:
        raise ValueError("isu_number must be positive")
    if variants_count <= 0:
        raise ValueError("variants_count must be positive")

    remainder = isu_number % variants_count
    return variants_count if remainder == 0 else remainder


def load_variant_data(csv_path: str | Path, variant: int) -> pd.DataFrame:
    """Load one variant block from the sales CSV file.

    Args:
        csv_path: Path to ``sell.csv``.
        variant: Variant number from 1 to 60.

    Returns:
        DataFrame with columns ``day`` and product/profit columns.
    """
    if not 1 <= variant <= 60:
        raise ValueError("variant must be in range 1..60")

    raw_data = pd.read_csv(
        csv_path,
        sep=";",
        header=None,
        skiprows=3,
        decimal=",",
        encoding="utf-8",
    )

    start_column = 1 + (variant - 1) * len(PRODUCT_COLUMNS)
    selected_columns = [0, *range(start_column, start_column + len(PRODUCT_COLUMNS))]
    data = raw_data.iloc[:, selected_columns].copy()
    data.columns = ["day", *PRODUCT_COLUMNS]

    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    return data
