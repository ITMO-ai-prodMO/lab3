from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pandas as pd


KAGGLE_DATASET = "krishujeniya/fitness-tracker-accelerometer-and-gyroscope-data"
SENSOR_COLUMNS = [
    "timestamp",
    "accelerometer_x",
    "accelerometer_y",
    "accelerometer_z",
    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z",
]


def ensure_fitness_tracker_dataset(data_dir: str | Path = "data/hard") -> Path:
    """Download and extract the Kaggle fitness tracker dataset if needed.

    The function uses the Kaggle CLI, so the machine must have Kaggle API
    credentials configured. If the data is already present, no network request
    is made.

    Args:
        data_dir: Directory where the dataset archive and extracted files live.

    Returns:
        Path to the extracted dataset directory.
    """
    target_dir = Path(data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    extracted_dir = target_dir / "fitness_tracker"
    if extracted_dir.exists() and any(extracted_dir.rglob("*.csv")):
        return extracted_dir

    archive_path = target_dir / f"{KAGGLE_DATASET.split('/')[-1]}.zip"
    if not archive_path.exists():
        command = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "-p",
            str(target_dir),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("Kaggle CLI is not installed or not available in PATH") from exc
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.strip() or exc.stdout.strip()
            raise RuntimeError(f"Kaggle dataset download failed: {message}") from exc

    extracted_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted_dir)

    return extracted_dir


def load_fitness_tracker_data(dataset_dir: str | Path, max_rows: int | None = 5000) -> pd.DataFrame:
    """Load accelerometer and gyroscope time series from the extracted dataset."""
    dataset_path = Path(dataset_dir)
    csv_path = _find_sensor_csv(dataset_path)
    raw_data = pd.read_csv(csv_path, nrows=max_rows)
    normalized = _normalize_sensor_columns(raw_data)
    normalized = normalized.dropna(subset=SENSOR_COLUMNS[1:]).reset_index(drop=True)
    normalized["sample_index"] = normalized.index
    normalized["acceleration_magnitude"] = (
        normalized[["accelerometer_x", "accelerometer_y", "accelerometer_z"]].pow(2).sum(axis=1) ** 0.5
    )
    normalized["gyroscope_magnitude"] = (
        normalized[["gyroscope_x", "gyroscope_y", "gyroscope_z"]].pow(2).sum(axis=1) ** 0.5
    )
    return normalized


def _find_sensor_csv(dataset_dir: Path) -> Path:
    for csv_path in dataset_dir.rglob("*.csv"):
        header = pd.read_csv(csv_path, nrows=0)
        normalized_columns = {_normalize_column_name(column) for column in header.columns}
        if {"accelerometer_x", "gyroscope_x"}.issubset(normalized_columns):
            return csv_path
    raise FileNotFoundError("Could not find a CSV file with accelerometer and gyroscope columns")


def _normalize_sensor_columns(data: pd.DataFrame) -> pd.DataFrame:
    renamed = data.rename(columns={column: _normalize_column_name(column) for column in data.columns})

    timestamp_candidates = ["timestamp", "epoch_ms", "epoch", "time"]
    timestamp_column = next((column for column in timestamp_candidates if column in renamed.columns), None)
    if timestamp_column is None:
        renamed["timestamp"] = renamed.index
    elif timestamp_column != "timestamp":
        renamed = renamed.rename(columns={timestamp_column: "timestamp"})

    missing = [column for column in SENSOR_COLUMNS if column not in renamed.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    result = renamed[SENSOR_COLUMNS].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    if result["timestamp"].isna().all():
        result["timestamp"] = renamed["timestamp"]

    for column in SENSOR_COLUMNS[1:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _normalize_column_name(column: str) -> str:
    cleaned = column.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "epoch_(ms)": "timestamp",
        "epoch_ms": "timestamp",
        "accelerometer_x": "accelerometer_x",
        "accelerometer_y": "accelerometer_y",
        "accelerometer_z": "accelerometer_z",
        "gyroscope_x": "gyroscope_x",
        "gyroscope_y": "gyroscope_y",
        "gyroscope_z": "gyroscope_z",
    }
    return aliases.get(cleaned, cleaned)
