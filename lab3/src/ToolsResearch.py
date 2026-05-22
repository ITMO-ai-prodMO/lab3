from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .DataProcessing import load_air_quality_sensors, load_sell_variant
from .Filters import apply_filters
from .TimeSeriesAnalysis import autocorrelation, summarize_frame


ISU = 464938
FILTER_LABELS = {
    "raw": "Raw",
    "moving_average": "Moving average",
    "kalman": "Kalman",
    "savitzky_golay": "Savitzky-Golay",
    "haar_wavelet": "Haar wavelet",
    "lms": "LMS predictor",
}


def _prepared_signal(series: pd.Series) -> np.ndarray:
    return series.astype(float).interpolate(limit_direction="both").to_numpy()


def filtered_frame(series: pd.Series) -> pd.DataFrame:
    values = apply_filters(_prepared_signal(series))
    return pd.DataFrame(values, index=series.index)


def _roughness(values: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(values)))) if values.size > 1 else 0.0


def filter_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for series_name in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[series_name]):
            continue
        filtered = filtered_frame(frame[series_name])
        raw = filtered["raw"].to_numpy()
        raw_roughness = max(_roughness(raw), 1e-12)
        for method in filtered.columns:
            values = filtered[method].to_numpy()
            rows.append(
                {
                    "series": series_name,
                    "method": method,
                    "roughness": _roughness(values),
                    "roughness_ratio_to_raw": _roughness(values) / raw_roughness,
                    "mean_abs_change_from_raw": float(np.mean(np.abs(values - raw))),
                    "std": float(np.std(values)),
                }
            )
    return pd.DataFrame(rows)


def _plot_filtered_axes(ax: plt.Axes, series: pd.Series, title: str) -> None:
    filtered = filtered_frame(series)
    index = filtered.index
    ax.plot(index, filtered["raw"], color="#263238", linewidth=1.0, alpha=0.6, label=FILTER_LABELS["raw"])
    for method, color in zip(
        filtered.columns[1:],
        ("#c62828", "#1565c0", "#2e7d32", "#6a1b9a", "#ef6c00"),
    ):
        ax.plot(index, filtered[method], linewidth=1.35, label=FILTER_LABELS[method], color=color)
    ax.set_title(title)
    ax.grid(alpha=0.22)


def _save_sell_filter_plot(sell: pd.DataFrame, path: Path) -> None:
    plotted_columns = [column for column in sell.columns if column != "day"]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    for ax, column in zip(axes.flat, plotted_columns):
        indexed = pd.Series(sell[column].to_numpy(), index=sell["day"], name=column)
        _plot_filtered_axes(ax, indexed, column)
        ax.set_xlabel("Day")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Variant 34: sell.csv time-series filtering", fontsize=16)
    fig.tight_layout(rect=(0, 0.065, 1, 0.965))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_autocorrelation_plot(sell: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey=True)
    for ax, column in zip(axes.flat, [column for column in sell.columns if column != "day"]):
        correlations = autocorrelation(sell[column], max_lag=min(24, len(sell) - 1))
        ax.stem(correlations.index, correlations.to_numpy(), basefmt=" ")
        ax.axhline(0.0, color="#263238", linewidth=0.8)
        ax.set_title(column)
        ax.set_xlabel("Lag, days")
        ax.grid(alpha=0.2)
    fig.suptitle("Variant 34: autocorrelation by sales series", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_correlation_plot(correlation: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(correlation, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    ax.set_xticks(np.arange(len(correlation.columns)), labels=correlation.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(correlation.index)), labels=correlation.index)
    for row in range(len(correlation.index)):
        for column in range(len(correlation.columns)):
            ax.text(column, row, f"{correlation.iloc[row, column]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def build_sell_assets(lab_root: Path) -> dict[str, pd.DataFrame]:
    assets = lab_root / "report_assets"
    assets.mkdir(exist_ok=True)

    sell = load_sell_variant(lab_root / "sell.csv", isu=ISU)
    sell_series = sell.drop(columns="day")
    summary = summarize_frame(sell_series, max_lag=min(24, len(sell) - 1))
    correlations = sell_series.corr()
    metrics = filter_metrics(sell_series)

    sell.to_csv(assets / "sell_variant_34.csv", index=False)
    summary.to_csv(assets / "sell_series_summary.csv", index=False)
    correlations.to_csv(assets / "sell_correlations.csv")
    metrics.to_csv(assets / "sell_filter_metrics.csv", index=False)

    _save_sell_filter_plot(sell, assets / "sell_filter_comparison.png")
    _save_autocorrelation_plot(sell, assets / "sell_autocorrelation.png")
    _save_correlation_plot(correlations, assets / "sell_correlation_heatmap.png", "Variant 34: sell.csv correlations")
    return {"sell": sell, "summary": summary, "correlations": correlations, "metrics": metrics}


def _save_air_filter_plot(sensors: pd.DataFrame, path: Path, plotted_rows: int = 24 * 14) -> None:
    selection = sensors.iloc[:plotted_rows]
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    for ax, column in zip(axes, selection.columns):
        _plot_filtered_axes(ax, selection[column], column)
        ax.set_xlabel("Timestamp")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("UCI Air Quality: two sensor channels, first 14 days", fontsize=16)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def build_air_quality_assets(lab_root: Path) -> dict[str, pd.DataFrame]:
    assets = lab_root / "report_assets"
    assets.mkdir(exist_ok=True)

    sensors = load_air_quality_sensors(lab_root / "data" / "AirQualityUCI.csv")
    summary = summarize_frame(sensors, min_period_lag=12, max_lag=24 * 14)
    correlations = sensors.corr()
    metrics = filter_metrics(sensors)

    summary.to_csv(assets / "air_sensor_summary.csv", index=False)
    correlations.to_csv(assets / "air_sensor_correlations.csv")
    metrics.to_csv(assets / "air_sensor_filter_metrics.csv", index=False)

    _save_air_filter_plot(sensors, assets / "air_sensor_filter_comparison.png")
    _save_correlation_plot(correlations, assets / "air_sensor_correlation_heatmap.png", "UCI Air Quality sensor correlation")
    return {"sensors": sensors, "summary": summary, "correlations": correlations, "metrics": metrics}


def build_report_assets(lab_root: Path | None = None) -> dict[str, dict[str, pd.DataFrame]]:
    resolved_root = lab_root or Path(__file__).resolve().parents[1]
    return {
        "sell": build_sell_assets(resolved_root),
        "air_quality": build_air_quality_assets(resolved_root),
    }


if __name__ == "__main__":
    build_report_assets()
