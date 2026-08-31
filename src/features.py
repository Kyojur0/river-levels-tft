"""Known-future calendar and observed lag features for the river study."""

from __future__ import annotations

from typing import Any, Sequence


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Feature construction requires pandas; install environment.yml first.") from exc
    return pd


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Feature construction requires numpy; install environment.yml first.") from exc
    return np


def add_calendar_features(
    frame: Any,
    *,
    timestamp_col: str = "dateTime",
    require_timezone: bool = True,
) -> Any:
    """Add deterministic calendar covariates that are known at forecast time."""

    pd = _require_pandas()
    np = _require_numpy()
    if timestamp_col not in frame.columns:
        raise ValueError(f"missing timestamp column: {timestamp_col}")
    output = frame.copy()
    timestamps = pd.to_datetime(output[timestamp_col], errors="raise")
    if require_timezone and getattr(timestamps.dt, "tz", None) is None:
        raise ValueError("calendar features require timezone-aware timestamps")
    output[timestamp_col] = timestamps
    output["hour"] = timestamps.dt.hour.astype("int16")
    output["day_of_week"] = timestamps.dt.dayofweek.astype("int16")
    output["day_of_year"] = timestamps.dt.dayofyear.astype("int16")
    output["week_of_year"] = timestamps.dt.isocalendar().week.astype("int16")
    output["is_weekend"] = (timestamps.dt.dayofweek >= 5).astype("int8")
    output["hour_sin"] = np.sin(2.0 * np.pi * output["hour"] / 24.0)
    output["hour_cos"] = np.cos(2.0 * np.pi * output["hour"] / 24.0)
    output["day_of_year_sin"] = np.sin(
        2.0 * np.pi * (output["day_of_year"] - 1) / 365.25
    )
    output["day_of_year_cos"] = np.cos(
        2.0 * np.pi * (output["day_of_year"] - 1) / 365.25
    )
    return output


def add_lag_features(
    frame: Any,
    *,
    value_col: str = "value",
    group_col: str = "station_guid",
    lags: Sequence[int] = (1, 24, 168),
) -> Any:
    """Add observed lags without looking into the forecast horizon."""

    _require_pandas()
    if value_col not in frame.columns:
        raise ValueError(f"missing value column: {value_col}")
    output = frame.copy()
    if group_col in output.columns:
        grouped = output.groupby(group_col, sort=False)[value_col]
        for lag in lags:
            if lag <= 0:
                raise ValueError("lags must be positive")
            output[f"lag_{lag}"] = grouped.shift(lag)
    else:
        for lag in lags:
            if lag <= 0:
                raise ValueError("lags must be positive")
            output[f"lag_{lag}"] = output[value_col].shift(lag)
    return output


def add_time_index(
    frame: Any,
    *,
    timestamp_col: str = "dateTime",
    group_col: str = "station_guid",
    freq: str = "h",
) -> Any:
    """Add a contiguous integer time index per station for TFT datasets."""

    pd = _require_pandas()
    if timestamp_col not in frame.columns:
        raise ValueError(f"missing timestamp column: {timestamp_col}")
    output = frame.copy()
    timestamps = pd.to_datetime(output[timestamp_col], errors="raise")
    if getattr(timestamps.dt, "tz", None) is None:
        raise ValueError("time index requires timezone-aware timestamps")
    output[timestamp_col] = timestamps
    if group_col not in output.columns:
        origin = timestamps.min().floor(freq)
        output["time_idx"] = ((timestamps - origin) / pd.Timedelta(hours=1)).astype("int64")
        return output
    output["time_idx"] = (
        output.sort_values([group_col, timestamp_col])
        .groupby(group_col, sort=False)
        .cumcount()
        .astype("int64")
    )
    return output


__all__ = ["add_calendar_features", "add_lag_features", "add_time_index"]
