"""Rolling-origin forecast evaluation shared by baselines and the river TFT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from .baselines import DEFAULT_QUANTILES, score_quantiles


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Evaluation requires numpy; install environment.yml first.") from exc
    return np


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Evaluation requires pandas; install environment.yml first.") from exc
    return pd


@dataclass(frozen=True)
class OriginWindow:
    """One chronological encoder/forecast window."""

    station: str
    origin: Any
    target_times: tuple[Any, ...]
    history: Any
    target: Any


def iter_origins(
    frame: Any,
    *,
    timestamp_col: str = "dateTime",
    value_col: str = "value",
    group_col: str = "station_guid",
    encoder_length: int = 168,
    horizon: int = 24,
    step: int = 24,
    first_target: Any | None = None,
    last_target: Any | None = None,
    skip_missing_targets: bool = True,
) -> Iterable[OriginWindow]:
    """Yield rolling origins in time order for each station.

    The function does not impute. A caller can either skip a target containing
    non-finite values or retain it and apply a downstream mask.
    """

    np = _require_numpy()
    pd = _require_pandas()
    if encoder_length <= 0 or horizon <= 0 or step <= 0:
        raise ValueError("encoder_length, horizon, and step must be positive")
    required = {timestamp_col, value_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing columns: {sorted(missing)}")
    working = frame.copy()
    working[timestamp_col] = pd.to_datetime(working[timestamp_col], errors="raise")
    if getattr(working[timestamp_col].dt, "tz", None) is None:
        raise ValueError("rolling evaluation requires timezone-aware timestamps")
    if group_col not in working.columns:
        working[group_col] = "series"
    for station, group in working.groupby(group_col, sort=True):
        group = group.sort_values(timestamp_col).reset_index(drop=True)
        timestamps = group[timestamp_col].tolist()
        values = pd.to_numeric(group[value_col], errors="coerce").to_numpy(dtype=float)
        if len(values) < encoder_length + horizon:
            continue
        for target_start in range(encoder_length, len(values) - horizon + 1, step):
            target_times = tuple(timestamps[target_start : target_start + horizon])
            if first_target is not None and target_times[0] < first_target:
                continue
            if last_target is not None and target_times[-1] > last_target:
                continue
            target = values[target_start : target_start + horizon]
            if skip_missing_targets and not np.isfinite(target).all():
                continue
            yield OriginWindow(
                station=str(station),
                origin=target_times[0],
                target_times=target_times,
                history=values[target_start - encoder_length : target_start].copy(),
                target=target.copy(),
            )


def _prediction_quantiles(prediction: Any, horizon: int, n_quantiles: int = 3) -> Any:
    np = _require_numpy()
    if isinstance(prediction, dict):
        if all(key in prediction for key in ("p10", "p50", "p90")):
            prediction = np.column_stack(
                [prediction["p10"], prediction["p50"], prediction["p90"]]
            )
        elif "predictions" in prediction:
            prediction = prediction["predictions"]
    array = np.asarray(prediction, dtype=float)
    if array.ndim == 1:
        if array.size != horizon:
            raise ValueError(f"point prediction has {array.size} values; expected {horizon}")
        array = np.repeat(array[:, None], n_quantiles, axis=1)
    if array.shape != (horizon, n_quantiles):
        raise ValueError(
            f"prediction must have shape ({horizon}, {n_quantiles}), got {array.shape}"
        )
    return array


def rolling_evaluate(
    frame: Any,
    predictor: Callable[[Any, int, str], Any],
    *,
    timestamp_col: str = "dateTime",
    value_col: str = "value",
    group_col: str = "station_guid",
    encoder_length: int = 168,
    horizon: int = 24,
    step: int = 24,
    first_target: Any | None = None,
    last_target: Any | None = None,
    skip_missing_targets: bool = True,
) -> Any:
    """Run a predictor at each origin and return long forecast records."""

    pd = _require_pandas()
    records: list[dict[str, Any]] = []
    for window in iter_origins(
        frame,
        timestamp_col=timestamp_col,
        value_col=value_col,
        group_col=group_col,
        encoder_length=encoder_length,
        horizon=horizon,
        step=step,
        first_target=first_target,
        last_target=last_target,
        skip_missing_targets=skip_missing_targets,
    ):
        prediction = _prediction_quantiles(
            predictor(window.history, horizon, window.station), horizon
        )
        for offset, (target_time, actual, values) in enumerate(
            zip(window.target_times, window.target, prediction), start=1
        ):
            records.append(
                {
                    "station_guid": window.station,
                    "origin": window.origin,
                    "target_time": target_time,
                    "horizon_step": offset,
                    "actual": float(actual),
                    "p10": float(values[0]),
                    "p50": float(values[1]),
                    "p90": float(values[2]),
                    "residual": float(actual - values[1]),
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "station_guid",
            "origin",
            "target_time",
            "horizon_step",
            "actual",
            "p10",
            "p50",
            "p90",
            "residual",
        ],
    )


def summarize_forecasts(
    forecasts: Any,
    *,
    group_cols: Sequence[str] = ("station_guid", "horizon_step"),
) -> Any:
    """Calculate pinball loss and MAE by station/horizon from long records."""

    np = _require_numpy()
    pd = _require_pandas()
    required = {"actual", "p10", "p50", "p90", *group_cols}
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"forecast frame is missing columns: {sorted(missing)}")
    records: list[dict[str, Any]] = []
    for key, group in forecasts.groupby(list(group_cols), sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        actual = group["actual"].to_numpy(dtype=float)
        prediction = group[["p10", "p50", "p90"]].to_numpy(dtype=float)
        metrics = score_quantiles(actual, prediction, DEFAULT_QUANTILES)
        record = {column: value for column, value in zip(group_cols, key)}
        record.update(metrics.as_dict())
        record["n_forecasts"] = int(len(group))
        record["interval_coverage"] = float(
            np.mean((actual >= prediction[:, 0]) & (actual <= prediction[:, 2]))
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


__all__ = ["OriginWindow", "iter_origins", "rolling_evaluate", "summarize_forecasts"]
