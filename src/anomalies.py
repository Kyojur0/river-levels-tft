"""Retrospective residual and interval anomaly review.

The outputs describe model misses for offline analysis. They are deliberately
not alerting logic and must not be used as a flood-warning service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Anomaly review requires numpy; install environment.yml first.") from exc
    return np


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Anomaly review requires pandas; install environment.yml first.") from exc
    return pd


@dataclass(frozen=True)
class ResidualReference:
    """Robust residual location/scale for one station and horizon."""

    station_guid: str
    horizon_step: int
    median: float
    scale: float
    n_values: int


def fit_residual_references(
    forecasts: Any,
    *,
    residual_col: str = "residual",
    group_cols: Sequence[str] = ("station_guid", "horizon_step"),
    min_scale: float = 1e-9,
) -> Any:
    """Fit median/MAD references on a training-only forecast frame."""

    np = _require_numpy()
    pd = _require_pandas()
    required = {residual_col, *group_cols}
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"forecast frame is missing columns: {sorted(missing)}")
    records: list[dict[str, Any]] = []
    for key, group in forecasts.groupby(list(group_cols), sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        residuals = pd.to_numeric(group[residual_col], errors="coerce").to_numpy(dtype=float)
        residuals = residuals[np.isfinite(residuals)]
        if residuals.size == 0:
            continue
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        scale = max(1.4826 * mad, min_scale)
        record = {column: value for column, value in zip(group_cols, key)}
        record.update({"residual_median": median, "residual_scale": scale, "n_values": int(residuals.size)})
        records.append(record)
    return pd.DataFrame.from_records(records)


def score_residual_anomalies(
    forecasts: Any,
    references: Any,
    *,
    actual_col: str = "actual",
    p10_col: str = "p10",
    p50_col: str = "p50",
    p90_col: str = "p90",
    group_cols: Sequence[str] = ("station_guid", "horizon_step"),
    threshold: float = 3.5,
) -> Any:
    """Flag out-of-interval and robust-residual events for retrospective review.

    An event is ``outside_interval`` when the actual lies below P10 or above
    P90. It is ``large_residual`` when its training-calibrated robust score is
    at least ``threshold``. The combined ``anomaly`` flag is their OR.
    """

    np = _require_numpy()
    pd = _require_pandas()
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    required = {actual_col, p10_col, p50_col, p90_col, *group_cols}
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"forecast frame is missing columns: {sorted(missing)}")
    reference_cols = list(group_cols) + ["residual_median", "residual_scale"]
    missing_reference = set(reference_cols) - set(references.columns)
    if missing_reference:
        raise ValueError(f"reference frame is missing columns: {sorted(missing_reference)}")
    output = forecasts.copy()
    output["residual"] = output[actual_col] - output[p50_col]
    output = output.merge(
        references[reference_cols], on=list(group_cols), how="left", validate="many_to_one"
    )
    output["outside_interval"] = (output[actual_col] < output[p10_col]) | (
        output[actual_col] > output[p90_col]
    )
    output["residual_score"] = (
        (output["residual"] - output["residual_median"]).abs()
        / output["residual_scale"].replace(0, np.nan)
    )
    output["large_residual"] = output["residual_score"] >= threshold
    output["anomaly"] = output["outside_interval"] | output["large_residual"]
    output["review_reason"] = ""
    output.loc[output["outside_interval"], "review_reason"] = "outside_p10_p90"
    both = output["outside_interval"] & output["large_residual"]
    output.loc[both, "review_reason"] = "outside_p10_p90|large_residual"
    output.loc[~output["outside_interval"] & output["large_residual"], "review_reason"] = "large_residual"
    return output


def summarize_anomalies(
    scored: Any,
    *,
    group_cols: Sequence[str] = ("station_guid",),
) -> Any:
    """Summarize retrospective event counts and rates by station or horizon."""

    pd = _require_pandas()
    required = {"anomaly", "outside_interval", "large_residual", *group_cols}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"scored frame is missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for key, group in scored.groupby(list(group_cols), sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        record = {column: value for column, value in zip(group_cols, key)}
        n = len(group)
        record.update(
            {
                "n_forecasts": int(n),
                "anomaly_count": int(group["anomaly"].sum()),
                "outside_interval_count": int(group["outside_interval"].sum()),
                "large_residual_count": int(group["large_residual"].sum()),
                "anomaly_rate": float(group["anomaly"].mean()) if n else float("nan"),
            }
        )
        rows.append(record)
    return pd.DataFrame.from_records(rows)


__all__ = [
    "ResidualReference",
    "fit_residual_references",
    "score_residual_anomalies",
    "summarize_anomalies",
]
