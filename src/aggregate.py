"""Timezone-aware quality filtering and hourly aggregation for EA levels.

The archive returns 15-minute observations, but its naive timestamp convention
must be resolved by the caller. This module therefore refuses to localize naive
timestamps implicitly. It also keeps quality and validity masks next to the
hourly value instead of silently filling gaps.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The hourly pipeline requires pandas; install environment.yml first."
        ) from exc
    return pd


def normalize_timestamps(
    frame: Any,
    *,
    timestamp_col: str = "dateTime",
    source_timezone: str | None = None,
    target_timezone: str = "UTC",
) -> Any:
    """Return a copy with explicit timezone-aware timestamps.

    ``source_timezone`` is required when the input contains naive timestamps.
    This is intentional: the EA archive responses sampled for this project did
    not include an offset, and guessing one would shift bins around DST.
    """

    pd = _require_pandas()
    if timestamp_col not in frame.columns:
        raise ValueError(f"missing timestamp column: {timestamp_col}")
    output = frame.copy()
    parsed = pd.to_datetime(output[timestamp_col], errors="raise")
    if getattr(parsed.dt, "tz", None) is None:
        if source_timezone is None:
            raise ValueError(
                "source_timezone is required for naive archive timestamps; "
                "do not assume UTC without documenting the source convention"
            )
        parsed = parsed.dt.tz_localize(
            source_timezone, ambiguous="raise", nonexistent="raise"
        )
    output[timestamp_col] = parsed.dt.tz_convert(target_timezone)
    return output


def quality_columns(
    frame: Any,
    *,
    quality_col: str = "quality",
    qcode_col: str = "qcode",
) -> Any:
    """Normalize the live ``qcode``/documented ``qflag`` naming difference."""

    output = frame.copy()
    if qcode_col not in output.columns and "qflag" in output.columns:
        output[qcode_col] = output["qflag"]
    if quality_col not in output.columns:
        output[quality_col] = ""
    if qcode_col not in output.columns:
        output[qcode_col] = ""
    return output


def apply_quality_rule(
    frame: Any,
    *,
    value_col: str = "value",
    quality_col: str = "quality",
    include: Sequence[str] = ("Good", "Estimated"),
    mask: Sequence[str] = ("Suspect", "Unchecked", "Missing"),
) -> Any:
    """Add explicit value/quality masks without deleting source rows."""

    pd = _require_pandas()
    if value_col not in frame.columns:
        raise ValueError(f"missing value column: {value_col}")
    output = quality_columns(frame, quality_col=quality_col)
    output[value_col] = pd.to_numeric(output[value_col], errors="coerce")
    quality = output[quality_col].astype("string").fillna("")
    included = set(include)
    masked = set(mask)
    output["value_missing"] = output[value_col].isna()
    output["quality_allowed"] = quality.isin(included) & ~output["value_missing"]
    output["quality_masked"] = output["value_missing"] | quality.isin(masked) | ~quality.isin(
        included | masked
    )
    return output


def validate_grid(
    frame: Any,
    *,
    timestamp_col: str = "dateTime",
    period_seconds: int = 900,
    anchor: Any | None = None,
) -> Any:
    """Add a boolean ``grid_ok`` column for the requested regular period."""

    pd = _require_pandas()
    if period_seconds <= 0:
        raise ValueError("period_seconds must be positive")
    if timestamp_col not in frame.columns:
        raise ValueError(f"missing timestamp column: {timestamp_col}")
    output = frame.copy()
    timestamps = pd.to_datetime(output[timestamp_col], errors="raise")
    if getattr(timestamps.dt, "tz", None) is None:
        raise ValueError("validate_grid requires timezone-aware timestamps")
    if anchor is None:
        anchor = timestamps.min()
    anchor_timestamp = pd.Timestamp(anchor)
    if anchor_timestamp.tzinfo is None and getattr(timestamps.dt, "tz", None) is not None:
        anchor_timestamp = anchor_timestamp.tz_localize(timestamps.dt.tz)
    period = pd.to_timedelta(int(period_seconds), unit="s")
    # Use Timedelta arithmetic instead of integer nanoseconds: pandas may store
    # timezone-aware values at microsecond resolution on newer releases.
    output["grid_ok"] = (timestamps - anchor_timestamp) % period == pd.Timedelta(0)
    return output


def validate_no_duplicates(
    frame: Any,
    *,
    timestamp_col: str = "dateTime",
    group_cols: Iterable[str] = ("station_guid", "measure_id"),
) -> None:
    """Raise with an actionable count when a station has duplicate timestamps."""

    columns = [column for column in group_cols if column in frame.columns]
    keys = columns + [timestamp_col]
    if frame.duplicated(keys).any():
        count = int(frame.duplicated(keys, keep=False).sum())
        raise ValueError(f"duplicate timestamp rows detected ({count} rows) for keys {keys}")


def prepare_level_frame(
    frame: Any,
    *,
    source_timezone: str | None = None,
    target_timezone: str = "UTC",
    period_seconds: int = 900,
    reject_duplicates: bool = True,
    reject_off_grid: bool = True,
    include_quality: Sequence[str] = ("Good", "Estimated"),
    mask_quality: Sequence[str] = ("Suspect", "Unchecked", "Missing"),
) -> Any:
    """Normalize, quality-tag, and validate an EA 15-minute level frame."""

    output = normalize_timestamps(
        frame,
        source_timezone=source_timezone,
        target_timezone=target_timezone,
    )
    output = apply_quality_rule(
        output,
        include=include_quality,
        mask=mask_quality,
    )
    output = validate_grid(output, period_seconds=period_seconds)
    if reject_duplicates:
        validate_no_duplicates(output)
    if reject_off_grid and not bool(output["grid_ok"].all()):
        count = int((~output["grid_ok"]).sum())
        raise ValueError(f"off-grid timestamps detected ({count} rows)")
    return output.sort_values("dateTime").reset_index(drop=True)


def aggregate_hourly(
    frame: Any,
    *,
    timestamp_col: str = "dateTime",
    value_col: str = "value",
    group_cols: Sequence[str] = ("station_guid", "measure_id"),
    expected_per_bin: int = 4,
    min_valid: int = 3,
) -> Any:
    """Aggregate 15-minute rows to hourly means with explicit validity fields."""

    pd = _require_pandas()
    if expected_per_bin <= 0 or min_valid <= 0 or min_valid > expected_per_bin:
        raise ValueError("require 0 < min_valid <= expected_per_bin")
    if value_col not in frame.columns or timestamp_col not in frame.columns:
        raise ValueError(f"frame must contain {timestamp_col!r} and {value_col!r}")
    timestamps = pd.to_datetime(frame[timestamp_col], errors="raise")
    if getattr(timestamps.dt, "tz", None) is None:
        raise ValueError("aggregate_hourly requires timezone-aware timestamps")

    output = frame.copy()
    output[timestamp_col] = timestamps
    output[value_col] = pd.to_numeric(output[value_col], errors="coerce")
    if "quality_allowed" not in output.columns:
        output = apply_quality_rule(output, value_col=value_col)
    output["_hour"] = output[timestamp_col].dt.floor("h")
    keys = [column for column in group_cols if column in output.columns]
    keys.append("_hour")
    records: list[dict[str, Any]] = []
    for key, group in output.groupby(keys, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        valid = group["quality_allowed"] & group[value_col].notna()
        values = group.loc[valid, value_col]
        record = {column: value for column, value in zip(keys, key)}
        record.update(
            {
                timestamp_col: record.pop("_hour"),
                value_col: float(values.mean()) if len(values) >= min_valid else float("nan"),
                "n_observations": int(len(group)),
                "n_valid": int(valid.sum()),
                "aggregation_valid": bool(len(values) >= min_valid),
                "quality_masked": bool((~valid).any()),
                "quality_values": "|".join(
                    sorted(
                        {
                            str(value)
                            for value in group.get("quality", pd.Series(dtype="string"))
                            if value is not None and str(value)
                        }
                    )
                ),
            }
        )
        records.append(record)
    if not records:
        columns = keys[:-1] + [
            timestamp_col,
            value_col,
            "n_observations",
            "n_valid",
            "aggregation_valid",
            "quality_masked",
            "quality_values",
        ]
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame.from_records(records)
        .sort_values(keys[:-1] + [timestamp_col])
        .reset_index(drop=True)
    )


__all__ = [
    "aggregate_hourly",
    "apply_quality_rule",
    "normalize_timestamps",
    "prepare_level_frame",
    "quality_columns",
    "validate_grid",
    "validate_no_duplicates",
]
