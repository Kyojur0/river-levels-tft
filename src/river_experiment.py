"""Bounded baseline/TFT experiment runner for the frozen hourly EA data.

The runner consumes one hourly CSV per station from data/hourly, applies the
chronological boundaries in configs/splits.yml, and evaluates a bounded number
of rolling test origins. It uses the shared evaluate.py and baselines.py
implementations. Optional ETS, LightGBM, and TFT paths are reported as
unavailable when their dependencies or implementation are absent; no score is
invented for a missing model.

No result file is written unless --results-json is supplied.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
import os
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    import pandas as pd
except ImportError:  # pragma: no cover - depends on project environment
    pd = None  # type: ignore[assignment]

try:
    import yaml
except ImportError:  # pragma: no cover - clear runtime error is preferable
    yaml = None  # type: ignore[assignment]

try:
    from . import evaluate
    from .baselines import (
        OptionalBaselineUnavailable,
        ets_forecast,
        lightgbm_direct_forecast,
        lightgbm_recursive_forecast,
        score_quantiles,
        seasonal_naive_forecast,
    )
except ImportError:  # pragma: no cover - supports python src/river_experiment.py
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import evaluate  # type: ignore[no-redef]
    from src.baselines import (  # type: ignore[no-redef]
        OptionalBaselineUnavailable,
        ets_forecast,
        lightgbm_direct_forecast,
        lightgbm_recursive_forecast,
        score_quantiles,
        seasonal_naive_forecast,
    )


DEFAULT_DATA_DIR = Path("data/hourly")
DEFAULT_SPLITS = Path("configs/splits.yml")
DEFAULT_RESULTS = Path("results/river_experiment.json")
DEFAULT_MAX_WINDOWS = 16
DEFAULT_SEASONALITY = 24
DEFAULT_GROUP_COL = "station_guid"
DEFAULT_TIMESTAMP_COL = "dateTime"
DEFAULT_VALUE_COL = "value"


class ExperimentError(RuntimeError):
    """Raised when input data/configuration cannot support a valid run."""


def _require_pandas() -> Any:
    if pd is None:
        raise ExperimentError(
            "river_experiment requires pandas; install environment.yml first"
        )
    return pd


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy values and NaN/inf to JSON-safe primitives."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _load_mapping(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ExperimentError(f"configuration file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExperimentError(f"cannot read configuration {path}: {exc}") from exc
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ExperimentError(f"invalid JSON configuration {path}") from exc
    else:
        if yaml is None:
            value = _simple_yaml_mapping(text)
        else:
            try:
                value = yaml.safe_load(text)
            except Exception as exc:
                raise ExperimentError(f"invalid YAML configuration {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ExperimentError(f"configuration root must be a mapping: {path}")
    return dict(value)


def _simple_yaml_scalar(value: str) -> Any:
    """Parse the small scalar subset used by this project's YAML configs."""

    value = value.strip()
    if not value:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "~"}:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _simple_yaml_mapping(text: str) -> dict[str, Any]:
    """Fallback parser for flat splits and station catalogue metadata.

    The declared environment includes PyYAML, but the runner remains usable in
    minimal smoke environments. This parser intentionally handles only the
    project's scalar keys and station list; it does not pretend to be a general
    YAML implementation.
    """

    result: dict[str, Any] = {}
    stations: list[dict[str, Any]] = []
    in_stations = False
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        line = raw_line.split("#", 1)[0].rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "stations:":
            in_stations = True
            result["stations"] = stations
            continue
        if in_stations and stripped.startswith("- "):
            current = {}
            stations.append(current)
            key_value = stripped[2:].strip()
            if ":" in key_value:
                key, value = key_value.split(":", 1)
                current[key.strip()] = _simple_yaml_scalar(value)
            continue
        if in_stations and current is not None and indent > 0 and ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            # Crosswalk keys are irrelevant to the hourly runner; retaining
            # scalar station fields avoids guessing any identifier.
            if key in {"station_guid", "name", "river", "role", "measure_id"}:
                current[key] = _simple_yaml_scalar(value)
            continue
        if ":" in stripped and indent == 0:
            key, value = stripped.split(":", 1)
            result[key.strip()] = _simple_yaml_scalar(value)
            in_stations = False
    if not result:
        raise ExperimentError("fallback YAML parser found no scalar settings")
    return result


def _date_value(value: Any, key: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ExperimentError(f"{key} must be YYYY-MM-DD") from exc
    raise ExperimentError(f"{key} is missing or is not a date")


def _load_station_names(path: Path | str) -> dict[str, dict[str, str]]:
    """Read names/roles from stations.yml without inferring any IDs."""

    config = _load_mapping(path)
    stations = config.get("stations")
    if not isinstance(stations, list):
        raise ExperimentError("stations config must contain a list")
    result: dict[str, dict[str, str]] = {}
    for raw in stations:
        if not isinstance(raw, Mapping):
            raise ExperimentError("each station config entry must be a mapping")
        guid = str(raw.get("station_guid") or "").strip()
        if not guid:
            raise ExperimentError("station config entry is missing station_guid")
        if guid in result:
            raise ExperimentError(f"duplicate station_guid in station config: {guid}")
        result[guid] = {
            "name": str(raw.get("name") or guid),
            "river": str(raw.get("river") or ""),
            "role": str(raw.get("role") or ""),
            "measure_id": str(raw.get("measure_id") or ""),
        }
    return result


def load_split_config(path: Path | str = DEFAULT_SPLITS) -> dict[str, Any]:
    """Load and validate the split dates/lengths used for evaluation."""

    config = _load_mapping(path)
    data_start = _date_value(config.get("data_start"), "data_start")
    test_end = _date_value(config.get("test_end_exclusive"), "test_end_exclusive")
    train_end = _date_value(config.get("train_end_exclusive"), "train_end_exclusive")
    validation_end = _date_value(
        config.get("validation_end_exclusive"), "validation_end_exclusive"
    )
    if not data_start < train_end <= validation_end < test_end:
        raise ExperimentError(
            "split dates must satisfy data_start < train_end <= "
            "validation_end < test_end_exclusive"
        )
    try:
        encoder_hours = int(config.get("encoder_hours", 168))
        horizon_hours = int(config.get("horizon_hours", 24))
        step_hours = int(config.get("rolling_origin_step_hours", 24))
    except (TypeError, ValueError) as exc:
        raise ExperimentError("encoder/horizon/rolling step must be integers") from exc
    if encoder_hours <= 0 or horizon_hours <= 0 or step_hours <= 0:
        raise ExperimentError("encoder/horizon/rolling step must be positive")
    return {
        "data_start": data_start,
        "train_end_exclusive": train_end,
        "validation_end_exclusive": validation_end,
        "test_end_exclusive": test_end,
        "encoder_hours": encoder_hours,
        "horizon_hours": horizon_hours,
        "rolling_origin_step_hours": step_hours,
    }


def _to_bool(series: Any) -> Any:
    pd_module = _require_pandas()
    if getattr(series, "dtype", None) == bool:
        return series.fillna(False)
    lowered = series.astype("string").str.strip().str.lower()
    return lowered.isin({"1", "true", "yes", "y", "t"})


def _station_input_summary(
    frame: Any,
    *,
    station_guid: str,
    path: Path,
    data_start: date,
    test_end: date,
) -> dict[str, Any]:
    pd_module = _require_pandas()
    subset = frame[
        (frame[DEFAULT_TIMESTAMP_COL] >= pd_module.Timestamp(data_start, tz="UTC"))
        & (frame[DEFAULT_TIMESTAMP_COL] < pd_module.Timestamp(test_end, tz="UTC"))
    ]
    values = pd_module.to_numeric(subset[DEFAULT_VALUE_COL], errors="coerce")
    return {
        "station_guid": station_guid,
        "file": str(path),
        "rows": int(len(subset)),
        "first_timestamp": (
            subset[DEFAULT_TIMESTAMP_COL].min().isoformat() if len(subset) else None
        ),
        "last_timestamp": (
            subset[DEFAULT_TIMESTAMP_COL].max().isoformat() if len(subset) else None
        ),
        "missing_values": int(values.isna().sum()),
        "finite_values": int(values.notna().sum()),
        "quality_masked_rows": int(
            _to_bool(subset["quality_masked"]).sum()
            if "quality_masked" in subset
            else 0
        ),
        "aggregation_invalid_rows": int(
            (~_to_bool(subset["aggregation_valid"])).sum()
            if "aggregation_valid" in subset
            else 0
        ),
    }


def load_hourly_data(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    *,
    station_guids: Sequence[str] | None = None,
    station_config: Path | str = Path("configs/stations.yml"),
    data_start: date | None = None,
    test_end: date | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Load and validate hourly station files.

    Missing/invalid aggregates are represented as NaN; no imputation occurs.
    Timestamp axes must be timezone-aware and exactly hourly so positional
    rolling windows cannot silently cross a gap.
    """

    pd_module = _require_pandas()
    data_root = Path(data_dir)
    if not data_root.exists():
        raise ExperimentError(f"hourly data directory not found: {data_root}")
    names = _load_station_names(station_config)
    selected = list(station_guids) if station_guids else list(names)
    unknown = [guid for guid in selected if guid not in names]
    if unknown:
        raise ExperimentError(
            "station GUID(s) not present in station config: " + ", ".join(unknown)
        )
    frames: list[Any] = []
    summaries: list[dict[str, Any]] = []
    for guid in selected:
        path = data_root / f"{guid}.csv"
        if not path.exists():
            raise ExperimentError(f"hourly station file not found: {path}")
        try:
            frame = pd_module.read_csv(path)
        except Exception as exc:
            raise ExperimentError(f"cannot read hourly file {path}: {exc}") from exc
        required = {DEFAULT_TIMESTAMP_COL, DEFAULT_VALUE_COL}
        missing = required - set(frame.columns)
        if missing:
            raise ExperimentError(f"{path} is missing columns: {sorted(missing)}")
        if "station_guid" not in frame.columns:
            frame["station_guid"] = guid
        frame["station_guid"] = frame["station_guid"].astype(str)
        if set(frame["station_guid"].dropna()) != {guid}:
            raise ExperimentError(f"{path} contains a station_guid mismatch")
        # Keep the verified station metadata alongside the numeric panel so a
        # pooled TFT can use it as a static categorical when requested.
        frame["station_name"] = names[guid]["name"]
        frame["river"] = names[guid]["river"]
        frame["role"] = names[guid]["role"]
        frame[DEFAULT_TIMESTAMP_COL] = pd_module.to_datetime(
            frame[DEFAULT_TIMESTAMP_COL], errors="coerce", utc=True
        )
        if frame[DEFAULT_TIMESTAMP_COL].isna().any():
            raise ExperimentError(f"{path} contains an invalid dateTime")
        frame[DEFAULT_VALUE_COL] = pd_module.to_numeric(
            frame[DEFAULT_VALUE_COL], errors="coerce"
        )
        if "aggregation_valid" in frame.columns:
            valid = _to_bool(frame["aggregation_valid"])
            frame.loc[~valid, DEFAULT_VALUE_COL] = float("nan")
        frame = frame.sort_values(DEFAULT_TIMESTAMP_COL).reset_index(drop=True)
        if frame[DEFAULT_TIMESTAMP_COL].duplicated().any():
            count = int(frame[DEFAULT_TIMESTAMP_COL].duplicated(keep=False).sum())
            raise ExperimentError(f"{path} contains {count} duplicate timestamps")
        deltas = frame[DEFAULT_TIMESTAMP_COL].diff().dropna()
        one_hour = pd_module.to_timedelta(1, unit="h")
        if len(deltas) and not bool((deltas == one_hour).all()):
            bad = int((deltas != one_hour).sum())
            raise ExperimentError(
                f"{path} is not a contiguous hourly axis ({bad} non-1-hour gaps)"
            )
        summaries.append(
            _station_input_summary(
                frame,
                station_guid=guid,
                path=path,
                data_start=data_start or frame[DEFAULT_TIMESTAMP_COL].min().date(),
                test_end=test_end
                or (
                    frame[DEFAULT_TIMESTAMP_COL].max()
                    + pd_module.to_timedelta(1, unit="h")
                ).date(),
            )
        )
        frames.append(frame)
    if not frames:
        raise ExperimentError("no hourly station files selected")
    panel = pd_module.concat(frames, ignore_index=True)
    panel = panel.sort_values(["station_guid", DEFAULT_TIMESTAMP_COL]).reset_index(drop=True)
    return panel, summaries


def _test_bounds(split: Mapping[str, Any]) -> tuple[Any, Any]:
    pd_module = _require_pandas()
    first = pd_module.Timestamp(split["validation_end_exclusive"], tz="UTC")
    end = pd_module.Timestamp(split["test_end_exclusive"], tz="UTC")
    return first, end


def _capped_last_target(
    first_target: Any,
    test_end: Any,
    *,
    horizon: int,
    step: int,
    max_windows: int | None,
) -> Any:
    pd_module = _require_pandas()
    natural_last = test_end - pd_module.to_timedelta(1, unit="h")
    if max_windows is None:
        return natural_last
    cap_last = first_target + pd_module.to_timedelta(
        (max_windows - 1) * step + horizon - 1, unit="h"
    )
    return min(natural_last, cap_last)


def _forecast_records(
    frame: Any,
    predictor: Callable[[Any, int, str], Any],
    *,
    split: Mapping[str, Any],
    max_windows: int | None,
    first_target: Any | None = None,
    last_target: Any | None = None,
) -> Any:
    default_first, test_end = _test_bounds(split)
    first_target = default_first if first_target is None else first_target
    horizon = int(split["horizon_hours"])
    step = int(split["rolling_origin_step_hours"])
    if last_target is None:
        last_target = _capped_last_target(
            first_target,
            test_end,
            horizon=horizon,
            step=step,
            max_windows=max_windows,
        )
    return evaluate.rolling_evaluate(
        frame,
        predictor,
        timestamp_col=DEFAULT_TIMESTAMP_COL,
        value_col=DEFAULT_VALUE_COL,
        group_col=DEFAULT_GROUP_COL,
        encoder_length=int(split["encoder_hours"]),
        horizon=horizon,
        step=step,
        first_target=first_target,
        last_target=last_target,
        skip_missing_targets=True,
    )


def _metric_summary(forecasts: Any) -> dict[str, Any]:
    if forecasts is None or len(forecasts) == 0:
        return {
            "status": "no_forecasts",
            "n_forecasts": 0,
            "n_origins": 0,
            "metrics": {
                "p10_pinball": None,
                "p50_pinball": None,
                "p90_pinball": None,
                "mae_from_p50": None,
                "n_values": 0,
            },
            "by_station": [],
        }
    import numpy as np

    metrics = score_quantiles(
        forecasts["actual"].to_numpy(dtype=float),
        forecasts[["p10", "p50", "p90"]].to_numpy(dtype=float),
    )
    by_station = evaluate.summarize_forecasts(
        forecasts, group_cols=(DEFAULT_GROUP_COL,)
    )
    station_records = [
        _json_safe(record) for record in by_station.to_dict(orient="records")
    ]
    return {
        "status": "ok",
        "n_forecasts": int(len(forecasts)),
        "n_origins": int(forecasts[["station_guid", "origin"]].drop_duplicates().shape[0]),
        "metrics": _json_safe(metrics.as_dict()),
        "by_station": station_records,
        "interval_coverage_mean": float(
            np.mean(
                (forecasts["actual"].to_numpy(dtype=float)
                 >= forecasts["p10"].to_numpy(dtype=float))
                & (forecasts["actual"].to_numpy(dtype=float)
                   <= forecasts["p90"].to_numpy(dtype=float))
            )
        ),
    }

def _run_baseline(
    frame: Any,
    *,
    name: str,
    predictor: Callable[[Any, int, str], Any],
    split: Mapping[str, Any],
    max_windows: int | None,
) -> tuple[dict[str, Any], Any | None]:
    try:
        forecasts = _forecast_records(
            frame, predictor, split=split, max_windows=max_windows
        )
    except OptionalBaselineUnavailable as exc:
        return {
            "status": "unavailable",
            "reason": str(exc),
            "n_forecasts": 0,
            "by_station": [],
        }, None
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "n_forecasts": 0,
            "by_station": [],
        }, None
    summary = _metric_summary(forecasts)
    summary["max_windows_per_station"] = max_windows
    return summary, forecasts


def _attempt_tft(
    frame: Any,
    *,
    split: Mapping[str, Any],
    max_windows: int | None,
    requested: bool,
    station_guids: Sequence[str] | None = None,
    max_epochs: int = 5,
    batch_size: int = 64,
    accelerator: str = "cpu",
) -> dict[str, Any]:
    if not requested:
        return {
            "status": "not_requested",
            "reason": "TFT execution requires --run-tft and an implemented src.tft runner.",
        }
    try:
        importlib.import_module("torch")
        importlib.import_module("pytorch_forecasting")
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"TFT dependency unavailable: {type(exc).__name__}: {exc}",
        }
    try:
        module = importlib.import_module(f"{__package__}.tft" if __package__ else "tft")
    except Exception as exc:
        return {"status": "unavailable", "reason": f"cannot import src.tft: {exc}"}
    runner = getattr(module, "run_river_tft", None) or getattr(module, "run_tft", None)
    if runner is None:
        return {
            "status": "unavailable",
            "reason": "src.tft exposes no run_river_tft/run_tft function",
        }
    try:
        value = runner(
            frame=frame,
            split=split,
            max_windows=max_windows,
            station_guids=station_guids,
            max_epochs=max_epochs,
            batch_size=batch_size,
            accelerator=accelerator,
        )
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {"status": "ok", "result": _json_safe(value)}


def _markdown_summary(
    *,
    split: Mapping[str, Any],
    baseline_results: Mapping[str, Mapping[str, Any]],
    tft_result: Mapping[str, Any],
    station_count: int,
) -> str:
    lines = [
        "### River baseline evaluation",
        (
            f"Test targets: {split['validation_end_exclusive'].isoformat()} to "
            f"{split['test_end_exclusive'].isoformat()} (UTC); "
            f"{station_count} station(s); max origins per station are bounded."
        ),
        "",
        "| Model | Status | Forecast values | P50 pinball | P90 pinball | MAE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, result in baseline_results.items():
        metrics = result.get("metrics", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(result.get("status", "")),
                    str(result.get("n_forecasts", 0)),
                    _format_number(metrics.get("p50_pinball")),
                    _format_number(metrics.get("p90_pinball")),
                    _format_number(metrics.get("mae_from_p50")),
                ]
            )
            + " |"
        )
    tft_payload = tft_result.get("result", tft_result)
    tft_metrics = tft_payload.get("pooled", {}).get("metrics", {}) if isinstance(tft_payload, Mapping) else {}
    tft_status = tft_payload.get("status", tft_result.get("status", "")) if isinstance(tft_payload, Mapping) else tft_result.get("status", "")
    lines.append(
        "| TFT | "
        + str(tft_status)
        + " | "
        + str(tft_payload.get("pooled", {}).get("test_windows", "n/a") if isinstance(tft_payload, Mapping) else "n/a")
        + " | "
        + _format_number(tft_metrics.get("p50_pinball"))
        + " | "
        + _format_number(tft_metrics.get("p90_pinball"))
        + " | "
        + _format_number(tft_metrics.get("mae_from_p50"))
        + " |"
    )
    lines.append("")
    lines.append(
        "Scores are measured only on finite target windows; unavailable/error "
        "models have no fabricated metrics."
    )
    return "\n".join(lines)


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.6g}" if math.isfinite(number) else "n/a"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".part",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(_json_safe(value), stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def run_experiment(
    *,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    splits_path: Path | str = DEFAULT_SPLITS,
    station_config: Path | str = Path("configs/stations.yml"),
    station_guids: Sequence[str] | None = None,
    max_windows: int | None = DEFAULT_MAX_WINDOWS,
    run_tft: bool = False,
    tft_max_epochs: int = 5,
    tft_batch_size: int = 64,
    tft_accelerator: str = "cpu",
    results_json: Path | str | None = None,
) -> dict[str, Any]:
    """Run capped baseline evaluations and optionally attempt TFT."""

    if max_windows is not None and max_windows <= 0:
        raise ExperimentError("max_windows must be positive or None")
    split = load_split_config(splits_path)
    frame, station_summaries = load_hourly_data(
        data_dir,
        station_guids=station_guids,
        station_config=station_config,
        data_start=split["data_start"],
        test_end=split["test_end_exclusive"],
    )
    # Run the optional PyTorch stage before LightGBM's native library is
    # loaded. On the local macOS stack, importing/fitting both in one process
    # can otherwise segfault inside PyTorch Forecasting's tensor conversion.
    tft = _attempt_tft(
        frame,
        split=split,
        max_windows=max_windows,
        requested=run_tft,
        station_guids=station_guids,
        max_epochs=tft_max_epochs,
        batch_size=tft_batch_size,
        accelerator=tft_accelerator,
    )
    seasonal, _ = _run_baseline(
        frame,
        name="seasonal_naive",
        predictor=lambda history, horizon, _station: seasonal_naive_forecast(
            history, horizon, DEFAULT_SEASONALITY
        ),
        split=split,
        max_windows=max_windows,
    )
    ets, _ = _run_baseline(
        frame,
        name="ets",
        predictor=lambda history, horizon, _station: ets_forecast(
            history, horizon, DEFAULT_SEASONALITY
        ),
        split=split,
        max_windows=max_windows,
    )
    lightgbm, _ = _run_baseline(
        frame,
        name="lightgbm",
        predictor=lambda history, horizon, _station: lightgbm_recursive_forecast(
            history,
            horizon,
            seasonality=DEFAULT_SEASONALITY,
            lags=48,
        ),
        split=split,
        max_windows=max_windows,
    )
    baseline_results = {
        "seasonal_naive": seasonal,
        "ets": ets,
        "lightgbm": lightgbm,
    }
    output: dict[str, Any] = {
        "schema": "river-experiment-v1",
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "data_dir": str(Path(data_dir)),
        "splits_path": str(Path(splits_path)),
        "station_config": str(Path(station_config)),
        "split": {
            key: value.isoformat() if isinstance(value, date) else value
            for key, value in split.items()
        },
        "stations": station_summaries,
        "baselines": baseline_results,
        "tft": tft,
        "markdown_summary": _markdown_summary(
            split=split,
            baseline_results=baseline_results,
            tft_result=tft,
            station_count=len(station_summaries),
        ),
        "notes": [
            "Evaluation uses the shared rolling_evaluate/summarize_forecasts functions.",
            "No imputation is applied before target-window selection.",
            "This is retrospective research evaluation, not operational warning logic.",
        ],
    }
    output = _json_safe(output)
    if results_json is not None:
        _atomic_json(Path(results_json), output)
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits", dest="splits_path", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--station-config", type=Path, default=Path("configs/stations.yml"))
    parser.add_argument(
        "--station",
        action="append",
        dest="station_guids",
        metavar="GUID",
        help="station GUID; repeat to select an explicit subset",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=DEFAULT_MAX_WINDOWS,
        help="maximum rolling test origins per station (default: 16)",
    )
    parser.add_argument(
        "--run-tft",
        action="store_true",
        help="attempt an implemented TFT runner only when dependencies exist",
    )
    parser.add_argument("--tft-max-epochs", type=int, default=5)
    parser.add_argument("--tft-batch-size", type=int, default=64)
    parser.add_argument(
        "--tft-accelerator",
        choices=("cpu", "mps", "auto"),
        default="cpu",
        help="TFT device; CPU is the reproducible default",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        help="write measured results to this path; otherwise print only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_experiment(
            data_dir=args.data_dir,
            splits_path=args.splits_path,
            station_config=args.station_config,
            station_guids=args.station_guids,
            max_windows=args.max_windows,
            run_tft=args.run_tft,
            tft_max_epochs=args.tft_max_epochs,
            tft_batch_size=args.tft_batch_size,
            tft_accelerator=args.tft_accelerator,
            results_json=args.results_json,
        )
    except (ExperimentError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(result["markdown_summary"])
    if args.results_json:
        print(f"\nresults_json={args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
