"""Run a retrospective residual/interval review for one river baseline.

The command deliberately separates validation calibration from test scoring.
It writes no alert stream and makes no operational or safety-critical claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    import pandas as pd
except ImportError:  # pragma: no cover - environment dependent
    pd = None  # type: ignore[assignment]

try:
    from .anomalies import fit_residual_references, score_residual_anomalies, summarize_anomalies
    from .baselines import ets_forecast, lightgbm_recursive_forecast, seasonal_naive_forecast
    from .river_experiment import (
        DEFAULT_SEASONALITY,
        _forecast_records,
        _json_safe,
        _require_pandas,
        load_hourly_data,
        load_split_config,
    )
except ImportError:  # pragma: no cover - supports python src/river_review.py
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.anomalies import (  # type: ignore[no-redef]
        fit_residual_references,
        score_residual_anomalies,
        summarize_anomalies,
    )
    from src.baselines import (  # type: ignore[no-redef]
        ets_forecast,
        lightgbm_recursive_forecast,
        seasonal_naive_forecast,
    )
    from src.river_experiment import (  # type: ignore[no-redef]
        DEFAULT_SEASONALITY,
        _forecast_records,
        _json_safe,
        _require_pandas,
        load_hourly_data,
        load_split_config,
    )


def _predictor(name: str) -> Callable[[Any, int, str], Any]:
    if name == "seasonal_naive":
        return lambda history, horizon, _station: seasonal_naive_forecast(
            history, horizon, DEFAULT_SEASONALITY
        )
    if name == "ets":
        return lambda history, horizon, _station: ets_forecast(
            history, horizon, DEFAULT_SEASONALITY
        )
    if name == "lightgbm":
        return lambda history, horizon, _station: lightgbm_recursive_forecast(
            history, horizon, seasonality=DEFAULT_SEASONALITY, lags=48
        )
    raise ValueError(f"unsupported review model: {name}")


def run_review(
    *,
    model: str = "seasonal_naive",
    data_dir: Path | str = Path("data/hourly"),
    splits_path: Path | str = Path("configs/splits.yml"),
    station_config: Path | str = Path("configs/stations.yml"),
    station_guids: Sequence[str] | None = None,
    max_windows: int | None = 16,
    threshold: float = 3.5,
    scored_csv: Path | str | None = None,
    results_json: Path | str | None = None,
) -> dict[str, Any]:
    """Calibrate on validation origins and score bounded held-out test origins."""

    pd_module = _require_pandas()
    split = load_split_config(splits_path)
    frame, station_summaries = load_hourly_data(
        data_dir,
        station_guids=station_guids,
        station_config=station_config,
        data_start=split["data_start"],
        test_end=split["test_end_exclusive"],
    )
    predictor = _predictor(model)
    train_end = pd_module.Timestamp(split["train_end_exclusive"], tz="UTC")
    validation_end = pd_module.Timestamp(split["validation_end_exclusive"], tz="UTC")
    validation_forecasts = _forecast_records(
        frame,
        predictor,
        split=split,
        max_windows=None,
        first_target=train_end,
        last_target=validation_end - pd_module.Timedelta(hours=1),
    )
    test_forecasts = _forecast_records(
        frame,
        predictor,
        split=split,
        max_windows=max_windows,
    )
    references = fit_residual_references(validation_forecasts)
    scored = score_residual_anomalies(test_forecasts, references, threshold=threshold)
    summary = summarize_anomalies(scored, group_cols=("station_guid",))
    by_horizon = summarize_anomalies(scored, group_cols=("station_guid", "horizon_step"))

    if scored_csv is not None:
        scored_path = Path(scored_csv)
        scored_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(scored_path, index=False)
    output: dict[str, Any] = {
        "schema": "river-anomaly-review-v1",
        "model": model,
        "threshold": threshold,
        "calibration": {
            "start": split["train_end_exclusive"].isoformat(),
            "end_exclusive": split["validation_end_exclusive"].isoformat(),
            "n_forecasts": int(len(validation_forecasts)),
            "n_references": int(len(references)),
        },
        "test": {
            "start": split["validation_end_exclusive"].isoformat(),
            "end_exclusive": split["test_end_exclusive"].isoformat(),
            "n_forecasts": int(len(test_forecasts)),
            "max_windows_per_station": max_windows,
        },
        "stations": station_summaries,
        "summary_by_station": summary.to_dict(orient="records"),
        "summary_by_station_horizon": by_horizon.to_dict(orient="records"),
        "notes": [
            "Residual references use validation-period forecasts only.",
            "Point baselines have degenerate P10/P50/P90 intervals; interval flags are therefore diagnostic, not calibrated probability statements.",
            "Events are retrospective model-miss review records, not alerts or flood warnings.",
        ],
    }
    output = _json_safe(output)
    if results_json is not None:
        path = Path(results_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def run_saved_tft_review(
    result_path: Path | str,
    *,
    threshold: float = 3.5,
    scored_csv: Path | str | None = None,
    results_json: Path | str | None = None,
) -> dict[str, Any]:
    """Review validation/test forecast records emitted by ``run_river_tft``."""

    pd_module = _require_pandas()
    payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    tft = payload.get("tft", payload)
    if isinstance(tft, Mapping) and isinstance(tft.get("result"), Mapping):
        tft = tft["result"]
    if not isinstance(tft, Mapping):
        raise ValueError("TFT result JSON has no result mapping")
    calibration = pd_module.DataFrame.from_records(tft.get("calibration_forecasts", []))
    test = pd_module.DataFrame.from_records(tft.get("forecasts", []))
    if calibration.empty or test.empty:
        raise ValueError("TFT result JSON lacks calibration or test forecast records")
    references = fit_residual_references(calibration)
    scored = score_residual_anomalies(test, references, threshold=threshold)
    summary = summarize_anomalies(scored, group_cols=("station_guid",))
    by_horizon = summarize_anomalies(scored, group_cols=("station_guid", "horizon_step"))
    if scored_csv is not None:
        path = Path(scored_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(path, index=False)
    output: dict[str, Any] = {
        "schema": "river-anomaly-review-v1",
        "model": "tft",
        "source_results": str(result_path),
        "threshold": threshold,
        "calibration": {"n_forecasts": int(len(calibration)), "n_references": int(len(references))},
        "test": {"n_forecasts": int(len(test))},
        "summary_by_station": summary.to_dict(orient="records"),
        "summary_by_station_horizon": by_horizon.to_dict(orient="records"),
        "notes": [
            "Residual references use the validation forecasts emitted by the same fitted TFT run.",
            "This is a smoke-run diagnostic; the one-epoch/small-window model is not a calibrated operational predictor.",
            "Events are retrospective review records, not alerts or flood warnings.",
        ],
    }
    output = _json_safe(output)
    if results_json is not None:
        path = Path(results_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("seasonal_naive", "ets", "lightgbm"), default="seasonal_naive")
    parser.add_argument("--tft-results", type=Path, help="review calibration/test forecasts from a saved river TFT run")
    parser.add_argument("--data-dir", type=Path, default=Path("data/hourly"))
    parser.add_argument("--splits", dest="splits_path", type=Path, default=Path("configs/splits.yml"))
    parser.add_argument("--station-config", type=Path, default=Path("configs/stations.yml"))
    parser.add_argument("--station", action="append", dest="station_guids", metavar="GUID")
    parser.add_argument("--max-windows", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=3.5)
    parser.add_argument("--scored-csv", type=Path, default=Path("results/anomaly_scored.csv"))
    parser.add_argument("--results-json", type=Path, default=Path("results/anomaly_review.json"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.tft_results:
            result = run_saved_tft_review(
                args.tft_results,
                threshold=args.threshold,
                scored_csv=args.scored_csv,
                results_json=args.results_json,
            )
        else:
            result = run_review(
                model=args.model,
                data_dir=args.data_dir,
                splits_path=args.splits_path,
                station_config=args.station_config,
                station_guids=args.station_guids,
                max_windows=args.max_windows,
                threshold=args.threshold,
                scored_csv=args.scored_csv,
                results_json=args.results_json,
            )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
