"""Phase 1 Electricity benchmark harness for the TFT project.

The module has two intentionally separate paths:

* ``--smoke`` runs with NumPy only on deterministic synthetic hourly series.
  It checks the metric/window plumbing and does not claim to reproduce the
  paper.
* the normal path downloads/reads the UCI ElectricityLoadDiagrams20112014
  data, prepares the paper-style hourly panel, evaluates optional baselines,
  and trains a PyTorch Forecasting TFT when the environment is installed.

Full benchmark numbers are printed as JSON and are never written into the
repository automatically.  This keeps ``BENCHMARKS.md`` honest: a result is
only reported there after the command, environment, subset, and data checksum
have been recorded by the researcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .baselines import (
    DEFAULT_QUANTILES,
    OptionalBaselineUnavailable,
    evaluate_ets,
    evaluate_lightgbm,
    evaluate_seasonal_naive,
    score_quantiles,
)


UCI_ELECTRICITY_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/"
    "LD2011_2014.txt.zip"
)
PAPER_REFERENCE = {"p50_pinball": 0.055, "p90_pinball": 0.027}
DEFAULT_ENCODER_LENGTH = 168
DEFAULT_HORIZON = 24
DEFAULT_PAPER_START_DAY = 1096
DEFAULT_PAPER_END_DAY = 1346


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Full benchmark mode requires pandas. Install the project "
            "environment (environment.yml) and retry; --smoke needs NumPy only."
        ) from exc
    return pd


def _require_requests() -> Any:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Full benchmark mode requires requests for the UCI download. "
            "Install the project environment and retry."
        ) from exc
    return requests


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 checksum without loading a dataset into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_electricity_archive(
    data_dir: str | os.PathLike[str],
    force: bool = False,
    url: str = UCI_ELECTRICITY_URL,
) -> tuple[Path, str]:
    """Download and safely extract the UCI quarter-hour source file.

    Returns ``(txt_path, archive_sha256)``.  The raw archive is retained so a
    future report can cite the exact input checksum.  Existing files are
    reused unless ``force`` is supplied.
    """

    root = Path(data_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "LD2011_2014.txt.zip"
    text_path = root / "LD2011_2014.txt"

    if force or not archive_path.exists():
        requests = _require_requests()
        response = requests.get(url, stream=True, timeout=(30, 180), allow_redirects=True)
        response.raise_for_status()
        temporary = archive_path.with_suffix(archive_path.suffix + ".part")
        try:
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
            temporary.replace(archive_path)
        finally:
            temporary.unlink(missing_ok=True)

    archive_sha256 = sha256_file(archive_path)
    if force or not text_path.exists():
        _extract_source_member(archive_path, text_path)
    return text_path, archive_sha256


def _extract_source_member(archive_path: Path, output_path: Path) -> None:
    """Extract only the expected UCI member and reject path traversal."""

    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if Path(name).name == "LD2011_2014.txt"]
        if len(members) != 1:
            raise RuntimeError(
                "UCI archive did not contain exactly one LD2011_2014.txt member; "
                f"found {members!r}"
            )
        member = Path(members[0])
        if member.is_absolute() or ".." in member.parts:
            raise RuntimeError(f"Refusing unsafe archive member: {members[0]!r}")
        temporary = output_path.with_suffix(output_path.suffix + ".part")
        try:
            with archive.open(members[0]) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary.replace(output_path)
        finally:
            temporary.unlink(missing_ok=True)


def load_electricity_hourly(source_path: str | os.PathLike[str]) -> Any:
    """Read UCI quarter-hour data and aggregate it to hourly means."""

    pd = _require_pandas()
    frame = pd.read_csv(source_path, index_col=0, sep=";", decimal=",")
    frame.index = pd.to_datetime(frame.index, errors="raise")
    frame.sort_index(inplace=True)
    # The Google reference script uses hourly means and treats zero as a
    # missing reading before defining each meter's active period.
    hourly = frame.resample("1h").mean(numeric_only=True).replace(0.0, np.nan)
    hourly.index.name = "date"
    return hourly


def format_electricity_panel(
    hourly: Any,
    *,
    start_day: int | None = DEFAULT_PAPER_START_DAY,
    end_day: int | None = DEFAULT_PAPER_END_DAY,
    max_series: int | None = None,
) -> Any:
    """Return the paper-style long hourly panel.

    ``time_idx`` is an integer number of hours from the global first timestamp.
    Series are restricted to their observed active range and interior missing
    hours are filled with zero, matching the published Google preprocessing.
    Pass ``start_day=None``/``end_day=None`` to retain the complete source.
    """

    pd = _require_pandas()
    if hourly.empty:
        raise ValueError("hourly Electricity frame is empty")
    earliest_time = hourly.index.min()
    columns = list(hourly.columns)
    if max_series is not None:
        if max_series <= 0:
            raise ValueError("max_series must be positive")
        columns = sorted(columns)[:max_series]

    frames = []
    for label in columns:
        series = hourly[label]
        valid = series.dropna()
        if valid.empty:
            continue
        start = valid.index.min()
        end = valid.index.max()
        active = series.loc[start:end].fillna(0.0)
        dates = active.index
        time_idx = ((dates - earliest_time) / pd.Timedelta(hours=1)).astype("int64")
        frame = pd.DataFrame(
            {
                "series": str(label),
                "date": dates,
                "time_idx": time_idx,
                "power_usage": active.to_numpy(dtype=float),
                "hour": dates.hour.astype("int16"),
                "day_of_week": dates.dayofweek.astype("int16"),
                "days_from_start": (dates - earliest_time).days.astype("int32"),
            }
        )
        frames.append(frame)
    if not frames:
        raise ValueError("no non-empty Electricity series remained after formatting")
    output = pd.concat(frames, ignore_index=True)
    if start_day is not None:
        output = output.loc[output["days_from_start"] >= int(start_day)]
    if end_day is not None:
        output = output.loc[output["days_from_start"] < int(end_day)]
    output.sort_values(["series", "time_idx"], inplace=True)
    output.reset_index(drop=True, inplace=True)
    if output.empty:
        raise ValueError("paper window removed every Electricity row")
    return output


def split_paper_panel(
    panel: Any,
    valid_boundary: int = 1315,
    test_boundary: int = 1339,
) -> tuple[Any, Any, Any]:
    """Split the panel using the boundaries from the Google TFT formatter."""

    train = panel.loc[panel["days_from_start"] < valid_boundary].copy()
    valid = panel.loc[
        (panel["days_from_start"] >= valid_boundary - 7)
        & (panel["days_from_start"] < test_boundary)
    ].copy()
    test = panel.loc[panel["days_from_start"] >= test_boundary - 7].copy()
    if train.empty or valid.empty or test.empty:
        raise ValueError(
            "paper split produced an empty partition; inspect the selected "
            "window and boundaries"
        )
    return train, valid, test


def _import_tft_stack() -> tuple[Any, Any, Any, Any, Any]:
    """Import PyTorch/Lightning/PyTorch Forecasting only in full mode."""

    try:
        import torch
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
        from pytorch_forecasting.metrics import QuantileLoss
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Full TFT mode requires torch, lightning, and pytorch-forecasting. "
            "Install the pinned project environment and retry; --smoke remains "
            "available without those packages."
        ) from exc

    try:
        import lightning.pytorch as pl
    except ImportError:
        try:
            import pytorch_lightning as pl  # type: ignore[no-redef]
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "PyTorch Forecasting needs Lightning (lightning or "
                "pytorch-lightning). Install the project environment and retry."
            ) from exc
    return torch, pl, TemporalFusionTransformer, TimeSeriesDataSet, (GroupNormalizer, QuantileLoss)


def resolve_accelerator(requested: str, torch: Any) -> str:
    """Resolve ``auto`` to MPS on Apple Silicon when it is available."""

    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("--accelerator mps was requested but MPS is unavailable")
        return requested
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _make_tft_datasets(
    panel: Any,
    TimeSeriesDataSet: Any,
    GroupNormalizer: Any,
    *,
    encoder_length: int,
    horizon: int,
) -> tuple[Any, Any, Any]:
    train, valid, test = split_paper_panel(panel)
    training = TimeSeriesDataSet(
        train,
        time_idx="time_idx",
        target="power_usage",
        group_ids=["series"],
        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,
        min_prediction_length=horizon,
        max_prediction_length=horizon,
        static_categoricals=["series"],
        time_varying_known_reals=["time_idx", "hour", "day_of_week"],
        time_varying_unknown_reals=["power_usage"],
        target_normalizer=GroupNormalizer(groups=["series"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training, valid, predict=True, stop_randomization=True
    )
    test_dataset = TimeSeriesDataSet.from_dataset(
        training, test, predict=True, stop_randomization=True
    )
    return training, validation, test_dataset


def _prediction_arrays(prediction: Any) -> tuple[np.ndarray, np.ndarray]:
    """Normalize PyTorch Forecasting's prediction wrapper to NumPy arrays."""

    output = prediction.output if hasattr(prediction, "output") else prediction
    target = prediction.y if hasattr(prediction, "y") else None
    if target is None:
        raise RuntimeError("TFT prediction did not return targets; use return_y=True")
    if isinstance(target, (tuple, list)):
        target = target[0]
    if hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    if hasattr(target, "detach"):
        target = target.detach().cpu().numpy()
    output = np.asarray(output, dtype=float)
    target = np.asarray(target, dtype=float)
    if output.ndim == 2:
        output = output[..., None]
    if target.ndim == 3 and target.shape[-1] == 1:
        target = target[..., 0]
    if output.shape[:-1] != target.shape:
        raise RuntimeError(
            f"unexpected TFT prediction shapes: output={output.shape}, target={target.shape}"
        )
    return target, output


def run_tft(
    panel: Any,
    *,
    encoder_length: int = DEFAULT_ENCODER_LENGTH,
    horizon: int = DEFAULT_HORIZON,
    max_epochs: int = 5,
    batch_size: int = 64,
    hidden_size: int = 16,
    attention_head_size: int = 2,
    hidden_continuous_size: int = 8,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    accelerator: str = "auto",
    limit_train_batches: float | int = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Train/evaluate one PyTorch Forecasting TFT and return measured metrics."""

    torch, pl, TemporalFusionTransformer, TimeSeriesDataSet, extras = _import_tft_stack()
    GroupNormalizer, QuantileLoss = extras
    pl.seed_everything(seed, workers=True)
    device = resolve_accelerator(accelerator, torch)
    training, validation, test = _make_tft_datasets(
        panel,
        TimeSeriesDataSet,
        GroupNormalizer,
        encoder_length=encoder_length,
        horizon=horizon,
    )
    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    valid_loader = validation.to_dataloader(
        train=False, batch_size=max(batch_size, 128), num_workers=0
    )
    test_loader = test.to_dataloader(train=False, batch_size=max(batch_size, 128), num_workers=0)
    loss = QuantileLoss(quantiles=list(DEFAULT_QUANTILES))
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=loss,
        optimizer="adam",
        reduce_on_plateau_patience=4,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=device,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        gradient_clip_val=0.1,
        limit_train_batches=limit_train_batches,
        deterministic=True,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    prediction = model.predict(
        test_loader,
        mode="quantiles",
        return_y=True,
        trainer_kwargs={"accelerator": device, "devices": 1, "logger": False},
    )
    y_true, quantile_prediction = _prediction_arrays(prediction)
    metrics = score_quantiles(y_true, quantile_prediction, DEFAULT_QUANTILES)
    return {
        "status": "measured",
        "device": device,
        "series": int(panel["series"].nunique()),
        # PyTorch Forecasting stores the integer time axis under ``time`` in
        # the current dataset object; ``len(training)`` is the number of
        # training windows and is stable across supported versions.
        "train_rows": int(len(training)),
        "test_forecast_values": metrics.n_values,
        "metrics": metrics.as_dict(),
        "paper_reference": PAPER_REFERENCE,
        "config": {
            "encoder_length": encoder_length,
            "horizon": horizon,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "attention_head_size": attention_head_size,
            "hidden_continuous_size": hidden_continuous_size,
            "learning_rate": learning_rate,
            "seed": seed,
        },
    }


def _series_arrays(panel: Any) -> Iterable[tuple[str, np.ndarray]]:
    for name, group in panel.groupby("series", sort=True):
        yield str(name), group.sort_values("time_idx")["power_usage"].to_numpy(dtype=float)


def run_baselines(
    panel: Any,
    *,
    encoder_length: int = DEFAULT_ENCODER_LENGTH,
    horizon: int = DEFAULT_HORIZON,
    max_series: int | None = None,
    max_windows: int = 64,
    include_optional: bool = True,
) -> dict[str, Any]:
    """Evaluate baseline ladder with explicit availability/status fields."""

    results: dict[str, Any] = {}
    grouped = [
        (str(name), group.sort_values("time_idx"))
        for name, group in panel.groupby("series", sort=True)
    ]
    series_items = grouped
    if max_series is not None:
        series_items = series_items[:max_series]
    baseline_functions = {"seasonal_naive": evaluate_seasonal_naive}
    if include_optional:
        baseline_functions.update({"ets": evaluate_ets, "lightgbm": evaluate_lightgbm})
    for name, function in baseline_functions.items():
        per_series: dict[str, Any] = {}
        failures: dict[str, str] = {}
        for series, group in series_items:
            values = group["power_usage"].to_numpy(dtype=float)
            # The public Google formatter keeps seven days of context in the
            # held-out partition.  Start baseline origins at that boundary so
            # training/validation rows are not scored as test predictions.
            test_start_day = 1339 - 7
            days = group["days_from_start"].to_numpy(dtype=int)
            candidate = np.flatnonzero(days >= test_start_day)
            start = int(candidate[0]) if candidate.size else None
            try:
                metrics = function(
                    values,
                    encoder_length=encoder_length,
                    horizon=horizon,
                    max_windows=max_windows,
                    start=start,
                )
                per_series[series] = metrics.as_dict()
            except OptionalBaselineUnavailable as exc:
                failures[series] = str(exc)
                break
        if failures:
            results[name] = {"status": "unavailable", "reason": next(iter(failures.values()))}
        else:
            results[name] = {"status": "measured", "per_series": per_series}
    return results


def run_smoke(seed: int = 7) -> dict[str, Any]:
    """Run a deterministic metric/window smoke test without pandas or PyTorch."""

    rng = np.random.default_rng(seed)
    n_series = 3
    n_steps = 168 + (24 * 8)
    values = []
    for series in range(n_series):
        t = np.arange(n_steps)
        values.append(10.0 + series + np.sin(2.0 * np.pi * t / 24.0) + rng.normal(0, 0.05, n_steps))
    metrics = [
        evaluate_seasonal_naive(
            series_values,
            encoder_length=168,
            horizon=24,
            max_windows=8,
        ).as_dict()
        for series_values in values
    ]
    result = {
        "status": "smoke_only",
        "message": "Synthetic NumPy check passed; these are not paper benchmark results.",
        "seed": seed,
        "shape": {"series": n_series, "steps": n_steps, "encoder": 168, "horizon": 24},
        "seasonal_naive_metrics": metrics,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run only the dependency-free synthetic check")
    parser.add_argument("--data-dir", default="data/benchmark/electricity", help="cache directory for the UCI source")
    parser.add_argument("--force-download", action="store_true", help="redownload and re-extract the source")
    parser.add_argument("--max-series", type=int, default=8, help="deterministic series cap (default: 8; use 0 for all)")
    parser.add_argument("--max-epochs", type=int, default=5, help="TFT epochs; keep small on CPU/MPS")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--attention-head-size", type=int, default=2)
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "mps"),
        default="cpu",
        help="training device (default: cpu; use mps explicitly on Apple Silicon)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-tft", action="store_true", help="run baselines/data preparation only")
    parser.add_argument("--skip-optional-baselines", action="store_true", help="only evaluate seasonal naive")
    parser.add_argument("--max-baseline-windows", type=int, default=16)
    parser.add_argument("--results-json", help="optional explicit path for measured JSON output")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.smoke:
        run_smoke(args.seed)
        return 0
    try:
        text_path, archive_sha256 = download_electricity_archive(
            args.data_dir, force=args.force_download
        )
        hourly = load_electricity_hourly(text_path)
        max_series = None if args.max_series == 0 else args.max_series
        panel = format_electricity_panel(hourly, max_series=max_series)
        result: dict[str, Any] = {
            "status": "measured_data_prepared",
            "source": {"url": UCI_ELECTRICITY_URL, "archive_sha256": archive_sha256},
            "panel": {
                "series": int(panel["series"].nunique()),
                "rows": int(len(panel)),
                "start": str(panel["date"].min()),
                "end": str(panel["date"].max()),
                "paper_window_days": [DEFAULT_PAPER_START_DAY, DEFAULT_PAPER_END_DAY],
            },
            "baselines": run_baselines(
                panel,
                max_series=max_series,
                max_windows=args.max_baseline_windows,
                include_optional=not args.skip_optional_baselines,
            ),
        }
        if not args.skip_tft:
            result["tft"] = run_tft(
                panel,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                hidden_size=args.hidden_size,
                attention_head_size=args.attention_head_size,
                accelerator=args.accelerator,
                seed=args.seed,
            )
        else:
            result["tft"] = {"status": "skipped", "reason": "--skip-tft"}
        rendered = json.dumps(result, indent=2, sort_keys=True)
        print(rendered)
        if args.results_json:
            output_path = Path(args.results_json).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered + "\n", encoding="utf-8")
            print(f"Wrote measured results to {output_path}", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"BENCHMARK NOT RUN: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
