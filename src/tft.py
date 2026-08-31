"""Lazy PyTorch Forecasting adapter for the per-station river TFT."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


def _require_stack() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
        from pytorch_forecasting.metrics import QuantileLoss
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "River TFT training requires torch, lightning, and pytorch-forecasting; "
            "install environment.yml first."
        ) from exc
    try:
        import lightning.pytorch as pl
    except ImportError:
        try:
            import pytorch_lightning as pl  # type: ignore[no-redef]
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Install lightning or pytorch-lightning for TFT training.") from exc
    return torch, pl, TemporalFusionTransformer, TimeSeriesDataSet, (GroupNormalizer, QuantileLoss)


def resolve_accelerator(requested: str = "auto") -> tuple[str, Any]:
    """Resolve CPU/MPS without importing PyTorch at module import time."""

    torch, _, _, _, _ = _require_stack()
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return "mps", torch
    if requested == "cpu":
        return "cpu", torch
    if requested != "auto":
        raise ValueError("requested accelerator must be auto, cpu, or mps")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps", torch
    return "cpu", torch


def make_river_dataset(
    frame: Any,
    *,
    target_col: str = "value",
    station_col: str = "station_guid",
    time_idx_col: str = "time_idx",
    encoder_length: int = 168,
    horizon: int = 24,
    known_reals: Sequence[str] = (
        "hour",
        "day_of_week",
        "day_of_year",
        "hour_sin",
        "hour_cos",
    ),
    train_end: Any | None = None,
) -> Any:
    """Build a PyTorch Forecasting ``TimeSeriesDataSet`` for river levels."""

    _, _, _, TimeSeriesDataSet, extras = _require_stack()
    GroupNormalizer, _ = extras
    required = {target_col, station_col, time_idx_col, *known_reals}
    if train_end is not None:
        required.add("dateTime")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"river frame is missing TFT columns: {sorted(missing)}")
    working = frame.copy()
    working[station_col] = working[station_col].astype(str)
    if train_end is not None:
        training_frame = working.loc[working["dateTime"] < train_end].copy()
    else:
        training_frame = working
    if training_frame.empty:
        raise ValueError("TFT training frame is empty after train_end filtering")
    static_categoricals = [station_col]
    for column in ("station_name", "river", "role"):
        if column in training_frame.columns:
            training_frame[column] = training_frame[column].astype(str).fillna("")
            static_categoricals.append(column)
    return TimeSeriesDataSet(
        training_frame,
        time_idx=time_idx_col,
        target=target_col,
        group_ids=[station_col],
        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,
        min_prediction_length=horizon,
        max_prediction_length=horizon,
        static_categoricals=static_categoricals,
        time_varying_known_reals=list(known_reals),
        time_varying_unknown_reals=[target_col],
        target_normalizer=GroupNormalizer(groups=[station_col]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )


def make_validation_dataset(training: Any, frame: Any, *, predict: bool = True) -> Any:
    """Reuse training encoders/scalers for a chronological evaluation frame."""

    _, _, _, TimeSeriesDataSet, _ = _require_stack()
    if frame.empty:
        raise ValueError("validation frame is empty")
    return TimeSeriesDataSet.from_dataset(training, frame, predict=predict, stop_randomization=True)


def fit_river_tft(
    training: Any,
    validation: Any,
    *,
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
) -> tuple[Any, Any, str]:
    """Fit a small, reproducible TFT and return ``(model, trainer, device)``."""

    _, pl, TemporalFusionTransformer, _, extras = _require_stack()
    device, _ = resolve_accelerator(accelerator)
    GroupNormalizer, QuantileLoss = extras
    del GroupNormalizer
    pl.seed_everything(seed, workers=True)
    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    validation_loader = validation.to_dataloader(
        train=False, batch_size=max(batch_size, 128), num_workers=0
    )
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=QuantileLoss(quantiles=list(QUANTILES)),
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
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    return model, trainer, device


def predict_quantiles(
    model: Any,
    dataloader: Any,
    *,
    accelerator: str = "cpu",
) -> tuple[Any, Any]:
    """Return ``(actual, quantiles)`` arrays using a specified prediction device."""

    np = __import__("numpy")
    prediction = model.predict(
        dataloader,
        mode="quantiles",
        return_y=True,
        trainer_kwargs={"accelerator": accelerator, "devices": 1, "logger": False},
    )
    output = prediction.output if hasattr(prediction, "output") else prediction
    target = prediction.y if hasattr(prediction, "y") else None
    if target is None:
        raise RuntimeError("TFT prediction did not include targets; use return_y=True")
    if isinstance(target, (tuple, list)):
        target = target[0]
    if hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    if hasattr(target, "detach"):
        target = target.detach().cpu().numpy()
    output = np.asarray(output, dtype=float)
    target = np.asarray(target, dtype=float)
    if target.ndim == 3 and target.shape[-1] == 1:
        target = target[..., 0]
    if output.ndim != target.ndim + 1 or output.shape[:-1] != target.shape:
        raise RuntimeError(f"unexpected TFT shapes: output={output.shape}, target={target.shape}")
    return target, output


def interpret_quantiles(model: Any, raw_output: Any) -> Any:
    """Expose PyTorch Forecasting interpretation tensors without ranking prose."""

    if not hasattr(model, "interpret_output"):
        raise TypeError("model does not expose interpret_output")
    return model.interpret_output(raw_output, reduction="sum")


def collect_interpretation(model: Any, dataloader: Any, *, accelerator: str = "cpu") -> dict[str, Any]:
    """Collect JSON-safe variable-selection and attention summaries."""

    np = __import__("numpy")
    raw = model.predict(
        dataloader,
        mode="raw",
        return_x=True,
        trainer_kwargs={"accelerator": accelerator, "devices": 1, "logger": False},
    )
    network_output = raw.output if hasattr(raw, "output") else raw
    interpreted = interpret_quantiles(model, network_output)

    def _values(value: Any) -> list[float]:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=float).reshape(-1)
        return [float(item) for item in array]

    def _labels(attribute: str, count: int, fallback: Sequence[str]) -> list[str]:
        values = getattr(model, attribute, None)
        if values is None:
            return list(fallback[:count]) + [f"variable_{i}" for i in range(len(fallback), count)]
        labels = [str(value) for value in values]
        return labels[:count] + [f"variable_{i}" for i in range(len(labels), count)]

    encoder_values = _values(interpreted.get("encoder_variables", []))
    decoder_values = _values(interpreted.get("decoder_variables", []))
    static_values = _values(interpreted.get("static_variables", []))
    attention_values = _values(interpreted.get("attention", []))
    return {
        "encoder_variables": dict(
            zip(
                _labels("encoder_variables", len(encoder_values), ["value"]),
                encoder_values,
            )
        ),
        "decoder_variables": dict(
            zip(
                _labels("decoder_variables", len(decoder_values), ["calendar"]),
                decoder_values,
            )
        ),
        "static_variables": dict(
            zip(
                _labels("static_variables", len(static_values), ["station_guid"]),
                static_values,
            )
        ),
        "attention": attention_values,
        "notes": [
            "Values are PyTorch Forecasting interpret_output reductions for the supplied evaluation loader.",
            "They are descriptive model diagnostics, not causal effects or operational guidance.",
        ],
    }


def run_river_tft(
    *,
    frame: Any,
    split: Mapping[str, Any],
    max_windows: int | None = 16,
    station_guids: Sequence[str] | None = None,
    max_epochs: int = 5,
    batch_size: int = 64,
    hidden_size: int = 16,
    attention_head_size: int = 2,
    hidden_continuous_size: int = 8,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    accelerator: str = "cpu",
    limit_train_batches: float | int = 1.0,
    seed: int = 42,
    collect_interpretability: bool = True,
) -> dict[str, Any]:
    """Train/evaluate one small per-station TFT on chronological river data.

    Invalid hourly aggregates are excluded from the dataset rather than
    imputed. ``allow_missing_timesteps`` preserves their positions through
    the integer hourly axis. Test predictions use all eligible windows and are
    sub-sampled at the configured rolling-origin step before the optional
    ``max_windows`` cap.
    """

    pd = __import__("pandas")
    np = __import__("numpy")
    from .baselines import score_quantiles
    required_split = {
        "train_end_exclusive",
        "validation_end_exclusive",
        "test_end_exclusive",
        "encoder_hours",
        "horizon_hours",
        "rolling_origin_step_hours",
    }
    missing_split = required_split - set(split)
    if missing_split:
        raise ValueError(f"river TFT split is missing keys: {sorted(missing_split)}")
    required_frame = {"station_guid", "dateTime", "value", "time_idx"}
    missing_frame = required_frame - set(frame.columns)
    if missing_frame:
        raise ValueError(f"river TFT frame is missing columns: {sorted(missing_frame)}")

    def _timestamp(value: Any) -> Any:
        parsed = pd.Timestamp(value)
        return parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")

    train_end = _timestamp(split["train_end_exclusive"])
    validation_end = _timestamp(split["validation_end_exclusive"])
    test_end = _timestamp(split["test_end_exclusive"])
    encoder = int(split["encoder_hours"])
    horizon = int(split["horizon_hours"])
    step = int(split["rolling_origin_step_hours"])
    if max_epochs <= 0 or batch_size <= 0:
        raise ValueError("max_epochs and batch_size must be positive")

    working = frame.copy()
    working["dateTime"] = pd.to_datetime(working["dateTime"], errors="raise", utc=True)
    working["station_guid"] = working["station_guid"].astype(str)
    working["value"] = pd.to_numeric(working["value"], errors="coerce")
    selected = list(station_guids) if station_guids else sorted(working["station_guid"].unique())
    per_station: list[dict[str, Any]] = []
    all_actual: list[Any] = []
    all_prediction: list[Any] = []
    calibration_forecast_records: list[dict[str, Any]] = []
    forecast_records: list[dict[str, Any]] = []

    for guid in selected:
        group = working.loc[working["station_guid"] == guid].sort_values("dateTime").copy()
        finite = group["value"].notna()
        train_frame = group.loc[(group["dateTime"] < train_end) & finite].copy()
        if len(train_frame) < encoder + horizon:
            per_station.append(
                {"station_guid": guid, "status": "skipped", "reason": "insufficient training rows"}
            )
            continue
        training = make_river_dataset(
            train_frame,
            encoder_length=encoder,
            horizon=horizon,
        )
        validation_context = group.loc[
            (group["dateTime"] >= train_end - pd.Timedelta(hours=encoder))
            & (group["dateTime"] < validation_end)
            & finite
        ].copy()
        if validation_context.empty:
            per_station.append(
                {"station_guid": guid, "status": "skipped", "reason": "empty validation context"}
            )
            continue
        validation = make_validation_dataset(training, validation_context, predict=True)
        model, trainer, device = fit_river_tft(
            training,
            validation,
            max_epochs=max_epochs,
            batch_size=batch_size,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            hidden_continuous_size=hidden_continuous_size,
            dropout=dropout,
            learning_rate=learning_rate,
            accelerator=accelerator,
            limit_train_batches=limit_train_batches,
            seed=seed,
        )
        del trainer
        # Generate validation-period rolling windows for residual calibration;
        # these are never mixed with the held-out test records below.
        validation_eval_dataset = make_validation_dataset(
            training, validation_context, predict=False
        )
        validation_loader = validation_eval_dataset.to_dataloader(
            train=False, batch_size=max(batch_size, 128), num_workers=0
        )
        validation_actual, validation_prediction = predict_quantiles(
            model, validation_loader, accelerator=device
        )
        if validation_actual.ndim != 2 or validation_prediction.ndim != 3:
            raise RuntimeError(
                f"unexpected river TFT validation arrays for {guid}: "
                f"actual={validation_actual.shape}, prediction={validation_prediction.shape}"
            )
        validation_indices = np.arange(validation_actual.shape[0])[::step]
        if max_windows is not None:
            validation_indices = validation_indices[:max_windows]
        validation_actual = validation_actual[validation_indices]
        validation_prediction = validation_prediction[validation_indices]
        validation_lookup = {
            int(row["time_idx"]): row["dateTime"]
            for _, row in validation_context.iterrows()
        }
        selected_validation_index = validation_eval_dataset.index.iloc[
            validation_indices
        ].reset_index(drop=True)
        for index_row, actual_path, prediction_path in zip(
            selected_validation_index.to_dict(orient="records"),
            validation_actual,
            validation_prediction,
        ):
            target_start_idx = int(index_row["time"]) + encoder
            origin = validation_lookup.get(target_start_idx)
            for offset, (actual_value, prediction_values) in enumerate(
                zip(actual_path, prediction_path), start=1
            ):
                target_time = validation_lookup.get(target_start_idx + offset - 1)
                calibration_forecast_records.append(
                    {
                        "station_guid": guid,
                        "origin": origin,
                        "target_time": target_time,
                        "horizon_step": offset,
                        "actual": float(actual_value),
                        "p10": float(prediction_values[0]),
                        "p50": float(prediction_values[1]),
                        "p90": float(prediction_values[2]),
                        "residual": float(actual_value - prediction_values[1]),
                    }
                )
        test_context = group.loc[
            (group["dateTime"] >= validation_end - pd.Timedelta(hours=encoder))
            & (group["dateTime"] < test_end)
            & finite
        ].copy()
        if test_context.empty:
            per_station.append(
                {"station_guid": guid, "status": "skipped", "reason": "empty test context"}
            )
            continue
        test_dataset = make_validation_dataset(training, test_context, predict=False)
        test_loader = test_dataset.to_dataloader(
            train=False, batch_size=max(batch_size, 128), num_workers=0
        )
        actual, prediction = predict_quantiles(model, test_loader, accelerator=device)
        if actual.ndim != 2 or prediction.ndim != 3:
            raise RuntimeError(
                f"unexpected river TFT arrays for {guid}: actual={actual.shape}, prediction={prediction.shape}"
            )
        indices = np.arange(actual.shape[0])[::step]
        if max_windows is not None:
            indices = indices[:max_windows]
        actual = actual[indices]
        prediction = prediction[indices]
        metrics = score_quantiles(actual, prediction, QUANTILES)
        interpretation = (
            collect_interpretation(model, test_loader, accelerator=device)
            if collect_interpretability
            else None
        )
        time_lookup = {
            int(row["time_idx"]): row["dateTime"]
            for _, row in test_context.iterrows()
        }
        selected_index = test_dataset.index.iloc[indices].reset_index(drop=True)
        for sample_number, (index_row, actual_path, prediction_path) in enumerate(
            zip(selected_index.to_dict(orient="records"), actual, prediction)
        ):
            target_start_idx = int(index_row["time"]) + encoder
            origin = time_lookup.get(target_start_idx)
            for offset, (actual_value, prediction_values) in enumerate(
                zip(actual_path, prediction_path), start=1
            ):
                target_time = time_lookup.get(target_start_idx + offset - 1)
                forecast_records.append(
                    {
                        "station_guid": guid,
                        "origin": origin,
                        "target_time": target_time,
                        "horizon_step": offset,
                        "actual": float(actual_value),
                        "p10": float(prediction_values[0]),
                        "p50": float(prediction_values[1]),
                        "p90": float(prediction_values[2]),
                        "residual": float(actual_value - prediction_values[1]),
                    }
                )
        per_station.append(
            {
                "station_guid": guid,
                "status": "measured",
                "device": device,
                "train_windows": int(len(training)),
                "test_windows": int(actual.shape[0]),
                "metrics": metrics.as_dict(),
                "interpretability": interpretation,
            }
        )
        all_actual.append(actual)
        all_prediction.append(prediction)

    pooled: dict[str, Any] = {"status": "no_forecasts"}
    if all_actual:
        pooled_metrics = score_quantiles(
            np.concatenate(all_actual), np.concatenate(all_prediction), QUANTILES
        )
        pooled = {
            "status": "measured",
            "metrics": pooled_metrics.as_dict(),
            "test_windows": int(sum(values.shape[0] for values in all_actual)),
        }
    return {
        "status": "measured" if all_actual else "no_forecasts",
        "config": {
            "encoder_hours": encoder,
            "horizon_hours": horizon,
            "rolling_origin_step_hours": step,
            "max_windows": max_windows,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "attention_head_size": attention_head_size,
            "hidden_continuous_size": hidden_continuous_size,
            "learning_rate": learning_rate,
            "accelerator": accelerator,
            "seed": seed,
        },
        "by_station": per_station,
        "pooled": pooled,
        "calibration_forecasts": calibration_forecast_records,
        "forecasts": forecast_records,
        "notes": [
            "Invalid hourly aggregates were excluded; no target imputation was applied.",
            "Windows are sampled from the test context at rolling_origin_step_hours.",
            "The result is retrospective research evaluation, not a warning feed.",
        ],
    }


__all__ = [
    "QUANTILES",
    "fit_river_tft",
    "interpret_quantiles",
    "collect_interpretation",
    "make_river_dataset",
    "make_validation_dataset",
    "predict_quantiles",
    "resolve_accelerator",
    "run_river_tft",
]
