"""Small, dependency-light baselines for the Electricity benchmark.

The benchmark runner imports this module before importing PyTorch.  Seasonal
naive therefore remains usable in a minimal Python environment, while ETS and
LightGBM are optional and fail with an actionable message when their optional
dependencies are not installed.

No function in this module writes results.  Callers decide where (or whether)
to persist a result after inspecting the values returned here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


class OptionalBaselineUnavailable(RuntimeError):
    """Raised when an optional baseline dependency is not installed."""


@dataclass(frozen=True)
class ForecastMetrics:
    """Pinball losses and MAE for one forecast collection."""

    p10: float
    p50: float
    p90: float
    mae: float
    n_values: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "p10_pinball": self.p10,
            "p50_pinball": self.p50,
            "p90_pinball": self.p90,
            "mae_from_p50": self.mae,
            "n_values": self.n_values,
        }


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Return mean pinball loss for one quantile.

    This intentionally mirrors the quantile-loss convention used by
    ``pytorch_forecasting.metrics.QuantileLoss``.  Inputs are flattened and
    finite pairs only; callers should report how many values were retained.
    """

    truth, prediction = _finite_pairs(y_true, y_pred)
    if truth.size == 0:
        return float("nan")
    error = truth - prediction
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def score_quantiles(
    y_true: np.ndarray,
    predictions: np.ndarray,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> ForecastMetrics:
    """Score ``[..., n_quantiles]`` predictions against ``y_true``.

    ``predictions`` may also be ``[..., 1]`` for point baselines.  A point
    forecast is treated as a deliberately degenerate forecast at every
    requested quantile; this is useful for comparison but is not a calibrated
    probabilistic baseline.
    """

    y_true = np.asarray(y_true, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim == y_true.ndim:
        predictions = predictions[..., None]
    if predictions.ndim != y_true.ndim + 1:
        raise ValueError(
            "predictions must have shape y_true.shape + (n_quantiles,), "
            f"got y_true={y_true.shape}, predictions={predictions.shape}"
        )
    if predictions.shape[:-1] != y_true.shape:
        raise ValueError(
            "prediction and target shapes do not align: "
            f"y_true={y_true.shape}, predictions={predictions.shape}"
        )

    q_values = tuple(float(q) for q in quantiles)
    if len(q_values) != predictions.shape[-1]:
        if predictions.shape[-1] == 1:
            predictions = np.repeat(predictions, len(q_values), axis=-1)
        else:
            raise ValueError(
                "number of quantiles does not match the final prediction axis: "
                f"{len(q_values)} != {predictions.shape[-1]}"
            )

    p_losses = [
        pinball_loss(y_true, predictions[..., i], q)
        for i, q in enumerate(q_values)
    ]
    median_index = min(range(len(q_values)), key=lambda i: abs(q_values[i] - 0.5))
    truth, median_prediction = _finite_pairs(y_true, predictions[..., median_index])
    mae = float(np.mean(np.abs(truth - median_prediction))) if truth.size else float("nan")
    return ForecastMetrics(
        p10=p_losses[q_values.index(0.1)] if 0.1 in q_values else float("nan"),
        p50=p_losses[q_values.index(0.5)] if 0.5 in q_values else float("nan"),
        p90=p_losses[q_values.index(0.9)] if 0.9 in q_values else float("nan"),
        mae=mae,
        n_values=int(truth.size),
    )


def seasonal_naive_forecast(
    history: Sequence[float], horizon: int, seasonality: int = 24
) -> np.ndarray:
    """Repeat the most recent seasonal cycle for ``horizon`` steps."""

    values = np.asarray(history, dtype=float)
    if horizon <= 0:
        return np.empty(0, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full(horizon, np.nan, dtype=float)
    if seasonality <= 0:
        raise ValueError("seasonality must be positive")
    if values.size < seasonality:
        return np.full(horizon, finite[-1], dtype=float)

    cycle = values[-seasonality:]
    if not np.isfinite(cycle).all():
        # Keep the baseline deterministic when a recent cycle contains gaps.
        fill = finite[-1]
        cycle = np.where(np.isfinite(cycle), cycle, fill)
    return np.resize(cycle, horizon).astype(float, copy=False)


def rolling_windows(
    values: Sequence[float],
    encoder_length: int,
    horizon: int,
    max_windows: int | None = None,
    start: int | None = None,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield history/target windows in chronological order.

    ``start`` is an index into ``values`` at which the first target starts.
    The default leaves exactly ``encoder_length`` observations before the
    first target.  Windows containing non-finite targets are skipped; history
    gaps are left for the baseline to handle deterministically.
    """

    array = np.asarray(values, dtype=float)
    if encoder_length <= 0 or horizon <= 0:
        raise ValueError("encoder_length and horizon must be positive")
    first = encoder_length if start is None else max(encoder_length, int(start))
    emitted = 0
    for target_start in range(first, array.size - horizon + 1, horizon):
        history = array[target_start - encoder_length : target_start]
        target = array[target_start : target_start + horizon]
        if not np.isfinite(target).all():
            continue
        yield history, target
        emitted += 1
        if max_windows is not None and emitted >= max_windows:
            return


def evaluate_seasonal_naive(
    values: Sequence[float],
    encoder_length: int = 168,
    horizon: int = 24,
    seasonality: int = 24,
    max_windows: int | None = 64,
    start: int | None = None,
) -> ForecastMetrics:
    """Evaluate seasonal naive on rolling, non-overlapping windows.

    ``start`` is useful for keeping a baseline evaluation on the held-out
    portion of a panel rather than accidentally scoring training rows.
    """

    truths: list[np.ndarray] = []
    forecasts: list[np.ndarray] = []
    for history, target in rolling_windows(
        values, encoder_length, horizon, max_windows=max_windows, start=start
    ):
        truths.append(target)
        forecasts.append(seasonal_naive_forecast(history, horizon, seasonality))
    if not truths:
        return ForecastMetrics(*(float("nan"),) * 4, n_values=0)
    y_true = np.concatenate(truths)
    point = np.concatenate(forecasts)
    return score_quantiles(y_true, point)


def ets_forecast(
    history: Sequence[float], horizon: int, seasonality: int = 24
) -> np.ndarray:
    """Forecast with statsmodels' additive seasonal Holt-Winters model."""

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise OptionalBaselineUnavailable(
            "ETS requires statsmodels; install the benchmark environment first."
        ) from exc

    values = np.asarray(history, dtype=float)
    if values.size < max(2 * seasonality, 8):
        return seasonal_naive_forecast(values, horizon, seasonality)
    finite = np.isfinite(values)
    if not finite.all():
        valid = values[finite]
        if valid.size == 0:
            return np.full(horizon, np.nan)
        values = np.where(finite, values, valid[-1])
    model = ExponentialSmoothing(
        values,
        trend="add",
        seasonal="add",
        seasonal_periods=seasonality,
        initialization_method="estimated",
    ).fit(optimized=True, use_brute=False)
    return np.asarray(model.forecast(horizon), dtype=float)


def evaluate_ets(
    values: Sequence[float],
    encoder_length: int = 168,
    horizon: int = 24,
    seasonality: int = 24,
    max_windows: int | None = 16,
    start: int | None = None,
) -> ForecastMetrics:
    """Evaluate ETS on a small rolling sample (ETS is intentionally slower)."""

    truths: list[np.ndarray] = []
    forecasts: list[np.ndarray] = []
    for history, target in rolling_windows(
        values, encoder_length, horizon, max_windows=max_windows, start=start
    ):
        truths.append(target)
        forecasts.append(ets_forecast(history, horizon, seasonality))
    if not truths:
        return ForecastMetrics(*(float("nan"),) * 4, n_values=0)
    return score_quantiles(np.concatenate(truths), np.concatenate(forecasts))


def lightgbm_direct_forecast(
    history: Sequence[float],
    horizon: int,
    seasonality: int = 24,
    lags: int = 168,
    max_train_rows: int = 2_000,
    seed: int = 42,
) -> np.ndarray:
    """Fit a direct multi-output LightGBM lag model for one forecast origin.

    The function is deliberately conservative: it trains one small model per
    horizon and caps the number of training rows.  It is a contextual baseline,
    not a claim of parity with the paper's tuned competing methods.
    """

    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise OptionalBaselineUnavailable(
            "LightGBM requires lightgbm; install the benchmark environment first."
        ) from exc

    values = np.asarray(history, dtype=float)
    if values.size <= lags + horizon:
        return seasonal_naive_forecast(values, horizon, seasonality)
    finite = np.isfinite(values)
    valid = values[finite]
    if valid.size == 0:
        return np.full(horizon, np.nan)
    values = np.where(finite, values, valid[-1])

    # Lag 1..lags plus two deterministic calendar proxies.  The exact
    # timestamp is not needed for this baseline because the index is hourly.
    rows = values.size - lags - horizon + 1
    row_indices = np.arange(lags, lags + rows)
    if rows > max_train_rows:
        row_indices = row_indices[-max_train_rows:]
    x_rows = []
    y_rows = [[] for _ in range(horizon)]
    for idx in row_indices:
        lag_values = values[idx - lags : idx][::-1]
        hour = idx % 24
        dow = (idx // 24) % 7
        x_rows.append(np.concatenate((lag_values, [hour, dow])))
        for step in range(horizon):
            y_rows[step].append(values[idx + step])
    x_train = np.asarray(x_rows, dtype=float)
    models = []
    for step in range(horizon):
        model = LGBMRegressor(
            # This is a contextual rung in the ladder. Keep the direct
            # horizon models small enough to run on a laptop while retaining
            # a meaningful nonlinear lag comparison.
            n_estimators=50,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=8,
            n_jobs=1,
            random_state=seed + step,
            verbosity=-1,
        )
        model.fit(x_train, np.asarray(y_rows[step], dtype=float))
        models.append(model)

    last = values[-lags:][::-1]
    hour = values.size % 24
    dow = (values.size // 24) % 7
    features = np.concatenate((last, [hour, dow]))[None, :]
    return np.asarray([model.predict(features)[0] for model in models], dtype=float)


def evaluate_lightgbm(
    values: Sequence[float],
    encoder_length: int = 168,
    horizon: int = 24,
    seasonality: int = 24,
    max_windows: int | None = 8,
    start: int | None = None,
) -> ForecastMetrics:
    """Evaluate the direct LightGBM baseline on capped rolling windows."""

    truths: list[np.ndarray] = []
    forecasts: list[np.ndarray] = []
    array = np.asarray(values, dtype=float)
    first = encoder_length if start is None else max(encoder_length, int(start))
    emitted = 0
    for target_start in range(first, array.size - horizon + 1, horizon):
        # Unlike ETS/seasonal-naive, a lag model needs more than one encoder
        # window to learn its coefficients.  Give it all observations available
        # before this origin; the target horizon is never included in history.
        history = array[:target_start]
        target = array[target_start : target_start + horizon]
        if not np.isfinite(target).all():
            continue
        truths.append(target)
        forecasts.append(lightgbm_direct_forecast(history, horizon, seasonality))
        emitted += 1
        if max_windows is not None and emitted >= max_windows:
            break
    if not truths:
        return ForecastMetrics(*(float("nan"),) * 4, n_values=0)
    return score_quantiles(np.concatenate(truths), np.concatenate(forecasts))


def lightgbm_recursive_forecast(
    history: Sequence[float],
    horizon: int,
    *,
    seasonality: int = 24,
    lags: int = 48,
    max_train_rows: int = 2_000,
    seed: int = 42,
) -> np.ndarray:
    """Fit one one-step lag model and roll it forward recursively.

    This is the laptop-sized LightGBM rung used by the river experiment. The
    direct variant above is retained for the Electricity context benchmark,
    while this one avoids fitting 24 separate models at every rolling origin.
    The returned forecast is a point path; the evaluator records it as a
    deliberately degenerate P10/P50/P90 forecast.
    """

    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise OptionalBaselineUnavailable(
            "LightGBM requires lightgbm; install the benchmark environment first."
        ) from exc

    values = np.asarray(history, dtype=float)
    if horizon <= 0:
        return np.empty(0, dtype=float)
    if lags <= 0:
        raise ValueError("lags must be positive")
    finite = np.isfinite(values)
    valid = values[finite]
    if valid.size == 0:
        return np.full(horizon, np.nan, dtype=float)
    values = np.where(finite, values, valid[-1])
    if values.size <= lags + 1:
        return seasonal_naive_forecast(values, horizon, seasonality)

    row_indices = np.arange(lags, values.size)
    if row_indices.size > max_train_rows:
        row_indices = row_indices[-max_train_rows:]
    x_rows = []
    y_rows = []
    for idx in row_indices:
        x_rows.append(
            np.concatenate((values[idx - lags : idx][::-1], [idx % 24, (idx // 24) % 7]))
        )
        y_rows.append(values[idx])
    model = LGBMRegressor(
        n_estimators=60,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=8,
        n_jobs=1,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float))
    context = list(values)
    forecast: list[float] = []
    for _ in range(horizon):
        idx = len(context)
        features = np.concatenate(
            (np.asarray(context[-lags:], dtype=float)[::-1], [idx % 24, (idx // 24) % 7])
        )[None, :]
        value = float(model.predict(features)[0])
        forecast.append(value)
        context.append(value)
    return np.asarray(forecast, dtype=float)


def _finite_pairs(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    prediction = np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.shape != prediction.shape:
        raise ValueError(f"shape mismatch: y_true={truth.shape}, y_pred={prediction.shape}")
    keep = np.isfinite(truth) & np.isfinite(prediction)
    return truth[keep], prediction[keep]
