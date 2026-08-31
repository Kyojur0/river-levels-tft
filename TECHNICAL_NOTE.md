# From Leaks to Levels: technical note

## Question

Can a Temporal Fusion Transformer (TFT) provide a reproducible, probabilistic
24-hour forecast of UK river level, and can its residuals support an offline
anomaly review? This repository reimplements the paper-shaped experiment using
the supported PyTorch Forecasting TFT route and transfers the pipeline to
Environment Agency (EA) station data. The goal is research evidence, not a
flood-warning service.

## Data and provenance

The EA hydrology archive is used for bulk history; the EA flood-monitoring API
is reserved for live incremental collection. The live API probe and endpoint
evidence are in `EA_HYDROLOGY_FIT_REPORT.md`. Nine geographically spread
qualified 15-minute water-level measures were frozen for
`2024-01-01 <= t < 2026-01-01`. `data/frozen/manifest.json` records request
policy, API version, row counts, retrieval time, station crosswalk fields, and
SHA-256 checksums. Raw quality/qcode/completeness fields are retained.

`src.live_collect` provides the separate `latest`/`since` station collector.
It keeps the explicit `same_as`/station-reference crosswalk and the required
real-time API attribution; it is not used to backfill the training archive.

The declared quality rule includes `Good` and `Estimated`, masks
`Suspect`/`Unchecked`/`Missing`/unknown values, rejects duplicate and off-grid
timestamps, and does not impute before a chronological split. At least three
valid quarter-hour rows are required for an hourly mean. Archive timestamps
were naive in the sampled responses, so the checked-in preparation used an
explicit UTC pipeline assumption; authoritative timezone semantics remain
unconfirmed and must be resolved before publication.

## Method

The modelling axis is hourly. Calendar variables (hour, weekday, day of year,
cyclic encodings) are known future inputs; station metadata is static when a
pooled dataset is used; level history is an observed unknown input. The TFT
uses P10/P50/P90 `QuantileLoss`, a 168-hour encoder, and a 24-hour decoder.
`src/tft.py` pins prediction to the training device and supports CPU or an
explicit MPS request. The river runner fits per-station models and excludes
masked target rows without filling them.

The baseline ladder is seasonal-naive, additive ETS, and a bounded recursive
48-lag LightGBM model. `src/evaluate.py` advances held-out origins every 24
hours. A target window is scored only when all 24 hours are finite. Scores are
P10/P50/P90 pinball loss, P50-derived MAE, and interval coverage where an
interval exists.

## Evidence

The Electricity harness downloads the UCI archive, prints its SHA-256, applies
the documented paper-style hourly preprocessing, and trains the current
PyTorch Forecasting implementation. The measured representative run used
eight series, five CPU epochs, a 128 batch, and the 168-to-24 setting. Its
P50/P90 values are recorded in `BENCHMARKS.md`; the paper's `0.055/0.027`
values remain references, not silently substituted results.

On the frozen EA panel, the bounded Q4 2025 test run produced 3,072 finite
forecast values across eight stations: seasonal naive P50/P90 `0.0740606 /
0.0726406`, ETS `0.0598863 / 0.0634740`, and recursive LightGBM `0.0785723 /
0.0655181`. Hereford Bridge had no complete test windows because its series is
masked for an extended outage. A one-epoch TFT smoke run across the remaining
eight stations produced pooled P50/P90 `0.0377434 / 0.0227445` over 384
held-out values; Kingston's corresponding values were `0.0242913 / 0.0087182`.
It demonstrates the end-to-end fit and quantile path, not model selection or a
TFT win.

The residual review calibrates station/horizon median/MAD references on the
validation period only, then scores the test period. Point-baseline intervals
are degenerate, so their outside-interval counts are diagnostic and not
calibrated false-alert rates. The Kingston TFT smoke review used its learned
intervals and the all-station smoke review flagged 213 large-residual, 79
outside-interval, and 241 combined events across 384 test rows. `ANOMALIES.md`
reports the measured counts and the required interpretation boundary.

## Reproduction

```text
conda env create -f environment.yml
conda activate river-levels-tft
python -m src.freeze --output-dir data/frozen --format jsonl
python -m src.prepare --source-timezone UTC
python -m src.overview
python -m src.river_experiment --max-windows 16 --results-json results/river_baselines.json
python -m src.river_review --model seasonal_naive --max-windows 16
```

For a real river TFT smoke run, use the Kingston command in `RESULTS.md`.
The Electricity command and checksum are in `BENCHMARKS.md`.

## Limitations

- The current benchmark is an honest PyTorch Forecasting reproduction attempt,
  not byte-for-byte TensorFlow 1.15/V100 parity or a four-dataset replication.
- The frozen EA window is two years and the raw archive is mutable; a new
  freeze can change revised rows, availability, and scores.
- Cross-station level values may use different datums/scales. Do not pool or
  compare absolute levels until station metadata is mapped.
- Hereford, Skelton, Bywell, and other stations contain material masked spans;
  missingness is part of the study and affects eligible origins.
- The one-epoch river TFT result is a smoke test. A final claim needs longer
  training, all predeclared origins, calibration diagnostics, and a locked
  model-selection protocol.
- No output is an operational warning, flood probability, or safety-critical
  recommendation.

## Attribution

Derived outputs must retain:

> Contains public sector information licensed under the Open Government Licence v3.0.
