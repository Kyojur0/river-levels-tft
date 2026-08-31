# Residual anomaly review

This module is a retrospective review of forecast misses. It is not an alert
stream, flood-warning service, or safety-critical decision rule.

## Definitions

For each station and forecast horizon, residuals are `actual - P50`. The
reference location is the validation-period residual median and the scale is
`max(1.4826 * MAD, 1e-9)`. A `large_residual` event has absolute robust score at
least `3.5`. An `outside_interval` event is an actual below P10 or above P90.
The combined `anomaly` flag is their OR. References are fitted on
`2025-07-01 <= t < 2025-10-01`; only the held-out `2025-10-01 <= t < 2026-01-01`
period is scored. Target windows with masked hourly bins are skipped rather
than imputed.

Run the review with:

```text
python -m src.river_review --model seasonal_naive --max-windows 16
```

The command writes `results/anomaly_review.json` and
`results/anomaly_scored.csv`. The implementation lives in
`src/anomalies.py` and `src/river_review.py`.

## Measured diagnostic run

The checked-in seasonal-naive review used all nine configured stations for
validation calibration and 16 held-out origins per station. It produced
17,640 validation forecast rows, 192 station/horizon residual references, and
3,072 test forecast rows across eight stations. Hereford Bridge had no complete
test target windows because its series is masked during the selected test
period.

| Station group | Test forecasts | Large-residual events | Outside-interval events | Combined events |
|---|---:|---:|---:|---:|
| Colwick (Trent) | 384 | 64 | 384 | 384 |
| Skelton (Ouse) | 384 | 123 | 384 | 384 |
| Roxton (Great Ouse) | 384 | 58 | 381 | 381 |
| Thorverton (Exe) | 384 | 63 | 384 | 384 |
| Kingston (Thames) | 384 | 7 | 382 | 382 |
| Bewdley (Severn) | 384 | 143 | 383 | 383 |
| Caton (Lune) | 384 | 76 | 384 | 384 |
| Bywell (Tyne) | 384 | 60 | 384 | 384 |
| **Pooled** | **3,072** | **594** | **3,066** | **3,066** |

The outside-interval count is intentionally not interpreted as a calibrated
false-alert rate: point baselines expose identical P10/P50/P90 values, so
their intervals are degenerate. The robust residual counts are the useful
diagnostic from this baseline run. A final TFT review must fit residual
references on validation predictions from the selected TFT checkpoint and
report event counts by station and horizon with the learned P10/P90 paths.
No ground-truth anomaly labels are present in this archive, so a true/false
alert rate cannot be estimated from this study; the reported anomaly rates are
model-miss diagnostics only.

## TFT smoke review

The one-station Kingston TFT smoke run now emits separate validation and test
forecast records. Reviewing `results/river_one_station_tft.json` with:

```text
python -m src.river_review \
  --tft-results results/river_one_station_tft.json \
  --results-json results/tft_anomaly_review.json \
  --scored-csv results/tft_anomaly_scored.csv
```

The all-station smoke result `results/river_all_stations_tft_epoch1.json` was
reviewed with the same command (substitute that path). It uses 384 validation
rows for 192 station/horizon references and 384 held-out rows across eight
stations; Hereford Bridge is skipped because its outage leaves no validation
context.

| Station | Test rows | Large residual | Outside P10/P90 | Combined |
|---|---:|---:|---:|---:|
| Colwick | 48 | 20 | 2 | 22 |
| Skelton | 48 | 41 | 0 | 41 |
| Roxton | 48 | 30 | 22 | 45 |
| Thorverton | 48 | 45 | 41 | 48 |
| Kingston | 48 | 30 | 12 | 38 |
| Bewdley | 48 | 29 | 2 | 29 |
| Caton | 48 | 2 | 0 | 2 |
| Bywell | 48 | 16 | 0 | 16 |
| **Pooled** | **384** | **213** | **79** | **241** |

These are outputs of one-epoch, two-origin fits and are not estimates of a
production false-alert rate. They should be rerun after any model-selection or
data-freeze change.

## Caveats

- The archive is revised and quality-masked; rerunning the freeze can change
  both forecast availability and event counts.
- The UTC axis is the explicit pipeline assumption documented in
  `DATA_QUALITY.md`; authoritative archive timezone semantics remain
  `[UNVERIFIED]`.
- No event count here is a flood likelihood, warning, or operational alarm.
