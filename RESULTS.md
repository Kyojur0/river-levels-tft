# River TFT results

## Evaluation contract

The checked-in run uses `configs/splits.yml`: train ends `2025-07-01`,
validation ends `2025-10-01`, and the held-out test window is
`2025-10-01 <= t < 2026-01-01` on the explicit UTC modelling axis. The model
input is the hourly aggregate of each qualified 15-minute level series. A
target window is scored only when all 24 target hours are finite; masked bins
are not imputed. Origins advance every 24 hours and are capped at 16 per
station for a laptop-sized, reproducible check. The corresponding measured
JSON is `results/river_baselines.json`.

## Measured baseline ladder

| Model | Status | Stations with eligible test windows | Forecast values | P50 pinball | P90 pinball | MAE from P50 |
|---|---|---:|---:|---:|---:|---:|
| Seasonal naive (latest 24-hour cycle) | measured | 8 | 3,072 | 0.0740606 | 0.0726406 | 0.148121 |
| ETS (additive Holt-Winters) | measured | 8 | 3,072 | 0.0598863 | 0.0634740 | 0.119773 |
| LightGBM (recursive 48-lag context) | measured | 8 | 3,072 | 0.0785723 | 0.0655181 | 0.157145 |
| TFT | measured for one station smoke run only | 1 | 48 | 0.0242913 | 0.0087182 | 0.0485825 |
| TFT | measured per-station smoke run | 8 | 384 | 0.0377434 | 0.0227445 | 0.0754868 |

The baseline scores are pooled across the eight stations with finite held-out
windows. Hereford Bridge has no eligible complete 24-hour target windows in
the selected Q4 2025 test period because its archive series is masked for a
long outage; it is retained as a stress-test station rather than silently
dropped from the catalogue. Station-level baseline metrics are in the JSON.

ETS is the lowest pooled P50 loss in this bounded run. LightGBM does not win
this comparison. The one-station TFT value is a real CPU, one-epoch
PyTorch-Forecasting run on Kingston with the requested 168-hour encoder and
24-hour horizon, two validation-calibrated and two held-out rolling origins;
it is a pipeline smoke/evidence run, not a tuned claim that TFT beats ETS or
the paper. The full per-station TFT study still requires a longer training
schedule and a predeclared model-selection protocol.

The nine-station, one-epoch TFT smoke run is in
`results/river_all_stations_tft_epoch1.json`. Eight stations produced two
validation-calibrated and two held-out origins each. Hereford Bridge was
skipped because its masked outage leaves no validation context. Per-station
P50/P90/MAE values were: Colwick `0.0279538/0.0212620/0.0559076`, Skelton
`0.0532578/0.0321248/0.106516`, Roxton `0.0191547/0.0114058/0.0383094`,
Thorverton `0.0467124/0.0146771/0.0934248`, Kingston
`0.0242913/0.0087182/0.0485825`, Bewdley `0.0498073/0.0433767/0.0996145`,
Caton `0.0533705/0.0194122/0.106741`, and Bywell
`0.0273994/0.0309792/0.0547987`.

## River TFT run

Run a bounded real station fit with:

```text
python -m src.river_experiment \
  --station 8496ce69-482c-406a-a2f0-ac418ef8f099 \
  --run-tft --tft-accelerator cpu --tft-max-epochs 1 --tft-batch-size 128 \
  --max-windows 2 --results-json results/river_one_station_tft.json
```

The adapter uses `pytorch-forecasting`'s `TemporalFusionTransformer`, static
station metadata where available, known calendar covariates, and observed
level history. Invalid hourly bins are excluded without filling. The default
runner forces CPU for reproducible local evidence; MPS is available as an
explicit experiment option but must be validated separately on the target
machine.

## Limitations and honest losses

- The current river TFT score is not a publication-ready comparison: one
  station, one epoch, one capped test origin, and no hyperparameter search.
- Point baselines are represented by identical P10/P50/P90 paths. Their
  interval coverage is therefore diagnostic and not probabilistically
  calibrated.
- The Q4 2025 test boundary is a fixed study example. A final analysis should
  rerun the frozen-data quality checks and report all rolling origins, not only
  the cap used here.
- Cross-station pooling is not interpreted until datum/scale metadata is
  mapped. The current TFT path is per-station.

## Reproduction

```text
python -m src.river_experiment --max-windows 16 --results-json results/river_baselines.json
python -m src.river_review --model seasonal_naive --max-windows 16
python -m src.river_experiment --run-tft --tft-accelerator cpu --tft-max-epochs 1 --max-windows 2 --results-json results/river_all_stations_tft_epoch1.json
python -m src.river_review --tft-results results/river_all_stations_tft_epoch1.json
```

Every measured output is retrospective research evidence. This project is
not a flood-warning service and must not be used for safety-critical decisions.
