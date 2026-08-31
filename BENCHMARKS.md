# Electricity Benchmark

Phase 1 has a runnable harness in
[`src/benchmark_electricity.py`](src/benchmark_electricity.py). This file is a
reproduction contract, not a claim that the published score has already been
matched. The checkout records one measured current-stack result below. Any new
result must be recorded with its environment, series cap, data checksum, and
command line.

## What Is Being Compared

The reference target is the Temporal Fusion Transformer paper's Electricity
experiment (Lim et al., arXiv:1912.09363): P50 pinball loss `0.055` and P90
pinball loss `0.027`. Those are reference values only. They are not a result of
this repository and should not be presented as one.

The source is the UCI ElectricityLoadDiagrams20112014 archive used by the
Google Research TFT code:

`https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip`

The runner follows the source preprocessing closely: parse semicolon-separated
quarter-hour readings with comma decimals, aggregate to hourly means, treat
zero as missing while finding each meter's active range, fill interior active
range gaps with zero, and use the historical paper window (days 1096 through
1345). It then uses a 168-hour encoder and 24-hour forecast horizon.

## Run

Create the environment described in `environment.yml`, then from the project
root run:

```text
python -m src.benchmark_electricity --smoke
python -m src.benchmark_electricity --max-series 8 --max-epochs 5
```

The first command is a dependency-light synthetic plumbing check. Its metrics
are explicitly labelled `smoke_only` and must never be reported as Electricity
or paper results. The second command downloads the archive into the requested
cache directory, prints a SHA-256 for the input archive, evaluates the baseline
ladder, trains a small TFT, and prints measured JSON. It may be run on CPU or
MPS (`--accelerator cpu|mps|auto`). `--max-series 0` removes the deterministic
series cap; this is a heavier run and is not claimed to fit a one-hour budget.
Use `--skip-tft` to inspect data/baselines first. Optional ETS and LightGBM
baselines report `unavailable` if their packages are not installed.

The runner does not write a results file unless an explicit `--results-json`
path is supplied. It does not edit this document automatically.

## Documented Delta Table

| Reproduction component | Status | What this harness does | Delta or reason |
|---|---|---|---|
| Electricity source | **Exact source** | Downloads the UCI `LD2011_2014.txt.zip` file used by the Google Research TFT data script. | The archive is mutable/redirected; record the retrieval date and printed SHA-256 for every run. |
| Quarter-hour to hourly preprocessing | **Approximate** | Semicolon/comma parsing, hourly mean, zero-to-missing active-range handling, and interior zero fill follow the public Google script. | The original TensorFlow-era script and dependency versions are not recreated byte-for-byte. |
| Historical date window | **Approximate** | Defaults to days `1096 <= days_from_start < 1346`, matching the public formatter's filtered period. | The exact paper split/checkpoint orchestration is not exposed as a single reproducible artifact. |
| Train/validation/test boundaries | **Approximate** | Uses the public formatter boundaries (`1315` and `1339`) with seven-day context overlap. | The runner uses PyTorch Forecasting dataset objects and can differ at boundary rows. |
| Encoder/horizon | **Exact study setting** | `168` hourly encoder steps and `24` hourly forecast steps. | This is the requested comparison setting, not proof of paper parity. |
| Quantiles and loss | **Approximate** | P10/P50/P90 with pinball scoring and P50/P90 reported. | PyTorch Forecasting's current `QuantileLoss` and inverse-normalisation path replace the original TF implementation. Verify output scale before publishing a score. |
| TFT architecture | **Approximate** | `TemporalFusionTransformer.from_dataset` with configurable small hidden size, heads, dropout, and Adam. | The paper used its TensorFlow 1.15 reference code, a V100-era configuration, and tuned hyperparameters; this project deliberately uses the supported PyTorch Forecasting route on M4 CPU/MPS. |
| Optimisation/search | **Not reproduced** | One deterministic configuration is trained; no paper-scale random search is run. | Search cost and old framework constraints exceed this phase's scope. |
| Hardware/training budget | **Not reproduced** | CPU/MPS-compatible trainer with a small default epoch count. | The paper's V100 training schedule and throughput are not comparable. |
| Seasonal-naive baseline | **Context baseline** | Repeats the latest 24-hour cycle and scores rolling windows. | It is not a claim that this was one of the paper's competing models. Point forecasts are degenerate at each quantile. |
| ETS baseline | **Context baseline** | Optional statsmodels additive Holt-Winters forecast, capped rolling evaluation. | Library defaults and capped windows are intentionally documented rather than paper parity. |
| LightGBM baseline | **Context baseline** | Optional capped direct lag model. | This is scaffolding for the planned ladder, not the paper's tuned benchmark. |
| Published P50/P90 values | **Reference only; exact parity not reproduced** | `0.055` and `0.027` remain the paper comparison values. | The measured current-stack run below is a documented-delta result, not a claim of matching the TensorFlow/V100 experiment. |

## Result Recording Contract

The accepted measured run is:

| Run | P50 pinball | P90 pinball | P50 delta vs `0.055` | P90 delta vs `0.027` | Series | Epochs | Device | Archive SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 2026-08-27, eight-series CPU run | 2.9622848 | 1.4711090 | +2.9072848 | +1.4441090 | 8 | 5 | cpu | `f6c4d0e0df12ecdb9ea008dd6eef3518adb52c559d04a9bac2e1b81dcfc8d4e1` |

Command:

```text
python -m src.benchmark_electricity --accelerator cpu --max-series 8 \
  --max-epochs 5 --batch-size 128 --max-baseline-windows 4 \
  --skip-optional-baselines \
  --results-json data/benchmark/electricity/phase1_tft_8series_epoch5_cpu_fixed.json
```

Environment evidence for that run: Python 3.12.11, torch 2.13.0,
pytorch-forecasting 1.8.0, lightning 2.6.5, pandas 3.0.5, and NumPy 2.5.2.
The raw UCI archive SHA-256 is shown in the table and in the JSON result.
The measured score is intentionally retained as a current-stack reproduction
attempt; it must not be replaced with a number copied from the paper, a smoke
run, or another implementation.

## Scope Boundary

This benchmark does not claim four-dataset parity or a deployable warning
system. The EA extraction and river transfer are reported separately in
`DATA_QUALITY.md` and `RESULTS.md`. The river study is research decision-support
and must not be used for safety-critical decisions.
