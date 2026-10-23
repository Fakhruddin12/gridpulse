# GridPulse: Great Britain electricity demand forecasting

GridPulse is a reproducible day-ahead forecasting study using National Energy System Operator (NESO) historic demand data. It compares seasonal baselines, regularized linear regression, and a temporal convolutional network (TCN), then evaluates them with expanding-window backtests and a final chronological holdout.

The project is designed around a practical energy-planning question: **how accurately can the next 24 hours of Great Britain National Demand be forecast using only information available at the forecast origin?** The resulting forecasts could inform planning discussions, but this project does not simulate dispatch, trading decisions, or financial returns.

## Results at a glance

The included benchmark covers NESO data from **1 January 2015 to 25 September 2026**. Three expanding-window validation folds were used for model comparison; the final contiguous period was kept as a locked holdout.

| Model | Mean validation MAE | Holdout MAE | Holdout RMSE |
|---|---:|---:|---:|
| Weekly seasonal naive | 2,050 MW | 2,053 MW | 2,781 MW |
| **Ridge regression (selected)** | **982 MW** | **1,164 MW** | **1,661 MW** |
| Temporal convolutional network | 1,135 MW | Not evaluated on the locked holdout | — |

Ridge reduced holdout MAE by **43.3%** relative to weekly seasonal naive. It also had lower mean validation MAE than the TCN, so the simpler model was selected. The empirical 80% residual interval covered 75.0% of holdout observations; interval calibration remains a limitation.

These are retrospective results on NESO's published National Demand series. They do not demonstrate improved dispatch, reserve savings, commercial value, or trading profit. The full evaluation details and run metadata are in [`reports/metrics.json`](../reports/metrics.json) and [`reports/run_metadata.json`](../reports/run_metadata.json).

## Data and processing

- **Source:** NESO Open Data, [Historic Demand Data](https://www.neso.energy/data-portal/historic-demand-data).
- **Target:** `ND` (National Demand, MW), retaining records marked as actuals.
- **Time handling:** NESO settlement timestamps use UK local time and cross daylight-saving transitions. The ingestion pipeline converts them to timezone-aware UTC, then aggregates half-hourly values to hourly observations.
- **Quality checks:** duplicate timestamps, missing hours, invalid demand values, and source metadata are checked and recorded.
- **Reproducibility:** source resource IDs, modification timestamps, URLs, and checksums are written to `data/raw/source_metadata.json` when the data are downloaded.

Source data are intentionally not included in this repository. The downloader retrieves them from the public NESO data portal; data files and generated model artifacts are excluded by `.gitignore`.

## Methods

The prediction task uses the previous **168 hourly demand observations** and calendar features to forecast the next **24 hours**.

- **Baselines:** previous-day and previous-week seasonal naive forecasts.
- **Statistical model:** Ridge regression using recent load history and deterministic calendar features.
- **Deep-learning benchmark:** causal dilated TCN trained to emit all 24 forecast hours directly.
- **Validation:** three expanding-window folds with non-overlapping 24-hour validation windows. The final contiguous holdout is not used for model selection. Feature scaling is fit using training data only.
- **Metrics and diagnostics:** MAE, RMSE, sMAPE, MASE, peak-hour MAE, residual intervals, seasonal slices, autocorrelation, and large-error examples.

The experiment deliberately excludes realized future weather and generation. Adding weather forecasts responsibly would require archived forecast vintages so that the backtest uses forecasts that were actually available at each historical origin.

## Quick start

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,api]"
python -m gridpulse.cli download --config configs/default.yaml
python -m gridpulse.cli prepare --config configs/default.yaml
python -m gridpulse.cli train --config configs/default.yaml
python -m gridpulse.cli report --config configs/default.yaml
```

The first run downloads the configured NESO history. Data are stored under `data/`, and model outputs are written under `artifacts/` and `reports/`.

Run the tests with:

```bash
pytest
```

## Local API

After training a model, launch the demonstration API:

```bash
uvicorn gridpulse.api:app --host 127.0.0.1 --port 8000
```

`GET /health` reports whether a trained model artifact is loaded. `POST /forecast` accepts 168 hourly load values and a timezone-aware `origin_utc`, and returns a 24-hour forecast. The API is a research demonstration, not an operational forecasting service.

## Repository layout

```text
configs/                 Reproducible data, model, and validation settings
reports/                 Included benchmark metrics and run diagnostics
src/gridpulse/           Data ingestion, EDA, features, models, evaluation, CLI, API
tests/                   Data, API, and contract tests
.github/workflows/       Continuous integration test workflow
Dockerfile               Container definition for the local API
```

## Limitations

- National Demand is an operational measure and is not the same as total end-use consumption.
- The historical series contains revised actual observations; it is not an archived stream of forecasts available to market participants at each origin.
- The forecast uses load history and calendar variables only; it does not model weather, outages, fuel prices, interconnector flows, or generation availability.
- Forecast quality has not been translated into trading P&L, reserve savings, or other business outcomes.
- The residual interval under-covered the holdout relative to its nominal 80% target and should not be described as calibrated.

## CV-ready summary

> Built a leakage-aware GB electricity-demand forecasting pipeline on 2015–2026 NESO data; compared seasonal-naive, Ridge, and TCN models with expanding-window validation, selecting Ridge at 982 MW mean CV MAE and 1,164 MW locked-holdout MAE (43% below weekly seasonal naive), with peak-load and interval-coverage diagnostics.

For a CV, describe this as a retrospective demand-forecasting benchmark. Do not imply that it generated trading returns or measured operational savings.
