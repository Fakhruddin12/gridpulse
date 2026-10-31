# GridPulse: Forecasting Tomorrow's Electricity Demand

[**Project report (PDF)**](reports/PROJECT_REPORT.pdf) · [**Research decision and evidence limits**](docs/RESEARCH_DECISION.md)

Saved holdout MAE is 1,164.4194 MW versus weekly copying 2,053.3037 MW, a 43.29% reduction. The fresh-history replay selects the same family and gives 1,164.4299 MW. Nominal 80% intervals cover only 74.96% of holdout hours.

**Research decision:** Retain Ridge for the studied retrospective task; establish timely inputs and improve interval coverage before an operational trial.

**How much electricity will Great Britain need in each hour of the next day?**

Electricity planning depends on anticipating demand rather than reacting after it arrives. GridPulse uses the previous week's demand and calendar information to forecast the next 24 hourly values of Great Britain's National Demand series.

I built the data pipeline and compared a simple regularised regression model with a neural network. The selected regression model reduced average absolute forecast error from **2,053 MW to 1,164 MW**, a **43.3% reduction**, compared with copying demand from the same hour one week earlier. The neural model was less accurate during validation, so the simpler model was selected.

[**Saved results**](reports/metrics.json) · [**Methods and limitations**](docs/METHODOLOGY.md) · [**Source and run record**](reports/run_metadata.json)

## What I built

- A public-data ingestion pipeline with timestamp, daylight-saving and source-integrity checks.
- A forecasting comparison using three successive validation periods and a later chronological test period.
- Training-only feature scaling and a direct forecast of all 24 hours.
- Analysis of peak-hour errors and prediction-interval coverage, with a local API demonstration and tests.

## Results in context

The test period contains **856 daily forecasts: 20,544 hourly predictions**. Mean absolute error (MAE) is the average size of an hourly mistake, measured in megawatts. Lower is better.

| Approach | Average validation MAE | Later test MAE |
|---|---:|---:|
| Copy the same hour from the previous week | 2,050 MW | 2,053 MW |
| **Ridge regression** | **982 MW** | **1,164 MW** |
| Temporal convolutional neural network | 1,135 MW | Not evaluated |

The neural network was not evaluated on the locked test period after losing the validation comparison. Test peak-hour MAE was **1,273 MW** for ridge, versus **2,893 MW** for the weekly baseline. The nominal 80% prediction interval covered only **75.0%** of observations, so its uncertainty estimates need improvement.

![Validation and test forecast errors](docs/assets/model-comparison.png)

## How it works

The pipeline downloads NESO National Demand actuals, handles UK settlement-time changes, converts observations to UTC and constructs a 168-hour history for each 24-hour forecast. Ridge is a linear model with a penalty that limits overly large coefficients. The neural comparison is a causal temporal convolutional network. Three expanding training windows compare the models before the final contiguous test period.

The saved benchmark uses data from January 2015 to September 2026. [Exact dates, metrics and evaluation details](docs/METHODOLOGY.md).

## Run it

Python 3.10+; from the repository root:

```bash
python -m pip install -e ".[dev,api,deep-learning]"
python -m gridpulse.cli download --config configs/default.yaml
python -m gridpulse.cli prepare --config configs/default.yaml
python -m gridpulse.cli train --config configs/default.yaml
python -m gridpulse.cli report --config configs/default.yaml
python -m pytest -q
```

After fitting, `uvicorn gridpulse.api:app --host 127.0.0.1 --port 8000` starts the local forecast demonstration. [Source code](src/gridpulse/) · [Tests](tests/).

## Scope

The series contains revised historical actuals published 21 days in arrears, not archived real-time data vintages. A point-in-time day-ahead claim is therefore not established. Ridge and TCN receive different effective context and calendar information; this is not an equal-information architecture comparison. Forecasts cover 24 UTC hours, not a UK settlement day. Weather forecasts are excluded. National Demand is not identical to total end-use consumption. Forecast improvements have not been translated into dispatch decisions, savings or trading profit.

**Author:** Fakhruddin Hakim Hussain · [Portfolio](https://github.com/Fakhruddin12)


## October 2026 code review

Only complete internal missing runs of at most two hours are now interpolated; longer gaps remain missing in full. Nine tests pass, including two gap regressions. The complete replay series has no missing input hours, so this correction leaves its benchmark unchanged. Short-gap interpolation still uses a later endpoint. Unused `minimum_train_days` and `validation_days` defaults were removed; the actual expanding-fold logic is unchanged.
