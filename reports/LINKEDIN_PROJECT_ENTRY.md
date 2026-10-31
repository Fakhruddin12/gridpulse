# GridPulse — LinkedIn Projects entry

## Project name
GridPulse — Great Britain Electricity-Demand Forecasting

## Description
Built a public-data forecasting pipeline for the next 24 hourly values of Great Britain's National Demand, using the previous 168 hours and calendar information.

Implemented UK settlement-time and daylight-saving handling, UTC aggregation, forecast-window construction and training-only scaling. Compared a weekly seasonal baseline, multi-output Ridge regression and a temporal convolutional network using three expanding-window validation blocks. Selected Ridge on validation before evaluating 856 later daily forecasts, covering 20,544 target hours.

Ridge reduced holdout MAE from 2,053 MW to 1,164 MW, a 43.3% reduction. On the highest-demand decile, MAE was 1,273 MW versus 2,893 MW for the weekly baseline. The nominal 80% residual interval covered 75.0% of observations, identifying uncertainty estimation as an area for improvement.

The report includes a fresh-data replay, a saved prediction ledger and an audit of the differences between model inputs. Results describe a retrospective benchmark on revised actuals published 21 days in arrears; live data availability and operational value remain to be tested.

## Skills to associate
Time Series Analysis; Forecasting; Python; Machine Learning; Model Validation

Choose the closest existing label in LinkedIn's selector if needed.

## Project URL
https://github.com/Fakhruddin12/gridpulse

## Media attachment
- File: GridPulse_Project_Report.pdf
- Title: GridPulse — Electricity-Demand Forecasting: Methods and Results
- Description: Research report covering NESO demand data, settlement-time handling, direct Ridge and temporal-convolution forecasts, chronological model selection, holdout errors, interval coverage and replay evidence. Explains model-input differences and historical-data availability limits.

## Dates and association
Enter the actual project dates from your records. Data periods and the report assembly date are not project start/end dates. Add an institutional association only if it reflects the work's actual origin.

## Evidence notes — not part of the pasted description
- Source version: be8e0c15049dc46c23d985f46d64953443ff6d1b.
- Recorded series: 102,863 hours, 1 January 2015–25 September 2026 at 22:00 UTC.
- 4,278 supervised daily windows: 3,422 development and 856 holdout.
- Validation block lengths are 342 origins each; the original configuration's validation_days value did not control the block size; that unused field has now been removed.
- Test target blocks start on 22 May 2024 through 24 September 2026. The last predicted hour is 24 September at 23:00 UTC.
- Mean validation MAE: Ridge 981.9187 MW; TCN 1,135.2253 MW; weekly baseline 2,050.1463 MW. The TCN was not evaluated on the final holdout after losing selection.
- Recorded holdout Ridge/weekly MAE: 1,164.4194 / 2,053.3037 MW; RMSE: 1,661.1545 / 2,781.2666 MW.
- Peak-hour MAE means the highest-demand decile defined by realised evaluation outcomes, not fixed evening clock hours.
- Ridge accesses 168 history hours plus an explicit future-calendar matrix. The supplied TCN has a maximum 61-hour receptive field and no explicit future-calendar matrix. This is a comparison of supplied specifications, not a universal model-class ranking.
- The fresh-download replay is clipped to the recorded endpoint. Original source byte identity is not established; old and new manifests are recorded separately.
- NESO's historical dataset is published with a delay and subject to revisions. Chronological splitting alone does not establish real-time input availability.
- Nine repository tests passed. Thirty-eight independent evidence checks passed, including the complete saved-model prediction roundtrip. Exact checks and replay differences are in GridPulse_Verification.json.
- Full new replay evidence is in GridPulse_Replay_Evidence.zip. It does not establish dispatch savings or trading returns.

## Public report link
https://github.com/Fakhruddin12/gridpulse/blob/master/reports/PROJECT_REPORT.pdf

## October review clarification — evidence note
Saved holdout MAE is 1,164.4194 MW versus weekly copying 2,053.3037 MW, a 43.29% reduction. The fresh-history replay selects the same family and gives 1,164.4299 MW. Nominal 80% intervals cover only 74.96% of holdout hours.

NESO historical actuals are revised and published 21 days in arrears. This is not a point-in-time day-ahead deployment test. Ridge receives 168 history hours and future calendar features; the TCN receptive field is at most 61 hours and receives no equivalent future-calendar matrix. Forecast blocks are 24 UTC hours, not 23/25-hour UK settlement days.
