# GridPulse: research decision

Reviewed 31 October 2026. This note separates the supported result from the next research decision.

## Decision

Retain Ridge for the studied retrospective task; establish timely inputs and improve interval coverage before an operational trial.

## Strongest relevant comparison

Saved holdout MAE is 1,164.4194 MW versus weekly copying 2,053.3037 MW, a 43.29% reduction. The fresh-history replay selects the same family and gives 1,164.4299 MW. Nominal 80% intervals cover only 74.96% of holdout hours.

## Evidence limits

NESO historical actuals are revised and published 21 days in arrears. This is not a point-in-time day-ahead deployment test. Ridge receives 168 history hours and future calendar features; the TCN receptive field is at most 61 hours and receives no equivalent future-calendar matrix. Forecast blocks are 24 UTC hours, not 23/25-hour UK settlement days.

## Checks and changes

The prior full trainer replay and 38 evidence checks remain valid for the complete input series. The short-gap loader fix and removal of unused default fields change no complete-series benchmark values. All nine current tests pass. Interpolation is still retrospective.

## Next independent milestone

Acquire timely archived load vintages, make gap filling causal, compare equal-context models and evaluate conditional interval coverage on a new period.

[Project report](../reports/PROJECT_REPORT.pdf) · [Prepared LinkedIn Projects entry](../reports/LINKEDIN_PROJECT_ENTRY.md)

The report and entry are portfolio summaries. Original experimental source version: `be8e0c15049dc46c23d985f46d64953443ff6d1b`. The report records verification scope and distinguishes earlier numerical replays from this presentation/code review.
