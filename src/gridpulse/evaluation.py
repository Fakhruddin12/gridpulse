from __future__ import annotations

import numpy as np


def regression_metrics(y_true, y_pred, seasonal_scale=None):
    y, p = np.asarray(y_true), np.asarray(y_pred)
    err = p - y
    denom = np.abs(y) + np.abs(p)
    smape = np.where(denom > 0, 2 * np.abs(err) / denom, 0).mean() * 100
    out = {
        "mae_mw": float(np.mean(np.abs(err))),
        "rmse_mw": float(np.sqrt(np.mean(err ** 2))),
        "smape_pct": float(smape),
        "peak_hour_mae_mw": float(np.mean(np.abs(err)[y >= np.quantile(y, .9)])),
        "n_forecasts": int(y.shape[0]),
        "n_hourly_predictions": int(y.size),
    }
    if seasonal_scale is not None and seasonal_scale > 0:
        out["mase"] = float(np.mean(np.abs(err)) / seasonal_scale)
    return out


def interval_metrics(y_true, y_pred, train_residuals, alpha=.2):
    lo, hi = np.quantile(train_residuals, [alpha / 2, 1 - alpha / 2])
    lower, upper = np.asarray(y_pred) + lo, np.asarray(y_pred) + hi
    y = np.asarray(y_true)
    return {
        "nominal_coverage": 1 - alpha,
        "empirical_coverage": float(np.mean((y >= lower) & (y <= upper))),
        "mean_interval_width_mw": float(upper.mean() - lower.mean()),
        "residual_lower_mw": float(lo),
        "residual_upper_mw": float(hi),
    }
