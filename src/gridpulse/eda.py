from __future__ import annotations

import numpy as np
import pandas as pd


def profile(series: pd.Series) -> dict:
    clean = series.dropna()
    if clean.empty:
        raise ValueError("No valid load observations to profile")
    frame = clean.rename("load_mw").to_frame()
    frame["hour_utc"] = frame.index.hour
    frame["day_of_week"] = frame.index.day_name()
    frame["month"] = frame.index.month
    daily = clean.resample("D").agg(["mean", "max", "min"])
    hourly = frame.groupby("hour_utc")["load_mw"].agg(["mean", "median", "std"])
    weekday = frame.groupby("day_of_week")["load_mw"].mean().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    return {
        "n_valid_hours": int(len(clean)),
        "time_range_utc": [str(clean.index.min()), str(clean.index.max())],
        "missing_fraction": float(series.isna().mean()),
        "load_quantiles_mw": {str(q): float(clean.quantile(q)) for q in [.01, .1, .5, .9, .99]},
        "hour_of_day_mean_mw": {str(k): float(v) for k, v in hourly["mean"].items()},
        "weekday_mean_mw": {str(k): float(v) for k, v in weekday.items()},
        "monthly_mean_mw": {str(k): float(v) for k, v in frame.groupby("month")["load_mw"].mean().items()},
        "daily_mean_range_mw": [float(daily["mean"].min()), float(daily["mean"].max())],
        "daily_peak_p99_mw": float(daily["max"].quantile(.99)),
        "first_order_autocorrelation_24h": float(clean.autocorr(24)),
        "first_order_autocorrelation_168h": float(clean.autocorr(168)),
        "interpretation_caution": "Descriptive patterns only; no causal attribution.",
    }
