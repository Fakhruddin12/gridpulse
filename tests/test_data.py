import numpy as np
import pandas as pd

from gridpulse.data import calendar_matrix, load_country_series, supervised_windows
from gridpulse.evaluation import regression_metrics
from gridpulse.eda import profile
from gridpulse.models import seasonal_naive


def test_daily_seasonal_forecast_returns_the_previous_24_hours():
    history = np.arange(168.0).reshape(1, 168, 1)
    forecast = seasonal_naive(history, period=24, horizon=24)
    assert forecast.shape == (1, 24)
    np.testing.assert_array_equal(forecast[0], np.arange(144.0, 168.0))


def test_series_cleaning_and_short_gap_fill(tmp_path):
    path = tmp_path / "historic_demand_data_2020.csv"
    pd.DataFrame({
        "SETTLEMENT_DATE": ["01/01/2020"] * 6,
        "SETTLEMENT_PERIOD": [1, 2, 3, 4, 5, 6],
        "ND": [100.0, 110.0, 999.0, 999.0, 140.0, 150.0],
        "FORECAST_ACTUAL_INDICATOR": ["A", "A", "F", "F", "A", "A"],
    }).to_csv(path, index=False)
    s = load_country_series(tmp_path, "SETTLEMENT_DATE", "ND", "FORECAST_ACTUAL_INDICATOR")
    assert len(s) == 3
    assert s.isna().sum() == 0
    assert s.iloc[1] == 125.0
    assert str(s.index.tz) == "UTC"


def test_windows_do_not_overlap_forecast_targets():
    index = pd.date_range("2020-01-01", periods=24 * 10, freq="h", tz="UTC")
    s = pd.Series(np.arange(len(index), dtype=float), index=index)
    X, C, Y, origins, starts = supervised_windows(s, 48, 24, step=24)
    assert X.shape[1:] == (48, 8)
    assert C.shape[1:] == (24, 7)
    assert Y.shape[1] == 24
    assert np.all(np.diff(starts.asi8) == 24 * 60 * 60 * 10**9)


def test_metrics_are_finite():
    y = np.array([[10., 20.], [30., 40.]])
    p = y + 1
    metrics = regression_metrics(y, p)
    assert metrics["mae_mw"] == 1.0
    assert metrics["rmse_mw"] == 1.0
    assert metrics["n_forecasts"] == 2


def test_eda_profile_captures_seasonality_and_range():
    index = pd.date_range("2020-01-01", periods=24 * 21, freq="h", tz="UTC")
    s = pd.Series(100 + 10 * np.sin(2 * np.pi * index.hour / 24), index=index)
    result = profile(s)
    assert result["n_valid_hours"] == len(index)
    assert "Monday" in result["weekday_mean_mw"]
    assert result["first_order_autocorrelation_24h"] > .99


def test_settlement_periods_map_through_uk_spring_clock_change(tmp_path):
    rows = []
    for period in range(1, 47):
        rows.append({"SETTLEMENT_DATE": "29/03/2026", "SETTLEMENT_PERIOD": period, "ND": 1000 + period})
    for period in (1, 2):
        rows.append({"SETTLEMENT_DATE": "30/03/2026", "SETTLEMENT_PERIOD": period, "ND": 1100 + period})
    pd.DataFrame(rows).to_csv(tmp_path / "historic_demand_data_2026.csv", index=False)
    s = load_country_series(tmp_path, "SETTLEMENT_DATE", "ND", "FORECAST_ACTUAL_INDICATOR")
    assert len(s) == 24
    assert s.index.is_unique
    assert s.isna().sum() == 0


def test_long_internal_gaps_are_not_partially_filled(tmp_path):
    pd.DataFrame({"SETTLEMENT_DATE": ["01/01/2026"] * 4,
                  "SETTLEMENT_PERIOD": [1, 2, 9, 10], "ND": [100., 100., 500., 500.]}).to_csv(
        tmp_path / "historic_demand_data_2026.csv", index=False)
    s = load_country_series(tmp_path, "SETTLEMENT_DATE", "ND", "FORECAST_ACTUAL_INDICATOR")
    assert s.iloc[1:4].isna().all()
    assert s.iloc[0] == 100 and s.iloc[4] == 500


def test_two_hour_internal_gap_is_filled_completely(tmp_path):
    pd.DataFrame({"SETTLEMENT_DATE": ["01/01/2026"] * 4,
                  "SETTLEMENT_PERIOD": [1, 2, 7, 8], "ND": [100., 100., 400., 400.]}).to_csv(
        tmp_path / "historic_demand_data_2026.csv", index=False)
    s = load_country_series(tmp_path, "SETTLEMENT_DATE", "ND", "FORECAST_ACTUAL_INDICATOR")
    np.testing.assert_allclose(s.to_numpy(), [100., 200., 300., 400.])
