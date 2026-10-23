import joblib
import numpy as np
from fastapi.testclient import TestClient

from gridpulse.models import fit_ridge
from gridpulse.api import create_app


def test_ridge_api_returns_24_hour_forecast(tmp_path):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(20, 168, 8)).astype("float32")
    cal = rng.normal(size=(20, 24, 7)).astype("float32")
    y = rng.normal(size=(20, 24)).astype("float32")
    model = fit_ridge(X, cal, y)
    joblib.dump({"selected_model": "ridge", "model": model, "load_mean": 50000., "load_std": 5000.,
                 "config": {"lookback_hours": 168, "horizon_hours": 24}}, tmp_path / "model.joblib")
    client = TestClient(create_app(tmp_path))
    response = client.post("/forecast", json={
        "last_168_hourly_load_mw": [50000.0] * 168,
        "origin_utc": "2020-01-08T00:00:00Z",
    })
    assert response.status_code == 200
    body = response.json()
    assert len(body["forecast_mw"]) == 24
    assert body["selected_model"] == "ridge"
