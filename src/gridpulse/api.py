"""Local demonstration API for 24-hour-ahead load forecasts."""
from pathlib import Path

import joblib


def create_app(artifact_dir="artifacts"):
    from fastapi import FastAPI, HTTPException
    app = FastAPI(title="GridPulse Forecast API", version="0.1.0")
    root = Path(artifact_dir)
    bundle_path = root / "model.joblib"
    bundle = joblib.load(bundle_path) if bundle_path.exists() else None
    tcn_bundle = None
    if (root / "tcn.pt").exists():
        import torch
        tcn_bundle = torch.load(root / "tcn.pt", map_location="cpu", weights_only=False)

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": bundle is not None or tcn_bundle is not None}

    @app.post("/forecast")
    def forecast(payload: dict):
        history = payload.get("last_168_hourly_load_mw")
        origin = payload.get("origin_utc")
        if not isinstance(history, list) or len(history) != 168 or not origin:
            raise HTTPException(status_code=422, detail="Provide 168 load values and origin_utc")
        import numpy as np
        import pandas as pd
        from .data import calendar_matrix
        values = np.asarray(history, dtype="float32")
        if not np.isfinite(values).all():
            raise HTTPException(status_code=422, detail="History must contain finite values")
        origin_dt = pd.Timestamp(origin)
        if origin_dt.tzinfo is None:
            raise HTTPException(status_code=422, detail="origin_utc must include a timezone")
        origin_dt = origin_dt.tz_convert("UTC")
        past_idx = pd.date_range(end=origin_dt, periods=168, freq="h")
        future_idx = pd.date_range(start=origin_dt + pd.Timedelta(hours=1), periods=24, freq="h")
        x = np.column_stack([values, calendar_matrix(past_idx)])[None, :, :]
        cal = calendar_matrix(future_idx)[None, :, :]
        selected = None
        if tcn_bundle is not None:
            import torch
            from .models import _tcn_classes
            cfg = tcn_bundle["config"]
            TCN = _tcn_classes()
            model = TCN(x.shape[-1], int(cfg["hidden_channels"]), int(cfg["kernel_size"]), float(cfg["dropout"]), 24)
            model.load_state_dict(tcn_bundle["state_dict"])
            model.eval()
            xn = x.copy(); xn[:, :, 0] = (xn[:, :, 0] - float(tcn_bundle["load_mean"])) / float(tcn_bundle["load_std"])
            with torch.no_grad():
                pred = model(torch.tensor(xn, dtype=torch.float32)).numpy()[0] * float(tcn_bundle["load_std"]) + float(tcn_bundle["load_mean"])
            selected = "tcn"
        elif bundle is not None:
            selected = bundle["selected_model"]
            if selected == "ridge":
                from .models import predict_ridge
                xn = x.copy(); xn[:, :, 0] = (xn[:, :, 0] - float(bundle["load_mean"])) / float(bundle["load_std"])
                yn = predict_ridge(bundle["model"], xn, cal)
                pred = yn[0] * float(bundle["load_std"]) + float(bundle["load_mean"])
            else:
                pred = values[-168:-144]
        else:
            raise HTTPException(status_code=503, detail="Train a model first")
        return {"origin_utc": origin_dt.isoformat(), "horizon_hours": 24,
                "forecast_mw": [float(v) for v in pred], "selected_model": selected,
                "research_prototype": True}

    return app


app = create_app()
