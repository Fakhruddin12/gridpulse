from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from .data import download_source, load_country_series, quality_report, sha256, supervised_windows
from .evaluation import regression_metrics, interval_metrics
from .models import fit_ridge, predict_ridge, seasonal_naive, train_tcn, predict_tcn
from .eda import profile


def read_config(path):
    return yaml.safe_load(Path(path).read_text())


def _mkdirs(cfg):
    for p in cfg["paths"].values():
        Path(p).mkdir(parents=True, exist_ok=True)


def _series(cfg):
    raw = Path(cfg["data"]["raw_dir"])
    manifest = raw / "source_metadata.json"
    return load_country_series(raw, cfg["data"]["timestamp_column"], cfg["data"]["target_column"], cfg["data"]["indicator_column"]), manifest


def prepare(cfg):
    _mkdirs(cfg)
    series, raw = _series(cfg)
    out = Path(cfg["data"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    frame = series.rename("load_mw").to_frame()
    frame.to_csv(out / "load.csv", index_label="timestamp_utc")
    report = quality_report(series)
    report.update({"source_manifest_sha256": sha256(raw), "source": "NESO Historic Demand Data", "region": "Great Britain"})
    (out / "quality_report.json").write_text(json.dumps(report, indent=2))
    Path(cfg["paths"]["reports"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["paths"]["reports"]) / "eda_profile.json").write_text(json.dumps(profile(series), indent=2))
    return report


def run_train(cfg):
    _mkdirs(cfg)
    series, raw_path = _series(cfg)
    mcfg, vcfg = cfg["model"], cfg["validation"]
    X, C, Y, origins, starts = supervised_windows(series, int(mcfg["lookback_hours"]), int(mcfg["horizon_hours"]), step=int(mcfg["horizon_hours"]))
    if len(X) < 500:
        raise ValueError(f"Only {len(X)} usable forecast windows; need at least 500")
    cutoff = int(len(X) * (1 - float(vcfg["test_fraction"])))
    dev_idx = np.arange(cutoff)
    test_idx = np.arange(cutoff, len(X))
    if len(dev_idx) < 365:
        raise ValueError("Not enough development history; lower minimum history only with justification")
    # Expanding-window CV: validation blocks are consecutive 24-hour origins and do not
    # overlap each other. All scalers are fitted within each fold.
    n_folds = int(vcfg["folds"])
    block = max(30, int(len(dev_idx) * .10))
    first_val = len(dev_idx) - n_folds * block
    if first_val < 180:
        raise ValueError("Insufficient development history for the requested expanding-window CV")
    fold_results, fold_residuals, tcn_epochs = [], {}, []
    for fold in range(n_folds):
        train_end = first_val + fold * block
        val_end = min(train_end + block, len(dev_idx))
        tr_idx = np.arange(train_end)
        va_idx = np.arange(train_end, val_end)
        if len(va_idx) == 0:
            continue
        mu = float(np.mean(X[tr_idx, :, 0]))
        sd = float(np.std(X[tr_idx, :, 0])) or 1.0
        seasonal_scale = float(np.mean(np.abs(Y[tr_idx] - X[tr_idx, :Y.shape[1], 0])))
        Xn = X.copy()
        Xn[:, :, 0] = (Xn[:, :, 0] - mu) / sd
        Yn = (Y - mu) / sd
        inner_cut = max(1, int(len(tr_idx) * .9))
        inner_tr, inner_va = tr_idx[:inner_cut], tr_idx[inner_cut:]
        if not len(inner_va):
            raise ValueError("Need a trailing validation segment for TCN early stopping")
        ridge = fit_ridge(Xn[tr_idx], C[tr_idx], Yn[tr_idx])
        ridge_p = predict_ridge(ridge, Xn[va_idx], C[va_idx]) * sd + mu
        naive_p = seasonal_naive(X[va_idx], period=int(mcfg["lookback_hours"]), horizon=int(mcfg["horizon_hours"]))
        # Early stopping uses only a trailing segment of the training period.
        tcfg = {**mcfg, "seed": int(vcfg["seed"]) + fold}
        tcn, history, device = train_tcn(Xn[inner_tr], Yn[inner_tr], Xn[inner_va], Yn[inner_va], tcfg)
        tcn_epochs.append(history[-1]["epoch"])
        tcn_p = predict_tcn(tcn, Xn[va_idx], device) * sd + mu
        preds = {"ridge": ridge_p, "seasonal_naive_week": naive_p, "tcn": tcn_p}
        scores = {name: regression_metrics(Y[va_idx], value, seasonal_scale) for name, value in preds.items()}
        fold_results.append({"fold": fold + 1, "train_origins": len(tr_idx), "validation_origins": len(va_idx), "scores": scores})
        for name, value in preds.items():
            fold_residuals.setdefault(name, []).append((Y[va_idx] - value).reshape(-1))
    mean_mae = {name: float(np.mean([f["scores"][name]["mae_mw"] for f in fold_results]) )
                for name in ["ridge", "seasonal_naive_week", "tcn"]}
    selected = min(mean_mae, key=mean_mae.get)
    residuals = np.concatenate(fold_residuals[selected])

    # Final model refit uses all development examples. The chosen TCN epoch count is
    # fixed from fold-local early stopping; the untouched final block is not consulted.
    mu = float(np.mean(X[dev_idx, :, 0]))
    sd = float(np.std(X[dev_idx, :, 0])) or 1.0
    seasonal_scale = float(np.mean(np.abs(Y[dev_idx] - X[dev_idx, :Y.shape[1], 0])))
    Xn = X.copy()
    Xn[:, :, 0] = (Xn[:, :, 0] - mu) / sd
    Yn = (Y - mu) / sd
    if selected == "ridge":
        model = fit_ridge(Xn[dev_idx], C[dev_idx], Yn[dev_idx])
        pred = predict_ridge(model, Xn[test_idx], C[test_idx]) * sd + mu
    elif selected == "tcn":
        final_cfg = {**mcfg, "epochs": max(1, int(np.median(tcn_epochs))), "seed": int(vcfg["seed"])}
        model, history, device = train_tcn(Xn[dev_idx], Yn[dev_idx], None, None, final_cfg)
        pred = predict_tcn(model, Xn[test_idx], device) * sd + mu
    else:
        model = None
        pred = seasonal_naive(X[test_idx], period=int(mcfg["lookback_hours"]), horizon=int(mcfg["horizon_hours"]))
    naive_test = seasonal_naive(X[test_idx], period=int(mcfg["lookback_hours"]), horizon=int(mcfg["horizon_hours"]))
    test_scores = {
        "selected_model": selected,
        "selected": regression_metrics(Y[test_idx], pred, seasonal_scale),
        "seasonal_naive_week": regression_metrics(Y[test_idx], naive_test, seasonal_scale),
        "cv_mean_mae_mw": mean_mae,
        "expanding_window_folds": fold_results,
        "interval": interval_metrics(Y[test_idx], pred, residuals),
    }
    pred_rows = []
    for i, window_i in enumerate(test_idx):
        for h in range(int(mcfg["horizon_hours"])):
            pred_rows.append({"forecast_origin_utc": str(origins[window_i]),
                              "target_time_utc": str(starts[window_i] + pd.Timedelta(hours=h)),
                              "actual_mw": float(Y[window_i, h]), "selected_forecast_mw": float(pred[i, h]),
                              "seasonal_naive_week_mw": float(naive_test[i, h]), "selected_model": selected})
    result = {
        "project": "GridPulse", "region": "Great Britain",
        "source_sha256": sha256(raw_path), "data_start": str(series.index.min()),
        "data_end": str(series.index.max()), "test_start": str(starts[test_idx[0]]),
        "test_end": str(starts[test_idx[-1]]), "lookback_hours": int(mcfg["lookback_hours"]),
        "horizon_hours": int(mcfg["horizon_hours"]), "metrics": test_scores,
    }
    report = Path(cfg["paths"]["reports"])
    (report / "metrics.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(pred_rows).to_csv(report / "holdout_predictions.csv", index=False)
    (report / "run_metadata.json").write_text(json.dumps({"config": cfg, "quality": quality_report(series), "epochs": history}, indent=2))
    artifact = Path(cfg["paths"]["artifacts"])
    artifact.mkdir(parents=True, exist_ok=True)
    if selected == "tcn":
        import torch
        torch.save({"state_dict": model.state_dict(), "load_mean": mu, "load_std": sd, "config": mcfg}, artifact / "tcn.pt")
    elif selected == "ridge":
        joblib.dump({"selected_model": "ridge", "model": model, "load_mean": mu, "load_std": sd, "config": mcfg}, artifact / "model.joblib")
    else:
        joblib.dump({"selected_model": "seasonal_naive_week", "config": mcfg}, artifact / "model.joblib")
    return result


def main():
    parser = argparse.ArgumentParser(prog="gridpulse")
    parser.add_argument("command", choices=["download", "prepare", "train", "report"])
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.command == "download":
        paths = download_source(cfg["data"]["raw_dir"], int(cfg["data"]["start_year"]), int(cfg["data"]["end_year"]))
        print(json.dumps({"files": [{"path": str(p), "sha256": sha256(p)} for p in paths]}, indent=2))
    elif args.command == "prepare":
        print(json.dumps(prepare(cfg), indent=2))
    elif args.command == "report":
        metrics_path = Path(cfg["paths"]["reports"]) / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError("Run the training command before requesting its report")
        print(metrics_path.read_text())
    else:
        print(json.dumps(run_train(cfg), indent=2))


if __name__ == "__main__":
    main()
