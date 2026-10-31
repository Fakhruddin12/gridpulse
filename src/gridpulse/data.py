from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(output_dir: str | Path, start_year: int, end_year: int) -> list[Path]:
    """Download annual NESO historic demand CSV files from its public CKAN API."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    api = "https://api.neso.energy/api/3/action/datapackage_show?id=historic-demand-data"
    with urlopen(Request(api, headers={"User-Agent": "GridPulse-research/0.1"}), timeout=60) as response:
        package = json.load(response)
    if not package.get("success"):
        raise RuntimeError("NESO CKAN API did not return a successful package response")
    resources = package["result"]["resources"]
    by_year = {}
    for item in resources:
        if str(item.get("format", "")).upper() != "CSV":
            continue
        title = item.get("title", "")
        digits = "".join(ch for ch in title if ch.isdigit())
        if len(digits) == 4 and start_year <= int(digits) <= end_year:
            by_year[int(digits)] = item
    missing = sorted(set(range(start_year, end_year + 1)) - set(by_year))
    if missing:
        raise ValueError(f"NESO package has no annual CSV resources for years: {missing}")
    jobs = []
    for year, resource in sorted(by_year.items()):
        url = resource.get("path", "")
        host = urlparse(url).hostname or ""
        if host not in {"api.neso.energy", "www.neso.energy"}:
            raise ValueError(f"Unexpected NESO resource host: {host}")
        dest = out / f"historic_demand_data_{year}.csv"
        jobs.append((year, resource, url, dest))

    def fetch(job):
        year, resource, url, dest = job
        existing_rows = 0
        if dest.exists() and dest.stat().st_size >= 100_000:
            with dest.open("rb") as old:
                existing_rows = sum(chunk.count(b"\n") for chunk in iter(lambda: old.read(1024 * 1024), b""))
        if existing_rows >= 10_000:
            return dest, {"year": year, "title": resource.get("title"), "resource_id": resource.get("id"),
                          "modified": resource.get("metadata_modified"), "url": url, "sha256": sha256(dest)}
        part = dest.with_suffix(".csv.part")
        request = Request(url, headers={"User-Agent": "GridPulse-research/0.1"})
        try:
            with urlopen(request, timeout=180) as response, part.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            if part.stat().st_size < 100_000:
                raise ValueError(f"Downloaded NESO file is unexpectedly small: {year}")
            with part.open("rb") as downloaded:
                row_count = sum(chunk.count(b"\n") for chunk in iter(lambda: downloaded.read(1024 * 1024), b""))
            # Complete years have ~17.5k half-hourly rows; current-year data should
            # still be well above this floor by the time the portfolio run is made.
            if row_count < 10_000:
                raise ValueError(f"NESO CSV appears truncated ({row_count} lines): {year}")
            part.replace(dest)
        except Exception:
            part.unlink(missing_ok=True)
            raise
        return dest, {"year": year, "title": resource.get("title"), "resource_id": resource.get("id"),
                      "modified": resource.get("metadata_modified"), "url": url, "sha256": sha256(dest)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, jobs))
    saved = [item[0] for item in results]
    metadata = [item[1] for item in results]
    (out / "source_metadata.json").write_text(json.dumps({"package": "historic-demand-data", "downloaded": metadata}, indent=2))
    return saved


def load_country_series(
    source_dir: str | Path, timestamp_column: str, target_column: str, indicator_column: str
) -> pd.Series:
    """Read actual half-hourly NESO outturns and resample them to UTC hourly means."""
    paths = sorted(Path(source_dir).glob("historic_demand_data_*.csv"))
    paths = [p for p in paths if p.name != "source_metadata.json"]
    if not paths:
        raise FileNotFoundError(f"No historic_demand_data_YYYY.csv files found under {source_dir}")
    rows = []
    for path in paths:
        header = pd.read_csv(path, nrows=0).columns
        required = {timestamp_column, "SETTLEMENT_PERIOD", target_column}
        missing = required - set(header)
        if missing:
            raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")
        usecols = list(required | ({indicator_column} if indicator_column in header else set()))
        frame = pd.read_csv(path, usecols=usecols)
        # Earlier annual files contain final outturns only; the current-year file
        # explicitly distinguishes forecasts and actuals.
        if indicator_column in frame.columns:
            frame = frame.loc[frame[indicator_column].astype(str).str.upper().eq("A")].copy()
        dates = pd.to_datetime(frame[timestamp_column], format="mixed", dayfirst=True, errors="coerce")
        periods = pd.to_numeric(frame["SETTLEMENT_PERIOD"], errors="coerce")
        values = pd.to_numeric(frame[target_column], errors="coerce")
        keep = dates.notna() & periods.notna() & values.notna()
        frame = frame.loc[keep].copy()
        local_midnights = pd.DatetimeIndex(dates.loc[keep].dt.normalize()).tz_localize("Europe/London")
        frame["timestamp_utc"] = (local_midnights.tz_convert("UTC") +
            pd.to_timedelta(periods.loc[keep].to_numpy() - 1, unit="m") * 30)
        frame["load_mw"] = values.loc[keep].astype(float).to_numpy()
        rows.append(frame[["timestamp_utc", "load_mw"]])
    frame = pd.concat(rows, ignore_index=True).sort_values("timestamp_utc")
    frame = frame.drop_duplicates("timestamp_utc", keep="last")
    half_hourly = frame.set_index("timestamp_utc")["load_mw"].astype("float64").sort_index()
    if half_hourly.index.tz is None:
        raise ValueError("Expected timezone-aware UTC timestamps")
    series = half_hourly.resample("h").mean()
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="h", tz="UTC")
    series = series.reindex(full_index)
    # Only short internal gaps are interpolated; larger gaps stay missing and are removed
    # from windows rather than smoothed across outages.
    # Fill only entire internal runs of length <= 2. pandas' limit alone
    # would also fill the first two positions of a longer gap.
    missing = series.isna()
    runs = missing.ne(missing.shift(fill_value=False)).cumsum()
    run_lengths = missing.groupby(runs).transform("sum")
    interpolated = series.interpolate(method="time", limit_area="inside")
    short_fill = series.where(~(missing & run_lengths.le(2)), interpolated)
    return short_fill.rename("load_mw")


def quality_report(series: pd.Series) -> dict:
    return {
        "n_hours": int(len(series)),
        "start_utc": str(series.index.min()),
        "end_utc": str(series.index.max()),
        "missing_hours_after_short_gap_fill": int(series.isna().sum()),
        "negative_load_hours": int((series < 0).sum()),
        "duplicate_timestamps": int(series.index.duplicated().sum()),
        "median_mw": float(series.median()),
        "p99_mw": float(series.quantile(0.99)),
    }


def calendar_matrix(index: pd.DatetimeIndex) -> np.ndarray:
    """Deterministic, forecast-time-known calendar features."""
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    doy = index.dayofyear.to_numpy()
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        (dow >= 5).astype(float),
    ]).astype("float32")


def supervised_windows(series: pd.Series, lookback: int, horizon: int, step: int | None = None):
    """Yield forecast-origin windows; reject any input/target window crossing a gap."""
    values = series.to_numpy(dtype="float32")
    cal = calendar_matrix(series.index)
    X, C, Y, origins, target_starts = [], [], [], [], []
    step = int(step or horizon)
    for origin in range(lookback - 1, len(series) - horizon, step):
        left = origin - lookback + 1
        right = origin + 1
        future = slice(right, right + horizon)
        if not np.isfinite(values[left:right]).all() or not np.isfinite(values[future]).all():
            continue
        X.append(np.column_stack([values[left:right], cal[left:right]]))
        C.append(cal[future])
        Y.append(values[future])
        origins.append(series.index[origin])
        target_starts.append(series.index[right])
    return (np.asarray(X), np.asarray(C), np.asarray(Y),
            pd.DatetimeIndex(origins), pd.DatetimeIndex(target_starts))
