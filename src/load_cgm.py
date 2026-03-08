"""
CMM upstream station loader — cruesgrandmontreal.ca API.

Fetches hourly level (m) and flow (m³/s) for three upstream stations on the
Rivière des Prairies / Lac des Deux Montagnes system, all upstream of 043301:

  39_RDP09  Rue Marceau, Pierrefonds-Roxboro      (~0.8 km upstream)
  01_RDP11  Parc Terrasse-Sacré-Cœur, Île-Bizard  (~3.5 km upstream)
  11_LDM01  Parc Philippe-Lavallée, Oka            (~22 km upstream, Ottawa River inflow)

Source: GET https://www.cruesgrandmontreal.ca/api/stationsGeoJSON
  - forcast array: indices 0–239  = 240 hourly history points (60-min intervals)
  -                indices 242–313 = 72 hourly forecast points
  - Updated every ~6 hours by CMM; no authentication required

Public API:
    load_cgm_history(cache)   -> pd.DataFrame  daily means, indexed by date
    load_cgm_forecast(n_days) -> pd.DataFrame  daily mean forecast, indexed by date
"""

import numpy as np
import pandas as pd
import requests
from pathlib import Path

DATA_DIR   = Path(__file__).parent.parent / "data"
CACHE_PATH = DATA_DIR / "cgm_daily.parquet"
CGM_URL    = "https://www.cruesgrandmontreal.ca/api/stationsGeoJSON"

# station_id -> (level_col, flow_col)
STATIONS = {
    "39_RDP09": ("rdp09_level_m", "rdp09_flow_m3s"),
    "01_RDP11": ("rdp11_level_m", "rdp11_flow_m3s"),
    "11_LDM01": ("ldm01_level_m", "ldm01_flow_m3s"),
}

# All CGM column names in order
CGM_COLS = [col for pair in STATIONS.values() for col in pair]


def _fetch_raw() -> dict:
    r = requests.get(CGM_URL, timeout=30)
    r.raise_for_status()
    return {f["properties"]["id"]: f for f in r.json()["features"]}


def _station_hourly(feat: dict, slice_start: int, slice_end: int) -> pd.DataFrame:
    """Extract hourly level + flow for one station from the forcast array slice."""
    t0    = pd.Timestamp(feat["niveau"]["t0_history"])
    level = np.array(feat["niveau"]["forcast"][slice_start:slice_end], dtype=float)
    flow  = np.array(feat["debit"]["forcast"][slice_start:slice_end],  dtype=float)
    level[level == -99999] = np.nan
    flow[flow   == -99999] = np.nan
    idx = pd.date_range(t0 + pd.Timedelta(hours=slice_start), periods=len(level), freq="h")
    return pd.DataFrame({"level_m": level, "flow_m3s": flow}, index=idx)


def _build_daily(raw: dict, slice_start: int, slice_end: int) -> pd.DataFrame:
    """Combine all three stations into a daily-mean DataFrame (tz-naive UTC dates)."""
    dfs = []
    for sid, (level_col, flow_col) in STATIONS.items():
        hourly = _station_hourly(raw[sid], slice_start, slice_end)
        daily  = hourly.resample("D").mean()
        daily.columns = [level_col, flow_col]
        dfs.append(daily)
    merged = pd.concat(dfs, axis=1)
    merged.index = merged.index.tz_localize(None)  # strip tz to match other data sources
    merged.index.name = "date"
    return merged


def load_cgm_history(cache: bool = True) -> pd.DataFrame:
    """
    Load daily CGM history for all three upstream stations.

    When cache=True and cache exists: return cached data.
    When cache=False or no cache: fetch API, extract the last 10 days of
    hourly history, resample to daily means, merge with existing cache.

    Returns DataFrame indexed by date with columns:
      rdp09_level_m, rdp09_flow_m3s,
      rdp11_level_m, rdp11_flow_m3s,
      ldm01_level_m, ldm01_flow_m3s
    """
    if cache and CACHE_PATH.exists():
        print(f"Loading CGM from cache: {CACHE_PATH}")
        df = pd.read_parquet(CACHE_PATH)
        print(f"Loaded {len(df)} CGM days from {df.index.min().date()} to {df.index.max().date()}")
        return df

    print("Fetching CGM upstream station data...")
    raw      = _fetch_raw()
    new_data = _build_daily(raw, slice_start=0, slice_end=240)

    if CACHE_PATH.exists():
        existing = pd.read_parquet(CACHE_PATH)
        combined = pd.concat([existing, new_data])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_data

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(CACHE_PATH)
    print(f"CGM cache saved: {len(combined)} days from "
          f"{combined.index.min().date()} to {combined.index.max().date()}")
    return combined


def load_cgm_forecast(n_days: int = 5) -> pd.DataFrame:
    """
    Fetch daily mean CGM upstream forecasts for the next n_days.

    Returns DataFrame indexed by forecast date with same columns as
    load_cgm_history().
    """
    raw    = _fetch_raw()
    daily  = _build_daily(raw, slice_start=242, slice_end=314)
    result = daily.head(n_days)
    print(f"Loaded {len(result)}-day CGM forecast: "
          f"{result.index[0].date()} → {result.index[-1].date()}")
    return result


if __name__ == "__main__":
    hist = load_cgm_history(cache=False)
    print(hist.tail(5).to_string())
    print()
    fc = load_cgm_forecast()
    print(fc.to_string())
