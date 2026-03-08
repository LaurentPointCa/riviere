"""
Weather forecast loader — basin-mean daily forecast from Open-Meteo.

Fetches the N-day ahead forecast for the same basin grid used by load_climate.py,
using the same variables (temperature, precipitation, rain, snowfall).

No API key required (Open-Meteo free tier).
"""

import requests
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from load_climate import fetch_basin_boundary, generate_grid_points, CLIMATE_VARIABLES

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def load_weather_forecast(days: int = 5, spacing_km: float = 50.0) -> pd.DataFrame:
    """
    Fetch basin-mean daily weather forecast from Open-Meteo.

    Parameters
    ----------
    days : number of forecast days to fetch (1–16)
    spacing_km : grid spacing for basin averaging (should match load_climate)

    Returns
    -------
    DataFrame indexed by date with columns:
      temperature_2m_mean, precipitation_sum, snowfall_sum, rain_sum
    Rows cover today + the next `days` days.
    """
    geojson = fetch_basin_boundary(cache=True)
    points = generate_grid_points(geojson, spacing_km=spacing_km)

    lats = ",".join(str(lat) for lat, _ in points)
    lons = ",".join(str(lon) for _, lon in points)

    params = [
        ("latitude",      lats),
        ("longitude",     lons),
        ("forecast_days", days + 1),   # +1 to include today
        ("timezone",      "America/Toronto"),
    ] + [("daily", v) for v in CLIMATE_VARIABLES]

    response = requests.get(FORECAST_URL, params=params, timeout=30)
    response.raise_for_status()

    location_responses = response.json()

    all_dfs = []
    for loc in location_responses:
        df = pd.DataFrame(loc["daily"])
        df["date"] = pd.to_datetime(df["time"])
        df = df.drop(columns="time").set_index("date")
        all_dfs.append(df)

    basin_mean = pd.concat(all_dfs).groupby(level=0).mean()
    basin_mean.index.name = "date"
    print(f"Loaded {len(basin_mean)}-day weather forecast: "
          f"{basin_mean.index[0].date()} → {basin_mean.index[-1].date()}")
    return basin_mean


if __name__ == "__main__":
    fc = load_weather_forecast(days=5)
    print(fc.to_string())
