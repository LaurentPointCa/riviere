"""
Climate data loader — basin-mean daily temperature, precipitation, and snow.

Approach:
  1. Download the basin boundary polygon from mghydro.com (GeoJSON).
  2. Generate a regular grid of points inside the polygon at ~50 km spacing.
  3. Query the Open-Meteo historical archive API (ERA5) for all points in one request.
  4. Average all points to produce basin-mean daily values.
  5. Cache result to data/climate_daily.parquet.

Variables (daily):
  temperature_2m_mean   °C    — basin-mean air temperature
  precipitation_sum     mm    — total water input (rain + snow water equiv.)
  snowfall_sum          cm    — daily snowfall; used to build snowpack proxy
  rain_sum              mm    — liquid precipitation only
"""

import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from shapely.geometry import shape, Point


DATA_DIR = Path(__file__).parent.parent / "data"

BASIN_URL = (
    "https://mghydro.com/app/watershed_api"
    "?lat=45.426&lng=-75.926&precision=low&simplify=true"
)

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

CLIMATE_VARIABLES = [
    "temperature_2m_mean",  # °C   — drives snowmelt
    "precipitation_sum",    # mm   — total water input (rain + snow water equiv.)
    "snowfall_sum",         # cm   — daily snowfall; used to build snowpack proxy
    "rain_sum",             # mm   — liquid precipitation only
]

# ERA5 starts 1940-01-01
CLIMATE_START = "1940-01-01"


def fetch_basin_boundary(cache: bool = True) -> dict:
    """Download and cache the basin boundary as a GeoJSON FeatureCollection."""
    cache_path = DATA_DIR / "basin_boundary.geojson"

    if cache and cache_path.exists():
        print(f"Loading basin boundary from cache: {cache_path}")
        return json.loads(cache_path.read_text())

    print("Downloading basin boundary from mghydro.com...")
    response = requests.get(BASIN_URL, timeout=30)
    response.raise_for_status()
    geojson = response.json()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(geojson, indent=2))
    print(f"Cached to: {cache_path}")
    return geojson


def generate_grid_points(geojson: dict, spacing_km: float = 50.0) -> list[tuple]:
    """
    Generate a regular lat/lon grid of points inside the basin polygon.

    Args:
        geojson: GeoJSON FeatureCollection with one Polygon feature
        spacing_km: approximate grid spacing in kilometres

    Returns:
        List of (lat, lon) tuples
    """
    polygon = shape(geojson["features"][0]["geometry"])
    minx, miny, maxx, maxy = polygon.bounds
    mid_lat = (miny + maxy) / 2.0

    lat_step = spacing_km / 111.0
    lon_step = spacing_km / (111.0 * np.cos(np.radians(mid_lat)))

    points = []
    lat = miny
    while lat <= maxy:
        lon = minx
        while lon <= maxx:
            if polygon.contains(Point(lon, lat)):
                points.append((round(lat, 4), round(lon, 4)))
            lon += lon_step
        lat += lat_step

    return points


def fetch_climate_all_points(
    points: list[tuple],
    start_date: str = CLIMATE_START,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Query Open-Meteo historical archive (ERA5) for all grid points in one request.

    Open-Meteo supports multi-location queries via comma-separated lat/lon strings.
    Returns a DataFrame indexed by date with basin-mean values.
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    lats = ",".join(str(lat) for lat, _ in points)
    lons = ",".join(str(lon) for _, lon in points)

    # Each variable must be a separate `daily=` param (comma-joined string causes 400)
    params = [
        ("latitude",   lats),
        ("longitude",  lons),
        ("start_date", start_date),
        ("end_date",   end_date),
        ("timezone",   "America/Toronto"),
    ] + [("daily", v) for v in CLIMATE_VARIABLES]

    response = requests.get(OPEN_METEO_URL, params=params, timeout=120)
    response.raise_for_status()

    # Response is a list — one dict per location
    location_responses = response.json()

    all_dfs = []
    for loc in location_responses:
        df = pd.DataFrame(loc["daily"])
        df["date"] = pd.to_datetime(df["time"])
        df = df.drop(columns="time").set_index("date")
        all_dfs.append(df)

    return pd.concat(all_dfs).groupby(level=0).mean()


def load_climate(cache: bool = True, spacing_km: float = 50.0) -> pd.DataFrame:
    """
    Load basin-mean daily climate data.

    Fetches ERA5 reanalysis for all grid points within the basin in a single
    API call, averages them to produce a basin-mean time series, and caches
    the result to data/climate_daily.parquet.

    Returns a DataFrame indexed by date with columns:
      temperature_2m_mean, precipitation_sum, snowfall_sum, rain_sum
    """
    cache_path = DATA_DIR / "climate_daily.parquet"

    if cache and cache_path.exists():
        print(f"Loading climate from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
        print(f"Loaded {len(df):,} rows from {df.index.min().date()} to {df.index.max().date()}")
        return df

    geojson = fetch_basin_boundary(cache=cache)
    points = generate_grid_points(geojson, spacing_km=spacing_km)
    print(f"Fetching ERA5 for {len(points)} grid points in one request...")

    climate = fetch_climate_all_points(points)
    climate.index.name = "date"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    climate.to_parquet(cache_path)
    print(f"Saved to: {cache_path}")
    print(f"Loaded {len(climate):,} rows from {climate.index.min().date()} to {climate.index.max().date()}")
    return climate


if __name__ == "__main__":
    df = load_climate()
    print(df.tail(5))
    print(df.describe().round(2))
    print(f"\nMissing values:\n{df.isnull().sum()}")
