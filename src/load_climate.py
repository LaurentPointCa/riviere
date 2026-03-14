"""
Climate data loader — basin-mean daily temperature, precipitation, and snow.

Approach:
  1. Download the basin boundary polygon from mghydro.com (GeoJSON).
  2. Generate a regular grid of points inside the polygon at ~50 km spacing.
  3. Query the Open-Meteo historical archive API (ERA5) for all points in one request.
  4. Average all points to produce basin-mean daily values.
  5. Cache result to data/climate_daily.parquet.

Watershed:
  mghydro.com watershed ID M72047806, area ≈ 148 202 km².
  Direct download: https://mghydro.com/app/download_watershed?format=json&wid=M72047806
  Full report:     https://mghydro.com/app/report?lat=45.454&lng=-74.106&precision=low&simplify=true

Variables (daily):
  temperature_2m_mean   °C    — basin-mean air temperature
  precipitation_sum     mm    — total water input (rain + snow water equiv.)
  snowfall_sum          cm    — daily snowfall
  rain_sum              mm    — liquid precipitation only
  snow_depth            m     — basin-mean snow water equivalent (ERA5 land surface model)
"""

import json
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from shapely.geometry import shape, Point


DATA_DIR = Path(__file__).parent.parent / "data"

BASIN_URL = (
    "https://mghydro.com/app/download_watershed"
    "?format=json&wid=M72047806"
)

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

CLIMATE_VARIABLES = [
    "temperature_2m_mean",  # °C   — drives snowmelt
    "precipitation_sum",    # mm   — total water input (rain + snow water equiv.)
    "snowfall_sum",         # cm   — daily snowfall
    "rain_sum",             # mm   — liquid precipitation only
]

# snow_depth is only available as hourly in ERA5-Land; fetched separately and
# aggregated to daily mean via fetch_snow_depth_daily() / load_snow_depth().
# We use a single representative point (basin centroid) to keep it one API call.
SNOW_DEPTH_CACHE = DATA_DIR / "snow_depth_daily.parquet"
SNOW_DEPTH_LAT  = 46.8   # basin centroid (approximate)
SNOW_DEPTH_LON  = -77.4

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


def fetch_snow_depth_daily(start_date: str, end_date: str) -> pd.Series:
    """
    Fetch hourly snow_depth from ERA5-Land at the basin centroid,
    aggregate to a daily mean Series.

    ERA5-Land provides snow_depth (m, water equivalent) as hourly only.
    Using a single representative point (basin centroid) keeps this one API call.
    Snowpack is spatially smooth at the watershed scale so a centroid is sufficient.
    """
    params = [
        ("latitude",   SNOW_DEPTH_LAT),
        ("longitude",  SNOW_DEPTH_LON),
        ("start_date", start_date),
        ("end_date",   end_date),
        ("timezone",   "America/Toronto"),
        ("hourly",     "snow_depth"),
        ("models",     "era5_land"),
    ]

    for attempt in range(5):
        response = requests.get(OPEN_METEO_URL, params=params, timeout=300)
        if response.status_code == 429:
            wait = 30 * (2 ** attempt)
            print(f"  Rate limited; retrying in {wait}s...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        break

    data = response.json()
    df = pd.DataFrame({"time": data["hourly"]["time"],
                       "snow_depth": data["hourly"]["snow_depth"]})
    df["date"] = pd.to_datetime(df["time"]).dt.normalize()
    daily = df.groupby("date")["snow_depth"].mean()
    daily.index.name = "date"
    return daily.rename("snow_depth")


def load_snow_depth(cache: bool = True) -> pd.Series:
    """
    Load daily snow depth (m SWE) from ERA5-Land at the basin centroid.

    Fetches the full history in a single API call (one point, hourly → daily mean)
    and caches to data/snow_depth_daily.parquet.
    When cache=False, performs an incremental update from the last cached date.
    """
    if cache and SNOW_DEPTH_CACHE.exists():
        print(f"Loading snow depth from cache: {SNOW_DEPTH_CACHE}")
        return pd.read_parquet(SNOW_DEPTH_CACHE)["snow_depth"]

    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    if not cache and SNOW_DEPTH_CACHE.exists():
        existing = pd.read_parquet(SNOW_DEPTH_CACHE)["snow_depth"]
        start = (existing.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if start > today:
            print(f"Snow depth cache already up to date ({existing.index.max().date()}).")
            return existing
        print(f"Fetching ERA5-Land hourly snow_depth ({start} → {today})...")
        new = fetch_snow_depth_daily(start, today)
        result = pd.concat([existing, new])
        result = result[~result.index.duplicated(keep="last")].sort_index()
    else:
        print(f"Fetching ERA5-Land hourly snow_depth (1940 → {today}), single centroid point...")
        result = fetch_snow_depth_daily(CLIMATE_START, today)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    result.to_frame().to_parquet(SNOW_DEPTH_CACHE)
    print(f"Saved snow depth to: {SNOW_DEPTH_CACHE}")
    return result


def load_climate(cache: bool = True, spacing_km: float = 50.0) -> pd.DataFrame:
    """
    Load basin-mean daily climate data.

    Fetches ERA5 reanalysis for all grid points within the basin in a single
    API call, averages them to produce a basin-mean time series, and caches
    the result to data/climate_daily.parquet.

    When cache=True and a cache exists, returns the cached data.
    When cache=False and a cache exists, performs an incremental update:
      only fetches dates newer than the last cached date, then appends.
    When no cache exists, performs a full download from CLIMATE_START.

    Returns a DataFrame indexed by date with columns:
      temperature_2m_mean, precipitation_sum, snowfall_sum, rain_sum, snow_depth
    where snow_depth is basin-mean SWE (m) from ERA5-Land (hourly, aggregated to daily).
    """
    cache_path = DATA_DIR / "climate_daily.parquet"

    if cache and cache_path.exists():
        print(f"Loading climate from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
        print(f"Loaded {len(df):,} rows from {df.index.min().date()} to {df.index.max().date()}")
        # Back-fill snow_depth if an older cache is missing it
        if "snow_depth" not in df.columns:
            try:
                snow = load_snow_depth(cache=True)
                df = df.join(snow, how="left")
            except Exception as e:
                print(f"Warning: could not load snow depth ({e}); snow_depth will be NaN.")
                df["snow_depth"] = float("nan")
        return df

    geojson = fetch_basin_boundary(cache=True)
    points = generate_grid_points(geojson, spacing_km=spacing_km)

    if not cache and cache_path.exists():
        # Incremental update: only fetch missing days
        existing = pd.read_parquet(cache_path)
        last_date = existing.index.max()
        start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        today = pd.Timestamp.today().strftime("%Y-%m-%d")

        if start_date > today:
            print(f"Climate cache already up to date ({last_date.date()}).")
            if "snow_depth" not in existing.columns:
                snow = load_snow_depth(cache=False)
                existing = existing.join(snow, how="left")
            return existing

        print(f"Fetching ERA5 for {len(points)} grid points ({start_date} → {today})...")
        new_data = fetch_climate_all_points(points, start_date=start_date, end_date=today)
        new_data.index.name = "date"
        climate = pd.concat([existing, new_data])
        climate = climate[~climate.index.duplicated(keep="last")].sort_index()
    else:
        print(f"Fetching ERA5 for {len(points)} grid points in one request...")
        climate = fetch_climate_all_points(points)
        climate.index.name = "date"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    climate.to_parquet(cache_path)
    print(f"Saved to: {cache_path}")
    print(f"Loaded {len(climate):,} rows from {climate.index.min().date()} to {climate.index.max().date()}")

    # Join ERA5-Land snow depth (fetched separately as hourly and aggregated)
    try:
        snow = load_snow_depth(cache=cache)
        climate = climate.join(snow, how="left")
    except Exception as e:
        print(f"Warning: could not load snow depth ({e}); snow_depth will be NaN.")
        climate["snow_depth"] = float("nan")
    return climate


if __name__ == "__main__":
    df = load_climate()
    print(df.tail(5))
    print(df.describe().round(2))
    print(f"\nMissing values:\n{df.isnull().sum()}")
