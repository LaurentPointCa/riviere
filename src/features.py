"""
Feature engineering for station 043301 — Des Prairies daily forecast.

Public API:
    build_dataset() -> tuple[pd.DataFrame, pd.DataFrame]
        Returns (X, y) where:
          X: feature matrix, indexed by date
          y: target matrix with 10 columns:
             flow_t1…flow_t5, level_t1…level_t5
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).parent))

from load_data import load_flow, load_level, load_upstream_level, load_ottawa_flow, load_hull_level
from load_climate import load_climate

DAM_ERA_START = "1978-01-01"

# Lag days applied to hydro and climate variables
LAG_DAYS = [1, 2, 3, 4, 5, 7, 14, 30]

# Variables to lag
LAG_VARS_HYDRO = [
    "flow_m3s", "level_m", "upstream_level_m", "ottawa_flow_m3s", "hull_level_m",
    "temperature_2m_mean", "precipitation_sum", "rain_sum", "snow_depth",
]


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for var in LAG_VARS_HYDRO:
        if var not in df.columns:
            continue
        for n in LAG_DAYS:
            new_cols[f"{var}_lag{n}"] = df[var].shift(n)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}

    # Hydro rolling features
    for window, suffix in [(3, "3d"), (7, "7d"), (14, "14d"), (30, "30d")]:
        new_cols[f"flow_roll_mean_{suffix}"]  = df["flow_m3s"].rolling(window, min_periods=1).mean()
        new_cols[f"level_roll_mean_{suffix}"] = df["level_m"].rolling(window, min_periods=1).mean()

    for window, suffix in [(3, "3d"), (7, "7d"), (14, "14d")]:
        new_cols[f"flow_roll_max_{suffix}"]  = df["flow_m3s"].rolling(window, min_periods=1).max()
        new_cols[f"level_roll_max_{suffix}"] = df["level_m"].rolling(window, min_periods=1).max()

    for window, suffix in [(7, "7d"), (14, "14d")]:
        new_cols[f"flow_roll_std_{suffix}"]  = df["flow_m3s"].rolling(window, min_periods=1).std()
        new_cols[f"level_roll_std_{suffix}"] = df["level_m"].rolling(window, min_periods=1).std()

    # Upstream level rolling features
    for window, suffix in [(3, "3d"), (7, "7d"), (14, "14d"), (30, "30d")]:
        new_cols[f"upstream_level_roll_mean_{suffix}"] = df["upstream_level_m"].rolling(window, min_periods=1).mean()
    for window, suffix in [(3, "3d"), (7, "7d"), (14, "14d")]:
        new_cols[f"upstream_level_roll_max_{suffix}"] = df["upstream_level_m"].rolling(window, min_periods=1).max()

    # Ottawa River (02KF005) rolling features
    if "ottawa_flow_m3s" in df.columns:
        for window, suffix in [(3, "3d"), (7, "7d"), (14, "14d"), (30, "30d")]:
            new_cols[f"ottawa_flow_roll_mean_{suffix}"] = df["ottawa_flow_m3s"].rolling(window, min_periods=1).mean()
        for window, suffix in [(3, "3d"), (7, "7d"), (14, "14d")]:
            new_cols[f"ottawa_flow_roll_max_{suffix}"] = df["ottawa_flow_m3s"].rolling(window, min_periods=1).max()

    # Hull level (02LA015) rolling features
    if "hull_level_m" in df.columns:
        for window, suffix in [(3, "3d"), (7, "7d"), (14, "14d"), (30, "30d")]:
            new_cols[f"hull_level_roll_mean_{suffix}"] = df["hull_level_m"].rolling(window, min_periods=1).mean()
        for window, suffix in [(3, "3d"), (7, "7d"), (14, "14d")]:
            new_cols[f"hull_level_roll_max_{suffix}"] = df["hull_level_m"].rolling(window, min_periods=1).max()

    # Climate rolling features
    for window, suffix in [(7, "7d"), (14, "14d"), (30, "30d")]:
        new_cols[f"temp_roll_mean_{suffix}"]   = df["temperature_2m_mean"].rolling(window, min_periods=1).mean()
        new_cols[f"precip_roll_sum_{suffix}"]  = df["precipitation_sum"].rolling(window, min_periods=1).sum()
        new_cols[f"snow_roll_sum_{suffix}"]    = df["snowfall_sum"].rolling(window, min_periods=1).sum()

    for window, suffix in [(3, "3d"), (7, "7d"), (14, "14d")]:
        new_cols[f"rain_roll_sum_{suffix}"] = df["rain_sum"].rolling(window, min_periods=1).sum()

    new_cols["snow_roll_sum_90d"] = df["snowfall_sum"].rolling(90, min_periods=1).sum()

    # ERA5 snow depth (SWE) rolling features
    if "snow_depth" in df.columns:
        for window, suffix in [(7, "7d"), (14, "14d"), (30, "30d")]:
            new_cols[f"snow_depth_roll_mean_{suffix}"] = df["snow_depth"].rolling(window, min_periods=1).mean()
        new_cols["snow_depth_roll_max_30d"] = df["snow_depth"].rolling(30, min_periods=1).max()

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def _add_snowpack_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Degree-day snowpack model iterating row by row."""
    MELT_FACTOR = 4.0   # mm of melt per °C per day
    SNOW_RATIO = 0.1    # cm snowfall → mm SWE (10:1 ratio)

    snowpack = np.zeros(len(df))
    swe = 0.0
    # Note: swe starts at 0 regardless of season. Rows before the first Oct 1
    # in the dataset will have underestimated snowpack (no prior-year accumulation
    # available). This affects only the first partial snow season (~9 months for
    # the 1978-01-01 dataset start) and is an accepted limitation.

    snowfall = df["snowfall_sum"].values
    temp = df["temperature_2m_mean"].values
    months = df.index.month
    days = df.index.day

    for i in range(len(df)):
        # Reset on October 1 (start of new snow season)
        if months[i] == 10 and days[i] == 1:
            swe = 0.0

        # Accumulation
        sf = snowfall[i]
        if not np.isnan(sf):
            swe += sf * SNOW_RATIO

        # Melt
        t = temp[i]
        if not np.isnan(t) and t > 0:
            swe -= t * MELT_FACTOR

        swe = max(0.0, swe)
        snowpack[i] = swe

    return pd.concat([df, pd.DataFrame({"snowpack_proxy_mm": snowpack}, index=df.index)], axis=1)


def _add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    doy = df.index.day_of_year.astype(float)
    new_cols = {
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "month":   df.index.month,
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def _add_flow_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """Flow deviation from long-term seasonal median by day-of-year."""
    doy = df.index.day_of_year
    seasonal_median = df.groupby(doy)["flow_m3s"].transform("median")
    new_cols = {"flow_anom": df["flow_m3s"] - seasonal_median}
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def _add_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 5-day weather forecast features.

    During training, ERA5 future observed values serve as a perfect-forecast proxy
    (e.g. temp_forecast_t1 = tomorrow's actual temperature). At inference time,
    predict.py replaces these columns with real Open-Meteo forecast values.

    LightGBM handles the NaNs in the last 5 rows of training data natively.
    """
    new_cols = {}
    for h in range(1, 6):
        new_cols[f"temp_forecast_t{h}"]   = df["temperature_2m_mean"].shift(-h)
        new_cols[f"precip_forecast_t{h}"] = df["precipitation_sum"].shift(-h)
        new_cols[f"rain_forecast_t{h}"]   = df["rain_sum"].shift(-h)
        new_cols[f"snow_forecast_t{h}"]   = df["snowfall_sum"].shift(-h)

    # Derived aggregates over the full 5-day window
    new_cols["precip_forecast_sum_5d"] = sum(
        new_cols[f"precip_forecast_t{h}"] for h in range(1, 6)
    )
    new_cols["temp_forecast_mean_5d"] = sum(
        new_cols[f"temp_forecast_t{h}"] for h in range(1, 6)
    ) / 5

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for h in range(1, 6):
        new_cols[f"flow_t{h}"]  = df["flow_m3s"].shift(-h)
        new_cols[f"level_t{h}"] = df["level_m"].shift(-h)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def build_dataset(drop_incomplete: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build feature matrix X and target matrix y for the Des Prairies forecast model.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix indexed by date (~17,500 rows, ~70 columns).
    y : pd.DataFrame
        Target matrix with columns flow_t1…flow_t5, level_t1…level_t5.
    """
    # ── 1. Load and merge data sources ─────────────────────────────────────
    flow = load_flow()[["flow_m3s"]]
    level = load_level()[["level_m"]]
    upstream = load_upstream_level()[["upstream_level_m"]]
    climate = load_climate()
    ottawa = load_ottawa_flow()[["ottawa_flow_m3s"]]
    hull   = load_hull_level()[["hull_level_m"]]

    df = (flow
          .join(level,    how="outer")
          .join(upstream, how="outer")
          .join(ottawa,   how="outer")
          .join(hull,     how="outer")
          .join(climate,  how="outer"))
    df.index.name = "date"
    df = df.sort_index()

    # Fill the one known missing date (2024-04-24) via forward-fill
    df = df.ffill()

    # Restrict to post-dam era
    df = df.loc[DAM_ERA_START:]

    # ── 2. Feature engineering ──────────────────────────────────────────────
    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    df = _add_snowpack_proxy(df)
    df = _add_seasonal_features(df)
    df = _add_flow_anomaly(df)
    df = _add_forecast_features(df)

    # ── 3. Build targets ────────────────────────────────────────────────────
    df = _add_targets(df)

    # ── 4. Split X / y and drop rows with missing targets ──────────────────
    target_cols = [f"flow_t{h}" for h in range(1, 6)] + [f"level_t{h}" for h in range(1, 6)]
    feature_cols = [c for c in df.columns if c not in target_cols]

    y = df[target_cols].copy()
    X = df[feature_cols].copy()

    # Drop rows where any target is NaN (last 5 rows due to forward shift)
    if drop_incomplete:
        valid = y.notna().all(axis=1)
        X = X.loc[valid]
        y = y.loc[valid]

    return X, y


if __name__ == "__main__":
    X, y = build_dataset()

    print("\n── X shape:", X.shape)
    print("── X columns:")
    for col in X.columns:
        print(f"   {col}")

    print("\n── y shape:", y.shape)
    print("── y columns:", list(y.columns))

    print("\n── NaN counts in y:")
    print(y.isnull().sum().to_dict())

    print("\n── NaN counts in X (top 20):")
    nan_counts = X.isnull().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
    print(nan_counts.head(20).to_dict())

    print("\n── X first 3 rows:")
    print(X.head(3).to_string())

    print("\n── X last 3 rows:")
    print(X.tail(3).to_string())
