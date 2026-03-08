# Rivière des Prairies — 5-Day Forecast

Daily flow (m³/s) and water level (m) forecast for CEHQ station **043301**
(Rivière des Prairies at Laval), using a LightGBM model trained on 45+ years
of hydrological and climate data, with real-time weather and upstream hydrological
forecast injection.

![5-day forecast chart](docs/forecast.png)

Latest forecast also available as machine-readable JSON: [`docs/forecast.json`](docs/forecast.json)

## Watershed and stations

![Basin map](docs/basin_map.png)

The watershed is delineated by [mghydro.com](https://mghydro.com) from the outlet point near Ottawa. ERA5 climate data and the Open-Meteo forecast are averaged across this full basin.

![Monitoring stations](docs/basin_stations.png)

The five monitoring stations used as model inputs in the Montréal–Laval area.

## Results

Held-out test set (2024-03-04 → 2026-03-03, 730 days) used for evaluation.
The deployed model is retrained on the full dataset (1978-01-01 → 2026-03-03).

| Horizon | Flow RMSE (m³/s) | Level RMSE (m) | Skill vs. persistence |
|---------|-----------------|----------------|-----------------------|
| t+1     | 38.2            | 0.057          | +22%                  |
| t+2     | 64.0            | 0.091          | +23%                  |
| t+3     | 85.5            | 0.114          | +22%                  |
| t+4     | 100.5           | 0.133          | +24%                  |
| t+5     | 109.6           | 0.145          | +28%                  |

## Data sources

| Source | Variables | Period |
|--------|-----------|--------|
| [CEHQ](https://www.cehq.gouv.qc.ca) | Flow (m³/s), Level (m) — station 043301 | 1922–present |
| [CEHQ](https://www.cehq.gouv.qc.ca) | Upstream level (m) — station 043108 (Lac des Deux Montagnes) | 1986–present |
| [Open-Meteo ERA5](https://open-meteo.com) | Temperature, precipitation, snowfall, rain (observed) | 1940–present |
| [Open-Meteo Forecast](https://open-meteo.com) | 5-day weather forecast (temperature, precipitation, rain, snow) | real-time |
| [Crues Grand Montréal](https://www.cruesgrandmontreal.ca) | Level (m) + flow (m³/s) — upstream stations 39_RDP09, 01_RDP11, 11_LDM01 | real-time + rolling history |
| [mghydro.com](https://mghydro.com/app/report?lat=45.454&lng=-74.106&precision=low&simplify=true) | Basin boundary polygon (GeoJSON) — ID M72047806, ~148,202 km² | static |

### Crues Grand Montréal upstream stations

| Station | Location | Distance upstream | Max flow ref. |
|---------|----------|-------------------|---------------|
| 39_RDP09 | Rue Marceau, Pierrefonds-Roxboro | ~0.8 km | 3,172 m³/s |
| 01_RDP11 | Parc Terrasse-Sacré-Cœur, Île-Bizard | ~3.5 km | 3,172 m³/s |
| 11_LDM01 | Parc Philippe-Lavallée, Oka | ~22 km (Lac des Deux Montagnes) | 11,340 m³/s |

## Pipeline

```
load_data.py      CEHQ historical + live feed (stations 043301 + upstream 043108)
load_climate.py   basin boundary (mghydro.com) → ERA5 basin-mean daily climate (Open-Meteo)
load_forecast.py  5-day weather forecast (Open-Meteo) → injected at inference time
load_cgm.py       hourly level + flow for 3 upstream stations (cruesgrandmontreal.ca)
                  → daily cache (data/cgm_daily.parquet) + 5-day forecast
     │
     ▼
features.py       build_dataset() → (X, y)   [222 columns]
                  • Lags 1–30 days (flow, level, upstream 043108 + 3 CGM stations, climate)
                  • Rolling mean/max/std (3–30 days)
                  • Snowpack proxy (degree-day model)
                  • Seasonal encoding (sin/cos DOY)
                  • Flow anomaly vs seasonal median
                  • Weather forecast t+1…t+5 (ERA5 perfect-forecast proxy at
                    training time, real Open-Meteo forecast at inference time)
                  • CGM upstream forecast t+1…t+5 (observed proxy at training time,
                    real CMM forecast at inference time)
     │
     ▼
model.py          10 × LGBMRegressor (one per horizon)
                  Evaluated on 2024–2026, deployed on full 1978–2026
     │
     ▼
predict.py        5-day forecast CLI → docs/forecast.png + docs/forecast_30d.png
                  + docs/forecast.json
```

## Usage

```bash
# Set up environment
python -m venv .venv
source .venv/bin/activate
pip install lightgbm scikit-learn pandas numpy requests shapely pyarrow
brew install libomp   # macOS only

# Build features and train (downloads data on first run)
python src/model.py

# Forecast from latest available date
python src/predict.py

# Forecast from a specific past date (shows observed vs predicted)
python src/predict.py --date 2025-06-01
```

## Model details

- **Strategy:** direct multi-output — one `LGBMRegressor` per horizon (t+1…t+5),
  separately for flow and level
- **Training period:** 1978-01-01 onward (post-dam era)
- **Features:** 222 columns — lags, rolling statistics (upstream station 043108 +
  3 CGM upstream stations), snowpack proxy, seasonal encoding, flow anomaly,
  5-day weather forecast, 5-day upstream hydrological forecast
- **Hyperparameters:** 500 trees, lr=0.05, 63 leaves, subsample=0.8
- **Top features:** current flow, 3-day rolling max flow, current level,
  day-of-year (sin), 5-day cumulative forecast precipitation, 30-day mean temperature
