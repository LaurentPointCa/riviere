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

The seven monitoring stations used as model inputs, from the Ottawa region to Laval.

## Results

Held-out test set (2024-03-04 → 2026-03-03, 730 days) used for evaluation.
The deployed model is retrained on the full dataset (1978-01-01 → 2026-03-03).

Two separate seasonal models: **cold** (Nov–May, snowmelt/spring freshet) and **warm** (Jun–Oct, rain/baseflow).

### Cold season (Nov–May)

| Horizon | Flow RMSE (m³/s) | Level RMSE (m) | Skill vs. persistence |
|---------|-----------------|----------------|-----------------------|
| t+1     | 36.8            | 0.055          | +26%                  |
| t+2     | 56.7            | 0.082          | +32%                  |
| t+3     | 77.2            | 0.102          | +33%                  |
| t+4     | 97.4            | 0.121          | +32%                  |
| t+5     | 107.0           | 0.138          | +37%                  |

### Warm season (Jun–Oct)

| Horizon | Flow RMSE (m³/s) | Level RMSE (m) | Skill vs. persistence |
|---------|-----------------|----------------|-----------------------|
| t+1     | 30.9            | 0.040          | +38%                  |
| t+2     | 55.2            | 0.072          | +35%                  |
| t+3     | 74.2            | 0.095          | +33%                  |
| t+4     | 89.2            | 0.111          | +31%                  |
| t+5     | 98.0            | 0.123          | +33%                  |

## Data sources

| Source | Variables | Period |
|--------|-----------|--------|
| [CEHQ](https://www.cehq.gouv.qc.ca) | Flow (m³/s), Level (m) — station 043301 | 1922–present |
| [CEHQ](https://www.cehq.gouv.qc.ca) | Upstream level (m) — station 043108 (Lac des Deux Montagnes) | 1986–present |
| [ECCC / HYDAT](https://eau.ec.gc.ca) | Flow (m³/s) — station 02KF005 (Ottawa River at Britannia) | 1960–present |
| [ECCC / HYDAT](https://eau.ec.gc.ca) | Level (m) — station 02LA015 (Ottawa River at Hull) | 1964–present |
| [Open-Meteo ERA5](https://open-meteo.com) | Temperature, precipitation, snowfall, rain (observed) | 1940–present |
| [Open-Meteo Forecast](https://open-meteo.com) | 5-day weather forecast (temperature, precipitation, rain, snow) | real-time |
| [Crues Grand Montréal](https://www.cruesgrandmontreal.ca) | Level (m) + flow (m³/s) — upstream stations 39_RDP09, 01_RDP11, 11_LDM01 | real-time + rolling history |
| [mghydro.com](https://mghydro.com/app/report?lat=45.454&lng=-74.106&precision=low&simplify=true) | Basin boundary polygon (GeoJSON) — ID M72047806, ~148,202 km² | static |

### Upstream hydrological stations

| Station | Source | Location | Distance upstream | Variable |
|---------|--------|----------|-------------------|----------|
| 043108 | CEHQ | Lac des Deux Montagnes | ~22 km | Level (m) |
| 02LA015 | ECCC | Ottawa River at Hull | ~50 km | Level (m) |
| 02KF005 | ECCC | Ottawa River at Britannia | ~100 km | Flow (m³/s) |

### Crues Grand Montréal upstream stations

| Station | Location | Distance upstream | Variable |
|---------|----------|-------------------|----------|
| 39_RDP09 | Rue Marceau, Pierrefonds-Roxboro | ~0.8 km | Level + flow |
| 01_RDP11 | Parc Terrasse-Sacré-Cœur, Île-Bizard | ~3.5 km | Level + flow |
| 11_LDM01 | Parc Philippe-Lavallée, Oka | ~22 km (Lac des Deux Montagnes) | Level + flow |

## Pipeline

```
load_data.py      CEHQ historical (043301 + 043108) + ECCC (02KF005 + 02LA015)
                  HYDAT CSV format + real-time XML exports → loaded via glob
load_climate.py   basin boundary (mghydro.com) → ERA5 basin-mean daily climate (Open-Meteo)
load_forecast.py  5-day weather forecast (Open-Meteo) → injected at inference time
load_cgm.py       hourly level + flow for 3 upstream stations (cruesgrandmontreal.ca)
                  → daily cache (data/cgm_daily.parquet) + 5-day forecast
     │
     ▼
features.py       build_dataset() → (X, y)   [254 columns]
                  • Lags 1–30 days (flow, level, upstream 043108, Ottawa 02KF005 +
                    02LA015, 3 CGM stations, climate)
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
model.py          2 seasonal sets × 10 LGBMRegressor (one per horizon)
                  Cold: Nov–May (snowmelt, spring freshet)
                  Warm: Jun–Oct (rain, baseflow)
                  Evaluated on 2024–2026, deployed on full 1978–2026
     │
     ▼
predict.py        5-day forecast CLI → docs/forecast.png + docs/forecast_30d.png
                  + docs/forecast.json
                  Automatically selects the seasonal model set based on the anchor date
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
  separately for flow and level, with two seasonal sets (cold / warm)
- **Seasonal selection:** cold = Nov–May, warm = Jun–Oct; `predict.py` automatically
  picks the right set via `season_for()`
- **Training period:** 1978-01-01 onward (post-dam era)
- **Features:** 254 columns — lags, rolling statistics (upstream stations 043108,
  02KF005 Ottawa at Britannia, 02LA015 Ottawa at Hull, 3 CGM upstream stations),
  snowpack proxy, seasonal encoding, flow anomaly, 5-day weather forecast,
  5-day upstream hydrological forecast
- **Hyperparameters:** 500 trees, lr=0.05, 63 leaves, subsample=0.8
- **Top features (cold):** current flow, 3-day rolling max, current level,
  day-of-year (sin), 5-day cumulative forecast precipitation
- **Top features (warm):** current flow, Hull level (02LA015), 3-day rolling max,
  current level, flow anomaly
