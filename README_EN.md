# Rivière des Prairies — 5-Day Forecast

Daily flow (m³/s) and water level (m) forecast for CEHQ station **043301**
(Rivière des Prairies at Laval), using a LightGBM model trained on 45+ years
of hydrological and climate data, with real-time weather and upstream hydrological
forecast injection.

![5-day forecast chart](docs/forecast.png)

Latest forecast also available as machine-readable JSON: [`docs/forecast.json`](docs/forecast.json)

## Deployed models

Three models run on every forecast cycle, each writing its own set of output files.

| Model | Output files | Purpose |
|-------|-------------|---------|
| **Quantile CV-tuned** (production) | `forecast.*` | Optimized for flood detection — biases predictions upward |
| MSE seasonal | `forecast_mse.*` | Standard regression baseline — good for normal conditions |
| Ext10 MSE | `forecast_ext10.*` | MSE variant with weather forecast features extended to t+10 |

### Production model — quantile regression (α=0.85)

The production model uses LightGBM quantile regression instead of standard MSE. Predictions are biased upward under uncertainty: under-prediction is penalized 85% of the time. Hyperparameters were tuned by minimizing pinball loss on days where flow exceeds 1,500 m³/s (approaching flood level), using walk-forward CV folds 2020–2023, with 2017 and 2019 held out for final evaluation.

**Why not MSE?** The 2024–2026 test set contains zero flood days (max: 2,269 m³/s). RMSE on those two years says nothing about what matters most. Of the 9,490 cold-season days in the full dataset, only 145 (1.5%) exceed the 2,500 m³/s concern threshold. An MSE model learns to predict normal conditions precisely while missing flood rises.

**Flood detection results** — walk-forward out-of-sample evaluation on flood years 2017, 2019, 2023 (threshold: 2,500 m³/s):

| Horizon | Recall — MSE | Recall — quantile CV | Precision — quantile CV |
|---------|-------------|---------------------|------------------------|
| t+1 | 0.935 | 0.989 | ≥ 0.876 |
| t+3 | 0.828 | 0.946 | ≥ 0.876 |
| t+5 | 0.656 | 0.839 | ≥ 0.876 |

The quantile CV model also gives 1–2 days of advance warning for the 2017 and 2019 flood onsets, versus zero days for the MSE model.

The `forecast.json` output includes a `flood_risk` block with two boolean flags:
- `concern`: any predicted horizon exceeds 2,500 m³/s
- `near_flood`: any predicted horizon exceeds 3,000 m³/s

### Reference models

- **`forecast_mse`** — the MSE seasonal model that was in production before the quantile switch. Useful as a reference for normal-conditions accuracy. The RMSE tables below reflect this model.
- **`forecast_ext10`** — MSE variant trained with weather forecast features extended to t+10 horizons (vs. t+5). Allows the model to capture a longer meteorological signal.

## Watershed and stations

![Basin map](docs/basin_map.png)

The watershed is delineated by [mghydro.com](https://mghydro.com) from the outlet point near Ottawa. ERA5 climate data and the Open-Meteo forecast are averaged across this full basin.

![Monitoring stations](docs/basin_stations.png)

The seven monitoring stations used as model inputs, from the Ottawa region to Laval.

## Results (MSE reference model)

RMSE on the held-out test set (2024-03-10 → 2026-03-09, 730 days). These numbers reflect the MSE seasonal model — not the production model's optimization target (see flood detection results above). The deployed models are retrained on the full dataset (1978-01-01 → 2026-03-09).

Two separate seasonal models: **cold** (Nov–May, snowmelt/spring freshet) and **warm** (Jun–Oct, rain/baseflow).

### Cold season (Nov–May)

| Horizon | Flow RMSE (m³/s) | Level RMSE (m) | Skill vs. persistence |
|---------|-----------------|----------------|-----------------------|
| t+1     | 36.2            | 0.054          | +26%                  |
| t+2     | 55.4            | 0.081          | +33%                  |
| t+3     | 74.8            | 0.102          | +34%                  |
| t+4     | 94.5            | 0.120          | +33%                  |
| t+5     | 104.3           | 0.136          | +37%                  |

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
| [mghydro.com](https://mghydro.com/app/report?lat=45.454&lng=-74.106&precision=low&simplify=true) | Basin boundary polygon (GeoJSON) — ID M72047806, ~148,202 km² | static |

> **Note:** [Crues Grand Montréal](https://www.cruesgrandmontreal.ca) stations (39_RDP09, 01_RDP11, 11_LDM01) are disabled pending multi-year historical data.

### Upstream hydrological stations

| Station | Source | Location | Distance upstream | Variable |
|---------|--------|----------|-------------------|----------|
| 043108 | CEHQ | Lac des Deux Montagnes | ~22 km | Level (m) |
| 02LA015 | ECCC | Ottawa River at Hull | ~50 km | Level (m) |
| 02KF005 | ECCC | Ottawa River at Britannia | ~100 km | Flow (m³/s) |

## Pipeline

```
load_data.py      CEHQ historical (043301 + 043108) + ECCC (02KF005 + 02LA015)
                  HYDAT CSV format + real-time XML exports → loaded via glob
load_climate.py   basin boundary (mghydro.com) → ERA5 basin-mean daily climate (Open-Meteo)
load_forecast.py  5-day weather forecast (Open-Meteo) → injected at inference time
     │
     ▼
features.py       build_dataset() → (X, y)   [165 columns]
                  • Lags 1–30 days (flow, level, upstream 043108, Ottawa 02KF005 +
                    02LA015, climate, ERA5-Land snow depth)
                  • Rolling mean/max/std (3–30 days)
                  • Snowpack proxy (degree-day model)
                  • Seasonal encoding (sin/cos DOY)
                  • Flow anomaly vs seasonal median
                  • Weather forecast t+1…t+5 (ERA5 perfect-forecast proxy at
                    training time, real Open-Meteo forecast at inference time)
     │
     ▼
model.py          2 seasonal sets × 10 LGBMRegressor (one per horizon)
                  Cold: Nov–May (snowmelt, spring freshet)
                  Warm: Jun–Oct (rain, baseflow)
                  Evaluated on 2024-03-10 → 2026-03-09, deployed on full 1978–2026
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
- **Features:** 165 columns — lags, rolling statistics (upstream stations 043108,
  02KF005 Ottawa at Britannia, 02LA015 Ottawa at Hull), ERA5-Land snow depth,
  snowpack proxy, seasonal encoding, flow anomaly, 5-day weather forecast
- **Production model:** quantile regression α=0.85, hyperparameters tuned via
  event-focused CV (pinball loss on days with flow > 1,500 m³/s, folds 2020–2023)
- **MSE/Ext10 hyperparameters:** 500 trees, lr=0.05, 63 leaves, subsample=0.8
- **Top features (cold):** current flow, 3-day rolling max, current level,
  day-of-year (sin), 5-day cumulative forecast precipitation
- **Top features (warm):** current flow, Hull level (02LA015), 3-day rolling max,
  current level, flow anomaly
