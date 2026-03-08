# Prévision du niveau de la station 043301

Ce projet est une simple expérimentation que j'ai créé afin de ma familiariser
avec Claude Code. Aucune donnée ni méthode utilisée ici n'est réputée être valide,
fonctionnelle ni exacte.

Les données de ce dépôt GitHub ainsi que celles produites par les outils ne
peuvent pas être utilisées pour aucune fin de planification ou de décision
par rapport au niveau de la rivière.

L'auteur se dégage de toute responsabilité face à l'utilisation par quiconque
des données ou outils sur ce site. Le tout n'est qu'une exploration et ne sert
qu'à discuter des outils et techniques explorés ici.

  -Laurent

# Rivière des Prairies — Prévision sur 5 jours

Prévision quotidienne du débit (m³/s) et du niveau d'eau (m) pour la station CEHQ **043301**
(Rivière des Prairies à Laval), à l'aide d'un modèle LightGBM entraîné sur plus de 45 ans
de données hydrologiques et climatiques, avec injection de la prévision météo et des conditions
hydrologiques en amont en temps réel.

![Graphique de prévision sur 5 jours](docs/forecast.png)

La prévision la plus récente est également disponible en JSON lisible par machine : [`docs/forecast.json`](docs/forecast.json)

## Résultats

Ensemble de test retenu (2024-03-04 → 2026-03-03, 730 jours) utilisé pour l'évaluation.
Le modèle déployé est ré-entraîné sur l'ensemble complet des données (1978-01-01 → 2026-03-03).

| Horizon | RMSE débit (m³/s) | RMSE niveau (m) | Gain vs. persistance |
|---------|-------------------|-----------------|----------------------|
| t+1     | 40,2              | 0,060           | +17 %                |
| t+2     | 68,8              | 0,096           | +17 %                |
| t+3     | 91,7              | 0,123           | +17 %                |
| t+4     | 106,5             | 0,143           | +20 %                |
| t+5     | 120,8             | 0,159           | +21 %                |

## Sources de données

| Source | Variables | Période |
|--------|-----------|---------|
| [CEHQ](https://www.cehq.gouv.qc.ca) | Débit (m³/s), Niveau (m) — station 043301 | 1922–présent |
| [CEHQ](https://www.cehq.gouv.qc.ca) | Niveau amont (m) — station 043108 (Lac des Deux Montagnes) | 1986–présent |
| [Open-Meteo ERA5](https://open-meteo.com) | Température, précipitations, chutes de neige, pluie (observé) | 1940–présent |
| [Open-Meteo Forecast](https://open-meteo.com) | Prévision météo sur 5 jours (température, précipitations, pluie, neige) | temps réel |
| [Crues Grand Montréal](https://www.cruesgrandmontreal.ca) | Niveau (m) + débit (m³/s) — stations amont 39_RDP09, 01_RDP11, 11_LDM01 | temps réel + historique glissant |
| [mghydro.com](https://mghydro.com) | Polygone du bassin versant (GeoJSON) | statique |

### Stations amont Crues Grand Montréal

| Station | Localisation | Distance amont | Débit max ref. |
|---------|-------------|----------------|----------------|
| 39_RDP09 | Rue Marceau, Pierrefonds-Roxboro | ~0,8 km | 3 172 m³/s |
| 01_RDP11 | Parc Terrasse-Sacré-Cœur, Île-Bizard | ~3,5 km | 3 172 m³/s |
| 11_LDM01 | Parc Philippe-Lavallée, Oka | ~22 km (Lac des Deux Montagnes) | 11 340 m³/s |

## Pipeline

```
load_data.py      Données historiques CEHQ + flux en direct (stations 043301 + amont 043108)
load_climate.py   Bassin versant (mghydro.com) → Climat journalier moyen ERA5 (Open-Meteo)
load_forecast.py  Prévision météo 5 jours (Open-Meteo) → injectée à l'inférence
load_cgm.py       Niveau + débit horaires des 3 stations amont (cruesgrandmontreal.ca)
                  → cache journalier (data/cgm_daily.parquet) + prévision 5 jours
     │
     ▼
features.py       build_dataset() → (X, y)   [222 colonnes]
                  • Décalages 1–30 jours (débit, niveau, amont 043108 + 3 stations CGM, climat)
                  • Moyenne/max/écart-type glissants (3–30 jours)
                  • Proxy d'enneigement (modèle degré-jour)
                  • Encodage saisonnier (sin/cos jour de l'année)
                  • Anomalie de débit vs médiane saisonnière
                  • Prévision météo t+1…t+5 (proxy ERA5 à l'entraînement,
                    prévision réelle Open-Meteo à l'inférence)
                  • Prévision CGM t+1…t+5 (proxy observé à l'entraînement,
                    prévision réelle CMM à l'inférence)
     │
     ▼
model.py          10 × LGBMRegressor (un par horizon)
                  Évalué sur 2024–2026, déployé sur l'ensemble 1978–2026
     │
     ▼
predict.py        Interface CLI de prévision 5 jours → docs/forecast.png + docs/forecast_30d.png
                  + docs/forecast.json
```

## Utilisation

```bash
# Configurer l'environnement
python -m venv .venv
source .venv/bin/activate
pip install lightgbm scikit-learn pandas numpy requests shapely pyarrow
brew install libomp   # macOS seulement

# Construire les caractéristiques et entraîner le modèle (télécharge les données au premier lancement)
python src/model.py

# Prévision à partir de la dernière date disponible
python src/predict.py

# Prévision à partir d'une date passée (affiche les valeurs observées vs prédites)
python src/predict.py --date 2025-06-01
```

## Détails du modèle

- **Stratégie :** multi-sortie directe — un `LGBMRegressor` par horizon (t+1…t+5),
  séparément pour le débit et le niveau
- **Période d'entraînement :** à partir du 1978-01-01 (ère post-barrage)
- **Caractéristiques :** 222 colonnes — décalages, statistiques glissantes (station 043108 +
  3 stations CGM amont), proxy d'enneigement, encodage saisonnier, anomalie de débit,
  prévision météo 5 jours, prévision hydrologique amont 5 jours
- **Hyperparamètres :** 500 arbres, lr=0,05, 63 feuilles, subsample=0,8
- **Meilleures caractéristiques :** débit actuel, max glissant sur 3 jours, niveau actuel,
  jour de l'année (sin), précipitations cumulées prévues sur 5 jours, température moyenne sur 30 jours

---

# Rivière des Prairies — 5-Day Forecast

Daily flow (m³/s) and water level (m) forecast for CEHQ station **043301**
(Rivière des Prairies at Laval), using a LightGBM model trained on 45+ years
of hydrological and climate data, with real-time weather and upstream hydrological
forecast injection.

![5-day forecast chart](docs/forecast.png)

Latest forecast also available as machine-readable JSON: [`docs/forecast.json`](docs/forecast.json)

## Results

Held-out test set (2024-03-04 → 2026-03-03, 730 days) used for evaluation.
The deployed model is retrained on the full dataset (1978-01-01 → 2026-03-03).

| Horizon | Flow RMSE (m³/s) | Level RMSE (m) | Skill vs. persistence |
|---------|-----------------|----------------|-----------------------|
| t+1     | 40.2            | 0.060          | +17%                  |
| t+2     | 68.8            | 0.096          | +17%                  |
| t+3     | 91.7            | 0.123          | +17%                  |
| t+4     | 106.5           | 0.143          | +20%                  |
| t+5     | 120.8           | 0.159          | +21%                  |

## Data sources

| Source | Variables | Period |
|--------|-----------|--------|
| [CEHQ](https://www.cehq.gouv.qc.ca) | Flow (m³/s), Level (m) — station 043301 | 1922–present |
| [CEHQ](https://www.cehq.gouv.qc.ca) | Upstream level (m) — station 043108 (Lac des Deux Montagnes) | 1986–present |
| [Open-Meteo ERA5](https://open-meteo.com) | Temperature, precipitation, snowfall, rain (observed) | 1940–present |
| [Open-Meteo Forecast](https://open-meteo.com) | 5-day weather forecast (temperature, precipitation, rain, snow) | real-time |
| [Crues Grand Montréal](https://www.cruesgrandmontreal.ca) | Level (m) + flow (m³/s) — upstream stations 39_RDP09, 01_RDP11, 11_LDM01 | real-time + rolling history |
| [mghydro.com](https://mghydro.com) | Basin boundary polygon (GeoJSON) | static |

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
