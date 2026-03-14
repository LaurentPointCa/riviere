# CLAUDE.md — Rivière des Prairies Forecast

Personal experiment built entirely with Claude Code. No outputs are validated or suitable for real decisions.

## What this is

5-day daily forecast of flow (m³/s) and water level (m) for CEHQ station 043301, auto-committed to GitHub twice a day.

## Environment

- Python: `.venv/bin/python` (no pyproject.toml — always use this interpreter)
- Retrain model: `python src/model.py` → saves `models/lgbm_forecast.pkl`
- Run forecast manually: `python src/predict.py`
- LaunchAgent: `~/Library/LaunchAgents/ca.laurent.riviere-forecast.plist` (07:00 + 16:00)

## Non-obvious design decisions

**Training proxy for future features:** At training time, `temp_forecast_t{h}` = actual ERA5 value h days ahead (perfect information). At inference, `predict.py` overwrites these columns with real Open-Meteo / CGM forecasts. LightGBM handles the NaN in the last 5 training rows natively. Same pattern for CGM upstream forecast columns.

**Datum offset:** Historical CEHQ level for 043301 is in a different datum than the live feed. `load_level()` applies `+27.62 m` to historical values.

**CGM cache is recent-only:** `data/cgm_daily.parquet` covers only the last ~10 rolling days. Most training rows are NaN for CGM columns — intentional, LightGBM handles it.

**dict + pd.concat in features.py:** All `_add_*` helpers build a dict of Series then concat once. Do not revert to column-by-column assignment (causes fragmentation warnings).

**`v0.1` tag** marks the pre-CGM state of the repo.

**Seasonal model pkl structure:** `models/lgbm_forecast.pkl` stores `{"cold": {target: model, …}, "warm": {target: model, …}}`. `predict.py` calls `season_for(anchor_date)` to pick the right set. `cold` = Nov–May (snowmelt/freshet), `warm` = Jun–Oct (rain/baseflow). Legacy flat-dict format (pre-seasonal) is still supported as a fallback.

**ECCC data loading pattern:** HYDAT CSV (via `_parse_eccc_csv()`, PARAM column filters flow vs level) is the historical base. Recent data is extended with real-time XML exports placed manually in `data/` and loaded via glob (e.g. `data/02KF005_QRD_*.xml`). If the XML still leaves a gap, `_fetch_eccc_recent()` queries the MSC GeoMet API (`api.weather.gc.ca/collections/hydrometric-daily-mean`). All three sources are concatenated and deduplicated via `.groupby(level=0).mean()`.

**02KF005 vs 02LA015:** 02KF005 (Britannia) provides flow; 02LA015 (Hull) provides level only — flow is not measured there. Both are Ottawa River stations upstream of the Des Prairies confluence. `hull_level_m` ranked #2 by feature importance in the warm-season model.

## Backlog

- Add historical data for the Carillon dam (Ottawa River, upstream inflow control)
- ~~More upstream CEHQ stations~~ (added 02KF005 + 02LA015 ECCC stations)
- ~~Seasonal/regime-specific models~~ (done: cold Nov–May / warm Jun–Oct)
- Recursive multi-step forecasting
- Ice flag features (CEHQ marks ice-corrected values with `*`)
- Proper walk-forward cross-validation
- Prediction intervals
