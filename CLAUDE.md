# CLAUDE.md — Rivière des Prairies Forecast

Personal experiment built entirely with Claude Code. No outputs are validated or suitable for real decisions.

## What this is

Daily forecast of flow (m³/s) and water level (m) for CEHQ station 043301, auto-committed to GitHub twice a day. Two horizons run side-by-side:

- **5-day production** (`forecast.*`) — quantile α=0.85 CV-tuned, flood-biased. `models/lgbm_forecast.pkl`.
- **5-day MSE reference** (`forecast_mse.*`) — unbiased point forecast. `models/lgbm_forecast_mse.pkl`.
- **10-day experimental** (`forecast_10d.*`) — quantile α=0.85 tuned, 20 targets × 2 seasons. `models/lgbm_forecast_10d_quantile_tuned.pkl`. Surfaced on the site in a dedicated "Expérimental / Experimental" card at the bottom; horizons t+6..t+10 render dimmed.

## Environment

- Python: `.venv/bin/python` (no pyproject.toml — always use this interpreter)
- Retrain model: `python src/model.py` → saves `models/lgbm_forecast.pkl`
- Run forecast manually: `python src/predict.py`
- LaunchAgent: `~/Library/LaunchAgents/ca.laurent.riviere-forecast.plist` (07:00 + 16:00)

## Non-obvious design decisions

**Training proxy for future features:** At training time, `temp_forecast_t{h}` = actual ERA5 value h days ahead (perfect information, h=1..10). At inference, `predict.py` overwrites these columns with real Open-Meteo forecasts (fetches up to 10 days). LightGBM handles the NaN in the last 10 training rows natively.

**Horizon auto-detection in predict.py:** `forecast()` infers max horizon from the loaded model's keys (`flow_t{h}`). A 5-day pkl yields 5 rows; a 10-day pkl yields 10. The `_RMSE` band table has both 5-horizon and 10-horizon versions; the right one is picked by `len(result)`.

**Datum offset:** Historical CEHQ level for 043301 is in a different datum than the live feed. `load_level()` applies `+27.62 m` to historical values.

**CGM cache is recent-only:** `data/cgm_daily.parquet` covers only the last ~10 rolling days. Most training rows are NaN for CGM columns — intentional, LightGBM handles it.

**dict + pd.concat in features.py:** All `_add_*` helpers build a dict of Series then concat once. Do not revert to column-by-column assignment (causes fragmentation warnings).

**`v0.1` tag** marks the pre-CGM state of the repo.

**Seasonal model pkl structure:** `models/lgbm_forecast.pkl` stores `{"cold": {target: model, …}, "warm": {target: model, …}}`. `predict.py` calls `season_for(anchor_date)` to pick the right set. `cold` = Nov–May (snowmelt/freshet), `warm` = Jun–Oct (rain/baseflow). Legacy flat-dict format (pre-seasonal) is still supported as a fallback.

**ECCC data loading pattern:** HYDAT CSV (via `_parse_eccc_csv()`, PARAM column filters flow vs level) is the historical base. Recent data is extended with real-time XML exports placed manually in `data/` and loaded via glob (e.g. `data/02KF005_QRD_*.xml`). If the XML still leaves a gap, `_fetch_eccc_recent()` queries the MSC GeoMet API: first `hydrometric-daily-mean` (reviewed values, typically ~12 month lag), then falls back to `hydrometric-realtime` (5-min samples aggregated to daily means) for anything after the daily-mean cutoff. All sources are concatenated and deduplicated via `.groupby(level=0).mean()` — XML refreshes are now optional since the realtime fallback auto-fills the trailing gap.

**02KF005 vs 02LA015:** 02KF005 (Britannia) provides flow; 02LA015 (Hull) provides level only — flow is not measured there. Both are Ottawa River stations upstream of the Des Prairies confluence. `hull_level_m` ranked #2 by feature importance in the warm-season model.

## Backlog

- Add historical data for the Carillon dam (Ottawa River, upstream inflow control) — data source obtained
- ~~Add historical data for Crues Grand Montréal stations (39_RDP09, 01_RDP11, 11_LDM01)~~ (obtained, evaluated, discarded — useless to model)
- ~~More upstream CEHQ stations~~ (added 02KF005 + 02LA015 ECCC stations)
- ~~Seasonal/regime-specific models~~ (done: cold Nov–May / warm Jun–Oct)
- Recursive multi-step forecasting
- ~~Ice flag features~~ (won't do: CEHQ historical files have no ice flags — remark codes are quality/method codes, not ice indicators. CEHQ applies corrections before publishing. Temperature lags and snow depth already encode ice season implicitly.)
- ~~Extended forecast horizon (10 days)~~ (done: experimental 10-day quantile tuned model, CV-tuned, surfaced as separate card on site)
- Proper walk-forward cross-validation
- Prediction intervals
