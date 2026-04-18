# Rivière des Prairies — Task Backlog

## SEO pass + English version (complete)

Goal: make the GitHub Pages site discoverable in both French and English.
Canonical base URL: `https://laurentpointca.github.io/riviere/`.

- [x] Add SEO meta tags to `docs/index.html` (title, description, canonical, robots, OG, Twitter)
- [x] Add hreflang alternates (fr, en, x-default) on both pages
- [x] Add JSON-LD structured data (WebApplication + Dataset) to both pages
- [x] Create `docs/en/index.html` — full English translation (UI, about, legal, disclaimer, legend)
- [x] Add FR ↔ EN language switcher to both pages (4th button in info-links row)
- [x] Improve image `alt` text on charts (descriptive rather than generic)
- [x] Create `docs/robots.txt`
- [x] Create `docs/sitemap.xml` listing both language URLs with xhtml:link alternates

### Review
- French page: added full SEO head block (title, description, canonical, robots, hreflang × 3,
  OG, Twitter Card, JSON-LD with WebApplication + Dataset schema), language switcher, better
  alt text on the two forecast charts. No layout changes, no JS logic changes.
- English page (`docs/en/index.html`): full standalone translation with adjusted relative paths
  (`../forecast.json`, `../forecast_30d.png`, `../forecast.png`) and back-link to `../` for FR.
  Uses `lang="en"`, `en-CA` locale, English day names and legal copy.
- SEO artifacts: `robots.txt` allows all and points to sitemap; `sitemap.xml` lists both URLs
  with `xhtml:link rel="alternate"` for Google's international SEO guidelines.
- Post-deploy follow-ups (manual, outside code): submit sitemap in Google Search Console and
  Bing Webmaster Tools under both language URLs; optionally create a dedicated 1200×630 OG
  image instead of reusing `forecast_30d.png`.

---

## Carillon Dam Integration (current focus)

Context: Carillon dam (Hydro-Québec) controls Ottawa River inflow upstream of the Des Prairies
confluence. Historical discharge data has been obtained — integrate into the model as a new
upstream feature.

- [ ] Make flow graphs log-scale instead of linear
- [ ] Fix VM/Mac git push conflicts: move VM forecast commits to a separate `forecast` branch (preferred), and have GitHub Pages serve from that branch — or alternatively merge `forecast` into `master` via Action. Goal: single writer per branch, no more force-with-lease races.
- [ ] Load and explore Carillon historical discharge data
- [ ] Add `load_carillon()` to `src/load_data.py`
- [ ] Add Carillon features (lags, rolling stats) to `src/features.py`
- [ ] Retrain model with Carillon features
- [ ] Evaluate with flood detection framework — must meet: recall@t+3>=0.90, precision>=0.85
- [ ] If improved, promote to production and SCP to VM

---

## Flood Detection Reorientation (complete)

Context: The model's real goal is to predict upcoming flow increases concerning to residents
(2500 m³/s = concern threshold, 3000 m³/s = near flood). RMSE optimization on a test set
with no flood events (2024–2026 max: 2269 m³/s) does not measure what matters. Full
analysis in memory/project_flood_detection_reorientation.md.

### Step 1 — Build flood detection evaluation framework
- [x] Write `src/evaluate_flood.py`:
  - Load flow history back to 1978
  - For each horizon t+1..t+5, compute precision/recall on flow > 2500 m³/s threshold
  - Also report lead time: how many days before threshold crossing did we first predict it?
  - Run against the current production model on 2017, 2019, 2023 (the flood years)
  - This becomes the baseline to beat before any model gets promoted
- [x] Baseline results (MSE walk-forward):
  - Precision: 0.95–1.00 across all horizons (almost no false alarms)
  - Recall: 0.935 (t+1) → 0.656 (t+5) — collapses at longer horizons
  - 2017 worst: recall=0.552 at t+3, 0.103 at t+5
  - Lead time: 2017 and 2019 onset events had ZERO advance warning (model only
    detected flood once it was already happening); 2023 had 2–5 day warnings

### Step 2 — Quantile regression model (alpha=0.85)
- [x] In `src/model.py`, add a `train_quantile(X, y, alpha=0.85)` variant
- [x] Train and save as `models/lgbm_forecast_quantile.pkl`
- [x] Evaluate with the flood detection framework from Step 1
- [x] Result: clear improvement over MSE baseline
  - Recall t+3: 0.828 → 0.925  |  t+5: 0.656 → 0.806
  - Precision stays high: 0.915 at t+3 and t+5 (was 0.951/0.953)
  - Lead time: 2017 onset 0 days → 5-day warning; 2019 onset 0 → 2-day warning
  - Verdict: quantile model advances to Step 3 (event-focused tuning)

### Step 3 — Event-focused hyperparameter tuning
- [x] Modify `src/tune_hyperparams.py` to support event-focused CV (`--mode quantile`)
  - CV metric: pinball loss on days where current flow > 1500 m³/s
  - CV folds: [2020, 2021, 2022, 2023] — 2017/2019 held out
- [x] Run 30-trial quantile tuning → `models/lgbm_forecast_quantile_tuned.pkl`
- [x] Evaluate with flood detection framework
- [x] Results vs untuned quantile:
  - Recall t+1: 0.978 → 0.989  |  t+3: 0.925 → 0.946  |  t+5: 0.806 → 0.839
  - Precision stays ≥ 0.876 at all horizons
  - Lead time: 2017 first onset regressed (5d → 1d); 2019 onset held at 2d

### Step 4 — Evaluate and promote
- [x] Compare on flood years (2017, 2019, 2023):
  - MSE (ext10):          recall t+3=0.828  t+5=0.656  precision≥0.951  lead=0d/0d
  - Quantile α=0.85:      recall t+3=0.925  t+5=0.806  precision≥0.915  lead=5d/2d
  - Quantile tuned CV:    recall t+3=0.946  t+5=0.839  precision≥0.876  lead=1d/2d
- [x] Promotion criteria: recall@t+3≥0.90, precision≥0.85 at all horizons, lead>0d
  - MSE fails recall (0.828) and lead time (0d/0d)
  - Winner: quantile CV-tuned (all criteria met, strictly best recall)
- [x] Promote winning model to production + SCP to VM
  - lgbm_forecast_quantile_tuned.pkl → lgbm_forecast.pkl (+ VM)

### Step 5 — Update forecast output for residents
- [x] Add flood risk signal to `docs/forecast.json`:
  - `flood_risk.concern` = true when any horizon > 2500 m³/s
  - `flood_risk.near_flood` = true when any horizon > 3000 m³/s
  - `flood_risk.max_predicted_flow_m3s` = max flow across all horizons
- [x] Add threshold lines to forecast charts (forecast.png, forecast_30d.png)
  - Amber line at 2500 m³/s ("Préoccupation") and red at 3000 m³/s ("Quasi-crue")

---

## Forecast Horizon Experiments

- [ ] **10-day forecast** — extend predictions from current t+5 to t+10 horizon
  - Evaluate performance: check recall@t+3/t+5/t+10 on flood detection framework, precision, lead time
  - Compare against current 5-day baseline
  - If marginal degradation (recall drop <5%), promote to prod
  
- [ ] **Same-day forecast (24h)** — add t+0 (next 24 hours) to the forecast output
  - Useful for immediate flood monitoring / decision support (current forecast starts at t+1)
  - Train on available features lagged appropriately
  - Evaluate on flood framework — should have highest recall/precision since it's the shortest horizon

---

## Model Retraining

- [ ] **Annual retrain — October** (after warm season closes, before cold season freshet risk)
  - Update ECCC XML exports for 02KF005 and 02LA015 (drop updated files in `data/` before retraining)
  - Run `python src/model.py` on Mac, SCP new pkl files to VM
  - Evaluate with `src/evaluate_flood.py` — must meet: recall@t+3≥0.90, precision≥0.85, lead>0d
  - Check validation plot for drift before and after

- [ ] **Event-triggered retrain — after any observed flow >2500 m³/s**
  - Same prep as above; new flood year is highest-value training addition
  - Update flood eval baseline in memory after

- [ ] **Drift monitoring** — watch validation plot for persistent bias across all horizons

---

## Backlog / Future Work

- [ ] Recursive multi-step forecasting
- [ ] Proper walk-forward cross-validation
- [ ] Prediction intervals (confidence bands on forecast output)
- [ ] Better snowpack modeling

---

## Completed

- [x] Weather forecast features extended to t+10 (ext10) — promoted to production
- [x] Walk-forward CV infrastructure (model.py `walk_forward_cv()`)
- [x] Hyperparameter tuning with 5-fold CV (tune_hyperparams.py) — evaluated, not promoted
      (reason: RMSE objective is wrong for flood detection goal)
- [x] Parallel tuning with ProcessPoolExecutor (8 workers)
- [x] Fix empty live feed crash (DatetimeIndex bug in _live_daily)
- [x] Establish Mac=train, VM=run workflow
