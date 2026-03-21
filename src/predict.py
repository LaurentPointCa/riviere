"""
5-day forecast script for station 043301 — Des Prairies.

Usage:
    python src/predict.py                       # forecast from latest available date
    python src/predict.py --date 2025-06-01     # forecast from a specific past date
"""

import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import build_dataset
from model import load_model, season_for
from load_forecast import load_weather_forecast

MODEL_PATH = Path("models/lgbm_forecast.pkl")


def _inject_weather_forecast(row: pd.DataFrame, wf: pd.DataFrame) -> pd.DataFrame:
    """
    Replace ERA5 perfect-forecast proxy columns with real Open-Meteo forecast values.

    Parameters
    ----------
    row : single-row feature DataFrame for the anchor date
    wf  : forecast DataFrame from load_weather_forecast(), indexed by date

    Returns
    -------
    Updated row with real forecast values injected.
    """
    row = row.copy()
    for h in range(1, 6):
        if h > len(wf):
            break
        fc = wf.iloc[h - 1]
        row[f"temp_forecast_t{h}"]   = fc["temperature_2m_mean"]
        row[f"precip_forecast_t{h}"] = fc["precipitation_sum"]
        row[f"rain_forecast_t{h}"]   = fc["rain_sum"]
        row[f"snow_forecast_t{h}"]   = fc["snowfall_sum"]

    # Recompute derived aggregates
    row["precip_forecast_sum_5d"] = sum(float(row[f"precip_forecast_t{h}"].iloc[0]) for h in range(1, 6))
    row["temp_forecast_mean_5d"]  = sum(float(row[f"temp_forecast_t{h}"].iloc[0])   for h in range(1, 6)) / 5

    return row


def _row_set(row: pd.DataFrame, col: str, val: float) -> None:
    """Set a scalar value in a 1-row DataFrame column, in-place."""
    if col in row.columns:
        row[col] = val


def _update_rolling_stats(row: pd.DataFrame, ext_series: list, prefix: str) -> None:
    """Update rolling statistics columns for flow or level in a 1-row DataFrame."""
    import numpy as np
    specs = [
        (f"{prefix}_roll_mean_3d",  3,  "mean"),
        (f"{prefix}_roll_mean_7d",  7,  "mean"),
        (f"{prefix}_roll_mean_14d", 14, "mean"),
        (f"{prefix}_roll_mean_30d", 30, "mean"),
        (f"{prefix}_roll_max_3d",   3,  "max"),
        (f"{prefix}_roll_max_7d",   7,  "max"),
        (f"{prefix}_roll_max_14d",  14, "max"),
        (f"{prefix}_roll_std_7d",   7,  "std"),
        (f"{prefix}_roll_std_14d",  14, "std"),
    ]
    for col, w, func in specs:
        if col not in row.columns or len(ext_series) < w:
            continue
        window = ext_series[-w:]
        if func == "mean":
            val = float(np.mean(window))
        elif func == "max":
            val = float(np.max(window))
        else:
            val = float(np.std(window))
        row[col] = val


def predict_recursive(
    models: dict,
    anchor_row: pd.DataFrame,
    flow_obs: pd.Series,
    level_obs: pd.Series,
    weather_df: pd.DataFrame,
    anchor_date: pd.Timestamp,
) -> list[dict]:
    """
    Roll flow_t1/level_t1 models forward 5 steps, updating the feature row at each step.

    Parameters
    ----------
    models     : dict containing at least "flow_t1" and "level_t1" keys
    anchor_row : 1-row feature DataFrame for anchor_date (X.loc[[anchor_date]])
    flow_obs   : observed flow series (X["flow_m3s"]), indexed by date
    level_obs  : observed level series (X["level_m"]), indexed by date
    weather_df : ≤5-row future weather DataFrame (row 0 = anchor+1, row 1 = anchor+2 …)
    anchor_date: forecast anchor as pd.Timestamp

    Returns
    -------
    list of 5 dicts: [{"flow_m3s": float, "level_m": float}, ...]
    """
    import numpy as np

    LAG_DAYS   = [1, 2, 3, 4, 5, 7, 14, 30]
    MELT_FACTOR = 4.0
    SNOW_RATIO  = 0.1

    # Extended series: observed history (≥35 days) + predictions accumulated in-loop
    ext_flow  = list(flow_obs.iloc[-35:].values)
    ext_level = list(level_obs.iloc[-35:].values)

    # Seasonal median by DOY (for flow_anom recomputation)
    doy_median = flow_obs.groupby(flow_obs.index.day_of_year).median()

    current_row = anchor_row.copy()
    predictions: list[dict] = []

    for k in range(1, 6):
        pred_f = float(models["flow_t1"].predict(current_row)[0])
        pred_l = float(models["level_t1"].predict(current_row)[0])
        predictions.append({"flow_m3s": pred_f, "level_m": pred_l})

        if k == 5:
            break

        # Append prediction: ext_flow[-1] = predicted flow at anchor+k
        ext_flow.append(pred_f)
        ext_level.append(pred_l)

        next_row = current_row.copy()

        # 1. Current flow / level (the "virtual today" = anchor+k)
        _row_set(next_row, "flow_m3s", pred_f)
        _row_set(next_row, "level_m",  pred_l)

        # 2. Lag columns  — ext_flow[-1] = anchor+k, lag_d = ext_flow[-(d+1)]
        for d in LAG_DAYS:
            if len(ext_flow) > d:
                _row_set(next_row, f"flow_m3s_lag{d}",  ext_flow[-(d + 1)])
                _row_set(next_row, f"level_m_lag{d}",   ext_level[-(d + 1)])

        # 3. Rolling statistics (flow and level only; upstream/hull held constant)
        _update_rolling_stats(next_row, ext_flow,  "flow")
        _update_rolling_stats(next_row, ext_level, "level")

        # 4. Weather forecast — shift by k so t+1 = original day k+1
        if not weather_df.empty and k < len(weather_df):
            next_row = _inject_weather_forecast(next_row, weather_df.iloc[k:])

        # 5. Seasonal features
        step_date = anchor_date + pd.Timedelta(days=k)
        doy = float(step_date.day_of_year)
        _row_set(next_row, "doy_sin", float(np.sin(2 * np.pi * doy / 365.25)))
        _row_set(next_row, "doy_cos", float(np.cos(2 * np.pi * doy / 365.25)))
        _row_set(next_row, "month",   step_date.month)

        # 6. Snowpack proxy — advance one day using forecast temp/snow at day k
        if "snowpack_proxy_mm" in next_row.columns:
            swe = float(next_row["snowpack_proxy_mm"].iloc[0])
            if k <= len(weather_df):
                wfc    = weather_df.iloc[k - 1]
                temp_k = float(wfc.get("temperature_2m_mean", 0) or 0)
                snow_k = float(wfc.get("snowfall_sum", 0) or 0)
            else:
                temp_k, snow_k = 0.0, 0.0
            swe += snow_k * SNOW_RATIO
            if temp_k > 0:
                swe -= temp_k * MELT_FACTOR
            _row_set(next_row, "snowpack_proxy_mm", max(0.0, swe))

        # 7. Flow anomaly
        if "flow_anom" in next_row.columns:
            step_doy    = step_date.day_of_year
            median_flow = float(doy_median.get(step_doy, doy_median.median()))
            _row_set(next_row, "flow_anom", pred_f - median_flow)

        current_row = next_row

    return predictions


def _load_models(model_path: "Path", anchor_date: pd.Timestamp) -> dict:
    """Load seasonal or flat models from path, return the season's model dict."""
    all_models = load_model(model_path)
    if "cold" in all_models and "warm" in all_models:
        season = season_for(anchor_date)
        models = all_models[season]
        print(f"Using {season} season model ({anchor_date.strftime('%b %d')}).")
    else:
        models = all_models
    return models


def _load_weather(anchor_date: pd.Timestamp, X: pd.DataFrame) -> "pd.DataFrame":
    """Fetch 5-day weather forecast for the anchor date; return empty DF on failure."""
    if anchor_date != X.index[-1] or "temp_forecast_t1" not in X.columns:
        return pd.DataFrame()
    try:
        wf = load_weather_forecast(days=5)
        future = wf[wf.index > anchor_date].head(5)
        if future.empty:
            return pd.DataFrame()
        if len(future) < 5:
            print(f"Warning: only {len(future)}/5 weather forecast days available; "
                  f"t+{len(future)+1}..t+5 will use ERA5 proxy.")
        return future
    except Exception as e:
        print(f"Warning: could not load weather forecast ({e}); using ERA5 proxy.")
        return pd.DataFrame()


def forecast(anchor_date: pd.Timestamp, X: pd.DataFrame, model_path=None) -> pd.DataFrame:
    """
    Generate a 5-day forecast of flow and level from the given anchor date.

    Parameters
    ----------
    anchor_date : the "today" date to forecast from; must be in X.index
    X           : feature matrix from build_dataset()
    model_path  : override model file (default: MODEL_PATH)

    Returns
    -------
    pd.DataFrame with columns date, flow_m3s, level_m; indexed 1..5 (horizon)
    """
    path   = model_path or MODEL_PATH
    models = _load_models(path, anchor_date)
    row    = X.loc[[anchor_date]]

    future = _load_weather(anchor_date, X)
    if not future.empty:
        row = _inject_weather_forecast(row, future)
        print("Weather forecast injected.")

    preds = {target: float(model.predict(row)[0]) for target, model in models.items()}

    return pd.DataFrame({
        "date":     [anchor_date + pd.Timedelta(days=h) for h in range(1, 6)],
        "flow_m3s": [preds[f"flow_t{h}"] for h in range(1, 6)],
        "level_m":  [preds[f"level_t{h}"] for h in range(1, 6)],
    }, index=range(1, 6))


def forecast_recursive_inference(
    anchor_date: pd.Timestamp,
    X: pd.DataFrame,
    model_path=None,
) -> pd.DataFrame:
    """
    Generate a 5-day forecast using recursive inference (Phase A).

    Loads flow_t1/level_t1 models and rolls them forward 5 steps,
    updating the feature row at each step.
    """
    path   = model_path or MODEL_PATH
    models = _load_models(path, anchor_date)
    row    = X.loc[[anchor_date]]

    future = _load_weather(anchor_date, X)
    if not future.empty:
        row = _inject_weather_forecast(row, future)

    preds = predict_recursive(
        models     = models,
        anchor_row = row,
        flow_obs   = X["flow_m3s"],
        level_obs  = X["level_m"],
        weather_df = future,
        anchor_date= anchor_date,
    )

    return pd.DataFrame({
        "date":     [anchor_date + pd.Timedelta(days=h) for h in range(1, 6)],
        "flow_m3s": [p["flow_m3s"] for p in preds],
        "level_m":  [p["level_m"]  for p in preds],
    }, index=range(1, 6))


def print_forecast(
    result: pd.DataFrame,
    anchor_date: pd.Timestamp,
    X: pd.DataFrame,
) -> None:
    """Print the 5-day forecast table with observed context."""
    W = 60
    print()
    print("═" * W)
    print("  5-day forecast — Station 043301 (Des Prairies)")
    print(f"  Anchor: {anchor_date.date()}")
    print("═" * W)

    # Check if actuals exist for the forecast window (past forecast validation)
    forecast_dates = result["date"].tolist()
    has_actuals = all(d in X.index for d in forecast_dates)

    if has_actuals:
        print(f"  {'Day':<5}  {'Date':<12}  {'Flow fcst':>10}  {'Flow obs':>10}  {'Lvl fcst':>9}  {'Lvl obs':>9}")
        print(f"  {'-'*5}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*9}")
        for h, row in result.iterrows():
            obs_flow  = X.loc[row["date"], "flow_m3s"]
            obs_level = X.loc[row["date"], "level_m"]
            print(
                f"  t+{h:<3}  {row['date'].strftime('%Y-%m-%d')}  "
                f"{row['flow_m3s']:>10.1f}  {obs_flow:>10.1f}  "
                f"{row['level_m']:>9.3f}  {obs_level:>9.3f}"
            )
        print(f"\n  Units: flow = m³/s, level = m  |  obs = observed actuals")
    else:
        print(f"  {'Day':<5}  {'Date':<12}  {'Flow (m³/s)':>12}  {'Level (m)':>10}")
        print(f"  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*10}")
        for h, row in result.iterrows():
            print(
                f"  t+{h:<3}  {row['date'].strftime('%Y-%m-%d')}  "
                f"{row['flow_m3s']:>12.1f}  {row['level_m']:>10.3f}"
            )
        print(f"\n  Units: flow = m³/s, level = m")

    print("═" * W)

    # Show observed values on the anchor date for context
    obs_flow  = X.loc[anchor_date, "flow_m3s"]
    obs_level = X.loc[anchor_date, "level_m"]
    print(f"  Observed on {anchor_date.date()}: flow = {obs_flow:.1f} m³/s, level = {obs_level:.3f} m")
    print()


def print_recursive_comparison(
    result_direct: pd.DataFrame,
    result_recursive: pd.DataFrame,
    anchor_date: pd.Timestamp,
) -> None:
    """Print direct vs recursive forecast side by side."""
    W = 74
    print()
    print("═" * W)
    print("  Direct vs Recursive inference — Station 043301")
    print(f"  Anchor: {anchor_date.date()}")
    print("═" * W)
    print(f"  {'Day':<5}  {'Date':<12}  {'Direct-F':>9}  {'Recur-F':>9}  "
          f"{'Δ-F':>7}  {'Direct-L':>9}  {'Recur-L':>9}  {'Δ-L':>7}")
    print(f"  {'-'*5}  {'-'*12}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*7}")
    for h in range(1, 6):
        df = result_direct.loc[h, "flow_m3s"]
        rf = result_recursive.loc[h, "flow_m3s"]
        dl = result_direct.loc[h, "level_m"]
        rl = result_recursive.loc[h, "level_m"]
        date = result_direct.loc[h, "date"].strftime("%Y-%m-%d")
        print(f"  t+{h:<3}  {date}  {df:>9.1f}  {rf:>9.1f}  "
              f"{rf-df:>+7.1f}  {dl:>9.3f}  {rl:>9.3f}  {rl-dl:>+7.3f}")
    print(f"\n  Units: flow = m³/s (F), level = m (L)  |  Δ = recursive − direct")
    print("═" * W)
    print()


def _doy_stats(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-DOY historical min, max, and mean for flow and level.

    Returns a DataFrame indexed 1–365 with MultiIndex columns
    (flow_m3s|level_m) × (min|max|mean).
    DOY 366 (leap Feb-29) is filled with DOY 365 values.
    """
    import numpy as np
    doy = np.clip(X.index.day_of_year, 1, 365)
    stats = X.groupby(doy)[["flow_m3s", "level_m"]].agg(["min", "max", "mean"])
    stats = stats.reindex(range(1, 366))  # ensure 1-365, fill any gaps
    stats = stats.ffill().bfill()
    return stats


def plot_forecast(
    result: pd.DataFrame,
    anchor_date: pd.Timestamp,
    X: pd.DataFrame,
    days_back: int = 365,
    docs_name: str = "forecast.png",
    figsize: tuple = (14, 8),
) -> Path:
    """
    Generate and save a 2-panel chart (flow + level) showing:
      - Past `days_back` days of observed data
      - Historical min/max envelope and mean per day-of-year
      - 5-day forecast

    Parameters
    ----------
    days_back  : how many days of observed history to show (default 365)
    docs_name  : filename to write under docs/ (default "forecast.png")

    Returns the path to the saved PNG.
    """
    # ── Data prep ────────────────────────────────────────────────────────────
    obs_start = anchor_date - pd.Timedelta(days=days_back)
    obs = X.loc[obs_start:anchor_date, ["flow_m3s", "level_m"]]

    stats = _doy_stats(X)

    # Map DOY stats to every date in the plot window
    all_dates = pd.date_range(obs_start, result["date"].iloc[-1], freq="D")
    import numpy as np
    doys = np.clip(all_dates.day_of_year, 1, 365)
    env = stats.loc[doys].copy()
    env.index = all_dates

    # ── Theme ─────────────────────────────────────────────────────────────────
    _BG     = "#07101f"   # page background
    _SURF   = "#0d1a2e"   # axes background (slightly lighter than page)
    _TEXT   = "#dbe8f5"
    _MUTED  = "#4a6280"
    _GRID   = "#132035"
    _FLOW   = "#0ee7c5"   # teal — matches website accent
    _LEVEL  = "#60a5fa"   # sky blue — second panel
    _FC     = "#fb923c"   # orange — forecast line
    _DANGER = "#ff5c7a"   # pink-red — danger threshold

    plt.rcParams.update({
        "figure.facecolor":   _BG,
        "axes.facecolor":     _SURF,
        "axes.edgecolor":     _GRID,
        "axes.labelcolor":    _TEXT,
        "xtick.color":        _MUTED,
        "ytick.color":        _MUTED,
        "text.color":         _TEXT,
        "grid.color":         _GRID,
        "grid.linewidth":     0.8,
        "legend.facecolor":   _BG,
        "legend.edgecolor":   _GRID,
        "legend.labelcolor":  _TEXT,
        "savefig.facecolor":  _BG,
        "savefig.edgecolor":  _BG,
    })

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, (ax_f, ax_l) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(
        f"Station 043301 — Rivière des Prairies\nPrévision du {anchor_date.strftime('%Y-%m-%d')}",
        fontsize=13, color=_TEXT,
    )

    # RMSE per horizon — cold season CV mean (7 folds, 2019–2025, no CGM)
    # Cold season used as conservative upper bound across seasons
    _RMSE = {
        "flow_m3s": [42.77, 63.36, 81.00, 97.20, 109.09],
        "level_m":  [0.06, 0.09, 0.11, 0.12, 0.14],
    }

    for ax, var, unit, obs_color in [
        (ax_f, "flow_m3s",  "Débit (m³/s)", _FLOW),
        (ax_l, "level_m",   "Niveau (m)",   _LEVEL),
    ]:
        lo  = env[(var, "min")]
        hi  = env[(var, "max")]
        avg = env[(var, "mean")]

        # Historical envelope
        ax.fill_between(env.index, lo, hi, color=obs_color, alpha=0.08, label="Min/max hist.")
        ax.plot(env.index, avg, color=obs_color, lw=1, alpha=0.40, linestyle="--", label="Moyenne hist.")

        # Observed history
        ax.plot(obs.index, obs[var], color=obs_color, lw=1.8, label="Observé")

        # 5-day forecast — connect last observed point for continuity
        bridge_dates  = [anchor_date] + result["date"].tolist()
        bridge_values = [obs[var].iloc[-1]] + result[var].tolist()
        ax.plot(bridge_dates, bridge_values, color=_FC, lw=2,
                linestyle="--", marker="o", markersize=5, zorder=5, label="Prévision")

        # Confidence bands (±1 RMSE per horizon)
        fc_dates  = result["date"].tolist()
        fc_values = result[var].tolist()
        rmse      = _RMSE[var]
        band_lo   = [v - r for v, r in zip(fc_values, rmse)]
        band_hi   = [v + r for v, r in zip(fc_values, rmse)]
        # Extend band from anchor (zero uncertainty) to first forecast point
        ax.fill_between(
            [anchor_date] + fc_dates,
            [obs[var].iloc[-1]] + band_lo,
            [obs[var].iloc[-1]] + band_hi,
            color=_FC, alpha=0.15, zorder=4, label="±1 RMSE",
        )

        # Danger zone line (level panel only)
        if var == "level_m":
            ax.axhline(22.5, color=_DANGER, lw=1.2, linestyle=":",
                       alpha=0.7, label="Zone de danger (22.5 m)", zorder=4)

        # "Today" marker
        ax.axvline(anchor_date, color=_MUTED, lw=0.8, linestyle=":")

        ax.set_ylabel(unit, fontsize=10)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.6)
        ax.grid(True)
        ax.margins(x=0.01)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)

    # French month abbreviations
    _FR_MONTHS = ["jan", "fév", "mar", "avr", "mai", "jun",
                  "jul", "aoû", "sep", "oct", "nov", "déc"]

    def _fr_month_fmt(x, pos=None):
        dt = mdates.num2date(x)
        m = _FR_MONTHS[dt.month - 1]
        if days_back <= 60:
            return f"{m} {dt.day:02d}"
        return f"{m}\n{dt.year}"

    # Tick density depends on window length
    if days_back <= 60:
        ax_l.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    else:
        ax_l.xaxis.set_major_locator(mdates.MonthLocator())
    ax_l.xaxis.set_major_formatter(plt.FuncFormatter(_fr_month_fmt))

    fig.autofmt_xdate(rotation=0, ha="center")
    plt.tight_layout()

    out_path = Path(f"forecast_{anchor_date.date()}.png")
    sample_path = Path("docs") / docs_name
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(sample_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved → {out_path}")
    print(f"Sample updated → {sample_path}")

    import subprocess, sys as _sys
    if _sys.platform == "darwin":
        subprocess.Popen(["open", str(out_path)])

    return out_path


def append_forecast_history(result: pd.DataFrame, anchor_date: pd.Timestamp) -> Path:
    """
    Append (or update) today's forecast in docs/forecast_history.json.

    The file is a JSON array ordered chronologically. If an entry for
    anchor_date already exists (e.g. a second daily run), it is replaced
    so the latest prediction is kept.
    """
    import json
    from datetime import datetime, UTC

    history_path = Path("docs/forecast_history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)

    if history_path.exists():
        entries = json.loads(history_path.read_text())
    else:
        entries = []

    entry = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "anchor_date": anchor_date.strftime("%Y-%m-%d"),
        "forecast": [
            {
                "day": int(h),
                "date": row["date"].strftime("%Y-%m-%d"),
                "flow_m3s": round(row["flow_m3s"], 1),
                "level_m": round(row["level_m"], 3),
            }
            for h, row in result.iterrows()
        ],
    }

    # Replace existing entry for this anchor_date, or append
    anchor_str = entry["anchor_date"]
    idx = next((i for i, e in enumerate(entries) if e["anchor_date"] == anchor_str), None)
    if idx is not None:
        entries[idx] = entry
    else:
        entries.append(entry)

    history_path.write_text(json.dumps(entries, indent=2))
    print(f"Forecast history updated → {history_path} ({len(entries)} entries)")
    return history_path


def save_forecast_json(
    result: pd.DataFrame,
    anchor_date: pd.Timestamp,
    recursive: pd.DataFrame | None = None,
) -> Path:
    """Save the 5-day forecast as JSON to docs/forecast.json."""
    import json
    from datetime import datetime, UTC

    out_path = Path("docs/forecast.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _rows(df):
        return [
            {
                "day":      int(h),
                "date":     row["date"].strftime("%Y-%m-%d"),
                "flow_m3s": round(row["flow_m3s"], 1),
                "level_m":  round(row["level_m"], 3),
            }
            for h, row in df.iterrows()
        ]

    data = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "anchor_date":  anchor_date.strftime("%Y-%m-%d"),
        "forecast":     _rows(result),
    }
    if recursive is not None:
        data["recursive"] = _rows(recursive)

    out_path.write_text(json.dumps(data, indent=2))
    print(f"Forecast JSON saved → {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 5-day forecast for station 043301.")
    parser.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        default=None,
        help="Anchor date to forecast from (default: latest available)",
    )
    args = parser.parse_args()

    print("Loading features...")
    X, _ = build_dataset(drop_incomplete=False)

    if args.date is None:
        anchor = X.index[-1]
        print(f"No date specified — using latest available: {anchor.date()}")
    else:
        anchor = pd.Timestamp(args.date)
        if anchor not in X.index:
            print(
                f"Error: {args.date} is not in the feature dataset.\n"
                f"Available range: {X.index[0].date()} – {X.index[-1].date()}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Direct forecast
    result = forecast(anchor, X)
    print_forecast(result, anchor, X)

    # Recursive inference (Phase A — always runs for comparison)
    print("\nRunning recursive inference...")
    result_rec = forecast_recursive_inference(anchor, X)
    print_recursive_comparison(result, result_rec, anchor)

    plot_forecast(result, anchor, X, days_back=365, docs_name="forecast.png",    figsize=(14, 8))
    plot_forecast(result, anchor, X, days_back=30,  docs_name="forecast_30d.png", figsize=(7, 8))
    save_forecast_json(result, anchor, recursive=result_rec)
    append_forecast_history(result, anchor)


if __name__ == "__main__":
    main()
