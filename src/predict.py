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
from model import load_model
from load_forecast import load_weather_forecast
from load_cgm import load_cgm_forecast, CGM_COLS

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


def _inject_cgm_forecast(row: pd.DataFrame, cgm_fc: pd.DataFrame) -> pd.DataFrame:
    """Replace CGM perfect-forecast proxy columns with real CGM forecast values."""
    row = row.copy()
    for h in range(1, 6):
        if h > len(cgm_fc):
            break
        fc = cgm_fc.iloc[h - 1]
        for col in CGM_COLS:
            feat = f"{col}_forecast_t{h}"
            if feat in row.columns and col in fc.index:
                row[feat] = fc[col]
    return row


def forecast(anchor_date: pd.Timestamp, X: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a 5-day forecast of flow and level from the given anchor date.

    Parameters
    ----------
    anchor_date : the "today" date to forecast from; must be in X.index
    X           : feature matrix from build_dataset()

    Returns
    -------
    pd.DataFrame with columns date, flow_m3s, level_m; indexed 1..5 (horizon)
    """
    models = load_model(MODEL_PATH)
    row = X.loc[[anchor_date]]

    # For the latest available date, inject real weather forecast.
    # For past dates the ERA5 perfect-forecast proxy is already correct.
    if anchor_date == X.index[-1] and "temp_forecast_t1" in X.columns:
        try:
            wf = load_weather_forecast(days=5)
            future = wf[wf.index > anchor_date].head(5)
            if not future.empty:
                row = _inject_weather_forecast(row, future)
                print("Weather forecast injected.")
        except Exception as e:
            print(f"Warning: could not load weather forecast ({e}); using ERA5 proxy.")

        try:
            cgm_fc = load_cgm_forecast(n_days=5)
            future_cgm = cgm_fc[cgm_fc.index > anchor_date].head(5)
            if not future_cgm.empty:
                row = _inject_cgm_forecast(row, future_cgm)
                print("CGM upstream forecast injected.")
        except Exception as e:
            print(f"Warning: could not load CGM forecast ({e}); using proxy.")

    preds = {target: float(model.predict(row)[0]) for target, model in models.items()}

    return pd.DataFrame({
        "date":     [anchor_date + pd.Timedelta(days=h) for h in range(1, 6)],
        "flow_m3s": [preds[f"flow_t{h}"] for h in range(1, 6)],
        "level_m":  [preds[f"level_t{h}"] for h in range(1, 6)],
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

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, (ax_f, ax_l) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(
        f"Station 043301 — Rivière des Prairies\nForecast from {anchor_date.strftime('%Y-%m-%d')}",
        fontsize=13,
    )

    for ax, var, unit, color in [
        (ax_f, "flow_m3s",  "Flow (m³/s)", "steelblue"),
        (ax_l, "level_m",   "Level (m)",   "teal"),
    ]:
        lo  = env[(var, "min")]
        hi  = env[(var, "max")]
        avg = env[(var, "mean")]

        # Historical envelope
        ax.fill_between(env.index, lo, hi, color=color, alpha=0.10, label="Hist. min/max")
        ax.plot(env.index, avg, color=color, lw=1, alpha=0.5, linestyle="--", label="Hist. mean")

        # Observed history
        ax.plot(obs.index, obs[var], color=color, lw=1.8, label="Observed")

        # 5-day forecast — connect last observed point for continuity
        bridge_dates  = [anchor_date] + result["date"].tolist()
        bridge_values = [obs[var].iloc[-1]] + result[var].tolist()
        ax.plot(bridge_dates, bridge_values, color="crimson", lw=2,
                linestyle="--", marker="o", markersize=5, zorder=5, label="Forecast")

        # "Today" marker
        ax.axvline(anchor_date, color="gray", lw=0.8, linestyle=":")

        ax.set_ylabel(unit, fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.01)

    # Tick density depends on window length
    if days_back <= 60:
        ax_l.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
        ax_l.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    else:
        ax_l.xaxis.set_major_locator(mdates.MonthLocator())
        ax_l.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

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


def save_forecast_json(result: pd.DataFrame, anchor_date: pd.Timestamp) -> Path:
    """Save the 5-day forecast as JSON to docs/forecast.json."""
    import json
    from datetime import datetime, UTC

    out_path = Path("docs/forecast.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
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

    result = forecast(anchor, X)
    print_forecast(result, anchor, X)
    plot_forecast(result, anchor, X, days_back=365, docs_name="forecast.png",    figsize=(14, 8))
    plot_forecast(result, anchor, X, days_back=30,  docs_name="forecast_30d.png", figsize=(7, 8))
    save_forecast_json(result, anchor)


if __name__ == "__main__":
    main()
