"""
5-day forecast script for station 043301 — Des Prairies.

Usage:
    python src/predict.py                       # forecast from latest available date
    python src/predict.py --date 2025-06-01     # forecast from a specific past date
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import build_dataset
from model import load_model

MODEL_PATH = Path("models/lgbm_forecast.pkl")


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
    X, _ = build_dataset()

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


if __name__ == "__main__":
    main()
