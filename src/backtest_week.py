"""
Backtest all 3 forecast modes over the past 7 anchor dates.

Models compared:
  direct   — lgbm_forecast.pkl  + direct inference  (current production)
  phase_a  — lgbm_forecast.pkl  + recursive inference
  phase_b  — lgbm_recursive.pkl + recursive inference

For each anchor date, each model's t+1…t+5 predictions are compared
against observed actuals.  RMSE and MAE are summarised per horizon.

Usage:
    python src/backtest_week.py
    python src/backtest_week.py --days 14    # extend lookback window
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import build_dataset
from model import load_model, season_for
from predict import (
    MODEL_PATH,
    _load_models,
    predict_recursive,
)


def _direct_forecast(anchor, X, model_path) -> pd.DataFrame:
    """Direct (non-recursive) forecast from anchor date."""
    models = _load_models(model_path, anchor)
    row    = X.loc[[anchor]]
    preds  = {t: float(m.predict(row)[0]) for t, m in models.items()}
    return pd.DataFrame({
        "date":     [anchor + pd.Timedelta(days=h) for h in range(1, 6)],
        "flow_m3s": [preds[f"flow_t{h}"] for h in range(1, 6)],
        "level_m":  [preds[f"level_t{h}"] for h in range(1, 6)],
    }, index=range(1, 6))


def _recursive_forecast(anchor, X, model_path) -> pd.DataFrame:
    """Phase A: roll t1 model forward 5 steps with full row update."""
    models = _load_models(model_path, anchor)
    row    = X.loc[[anchor]]
    preds  = predict_recursive(
        models      = models,
        anchor_row  = row,
        flow_obs    = X["flow_m3s"],
        level_obs   = X["level_m"],
        weather_df  = pd.DataFrame(),   # past dates: use ERA5 proxy (already in row)
        anchor_date = anchor,
    )
    return pd.DataFrame({
        "date":     [anchor + pd.Timedelta(days=h) for h in range(1, 6)],
        "flow_m3s": [p["flow_m3s"] for p in preds],
        "level_m":  [p["level_m"]  for p in preds],
    }, index=range(1, 6))



def run_backtest(X: pd.DataFrame, anchors: list[pd.Timestamp]) -> None:
    modes = [
        ("direct",  lambda a: _direct_forecast(a, X, MODEL_PATH)),
        ("phase_a", lambda a: _recursive_forecast(a, X, MODEL_PATH)),
    ]

    W = 90
    print()
    print("═" * W)
    print("  Backtest — past-week anchors, all 3 forecast modes")
    print(f"  {'direct':12s} = lgbm_forecast.pkl + direct inference (baseline)")
    print(f"  {'phase_a':12s} = lgbm_forecast.pkl + roll t1 forward 5 steps")
    print("═" * W)

    # Collect per-anchor per-horizon errors
    # errors[mode][var] = list of (horizon, abs_error) tuples
    errors: dict[str, dict[str, list]] = {
        m: {"flow_m3s": [], "level_m": []} for m, _ in modes
    }

    for anchor in sorted(anchors):
        season = season_for(anchor)
        print(f"\n── Anchor {anchor.date()}  [{season}] ────────────────────────────────")

        # Build results dict
        results = {}
        for name, fn in modes:
            try:
                results[name] = fn(anchor)
            except Exception as e:
                print(f"  {name}: ERROR — {e}")

        # Header
        mode_names = list(results.keys())
        header_parts = "".join(f"  {n+'-F':>10}  {n+'-L':>8}" for n in mode_names)
        print(f"  {'t':>3}  {'Date':<12}  {'Obs-F':>9}  {'Obs-L':>8}" + header_parts)
        print(f"  {'-'*3}  {'-'*12}  {'-'*9}  {'-'*8}" +
              "".join(f"  {'-'*10}  {'-'*8}" for _ in mode_names))

        for h in range(1, 6):
            target_date = anchor + pd.Timedelta(days=h)
            if target_date not in X.index:
                print(f"  t+{h}  {target_date.date()}  (no actuals yet)")
                continue
            obs_f = X.loc[target_date, "flow_m3s"]
            obs_l = X.loc[target_date, "level_m"]

            row_parts = f"  t+{h}  {target_date.date()}  {obs_f:>9.1f}  {obs_l:>8.3f}"
            for name, res in results.items():
                pf = res.loc[h, "flow_m3s"]
                pl = res.loc[h, "level_m"]
                row_parts += f"  {pf:>10.1f}  {pl:>8.3f}"
                errors[name]["flow_m3s"].append((h, abs(pf - obs_f)))
                errors[name]["level_m"].append((h, abs(pl - obs_l)))
            print(row_parts)

    # Summary: MAE per horizon per mode
    print()
    print("═" * W)
    print("  MAE by horizon — flow (m³/s)")
    print("═" * W)

    for var, unit in [("flow_m3s", "m³/s"), ("level_m", "m")]:
        print(f"\n  {var} ({unit})")
        hdr = f"  {'Mode':<10}" + "".join(f"  {'t+'+str(h):>7}" for h in range(1, 6)) + "   mean"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for name, _ in modes:
            errs = errors[name][var]
            by_h = {h: [] for h in range(1, 6)}
            for h, e in errs:
                by_h[h].append(e)
            row = f"  {name:<10}"
            mae_vals = []
            for h in range(1, 6):
                mae = np.mean(by_h[h]) if by_h[h] else float("nan")
                mae_vals.append(mae)
                row += f"  {mae:>7.2f}"
            row += f"   {np.nanmean(mae_vals):>6.2f}"
            print(row)

    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7,
                        help="Number of past anchor dates to evaluate (default: 7)")
    args = parser.parse_args()

    print("Loading features...")
    X, _ = build_dataset(drop_incomplete=False)

    # Anchor dates: last N dates that have full t+1..t+5 actuals
    all_with_actuals = X.index[X.index <= X.index[-6]]
    anchors = list(all_with_actuals[-(args.days):])
    print(f"Backtesting {len(anchors)} anchor dates: "
          f"{anchors[0].date()} → {anchors[-1].date()}")

    run_backtest(X, anchors)


if __name__ == "__main__":
    main()
