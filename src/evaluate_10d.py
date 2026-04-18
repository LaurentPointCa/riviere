"""
Evaluate the 10-horizon seasonal model against persistence and climatology baselines.

Reports per-horizon RMSE and skill score for flow and level, by season, on the
held-out test set (last 2 years). Skill > 0 means better than the reference.

Usage:
    python src/evaluate_10d.py                       # vs persistence (default)
    python src/evaluate_10d.py --ref climatology     # vs day-of-year mean
    python src/evaluate_10d.py --model models/lgbm_forecast_10d.pkl
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import build_dataset
from model import (
    load_model, time_split, evaluate,
    COLD_MONTHS, WARM_MONTHS, TARGET_COLS,
    _season_mask,
)

HORIZONS = list(range(1, 11))


def _climatology_preds(X: pd.DataFrame, X_train: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-horizon climatology: long-term mean of the target by day-of-year from training."""
    # For flow_th at date d, the reference is the DOY mean of flow on date d+h.
    preds: dict[str, np.ndarray] = {}
    flow_by_doy  = X_train.groupby(X_train.index.day_of_year)["flow_m3s"].mean()
    level_by_doy = X_train.groupby(X_train.index.day_of_year)["level_m"].mean()

    for h in HORIZONS:
        future_dates = X.index + pd.Timedelta(days=h)
        doys = np.clip(future_dates.day_of_year, 1, 365)
        preds[f"flow_t{h}"]  = flow_by_doy.reindex(doys).values
        preds[f"level_t{h}"] = level_by_doy.reindex(doys).values

    return preds


def _persistence_preds(X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Persistence: flow/level stays the same as the anchor day."""
    flow_today  = X["flow_m3s"].values
    level_today = X["level_m"].values
    preds: dict[str, np.ndarray] = {}
    for h in HORIZONS:
        preds[f"flow_t{h}"]  = flow_today
        preds[f"level_t{h}"] = level_today
    return preds


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def evaluate_season(models: dict, X_te: pd.DataFrame, y_te: pd.DataFrame,
                    ref_preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        for var in ("flow", "level"):
            tgt = f"{var}_t{h}"
            y_true  = y_te[tgt].values
            y_model = models[tgt].predict(X_te)
            y_ref   = ref_preds[tgt]

            rmse_m = _rmse(y_true, y_model)
            rmse_r = _rmse(y_true, y_ref)
            skill  = 1.0 - rmse_m / rmse_r if rmse_r > 0 else float("nan")

            rows.append({
                "h":         h,
                "var":       var,
                "rmse_model": rmse_m,
                "rmse_ref":   rmse_r,
                "skill":      skill,
            })
    return pd.DataFrame(rows)


def _print_table(df: pd.DataFrame, season_label: str, ref_name: str) -> None:
    print(f"\n── {season_label} (vs {ref_name}) ─────────────────────────────────────")
    print(f"  {'h':>3}  {'flow RMSE':>10}  {'flow ref':>10}  {'flow skill':>11}   "
          f"{'lvl RMSE':>9}  {'lvl ref':>9}  {'lvl skill':>10}")
    print(f"  {'-'*3}  {'-'*10}  {'-'*10}  {'-'*11}   {'-'*9}  {'-'*9}  {'-'*10}")
    for h in HORIZONS:
        f = df[(df["h"] == h) & (df["var"] == "flow")].iloc[0]
        l = df[(df["h"] == h) & (df["var"] == "level")].iloc[0]
        print(f"  t+{h:<2}  {f['rmse_model']:>10.2f}  {f['rmse_ref']:>10.2f}  {f['skill']:>+11.3f}   "
              f"{l['rmse_model']:>9.4f}  {l['rmse_ref']:>9.4f}  {l['skill']:>+10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/lgbm_forecast_10d.pkl",
                        help="Path to seasonal model pkl")
    parser.add_argument("--ref", choices=("persistence", "climatology", "both"),
                        default="both", help="Reference baseline")
    args = parser.parse_args()

    print("Loading dataset...")
    X, y = build_dataset()
    X_train, X_test, y_train, y_test = time_split(X, y, test_years=2)
    print(f"  Train: {X_train.index[0].date()} → {X_train.index[-1].date()} ({len(X_train)} rows)")
    print(f"  Test:  {X_test.index[0].date()} → {X_test.index[-1].date()} ({len(X_test)} rows)")

    seasonal = load_model(args.model)
    if "cold" not in seasonal:
        print(f"Error: {args.model} is not a seasonal model file.", file=sys.stderr)
        sys.exit(1)

    refs_to_run = ["persistence", "climatology"] if args.ref == "both" else [args.ref]

    for ref_name in refs_to_run:
        print(f"\n{'═'*75}")
        print(f"  Baseline: {ref_name.upper()}")
        print(f"{'═'*75}")

        for season, months in [("cold", COLD_MONTHS), ("warm", WARM_MONTHS)]:
            label = "Nov–May" if season == "cold" else "Jun–Oct"
            mask_te = _season_mask(X_test.index, months)
            X_te = X_test[mask_te]
            y_te = y_test[mask_te]
            if len(X_te) == 0:
                continue

            if ref_name == "persistence":
                refs = _persistence_preds(X_te)
            else:
                refs = _climatology_preds(X_te, X_train)

            df = evaluate_season(seasonal[season], X_te, y_te, refs)
            _print_table(df, f"{season.upper()} ({label})", ref_name)

    print()


if __name__ == "__main__":
    main()
