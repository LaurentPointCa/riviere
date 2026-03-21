"""
Flood detection evaluation framework for the Des Prairies forecast model.

Walk-forward evaluation on flood years (2017, 2019, 2023):
  - For each year Y, trains a cold-season model on all data before Y (honest OOS)
  - Reports precision/recall at each horizon t+1..t+5 for the concern threshold
  - Reports lead time: days before first threshold crossing that model first alerted

Primary threshold: 2500 m³/s (residents concerned)
Secondary threshold: 3000 m³/s (near historical flood)

Usage:
    python src/evaluate_flood.py                   # MSE baseline, 2500 threshold
    python src/evaluate_flood.py --alpha 0.85      # quantile model
    python src/evaluate_flood.py --threshold 3000
    python src/evaluate_flood.py --years 2019 2023
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).parent))
from features import build_dataset
from model import _season_mask, COLD_MONTHS, LGBM_PARAMS

FLOW_TARGETS       = [f"flow_t{h}" for h in range(1, 6)]
HORIZONS           = list(range(1, 6))
DEFAULT_THRESHOLD  = 2500   # m³/s — residents start to feel concerned
DEFAULT_EVAL_YEARS = [2017, 2019, 2023]


# ── Model training ───────────────────────────────────────────────────────────

def _train_cold(X_tr: pd.DataFrame, y_tr: pd.DataFrame,
                alpha: float | None = None) -> dict:
    """
    Train cold-season flow models (t+1..t+5 only).
    alpha=None  → MSE / regression (default)
    alpha=float → quantile regression at that quantile level
    """
    params = {**LGBM_PARAMS}
    if alpha is not None:
        params.update({"objective": "quantile", "alpha": alpha, "metric": "quantile"})

    models = {}
    for target in FLOW_TARGETS:
        m = LGBMRegressor(**params)
        m.fit(X_tr, y_tr[target])
        models[target] = m
    return models


# ── Prediction ───────────────────────────────────────────────────────────────

def predict_year(
    year: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    alpha: float | None = None,
) -> pd.DataFrame:
    """
    Train on all cold-season data before `year`, predict cold season of `year`.

    Returns DataFrame indexed by date with columns:
      flow_today, actual_t{h}, pred_t{h}  for h in 1..5
    """
    cutoff = pd.Timestamp(f"{year - 1}-12-31")
    y0     = pd.Timestamp(f"{year}-01-01")
    y1     = pd.Timestamp(f"{year}-12-31")

    tr_mask = (X.index <= cutoff) & _season_mask(X.index, COLD_MONTHS)
    te_mask = (X.index >= y0) & (X.index <= y1) & _season_mask(X.index, COLD_MONTHS)

    X_tr, y_tr = X[tr_mask], y[tr_mask]
    X_te, y_te = X[te_mask], y[te_mask]

    label = "MSE" if alpha is None else f"quantile α={alpha}"
    print(f"  [{year}] {label} | train={len(X_tr):,} cold-season rows"
          f" | test={len(X_te)} rows")

    models = _train_cold(X_tr, y_tr, alpha)

    df = pd.DataFrame(index=X_te.index)
    df["flow_today"] = X_te["flow_m3s"].values
    for h in HORIZONS:
        df[f"actual_t{h}"] = y_te[f"flow_t{h}"].values
        df[f"pred_t{h}"]   = models[f"flow_t{h}"].predict(X_te)
    return df


# ── Precision / Recall ───────────────────────────────────────────────────────

def threshold_metrics(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Precision / recall / F1 at `threshold` for each horizon.

    For horizon h at date D:
      positive  = actual flow at D+h exceeds threshold   (actual_th)
      predicted = model predicts flow at D+h > threshold (pred_th)
    """
    rows = []
    for h in HORIZONS:
        actual_pos = df[f"actual_t{h}"] > threshold
        pred_pos   = df[f"pred_t{h}"]   > threshold

        tp = int((actual_pos & pred_pos).sum())
        fp = int((~actual_pos & pred_pos).sum())
        fn = int((actual_pos & ~pred_pos).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1   = (2 * prec * rec / (prec + rec)
                if not np.isnan(prec) and not np.isnan(rec) and (prec + rec) > 0
                else float("nan"))

        rows.append({
            "horizon":   f"t+{h}",
            "n_actual":  int(actual_pos.sum()),
            "n_pred":    int(pred_pos.sum()),
            "TP":        tp,
            "FP":        fp,
            "FN":        fn,
            "precision": prec,
            "recall":    rec,
            "F1":        f1,
        })
    return pd.DataFrame(rows).set_index("horizon")


# ── Lead time ────────────────────────────────────────────────────────────────

def lead_time_events(df: pd.DataFrame, threshold: float) -> list[dict]:
    """
    For each event onset (flow_today first crosses threshold after being below),
    find the earliest advance warning the model gave.

    For onset date E, checks whether pred_t{h}[E - h days] > threshold,
    starting from h=5 downward. Reports the largest h for which the model
    raised an alert — that is the lead time in days.

    Returns list of dicts: onset, flow_today, lead_days
    """
    flow  = df["flow_today"]
    above = flow > threshold
    events = []

    for i in range(1, len(flow)):
        # Onset: first day above threshold after at least one day below
        if not (above.iloc[i] and not above.iloc[i - 1]):
            continue

        onset = flow.index[i]

        # Search from largest to smallest horizon for earliest warning
        lead = 0
        for h in HORIZONS[::-1]:  # 5 → 1
            lookback = onset - pd.Timedelta(days=h)
            if lookback not in df.index:
                continue
            if df.loc[lookback, f"pred_t{h}"] > threshold:
                lead = h
                break

        events.append({
            "onset":      onset.date(),
            "flow_today": int(round(float(flow.iloc[i]))),
            "lead_days":  lead,
        })

    return events


# ── Display ──────────────────────────────────────────────────────────────────

def _fmt(val: float, digits: int = 3) -> str:
    return f"{val:.{digits}f}" if not np.isnan(val) else "  —  "


def print_year_results(
    year: int,
    metrics: pd.DataFrame,
    events: list[dict],
    threshold: float,
) -> None:
    w = 67
    print(f"\n{'═' * w}")
    print(f"  Year {year} — threshold {threshold:,} m³/s")
    print(f"{'═' * w}")
    print(f"  {'Horizon':<8}  {'n_actual':>8}  {'n_pred':>7}  "
          f"{'TP':>4}  {'FP':>5}  {'FN':>4}  "
          f"{'Precision':>10}  {'Recall':>8}  {'F1':>6}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*4}  {'-'*5}  {'-'*4}  "
          f"{'-'*10}  {'-'*8}  {'-'*6}")

    for idx, row in metrics.iterrows():
        print(f"  {idx:<8}  {int(row['n_actual']):>8}  {int(row['n_pred']):>7}  "
              f"{int(row['TP']):>4}  {int(row['FP']):>5}  {int(row['FN']):>4}  "
              f"{_fmt(row['precision']):>10}  {_fmt(row['recall']):>8}  "
              f"{_fmt(row['F1']):>6}")

    print(f"\n  Event onsets (flow_today crossing {threshold:,} m³/s):")
    if events:
        for e in events:
            if e["lead_days"] > 0:
                warning = f"{e['lead_days']}-day advance warning"
            else:
                warning = "no advance warning within t+5"
            print(f"    {e['onset']}  flow={e['flow_today']:,} m³/s  →  {warning}")
    else:
        print(f"    (none — flow never crossed {threshold:,} m³/s in cold season)")


def print_summary(all_metrics: dict[int, pd.DataFrame]) -> None:
    """Aggregate precision/recall across all eval years."""
    print(f"\n{'═' * 67}")
    print(f"  Aggregated across all evaluation years")
    print(f"{'═' * 67}")
    print(f"  {'Horizon':<8}  {'tot_actual':>10}  {'tot_pred':>9}  "
          f"{'TP':>5}  {'FP':>6}  {'FN':>5}  "
          f"{'Precision':>10}  {'Recall':>8}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*9}  {'-'*5}  {'-'*6}  {'-'*5}  "
          f"{'-'*10}  {'-'*8}")

    for h in HORIZONS:
        idx = f"t+{h}"
        n_actual = sum(m.loc[idx, "n_actual"] for m in all_metrics.values())
        n_pred   = sum(m.loc[idx, "n_pred"]   for m in all_metrics.values())
        tp = sum(int(m.loc[idx, "TP"]) for m in all_metrics.values())
        fp = sum(int(m.loc[idx, "FP"]) for m in all_metrics.values())
        fn = sum(int(m.loc[idx, "FN"]) for m in all_metrics.values())

        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

        print(f"  {idx:<8}  {n_actual:>10}  {n_pred:>9}  "
              f"{tp:>5}  {fp:>6}  {fn:>5}  "
              f"{_fmt(prec):>10}  {_fmt(rec):>8}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flood detection evaluation — walk-forward, out-of-sample."
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=DEFAULT_EVAL_YEARS,
        help="Years to evaluate (default: 2017 2019 2023)",
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Quantile alpha, e.g. 0.85. Omit for MSE/regression.",
    )
    parser.add_argument(
        "--threshold", type=int, default=DEFAULT_THRESHOLD,
        help=f"Flow threshold in m³/s (default: {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args()

    label = "MSE (regression)" if args.alpha is None else f"quantile α={args.alpha}"
    print(f"\nFlood detection evaluation  [{label}]")
    print(f"Threshold : {args.threshold:,} m³/s")
    print(f"Years     : {args.years}")
    print(f"\nLoading dataset...")
    X, y = build_dataset()
    print(f"  {X.shape}  |  {X.index[0].date()} → {X.index[-1].date()}")

    print(f"\nWalk-forward predictions:")
    all_metrics: dict[int, pd.DataFrame] = {}

    for year in args.years:
        df      = predict_year(year, X, y, alpha=args.alpha)
        metrics = threshold_metrics(df, args.threshold)
        events  = lead_time_events(df, args.threshold)
        all_metrics[year] = metrics
        print_year_results(year, metrics, events, args.threshold)

    if len(args.years) > 1:
        print_summary(all_metrics)


if __name__ == "__main__":
    main()
