"""
Validation plot for the 10-day experimental model: observed flow/level vs
forecast horizons t+1..t+10. Reads from docs/forecast_10d_history.json.
Saves to docs/forecast_10d_validation.png.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from load_data import load_flow, load_level

MODEL_START    = "2026-04-18"   # first day 10-day experimental model produced history
HORIZONS       = list(range(1, 11))
HISTORY_PATH   = Path(__file__).parent.parent / "docs" / "forecast_10d_history.json"
OUT_PATH       = Path(__file__).parent.parent / "docs" / "forecast_10d_validation.png"

if not HISTORY_PATH.exists():
    print(f"No history yet at {HISTORY_PATH}; skipping 10-day validation plot.")
    sys.exit(0)

with open(HISTORY_PATH) as f:
    history = json.load(f)

pred = {h: [] for h in HORIZONS}
for entry in history:
    anchor = entry["anchor_date"]
    if anchor < MODEL_START:
        continue
    for fc in entry["forecast"]:
        h = fc["day"]
        if h in pred:
            pred[h].append({
                "date":  pd.Timestamp(fc["date"]),
                "flow":  fc["flow_m3s"],
                "level": fc["level_m"],
            })

pred_df = {}
for h in HORIZONS:
    if pred[h]:
        df = pd.DataFrame(pred[h]).drop_duplicates("date").set_index("date").sort_index()
        pred_df[h] = df

flow_obs  = load_flow().loc[MODEL_START:,  "flow_m3s"]
level_obs = load_level().loc[MODEL_START:, "level_m"]

COLORS  = {1: "#E91E63", 2: "#FF6D00", 3: "#FFC107", 4: "#7C4DFF", 5: "#00BCD4",
           6: "#4CAF50", 7: "#9C27B0", 8: "#3F51B5", 9: "#795548", 10: "#607D8B"}
MARKERS = {1: "o",       2: "s",       3: "^",       4: "D",       5: "P",
           6: "v",       7: "X",       8: "*",       9: "h",       10: "<"}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig.suptitle(
    f"Station 043301 — Observed vs 10-day Experimental Forecast (from {MODEL_START})",
    fontsize=13, fontweight="bold",
)

for ax, obs, col, ylabel in [
    (ax1, flow_obs,  "flow",  "Flow (m³/s)"),
    (ax2, level_obs, "level", "Level (m)"),
]:
    ax.plot(obs.index, obs.values, "-", color="#1565C0", lw=2.5, zorder=10, label="Observed")
    ax.scatter(obs.index, obs.values, color="#1565C0", s=35, zorder=11)

    for h in HORIZONS:
        df = pred_df.get(h)
        if df is None or df.empty:
            continue
        ax.plot(df.index, df[col], "--", color=COLORS[h], lw=1.3, alpha=0.75, zorder=5)
        ax.scatter(df.index, df[col],
                   marker=MARKERS[h], color=COLORS[h], s=55, zorder=6, label=f"t+{h}")

    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, ncol=5, framealpha=0.9)

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax2.xaxis.set_major_locator(mdates.DayLocator())
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
