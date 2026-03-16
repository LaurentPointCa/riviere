"""
Plot all historical forecasts vs actual observed levels.
Run from anywhere: python ~/forecast_history_plot.py
"""
import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

RIVIERE = Path.home() / "claude/riviere"
HISTORY = RIVIERE / "docs/forecast_history.json"

sys.path.insert(0, str(RIVIERE / "src"))
from load_data import load_level

# ── Load data ─────────────────────────────────────────────────────────────────
entries = json.loads(HISTORY.read_text())
level_obs = load_level()  # full observed series

# Collect all predicted dates so we can clip the observed window
all_pred_dates = [
    pd.Timestamp(row["date"])
    for e in entries
    for row in e["forecast"]
]
anchor_dates = [pd.Timestamp(e["anchor_date"]) for e in entries]
window_start = min(anchor_dates) - pd.Timedelta(days=5)
window_end   = max(all_pred_dates) + pd.Timedelta(days=2)
obs = level_obs.loc[window_start:window_end]

# ── Theme ──────────────────────────────────────────────────────────────────────
_BG    = "#07101f"
_SURF  = "#0d1a2e"
_TEXT  = "#dbe8f5"
_MUTED = "#4a6280"
_GRID  = "#132035"
_LEVEL = "#60a5fa"

plt.rcParams.update({
    "figure.facecolor":  _BG,
    "axes.facecolor":    _SURF,
    "axes.edgecolor":    _GRID,
    "axes.labelcolor":   _TEXT,
    "xtick.color":       _MUTED,
    "ytick.color":       _MUTED,
    "text.color":        _TEXT,
    "grid.color":        _GRID,
    "grid.linewidth":    0.8,
    "legend.facecolor":  _BG,
    "legend.edgecolor":  _GRID,
    "legend.labelcolor": _TEXT,
    "savefig.facecolor": _BG,
    "savefig.edgecolor": _BG,
})

# Colour ramp: one colour per forecast run (anchor date)
COLOURS = ["#fb923c", "#a78bfa", "#34d399", "#f472b6", "#facc15", "#38bdf8"]

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle("Prévisions vs observations — niveau (m)\nStation 043301, Rivière des Prairies",
             fontsize=12, color=_TEXT)

# Observed line
ax.plot(obs.index, obs.values, color=_LEVEL, lw=2, label="Observé", zorder=5)

# One forecast line per entry
for i, entry in enumerate(entries):
    colour     = COLOURS[i % len(COLOURS)]
    anchor     = pd.Timestamp(entry["anchor_date"])
    anchor_lbl = anchor.strftime("%Y-%m-%d")
    fc_dates   = [pd.Timestamp(r["date"]) for r in entry["forecast"]]
    fc_levels  = [r["level_m"] for r in entry["forecast"]]

    # Bridge from anchor observed value so the line connects visually
    anchor_obs = float(obs.loc[anchor].iloc[0]) if anchor in obs.index else fc_levels[0]
    plot_dates  = [anchor] + fc_dates
    plot_levels = [anchor_obs] + fc_levels

    ax.plot(plot_dates, plot_levels,
            color=colour, lw=1.5, linestyle="--",
            marker="o", markersize=4, zorder=4,
            label=f"Prévision {anchor_lbl}")

    # Mark anchor
    ax.axvline(anchor, color=colour, lw=0.6, linestyle=":", alpha=0.5)

ax.set_ylabel("Niveau (m)", fontsize=10)
ax.legend(fontsize=8, loc="upper left", framealpha=0.6)
ax.grid(True)
ax.margins(x=0.02)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
fig.autofmt_xdate(rotation=0, ha="center")
plt.tight_layout()

out = Path.home() / "forecast_history_plot.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {out}")
