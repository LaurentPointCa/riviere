"""Generate docs/basin_map.png and docs/basin_stations.png."""

import json
from pathlib import Path

import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import pyproj

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"

STATIONS = [
    # CEHQ coordinates from data file headers (NAD83 DMS → decimal)
    # 043301: 45°31'18"N / 73°50'44"W
    (-73.8456, 45.5217, "043301\n(cible)",        ( 8,  4), "red"),
    # 043108: 45°29'20"N / 73°58'41"W
    (-73.9781, 45.4889, "043108\n(amont CEHQ)",   (-55,  6), "steelblue"),
    # CGM stations: coordinates from cruesgrandmontreal.ca GeoJSON API
    (-73.8548, 45.5078, "39_RDP09",               (-44,  -8), "darkorange"),
    (-73.8799, 45.4804, "01_RDP11",               (-44, -14), "darkorange"),
    (-74.0906, 45.4590, "11_LDM01\n(Oka)",        (-50,   4), "darkorange"),
]

PROJ = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def to_wm(lon, lat):
    return PROJ.transform(lon, lat)


def ring_to_wm(ring):
    xs, ys = PROJ.transform([c[0] for c in ring], [c[1] for c in ring])
    return list(zip(xs, ys))


def make_ax(extent_wm, figsize):
    """Return (fig, ax) with correct aspect ratio for a Web Mercator extent."""
    x0, y0, x1, y1 = extent_wm
    dx = x1 - x0
    dy = y1 - y0
    # Enforce 1:1 pixel scale: adjust figsize height to match data aspect
    w, _ = figsize
    h = w * dy / dx
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig, ax


def add_basemap(ax, zoom="auto"):
    try:
        ctx.add_basemap(ax, crs="EPSG:3857",
                        source=ctx.providers.OpenStreetMap.Mapnik,
                        attribution_size=6, zoom=zoom)
    except Exception as e:
        print(f"  Basemap warning: {e}")


def main():
    geojson = json.loads((DATA_DIR / "basin_boundary.geojson").read_text())
    ring = geojson["features"][0]["geometry"]["coordinates"][0]
    wm_ring = ring_to_wm(ring)
    xs = [p[0] for p in wm_ring]
    ys = [p[1] for p in wm_ring]

    st_wm = [(to_wm(lon, lat), lbl, offset, col)
              for lon, lat, lbl, offset, col in STATIONS]

    # ── basin_map.png — full watershed ──────────────────────────────
    pad_x = (max(xs) - min(xs)) * 0.30
    pad_y = (max(ys) - min(ys)) * 0.30
    extent = (min(xs) - pad_x, min(ys) - pad_y,
              max(xs) + pad_x, max(ys) + pad_y)

    fig, ax = make_ax(extent, figsize=(8, 8))

    poly = MplPolygon(list(zip(xs, ys)), closed=True)
    pc = PatchCollection([poly], facecolor="royalblue", alpha=0.22,
                         edgecolor="steelblue", linewidth=1.5)
    ax.add_collection(pc)
    add_basemap(ax, zoom=6)

    # Delineation point
    ox, oy = to_wm(-75.926, 45.426)
    ax.plot(ox, oy, "s", ms=7, color="purple", zorder=5,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.annotate("Point de\ndélineation\ndu bassin", (ox, oy),
                xytext=(8, 8), textcoords="offset points", fontsize=7,
                color="purple",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, lw=0))

    legend_handles = [
        mpatches.Patch(facecolor="royalblue", alpha=0.35, edgecolor="steelblue",
                       label="Bassin versant (mghydro.com)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="purple",
                   markersize=8, label="Point de délineation"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title("Bassin versant — Rivière des Prairies (station 043301)",
                 fontsize=10, pad=6)

    out1 = DOCS_DIR / "basin_map.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out1}")

    # ── basin_stations.png — station detail ─────────────────────────
    inset_lon = (-74.35, -73.55)
    inset_lat = (45.35, 45.65)
    sx0, sy0 = to_wm(inset_lon[0], inset_lat[0])
    sx1, sy1 = to_wm(inset_lon[1], inset_lat[1])
    extent2 = (sx0, sy0, sx1, sy1)

    fig2, ax2 = make_ax(extent2, figsize=(8, 8))

    poly2 = MplPolygon(list(zip(xs, ys)), closed=True)
    pc2 = PatchCollection([poly2], facecolor="royalblue", alpha=0.15,
                          edgecolor="steelblue", linewidth=1.2)
    ax2.add_collection(pc2)
    add_basemap(ax2, zoom=11)

    for (wx, wy), lbl, offset, col in st_wm:
        ax2.plot(wx, wy, "o", ms=6, color=col, zorder=5,
                 markeredgecolor="white", markeredgewidth=0.8)
        ax2.annotate(lbl, (wx, wy), xytext=offset,
                     textcoords="offset points", fontsize=6.5,
                     color=col, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white",
                               alpha=0.78, lw=0))

    legend_handles2 = [
        mpatches.Patch(facecolor="royalblue", alpha=0.25, edgecolor="steelblue",
                       label="Bassin versant (mghydro.com)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                   markersize=9, label="043301 — station cible"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
                   markersize=9, label="043108 — amont CEHQ"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="darkorange",
                   markersize=9, label="Stations CMM (amont)"),
    ]
    ax2.legend(handles=legend_handles2, loc="lower left", fontsize=8, framealpha=0.9)
    ax2.set_title("Stations de mesure — région de Montréal–Laval", fontsize=10, pad=6)

    out2 = DOCS_DIR / "basin_stations.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
