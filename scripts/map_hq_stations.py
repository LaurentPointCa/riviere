#!/usr/bin/env python3
"""Plot all Hydro-Québec hydrometric stations on a map.

Highlights every station that falls inside the Ottawa River drainage basin
polygon (`data/basin_boundary.geojson`) — these are the targets of the HQ
historical-data request. Outputs a high-res PNG with a zoomed inset over the
basin so individual labels stay legible.
"""
import json
import math

import contextily as cx
import matplotlib.pyplot as plt
import requests
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Point, shape

API = "https://donnees.hydroquebec.com/api/explore/v2.1/catalog/datasets/donnees-hydrometriques/records"
BASIN_PATH = "data/basin_boundary.geojson"

# Manual label offsets for crowded clusters (xy in points).
# Any offset of magnitude > 15 gets a leader line.
LABEL_OFFSET = {
    "3-66": (8, -14),    # Chelsea — overlaps Rapides-Farmers
    "3-67": (8, 8),      # Rapides-Farmers
    # Western Ottawa cascade — three stations within ~10 km
    "3-31": (28, -2),    # Rapides-des-Quinze (eastmost of the cluster) → push right
    "3-32": (28, -22),   # Rapides-des-Îles (middle)                    → push right + down
    "3-33": (-90, 18),   # Première-Chute (westmost)                    → push left + up
}


def fetch_stations():
    r = requests.get(
        API,
        params={
            "select": "identifiant,nom,regionqc,xcoord,ycoord",
            "group_by": "identifiant,nom,regionqc,xcoord,ycoord",
            "limit": 100,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["results"]


def load_basin():
    """Load the Ottawa River basin polygon (WGS84)."""
    with open(BASIN_PATH) as f:
        gj = json.load(f)
    return shape(gj["features"][0]["geometry"])


def lonlat_to_mercator(lon, lat):
    """WGS84 → Web Mercator (EPSG:3857) meters."""
    R = 6378137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def draw_basin(ax, basin):
    """Draw the basin polygon on a Web-Mercator-projected matplotlib axis."""
    # basin is a shapely Polygon with (lon, lat) tuples
    exterior = [lonlat_to_mercator(lon, lat) for lon, lat in basin.exterior.coords]
    patch = MplPolygon(
        exterior,
        closed=True,
        facecolor="#1a9850",
        edgecolor="#0c5e30",
        alpha=0.13,
        linewidth=1.4,
        zorder=2,
    )
    ax.add_patch(patch)


def plot_panel(ax, stations, basin, *, label_others, title):
    """Plot all stations + basin on ax. label_others=True labels every station."""
    draw_basin(ax, basin)

    in_basin = {s["identifiant"] for s in stations if basin.contains(Point(s["xcoord"], s["ycoord"]))}
    others = [s for s in stations if s["identifiant"] not in in_basin]

    for s in others:
        x, y = lonlat_to_mercator(s["xcoord"], s["ycoord"])
        ax.scatter(x, y, s=22, c="#5a6c80", edgecolor="white", linewidth=0.4, zorder=3)
        if label_others:
            ax.annotate(
                f"{s['nom']}\n{s['identifiant']}",
                xy=(x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=4.2,
                color="#34404f",
                zorder=4,
            )

    # Stations inside the Ottawa basin polygon — green
    for s in stations:
        if s["identifiant"] not in in_basin:
            continue
        x, y = lonlat_to_mercator(s["xcoord"], s["ycoord"])
        ax.scatter(
            x, y, s=140, c="#1a9850", edgecolor="white",
            linewidth=1.6, zorder=5,
        )
        offset = LABEL_OFFSET.get(s["identifiant"], (9, 9))
        large = abs(offset[0]) > 15 or abs(offset[1]) > 15
        kw = dict(
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            color="#0c5e30",
            weight="bold",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1a9850", alpha=0.92),
        )
        if large:
            kw["arrowprops"] = dict(arrowstyle="-", color="#1a9850", lw=0.8, alpha=0.85)
        ax.annotate(f"{s['nom']} ({s['identifiant']})", xy=(x, y), **kw)

    ax.set_axis_off()
    ax.set_title(title, fontsize=12, pad=8)
    return in_basin


def main():
    stations = fetch_stations()
    basin = load_basin()
    print(f"Fetched {len(stations)} stations")
    print(f"Basin loaded: bounds={basin.bounds}, area={basin.area:.2f} sq deg")

    in_basin = sorted(
        [s for s in stations if basin.contains(Point(s["xcoord"], s["ycoord"]))],
        key=lambda s: -s["ycoord"],
    )
    print(f"Stations inside basin: {len(in_basin)}")
    for s in in_basin:
        print(f"  {s['identifiant']:>5}  {s['nom']}")

    fig, (ax_full, ax_zoom) = plt.subplots(
        1, 2, figsize=(20, 12), gridspec_kw={"width_ratios": [1.55, 1.0]}, dpi=200
    )

    # ── Left panel: full Quebec view (all stations + basin polygon)
    plot_panel(
        ax_full,
        stations,
        basin,
        label_others=True,
        title=(
            f"All 93 Hydro-Québec hydrometric stations\n"
            f"green polygon = Ottawa River basin · {len(in_basin)} HQ stations inside"
        ),
    )
    xs = [lonlat_to_mercator(s["xcoord"], s["ycoord"])[0] for s in stations]
    ys = [lonlat_to_mercator(s["xcoord"], s["ycoord"])[1] for s in stations]
    pad_x = (max(xs) - min(xs)) * 0.04
    pad_y = (max(ys) - min(ys)) * 0.04
    ax_full.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax_full.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

    # ── Right panel: zoom on the basin
    plot_panel(
        ax_zoom,
        stations,
        basin,
        label_others=True,
        title="Bassin de la rivière des Outaouais — request targets",
    )
    # Use the basin bounds for the zoom box (with a small margin)
    blon_min, blat_min, blon_max, blat_max = basin.bounds
    bxmin, bymin = lonlat_to_mercator(blon_min, blat_min)
    bxmax, bymax = lonlat_to_mercator(blon_max, blat_max)
    pad = max(bxmax - bxmin, bymax - bymin) * 0.04
    ax_zoom.set_xlim(bxmin - pad, bxmax + pad)
    ax_zoom.set_ylim(bymin - pad, bymax + pad)

    # ── Basemaps
    for ax in (ax_full, ax_zoom):
        try:
            cx.add_basemap(
                ax, source=cx.providers.CartoDB.Positron, crs="EPSG:3857"
            )
        except Exception as e:
            print(f"Basemap fetch failed for one panel: {e}")

    fig.suptitle(
        "Hydro-Québec hydrometric stations · Ottawa River basin overlay",
        fontsize=15,
        weight="bold",
        y=0.995,
    )

    out = "/Users/laurentchouinard/claude/riviere/docs/hq_stations_map.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
