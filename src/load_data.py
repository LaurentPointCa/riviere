"""
Data loader for CEHQ (Quebec hydrological data).

Historical files (fixed-width .txt):
  - Preamble lines then a header row containing "Station" and "Date"
  - Fixed-width columns: station (0-14), date (15-34), value (35-49), remark (50+)
  - Date format: YYYY/MM/DD
  - One row per day

Live feed (tab-separated):
  - URL: https://www.cehq.gouv.qc.ca/suivihydro/fichier_donnees.asp?NoStation=043301
  - Header row: Date / Heure / Niveau / Débit
  - One row per 15-minute interval, most-recent-first
  - Comma as decimal separator; asterisk suffix on Débit = ice-corrected
"""

import pandas as pd
import requests
from io import StringIO
from pathlib import Path


SOURCES = {
    "flow":  "https://www.cehq.gouv.qc.ca/depot/historique_donnees/fichier/043301_Q.txt",
    "level": "https://www.cehq.gouv.qc.ca/depot/historique_donnees/fichier/043301_N.txt",
    "live":  "https://www.cehq.gouv.qc.ca/suivihydro/fichier_donnees.asp?NoStation=043301",
}

DATA_DIR = Path(__file__).parent.parent / "data"


def fetch_raw(url: str, encoding: str = "latin-1") -> str:
    response = requests.get(url, timeout=30)
    response.encoding = encoding
    return response.text


def parse_cehq_file(text: str, value_col_name: str) -> pd.DataFrame:
    """
    Parse a CEHQ fixed-width data file into a DataFrame.

    Args:
        text: raw file content as a string
        value_col_name: name to give the measurement column (e.g. 'flow_m3s')

    Returns:
        DataFrame with columns [date, <value_col_name>, remark]
    """
    lines = text.splitlines()

    # Find the header row (contains "Station" and "Date"), data starts after it
    data_start = None
    for i, line in enumerate(lines):
        if "Station" in line and "Date" in line:
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError("Could not find header row in file")

    data_lines = lines[data_start:]

    rows = []
    for line in data_lines:
        line = line.rstrip()
        if not line:
            continue
        # Fixed-width slicing per format spec
        station = line[0:15].strip()
        date_str = line[15:35].strip()
        value_str = line[35:50].strip()
        remark = line[50:].strip() if len(line) > 50 else ""

        if not date_str or not value_str:
            continue

        rows.append({
            "station": station,
            "date": pd.to_datetime(date_str, format="%Y/%m/%d"),
            value_col_name: float(value_str),
            "remark": remark,
        })

    df = pd.DataFrame(rows)
    df = df.set_index("date").sort_index()
    return df


def _load_cehq(key: str, filename: str, value_col_name: str, cache: bool) -> pd.DataFrame:
    cache_path = DATA_DIR / filename

    if cache and cache_path.exists():
        print(f"Loading from cache: {cache_path}")
        text = cache_path.read_text(encoding="latin-1")
    else:
        print(f"Downloading: {SOURCES[key]}")
        text = fetch_raw(SOURCES[key])
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="latin-1")
        print(f"Cached to: {cache_path}")

    df = parse_cehq_file(text, value_col_name=value_col_name)
    print(f"Loaded {len(df):,} rows from {df.index.min().date()} to {df.index.max().date()}")
    return df


def load_live() -> pd.DataFrame:
    """
    Fetch the live 15-minute feed for station 043301.

    Returns a DataFrame indexed by datetime (ascending) with columns:
      level_m       - water level in metres
      flow_m3s      - flow in m³/s
      ice_corrected - True when the original value had an asterisk
    """
    text = fetch_raw(SOURCES["live"], encoding="utf-8")
    rows = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        # Remove empty strings from multiple consecutive tabs
        parts = [p.strip() for p in parts if p.strip()]
        # Skip header row
        if parts[0] == "Date":
            continue
        if len(parts) < 4:
            continue

        date_str, time_str, level_str, flow_str = parts[0], parts[1], parts[2], parts[3]

        ice_corrected = flow_str.endswith("*")
        flow_str = flow_str.rstrip("*")

        # French locale: non-breaking space as thousands sep, comma as decimal sep
        level_val = float(level_str.replace("\u00a0", "").replace(",", "."))
        flow_val  = float(flow_str.replace("\u00a0", "").replace(",", "."))

        rows.append({
            "datetime":      pd.to_datetime(f"{date_str} {time_str}"),
            "level_m":       level_val,
            "flow_m3s":      flow_val,
            "ice_corrected": ice_corrected,
        })

    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    print(f"Loaded {len(df):,} live rows from {df.index.min()} to {df.index.max()}")
    return df


def load_flow(cache: bool = True) -> pd.DataFrame:
    """Load daily flow data (m³/s) for station 043301."""
    return _load_cehq("flow", "043301_Q.txt", "flow_m3s", cache)


def load_level(cache: bool = True) -> pd.DataFrame:
    """Load daily water level data (m) for station 043301."""
    return _load_cehq("level", "043301_N.txt", "level_m", cache)


if __name__ == "__main__":
    flow = load_flow()
    print(flow.head(3))
    print(f"Missing: {flow.isnull().sum().to_dict()}\n")

    level = load_level()
    print(level.head(3))
    print(f"Missing: {level.isnull().sum().to_dict()}\n")

    live = load_live()
    print(live.head(3))
    print(live.tail(3))
    print(f"Missing: {live.isnull().sum().to_dict()}")
    print(f"Ice-corrected rows: {live['ice_corrected'].sum()}")
