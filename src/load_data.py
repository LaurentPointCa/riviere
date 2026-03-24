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
    "flow":           "https://www.cehq.gouv.qc.ca/depot/historique_donnees/fichier/043301_Q.txt",
    "level":          "https://www.cehq.gouv.qc.ca/depot/historique_donnees/fichier/043301_N.txt",
    "live":           "https://www.cehq.gouv.qc.ca/suivihydro/fichier_donnees.asp?NoStation=043301",
    "upstream_level": "https://www.cehq.gouv.qc.ca/depot/historique_donnees/fichier/043108_N.txt",
    "upstream_live":  "https://www.cehq.gouv.qc.ca/suivihydro/fichier_donnees.asp?NoStation=043108",
}

# MSC GeoMet API — daily hydrometric data (works for any station)
ECCC_DAILY_URL = (
    "https://api.weather.gc.ca/collections/hydrometric-daily-mean/items"
    "?STATION_NUMBER={station}&datetime={start}/{end}&limit=500&f=json&sortby=DATE"
)

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
    try:
        text = fetch_raw(SOURCES["live"], encoding="utf-8")
    except requests.exceptions.RequestException as e:
        print(f"Warning: live feed fetch failed ({e}) — skipping.")
        return pd.DataFrame(columns=["level_m", "flow_m3s", "ice_corrected"]).rename_axis("datetime")
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

    if not rows:
        print("Warning: live feed returned no parseable rows — skipping.")
        return pd.DataFrame(columns=["level_m", "flow_m3s", "ice_corrected"]).rename_axis("datetime")

    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    print(f"Loaded {len(df):,} live rows from {df.index.min()} to {df.index.max()}")
    return df


def _live_daily() -> pd.DataFrame:
    """Fetch live 15-min feed and resample to daily means (flow_m3s, level_m)."""
    live = load_live()
    if live.empty:
        return pd.DataFrame(columns=["flow_m3s", "level_m"],
                            index=pd.DatetimeIndex([], name="datetime"))
    return live[["flow_m3s", "level_m"]].resample("D").mean()


def load_flow(cache: bool = True) -> pd.DataFrame:
    """Load daily flow data (m³/s) for station 043301, extended with live feed."""
    hist = _load_cehq("flow", "043301_Q.txt", "flow_m3s", cache)
    live_daily = _live_daily()
    new_rows = live_daily[["flow_m3s"]][live_daily.index > hist.index.max()]
    if not new_rows.empty:
        hist = pd.concat([hist, new_rows])
        print(f"Extended with {len(new_rows)} days from live feed (up to {new_rows.index.max().date()})")
    return hist


def load_level(cache: bool = True) -> pd.DataFrame:
    """Load daily water level data (m) for station 043301, extended with live feed."""
    hist = _load_cehq("level", "043301_N.txt", "level_m", cache)
    hist["level_m"] -= 27.62  # convert historical datum to match live feed
    live_daily = _live_daily()
    new_rows = live_daily[["level_m"]][live_daily.index > hist.index.max()]
    if not new_rows.empty:
        hist = pd.concat([hist, new_rows])
        print(f"Extended with {len(new_rows)} days from live feed (up to {new_rows.index.max().date()})")
    return hist


def _load_upstream_live() -> pd.DataFrame:
    """Fetch live 15-min level feed for upstream station 043108 (Lac des Deux Montagnes)."""
    try:
        text = fetch_raw(SOURCES["upstream_live"], encoding="utf-8")
    except requests.exceptions.RequestException as e:
        print(f"Warning: upstream live feed fetch failed ({e}) — skipping.")
        return pd.DataFrame(columns=["upstream_level_m"]).rename_axis("datetime")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if parts[0] == "Date":
            continue
        if len(parts) < 3:
            continue
        date_str, time_str, level_str = parts[0], parts[1], parts[2]
        level_val = float(level_str.replace("\u00a0", "").replace(",", "."))
        rows.append({
            "datetime":         pd.to_datetime(f"{date_str} {time_str}"),
            "upstream_level_m": level_val,
        })
    if not rows:
        print("Warning: upstream live feed returned no parseable rows — skipping.")
        return pd.DataFrame(columns=["upstream_level_m"]).rename_axis("datetime")

    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    print(f"Loaded {len(df):,} upstream live rows from {df.index.min()} to {df.index.max()}")
    return df


def load_upstream_level(cache: bool = True) -> pd.DataFrame:
    """Load daily level (m) for upstream station 043108 (Lac des Deux Montagnes), extended with live feed."""
    hist = _load_cehq("upstream_level", "043108_N.txt", "upstream_level_m", cache)
    live = _load_upstream_live()
    if live.empty:
        live_daily = pd.DataFrame(columns=["upstream_level_m"],
                                  index=pd.DatetimeIndex([], name="datetime"))
    else:
        live_daily = live[["upstream_level_m"]].resample("D").mean()
    new_rows = live_daily[live_daily.index > hist.index.max()]
    if not new_rows.empty:
        hist = pd.concat([hist, new_rows])
        print(f"Extended with {len(new_rows)} days from upstream live feed (up to {new_rows.index.max().date()})")
    return hist


def _parse_eccc_csv(path: Path, param: int, col_name: str) -> pd.DataFrame:
    """
    Parse an ECCC Water Office CSV file (02KF005.csv format).

    Columns: ID, PARAM, Date, Valeur, SYM
    PARAM=1 → flow (m³/s), PARAM=2 → level (m)
    Date format: YYYY/MM/DD
    First line is a description preamble; second line is the header.
    """
    df = pd.read_csv(path, skiprows=1, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df[df["PARAM"] == param].copy()
    df["date"] = pd.to_datetime(df["Date"].str.strip(), format="%Y/%m/%d")
    df = df.rename(columns={"Valeur": col_name})[["date", col_name]]
    df = df.set_index("date").sort_index()
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    return df


def _fetch_eccc_recent(station: str, start: str, end: str, col_name: str,
                       field: str = "DISCHARGE") -> pd.DataFrame:
    """
    Fetch recent daily hydrometric data from the MSC GeoMet API.

    Parameters
    ----------
    station  : ECCC station number (e.g. '02KF005')
    start    : start date string YYYY-MM-DD
    end      : end date string YYYY-MM-DD
    col_name : output column name
    field    : 'DISCHARGE' or 'LEVEL'
    """
    url = ECCC_DAILY_URL.format(station=station, start=start, end=end)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        features = resp.json().get("features", [])
        if not features:
            return pd.DataFrame(columns=[col_name]).rename_axis("date")
        rows = [
            {"date": pd.to_datetime(f["properties"]["DATE"]),
             col_name: f["properties"][field]}
            for f in features
            if f["properties"].get(field) is not None
        ]
        if not rows:
            return pd.DataFrame(columns=[col_name]).rename_axis("date")
        return pd.DataFrame(rows).set_index("date").sort_index()
    except Exception as e:
        print(f"Warning: could not fetch ECCC recent data for {station} ({e})")
        return pd.DataFrame(columns=[col_name]).rename_axis("date")


def _parse_eccc_xml(path: Path, col_name: str) -> pd.DataFrame:
    """
    Parse an ECCC real-time XML export (02KF005_QRD_*.xml or *_HGD_*.xml).

    Each <record> contains <datestamp> and <value>.
    Returns a DataFrame indexed by date (daily, UTC-naive) with one column.
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    rows = []
    for record in tree.getroot().findall(".//record"):
        ds  = record.findtext("datestamp", "").strip()
        val = record.findtext("value", "").strip()
        if ds and val:
            rows.append({
                "date":    pd.to_datetime(ds[:10]),   # keep date part only
                col_name:  float(val),
            })
    if not rows:
        return pd.DataFrame(columns=[col_name]).rename_axis("date")
    df = pd.DataFrame(rows).set_index("date").sort_index()
    # Daily mean in case of duplicates
    return df[col_name].resample("D").mean().to_frame()


def load_ottawa_flow(cache: bool = True) -> pd.DataFrame:
    """
    Load daily flow (m³/s) for ECCC station 02KF005 (Ottawa River near Bells Corners).

    Reads from the cached CSV file data/02KF005.csv, then extends with the
    ECCC real-time API for dates beyond the file's coverage.
    """
    cache_path = DATA_DIR / "02KF005.csv"
    hist = _parse_eccc_csv(cache_path, param=1, col_name="ottawa_flow_m3s")
    print(f"Loaded {len(hist):,} rows of Ottawa flow from {hist.index.min().date()} to {hist.index.max().date()}")

    # Extend with any XML exports found in data/ (e.g. 02KF005_QRD_*.xml)
    xml_files = sorted(DATA_DIR.glob("02KF005_QRD_*.xml"))
    if xml_files:
        frames = [_parse_eccc_xml(f, "ottawa_flow_m3s") for f in xml_files]
        xml_df = pd.concat(frames).groupby(level=0).mean()  # deduplicate overlapping dates
        new_rows = xml_df[xml_df.index > hist.index.max()]
        if not new_rows.empty:
            hist = pd.concat([hist, new_rows])
            print(f"Extended with {len(new_rows)} days from XML export(s) (up to {new_rows.index.max().date()})")

    # Try ECCC API for anything still missing
    start = (hist.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end   = pd.Timestamp.today().strftime("%Y-%m-%d")
    if start <= end:
        recent = _fetch_eccc_recent("02KF005", start, end, "ottawa_flow_m3s", field="DISCHARGE")
        if not recent.empty:
            hist = pd.concat([hist, recent])
            print(f"Extended with {len(recent)} days from ECCC API (up to {recent.index.max().date()})")

    return hist


def load_hull_level(cache: bool = True) -> pd.DataFrame:
    """
    Load daily water level (m) for ECCC station 02LA015 (Ottawa River at Hull).

    Reads from data/02LA015.csv, extends with any 02LA015_HGD_*.xml exports,
    then tries the MSC GeoMet API for anything still missing.
    """
    cache_path = DATA_DIR / "02LA015.csv"
    hist = _parse_eccc_csv(cache_path, param=2, col_name="hull_level_m")
    print(f"Loaded {len(hist):,} rows of Hull level from {hist.index.min().date()} to {hist.index.max().date()}")

    # Extend with any XML exports found in data/ (e.g. 02LA015_HGD_*.xml)
    xml_files = sorted(DATA_DIR.glob("02LA015_HGD_*.xml"))
    if xml_files:
        frames = [_parse_eccc_xml(f, "hull_level_m") for f in xml_files]
        xml_df = pd.concat(frames).groupby(level=0).mean()
        new_rows = xml_df[xml_df.index > hist.index.max()]
        if not new_rows.empty:
            hist = pd.concat([hist, new_rows])
            print(f"Extended with {len(new_rows)} days from XML export(s) (up to {new_rows.index.max().date()})")

    # Try ECCC API for anything still missing
    start = (hist.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end   = pd.Timestamp.today().strftime("%Y-%m-%d")
    if start <= end:
        recent = _fetch_eccc_recent("02LA015", start, end, "hull_level_m", field="LEVEL")
        if not recent.empty:
            hist = pd.concat([hist, recent])
            print(f"Extended with {len(recent)} days from ECCC API (up to {recent.index.max().date()})")

    return hist


def _parse_rdp_csv(path: Path, col_name: str, level_min: float = 5.0) -> pd.DataFrame:
    """
    Parse a CMM historical CSV export (semicolon-separated, JS Date.toString() format).

    Expected columns: station;date;valeur
    Date format: "Wed Oct 28 2020 15:05:00 GMT-0400 (Eastern Daylight Time)"

    Drops obvious installation-period readings where level < level_min (e.g. sensor
    not yet submerged).  Resamples 5-minute readings to daily means.
    Returns a DataFrame indexed by date (UTC-naive) with one column named col_name.
    """
    df = pd.read_csv(
        path, sep=";", header=0,
        names=["station", "date_str", col_name],
        dtype={"station": str, "date_str": str, col_name: float},
    )

    # Drop installation-period artifacts (sensor not yet submerged)
    df = df[df[col_name] >= level_min].copy()

    # Extract local calendar date from JS Date.toString():
    # "Wed Oct 28 2020 15:05:00 GMT-0400 (Eastern Daylight Time)" → "Wed Oct 28 2020"
    date_part = df["date_str"].str.extract(r"(\w{3} \w{3} \d{1,2} \d{4})")[0]
    df["date"] = pd.to_datetime(date_part, format="%a %b %d %Y")

    daily = df.groupby("date")[col_name].mean().to_frame()
    daily.index.name = "date"
    print(
        f"Loaded {len(df):,} 5-min rows → {len(daily):,} daily means "
        f"from {daily.index.min().date()} to {daily.index.max().date()}"
    )
    return daily


def load_rdp09_level() -> pd.DataFrame:
    """
    Load daily water level (m) for CMM station 39_RDP09 (Rue Marceau, Pierrefonds-Roxboro).

    Source: semicolon-separated historical CSV exports matching data/39_RDP09__*.csv.
    Multiple files are concatenated and deduplicated.
    Returns a DataFrame indexed by date with column 'rdp09_level_m'.
    """
    paths = sorted(DATA_DIR.glob("39_RDP09__*.csv"))
    if not paths:
        raise FileNotFoundError("No 39_RDP09__*.csv file found in data/")
    frames = [_parse_rdp_csv(p, "rdp09_level_m") for p in paths]
    df = pd.concat(frames).groupby(level=0).mean()
    print(f"39_RDP09 combined: {len(df):,} daily rows ({df.index.min().date()} – {df.index.max().date()})")
    return df


def load_rdp11_level() -> pd.DataFrame:
    """
    Load daily water level (m) for CMM station 01_RDP11 (Parc Terrasse-Sacré-Cœur, Île-Bizard).

    Source: semicolon-separated historical CSV exports matching data/01_RDP11__*.csv.
    Multiple files are concatenated and deduplicated.
    Returns a DataFrame indexed by date with column 'rdp11_level_m'.
    """
    paths = sorted(DATA_DIR.glob("01_RDP11__*.csv"))
    if not paths:
        raise FileNotFoundError("No 01_RDP11__*.csv file found in data/")
    frames = [_parse_rdp_csv(p, "rdp11_level_m") for p in paths]
    df = pd.concat(frames).groupby(level=0).mean()
    print(f"01_RDP11 combined: {len(df):,} daily rows ({df.index.min().date()} – {df.index.max().date()})")
    return df


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
