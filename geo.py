"""US geography helpers.

State-level bubbles use the built-in centroid table below (no network, no extra package).
City-level bubbles geocode each City/State through the offline `zipcodes` package, using the
row's postal code first and the city name as a fallback. If `zipcodes` is not installed the
city map is simply unavailable and the app falls back to state level.
"""
from __future__ import annotations

import pandas as pd

try:  # offline zip-code database, bundled with the package
    import zipcodes
except ImportError:  # pragma: no cover - optional dependency
    zipcodes = None

STATE_CENTROIDS = {
    "Alabama": (32.8067, -86.7911), "Alaska": (61.3707, -152.4044), "Arizona": (33.7298, -111.4312),
    "Arkansas": (34.9697, -92.3731), "California": (36.1162, -119.6816), "Colorado": (39.0598, -105.3111),
    "Connecticut": (41.5978, -72.7554), "Delaware": (39.3185, -75.5071),
    "District of Columbia": (38.8974, -77.0268), "Florida": (27.7663, -81.6868),
    "Georgia": (33.0406, -83.6431), "Hawaii": (21.0943, -157.4983), "Idaho": (44.2405, -114.4788),
    "Illinois": (40.3495, -88.9861), "Indiana": (39.8494, -86.2583), "Iowa": (42.0115, -93.2105),
    "Kansas": (38.5266, -96.7265), "Kentucky": (37.6681, -84.6701), "Louisiana": (31.1695, -91.8678),
    "Maine": (44.6939, -69.3819), "Maryland": (39.0639, -76.8021), "Massachusetts": (42.2302, -71.5301),
    "Michigan": (43.3266, -84.5361), "Minnesota": (45.6945, -93.9002), "Mississippi": (32.7416, -89.6787),
    "Missouri": (38.4561, -92.2884), "Montana": (46.9219, -110.4544), "Nebraska": (41.1254, -98.2681),
    "Nevada": (38.3135, -117.0554), "New Hampshire": (43.4525, -71.5639), "New Jersey": (40.2989, -74.5210),
    "New Mexico": (34.8405, -106.2485), "New York": (42.1657, -74.9481), "North Carolina": (35.6301, -79.8064),
    "North Dakota": (47.5289, -99.7840), "Ohio": (40.3888, -82.7649), "Oklahoma": (35.5653, -96.9289),
    "Oregon": (44.5720, -122.0709), "Pennsylvania": (40.5908, -77.2098), "Rhode Island": (41.6809, -71.5118),
    "South Carolina": (33.8569, -80.9450), "South Dakota": (44.2998, -99.4388), "Tennessee": (35.7478, -86.6923),
    "Texas": (31.0545, -97.5635), "Utah": (40.1500, -111.8624), "Vermont": (44.0459, -72.7107),
    "Virginia": (37.7693, -78.1700), "Washington": (47.4009, -121.4905), "West Virginia": (38.4912, -80.9545),
    "Wisconsin": (44.2685, -89.6165), "Wyoming": (42.7560, -107.3025),
}

STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
    "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


def geocoder_available() -> bool:
    return zipcodes is not None


def _latlon(hit: dict):
    try:
        return float(hit["lat"]), float(hit["long"])
    except (KeyError, TypeError, ValueError):
        return None


def _from_zip(code):
    try:
        hits = zipcodes.matching(str(code))
    except Exception:
        return None
    for h in hits or []:
        p = _latlon(h)
        if p:
            return p
    return None


def _from_name(city, state):
    abbr = STATE_ABBR.get(state)
    if not abbr:
        return None
    for name in (city, str(city).upper()):
        try:
            hits = zipcodes.filter_by(city=name, state=abbr)
        except Exception:
            hits = []
        pts = [p for p in (_latlon(h) for h in hits or []) if p]
        if pts:
            return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    return None


def resolve_city_coords(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (City, State) with mean lat/lon. Empty frame if geocoding is unavailable."""
    cols = ["City", "State", "lat", "lon"]
    if zipcodes is None or not {"City", "State"} <= set(df.columns):
        return pd.DataFrame(columns=cols)

    zips = df["Postal Code"] if "Postal Code" in df.columns else pd.Series([None] * len(df), index=df.index)
    pairs = pd.DataFrame({"City": df["City"], "State": df["State"], "Zip": zips}).drop_duplicates()

    found: dict = {}
    for city, state, code in pairs.itertuples(index=False, name=None):
        pts = found.setdefault((city, state), [])
        if code is not None and not pd.isna(code):
            p = _from_zip(code)
            if p:
                pts.append(p)

    rows = []
    for (city, state), pts in found.items():
        if not pts:  # e.g. missing postal code (a few Burlington, VT rows in the public Superstore file)
            p = _from_name(city, state)
            pts = [p] if p else []
        if pts:
            rows.append((city, state, sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)))
    return pd.DataFrame(rows, columns=cols)
