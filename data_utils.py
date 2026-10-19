"""Loading, cleaning and aggregating the sales table. No plotting in here."""
from __future__ import annotations

import base64
import hashlib
import io
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DEFAULT_FILE = Path(os.getenv("SALES_CSV", BASE_DIR / "data" / "superstore.csv"))

CANONICAL = [
    "Row ID", "Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID", "Customer Name", "Segment",
    "Country", "City", "State", "Postal Code", "Region", "Product ID", "Category", "Sub-Category",
    "Product Name", "Sales",
]
_squash = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())  # noqa: E731
_NAME_MAP = {_squash(c): c for c in CANONICAL}

CAT_COLS = ["Ship Mode", "Segment", "Country", "Region", "State", "City", "Category", "Sub-Category"]
TEXT_COLS = CAT_COLS + ["Customer Name", "Product Name"]
_KEY_RE = re.compile(r"[0-9a-f]{16}")


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def default_available() -> bool:
    return DEFAULT_FILE.exists()


def read_table(raw: bytes, filename: str = "") -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".xls"):
        raise ValueError("Old .xls files are not supported. Save the sheet as .xlsx or .csv.")
    if name.endswith((".xlsx", ".xlsm")):
        return pd.read_excel(io.BytesIO(raw))
    last = None
    for enc in ("utf-8-sig", "latin-1"):  # the public Superstore CSV is often latin-1
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            if df.shape[1] == 1:  # semicolon / tab separated
                df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
            return df
        except UnicodeDecodeError as exc:
            last = exc
    raise ValueError(f"Could not read the file: {last}")


# --------------------------------------------------------------------------- #
# Dates (dd/mm/yyyy vs mm/dd/yyyy is the classic Superstore trap)
# --------------------------------------------------------------------------- #
def _parse_dmy(s: pd.Series, dayfirst: bool) -> pd.Series:
    txt = s.astype("string").str.extract(r"(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{4})")[0]
    txt = txt.str.replace(r"[\-.]", "/", regex=True)
    return pd.to_datetime(txt, format="%d/%m/%Y" if dayfirst else "%m/%d/%Y", errors="coerce")


def parse_dates(order: pd.Series, ship: pd.Series | None):
    """Return (order_dates, ship_dates_or_None, format_label)."""
    if pd.api.types.is_datetime64_any_dtype(order):
        s = pd.to_datetime(ship, errors="coerce").dt.normalize() if ship is not None else None
        return order.dt.normalize(), s, "native dates"

    parts = order.astype("string").str.extract(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.]\d{4}")
    if parts.notna().all(axis=1).mean() < 0.9:  # ISO (yyyy-mm-dd) or something else: let pandas infer
        o = pd.to_datetime(order, errors="coerce").dt.normalize()
        s = pd.to_datetime(ship, errors="coerce").dt.normalize() if ship is not None else None
        return o, s, "yyyy-mm-dd (auto)"

    a = pd.to_numeric(parts[0], errors="coerce").max()
    b = pd.to_numeric(parts[1], errors="coerce").max()
    if a > 12:
        dayfirst = True
    elif b > 12:
        dayfirst = False
    else:  # every value is ambiguous: the date that makes ship >= order wins
        dayfirst = True
        if ship is not None:
            def plausible(df_first: bool) -> float:
                gap = (_parse_dmy(ship, df_first) - _parse_dmy(order, df_first)).dt.days
                return float(gap.between(0, 30).mean())
            dayfirst = plausible(True) >= plausible(False)

    o = _parse_dmy(order, dayfirst).dt.normalize()
    s = _parse_dmy(ship, dayfirst).dt.normalize() if ship is not None else None
    return o, s, "dd/mm/yyyy" if dayfirst else "mm/dd/yyyy"


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean(raw: pd.DataFrame):
    """Return (clean_df, meta). Raises ValueError with a readable message on bad input."""
    df = raw.copy()
    df.columns = [_NAME_MAP.get(_squash(c), str(c).strip()) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    missing = [c for c in ("Order Date", "Sales") if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}. "
                         f"Found: {', '.join(map(str, df.columns[:15]))}")

    n_raw = len(df)
    od, sd, fmt = parse_dates(df["Order Date"], df["Ship Date"] if "Ship Date" in df.columns else None)
    df["Order Date"] = od
    if sd is not None:
        df["Ship Date"] = sd
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

    df = df.loc[df["Order Date"].notna() & df["Sales"].notna()].copy()
    if df.empty:
        raise ValueError("No rows with a valid Order Date and Sales value.")

    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip().fillna("Unknown").astype(object)

    if "Postal Code" in df.columns:  # numeric CSVs lose leading zeros (Boston 02122 -> 2122)
        pc = df["Postal Code"].astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
        short = pc.str.fullmatch(r"\d{1,4}").fillna(False)
        pc = pc.mask(short, pc.str.zfill(5))
        df["Postal Code"] = pc.astype(object).where(pc.notna(), np.nan)

    if "Ship Date" in df.columns:
        df["Ship Days"] = (df["Ship Date"] - df["Order Date"]).dt.days

    df = df.sort_values("Order Date", kind="mergesort").reset_index(drop=True)
    meta = dict(
        rows=len(df), rows_raw=n_raw, dropped=n_raw - len(df), date_format=fmt,
        date_min=df["Order Date"].min(), date_max=df["Order Date"].max(),
        missing_postal=int(df["Postal Code"].isna().sum()) if "Postal Code" in df.columns else 0,
    )
    return df, meta


# --------------------------------------------------------------------------- #
# Loading / caching. Datasets live on disk and are referred to by a short key, so callbacks
# only pass a 16-character string around instead of the whole table.
# --------------------------------------------------------------------------- #
def save_upload(contents: str, filename: str) -> str:
    if not contents or "," not in contents:
        raise ValueError("The upload was empty.")
    raw = base64.b64decode(contents.split(",", 1)[1])
    table = read_table(raw, filename)
    clean(table)  # validate now so the user sees a clear error immediately
    key = hashlib.sha1(raw).hexdigest()[:16]
    UPLOAD_DIR.mkdir(exist_ok=True)
    table.to_csv(UPLOAD_DIR / f"{key}.csv", index=False)
    return key


@lru_cache(maxsize=4)
def load_df(key: str):
    if key == "demo":
        from demo_data import make_demo
        raw = make_demo()
    elif key == "default":
        raw = read_table(DEFAULT_FILE.read_bytes(), DEFAULT_FILE.name)
    elif _KEY_RE.fullmatch(key or ""):  # keys come from the browser: never build a path from anything else
        raw = pd.read_csv(UPLOAD_DIR / f"{key}.csv")
    else:
        raise ValueError("Unknown dataset key.")
    return clean(raw)


def get_df(key: str) -> pd.DataFrame:
    """Shared, cached frame. Treat it as read-only: copy before modifying."""
    return load_df(key)[0]


def get_meta(key: str) -> dict:
    return load_df(key)[1]


# --------------------------------------------------------------------------- #
# Aggregations used by several tabs
# --------------------------------------------------------------------------- #
def daily_series(df: pd.DataFrame, category: str | None = None) -> pd.Series:
    """Total sales per calendar day. Days without orders are 0 (not dropped)."""
    d = df
    if category not in (None, "All") and "Category" in df.columns:
        d = df[df["Category"] == category]
    if d.empty:
        raise ValueError(f"No rows for category '{category}'.")
    s = d.groupby("Order Date")["Sales"].sum()
    idx = pd.date_range(df["Order Date"].min(), df["Order Date"].max(), freq="D")
    return s.reindex(idx, fill_value=0.0).astype(float).rename("Sales")


def daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Order Date")
    fr = pd.DataFrame({
        "Daily Sales": g["Sales"].sum(),
        "Lines": g["Sales"].size(),
        "Avg Line Sales": g["Sales"].mean(),
    })
    if "Order ID" in df.columns:
        fr["Orders"] = g["Order ID"].nunique()
    if "Customer ID" in df.columns:
        fr["Customers"] = g["Customer ID"].nunique()
    idx = pd.date_range(df["Order Date"].min(), df["Order Date"].max(), freq="D")
    fr = fr.reindex(idx).fillna(0.0)
    fr["Lag 1"] = fr["Daily Sales"].shift(1)
    fr["Lag 7"] = fr["Daily Sales"].shift(7)
    fr["Day of Week"] = fr.index.dayofweek
    fr["Month"] = fr.index.month
    fr.index.name = "Date"
    return fr


def numeric_view(df: pd.DataFrame, which: str):
    """(frame with date,value ; label) for the univariate tab."""
    if which == "daily":
        s = daily_series(df)
        return pd.DataFrame({"date": s.index, "value": s.to_numpy()}), "Daily sales (total per day)"
    if which == "ship" and "Ship Days" in df.columns:
        v = df[["Order Date", "Ship Days"]].dropna()
        return v.rename(columns={"Order Date": "date", "Ship Days": "value"}), "Ship days"
    return df[["Order Date", "Sales"]].rename(columns={"Order Date": "date", "Sales": "value"}), "Sales (order line)"


def profile(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    out = []
    for c in cols:
        s = df[c]
        ex = s.dropna().iloc[0] if s.notna().any() else ""
        if isinstance(ex, pd.Timestamp):
            ex = ex.strftime("%Y-%m-%d")
        out.append({"Column": c, "Type": str(s.dtype), "Non-null": int(s.notna().sum()),
                    "Missing %": float(round(100 * s.isna().mean(), 2)), "Unique": int(s.nunique()),
                    "Example": str(ex)[:40]})
    return out
