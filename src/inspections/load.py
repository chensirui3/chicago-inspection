"""Read the city's .Rds snapshot and the live Chicago data portal.

The guide suggests pyreadr for the .Rds files. With the pinned snapshot pyreadr 0.5 crashes the
interpreter on ``23_food_insp_features.Rds`` (a segfault, so it cannot be caught) and silently
reads data.table ``IDate`` columns in ``24_bus_features.Rds`` as 1970-01-01. This module therefore
reads every file with ``rdata`` (a pure-Python R serialization parser), registers converters for
R's ``Date``/``IDate``/``POSIXct`` classes, and caches each table once as Parquet.
"""

from __future__ import annotations

import io
import json
import os
import time
import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

RDS_NAMES = (
    "11_bus_license",
    "12_crime",
    "13_food_inspections",
    "14_garbage_carts",
    "15_sanitation_code",
    "17_mongo_weather_update",
    "19_inspector_assignments",
    "21_food_inspection_violation_matrix",
    "21_food_inspection_violation_matrix_nums",
    "22_burglary_heat",
    "22_garbageCarts_heat",
    "22_sanitationComplaints_heat",
    "23_food_insp_features",
    "24_bus_features",
    "30_xgboost_data",
    "inspectors",
)

# Float columns that are really integer identifiers (R stores them as doubles).
_ID_COLUMNS = {
    "Inspection_ID",
    "inspectionID",
    "License",
    "LICENSE_NUMBER",
    "LICENSE_ID",
    "ACCOUNT_NUMBER",
    "LICENSE_CODE",
}


# ----------------------------------------------------------------------------------------------
# .Rds files
# ----------------------------------------------------------------------------------------------
def _as_float(obj) -> np.ndarray:
    return np.ma.filled(np.ma.asarray(obj, dtype="float64"), np.nan)


def _r_date(obj, attrs):  # R "Date" and data.table "IDate": days since 1970-01-01
    return pd.to_datetime(_as_float(obj), unit="D")


def _r_posixct(obj, attrs):  # R "POSIXct": seconds since 1970-01-01 UTC
    return pd.to_datetime(_as_float(obj), unit="s")


def read_rds(path: str | Path) -> pd.DataFrame:
    """Read one .Rds file into a DataFrame, converting R date classes to datetime64."""
    import rdata  # imported lazily: only needed when (re)building the Parquet cache

    constructors = {
        **rdata.conversion.DEFAULT_CLASS_MAP,
        "Date": _r_date,
        "IDate": _r_date,
        "POSIXct": _r_posixct,
    }
    with warnings.catch_warnings():  # rdata warns that ".Rds" is not its preferred extension
        warnings.simplefilter("ignore")
        df = rdata.read_rds(path, constructor_dict=constructors)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{path} does not contain a data.frame (got {type(df).__name__})")
    return _normalize(df)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    for col in df.columns:
        s = df[col]
        if col in _ID_COLUMNS and pd.api.types.is_float_dtype(s):
            finite = s.dropna()
            if (finite == np.round(finite)).all():
                df[col] = s.astype("Int64")
        elif isinstance(s.dtype, pd.StringDtype):
            df[col] = s.astype("string")
    return df


def rds_cache_path(name: str) -> Path:
    return config.PROCESSED_DIR / "rds" / f"{name}.parquet"


def convert_rds(name: str, force: bool = False) -> Path:
    """Convert ``data/raw/<name>.Rds`` to ``data/processed/rds/<name>.parquet`` once."""
    out = rds_cache_path(name)
    if out.exists() and not force:
        return out
    src = config.RAW_DIR / f"{name}.Rds"
    if not src.exists():
        raise FileNotFoundError(f"{src} is missing; run `python scripts/get_data.py` first")
    df = read_rds(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    return out


def load_rds(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Load one of the city's tables (from the Parquet cache, building it if needed)."""
    return pd.read_parquet(convert_rds(name), columns=columns)


def data_available(names: tuple[str, ...] = ("23_food_insp_features",)) -> bool:
    """True when the snapshot (raw or cached) is present; tests use this to skip."""
    return all(
        rds_cache_path(n).exists() or (config.RAW_DIR / f"{n}.Rds").exists() for n in names
    )


# ----------------------------------------------------------------------------------------------
# Chicago data portal (Socrata) and NOAA
# ----------------------------------------------------------------------------------------------
PORTAL = "https://data.cityofchicago.org"
DATASETS = {
    "food_inspections": "4ijn-s7e5",
    "business_licenses": "r5kz-chrr",
    "crimes": "ijzp-q8t2",
    "service_requests_311": "v6vf-nfxy",
    "community_areas": "igwz-8jzy",  # "Boundaries - Community Areas" (cauq-8yn6 is a map view)
    "socioeconomic_2008_2012": "kn9c-c2s2",
}


def _session():
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = "chicago-inspections (open-source research project)"
    token = os.environ.get("SOCRATA_APP_TOKEN")
    if token:  # never hard-code the token; CI passes it as a GitHub Actions secret
        s.headers["X-App-Token"] = token
    return s


RETRY_STATUS = (429, 500, 502, 503, 504)


def _get_with_retries(session, url: str, params: dict, retries: int):
    """GET with exponential backoff on timeouts, dropped connections, 429 and 5xx."""
    import requests

    # A connection the portal drops partway through a response body raises
    # ChunkedEncodingError, which is neither a Timeout nor a ConnectionError.
    transient = (requests.Timeout, requests.ConnectionError,
                 requests.exceptions.ChunkedEncodingError)
    for attempt in range(retries):
        last = attempt == retries - 1
        try:
            resp = session.get(url, params=params, timeout=(30, 300))
        except transient:
            if last:
                raise
        else:
            if resp.status_code == 200:
                return resp
            if resp.status_code not in RETRY_STATUS or last:
                raise RuntimeError(f"{url} returned {resp.status_code}: {resp.text[:300]}")
        time.sleep(min(10 * 2**attempt, 120))
    raise AssertionError("unreachable")


def socrata_pages(
    dataset: str,
    where: str | None = None,
    select: str | None = None,
    page_size: int = 50_000,
    max_rows: int | None = None,
    retries: int = 5,
) -> Iterator[pd.DataFrame]:
    """Yield pages of a Socrata dataset as DataFrames (all columns as strings).

    Pages by row identifier (``$order=:id`` plus ``:id > last``) instead of ``$offset``, so
    every page costs the portal the same however deep it is; offset paging slows down on large
    datasets such as 311 Service Requests. When ``SOCRATA_APP_TOKEN`` is set it is sent as
    ``X-App-Token`` (required by the newer SODA3 endpoints and avoids throttling).
    """
    ds = DATASETS.get(dataset, dataset)
    url = f"{PORTAL}/resource/{ds}.csv"
    session = _session()
    fetched, last_id = 0, None
    while True:
        limit = page_size if max_rows is None else min(page_size, max_rows - fetched)
        if limit <= 0:
            return
        clauses = [f"({where})"] if where else []
        if last_id is not None:
            clauses.append(f":id > '{last_id}'")
        params = {"$select": f":id,{select or '*'}", "$order": ":id", "$limit": limit}
        if clauses:
            params["$where"] = " AND ".join(clauses)
        resp = _get_with_retries(session, url, params, retries)
        page = pd.read_csv(io.StringIO(resp.text), dtype=str, keep_default_na=False)
        if page.empty:
            return
        last_id = page[":id"].iloc[-1]
        yield page.drop(columns=[":id"])
        fetched += len(page)
        if len(page) < limit:
            return


def socrata_frame(dataset: str, **kwargs) -> pd.DataFrame:
    pages = list(socrata_pages(dataset, **kwargs))
    return pd.concat(pages, ignore_index=True) if pages else pd.DataFrame()


def download(url: str, dest: Path, overwrite: bool = False) -> Path:
    """Download a URL to ``dest`` (atomic write)."""
    if dest.exists() and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = _session().get(url, timeout=300)
    resp.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(resp.content)
    tmp.replace(dest)
    return dest


def community_areas_path() -> Path:
    return config.PORTAL_DIR / "community_areas.geojson"


def socioeconomic_path() -> Path:
    return config.PORTAL_DIR / "socioeconomic_2008_2012.csv"


def fetch_extension_b_data(overwrite: bool = False) -> dict[str, Path]:
    """Community-area polygons and 2008-2012 socioeconomic indicators (both small)."""
    ca = download(
        f"{PORTAL}/api/geospatial/{DATASETS['community_areas']}?method=export&format=GeoJSON",
        community_areas_path(),
        overwrite,
    )
    se = download(
        f"{PORTAL}/api/views/{DATASETS['socioeconomic_2008_2012']}/rows.csv?accessType=DOWNLOAD",
        socioeconomic_path(),
        overwrite,
    )
    return {"community_areas": ca, "socioeconomic": se}


def load_community_areas():
    import geopandas as gpd

    path = community_areas_path()
    if not path.exists():
        fetch_extension_b_data()
    gdf = gpd.read_file(path)
    gdf.columns = [c.lower() for c in gdf.columns]
    num = "area_numbe" if "area_numbe" in gdf.columns else "area_num_1"
    gdf["community_area"] = pd.to_numeric(gdf[num]).astype(int)
    gdf["community_name"] = gdf["community"].str.title()
    return gdf[["community_area", "community_name", "geometry"]].to_crs(4326)


def load_socioeconomic() -> pd.DataFrame:
    path = socioeconomic_path()
    if not path.exists():
        fetch_extension_b_data()
    se = pd.read_csv(path)
    se.columns = [c.strip() for c in se.columns]
    rename = {
        "Community Area Number": "community_area",
        "COMMUNITY AREA NAME": "community_name_se",
        "PER CAPITA INCOME": "per_capita_income",
        "PERCENT HOUSEHOLDS BELOW POVERTY": "pct_below_poverty",
        "HARDSHIP INDEX": "hardship_index",
    }
    se = se.rename(columns=rename)
    se = se[se["community_area"].notna()].copy()  # drops the city-wide "CHICAGO" row
    se["community_area"] = se["community_area"].astype(int)
    return se[
        ["community_area", "community_name_se", "per_capita_income", "pct_below_poverty",
         "hardship_index"]
    ]


def _get_json(url: str, params: dict, attempts: int):
    import requests

    for attempt in range(attempts):
        try:
            resp = _session().get(url, params=params, timeout=(30, 180))
            resp.raise_for_status()
            return json.loads(resp.text or "null")
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            time.sleep(10 * 2**attempt)
    raise AssertionError("unreachable")


def _weather_frame(dates, values) -> pd.DataFrame:
    df = pd.DataFrame({"date": pd.to_datetime(list(dates)),
                       "temperatureMax": pd.to_numeric(pd.Series(list(values), dtype="object"),
                                                       errors="coerce")})
    return df.sort_values("date").reset_index(drop=True)


def noaa_daily(start: str, end: str, station: str = "USW00094846",
               attempts: int = 3) -> pd.DataFrame:
    """Daily max temperature (deg F) from NOAA NCEI's GHCN-Daily service (O'Hare by default)."""
    rows = _get_json("https://www.ncei.noaa.gov/access/services/data/v1", {
        "dataset": "daily-summaries", "stations": station, "startDate": start,
        "endDate": end, "dataTypes": "TMAX", "units": "standard", "format": "json",
    }, attempts) or []
    if not rows:
        return pd.DataFrame(columns=["date", "temperatureMax"])
    return _weather_frame([r["DATE"] for r in rows], [r.get("TMAX") for r in rows])


def acis_daily(start: str, end: str, station: str = "USW00094846",
               attempts: int = 2) -> pd.DataFrame:
    """The same GHCN-Daily max temperature from NOAA's Regional Climate Centers (ACIS).

    Used when NCEI is unreachable. Missing days come back as "M" and become NaN.
    """
    out = _get_json("https://data.rcc-acis.org/StnData", {
        "sid": station, "sdate": start, "edate": end, "elems": "maxt", "output": "json",
    }, attempts) or {}
    if "error" in out:
        raise ValueError(f"ACIS: {out['error']}")
    rows = out.get("data", [])
    return _weather_frame([r[0] for r in rows], [r[1] for r in rows])
