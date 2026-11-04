"""Feature building.

Python ports of the City of Chicago's food-inspections-evaluation code (Copyright, 2014 City of
Chicago; MIT-style license): CODE/30a_glmnet_model.R (joins and transforms),
functions/weather_3day_calc.R, CODE/23_food_insp_features.R with functions/filter_foodInspect.R,
calculate_violation_matrix.R and calculate_violation_types.R (history features), and
CODE/22_calculate_heat_map_values.R with functions/calculate_heat_values.R and kde.R
(density features). New features for Core 2 and the as-of-date rebuild for Core 3 are below.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config
from .load import load_rds

DAY = pd.Timedelta(days=1)


# ----------------------------------------------------------------------------------------------
# Core 1: the city's modeling table
# ----------------------------------------------------------------------------------------------
def weather_3day(weather: pd.DataFrame) -> pd.DataFrame:
    """For each date, the mean of each weather column over the three preceding days.

    Port of weather_3day_calc.R (which assumes one row per consecutive day).
    """
    w = weather.sort_values("date", ascending=False).reset_index(drop=True)
    cols = [c for c in w.columns if c != "date"]
    vals = w[cols].to_numpy(dtype="float64")
    three = (vals[1:-2] + vals[2:-1] + vals[3:]) / 3
    out = pd.DataFrame(three, columns=cols)
    out.insert(0, "Inspection_Date", w["date"].iloc[:-3].to_numpy())
    return out.sort_values("Inspection_Date").reset_index(drop=True)


def published_ids() -> np.ndarray:
    """Inspection_IDs of the city's published 18,712-row modeling table (DATA/30_xgboost_data).

    The 2019 refactor of the city's repo regenerated the inspector assignments and business
    licenses; joining today's snapshot as CODE/30a does yields 69 more inspections than the
    published evaluation (1,665 vs 1,637 in the test window). The XGBoost table saved in 2017
    is the only file in the snapshot with exactly the published rows (17,075 / 1,637 / 258).
    """
    return load_rds("30_xgboost_data", columns=["Inspection_ID"])["Inspection_ID"].to_numpy()


def build_city_table(row_set: str = "published") -> pd.DataFrame:
    """Join the precomputed feature files exactly as CODE/30a_glmnet_model.R does.

    ``row_set="snapshot"`` keeps every row the joins produce from today's snapshot;
    ``row_set="published"`` keeps the city's published inspections only. Join counts are stored
    in ``df.attrs["join_counts"]``.
    """
    food = load_rds("23_food_insp_features")
    bus = load_rds("24_bus_features")
    san = load_rds("19_inspector_assignments").rename(columns={"inspectionID": "Inspection_ID"})
    weather = load_rds("17_mongo_weather_update")
    heat = {
        "heat_burglary": load_rds("22_burglary_heat"),
        "heat_garbage": load_rds("22_garbageCarts_heat"),
        "heat_sanitation": load_rds("22_sanitationComplaints_heat"),
    }
    shared = (set(food.columns) & set(bus.columns)) - {"Inspection_ID"}
    if shared:
        raise ValueError(f"unexpected shared columns between food and bus tables: {shared}")

    counts = {}
    dat = bus.merge(food, on="Inspection_ID", how="left")  # data.table food[bus]
    counts["food[bus]"] = len(dat)
    dat = dat.merge(san, on="Inspection_ID")
    counts["+ inspector clusters"] = len(dat)
    dat = dat.merge(weather_3day(weather), on="Inspection_Date")
    counts["+ 3-day weather"] = len(dat)
    for name, h in heat.items():
        dat = dat.merge(h.dropna().rename(columns={"heat_values": name}), on="Inspection_ID")
    counts["+ density features"] = len(dat)
    dat = dat[dat["LICENSE_DESCRIPTION"] == "Retail Food Establishment"]
    counts["Retail Food Establishment"] = len(dat)
    dat = dat.sort_values("Inspection_ID").reset_index(drop=True)
    dat["criticalFound"] = np.minimum(1, dat["criticalCount"]).astype(int)
    dat["Test"] = dat["Inspection_Date"] >= dat["Inspection_Date"].max() - 90 * DAY
    if row_set == "published":
        dat = dat[dat["Inspection_ID"].isin(published_ids())].reset_index(drop=True)
        counts["published rows"] = len(dat)
    elif row_set != "snapshot":
        raise ValueError(row_set)
    dat.attrs["join_counts"] = counts
    return dat


def city_design_matrix(df: pd.DataFrame, inspector: bool = True,
                       temperature: bool = True) -> pd.DataFrame:
    """The city's model matrix (CODE/30a ``xmat`` + ``model.matrix(criticalFound ~ . - 1)``)."""
    X = pd.DataFrame(index=df.index)
    if inspector:
        for c in config.INSPECTOR_CLUSTERS:
            X[f"Inspector_{c}"] = (df["sanitarian"] == c).astype(float)
    X["pastSerious"] = np.minimum(df["pastSerious"], 1).astype(float)
    X["pastCritical"] = np.minimum(df["pastCritical"], 1).astype(float)
    X["timeSinceLast"] = df["timeSinceLast"].astype(float)
    X["ageAtInspection"] = (df["ageAtInspection"] > 4).astype(float)
    X["consumption_on_premises_incidental_activity"] = df[
        "consumption_on_premises_incidental_activity"
    ].astype(float)
    X["tobacco"] = df["tobacco"].astype(float)
    if temperature:
        X["temperatureMax"] = df["temperatureMax"].astype(float)
    X["heat_burglary"] = np.minimum(df["heat_burglary"], 70).astype(float)
    X["heat_sanitation"] = np.minimum(df["heat_sanitation"], 70).astype(float)
    X["heat_garbage"] = np.minimum(df["heat_garbage"], 50).astype(float)
    if X.isna().any().any():
        raise ValueError(f"NaN in design matrix: {X.columns[X.isna().any()].tolist()}")
    return X


# ----------------------------------------------------------------------------------------------
# Rebuilding the history features from the raw inspections (CODE/23_food_insp_features.R)
# ----------------------------------------------------------------------------------------------
BAD_RESULTS = ("Out of Business", "Business Not Located", "No Entry")


def filter_food_inspections(food: pd.DataFrame) -> pd.DataFrame:
    """Port of filter_foodInspect.R: canvass inspections after 2011-09-01 with a license."""
    f = food[food["Inspection_Date"].notna() & food["License"].notna()]
    f = f[~f["Inspection_ID"].duplicated()]
    f = f[f["License"] != 0]
    f = f[f["Inspection_Date"] > pd.Timestamp("2011-09-01")]
    f = f[f["Inspection_Type"] == "Canvass"]
    f = f[~f["Results"].isin(BAD_RESULTS)]
    return f.reset_index(drop=True)


_CODE = re.compile(r"^([0-9]+)")


def violation_codes(text) -> list[int]:
    """Leading code number of every violation in a "| "-separated violation string."""
    if text is None or (isinstance(text, float) and np.isnan(text)) or text is pd.NA:
        return []
    out = []
    for piece in str(text).split("| "):
        m = _CODE.match(piece)
        if m:
            out.append(int(m.group(1)))
    return out


def violation_matrix(violations: pd.Series) -> pd.DataFrame:
    """Count of each violation code per inspection (port of calculate_violation_matrix.R)."""
    rows = [violation_codes(v) for v in violations]
    codes = sorted({c for r in rows for c in r})
    mat = np.zeros((len(rows), len(codes)), dtype=np.int16)
    pos = {c: j for j, c in enumerate(codes)}
    for i, r in enumerate(rows):
        for c in r:
            mat[i, pos[c]] += 1
    return pd.DataFrame(mat, columns=codes, index=violations.index)


def violation_types(vio: pd.DataFrame, scheme: str = "pre2018") -> pd.DataFrame:
    """Critical (codes 1-14), serious (15-29) and minor (30-44) counts, pre-July-2018 code."""
    if scheme != "pre2018":
        raise ValueError("only the pre-2018 numbering has fixed critical/serious ranges")
    cols = np.asarray(vio.columns, dtype=int)

    def total(lo, hi):
        return vio.loc[:, (cols >= lo) & (cols <= hi)].sum(axis=1)

    return pd.DataFrame(
        {"criticalCount": total(1, 14), "seriousCount": total(15, 29),
         "minorCount": total(30, 44)},
        index=vio.index,
    )


def history_features(insp: pd.DataFrame) -> pd.DataFrame:
    """pastFail/pastCritical/pastSerious/pastMinor/timeSinceLast/firstRecord per inspection.

    ``insp`` needs Inspection_ID, License, Inspection_Date, Results and the three counts. Rows
    are ordered by license and date (ties by Inspection_ID, as data.table's stable sort on a
    table keyed by Inspection_ID does) and each license's previous inspection is taken.
    """
    df = insp.sort_values("Inspection_ID").sort_values(
        ["License", "Inspection_Date"], kind="stable"
    )
    g = df.groupby("License", sort=False)
    out = pd.DataFrame({"Inspection_ID": df["Inspection_ID"].to_numpy()}, index=df.index)
    fail = (df["Results"] == "Fail").astype(float)
    out["pastFail"] = fail.groupby(df["License"]).shift(1).fillna(0).to_numpy()
    for src, dst in [("criticalCount", "pastCritical"), ("seriousCount", "pastSerious"),
                     ("minorCount", "pastMinor")]:
        out[dst] = g[src].shift(1).fillna(0).to_numpy(dtype=float)
    gap = (df["Inspection_Date"] - g["Inspection_Date"].shift(1)).dt.days / 365
    out["firstRecord"] = gap.isna().astype(float).to_numpy()
    out["timeSinceLast"] = np.minimum(gap.fillna(2).to_numpy(), 2.0)
    return out.sort_values("Inspection_ID").reset_index(drop=True)


def rebuild_food_features() -> pd.DataFrame:
    """Recompute the city's 23_food_insp_features columns from the raw 13_ inspections."""
    food = filter_food_inspections(load_rds("13_food_inspections"))
    counts = violation_types(violation_matrix(food["Violations"]))
    insp = pd.concat([food.drop(columns=["Violations"]), counts], axis=1)
    hist = history_features(insp)
    return insp.drop(columns=[]).merge(hist, on="Inspection_ID")


# ----------------------------------------------------------------------------------------------
# Density ("heat") features (CODE/22_calculate_heat_map_values.R)
# ----------------------------------------------------------------------------------------------
def heat_observations(kind: str) -> pd.DataFrame:
    """Date/Latitude/Longitude of burglaries, garbage-cart or sanitation requests (filtered)."""
    if kind == "burglary":
        crime = load_rds("12_crime", columns=["Date", "Latitude", "Longitude"])
        obs = crime[crime["Date"] > pd.Timestamp("2011-07-01")]
    else:
        name = {"garbage": "14_garbage_carts", "sanitation": "15_sanitation_code"}[kind]
        raw = load_rds(name, columns=["Creation_Date", "Status", "Latitude", "Longitude"])
        raw = raw[raw["Status"].isin(["Completed", "Open"])]
        obs = raw.rename(columns={"Creation_Date": "Date"})[["Date", "Latitude", "Longitude"]]
    return obs.dropna().sort_values("Date").reset_index(drop=True)


def kde_heat(lat, lon, obs_lat, obs_lon, h: float = 0.01) -> np.ndarray:
    """Bivariate normal kernel density at each (lat, lon) (port of functions/kde.R).

    As in the city's code the bandwidth ``h`` is divided by 4 and the density is normalized by
    the number of observations in the window.
    """
    h = h / 4
    lat = np.asarray(lat, float)[:, None]
    lon = np.asarray(lon, float)[:, None]
    out = np.empty(lat.shape[0])
    for start in range(0, lat.shape[0], 256):
        sl = slice(start, start + 256)
        ax = (lat[sl] - obs_lat[None, :]) / h
        ay = (lon[sl] - obs_lon[None, :]) / h
        k = np.exp(-0.5 * (ax * ax + ay * ay)) / (2 * np.pi)
        out[sl] = k.sum(axis=1)
    return out / (len(obs_lat) * h * h)


def heat_values(
    points: pd.DataFrame, obs: pd.DataFrame, window: int = 90, as_of=None
) -> np.ndarray:
    """Density of ``obs`` within ``window`` days up to and including each point's date.

    ``points`` needs Inspection_Date, Latitude and Longitude. With ``as_of`` every point uses
    the window ending on that date instead (Core 3's "as of day one" scoring).
    """
    dates = (
        pd.Series(pd.Timestamp(as_of), index=points.index)
        if as_of is not None
        else points["Inspection_Date"]
    )
    obs_d = obs["Date"].to_numpy()
    obs_lat = obs["Latitude"].to_numpy(float)
    obs_lon = obs["Longitude"].to_numpy(float)
    out = np.full(len(points), np.nan)
    lat = points["Latitude"].to_numpy(float)
    lon = points["Longitude"].to_numpy(float)
    ok = ~(np.isnan(lat) | np.isnan(lon))
    for d, idx in pd.Series(np.arange(len(points)))[ok].groupby(dates[ok].to_numpy()):
        lo = np.searchsorted(obs_d, np.datetime64(d - window * DAY), side="left")
        hi = np.searchsorted(obs_d, np.datetime64(d), side="right")
        if hi > lo:
            rows = idx.to_numpy()
            out[rows] = kde_heat(lat[rows], lon[rows], obs_lat[lo:hi], obs_lon[lo:hi])
    return out


# ----------------------------------------------------------------------------------------------
# Core 2: new features. Every one is computed from records dated strictly before the inspection.
# ----------------------------------------------------------------------------------------------
_KEY = 100_000  # (license, day) -> license * _KEY + day; days since 1970 stay below 100,000


def _days(dates) -> np.ndarray:
    return pd.to_datetime(pd.Series(dates)).to_numpy().astype("datetime64[D]").astype(np.int64)


class EventIndex:
    """Per-license event history answering "how many events before day d" in O(log n)."""

    def __init__(self, license_, days, values: dict[str, np.ndarray]):
        lic = np.asarray(license_, dtype=np.int64)
        days = np.asarray(days, dtype=np.int64)
        order = np.lexsort((days, lic))
        self.lic = lic[order]
        self.days = days[order]
        self.key = self.lic * _KEY + self.days
        self.order = order
        self.values = {k: np.asarray(v, dtype=float)[order] for k, v in values.items()}
        self.cums = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in self.values.items()}

    def _bounds(self, lic, day, lookback):
        lic = np.asarray(lic, dtype=np.int64)
        day = np.asarray(day, dtype=np.int64)
        hi = np.searchsorted(self.key, lic * _KEY + day, side="left")
        start = lic * _KEY + (day - lookback if lookback is not None else 0)
        lo = np.searchsorted(self.key, start, side="left")
        return lo, hi

    def total(self, name, lic, day, lookback=None) -> np.ndarray:
        lo, hi = self._bounds(lic, day, lookback)
        return self.cums[name][hi] - self.cums[name][lo]

    def last(self, lic, day) -> np.ndarray:
        """Position of the license's last event before ``day`` (in sorted order); -1 if none."""
        lo, hi = self._bounds(lic, day, None)
        return np.where(hi > lo, hi - 1, -1)


def raw_events() -> pd.DataFrame:
    """Every real visit in the raw 13_ inspections (all types), with violation counts."""
    raw = load_rds("13_food_inspections")
    ev = raw[raw["Inspection_Date"].notna() & raw["License"].notna() & (raw["License"] != 0)]
    ev = ev[~ev["Inspection_ID"].duplicated() & ~ev["Results"].isin(BAD_RESULTS)]
    ev = ev.reset_index(drop=True)
    t = ev["Inspection_Type"].fillna("").str.strip().str.lower()
    ev["kind"] = np.select(
        [t == "canvass", t.str.contains("re-insp"),
         t.str.contains("complaint") | t.str.contains("food poisoning"),
         t.str.startswith("license")],
        ["canvass", "reinspection", "complaint", "license"],
        "other",
    )
    vio = violation_matrix(ev["Violations"])
    ev = pd.concat([ev.drop(columns=["Violations"]), violation_types(vio)], axis=1)
    ev["fail"] = (ev["Results"] == "Fail").astype(float)
    ev["crit"] = (ev["criticalCount"] > 0).astype(float)
    ev["serious"] = (ev["seriousCount"] > 0).astype(float)
    ev["day"] = _days(ev["Inspection_Date"])
    ev.attrs["codes"] = vio
    return ev


FACILITY_TYPES = ("Restaurant", "Grocery Store", "Bakery", "School", "Catering")


def extended_features(table: pd.DataFrame, as_of=None, train_mask=None,
                      events: pd.DataFrame | None = None,
                      code_columns: list[int] | None = None) -> pd.DataFrame:
    """New features for each row of ``table`` (needs License, Inspection_Date, Latitude,
    Longitude, Facility_Type, Risk).

    With ``as_of`` every row is described as of that date instead of its own inspection date
    (Core 3). ``train_mask`` marks the rows used to choose which past violation codes to keep
    (codes seen at >= 2% of previous visits); ``code_columns`` fixes that list instead.
    """
    ev = raw_events() if events is None else events
    codes = ev.attrs["codes"]
    lic = table["License"].astype("int64").to_numpy()
    if as_of is not None:
        day = np.full(len(table), _days([as_of])[0])
    else:
        day = _days(table["Inspection_Date"])
    out = pd.DataFrame(index=table.index)

    kinds = {k: (ev["kind"] == k).to_numpy(float)
             for k in ("canvass", "complaint", "reinspection", "license")}
    crit = ev["crit"].to_numpy()
    vals = {"all": np.ones(len(ev)), "crit": crit, "fail": ev["fail"].to_numpy(),
            "serious": ev["serious"].to_numpy(), **kinds,
            "canvass_crit": kinds["canvass"] * crit,
            "complaint_crit": kinds["complaint"] * crit,
            "license_fail": kinds["license"] * ev["fail"].to_numpy()}
    idx = EventIndex(ev["License"], ev["day"], vals)

    out["n_insp_2y"] = idx.total("all", lic, day, 730)
    out["n_canvass_all"] = idx.total("canvass", lic, day)
    crit_canvass = idx.total("canvass_crit", lic, day)
    out["canvass_crit_share"] = (crit_canvass + 2 * 0.15) / (out["n_canvass_all"] + 2)
    out["n_crit_2y"] = idx.total("crit", lic, day, 730)
    out["n_serious_2y"] = idx.total("serious", lic, day, 730)
    out["n_fail_2y"] = idx.total("fail", lic, day, 730)
    out["n_complaint_1y"] = idx.total("complaint", lic, day, 365)
    out["n_complaint_crit_2y"] = idx.total("complaint_crit", lic, day, 730)
    out["n_reinspection_2y"] = idx.total("reinspection", lic, day, 730)
    out["license_insp_failed"] = (idx.total("license_fail", lic, day) > 0).astype(float)

    last = idx.last(lic, day)
    has = last >= 0
    safe = np.where(has, last, 0)
    out["days_since_any"] = np.where(has, day - idx.days[safe], 730).clip(0, 730).astype(float)
    out["last_any_critical"] = np.where(has, idx.values["crit"][safe], 0.0)
    out["last_any_fail"] = np.where(has, idx.values["fail"][safe], 0.0)

    comp = ev[ev["kind"] == "complaint"]
    cidx = EventIndex(comp["License"], comp["day"], {"n": np.ones(len(comp))})
    clast = cidx.last(lic, day)
    since = day - cidx.days[np.where(clast >= 0, clast, 0)]
    out["days_since_complaint"] = np.where(clast >= 0, since, 730).clip(0, 730).astype(float)

    # violation codes recorded at the previous canvass visit (back to 2010)
    can = ev[ev["kind"] == "canvass"]
    can_codes = codes.loc[can.index].to_numpy()
    cv_idx = EventIndex(can["License"], can["day"], {"n": np.ones(len(can))})
    prev = cv_idx.last(lic, day)
    rows = can_codes[cv_idx.order][np.where(prev >= 0, prev, 0)]
    prev_codes = pd.DataFrame(np.where(prev[:, None] >= 0, rows, 0) > 0,
                              columns=codes.columns, index=table.index).astype(float)
    if code_columns is None:
        keep_rows = np.ones(len(table), bool) if train_mask is None else np.asarray(train_mask)
        prevalence = prev_codes[keep_rows].mean()
        code_columns = list(prevalence[(prevalence >= 0.02) & (prevalence.index <= 44)].index)
    for c in code_columns:
        out[f"prev_code_{c}"] = (prev_codes[c].to_numpy() if c in prev_codes
                                 else np.zeros(len(table)))

    # facility type and risk level (both set before the inspection)
    ft = table["Facility_Type"].fillna("").str.strip()
    for f in FACILITY_TYPES:
        out[f"facility_{f.lower().replace(' ', '_')}"] = (ft == f).astype(float).to_numpy()
    risk = table["Risk"].fillna("").str.extract(r"Risk (\d)")[0].astype(float)
    out["risk_level"] = risk.fillna(1).to_numpy()

    # nearby restaurants' recent critical rate (other licenses' canvass visits, 180 days)
    rate, n = nearby_rate(table["Latitude"].to_numpy(float), table["Longitude"].to_numpy(float),
                          day, lic, can, radius_km=0.5, lookback=180)
    out["nearby_crit_rate_180d"] = rate
    out["nearby_n_180d"] = n
    return out


def nearby_rate(lat, lon, day, lic, events: pd.DataFrame, radius_km=0.5, lookback=180,
                prior_n=10.0):
    """Smoothed critical rate of other licenses' visits within ``radius_km`` before ``day``."""
    ev = events.dropna(subset=["Latitude", "Longitude"]).sort_values("day", kind="stable")
    e_day = ev["day"].to_numpy()
    e_lat = ev["Latitude"].to_numpy(float)
    e_lon = ev["Longitude"].to_numpy(float)
    e_lic = ev["License"].astype("int64").to_numpy()
    e_crit = ev["crit"].to_numpy(float)
    rate = np.full(len(day), np.nan)
    count = np.zeros(len(day))
    kx = 111.32 * np.cos(np.deg2rad(41.85))  # km per degree of longitude at Chicago
    ky = 110.57
    ok = ~(np.isnan(lat) | np.isnan(lon))
    for d in np.unique(day[ok]):
        rows = np.flatnonzero(ok & (day == d))
        lo = np.searchsorted(e_day, d - lookback, side="left")
        hi = np.searchsorted(e_day, d, side="left")
        if hi <= lo:
            continue
        base = e_crit[lo:hi].mean()
        dx = (lon[rows, None] - e_lon[None, lo:hi]) * kx
        dy = (lat[rows, None] - e_lat[None, lo:hi]) * ky
        near = (dx * dx + dy * dy <= radius_km**2) & (lic[rows, None] != e_lic[None, lo:hi])
        n = near.sum(axis=1)
        k = (near * e_crit[None, lo:hi]).sum(axis=1)
        rate[rows] = (k + prior_n * base) / (n + prior_n)
        count[rows] = n
    return rate, count
