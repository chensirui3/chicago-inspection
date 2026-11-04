"""Extension C: retrain and score every night on current City of Chicago data.

New label (post-July-2018 food code): an inspection "found a Priority or Priority Foundation
violation" when its result is **Fail** or **Pass w/ Conditions** (the 2019 CDPH inspection FAQ:
Pass = no P/PF violations; Pass w/ Conditions = P/PF found and corrected on site; Fail = P/PF
not corrected). It is broader than the 2014 target (critical only), so its numbers are not
comparable with the 7.44 days. For history before July 1, 2018 the old equivalent is "any
critical or serious violation" (codes 1-29); old results are not used there.

No inspector features (identities are not public). Published output is aggregate only: counts
and mean scores by community area plus model diagnostics, never a list of named restaurants.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, evaluate, features
from .load import socrata_frame

LIVE_DIR = config.DATA_DIR / "live"
RAW = LIVE_DIR / "raw"
CODE_CHANGE = pd.Timestamp(config.FOOD_CODE_CHANGE)
HISTORY_START = "2016-07-01"  # two years of history before the new code
DENSITY_START = pd.Timestamp("2018-12-18")  # the new 311 system starts here
MODEL_START = DENSITY_START + pd.Timedelta(days=90)  # first day with a full 90-day window
PANDEMIC = (pd.Timestamp("2020-03-16"), pd.Timestamp("2021-06-30"))  # see monthly chart
DROP_RESULTS = ("Out of Business", "No Entry", "Business Not Located", "Not Ready")
PPF_RESULTS = ("Fail", "Pass w/ Conditions")
SANITATION = "Sanitation Code Violation"
GARBAGE = "Garbage Cart Maintenance"
DUE_DAYS = {1: 182, 2: 365, 3: 730}  # city targets: twice a year, yearly, every other year


class StaleDataError(RuntimeError):
    """Raised when the portal data is too old or its schema changed: fail loudly."""


# ----------------------------------------------------------------------------------------------
# 1. Fetch (incremental) and 2. validate
# ----------------------------------------------------------------------------------------------
INSPECTION_COLS = ["inspection_id", "license_", "facility_type", "risk", "inspection_date",
                   "inspection_type", "results", "latitude", "longitude"]
LICENSE_COLS = ["id", "license_number", "account_number", "license_description",
                "license_start_date", "expiration_date", "license_status", "community_area",
                "latitude", "longitude"]


def _state_path() -> Path:
    return LIVE_DIR / "state.json"


def load_state() -> dict:
    p = _state_path()
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(state: dict) -> None:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    _state_path().write_text(json.dumps(state, indent=2, default=str))


def _upsert(name: str, new: pd.DataFrame, key: str) -> pd.DataFrame:
    path = RAW / f"{name}.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    df = pd.concat([old, new], ignore_index=True) if len(old) else new
    if len(df):
        df = df.drop_duplicates(subset=[key], keep="last").reset_index(drop=True)
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def _changed_since(since: str | None) -> str:
    return f" AND :updated_at > '{since}'" if since else ""


def fetch_all(since: str | None = None) -> dict[str, int]:
    """Pull rows changed since ``since`` (ISO timestamp) or everything when ``since`` is None."""
    ch = _changed_since(since)
    n = {}
    # Food inspections: pre-code-change rows need the violation text (codes 1-29), later ones not.
    post = socrata_frame("food_inspections", select=",".join(INSPECTION_COLS),
                         where=f"inspection_date >= '{config.FOOD_CODE_CHANGE}'{ch}")
    pre = socrata_frame("food_inspections", select=",".join(INSPECTION_COLS + ["violations"]),
                        where=f"inspection_date >= '{HISTORY_START}' AND inspection_date < "
                              f"'{config.FOOD_CODE_CHANGE}'{ch}")
    if len(pre):
        codes = pre["violations"].map(features.violation_codes)
        pre["old_ppf"] = codes.map(lambda c: float(any(1 <= x <= 29 for x in c)))
        pre = pre.drop(columns=["violations"])
    insp = pd.concat([d for d in (pre, post) if len(d)], ignore_index=True) if (
        len(pre) or len(post)) else pd.DataFrame(columns=INSPECTION_COLS)
    n["inspections"] = len(_upsert("inspections", insp, "inspection_id"))

    descs = "','".join(["Retail Food Establishment", "Tobacco",
                        "Consumption on Premises - Incidental Activity"])
    lic = socrata_frame("business_licenses", select=",".join(LICENSE_COLS),
                        where=f"license_description in ('{descs}'){ch}")
    n["licenses"] = len(_upsert("licenses", lic, "id"))

    crimes = socrata_frame("crimes", select="id,date,latitude,longitude",
                           where=f"primary_type = 'BURGLARY' AND date >= '2018-09-01'{ch}")
    n["burglaries"] = len(_upsert("burglaries", crimes, "id"))
    # One request type at a time and in smaller pages: 311 is the portal's largest dataset and
    # the slowest to filter.
    parts = [socrata_frame("service_requests_311", page_size=25_000,
                           select="sr_number,sr_type,status,created_date,duplicate,latitude,"
                                  "longitude",
                           where=f"sr_type = '{t}' AND created_date >= "
                                 f"'{DENSITY_START:%Y-%m-%d}'{ch}")
             for t in (SANITATION, GARBAGE)]
    parts = [p for p in parts if len(p)]
    sr = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    n["service_requests"] = len(_upsert("service_requests", sr, "sr_number"))
    return n


def fetch_weather() -> str:
    """Append recent days of O'Hare's daily high to the cached series.

    Sources in order: NOAA NCEI, then NOAA's Regional Climate Centers (ACIS, the same
    GHCN-Daily readings), then the cache. Weather is auxiliary (on a scoring day it shifts every
    score equally), so if all three fail the job trains that night's model without temperature
    instead of failing. The status is published with the other data checks.
    """
    import requests

    from . import load

    path = RAW / "weather.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(
        columns=["date", "temperatureMax"])
    start = HISTORY_START if old.empty else (
        pd.to_datetime(old["date"]).max() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    new, failed = None, []
    for name, fetch in (("NOAA NCEI", lambda: load.noaa_daily(start, end, attempts=2)),
                        ("NOAA ACIS", lambda: load.acis_daily(start, end))):
        try:
            new = fetch()
            break
        except (requests.RequestException, ValueError, KeyError, IndexError) as e:
            failed.append(f"{name} unreachable ({type(e).__name__})")
    if new is None:
        tail = "using cached weather" if len(old) else "temperature left out tonight"
        return "; ".join([*failed, tail])
    new = new.dropna(subset=["temperatureMax"])  # a missing day never overwrites a reading
    w = pd.concat([old, new], ignore_index=True) if len(old) else new
    w["date"] = pd.to_datetime(w["date"])
    w = w.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    RAW.mkdir(parents=True, exist_ok=True)
    w.to_parquet(path, index=False)
    return "; ".join([*failed, "used NOAA ACIS"]) if failed else "ok"


def load_raw() -> dict[str, pd.DataFrame]:
    def num(s):
        return pd.to_numeric(s, errors="coerce")

    insp = pd.read_parquet(RAW / "inspections.parquet")
    insp["Inspection_ID"] = num(insp["inspection_id"]).astype("Int64")
    insp["License"] = num(insp["license_"]).astype("Int64")
    insp["Inspection_Date"] = pd.to_datetime(insp["inspection_date"]).dt.normalize()
    insp["Latitude"], insp["Longitude"] = num(insp["latitude"]), num(insp["longitude"])
    insp["Results"] = insp["results"].astype(str)
    insp["Facility_Type"] = insp["facility_type"].astype("string")
    insp["Risk"] = insp["risk"].astype("string")
    insp["type"] = insp["inspection_type"].fillna("").str.strip().str.lower()
    lic = pd.read_parquet(RAW / "licenses.parquet")
    for c in ("license_number", "account_number", "community_area"):
        lic[c] = num(lic[c]).astype("Int64")
    for c in ("license_start_date", "expiration_date"):
        lic[c] = pd.to_datetime(lic[c], errors="coerce").dt.normalize()
    lic["latitude"], lic["longitude"] = num(lic["latitude"]), num(lic["longitude"])
    obs = {}
    b = pd.read_parquet(RAW / "burglaries.parquet")
    obs["burglary"] = pd.DataFrame({"Date": pd.to_datetime(b["date"]).dt.normalize(),
                                    "Latitude": num(b["latitude"]),
                                    "Longitude": num(b["longitude"])})
    sr = pd.read_parquet(RAW / "service_requests.parquet")
    sr = sr[(sr["duplicate"].astype(str).str.lower() != "true") & (sr["status"] != "Canceled")]
    for kind, t in (("sanitation", SANITATION), ("garbage", GARBAGE)):
        s = sr[sr["sr_type"] == t]
        obs[kind] = pd.DataFrame({"Date": pd.to_datetime(s["created_date"]).dt.normalize(),
                                  "Latitude": num(s["latitude"]),
                                  "Longitude": num(s["longitude"])})
    obs = {k: v.dropna().sort_values("Date").reset_index(drop=True) for k, v in obs.items()}
    weather = pd.read_parquet(RAW / "weather.parquet") if (RAW / "weather.parquet").exists() \
        else pd.DataFrame(columns=["date", "temperatureMax"])
    return {"inspections": insp, "licenses": lic, "obs": obs, "weather": weather}


def validate(raw: dict[str, pd.DataFrame], today: pd.Timestamp, max_age_days: int = 21) -> dict:
    """Check schema, row counts and freshness; raise StaleDataError on any failure."""
    insp, lic = raw["inspections"], raw["licenses"]
    problems = []
    for name, df, cols in (("inspections", insp, INSPECTION_COLS), ("licenses", lic,
                                                                    LICENSE_COLS)):
        missing = set(cols) - set(df.columns)
        if missing:
            problems.append(f"{name}: missing columns {sorted(missing)}")
    expected_min = {"inspections": 100_000, "licenses": 20_000}
    for name, df in (("inspections", insp), ("licenses", lic)):
        if len(df) < expected_min[name]:
            problems.append(f"{name}: only {len(df):,} rows (expected >= {expected_min[name]:,})")
    newest = insp["Inspection_Date"].max()
    if pd.isna(newest) or (today - newest).days > max_age_days:
        problems.append(f"inspections: newest date {newest} is more than {max_age_days} days old")
    for k, o in raw["obs"].items():
        if len(o) == 0 or (today - o["Date"].max()).days > max_age_days:
            problems.append(f"{k}: no recent rows (newest {o['Date'].max() if len(o) else None})")
    if problems:
        raise StaleDataError("; ".join(problems))
    rfe = lic[lic["license_description"] == "Retail Food Establishment"]
    canv = insp[insp["type"] == "canvass"]
    match = canv["License"].isin(set(rfe["license_number"].dropna())).mean()
    return {"inspections": len(insp), "licenses": len(lic), "newest_inspection": newest,
            "license_match_rate": float(match),
            **{f"{k}_rows": len(v) for k, v in raw["obs"].items()}}


# ----------------------------------------------------------------------------------------------
# 3. Labels and features
# ----------------------------------------------------------------------------------------------
def visits(insp: pd.DataFrame) -> pd.DataFrame:
    """Completed visits of any type, labelled with the P/PF (or pre-2018 critical/serious) flag."""
    v = insp[insp["Inspection_Date"].notna() & insp["License"].notna() & (insp["License"] != 0)
             & ~insp["Results"].isin(DROP_RESULTS)].copy()
    v = v[~v["Inspection_ID"].duplicated()]
    new = v["Inspection_Date"] >= CODE_CHANGE
    v["ppf"] = np.where(new, v["Results"].isin(PPF_RESULTS).astype(float),
                        pd.to_numeric(v.get("old_ppf"), errors="coerce").fillna(0.0))
    v["fail"] = (v["Results"] == "Fail").astype(float)
    v["kind"] = np.select(
        [v["type"] == "canvass", v["type"].str.contains("re-insp"),
         v["type"].str.contains("complaint") | v["type"].str.contains("food poisoning")],
        ["canvass", "reinspection", "complaint"], "other")
    v["day"] = features._days(v["Inspection_Date"])
    return v.reset_index(drop=True)


def license_features(rows: pd.DataFrame, lic: pd.DataFrame, day: np.ndarray) -> pd.DataFrame:
    """License age > 4 years, and whether the business account holds a liquor-consumption or
    tobacco license on that day (a simplification of the city's name/address match)."""
    rfe = lic[lic["license_description"] == "Retail Food Establishment"]
    first = rfe.groupby("license_number")["license_start_date"].min()
    account = rfe.sort_values("license_start_date").groupby("license_number")[
        "account_number"].last()
    d = pd.to_datetime(day.astype("datetime64[D]"))
    out = pd.DataFrame(index=rows.index)
    start = first.reindex(rows["License"].to_numpy()).to_numpy()
    age = (d - pd.to_datetime(start)).days / 365
    out["license_age_gt4"] = (np.nan_to_num(np.asarray(age, float), nan=0.0) > 4).astype(float)
    acct = pd.array(account.reindex(rows["License"].to_numpy()).to_numpy(), dtype="Int64")
    for flag, desc in (("consumption_on_premises", "Consumption on Premises - Incidental "
                                                  "Activity"), ("tobacco", "Tobacco")):
        held = lic[lic["license_description"] == desc][["account_number", "license_start_date",
                                                        "expiration_date"]].dropna()
        held = held.astype({"account_number": "Int64"})
        m = pd.DataFrame({"account_number": acct, "d": d, "row": np.arange(len(rows))}).merge(
            held, on="account_number", how="inner")
        ok = m[(m["license_start_date"] <= m["d"]) & (m["d"] <= m["expiration_date"])]["row"]
        v = np.zeros(len(rows))
        v[ok.unique()] = 1.0
        out[flag] = v
    return out


def weather_3day_before(weather: pd.DataFrame, day: np.ndarray) -> np.ndarray:
    w = weather.dropna().copy()
    if w.empty:
        return np.full(len(day), np.nan)
    w["day"] = features._days(w["date"])
    s = w.set_index("day")["temperatureMax"]
    vals = [s.reindex(day - k).to_numpy() for k in (1, 2, 3)]
    return np.nanmean(np.vstack(vals), axis=0)


def build_features(rows: pd.DataFrame, raw: dict, as_of: pd.Timestamp | None = None
                   ) -> pd.DataFrame:
    """Features for each row (License, Inspection_Date, Latitude, Longitude, Facility_Type,
    Risk), using only visits and observations dated before the row's date (or ``as_of``)."""
    v = visits(raw["inspections"])
    lic = rows["License"].astype("int64").to_numpy()
    day = (np.full(len(rows), features._days([as_of])[0]) if as_of is not None
           else features._days(rows["Inspection_Date"]))
    kinds = {k: (v["kind"] == k).to_numpy(float) for k in ("canvass", "complaint",
                                                           "reinspection")}
    ppf = v["ppf"].to_numpy()
    idx = features.EventIndex(v["License"], v["day"], {
        "all": np.ones(len(v)), "ppf": ppf, "fail": v["fail"].to_numpy(), **kinds,
        "canvass_ppf": kinds["canvass"] * ppf})
    X = pd.DataFrame(index=rows.index)
    can = v[v["kind"] == "canvass"]
    cidx = features.EventIndex(can["License"], can["day"], {"ppf": can["ppf"].to_numpy(),
                                                            "fail": can["fail"].to_numpy()})
    last = cidx.last(lic, day)
    has = last >= 0
    safe = np.where(has, last, 0)
    X["past_ppf"] = np.where(has, cidx.values["ppf"][safe], 0.0)
    X["past_fail"] = np.where(has, cidx.values["fail"][safe], 0.0)
    X["time_since_last"] = np.where(has, np.minimum((day - cidx.days[safe]) / 365, 2.0), 2.0)
    X["first_record"] = (~has).astype(float)
    X["n_insp_2y"] = idx.total("all", lic, day, 730)
    X["n_ppf_2y"] = idx.total("ppf", lic, day, 730)
    X["n_fail_2y"] = idx.total("fail", lic, day, 730)
    n_can = idx.total("canvass", lic, day)
    X["canvass_ppf_share"] = (idx.total("canvass_ppf", lic, day) + 2 * 0.4) / (n_can + 2)
    X["n_complaint_1y"] = idx.total("complaint", lic, day, 365)
    X["n_reinspection_2y"] = idx.total("reinspection", lic, day, 730)
    lastany = idx.last(lic, day)
    X["days_since_any"] = np.where(lastany >= 0, day - idx.days[np.where(lastany >= 0,
                                                                         lastany, 0)], 730)
    X["days_since_any"] = X["days_since_any"].clip(0, 730).astype(float)
    ft = rows["Facility_Type"].fillna("").str.strip()
    for f in features.FACILITY_TYPES:
        X[f"facility_{f.lower().replace(' ', '_')}"] = (ft == f).astype(float).to_numpy()
    risk = rows["Risk"].fillna("").str.extract(r"Risk (\d)")[0].astype(float)
    X["risk_level"] = risk.fillna(1).to_numpy()
    X = pd.concat([X, license_features(rows, raw["licenses"], day)], axis=1)
    pts = rows[["Latitude", "Longitude"]].copy()
    pts["Inspection_Date"] = pd.to_datetime(day.astype("datetime64[D]"))
    for k, o in raw["obs"].items():
        X[f"heat_{k}"] = features.heat_values(pts.reset_index(drop=True), o).clip(
            max=70 if k != "garbage" else 50)
    rate, n = features.nearby_rate(rows["Latitude"].to_numpy(float),
                                   rows["Longitude"].to_numpy(float), day, lic,
                                   can.assign(crit=can["ppf"]), radius_km=0.5, lookback=180)
    X["nearby_ppf_rate_180d"] = rate
    X["nearby_n_180d"] = n
    X["temperatureMax"] = weather_3day_before(raw["weather"], day)
    return X


def model_table(raw: dict) -> pd.DataFrame:
    """Completed canvass inspections since the density windows are complete, with features."""
    v = visits(raw["inspections"])
    rows = v[(v["kind"] == "canvass") & (v["Inspection_Date"] >= MODEL_START)
             & v["Latitude"].notna()].copy()
    rows = rows.sort_values("Inspection_ID").reset_index(drop=True)
    X = build_features(rows, raw)
    out = pd.concat([rows[["Inspection_ID", "License", "Inspection_Date", "Latitude",
                           "Longitude", "ppf"]], X], axis=1)
    out["pandemic"] = out["Inspection_Date"].between(*PANDEMIC)
    return out


FEATURES_EXCLUDED = {"Inspection_ID", "License", "Inspection_Date", "Latitude", "Longitude",
                     "ppf", "pandemic"}


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in FEATURES_EXCLUDED]


# ----------------------------------------------------------------------------------------------
# Model (the Core 2 family: ridge + LightGBM blend, no inspector)
# ----------------------------------------------------------------------------------------------
@dataclass
class LiveModel:
    ridge: object
    lgbm: object
    columns: list[str]
    medians: pd.Series

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        from .core2 import glm_transform

        X = X[self.columns].fillna(self.medians)
        p1 = self.ridge.predict_proba(glm_transform(X))[:, 1]
        p2 = self.lgbm.predict_proba(X)[:, 1]
        return (p1 + p2) / 2


def fit_live_model(train: pd.DataFrame) -> LiveModel:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from . import models
    from .core2 import glm_transform

    # A feature with no values at all (temperature on a night every weather source is down)
    # is left out of tonight's model rather than failing the job.
    cols = [c for c in feature_columns(train) if train[c].notna().any()]
    X = train[cols]
    med = X.median()
    X = X.fillna(med)
    y = train["ppf"].to_numpy()
    ridge = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000)).fit(
        glm_transform(X), y)
    lgbm = models.fit_lgbm(X, y, {"num_leaves": 7, "n_estimators": 300})
    return LiveModel(ridge, lgbm, cols, med)


def monthly_backtest(table: pd.DataFrame, first_month: str = "2019-09") -> pd.DataFrame:
    """Train on everything before each month (pandemic months excluded), score that month."""
    from sklearn.metrics import roc_auc_score

    rows = []
    months = pd.period_range(first_month, table["Inspection_Date"].max().to_period("M"), freq="M")
    for m in months:
        start, end = m.start_time, m.end_time.normalize() + pd.Timedelta(days=1)
        test = table[(table["Inspection_Date"] >= start) & (table["Inspection_Date"] < end)]
        train = table[(table["Inspection_Date"] < start) & ~table["pandemic"]]
        if len(test) < 100 or test["ppf"].nunique() < 2 or len(train) < 2000:
            continue
        p = fit_live_model(train).predict(test)
        rows.append({
            "month": str(m), "n": len(test), "ppf_rate": test["ppf"].mean(),
            "days_earlier": evaluate.days_earlier(test["Inspection_Date"], p, test["ppf"],
                                                  test["Inspection_ID"]),
            "auc": roc_auc_score(test["ppf"], p), "mean_pred": float(p.mean()),
            "pandemic": bool(test["pandemic"].mean() > 0.5),
            "partial": bool(end > table["Inspection_Date"].max() + pd.Timedelta(days=1)),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# 4. Score restaurants due for a canvass inspection, as of today
# ----------------------------------------------------------------------------------------------
def due_restaurants(raw: dict, today: pd.Timestamp) -> pd.DataFrame:
    lic = raw["licenses"]
    rfe = lic[(lic["license_description"] == "Retail Food Establishment")
              & (lic["license_start_date"] <= today) & (lic["expiration_date"] >= today)]
    rfe = rfe.sort_values("license_start_date").drop_duplicates("license_number", keep="last")
    v = visits(raw["inspections"])
    v = v[v["Inspection_Date"] < today]
    last_any = v.sort_values("Inspection_Date").drop_duplicates("License", keep="last")
    last_can = (v[v["kind"] == "canvass"].sort_values("Inspection_Date")
                .drop_duplicates("License", keep="last").set_index("License")["Inspection_Date"])
    d = rfe.merge(last_any[["License", "Facility_Type", "Risk", "Latitude", "Longitude"]],
                  left_on="license_number", right_on="License", how="inner")
    d["Latitude"] = d["Latitude"].fillna(d["latitude"])
    d["Longitude"] = d["Longitude"].fillna(d["longitude"])
    d["last_canvass"] = last_can.reindex(d["License"].to_numpy()).to_numpy()
    risk = d["Risk"].fillna("").str.extract(r"Risk (\d)")[0].astype(float).fillna(1).astype(int)
    since = (today - pd.to_datetime(d["last_canvass"])).dt.days
    d["due"] = since.isna() | (since >= risk.map(DUE_DAYS))
    d["Inspection_Date"] = today
    return d[d["due"] & d["Latitude"].notna()].reset_index(drop=True)


def area_of(points: pd.DataFrame) -> pd.Series:
    from .core3 import community_areas_for

    return community_areas_for(points[["Latitude", "Longitude"]])


# ----------------------------------------------------------------------------------------------
# 5. Monitor: earlier predictions vs outcomes since
# ----------------------------------------------------------------------------------------------
def monitor(raw: dict, pred_dir: Path, today: pd.Timestamp, days: int = 56) -> dict | None:
    files = sorted(pred_dir.glob("*.parquet"))
    if not files:
        return None
    preds = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    v = visits(raw["inspections"])
    out = v[(v["kind"] == "canvass") & (v["Inspection_Date"] >= today - pd.Timedelta(days=days))]
    m = out.merge(preds, on="License")
    m = m[m["scored_on"] < m["Inspection_Date"]]
    m = m.sort_values("scored_on").drop_duplicates("Inspection_ID", keep="last")
    if len(m) < 50:
        return {"matched": int(len(m))}
    m["bin"] = pd.qcut(m["score"].rank(method="first"), 10, labels=False)
    cal = m.groupby("bin").agg(pred=("score", "mean"), obs=("ppf", "mean"), n=("ppf", "size"))
    return {"matched": int(len(m)), "brier": float(((m["score"] - m["ppf"]) ** 2).mean()),
            "mean_pred": float(m["score"].mean()), "observed": float(m["ppf"].mean()),
            "calibration": cal.reset_index().to_dict(orient="list")}


def draw_monthly(fig, t, df: pd.DataFrame):
    """Two small multiples (never a dual axis): canvass volume and the share finding P/PF."""
    from . import plots

    axes = fig.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0.35})
    for ax, col, ylabel, fmt in [(axes[0], "n", "Canvass inspections", "{x:,.0f}"),
                                 (axes[1], "ppf_rate", "Share with a P/PF violation",
                                  "{x:.0%}")]:
        plots.style_axes(ax, t)
        ax.plot(df["date"], df[col], color=t["series"][0], lw=plots.LW)
        ax.axvspan(*PANDEMIC, color=t["neutral"], alpha=0.25, lw=0)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(plots.matplotlib.ticker.StrMethodFormatter(fmt))
        ax.set_ylim(0, df[col].max() * 1.15)
    axes[0].text(PANDEMIC[0], axes[0].get_ylim()[1], " pandemic", color=t["muted"],
                 fontsize=8.5, va="top")
    plots.title(axes[0], t, "Canvass inspections per month and P/PF rate, 2019 to today",
                "Completed canvass inspections in the live modeling table")


def write_report(bt: pd.DataFrame, monthly: pd.DataFrame, summary: dict) -> dict:
    """README artifacts for Extension C (tables, figures, results.json section)."""
    from . import plots, report

    bt = bt.copy()
    bt["date"] = pd.PeriodIndex(bt["month"], freq="M").to_timestamp()
    full = bt[~bt["partial"]]
    period = np.where(full["pandemic"], "2020-21 (pandemic, excluded from training)",
                      full["month"].str[:4])
    yearly = (full.assign(Period=period)
                  .groupby("Period", sort=True)
                  .agg(Months=("month", "size"), Inspections=("n", "sum"),
                       ppf=("ppf_rate", "mean"), days=("days_earlier", "mean"),
                       auc=("auc", "mean"))
                  .reset_index())
    yearly["Inspections"] = yearly["Inspections"].map("{:,}".format)
    yearly = yearly.rename(columns={"ppf": "P/PF rate", "days": "Mean days earlier per month",
                                    "auc": "Mean AUC"})
    report.save_table(yearly, "ext_c_backtest", {"P/PF rate": ".1%",
                                                 "Mean days earlier per month": ".2f",
                                                 "Mean AUC": ".3f"})
    full.to_csv(config.TABLES_DIR / "ext_c_backtest_monthly.csv", index=False)
    plots.render("ext_c_backtest", plots.draw_lines, df=full, x="date", series=["days_earlier"],
                 labels=["Days earlier"], ylabel="Days earlier within the month",
                 heading="Monthly forward backtest of the live model (P/PF label)",
                 subtitle="Each month scored by a model trained only on earlier months",
                 ref=0.0, shade=(PANDEMIC[0], PANDEMIC[1], "pandemic"))
    m = monthly.copy()
    m["date"] = pd.PeriodIndex(m["month"], freq="M").to_timestamp()
    plots.render("ext_c_monthly", draw_monthly, size=(8, 5.2), df=m.iloc[:-1])
    recent = full[~full["pandemic"]].tail(12)
    ch = summary["checks"]
    text = (
        f"**Label.** Under the 2018 food code a result of *Pass* means no Priority (P) or "
        f"Priority Foundation (PF) violation, *Pass w/ Conditions* means at least one was found "
        f"and corrected on site, *Fail* means at least one could not be corrected. So \"at least "
        f"one P or PF violation\" = Fail or Pass w/ Conditions. It is broader than the 2014 "
        f"critical-only target (about {full['ppf_rate'].mean():.0%} of canvass inspections vs "
        f"15.8% in the 2014 test window), so **these numbers are not comparable with the 7.44 "
        f"days**. History before "
        f"July 2018 uses \"any critical or serious violation\" (codes 1-29), never old results. "
        f"Completed canvass inspections only (Out of Business, No Entry, Business Not Located "
        f"and Not Ready dropped).\n\n"
        f"**Data.** Food Inspections from July 2016 (history), Business Licenses (license age "
        f"> 4 years, and whether the business account holds a liquor-consumption or tobacco "
        f"license), Crimes (burglaries) and 311 Service Requests (\"{SANITATION}\" and "
        f"\"{GARBAGE}\" are the exact type names; duplicates and canceled requests dropped) "
        f"for the 90-day densities, NOAA daily maximum temperature at O'Hare. The new 311 "
        f"system starts Dec 18, 2018, so the modeling table starts {MODEL_START:%Y-%m-%d}: "
        f"{summary['training_rows']:,} canvass inspections outside the pandemic period "
        f"(Mar 16, 2020 to Jun 30, 2021, when volumes collapsed and rates shifted; it is "
        f"excluded from training and flagged in the backtest). "
        f"{ch['license_match_rate']:.0%} of canvass inspections match a business license by "
        f"license number (logged every night).\n\n"
        f"**Forward validation.** Train through one month, score the next, slide month by "
        f"month (same ridge + LightGBM blend as Core 2, no inspector features). Over the last "
        f"12 full months the model finds P/PF violations "
        f"{recent['days_earlier'].mean():.1f} days earlier on average within each month "
        f"(range {recent['days_earlier'].min():.1f} to {recent['days_earlier'].max():.1f}; "
        f"AUC {recent['auc'].mean():.3f}). The days are smaller than a two-month window's "
        f"because each month spans half as many days. The label drifted: "
        f"{full.loc[full['month'].str[:4] == full['month'].iloc[0][:4], 'ppf_rate'].mean():.0%} "
        f"of canvass inspections found a P/PF violation in {full['month'].iloc[0][:4]}, "
        f"{recent['ppf_rate'].mean():.0%} over the last 12 months, so calibration drifts too; "
        f"retraining every night and the calibration monitor exist for that.\n\n"
        f"**Nightly job** (`.github/workflows/nightly.yml`, cron in UTC): fetch rows changed "
        f"since the last run with the app token from the `SOCRATA_APP_TOKEN` secret; validate "
        f"columns, row counts and freshness and fail loudly otherwise; rebuild features as of "
        f"today; retrain; score the {summary['n_due']:,} retail food establishments due for a "
        f"canvass (risk 1 after 6 months, risk 2 after a year, risk 3 after two); publish a "
        f"static GitHub Pages dashboard (map by community area, score distribution, monthly "
        f"backtest, last-updated time) and its CSV/JSON; compare earlier predictions with "
        f"outcomes since for a calibration chart. The numbers in this section are from the run "
        f"of {summary['updated']}; the dashboard shows the latest run.\n\n"
        f"**Publishing responsibly.** The page and its data files show only community-area "
        f"aggregates (areas with fewer than 5 due establishments are suppressed) and model "
        f"diagnostics. A score is a prediction, not an inspection result, so no ranked list of "
        f"named restaurants is published; per-restaurant predictions stay in the private "
        f"Actions cache for monitoring. Individual restaurants' actual results are already on "
        f"the city's portal.")
    out = {"text": text, "recent_days": float(recent["days_earlier"].mean()),
           "recent_auc": float(recent["auc"].mean()), "summary": summary}
    report.save_results("ext_c", out)
    return out


def run_nightly(today: pd.Timestamp | None = None, backtest: bool = False,
                full_refresh: bool = False) -> dict:
    """The nightly job: fetch -> validate -> features -> retrain -> score -> publish -> monitor."""
    from . import site

    t0 = time.time()
    today = (today or pd.Timestamp.now(tz="America/Chicago").tz_localize(None)).normalize()
    state = {} if full_refresh else load_state()
    since = state.get("last_fetch_utc")
    started = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%S")
    counts = fetch_all(since=since)
    weather_status = fetch_weather()
    raw = load_raw()
    checks = validate(raw, today)
    checks["weather"] = weather_status
    checks["weather_newest"] = raw["weather"]["date"].max() if len(raw["weather"]) else None
    table = model_table(raw)
    model = fit_live_model(table[~table["pandemic"]])
    due = due_restaurants(raw, today)
    X = build_features(due, raw, as_of=today)
    due["score"] = model.predict(X)
    due["community_area"] = area_of(due).to_numpy()
    pred_dir = LIVE_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    due[["License", "score"]].assign(scored_on=today).to_parquet(
        pred_dir / f"{today:%Y-%m-%d}.parquet", index=False)
    bt_path = LIVE_DIR / "backtest.parquet"
    if backtest or not bt_path.exists():
        monthly_backtest(table).to_parquet(bt_path, index=False)
    bt = pd.read_parquet(bt_path)
    monthly = (table.assign(month=table["Inspection_Date"].dt.to_period("M").astype(str))
               .groupby("month").agg(n=("ppf", "size"), ppf_rate=("ppf", "mean")).reset_index())
    mon = monitor(raw, pred_dir, today)
    summary = {
        "updated": pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d %H:%M %Z"),
        "scored_on": f"{today:%Y-%m-%d}", "n_due": int(len(due)),
        "expected_ppf": float(due["score"].sum()), "mean_score": float(due["score"].mean()),
        "checks": {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in checks.items()},
        "rows_after_fetch": counts, "training_rows": int((~table["pandemic"]).sum()),
        "runtime_s": round(time.time() - t0),
    }
    # Private copies (Actions cache, never published) so the page can be rebuilt offline.
    due[["License", "score", "community_area"]].to_parquet(LIVE_DIR / "due_latest.parquet",
                                                         index=False)
    monthly.to_parquet(LIVE_DIR / "monthly.parquet", index=False)
    (LIVE_DIR / "summary.json").write_text(json.dumps({"summary": summary, "monitor": mon},
                                                      indent=2, default=str))
    site.publish(due, bt, monthly, mon, summary)
    write_report(bt, monthly, summary)
    state.update({"last_fetch_utc": started, "last_success": summary["updated"],
                  "checks": summary["checks"]})
    save_state(state)
    return summary


def publish_only() -> dict:
    """Rebuild the dashboard and README artifacts from the last run's cached outputs."""
    from . import site

    saved = json.loads((LIVE_DIR / "summary.json").read_text())
    due = pd.read_parquet(LIVE_DIR / "due_latest.parquet")
    bt = pd.read_parquet(LIVE_DIR / "backtest.parquet")
    monthly = pd.read_parquet(LIVE_DIR / "monthly.parquet")
    site.publish(due, bt, monthly, saved["monitor"], saved["summary"])
    write_report(bt, monthly, saved["summary"])
    return saved["summary"]
