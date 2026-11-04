"""Tests for Extension C: the post-2018 label, the due rule and the data checks."""

import numpy as np
import pandas as pd
import pytest

from inspections import live


def inspections(rows):
    df = pd.DataFrame(rows, columns=["Inspection_ID", "License", "Inspection_Date",
                                     "inspection_type", "Results", "old_ppf"])
    df["Inspection_Date"] = pd.to_datetime(df["Inspection_Date"])
    df["type"] = df["inspection_type"].str.lower()
    df["Facility_Type"] = "Restaurant"
    df["Risk"] = "Risk 1 (High)"
    df["Latitude"], df["Longitude"] = 41.9, -87.6
    return df


def test_label_uses_results_after_july_2018_and_codes_before():
    df = inspections([
        (1, 7, "2018-06-30", "Canvass", "Fail", 0.0),  # old code: a fail is not the label
        (2, 7, "2018-06-29", "Canvass", "Pass", 1.0),  # old code: codes 1-29 present
        (3, 7, "2018-07-01", "Canvass", "Pass w/ Conditions", np.nan),
        (4, 7, "2018-08-01", "Canvass", "Fail", np.nan),
        (5, 7, "2018-09-01", "Canvass", "Pass", np.nan),
        (6, 7, "2018-10-01", "Canvass", "Out of Business", np.nan),
        (7, 7, "2018-10-02", "Canvass", "No Entry", np.nan),
    ])
    v = live.visits(df).set_index("Inspection_ID")
    assert v["ppf"].to_dict() == {1: 0.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.0}


def test_due_rule_follows_the_risk_level_targets():
    today = pd.Timestamp("2026-11-01")
    df = inspections([
        (1, 100, "2026-03-01", "Canvass", "Pass", np.nan),  # risk 1, 245 days ago -> due
        (2, 200, "2026-09-01", "Canvass", "Pass", np.nan),  # risk 1, 61 days ago -> not due
    ])
    lic = pd.DataFrame({"license_number": [100, 200, 300], "account_number": [1, 2, 3],
                        "license_description": "Retail Food Establishment",
                        "license_start_date": pd.Timestamp("2025-01-01"),
                        "expiration_date": pd.Timestamp("2027-01-01"),
                        "latitude": 41.9, "longitude": -87.6})
    due = live.due_restaurants({"inspections": df, "licenses": lic}, today)
    assert due["License"].tolist() == [100]  # 300 has no inspection record to describe it


def _down(*args, **kwargs):
    import requests

    raise requests.ConnectTimeout("service down")


def test_weather_falls_back_from_ncei_to_acis(tmp_path, monkeypatch):
    from inspections import load

    monkeypatch.setattr(live, "RAW", tmp_path)
    monkeypatch.setattr(load, "noaa_daily", _down)
    monkeypatch.setattr(load, "acis_daily", lambda *a, **k: pd.DataFrame({
        "date": pd.to_datetime(["2026-10-01", "2026-10-02"]),
        "temperatureMax": [61.0, np.nan]}))
    status = live.fetch_weather()
    assert "NCEI unreachable" in status and "used NOAA ACIS" in status
    saved = pd.read_parquet(tmp_path / "weather.parquet")
    assert saved["temperatureMax"].tolist() == [61.0]  # the missing day is not stored


def test_weather_outage_uses_cache_or_leaves_temperature_out(tmp_path, monkeypatch):
    from inspections import load

    monkeypatch.setattr(live, "RAW", tmp_path)
    monkeypatch.setattr(load, "noaa_daily", _down)
    monkeypatch.setattr(load, "acis_daily", _down)
    assert live.fetch_weather().endswith("temperature left out tonight")  # first run, no cache
    pd.DataFrame({"date": pd.to_datetime(["2026-10-01"]), "temperatureMax": [60.0]}).to_parquet(
        tmp_path / "weather.parquet")
    assert live.fetch_weather().endswith("using cached weather")


def test_acis_response_is_parsed_with_missing_days(monkeypatch):
    from inspections import load

    class Resp:
        text = '{"meta": {}, "data": [["2026-10-30", "58"], ["2026-10-31", "M"]]}'

        def raise_for_status(self):
            pass

    class Session:
        def get(self, url, params=None, timeout=None):
            assert params["sid"] == "USW00094846" and params["elems"] == "maxt"
            return Resp()

    monkeypatch.setattr(load, "_session", lambda: Session())
    df = load.acis_daily("2026-10-30", "2026-10-31")
    assert df["temperatureMax"].iloc[0] == 58 and np.isnan(df["temperatureMax"].iloc[1])


def test_model_trains_without_temperature_when_no_weather_exists():
    rng = np.random.default_rng(0)
    n = 400
    table = pd.DataFrame({
        "Inspection_ID": np.arange(n), "License": np.arange(n),
        "Inspection_Date": pd.Timestamp("2025-01-01"), "Latitude": 41.9, "Longitude": -87.6,
        "past_ppf": rng.integers(0, 2, n).astype(float),
        "n_insp_2y": rng.integers(0, 5, n).astype(float),
        "temperatureMax": np.nan, "pandemic": False,
    })
    table["ppf"] = (rng.random(n) < 0.3 + 0.3 * table["past_ppf"]).astype(float)
    model = live.fit_live_model(table)
    assert "temperatureMax" not in model.columns
    p = model.predict(table)
    assert np.all((p > 0) & (p < 1))


def test_validation_fails_loudly_on_stale_data():
    today = pd.Timestamp("2026-11-01")
    insp = inspections([(1, 1, "2026-09-01", "Canvass", "Pass", np.nan)])
    for c in live.INSPECTION_COLS:
        insp[c] = 0
    lic = pd.DataFrame({c: [0] for c in live.LICENSE_COLS})
    obs = {"burglary": pd.DataFrame({"Date": [today]})}
    with pytest.raises(live.StaleDataError, match="newest date"):
        live.validate({"inspections": insp, "licenses": lic, "obs": obs}, today)
