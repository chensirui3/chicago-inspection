"""Tests for the feature ports (weather, violations, history, density) and new features."""

import numpy as np
import pandas as pd
import pytest

from inspections import features as F


def test_weather_3day_averages_the_three_previous_days():
    w = pd.DataFrame({"date": pd.date_range("2014-09-01", periods=6),
                      "temperatureMax": [60.0, 63.0, 66.0, 69.0, 72.0, 75.0]})
    out = F.weather_3day(w).set_index("Inspection_Date")["temperatureMax"]
    # 09-04 averages 09-01..09-03; 09-06 averages 09-03..09-05; first three days have no value
    assert out.loc["2014-09-04"] == pytest.approx(63.0)
    assert out.loc["2014-09-06"] == pytest.approx(69.0)
    assert list(out.index) == list(pd.date_range("2014-09-04", periods=3))


def test_violation_codes_parse_leading_numbers():
    text = ("32. FOOD AND NON-FOOD CONTACT SURFACES - Comments: CLEAN | 33. FOOD AND NON-FOOD "
            "CONTACT EQUIPMENT - Comments: x | 3. POTENTIALLY HAZARDOUS FOOD - Comments: 41F")
    assert F.violation_codes(text) == [32, 33, 3]
    assert F.violation_codes(None) == []
    assert F.violation_codes(np.nan) == []


def test_violation_types_use_the_pre2018_ranges():
    vio = F.violation_matrix(pd.Series(["1. A | 14. B | 15. C", "29. D | 30. E | 44. F | 70. G",
                                        None]))
    t = F.violation_types(vio)
    assert t["criticalCount"].tolist() == [2, 0, 0]
    assert t["seriousCount"].tolist() == [1, 1, 0]
    assert t["minorCount"].tolist() == [0, 2, 0]  # code 70 counts in no category


def test_history_features_shift_within_license_and_cap_time():
    insp = pd.DataFrame({
        "Inspection_ID": [3, 1, 2, 4],
        "License": [10, 10, 10, 20],
        "Inspection_Date": pd.to_datetime(["2013-06-01", "2011-10-01", "2012-04-01",
                                           "2012-01-01"]),
        "Results": ["Pass", "Fail", "Pass", "Pass"],
        "criticalCount": [0, 2, 1, 0],
        "seriousCount": [1, 0, 0, 0],
        "minorCount": [0, 0, 0, 3],
    })
    h = F.history_features(insp).set_index("Inspection_ID")
    assert h.loc[1, "firstRecord"] == 1 and h.loc[1, "timeSinceLast"] == 2
    assert h.loc[2, "pastCritical"] == 2 and h.loc[2, "pastFail"] == 1
    assert h.loc[2, "timeSinceLast"] == pytest.approx(183 / 365)
    assert h.loc[3, "pastCritical"] == 1 and h.loc[3, "pastFail"] == 0
    assert h.loc[4, "firstRecord"] == 1  # another license starts fresh


def test_history_ties_on_the_same_day_follow_inspection_id():
    insp = pd.DataFrame({"Inspection_ID": [8, 5], "License": [1, 1],
                         "Inspection_Date": pd.to_datetime(["2013-01-01", "2013-01-01"]),
                         "Results": ["Pass", "Pass"], "criticalCount": [0, 4],
                         "seriousCount": [0, 0], "minorCount": [0, 0]})
    h = F.history_features(insp).set_index("Inspection_ID")
    assert h.loc[8, "pastCritical"] == 4  # id 5 sorts first, so it is 8's previous visit


def test_kde_matches_the_formula_of_the_city_code():
    h = 0.01 / 4
    one = F.kde_heat([41.9], [-87.6], np.array([41.9]), np.array([-87.6]))
    assert one[0] == pytest.approx(1 / (2 * np.pi * h * h))
    two = F.kde_heat([41.9], [-87.6], np.array([41.9, 41.9 + h]), np.array([-87.6, -87.6]))
    expected = (1 + np.exp(-0.5)) / (2 * np.pi) / (2 * h * h)
    assert two[0] == pytest.approx(expected)


def test_heat_window_includes_day_minus_90_and_the_day_itself():
    pts = pd.DataFrame({"Inspection_Date": pd.to_datetime(["2014-01-01"]),
                        "Latitude": [41.9], "Longitude": [-87.6]})
    dates = pd.to_datetime(["2013-10-03", "2013-10-02", "2014-01-01", "2014-01-02"])
    obs = pd.DataFrame({"Date": dates, "Latitude": 41.9, "Longitude": -87.6})
    # 2013-10-03 is exactly 90 days before; 2013-10-02 (91 days) and 2014-01-02 are outside
    v = F.heat_values(pts, obs.sort_values("Date").reset_index(drop=True))
    h = 0.01 / 4
    assert v[0] == pytest.approx(1 / (2 * np.pi * h * h))  # two identical points, n = 2


def test_event_index_counts_strictly_before_the_day():
    idx = F.EventIndex([1, 1, 1, 2], [100, 200, 300, 150], {"x": np.array([1, 0, 1, 1.0])})
    assert idx.total("x", [1], [300])[0] == 1  # the event on day 300 is not "before"
    assert idx.total("x", [1], [301])[0] == 2
    assert idx.total("x", [1], [301], lookback=150)[0] == 1  # window [151, 301)
    assert idx.total("x", [3], [301])[0] == 0
    assert idx.last([1], [250])[0] >= 0 and idx.last([2], [100])[0] == -1


def test_city_design_matrix_applies_the_city_caps():
    row = pd.DataFrame({"sanitarian": ["purple"], "pastSerious": [3], "pastCritical": [2],
                        "timeSinceLast": [1.5], "ageAtInspection": [4.2],
                        "consumption_on_premises_incidental_activity": [1], "tobacco": [0],
                        "temperatureMax": [70.0], "heat_burglary": [120.0],
                        "heat_sanitation": [10.0], "heat_garbage": [99.0]})
    X = F.city_design_matrix(row).iloc[0]
    assert X["Inspector_purple"] == 1 and X["Inspector_blue"] == 0
    assert X["pastSerious"] == 1 and X["pastCritical"] == 1 and X["ageAtInspection"] == 1
    assert X["heat_burglary"] == 70 and X["heat_garbage"] == 50 and X["heat_sanitation"] == 10


def test_filter_keeps_completed_canvass_inspections_after_sept_2011():
    food = pd.DataFrame({
        "Inspection_ID": [1, 2, 3, 4, 5, 5, 6],
        "License": [1, 2, 0, 4, 5, 5, 6],
        "Inspection_Date": pd.to_datetime(["2011-09-01", "2012-01-01", "2012-01-01",
                                           "2012-01-01", "2012-01-01", "2012-02-01",
                                           "2012-01-01"]),
        "Inspection_Type": ["Canvass", "Canvass", "Canvass", "Complaint", "Canvass", "Canvass",
                            "Canvass"],
        "Results": ["Pass", "Fail", "Pass", "Pass", "Pass", "Pass", "No Entry"],
    })
    assert F.filter_food_inspections(food)["Inspection_ID"].tolist() == [2, 5]
