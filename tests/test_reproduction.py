"""End-to-end checks against the city's published numbers (need the data snapshot)."""

import pytest

from inspections import config, core1, evaluate, features
from inspections.load import load_rds

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def published():
    return features.build_city_table("published")


def test_row_counts_match_the_published_evaluation(published):
    test = published[published["Test"]]
    assert (~published["Test"]).sum() == config.PUBLISHED["n_train"]
    assert len(test) == config.PUBLISHED["n_test"]
    assert test["criticalFound"].sum() == config.PUBLISHED["n_positive"]
    assert str(test["Inspection_Date"].min().date()) == "2014-09-02"
    assert str(test["Inspection_Date"].max().date()) == "2014-10-31"


def test_python_port_reproduces_7_44_days(published):
    train, test = published[~published["Test"]], published[published["Test"]]
    m, k, _ = core1.fit_city_model(train)
    r = evaluate.evaluate_ranking(test, core1.score_city_model(m, k, test))
    assert round(r["days_earlier"], 2) == config.PUBLISHED["days_earlier"]
    assert r["n_later"] == config.PUBLISHED["n_later"]
    assert r["first_half_actual"] == pytest.approx(config.PUBLISHED["first_half_actual"])
    assert r["first_half_model"] == pytest.approx(config.PUBLISHED["first_half_model"])
    coef = m.coefficients(k)
    for name, value in config.PUBLISHED_COEFFICIENTS.items():
        assert coef[name] == pytest.approx(value, abs=0.01)


def test_metric_reproduces_the_city_xgboost_7_79_days():
    xgb = load_rds("30_xgboost_data")
    test = xgb[xgb["Test"]].sort_values("Inspection_ID")
    days = evaluate.days_earlier(test["Inspection_Date"], test["score"],
                                 test["criticalFound"], test["Inspection_ID"])
    assert round(days, 2) == config.PUBLISHED["xgboost_days_earlier"]
