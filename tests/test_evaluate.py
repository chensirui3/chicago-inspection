"""Tests for the days-earlier simulation, first-half share and bootstrap."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from inspections import evaluate as ev


def hand_example():
    """Six inspections over three days, solved by hand in test_hand_example."""
    ids = np.array([1, 2, 3, 4, 5, 6])
    dates = pd.to_datetime(["2014-09-02"] * 2 + ["2014-09-03"] * 2 + ["2014-09-04"] * 2)
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.8])
    positives = np.array([0, 0, 0, 1, 1, 0])
    return dates, scores, positives, ids


def test_hand_example():
    dates, scores, positives, ids = hand_example()
    # Riskiest first: 5, 6, 4, 3, 2, 1 take the slots 09-02, 09-02, 09-03, 09-03, 09-04, 09-04.
    sim = pd.to_datetime(ev.simulated_dates(dates, scores, ids), unit="D")
    expected = pd.to_datetime(
        ["2014-09-04", "2014-09-04", "2014-09-03", "2014-09-03", "2014-09-02", "2014-09-02"]
    )
    assert (sim == expected).all()
    # Inspection 4: 09-03 -> 09-03 (0 days). Inspection 5: 09-04 -> 09-02 (2 days).
    np.testing.assert_array_equal(ev.date_diffs(dates, scores, positives, ids), [0.0, 2.0])
    assert ev.days_earlier(dates, scores, positives, ids) == 1.0
    # Half of 6 inspections is 3; only 09-02 (cumulative 2 < 3) is "first half". Actually no
    # violation was found that day; the model puts inspection 5 there: 1 of 2 violations.
    assert ev.first_half_share(dates, scores, positives, ids) == (0.0, 0.5)


def test_ties_are_broken_by_id_not_by_row_order():
    dates = pd.to_datetime(["2014-09-02", "2014-09-03", "2014-09-04", "2014-09-05"])
    ids = np.array([10, 11, 12, 13])
    scores = np.zeros(4)
    positives = np.array([0, 1, 0, 1])
    perm = np.array([3, 1, 0, 2])  # shuffle rows; result must not change
    a = ev.date_diffs(dates, scores, positives, ids)
    b = ev.date_diffs(dates[perm], scores[perm], positives[perm], ids[perm])
    np.testing.assert_array_equal(a, [0.0, 0.0])
    np.testing.assert_array_equal(np.sort(b), [0.0, 0.0])


def random_window(n=500, days=40, rate=0.16, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp("2014-09-02") + pd.to_timedelta(np.sort(rng.integers(0, days, n)), "D")
    ids = rng.permutation(n) + 1000
    positives = (rng.random(n) < rate).astype(int)
    return dates, positives, ids


def test_perfect_model_gives_best_possible_gain():
    dates, positives, ids = random_window()
    d = ev.as_days(dates)
    k = positives.sum()
    best = d[positives == 1].mean() - np.sort(d)[:k].mean()
    assert ev.days_earlier(dates, positives.astype(float), positives, ids) == pytest.approx(best)
    rng = np.random.default_rng(1)
    for _ in range(50):  # no ranking can beat it
        assert ev.days_earlier(dates, rng.random(len(d)), positives, ids) <= best + 1e-9


def test_reversed_scores_give_worst_possible_gain():
    dates, positives, ids = random_window()
    d = ev.as_days(dates)
    k = positives.sum()
    worst = d[positives == 1].mean() - np.sort(d)[-k:].mean()
    assert ev.days_earlier(dates, -positives.astype(float), positives, ids) == pytest.approx(worst)
    assert worst < 0


def test_random_scores_average_near_zero():
    dates, positives, ids = random_window(n=2000, seed=3)
    d = ev.as_days(dates)
    rng = np.random.default_rng(4)
    draws = [ev.days_earlier(dates, rng.random(len(d)), positives, ids) for _ in range(400)]
    # A random order moves each violation to a uniformly random slot, so the expectation is
    # exactly mean(actual date of violations) - mean(all dates), which is ~0 here.
    expected = d[positives == 1].mean() - d.mean()
    assert np.mean(draws) == pytest.approx(expected, abs=0.25)
    assert abs(np.mean(draws)) < 0.6


def test_bin_summary_matches_mean_difference():
    dates, positives, ids = random_window(seed=5)
    scores = np.random.default_rng(6).random(len(ids))
    bins = ev.bin_summary(dates, scores, positives, ids)
    days = ev.as_days(bins["date"])
    weighted = (days * bins["POS"]).sum() / bins["POS"].sum() - (
        days * bins["POS_SIM"]
    ).sum() / bins["POS_SIM"].sum()
    assert weighted == pytest.approx(ev.days_earlier(dates, scores, positives, ids))
    assert bins["NTOT"].iloc[-1] == len(ids)
    assert bins["POSTOT"].iloc[-1] == bins["POSTOT_SIM"].iloc[-1] == positives.sum()


def test_normalized_gini_equals_2auc_minus_1_without_ties():
    rng = np.random.default_rng(7)
    y = (rng.random(800) < 0.2).astype(int)
    s = rng.random(800) + 0.3 * y
    assert ev.normalized_gini(s, y) == pytest.approx(2 * roc_auc_score(y, s) - 1, abs=1e-12)


def test_paired_bootstrap_is_paired():
    dates, positives, ids = random_window(seed=8)
    s = np.random.default_rng(9).random(len(ids))
    boot = ev.bootstrap_days_earlier(dates, positives, ids, {"a": s, "b": s.copy()}, n_boot=50)
    assert (boot["a"] == boot["b"]).all()
    day_boot = ev.bootstrap_days_earlier(
        dates, positives, ids, {"a": s}, n_boot=20, block="day"
    )
    assert day_boot["a"].notna().all()


def test_day_block_resampling_keeps_whole_days():
    groups = np.array([1, 1, 1, 2, 2, 3])
    for idx in ev.bootstrap_indices(len(groups), 30, seed=1, groups=groups):
        picked = groups[idx]
        for g in np.unique(picked):
            assert (picked == g).sum() % (groups == g).sum() == 0
