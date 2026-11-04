"""Tests for the glmnet port and the validation helpers."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from inspections import models as M


def synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 3, n)
    X = pd.DataFrame({"Inspector_a": (g == 0) * 1.0, "Inspector_b": (g == 1) * 1.0,
                      "Inspector_c": (g == 2) * 1.0, "x1": rng.normal(size=n),
                      "x2": rng.binomial(1, 0.3, n) * 1.0})
    eta = -1.5 + np.array([0.8, -0.6, 0.0])[g] + 0.5 * X["x1"] + 0.4 * X["x2"]
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    return X, y


def test_unpenalized_fit_equals_maximum_likelihood():
    X, y = synthetic()
    Xn = X[["x1", "x2"]]
    m = M.GlmnetLogit(penalty_factor=np.ones(2)).fit(Xn, y, lambdas=[0.0])
    ref = sm.Logit(y, sm.add_constant(Xn)).fit(disp=0).params.to_numpy()
    np.testing.assert_allclose(m.coefficients(0).to_numpy(), ref, atol=1e-6)


def test_ridge_port_matches_glum_at_a_fixed_lambda():
    X, y = synthetic()
    pf = M.inspector_penalty(X.columns)
    lam = 0.02
    m = M.GlmnetLogit(penalty_factor=pf).fit(X, y, lambdas=[lam])
    g = M.fit_glum(X, y, pf, lam)
    np.testing.assert_allclose(m.coefficients(0).to_numpy(),
                               np.concatenate([[g.intercept_], g.coef_]), atol=1e-5)


def test_sklearn_route_matches_the_port():
    X, y = synthetic(seed=1)
    pf = M.inspector_penalty(X.columns)
    m = M.GlmnetLogit(penalty_factor=pf).fit(X, y, lambdas=[0.01])
    b0, beta = M.fit_sklearn(X, y, pf, 0.01)
    np.testing.assert_allclose(m.coefficients(0).to_numpy(), np.r_[b0, beta], atol=1e-3)


def test_only_penalized_columns_shrink_and_path_is_decreasing():
    X, y = synthetic(seed=2)
    pf = M.inspector_penalty(X.columns)
    m = M.GlmnetLogit(penalty_factor=pf).fit(X, y)
    assert np.all(np.diff(m.lambdas_) < 0) and len(m.lambdas_) == 100
    assert m.lambdas_[-1] / m.lambdas_[0] == pytest.approx(1e-4)
    first, last = m.coefficients(0), m.coefficients(99)
    insp = [c for c in X.columns if c.startswith("Inspector")]
    assert np.allclose(first[insp], 0)  # glmnet's first fit is at lambda = infinity
    assert np.abs(last[insp]).max() > 0.3
    assert first["x1"] != 0  # unpenalized columns are fitted even at lambda = infinity


def test_cv_glmnet_picks_an_index_on_the_path():
    X, y = synthetic(n=1500, seed=3)
    cv = M.cv_glmnet(X, y, M.inspector_penalty(X.columns), nfolds=5, seed=1)
    assert 0 <= cv.index_1se <= cv.index_min < 100
    assert cv.cvm[cv.index_min] == cv.cvm.min()
    assert sorted(np.unique(cv.foldid)) == [1, 2, 3, 4, 5]


def test_rolling_origin_splits_never_train_on_the_future():
    dates = pd.Series(pd.date_range("2013-01-01", "2014-04-30", freq="D"))
    for tr, va in M.rolling_origin_splits(dates, ["2013-09-01", "2014-01-01"], 61):
        assert dates.iloc[tr].max() < dates.iloc[va].min()
        assert (dates.iloc[va].max() - dates.iloc[va].min()).days <= 60
