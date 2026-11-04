"""Model fitting: an exact port of the city's glmnet ridge model, plus LightGBM and helpers.

``GlmnetLogit`` and ``cv_glmnet`` reproduce R glmnet's binomial ridge (alpha = 0) with
per-feature penalty factors: the same objective, standardization, penalty-factor rescaling,
lambda sequence and cross-validation rule as ``glmnet``/``cv.glmnet`` (glmnet 2.x/4.x), used by
CODE/30a_glmnet_model.R of the City of Chicago's food-inspections-evaluation repository
(Copyright, 2014 City of Chicago; MIT-style license). ``fit_glum`` and ``fit_sklearn`` are the
two library routes suggested in the project guide and are kept as cross-checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit

# ----------------------------------------------------------------------------------------------
# glmnet port
# ----------------------------------------------------------------------------------------------
_PROB_MIN = 1e-5  # glmnet clips held-out probabilities to [1e-5, 1 - 1e-5] for the deviance


def _newton_logit(X1, y, ridge, beta0, tol=1e-12, max_iter=200):
    """Minimize -(1/n) loglik(X1 @ b) + 0.5 * sum(ridge * b**2) by damped Newton steps.

    ``X1`` includes the intercept column; ``ridge`` is the per-coefficient penalty weight.
    """
    n = len(y)
    b = beta0.copy()

    def objective(bb):
        eta = X1 @ bb
        return (np.logaddexp(0.0, eta).sum() - y @ eta) / n + 0.5 * (ridge * bb * bb).sum()

    f = objective(b)
    for _ in range(max_iter):
        p = expit(X1 @ b)
        grad = -(X1.T @ (y - p)) / n + ridge * b
        w = p * (1.0 - p)
        hess = (X1.T * w) @ X1 / n + np.diag(ridge)
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while True:
            b_new = b - t * step
            f_new = objective(b_new)
            if f_new <= f + 1e-4 * t * (grad @ -step) or t < 1e-10:
                break
            t *= 0.5
        b, f_old, f = b_new, f, f_new
        if np.max(np.abs(t * step)) < tol or abs(f_old - f) < tol * 1e-3:
            break
    return b


@dataclass
class GlmnetLogit:
    """Binomial ridge regression with glmnet's conventions (``alpha = 0``).

    Minimizes, over standardized columns,
    ``-(1/N) sum_i [y_i eta_i - log(1 + exp(eta_i))] + lambda * sum_j pf_j beta_j^2 / 2``
    where the penalty factors ``pf`` are rescaled to sum to the number of columns, columns are
    standardized with the population SD, and coefficients are reported on the original scale.
    Columns with ``pf = 0`` are unpenalized.
    """

    penalty_factor: np.ndarray | None = None
    nlambda: int = 100
    lambda_min_ratio: float | None = None
    standardize: bool = True
    lambdas_: np.ndarray = field(init=False, repr=False)
    coef_path_: np.ndarray = field(init=False, repr=False)  # (nlambda, p) original scale
    intercept_path_: np.ndarray = field(init=False, repr=False)
    feature_names_: list[str] = field(init=False, repr=False)

    def fit(self, X, y, lambdas=None) -> GlmnetLogit:
        names = list(X.columns) if isinstance(X, pd.DataFrame) else None
        X = np.asarray(X, dtype="float64")
        y = np.asarray(y, dtype="float64")
        n, p = X.shape
        self.feature_names_ = names or [f"x{j}" for j in range(p)]
        pf = np.ones(p) if self.penalty_factor is None else np.asarray(self.penalty_factor, float)
        pf = pf * p / pf.sum()  # glmnet rescales penalty factors to sum to nvars
        mu = X.mean(axis=0) if self.standardize else np.zeros(p)
        sd = X.std(axis=0) if self.standardize else np.ones(p)
        sd[sd == 0] = 1.0
        Xs = (X - mu) / sd
        X1 = np.column_stack([np.ones(n), Xs])
        free = pf == 0

        # lambda = infinity: penalized coefficients are 0, intercept and free columns are fitted
        idx_free = np.concatenate([[0], 1 + np.flatnonzero(free)])
        b_inf = np.zeros(p + 1)
        ybar = np.clip(y.mean(), 1e-9, 1 - 1e-9)
        b_inf[0] = np.log(ybar / (1 - ybar))
        b_inf[idx_free] = _newton_logit(
            X1[:, idx_free], y, np.zeros(len(idx_free)), b_inf[idx_free]
        )

        first_is_infinite = lambdas is None
        if lambdas is None:
            # lambda_max comes from the gradient at the intercept-only model. With alpha = 0
            # glmnet's strong rule admits every variable on the first step, so its Fortran never
            # refreshes that gradient after fitting the unpenalized columns (lognet2n).
            resid = y - y.mean()
            grad = np.abs(Xs.T @ resid) / n
            pen = ~free
            lam_max = np.max(grad[pen] / pf[pen]) / 1e-3  # glmnet uses max(alpha, 1e-3)
            ratio = self.lambda_min_ratio or (1e-4 if n > p else 1e-2)
            lambdas = lam_max * ratio ** (np.arange(self.nlambda) / (self.nlambda - 1))
        lambdas = np.asarray(lambdas, dtype="float64")

        coefs = np.empty((len(lambdas), p + 1))
        b = b_inf.copy()
        for k, lam in enumerate(lambdas):
            if k == 0 and first_is_infinite:
                coefs[k] = b_inf  # glmnet's first fit is at lambda = infinity
                continue
            ridge = np.concatenate([[0.0], lam * pf])
            b = _newton_logit(X1, y, ridge, b)
            coefs[k] = b
        beta = coefs[:, 1:] / sd
        self.lambdas_ = lambdas
        self.coef_path_ = beta
        self.intercept_path_ = coefs[:, 0] - beta @ mu
        return self

    def decision_path(self, X) -> np.ndarray:
        X = np.asarray(X, dtype="float64")
        return X @ self.coef_path_.T + self.intercept_path_

    def predict_proba_path(self, X) -> np.ndarray:
        return expit(self.decision_path(X))

    def predict_proba(self, X, index: int) -> np.ndarray:
        X = np.asarray(X, dtype="float64")
        return expit(X @ self.coef_path_[index] + self.intercept_path_[index])

    def coefficients(self, index: int) -> pd.Series:
        return pd.Series(
            np.concatenate([[self.intercept_path_[index]], self.coef_path_[index]]),
            index=["(Intercept)", *self.feature_names_],
        )


@dataclass
class CVGlmnet:
    model: GlmnetLogit
    cvm: np.ndarray
    cvsd: np.ndarray
    foldid: np.ndarray
    index_min: int
    index_1se: int

    @property
    def lambda_min(self) -> float:
        return float(self.model.lambdas_[self.index_min])

    @property
    def lambda_1se(self) -> float:
        return float(self.model.lambdas_[self.index_1se])

    def predict_proba(self, X, which: str = "min") -> np.ndarray:
        idx = self.index_min if which == "min" else self.index_1se
        return self.model.predict_proba(X, idx)


def r_style_folds(n: int, nfolds: int, rng: np.random.Generator) -> np.ndarray:
    """``sample(rep(seq(nfolds), length = n))`` with a numpy generator."""
    return rng.permutation(np.resize(np.arange(1, nfolds + 1), n))


def cv_glmnet(
    X, y, penalty_factor=None, nfolds: int = 10, seed: int = 0, foldid=None
) -> CVGlmnet:
    """Port of ``cv.glmnet(family = "binomial", alpha = 0, type.measure = "deviance")``."""
    Xa = np.asarray(X, dtype="float64")
    y = np.asarray(y, dtype="float64")
    full = GlmnetLogit(penalty_factor=penalty_factor).fit(X, y)
    lambdas = full.lambdas_
    if foldid is None:
        foldid = r_style_folds(len(y), nfolds, np.random.default_rng(seed))
    folds = np.unique(foldid)
    fold_means = np.empty((len(folds), len(lambdas)))
    fold_sizes = np.empty(len(folds))
    for i, k in enumerate(folds):
        held = foldid == k
        m = GlmnetLogit(penalty_factor=penalty_factor).fit(Xa[~held], y[~held], lambdas=lambdas)
        p = np.clip(m.predict_proba_path(Xa[held]), _PROB_MIN, 1 - _PROB_MIN)
        yy = y[held][:, None]
        dev = -2 * (yy * np.log(p) + (1 - yy) * np.log(1 - p))
        fold_means[i] = dev.mean(axis=0)
        fold_sizes[i] = held.sum()
    w = fold_sizes / fold_sizes.sum()
    cvm = w @ fold_means
    cvsd = np.sqrt((w @ (fold_means - cvm) ** 2) / (len(folds) - 1))
    best = cvm.min()
    index_min = int(np.flatnonzero(cvm <= best)[0])  # largest lambda attaining the minimum
    index_1se = int(np.flatnonzero(cvm <= best + cvsd[index_min])[0])
    return CVGlmnet(full, cvm, cvsd, foldid, index_min, index_1se)


def inspector_penalty(columns) -> np.ndarray:
    """The city's penalty: 1 for inspector dummies, 0 (unpenalized) for everything else."""
    return np.array([1.0 if c.startswith("Inspector") else 0.0 for c in columns])


# ----------------------------------------------------------------------------------------------
# The two library routes from the guide (kept as cross-checks of the port)
# ----------------------------------------------------------------------------------------------
def fit_glum(X, y, penalty_factor, lam: float):
    """glum route: binomial, l1_ratio = 0, per-feature P2 = glmnet's rescaled penalty factors."""
    from glum import GeneralizedLinearRegressor

    pf = np.asarray(penalty_factor, float)
    pf = pf * len(pf) / pf.sum()
    model = GeneralizedLinearRegressor(
        family="binomial",
        alpha=lam,
        l1_ratio=0.0,
        P2=pf,
        scale_predictors=True,
        fit_intercept=True,
        gradient_tol=1e-10,
        max_iter=1000,
    )
    return model.fit(np.asarray(X, float), np.asarray(y, float))


def fit_sklearn(X, y, penalty_factor, lam: float, big: float = 100.0):
    """scikit-learn route: standardize, scale unpenalized columns by ``big`` so their penalty
    is negligible, and scale penalized columns by 1/sqrt(pf) so one C matches glmnet's lambda.

    Returns ``(intercept, coef)`` on the original scale.
    """
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    pf = np.asarray(penalty_factor, float)
    pf = pf * p / pf.sum()
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1
    scale = np.where(pf > 0, 1 / np.sqrt(np.where(pf > 0, pf, 1)), big)
    Z = (X - mu) / sd * scale
    clf = LogisticRegression(C=1.0 / (lam * n), solver="lbfgs", max_iter=10_000, tol=1e-12)
    clf.fit(Z, y)
    beta = clf.coef_[0] * scale / sd
    return float(clf.intercept_[0] - beta @ mu), beta


# ----------------------------------------------------------------------------------------------
# Gradient-boosted trees and time-respecting validation (Core 2)
# ----------------------------------------------------------------------------------------------
def fit_lgbm(X: pd.DataFrame, y, params: dict, seed: int = 0):
    import lightgbm as lgb

    base = {
        "objective": "binary",
        "learning_rate": 0.03,
        "n_estimators": 400,
        "num_leaves": 15,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": seed,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": 4,
        "verbose": -1,
    }
    base.update(params)
    model = lgb.LGBMClassifier(**base)
    model.fit(X, np.asarray(y))
    return model


def rolling_origin_splits(
    dates: pd.Series, cutoffs: list[str], horizon_days: int = 61
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Train on everything before each cutoff, validate on the next ``horizon_days``."""
    d = pd.to_datetime(dates).to_numpy()
    out = []
    for c in cutoffs:
        start = np.datetime64(pd.Timestamp(c))
        end = start + np.timedelta64(horizon_days, "D")
        tr = np.flatnonzero(d < start)
        va = np.flatnonzero((d >= start) & (d < end))
        out.append((tr, va))
    return out


def fit_isotonic(scores, y):
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip")
    return iso.fit(np.asarray(scores, float), np.asarray(y, float))
