"""Evaluation: the city's days-earlier simulation, first-half share, standard metrics, bootstrap.

``simulated_dates``/``date_diffs``/``bin_summary``/``normalized_gini`` are Python ports of
CODE/functions/simulated_date_diff_mean.R, simulated_bin_summary.R and gini.R, and
``first_half_share`` ports the end of CODE/30b_glmnet_model_evaluation.R, from the City of
Chicago's food-inspections-evaluation repository.
Copyright, 2014 City of Chicago (MIT-style license; see LICENSE).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def as_days(dates) -> np.ndarray:
    """Dates (datetime64, Timestamps or numbers of days) as float days since 1970-01-01."""
    arr = np.asarray(dates)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[D]").astype("int64").astype("float64")
    if arr.dtype == object:
        return pd.to_datetime(pd.Series(arr)).values.astype("datetime64[D]").astype(float)
    return arr.astype("float64")


def _ids(ids, n: int) -> np.ndarray:
    return np.arange(n) if ids is None else np.asarray(ids)


def ranking_order(scores, ids=None) -> np.ndarray:
    """Row order by descending score; ties broken by ``ids`` ascending.

    The city sorts a data.table keyed by Inspection_ID with R's stable ``order(-score)``, so
    tied scores keep ascending Inspection_ID order.
    """
    s = np.asarray(scores, dtype="float64")
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    return np.lexsort((_ids(ids, len(s)), -s))


def simulated_dates(dates, scores, ids=None) -> np.ndarray:
    """Give the i-th riskiest inspection the i-th earliest of the actual dates (time slots)."""
    d = as_days(dates)
    sim = np.empty_like(d)
    sim[ranking_order(scores, ids)] = np.sort(d, kind="stable")
    return sim


def date_diffs(dates, scores, positives, ids=None) -> np.ndarray:
    """Actual minus simulated date (days) for every inspection with a critical violation."""
    d = as_days(dates)
    pos = np.asarray(positives).astype(bool)
    return d[pos] - simulated_dates(d, scores, ids)[pos]


def days_earlier(dates, scores, positives, ids=None) -> float:
    """The city's headline metric: mean days earlier a critical violation would be found."""
    diffs = date_diffs(dates, scores, positives, ids)
    return float(diffs.mean()) if diffs.size else float("nan")


def bin_summary(dates, scores, positives, ids=None) -> pd.DataFrame:
    """Per-day counts, actual vs simulated (port of simulated_bin_summary.R).

    Columns: date, N (inspections that day), NTOT (cumulative), POS (critical violations found
    that day), POSTOT (cumulative), POS_SIM and POSTOT_SIM (the same under the model's order).
    """
    d = as_days(dates)
    pos = np.asarray(positives).astype(int)
    sim = simulated_dates(d, scores, ids)
    actual = pd.DataFrame({"date": d, "pos": pos}).groupby("date")["pos"].agg(["sum", "size"])
    simulated = pd.DataFrame({"date": sim, "pos": pos}).groupby("date")["pos"].sum()
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(actual.index.values, unit="D"),
            "N": actual["size"].values,
            "POS": actual["sum"].values,
            "POS_SIM": simulated.reindex(actual.index).values,
        }
    )
    out["NTOT"] = out["N"].cumsum()
    out["POSTOT"] = out["POS"].cumsum()
    out["POSTOT_SIM"] = out["POS_SIM"].cumsum()
    return out[["date", "N", "NTOT", "POS", "POSTOT", "POS_SIM", "POSTOT_SIM"]]


def first_half_share(dates, scores, positives, ids=None) -> tuple[float, float]:
    """Share of all critical violations found in the first half of the window, actual vs model.

    Days are split at the midpoint by inspection count; like the city's code, a day counts as
    "first half" only if the cumulative inspection count at its end is strictly below half.
    """
    bins = bin_summary(dates, scores, positives, ids)
    total = bins["POS"].sum()
    first = bins["NTOT"] < bins["N"].sum() / 2
    actual = bins.loc[first, "POS"].sum() / total
    simulated = bins.loc[first, "POS_SIM"].sum() / total
    return float(actual), float(simulated)


def normalized_gini(scores, y) -> float:
    """Normalized Gini of the city's gini.R (approximately 2 * AUC - 1)."""
    s = np.asarray(scores, dtype="float64")
    y = np.asarray(y, dtype="float64")
    n, npos = len(y), y.sum()
    y_sorted = y[np.argsort(-s, kind="stable")]
    best = np.minimum(np.arange(1, n + 1), npos)
    model = np.cumsum(y_sorted)
    random = np.cumsum(np.full(n, npos / n))
    return float((model.sum() - random.sum()) / (best.sum() - random.sum()))


def evaluate_ranking(
    df: pd.DataFrame,
    score,
    date_col: str = "Inspection_Date",
    y_col: str = "criticalFound",
    id_col: str = "Inspection_ID",
) -> dict:
    """Every metric the README reports for one model on one test window."""
    scores = df[score].to_numpy() if isinstance(score, str) else np.asarray(score)
    y = df[y_col].to_numpy().astype(int)
    ids = df[id_col].to_numpy()
    diffs = date_diffs(df[date_col], scores, y, ids)
    fh_actual, fh_model = first_half_share(df[date_col], scores, y, ids)
    out = {
        "n": len(df),
        "n_positive": int(y.sum()),
        "days_earlier": float(diffs.mean()),
        "sd": float(diffs.std(ddof=1)),
        "se": float(diffs.std(ddof=1) / np.sqrt(len(diffs))),
        "median": float(np.median(diffs)),
        "max": float(diffs.max()),
        "n_later": int((diffs < 0).sum()),
        "share_later": float((diffs < 0).mean()),
        "first_half_actual": fh_actual,
        "first_half_model": fh_model,
        "auc": float(roc_auc_score(y, scores)),
        "gini": normalized_gini(scores, y),
    }
    if np.all((scores >= 0) & (scores <= 1)):
        out["brier"] = float(brier_score_loss(y, scores))
        out["log_loss"] = float(log_loss(y, np.clip(scores, 1e-6, 1 - 1e-6)))
    return out


# ----------------------------------------------------------------------------------------------
# Paired bootstrap
# ----------------------------------------------------------------------------------------------
def bootstrap_indices(
    n: int, n_boot: int, seed: int, groups=None
) -> list[np.ndarray]:
    """Row indices for ``n_boot`` resamples, of rows or (when ``groups`` is given) whole days."""
    rng = np.random.default_rng(seed)
    if groups is None:
        return [rng.integers(0, n, n) for _ in range(n_boot)]
    groups = np.asarray(groups)
    labels, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(labels))]
    out = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(labels), len(labels))
        out.append(np.concatenate([members[k] for k in picked]))
    return out


def bootstrap_days_earlier(
    dates,
    positives,
    ids,
    scores: Mapping[str, np.ndarray],
    n_boot: int = 1000,
    seed: int = 0,
    block: str = "inspection",
) -> pd.DataFrame:
    """Re-run the days-earlier simulation for every model on the same resamples.

    Returns one row per replicate and one column per model, so differences between columns are
    paired. ``block="day"`` resamples whole inspection days instead of single inspections.
    """
    d = as_days(dates)
    pos = np.asarray(positives).astype(int)
    ids = np.asarray(ids)
    groups = d if block == "day" else None
    idx_list = bootstrap_indices(len(d), n_boot, seed, groups)
    rows = []
    for idx in idx_list:
        rows.append(
            {
                name: days_earlier(d[idx], np.asarray(s)[idx], pos[idx], ids[idx])
                for name, s in scores.items()
            }
        )
    return pd.DataFrame(rows)


def interval(values, level: float = 0.95) -> tuple[float, float]:
    lo, hi = np.nanpercentile(np.asarray(values), [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


def paired_comparison(boot: pd.DataFrame, model: str, baseline: str) -> dict:
    """Mean and 95% percentile interval of (model - baseline) over paired replicates."""
    diff = boot[model] - boot[baseline]
    lo, hi = interval(diff)
    return {"diff_mean": float(diff.mean()), "diff_lo": lo, "diff_hi": hi,
            "p_le_zero": float((diff <= 0).mean())}


def bootstrap_metric(
    df: pd.DataFrame,
    fn,
    n_boot: int = 1000,
    seed: int = 0,
    groups=None,
) -> np.ndarray:
    """Bootstrap any scalar statistic ``fn(sub_df)`` (used for per-group fairness metrics)."""
    idx_list = bootstrap_indices(len(df), n_boot, seed, groups)
    return np.array([fn(df.iloc[idx]) for idx in idx_list], dtype=float)
