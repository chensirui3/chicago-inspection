"""Core 3: how much of the 7.44 days depends on information a scheduler would not have had?

The backtest keeps, for every inspection, the score computed with the inspector who actually did
it, the temperature of its actual date and the time since its previous visit measured to its
actual date. Here we remove the inspector, then rebuild every feature as of 2014-09-02 (the first
day of the window) and rank once, which is what a scheduler could have known.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from . import config, core1, core2, evaluate, features, plots, report
from .load import load_rds

AS_OF = pd.Timestamp("2014-09-02")


# ----------------------------------------------------------------------------------------------
# Features as of a fixed date
# ----------------------------------------------------------------------------------------------
def city_features_as_of(rows: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """The city's feature columns for ``rows`` recomputed as of ``as_of``.

    History uses the city's own universe (filtered canvass visits, CODE/23) strictly before
    ``as_of``; license age is measured to ``as_of``; densities use the 90 days ending
    ``as_of``. License flags are unchanged (they describe the business, not the visit).
    """
    hist = features.rebuild_food_features()
    hist = hist[hist["Inspection_Date"] < as_of]
    last = (hist.sort_values(["License", "Inspection_Date", "Inspection_ID"])
                .groupby("License").tail(1)
                .set_index("License"))
    out = rows.copy()
    prev = last.reindex(rows["License"].to_numpy())
    has = prev["Inspection_Date"].notna().to_numpy()
    out["pastCritical"] = np.where(has, prev["criticalCount"].to_numpy(dtype=float), 0.0)
    out["pastSerious"] = np.where(has, prev["seriousCount"].to_numpy(dtype=float), 0.0)
    gap = (as_of - prev["Inspection_Date"]).dt.days.to_numpy(dtype=float) / 365
    out["timeSinceLast"] = np.where(has, np.minimum(gap, 2.0), 2.0)
    out["firstRecord"] = (~has).astype(float)
    shift_years = (rows["Inspection_Date"] - as_of).dt.days.to_numpy() / 365
    out["ageAtInspection"] = rows["ageAtInspection"].to_numpy() - shift_years
    for kind, col in [("burglary", "heat_burglary"), ("sanitation", "heat_sanitation"),
                      ("garbage", "heat_garbage")]:
        out[col] = features.heat_values(rows, features.heat_observations(kind), as_of=as_of)
    out["temperatureMax"] = np.nan  # not known in advance; the as-of models do not use it
    return out


# ----------------------------------------------------------------------------------------------
# Experiments 1-2: the backtest table
# ----------------------------------------------------------------------------------------------
def _fit_score(spec: core2.Spec, train, test_rows, ext_train, ext_test, temperature: bool):
    Xtr = core2.design(train, ext_train, spec.feature_set, spec.inspector, temperature)
    Xte = core2.design(test_rows, ext_test, spec.feature_set, spec.inspector, temperature)
    p, _ = core2.fit_predict(spec, Xtr, train["criticalFound"], Xte)
    return p


def backtest_table(pub, ext, selected) -> tuple[pd.DataFrame, dict]:
    train, test = pub[~pub["Test"]], pub[pub["Test"]]
    city = core2.Spec("city_glm", "city", True)
    city_ni = core2.Spec("city_glm", "city", False)
    best = core2.Spec(**selected["with_inspector"])
    best_ni = core2.Spec(**selected["without_inspector"])

    test_asof = city_features_as_of(test, AS_OF)
    codes = [int(c.rsplit("_", 1)[1]) for c in ext.columns if c.startswith("prev_code_")]
    ext_asof = features.extended_features(test, as_of=AS_OF, code_columns=codes)
    ext_asof = ext_asof[ext.columns]  # same columns, same order as the training design
    scores = {
        "published": _fit_score(city, train, test, ext, ext, True),
        "city_no_insp": _fit_score(city_ni, train, test, ext, ext, True),
        "city_no_insp_no_temp": _fit_score(city_ni, train, test, ext, ext, False),
        "city_asof": _fit_score(city_ni, train, test_asof, ext, ext_asof, False),
        "best": _fit_score(best, train, test, ext, ext, True),
        "best_no_insp": _fit_score(best_ni, train, test, ext, ext, True),
        "best_asof": _fit_score(best_ni, train, test_asof, ext, ext_asof, False),
    }
    boot = evaluate.bootstrap_days_earlier(
        test["Inspection_Date"], test["criticalFound"], test["Inspection_ID"], scores,
        n_boot=config.N_BOOT, seed=config.SEED)
    labels = {
        "published": ("City's published setup", "Yes", "Actual inspection date"),
        "city_no_insp": ("Same model, no inspector", "No", "Actual inspection date"),
        "city_no_insp_no_temp": ("Same model, no inspector or temperature", "No",
                                 "Actual inspection date"),
        "city_asof": ("Same model, as of day one", "No", "2014-09-02"),
        "best": (f"My best model ({best.label})", "Yes", "Actual inspection date"),
        "best_no_insp": ("My best model, no inspector", "No", "Actual inspection date"),
        "best_asof": ("My best model, as of day one", "No", "2014-09-02"),
    }
    rows, res = [], {}
    for key, p in scores.items():
        r = evaluate.evaluate_ranking(test, p)
        lo, hi = evaluate.interval(boot[key])
        cmp = evaluate.paired_comparison(boot, key, "published")
        res[key] = {**r, "lo": lo, "hi": hi, **cmp}
        name, insp, asof = labels[key]
        rows.append({
            "Setup": name, "Inspector known?": insp, "Features as of": asof,
            "Days earlier (95% interval)": f"{r['days_earlier']:.2f} {report.ci(lo, hi)}",
            "Change vs published [95% CI]": "-" if key == "published" else
            f"{cmp['diff_mean']:+.2f} {report.ci(cmp['diff_lo'], cmp['diff_hi'], '+.2f')}",
            "First-half share": f"{r['first_half_model']:.1%}",
            "Found later": r["n_later"], "AUC": f"{r['auc']:.3f}",
        })
    pd.DataFrame({"Inspection_ID": test["Inspection_ID"].to_numpy(), **scores}).to_parquet(
        config.CACHE_DIR / "core3_scores.parquet", index=False)
    changed = test_asof["timeSinceLast"].to_numpy() != test["timeSinceLast"].to_numpy()
    res["asof_feature_changes"] = {
        "timeSinceLast_mean_actual": float(test["timeSinceLast"].mean()),
        "timeSinceLast_mean_asof": float(test_asof["timeSinceLast"].mean()),
        "share_pastCritical_changed": float(
            (np.minimum(test_asof["pastCritical"], 1).to_numpy()
             != np.minimum(test["pastCritical"], 1).to_numpy()).mean()),
        "share_timeSinceLast_changed": float(changed.mean()),
    }
    return pd.DataFrame(rows), res


# ----------------------------------------------------------------------------------------------
# Experiment 3: strict inspectors or dirty kitchens?
# ----------------------------------------------------------------------------------------------
def cross_fitted_expected(train: pd.DataFrame, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold predictions of the city's model without inspectors (folds grouped by license,
    so a restaurant's own visits never inform its expected rate)."""
    X = features.city_design_matrix(train, inspector=False)
    y = train["criticalFound"].to_numpy()
    out = np.empty(len(train))
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups=train["License"]):
        m = core1.models.GlmnetLogit(penalty_factor=np.ones(X.shape[1])).fit(
            X.iloc[tr], y[tr], lambdas=[0.0])
        out[va] = m.predict_proba(X.iloc[va], 0)
    return out


def stringency(train: pd.DataFrame) -> pd.DataFrame:
    exp = cross_fitted_expected(train)
    df = pd.DataFrame({"cluster": train["sanitarian"].to_numpy(),
                       "y": train["criticalFound"].to_numpy(), "e": exp})
    rows = []
    rng = np.random.default_rng(config.SEED)
    for c in config.INSPECTOR_CLUSTERS:
        d = df[df["cluster"] == c]
        oe = d["y"].sum() / d["e"].sum()
        idx = rng.integers(0, len(d), (config.N_BOOT, len(d)))
        boots = d["y"].to_numpy()[idx].sum(1) / d["e"].to_numpy()[idx].sum(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"cluster": c, "n": len(d), "actual": d["y"].mean(),
                     "expected": d["e"].mean(), "oe": oe, "oe_lo": lo, "oe_hi": hi,
                     "coef": config.PUBLISHED_COEFFICIENTS[f"Inspector_{c}"]})
    return pd.DataFrame(rows).sort_values("oe").reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# Experiment 4: the knock-on effect through pastCritical
# ----------------------------------------------------------------------------------------------
def previous_cluster(pub: pd.DataFrame) -> pd.Series:
    """Inspector cluster of each inspection's previous canvass visit (city universe)."""
    hist = features.rebuild_food_features()[["Inspection_ID", "License", "Inspection_Date"]]
    hist = hist.sort_values(["License", "Inspection_Date", "Inspection_ID"])
    hist["prev_id"] = hist.groupby("License")["Inspection_ID"].shift(1)
    san = load_rds("19_inspector_assignments").rename(
        columns={"inspectionID": "prev_id", "sanitarian": "prev_cluster"})
    hist = hist.merge(san, on="prev_id", how="left")
    return pub[["Inspection_ID"]].merge(hist[["Inspection_ID", "prev_cluster"]],
                                        on="Inspection_ID", how="left")["prev_cluster"]


def knock_on(train: pd.DataFrame, strict: set[str]) -> dict:
    df = train.copy()
    df["prev_cluster"] = previous_cluster(train).to_numpy()
    df = df[df["prev_cluster"].notna() & (df["firstRecord"] == 0)]
    df["prev_strict"] = df["prev_cluster"].isin(strict).astype(float)
    df["pc"] = np.minimum(df["pastCritical"], 1)
    out = {"n": int(len(df))}
    rng = np.random.default_rng(config.SEED)
    for grp, d in [("lenient", df[df["prev_strict"] == 0]), ("strict", df[df["prev_strict"] == 1])]:
        y, x = d["criticalFound"].to_numpy(), d["pc"].to_numpy()
        auc = roc_auc_score(y, x)
        boots = []
        for _ in range(config.N_BOOT):
            i = rng.integers(0, len(d), len(d))
            if y[i].min() != y[i].max():
                boots.append(roc_auc_score(y[i], x[i]))
        out[grp] = {"n": int(len(d)), "share_pastCritical": float(x.mean()),
                    "rate_if_pc1": float(y[x == 1].mean()), "rate_if_pc0": float(y[x == 0].mean()),
                    "auc": float(auc), "auc_lo": float(np.percentile(boots, 2.5)),
                    "auc_hi": float(np.percentile(boots, 97.5))}
    # Logistic regression with the current inspector and the other city features as controls.
    X = features.city_design_matrix(df, inspector=True).drop(columns=["Inspector_blue"])
    X["prev_strict"] = df["prev_strict"].to_numpy()
    X["pastCritical_x_prev_strict"] = X["pastCritical"] * X["prev_strict"]
    fit = sm.Logit(df["criticalFound"].to_numpy(), sm.add_constant(X)).fit(disp=0)
    for term in ("pastCritical", "prev_strict", "pastCritical_x_prev_strict"):
        out[f"coef_{term}"] = float(fit.params[term])
        out[f"se_{term}"] = float(fit.bse[term])
        out[f"p_{term}"] = float(fit.pvalues[term])
    return out


# ----------------------------------------------------------------------------------------------
# Experiment 5: geography
# ----------------------------------------------------------------------------------------------
def community_areas_for(points: pd.DataFrame) -> pd.Series:
    import geopandas as gpd

    from .load import load_community_areas

    ca = load_community_areas()
    ok = points["Latitude"].notna() & points["Longitude"].notna()
    gdf = gpd.GeoDataFrame(points[ok], geometry=gpd.points_from_xy(
        points.loc[ok, "Longitude"], points.loc[ok, "Latitude"]), crs=4326)
    j = gpd.sjoin(gdf, ca, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    return j["community_area"].reindex(points.index)


def cramers_v(a, b) -> float:
    tab = pd.crosstab(a, b).to_numpy().astype(float)
    n = tab.sum()
    expected = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / n
    chi2 = ((tab - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / n / (min(tab.shape) - 1)))


def draw_cluster_maps(fig, t, df: pd.DataFrame, ca_shapes, order: list[str]):
    axes = fig.subplots(2, 3, gridspec_kw={"hspace": 0.08, "wspace": 0.02})
    for ax, c in zip(axes.ravel(), order, strict=True):
        ax.set_facecolor(t["surface"])
        ca_shapes.boundary.plot(ax=ax, color=t["grid"], linewidth=0.4)
        ax.scatter(df["Longitude"], df["Latitude"], s=0.6, color=t["neutral"], alpha=0.35,
                   lw=0, rasterized=True)
        d = df[df["sanitarian"] == c]
        ax.scatter(d["Longitude"], d["Latitude"], s=1.2, color=t["series"][0], lw=0,
                   rasterized=True)
        ax.set_title(f"{c} ({len(d):,} inspections)", color=t["ink"], fontsize=10, loc="left")
        ax.set_axis_off()
        ax.set_aspect(1 / np.cos(np.deg2rad(41.85)))
    fig.suptitle("Where each inspector cluster works (training period, 2011-2014)", x=0.02,
                 ha="left", color=t["ink"], fontsize=12, fontweight="bold")
    fig.text(0.02, 0.935, "Blue: the cluster's inspections; gray: all others. Clusters ordered "
             "from most lenient to strictest (observed / expected rate).", color=t["ink2"],
             fontsize=9)


# ----------------------------------------------------------------------------------------------
def run() -> dict:
    out: dict = {}
    selected = json.loads((config.CACHE_DIR / "core2_selected.json").read_text())
    pub, ext = core2.build_tables()
    train = pub[~pub["Test"]]

    table, res = backtest_table(pub, ext, selected)
    out["backtest"] = res
    out["table_backtest"] = report.save_table(table, "core3_backtest")

    st = stringency(train)
    out["stringency"] = st.to_dict(orient="records")
    tab = st.assign(**{
        "Cluster": st["cluster"],
        "Training inspections": st["n"].map("{:,}".format),
        "Actual critical rate": st["actual"].map("{:.1%}".format),
        "Expected from non-inspector features": st["expected"].map("{:.1%}".format),
        "Observed / expected [95% CI]": [f"{r.oe:.2f} {report.ci(r.oe_lo, r.oe_hi)}"
                                         for r in st.itertuples()],
        "City coefficient": st["coef"].map("{:+.2f}".format),
    })[["Cluster", "Training inspections", "Actual critical rate",
        "Expected from non-inspector features", "Observed / expected [95% CI]",
        "City coefficient"]]
    out["table_stringency"] = report.save_table(tab, "core3_stringency")
    plots.render("core3_stringency", plots.draw_intervals, size=(7.5, 3.6),
                 df=st.assign(label=st["cluster"]), label_col="label", est="oe", lo="oe_lo",
                 hi="oe_hi", ref=1.0, ref_label="as the kitchen predicts",
                 xlabel="Observed / expected critical-violation rate",
                 heading="Same kind of kitchen, very different findings",
                 subtitle="Training inspections by inspector cluster; expected rate from the "
                          "city's non-inspector features (cross-fitted)")
    strict = set(st.loc[st["oe"] > 1, "cluster"])
    out["strict_clusters"] = sorted(strict)
    out["knock_on"] = knock_on(train, strict)

    # geography
    pts = pub[["Inspection_ID", "Latitude", "Longitude", "sanitarian", "Test"]].copy()
    pts["community_area"] = community_areas_for(pts)
    trp = pts[~pts["Test"] & pts["community_area"].notna()]
    out["geography"] = {
        "cramers_v_cluster_area": cramers_v(trp["sanitarian"], trp["community_area"]),
        "unplaced": int(pts["community_area"].isna().sum()),
    }
    top = (trp.groupby("sanitarian")["community_area"]
              .agg(lambda s: s.value_counts(normalize=True).head(10).sum()))
    out["geography"]["top10_area_share"] = top.to_dict()
    from .load import load_community_areas

    ca = load_community_areas()
    order = list(st["cluster"])
    plots.render("core3_cluster_maps", draw_cluster_maps, size=(10, 8.2),
                 df=pub[~pub["Test"]], ca_shapes=ca, order=order)
    pts.to_parquet(config.CACHE_DIR / "core3_points.parquet", index=False)
    report.save_results("core3", out)
    return out
