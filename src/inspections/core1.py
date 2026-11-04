"""Core 1: reproduce the city's published model and evaluation in Python."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from . import config, evaluate, features, models, plots, report
from .load import load_rds

PUBLISHED_LAMBDA_INDEX = 99  # the report's model used model$lambda[100] (CODE/30_glmnet_model.R)


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Deciles of predicted risk: mean prediction, observed rate and a Wilson 95% interval."""
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    g = df.groupby("bin").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    z = 1.96
    centre = (g["obs"] + z * z / (2 * g["n"])) / (1 + z * z / g["n"])
    half = z * np.sqrt(g["obs"] * (1 - g["obs"]) / g["n"] + z * z / (4 * g["n"] ** 2)) / (
        1 + z * z / g["n"]
    )
    g["lo"], g["hi"] = centre - half, centre + half
    return g.reset_index()


def fit_city_model(train: pd.DataFrame, inspector: bool = True, temperature: bool = True):
    """The published configuration: glmnet path on the training rows, 100th lambda."""
    X = features.city_design_matrix(train, inspector=inspector, temperature=temperature)
    pf = models.inspector_penalty(X.columns)
    if pf.any():
        m = models.GlmnetLogit(penalty_factor=pf).fit(X, train["criticalFound"])
        return m, PUBLISHED_LAMBDA_INDEX, list(X.columns)
    # Without inspector columns nothing is penalized: plain logistic regression (lambda = 0).
    m = models.GlmnetLogit(penalty_factor=np.ones(X.shape[1])).fit(
        X, train["criticalFound"], lambdas=[0.0]
    )
    return m, 0, list(X.columns)


def score_city_model(m, lam_index: int, df: pd.DataFrame, inspector=True, temperature=True):
    X = features.city_design_matrix(df, inspector=inspector, temperature=temperature)
    return m.predict_proba(X, lam_index)


def run(n_cv_seeds: int = 50) -> dict:
    out: dict = {}
    pub = features.build_city_table("published")
    snap = features.build_city_table("snapshot")

    # ---- join counts ----------------------------------------------------------------------
    rows = []
    for step, n in snap.attrs["join_counts"].items():
        rows.append({"Step (CODE/30a order)": step, "Rows": n})
    rows.append({"Step (CODE/30a order)": "restricted to the published table "
                 "(DATA/30_xgboost_data.Rds)", "Rows": len(pub)})
    for name, df in [("2019 snapshot join", snap), ("published rows", pub)]:
        te = df[df["Test"]]
        rows.append({"Step (CODE/30a order)": f"{name}: train / test / positives",
                     "Rows": f"{(~df['Test']).sum():,} / {len(te):,} / "
                             f"{int(te['criticalFound'].sum())}"})
    report.save_table(pd.DataFrame(rows), "core1_join_counts")
    tr, te = pub[~pub["Test"]], pub[pub["Test"]]
    out["counts"] = {
        "train": int(len(tr)), "test": int(len(te)), "positives": int(te["criticalFound"].sum()),
        "train_first": tr["Inspection_Date"].min(), "train_last": tr["Inspection_Date"].max(),
        "test_first": te["Inspection_Date"].min(), "test_last": te["Inspection_Date"].max(),
        "snapshot_rows": len(snap), "snapshot_test": int(snap["Test"].sum()),
        "snapshot_positives": int(snap.loc[snap["Test"], "criticalFound"].sum()),
    }

    # ---- the published configuration ------------------------------------------------------
    m, k, cols = fit_city_model(tr)
    p_pub = score_city_model(m, k, te)
    res_pub = evaluate.evaluate_ranking(te, p_pub)
    out["published_config"] = {**res_pub, "lambda": float(m.lambdas_[k])}
    coef = m.coefficients(k)

    # ---- the current code (cv.glmnet lambda.min), several fold seeds -------------------------
    Xtr = features.city_design_matrix(tr)
    Xte = features.city_design_matrix(te)
    pf = models.inspector_penalty(Xtr.columns)
    from joblib import Parallel, delayed

    fits = Parallel(n_jobs=-2)(
        delayed(models.cv_glmnet)(Xtr, tr["criticalFound"], pf, nfolds=10, seed=config.SEED + s)
        for s in range(n_cv_seeds))
    cv_rows = []
    for s, cvfit in enumerate(fits):
        p = cvfit.predict_proba(Xte)
        cv_rows.append({"seed": s, "lambda_index": cvfit.index_min + 1,
                        "lambda_min": cvfit.lambda_min,
                        "days_earlier": evaluate.days_earlier(
                            te["Inspection_Date"], p, te["criticalFound"], te["Inspection_ID"])})
    cv0 = fits[0]
    p_cv = cv0.predict_proba(Xte)
    cvdf = pd.DataFrame(cv_rows)
    res_cv = evaluate.evaluate_ranking(te, p_cv)
    out["cv_config"] = {**res_cv, "lambda": cv0.lambda_min, "lambda_index": cv0.index_min + 1}
    out["cv_seeds"] = {
        "n": n_cv_seeds,
        "share_smallest_lambda": float((cvdf["lambda_index"] == 100).mean()),
        "lambda_index_min": int(cvdf["lambda_index"].min()),
        "days_min": float(cvdf["days_earlier"].min()),
        "days_max": float(cvdf["days_earlier"].max()),
        "days_mean": float(cvdf["days_earlier"].mean()),
    }
    cvdf.to_csv(config.TABLES_DIR / "core1_cv_seeds.csv", index=False)

    # ---- library routes at the same lambda ---------------------------------------------------
    lam = float(m.lambdas_[k])
    gl = models.fit_glum(Xtr, tr["criticalFound"], pf, lam)
    glum_coef = pd.Series(np.concatenate([[gl.intercept_], gl.coef_]), index=coef.index)
    sk_b0, sk_beta = models.fit_sklearn(Xtr, tr["criticalFound"], pf, lam)
    sk_coef = pd.Series(np.concatenate([[sk_b0], sk_beta]), index=coef.index)
    p_glum = gl.predict(Xte.to_numpy())
    p_sk = expit(Xte.to_numpy() @ sk_beta + sk_b0)
    out["routes"] = {
        "glum_max_abs_coef_diff": float((glum_coef - coef).abs().max()),
        "sklearn_max_abs_coef_diff": float((sk_coef - coef).abs().max()),
        "glum_days": evaluate.days_earlier(te["Inspection_Date"], p_glum, te["criticalFound"],
                                           te["Inspection_ID"]),
        "sklearn_days": evaluate.days_earlier(te["Inspection_Date"], p_sk, te["criticalFound"],
                                              te["Inspection_ID"]),
    }

    # ---- same model on today's full snapshot join (sensitivity) ------------------------------
    s_tr, s_te = snap[~snap["Test"]], snap[snap["Test"]]
    ms, ks, _ = fit_city_model(s_tr)
    res_snap = evaluate.evaluate_ranking(s_te, score_city_model(ms, ks, s_te))
    out["snapshot_rows"] = res_snap

    # ---- the city's own XGBoost scores, re-scored with this repo's metric ----------------
    xgb = load_rds("30_xgboost_data")
    xte = xgb[xgb["Test"]].sort_values("Inspection_ID")
    res_xgb = evaluate.evaluate_ranking(xte, "score")
    out["city_xgboost_saved_scores"] = res_xgb

    # ---- reproduction table ----------------------------------------------------------------
    P = config.PUBLISHED
    rep_rows = [
        ("Test inspections / with a critical violation", f"{P['n_test']:,} / {P['n_positive']}",
         f"{res_pub['n']:,} / {res_pub['n_positive']}",
         f"{res_snap['n']:,} / {res_snap['n_positive']}"),
        ("Training inspections", f"{P['n_train']:,}", f"{len(tr):,}", f"{len(s_tr):,}"),
        ("Avg. days earlier", f"{P['days_earlier']:.2f}", f"{res_pub['days_earlier']:.2f}",
         f"{res_snap['days_earlier']:.2f}"),
        ("SD of days earlier", f"{P['days_earlier_sd']:.2f}", f"{res_pub['sd']:.2f}",
         f"{res_snap['sd']:.2f}"),
        ("Median / max days earlier", "6 / 58", f"{res_pub['median']:.0f} / {res_pub['max']:.0f}",
         f"{res_snap['median']:.0f} / {res_snap['max']:.0f}"),
        ("Violations found later (share)", f"{P['n_later']} (38%)",
         f"{res_pub['n_later']} ({res_pub['share_later']:.0%})",
         f"{res_snap['n_later']} ({res_snap['share_later']:.0%})"),
        ("First-half share, actual -> model",
         f"{P['first_half_actual']:.1%} -> {P['first_half_model']:.1%}",
         f"{res_pub['first_half_actual']:.1%} -> {res_pub['first_half_model']:.1%}",
         f"{res_snap['first_half_actual']:.1%} -> {res_snap['first_half_model']:.1%}"),
        ("Test AUC (normalized Gini)", "not published",
         f"{res_pub['auc']:.3f} ({res_pub['gini']:.3f})",
         f"{res_snap['auc']:.3f} ({res_snap['gini']:.3f})"),
        ("Brier score", "not published", f"{res_pub['brier']:.4f}", f"{res_snap['brier']:.4f}"),
        ("City XGBoost (saved scores): days earlier / AUC",
         f"{P['xgboost_days_earlier']:.2f} / {P['xgboost_auc']:.3f}",
         f"{res_xgb['days_earlier']:.2f} / {res_xgb['auc']:.3f}", ""),
    ]
    rep = pd.DataFrame(rep_rows, columns=["Metric", "City (R, published)",
                                          "Python, published rows",
                                          "Python, 2019 snapshot join"])
    out["table_reproduction"] = report.save_table(rep, "core1_reproduction")

    names = {**{f"Inspector_{c}": f"Inspector {c}" for c in config.INSPECTOR_CLUSTERS}}
    coef_tab = pd.DataFrame({
        "Variable": [names.get(c, c) for c in coef.index],
        "City (R)": [config.PUBLISHED_COEFFICIENTS.get(c, np.nan) for c in coef.index],
        "Python port": coef.values,
        "glum": glum_coef.values,
        "scikit-learn": sk_coef.values,
    })
    coef_tab["Python - City"] = coef_tab["Python port"] - coef_tab["City (R)"]
    out["table_coefficients"] = report.save_table(
        coef_tab, "core1_coefficients",
        {"City (R)": ".3f", "Python port": ".3f", "glum": ".3f", "scikit-learn": ".3f",
         "Python - City": "+.3f"})
    out["coef_max_abs_diff"] = float(coef_tab["Python - City"].abs().max())

    # ---- rebuilt history and density features vs the city's files ---------------------------
    out["rebuild"] = rebuild_checks()
    chk = pd.DataFrame(
        [{"Feature": k, "Inspections compared": v["n"], "Exact match": f"{v['match']:.2%}",
          "Max abs. difference": f"{v['max_diff']:.2g}"} for k, v in out["rebuild"].items()]
    )
    out["table_rebuild"] = report.save_table(chk, "core1_rebuild_checks")

    # ---- calibration ----------------------------------------------------------------------
    cal = calibration_table(te["criticalFound"], p_pub)
    out["calibration"] = cal.to_dict(orient="list")
    plots.render("core1_calibration", plots.draw_calibration, size=(6.4, 5.2),
                 curves={"City model (Python port)": cal},
                 subtitle=f"{len(te):,} test inspections in deciles of predicted risk; "
                          "bars are 95% intervals")

    # ---- cache scores for the later parts -----------------------------------------------------
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    keep = ["Inspection_ID", "Inspection_Date", "License", "criticalFound", "Test",
            "Latitude", "Longitude", "sanitarian"]
    scores = pub[keep].copy()
    scores["city_score"] = score_city_model(m, k, pub)
    scores.to_parquet(config.CACHE_DIR / "core1_scores.parquet", index=False)
    report.save_results("core1", out)
    return out


def rebuild_checks() -> dict:
    """Compare features rebuilt from raw files with the city's precomputed files."""
    out = {}
    city = load_rds("23_food_insp_features")
    mine = features.rebuild_food_features()
    m = city.merge(mine, on="Inspection_ID", suffixes=("_city", "_mine"))
    for c in ["pastCritical", "pastSerious", "timeSinceLast", "firstRecord", "criticalCount"]:
        a, b = m[f"{c}_city"].astype(float), m[f"{c}_mine"].astype(float)
        out[c] = {"n": int(len(m)), "match": float(np.isclose(a, b, atol=1e-9).mean()),
                  "max_diff": float(np.abs(a - b).max())}
    pts = city[["Inspection_ID", "Inspection_Date", "Latitude", "Longitude"]]
    for kind, name in [("burglary", "22_burglary_heat"), ("sanitation",
                       "22_sanitationComplaints_heat"), ("garbage", "22_garbageCarts_heat")]:
        vals = features.heat_values(pts, features.heat_observations(kind))
        ref = load_rds(name).set_index("Inspection_ID")["heat_values"]
        ref = ref.reindex(pts["Inspection_ID"]).to_numpy()
        ok = ~np.isnan(ref)
        rel = np.abs(vals[ok] - ref[ok]) / np.abs(ref[ok])
        out[f"heat_{kind}"] = {"n": int(ok.sum()), "match": float((rel < 1e-9).mean()),
                               "max_diff": float(np.abs(vals[ok] - ref[ok]).max())}
    return out
