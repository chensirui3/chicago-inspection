"""Core 2: try to beat the published benchmark under the same rules.

Same rows, split, target and metric code as Core 1. Model and hyperparameter choices are made
only inside the training period with rolling-origin validation (train before a cutoff, validate
on the next two months, roll forward); the Sept-Oct 2014 test window is scored once per final
model. Wins are judged with a paired bootstrap against the city's model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config, core1, evaluate, features, models, plots, report

CUTOFFS = ["2013-05-01", "2013-07-01", "2013-09-01", "2013-11-01", "2014-01-01", "2014-03-01"]
HORIZON_DAYS = 61
LGBM_GRID = {"num_leaves": [7, 15, 31], "n_estimators": [150, 300, 600]}
RIDGE_GRID = {"C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]}
COUNT_PREFIXES = ("n_", "nearby_n")


@dataclass
class Spec:
    family: str  # city_glm | ridge | lgbm | blend
    feature_set: str  # city | extended
    inspector: bool
    params: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        names = {"city_glm": "City GLM (Core 1)", "ridge": "Ridge logistic",
                 "lgbm": "LightGBM", "blend": "Blend (ridge + LightGBM)"}
        fs = "" if self.family == "city_glm" else (
            ", city features" if self.feature_set == "city" else ", extended features")
        return names[self.family] + fs


# ----------------------------------------------------------------------------------------------
# Data and design matrices
# ----------------------------------------------------------------------------------------------
def build_tables(as_of=None):
    pub = features.build_city_table("published")
    ext = features.extended_features(pub, as_of=as_of, train_mask=~pub["Test"].to_numpy())
    return pub, ext


def design(pub: pd.DataFrame, ext: pd.DataFrame, feature_set: str, inspector: bool,
           temperature: bool = True) -> pd.DataFrame:
    X = features.city_design_matrix(pub, inspector=inspector, temperature=temperature)
    if feature_set == "extended":
        X = pd.concat([X, ext.loc[pub.index]], axis=1)
    return X


def glm_transform(X: pd.DataFrame) -> pd.DataFrame:
    """log1p for counts and years for day counts, so a linear model is not driven by outliers."""
    X = X.copy()
    for c in X.columns:
        if c.startswith(COUNT_PREFIXES):
            X[c] = np.log1p(X[c])
        elif c.startswith("days_since"):
            X[c] = X[c] / 365
    return X


def fit_predict(spec: Spec, X_tr: pd.DataFrame, y_tr, X_ev: pd.DataFrame,
                seed: int = config.SEED) -> tuple[np.ndarray, object]:
    y_tr = np.asarray(y_tr)
    if spec.family == "city_glm":
        pf = models.inspector_penalty(X_tr.columns)
        if pf.any():
            m = models.GlmnetLogit(penalty_factor=pf).fit(X_tr, y_tr)
            k = core1.PUBLISHED_LAMBDA_INDEX
        else:
            m = models.GlmnetLogit(penalty_factor=np.ones(X_tr.shape[1])).fit(
                X_tr, y_tr, lambdas=[0.0])
            k = 0
        return m.predict_proba(X_ev, k), m
    if spec.family == "ridge":
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=spec.params["C"], max_iter=5000))
        m.fit(glm_transform(X_tr), y_tr)
        return m.predict_proba(glm_transform(X_ev))[:, 1], m
    if spec.family == "lgbm":
        m = models.fit_lgbm(X_tr, y_tr, spec.params, seed=seed)
        return m.predict_proba(X_ev)[:, 1], m
    if spec.family == "blend":
        p1, m1 = fit_predict(Spec("ridge", spec.feature_set, spec.inspector,
                                  spec.params["ridge"]), X_tr, y_tr, X_ev, seed)
        p2, m2 = fit_predict(Spec("lgbm", spec.feature_set, spec.inspector,
                                  spec.params["lgbm"]), X_tr, y_tr, X_ev, seed)
        return (p1 + p2) / 2, (m1, m2)
    raise ValueError(spec.family)


# ----------------------------------------------------------------------------------------------
# Rolling-origin validation inside the training period
# ----------------------------------------------------------------------------------------------
def validate(spec: Spec, pub: pd.DataFrame, ext: pd.DataFrame, return_oof: bool = False):
    train = pub[~pub["Test"]]
    X = design(train, ext, spec.feature_set, spec.inspector)
    y = train["criticalFound"].to_numpy()
    rows, oof = [], []
    for cut, (tr, va) in zip(CUTOFFS, models.rolling_origin_splits(
            train["Inspection_Date"], CUTOFFS, HORIZON_DAYS), strict=True):
        p, _ = fit_predict(spec, X.iloc[tr], y[tr], X.iloc[va])
        v = train.iloc[va]
        rows.append({
            "cutoff": cut, "n": len(va), "positives": int(y[va].sum()),
            "days_earlier": evaluate.days_earlier(v["Inspection_Date"], p, y[va],
                                                  v["Inspection_ID"]),
            "auc": roc_auc_score(y[va], p),
            "log_loss": log_loss(y[va], np.clip(p, 1e-6, 1 - 1e-6)),
        })
        oof.append(pd.DataFrame({"Inspection_ID": v["Inspection_ID"].to_numpy(), "score": p,
                                 "y": y[va], "cutoff": cut}))
    res = pd.DataFrame(rows)
    return (res, pd.concat(oof, ignore_index=True)) if return_oof else res


def summarize_validation(res: pd.DataFrame) -> dict:
    return {"val_days": res["days_earlier"].mean(), "val_days_sd": res["days_earlier"].std(),
            "val_auc": res["auc"].mean(), "val_log_loss": res["log_loss"].mean()}


def tune(family: str, feature_set: str, inspector: bool, pub, ext, log: list) -> Spec:
    """Pick hyperparameters by mean validation log-loss (stable), not by the noisy metric."""
    grid = LGBM_GRID if family == "lgbm" else RIDGE_GRID
    best, best_ll = None, np.inf
    for values in product(*grid.values()):
        params = dict(zip(grid.keys(), values, strict=True))
        spec = Spec(family, feature_set, inspector, params)
        s = summarize_validation(validate(spec, pub, ext))
        log.append({"family": family, "feature_set": feature_set, "inspector": inspector,
                    "params": json.dumps(params), **s})
        if s["val_log_loss"] < best_ll:
            best, best_ll = spec, s["val_log_loss"]
    return best


FEATURE_GROUPS = {
    "Longer history": ("n_insp_2y", "n_canvass_all", "canvass_crit_share", "n_crit_2y",
                       "n_serious_2y", "n_fail_2y", "n_reinspection_2y", "license_insp_failed",
                       "days_since_any", "last_any_critical", "last_any_fail"),
    "Complaint inspections": ("n_complaint_1y", "n_complaint_crit_2y", "days_since_complaint"),
    "Past violation codes": ("prev_code_",),
    "Facility type and risk level": ("facility_", "risk_level"),
    "Nearby restaurants' critical rate": ("nearby_",),
}


def ablation(spec: Spec, pub, ext) -> list[dict]:
    """Validation-only: drop one group of new features at a time from the selected model."""
    base = summarize_validation(validate(spec, pub, ext))
    rows = [{"Feature group removed": "(none: selected model)",
             "Validation days earlier": base["val_days"], "Validation AUC": base["val_auc"],
             "Change in days": 0.0}]
    for group, prefixes in FEATURE_GROUPS.items():
        drop = [c for c in ext.columns if c.startswith(prefixes)]
        s = summarize_validation(validate(spec, pub, ext.drop(columns=drop)))
        rows.append({"Feature group removed": group, "Validation days earlier": s["val_days"],
                     "Validation AUC": s["val_auc"],
                     "Change in days": s["val_days"] - base["val_days"]})
    return rows


def first_half_end(test: pd.DataFrame) -> pd.Timestamp:
    """Last day whose cumulative inspection count stays below half (the city's period A)."""
    n = test.groupby("Inspection_Date").size().sort_index().cumsum()
    return n[n < n.iloc[-1] / 2].index[-1]


def hyperparameter_note(tlog: pd.DataFrame, best: Spec) -> str:
    lg = tlog[tlog["family"] == "lgbm"].copy()
    lg["leaves"] = lg["params"].map(lambda s: json.loads(s)["num_leaves"])
    groups = lg.groupby(["feature_set", "inspector"])
    deep_worse = all(g.loc[g["leaves"] == g["leaves"].max(), "val_log_loss"].min()
                     > g["val_log_loss"].min() for _, g in groups)
    p = best.params.get("lgbm", best.params)
    text = (f"**Hyperparameters**: {len(tlog)} configurations, chosen by validation log-loss. "
            f"The selected trees have {p.get('num_leaves')} leaves and "
            f"{p.get('n_estimators')} rounds")
    if deep_worse:
        text += f"; the deepest setting ({lg['leaves'].max()} leaves) was worse in every variant"
    return text + "."


def rolling_isotonic_check(oof: pd.DataFrame) -> dict:
    """Brier of raw vs isotonic scores, isotonic always fitted on earlier windows only."""
    from sklearn.metrics import brier_score_loss

    raw, iso = [], []
    cuts = sorted(oof["cutoff"].unique())
    for i, c in enumerate(cuts[1:], start=1):
        past = oof[oof["cutoff"].isin(cuts[:i])]
        cur = oof[oof["cutoff"] == c]
        f = models.fit_isotonic(past["score"], past["y"])
        raw.append(brier_score_loss(cur["y"], cur["score"]))
        iso.append(brier_score_loss(cur["y"], f.predict(cur["score"])))
    return {"brier_raw": float(np.mean(raw)), "brier_isotonic": float(np.mean(iso)),
            "windows": len(raw)}


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def run() -> dict:
    out: dict = {}
    pub, ext = build_tables()
    train, test = pub[~pub["Test"]], pub[pub["Test"]]
    city_scores = pd.read_parquet(config.CACHE_DIR / "core1_scores.parquet")
    city_test = city_scores.set_index("Inspection_ID").loc[test["Inspection_ID"], "city_score"]

    # ---- tuning and validation (training period only) -----------------------------------------
    tuning_log: list = []
    specs: dict[tuple, Spec] = {}
    for inspector in (True, False):
        specs[("city_glm", "city", inspector)] = Spec("city_glm", "city", inspector)
        specs[("ridge", "extended", inspector)] = tune("ridge", "extended", inspector, pub, ext,
                                                       tuning_log)
        for fs in ("city", "extended"):
            specs[("lgbm", fs, inspector)] = tune("lgbm", fs, inspector, pub, ext, tuning_log)
        specs[("blend", "extended", inspector)] = Spec(
            "blend", "extended", inspector,
            {"ridge": specs[("ridge", "extended", inspector)].params,
             "lgbm": specs[("lgbm", "extended", inspector)].params})
    pd.DataFrame(tuning_log).to_csv(config.TABLES_DIR / "core2_tuning_log.csv", index=False)

    val_rows, oofs = [], {}
    for key, spec in specs.items():
        res, oof = validate(spec, pub, ext, return_oof=True)
        oofs[key] = oof
        val_rows.append({"key": key, **summarize_validation(res)})
        res.assign(model=spec.label, inspector=spec.inspector).to_csv(
            config.CACHE_DIR / f"core2_val_{'_'.join(map(str, key))}.csv", index=False)
    val = pd.DataFrame(val_rows).set_index("key")

    # The pre-registered choice: best mean validation days-earlier among models with inspectors
    # (the published setup), then the same family without inspectors for Core 3.
    with_insp = [k for k in specs if k[2]]
    best_key = max(with_insp, key=lambda k: val.loc[[k], "val_days"].iloc[0])
    best_noinsp_key = (best_key[0], best_key[1], False)
    out["selected"] = {"with_inspector": asdict(specs[best_key]),
                       "without_inspector": asdict(specs[best_noinsp_key]),
                       "rule": "max mean validation days-earlier over 6 rolling 2-month windows"}

    # ---- ablation of the new feature groups (validation only, selected family) --------------
    out["ablation"] = ablation(specs[best_key], pub, ext)
    out["table_ablation"] = report.save_table(
        pd.DataFrame(out["ablation"]), "core2_ablation",
        {"Validation days earlier": ".2f", "Validation AUC": ".3f", "Change in days": "+.2f"})

    # ---- final fits on the full training period; one look at the test window ----------------
    from .load import load_rds

    xgb = load_rds("30_xgboost_data", columns=["Inspection_ID", "score"]).set_index(
        "Inspection_ID")["score"]
    rows, test_scores = [], {"city_published": city_test.to_numpy(),
                             "city_xgboost": xgb.loc[test["Inspection_ID"]].to_numpy()}
    models_fitted = {}
    for key, spec in specs.items():
        Xtr = design(train, ext, spec.feature_set, spec.inspector)
        Xte = design(test, ext, spec.feature_set, spec.inspector)
        p, m = fit_predict(spec, Xtr, train["criticalFound"], Xte)
        test_scores["_".join(map(str, key))] = p
        models_fitted[key] = (m, list(Xtr.columns))
    names = {"_".join(map(str, k)): k for k in specs}
    boot = evaluate.bootstrap_days_earlier(
        test["Inspection_Date"], test["criticalFound"], test["Inspection_ID"], test_scores,
        n_boot=config.N_BOOT, seed=config.SEED)
    boot_day = evaluate.bootstrap_days_earlier(
        test["Inspection_Date"], test["criticalFound"], test["Inspection_ID"], test_scores,
        n_boot=config.N_BOOT, seed=config.SEED, block="day")
    for name, p in test_scores.items():
        r = evaluate.evaluate_ranking(test, p)
        lo, hi = evaluate.interval(boot[name])
        cmp = evaluate.paired_comparison(boot, name, "city_published")
        cmp_day = evaluate.paired_comparison(boot_day, name, "city_published")
        key = names.get(name)
        spec = specs[key] if key else None
        vkey = key or (("city_glm", "city", True) if name == "city_published" else None)
        baseline_label = {"city_published": "City GLM (Core 1 port, published)",
                          "city_xgboost": "City XGBoost (saved 2017 scores)"}
        row = {
            "name": name,
            "Model": spec.label if spec else baseline_label[name],
            "Inspector": "yes" if (spec.inspector if spec else True) else "no",
            "Validation days earlier": val.loc[[vkey], "val_days"].iloc[0] if vkey else np.nan,
            "Validation AUC": val.loc[[vkey], "val_auc"].iloc[0] if vkey else np.nan,
            "Days earlier": r["days_earlier"], "lo": lo, "hi": hi,
            "Diff vs city": cmp["diff_mean"], "diff_lo": cmp["diff_lo"],
            "diff_hi": cmp["diff_hi"], "day_diff_lo": cmp_day["diff_lo"],
            "day_diff_hi": cmp_day["diff_hi"],
            "SD": r["sd"], "Found later": r["n_later"],
            "First-half share": r["first_half_model"], "AUC": r["auc"],
            "Brier": r.get("brier", np.nan),
            "selected": key == best_key,
        }
        rows.append(row)
    res = pd.DataFrame(rows)
    res = res[res["name"] != "city_glm_city_True"].reset_index(drop=True)  # same as city_published
    res.to_csv(config.CACHE_DIR / "core2_results_full.csv", index=False)

    # ---- README table -------------------------------------------------------------------
    def fmt_row(r):
        star = " **(selected)**" if r["selected"] else ""
        return {
            "Model": r["Model"] + star,
            "Inspector": r["Inspector"],
            "Validation days earlier": "" if np.isnan(r["Validation days earlier"])
            else f"{r['Validation days earlier']:.2f}",
            "Test days earlier [95% CI]": f"{r['Days earlier']:.2f} {report.ci(r['lo'], r['hi'])}",
            "Minus city [95% CI]": "-" if r["name"] == "city_published"
            else f"{r['Diff vs city']:+.2f} {report.ci(r['diff_lo'], r['diff_hi'], '+.2f')}",
            "SD": f"{r['SD']:.1f}",
            "Found later": int(r["Found later"]),
            "First-half share": f"{r['First-half share']:.1%}",
            "AUC": f"{r['AUC']:.3f}",
        }
    order = ["city_published", "city_xgboost", "lgbm_city_True", "ridge_extended_True",
             "lgbm_extended_True", "blend_extended_True", "city_glm_city_False",
             "lgbm_city_False", "ridge_extended_False", "lgbm_extended_False",
             "blend_extended_False"]
    table = pd.DataFrame([fmt_row(res.set_index("name").loc[n].to_dict() | {"name": n})
                          for n in order])
    out["table_results"] = report.save_table(table, "core2_results")
    out["results"] = res.drop(columns=["selected"]).to_dict(orient="records")
    best_name = "_".join(map(str, best_key))
    best = res.set_index("name").loc[best_name]
    out["best"] = {"name": best_name, "label": specs[best_key].label,
                   "days": best["Days earlier"], "lo": best["lo"], "hi": best["hi"],
                   "diff": best["Diff vs city"], "diff_lo": best["diff_lo"],
                   "diff_hi": best["diff_hi"], "day_diff_lo": best["day_diff_lo"],
                   "day_diff_hi": best["day_diff_hi"], "first_half": best["First-half share"],
                   "found_later": int(best["Found later"]), "sd": best["SD"],
                   "auc": best["AUC"], "win": bool(best["diff_lo"] > 0)}
    city_row = res.set_index("name").loc["city_published"]
    out["city"] = {"days": city_row["Days earlier"], "lo": city_row["lo"], "hi": city_row["hi"]}
    cmp_x = evaluate.paired_comparison(boot, best_name, "city_xgboost")
    out["best"].update({"diff_xgb": cmp_x["diff_mean"], "diff_xgb_lo": cmp_x["diff_lo"],
                        "diff_xgb_hi": cmp_x["diff_hi"]})
    xgb_row = res.set_index("name").loc["city_xgboost"]
    out["city_xgboost"] = {"days": xgb_row["Days earlier"], "lo": xgb_row["lo"],
                           "hi": xgb_row["hi"]}

    # ---- calibration: isotonic regression fitted on the validation months --------------------
    # Decide inside the training period: for each rolling window, fit isotonic on the earlier
    # windows' out-of-fold scores and compare Brier on that window with the raw scores.
    from sklearn.metrics import brier_score_loss

    y = test["criticalFound"].to_numpy()
    best_scores = test_scores[best_name]
    noinsp_name = "_".join(map(str, best_noinsp_key))
    cal_check = rolling_isotonic_check(oofs[best_key])
    use_iso = cal_check["brier_isotonic"] < cal_check["brier_raw"]
    if use_iso:
        best_cal = models.fit_isotonic(oofs[best_key]["score"],
                                       oofs[best_key]["y"]).predict(best_scores)
        noinsp_cal = models.fit_isotonic(
            oofs[best_noinsp_key]["score"], oofs[best_noinsp_key]["y"]
        ).predict(test_scores[noinsp_name])
    else:  # raw scores are already as well calibrated as isotonic can make them out of sample
        best_cal, noinsp_cal = best_scores, test_scores[noinsp_name]
    out["calibration"] = {
        **cal_check, "use_isotonic": bool(use_iso),
        "test_brier_selected": brier_score_loss(y, best_cal),
        "test_brier_city": brier_score_loss(y, city_test.to_numpy()),
        "test_mean_pred": float(np.mean(best_cal)), "test_observed_rate": float(y.mean()),
    }
    cal_label = f"{specs[best_key].label}" + (", isotonic" if use_iso else "")
    plots.render(
        "core2_calibration", plots.draw_calibration, size=(6.4, 5.2),
        curves={"City model": core1.calibration_table(y, city_test.to_numpy()),
                cal_label: core1.calibration_table(y, best_cal)},
        subtitle="Deciles of predicted risk on the test window; bars are 95% intervals")

    # ---- cumulative chart: actual vs city vs selected -------------------------------------------
    def curve(scores, simulated=True):
        b = evaluate.bin_summary(test["Inspection_Date"], scores, y, test["Inspection_ID"])
        return pd.DataFrame({"date": b["date"],
                             "found": b["POSTOT_SIM"] if simulated else b["POSTOT"]})
    curves = {"Actual schedule": curve(city_test.to_numpy(), simulated=False),
              f"City model ({out['city']['days']:.2f} days earlier)": curve(
                  city_test.to_numpy()),
              f"My model ({out['best']['days']:.2f} days earlier)": curve(best_scores)}
    plots.render("core2_cumulative", plots.draw_cumulative, curves=curves, total=int(y.sum()),
                 midpoint=first_half_end(test),
                 subtitle="Each model re-orders the same 1,637 inspections, same number per "
                          "day. My model: ridge + LightGBM blend with new features.")

    # ---- feature importance of the selected tree model, for the write-up ----------------------
    m, cols = models_fitted[best_key]
    lgb_model = m[1] if isinstance(m, tuple) else m
    if hasattr(lgb_model, "booster_"):
        gain = pd.Series(lgb_model.booster_.feature_importance("gain"), index=cols)
        imp = (gain / gain.sum()).sort_values(ascending=False).head(15).reset_index()
        imp.columns = ["Feature", "Share of gain"]
        out["table_importance"] = report.save_table(imp, "core2_importance",
                                                    {"Share of gain": ".1%"})

    # ---- what I tried (validation vs test for every candidate family) -----------------------
    tlog = pd.DataFrame(tuning_log)
    out["n_configs_tried"] = int(len(tlog))
    r = res.set_index("name")

    def diff(n):
        x = r.loc[n]
        return (f"{x['Days earlier']:.2f} days, {x['Diff vs city']:+.2f} "
                f"{report.ci(x['diff_lo'], x['diff_hi'], '+.2f')} vs the city model")

    abl = pd.DataFrame(out["ablation"]).iloc[1:]
    worst = abl.sort_values("Change in days").iloc[0]
    cal = out["calibration"]
    out["what_i_tried"] = [
        f"**Gradient-boosted trees on the city's own ten features** (LightGBM): "
        f"{diff('lgbm_city_True')}. Trees alone add little; the gain comes from new inputs.",
        f"**New features in a ridge logistic regression** (longer history, complaint "
        f"inspections, facility type and risk level, past violation codes, nearby restaurants' "
        f"critical rate): {diff('ridge_extended_True')}.",
        f"**The same features in LightGBM**: {diff('lgbm_extended_True')}. This is the best "
        f"test number, but it was not the validation winner, so it is not the headline model.",
        f"**A 50/50 blend of the two**, selected on validation: {diff('blend_extended_True')}.",
        f"**Which new features matter** (validation only): removing "
        f"{worst['Feature group removed'].lower()} costs the most "
        f"({worst['Change in days']:+.2f} days); see the ablation table.",
        hyperparameter_note(tlog, specs[best_key]),
        f"**Isotonic calibration** fitted on earlier validation windows "
        f"{'improved' if cal['use_isotonic'] else 'did not improve'} Brier score on later "
        f"windows ({cal['brier_raw']:.4f} raw vs {cal['brier_isotonic']:.4f}), so it was "
        f"{'kept' if cal['use_isotonic'] else 'dropped'}; the raw blend is already close "
        f"(mean prediction {cal['test_mean_pred']:.3f} vs observed rate "
        f"{cal['test_observed_rate']:.3f} on the test window).",
        "**Left out on purpose**: ten more license categories the city computed but never used "
        "(tavern, outdoor patio, ...). The city's code attaches them by license end date, which "
        "can postdate the inspection.",
        f"**Validation is optimistic**: windows right after each cutoff score "
        f"{val.loc[[best_key], 'val_days'].iloc[0]:.2f} days for the selected model, the test "
        f"window (4.5 months after training ends) {out['best']['days']:.2f}.",
    ]

    # ---- cache for Core 3 and the extensions ---------------------------------------------------
    sc = test[["Inspection_ID"]].copy()
    sc["best_score"] = best_scores
    sc["best_calibrated"] = best_cal
    sc["best_noinsp_score"] = test_scores[noinsp_name]
    sc["best_noinsp_calibrated"] = noinsp_cal
    sc["city_score"] = city_test.to_numpy()
    sc.to_parquet(config.CACHE_DIR / "core2_scores.parquet", index=False)
    (config.CACHE_DIR / "core2_selected.json").write_text(json.dumps(out["selected"], indent=2))
    report.save_results("core2", out)
    return out
