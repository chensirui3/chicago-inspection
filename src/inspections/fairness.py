"""Neighborhood fairness audit: does the priority list treat neighborhoods evenly?

Fairness definition used for the headline: **calibration within groups** (a group's mean
predicted risk should equal its actual critical-violation rate). A scheduler acts on predicted
risk, so a group whose risk is over-predicted gets inspected early more than its kitchens
warrant (a burden on owners) and an under-predicted group waits longer than it should (exposure
for residents). Four other gap metrics are reported alongside it, all with
bootstrap intervals; they can disagree, and choosing among them is a policy decision.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, core2, core3, evaluate, plots, report
from .load import load_community_areas, load_socioeconomic

GROUPS = ["Low hardship", "Middle", "High hardship"]
DENSITY = ["heat_burglary", "heat_sanitation", "heat_garbage"]


# ----------------------------------------------------------------------------------------------
# Groups
# ----------------------------------------------------------------------------------------------
def area_groups() -> pd.DataFrame:
    """Community areas with 2008-2012 socioeconomic indicators and hardship thirds."""
    se = load_socioeconomic()
    se["group"] = pd.qcut(se["hardship_index"], 3, labels=GROUPS).astype(str)
    return se


def attach_groups(pub: pd.DataFrame) -> pd.DataFrame:
    pts = pub[["Inspection_ID", "Latitude", "Longitude"]].copy()
    pts["community_area"] = core3.community_areas_for(pts)
    g = area_groups()[["community_area", "group", "hardship_index", "per_capita_income",
                       "pct_below_poverty"]]
    out = pts.merge(g, on="community_area", how="left")
    out.index = pub.index
    return out


# ----------------------------------------------------------------------------------------------
# Group metrics
# ----------------------------------------------------------------------------------------------
def group_metrics(test: pd.DataFrame, scores, midpoint: pd.Timestamp) -> pd.DataFrame:
    """Five gap metrics, per hardship group, for one model on the test window."""
    d = evaluate.as_days(test["Inspection_Date"])
    y = test["criticalFound"].to_numpy()
    s = np.asarray(scores, float)
    sim = evaluate.simulated_dates(d, s, test["Inspection_ID"].to_numpy())
    mid = evaluate.as_days([midpoint])[0]
    first = sim <= mid
    rows = []
    for g in GROUPS:
        m = (test["group"] == g).to_numpy()
        pos = m & (y == 1)
        rows.append({
            "group": g,
            "n": int(m.sum()),
            "positives": int(pos.sum()),
            "actual_rate": y[m].mean(),
            "mean_pred": s[m].mean(),
            "calibration_gap": s[m].mean() - y[m].mean(),
            "share_first_half": (m & first).sum() / first.sum(),
            "share_violations": pos.sum() / (y == 1).sum(),
            "days_earlier": (d[pos] - sim[pos]).mean(),
            "clean_first_half": (m & first & (y == 0)).sum() / max((m & (y == 0)).sum(), 1),
            "pushed_back": int((pos & (d <= mid) & ~first).sum()),
            "second_half_violations": int((pos & ~first).sum()),
        })
    out = pd.DataFrame(rows)
    out["attention_gap"] = out["share_first_half"] - out["share_violations"]
    return out


METRICS = ["calibration_gap", "attention_gap", "days_earlier", "clean_first_half",
           "second_half_violations"]


def bootstrap_groups(test: pd.DataFrame, scores: dict[str, np.ndarray], midpoint,
                     n_boot: int = config.N_BOOT, seed: int = config.SEED) -> pd.DataFrame:
    """Resample test inspections, re-run the simulation and the group metrics for each model."""
    rows = []
    for b, idx in enumerate(evaluate.bootstrap_indices(len(test), n_boot, seed)):
        sub = test.iloc[idx].reset_index(drop=True)
        for name, s in scores.items():
            gm = group_metrics(sub, np.asarray(s)[idx], midpoint)
            gm["model"], gm["rep"] = name, b
            rows.append(gm)
    return pd.concat(rows, ignore_index=True)


def summarize(point: dict[str, pd.DataFrame], boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, gm in point.items():
        bm = boot[boot["model"] == name]
        for _, r in gm.iterrows():
            b = bm[bm["group"] == r["group"]]
            row = {"model": name, "group": r["group"], "n": r["n"], "positives": r["positives"]}
            for m in METRICS + ["actual_rate", "mean_pred"]:
                lo, hi = evaluate.interval(b[m])
                row[m], row[f"{m}_lo"], row[f"{m}_hi"] = r[m], lo, hi
            rows.append(row)
        # largest between-group gap in calibration (percentage points), with its interval
        piv = bm.pivot_table(index="rep", columns="group", values="calibration_gap")
        spread = piv.max(axis=1) - piv.min(axis=1)
        lo, hi = evaluate.interval(spread)
        cg = gm.set_index("group")["calibration_gap"]
        rows.append({"model": name, "group": "max - min", "calibration_gap": cg.max() - cg.min(),
                     "calibration_gap_lo": lo, "calibration_gap_hi": hi})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Fixes
# ----------------------------------------------------------------------------------------------
def group_recalibration(oof: pd.DataFrame, groups: pd.Series):
    """Per-group logistic recalibration fitted on validation-month predictions."""
    from sklearn.linear_model import LogisticRegression

    fits = {}
    for g in GROUPS:
        d = oof[groups.reindex(oof["Inspection_ID"]).to_numpy() == g]
        z = np.log(np.clip(d["score"], 1e-6, 1 - 1e-6) / (1 - np.clip(d["score"], 1e-6, 1 - 1e-6)))
        fits[g] = LogisticRegression(C=1e6).fit(z.to_numpy()[:, None], d["y"])

    def apply(scores, group_labels):
        s = np.clip(np.asarray(scores, float), 1e-6, 1 - 1e-6)
        z = np.log(s / (1 - s))[:, None]
        out = s.copy()
        for g, f in fits.items():
            m = np.asarray(group_labels) == g
            if m.any():
                out[m] = f.predict_proba(z[m])[:, 1]
        return out

    return apply


def draw_tradeoff(fig, t, df: pd.DataFrame):
    ax = fig.add_subplot(111)
    plots.style_axes(ax, t, grid="both")
    for k, r in enumerate(df.itertuples()):
        c = t["series"][0] if "My model" in r.label else t["series"][1]
        ax.hlines(r.gap, r.days_lo, r.days_hi, color=c, lw=0.8, alpha=0.6)
        ax.vlines(r.days, r.gap_lo, r.gap_hi, color=c, lw=0.8, alpha=0.6)
        ax.plot(r.days, r.gap, "o", color=c, ms=8, mec=t["surface"], mew=1.5, zorder=3)
        ax.annotate(r.label, (r.days, r.gap), xytext=(7, 6 if k % 2 == 0 else -12),
                    textcoords="offset points", fontsize=8.5, color=t["ink2"])
    ax.set_xlabel("Days earlier, all critical violations (higher is better)")
    ax.set_ylabel("Largest calibration gap between hardship\ngroups, percentage points "
                  "(lower is fairer)")
    plots.title(ax, t, "Accuracy vs. the largest neighborhood gap",
                "Test window; whiskers are 95% bootstrap intervals")


def draw_area_maps(fig, t, areas, pred_col: str, act_col: str, n_col: str, vmax: float):
    axes = fig.subplots(1, 2, gridspec_kw={"wspace": 0.02})
    cmap = plots.seq_cmap(t)
    for ax, col, name in [(axes[0], pred_col, "Mean predicted risk (my model)"),
                          (axes[1], act_col, "Actual critical-violation rate")]:
        ax.set_facecolor(t["surface"])
        thin = areas[areas[n_col] < 10]
        areas.plot(column=col, ax=ax, cmap=cmap, vmin=0, vmax=vmax, edgecolor=t["surface"],
                   linewidth=0.6)
        if len(thin):
            thin.plot(ax=ax, color=t["neutral"], edgecolor=t["surface"], linewidth=0.6)
        ax.set_title(name, loc="left", color=t["ink"], fontsize=10)
        ax.set_axis_off()
    sm = matplotlib_scalar_mappable(cmap, vmax)
    cb = fig.colorbar(sm, ax=list(axes), shrink=0.6, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], labelsize=8)
    cb.ax.yaxis.set_major_formatter(plots.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    fig.suptitle("Predicted risk vs. actual critical-violation rate by community area, "
                 "Sept-Oct 2014", x=0.02, ha="left", color=t["ink"], fontsize=11.5,
                 fontweight="bold")
    fig.text(0.02, 0.9, "Gray: fewer than 10 test inspections. Same color scale in both maps.",
             color=t["ink2"], fontsize=9)


def matplotlib_scalar_mappable(cmap, vmax):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    return ScalarMappable(norm=Normalize(0, vmax), cmap=cmap)


# ----------------------------------------------------------------------------------------------
def run() -> dict:
    out: dict = {}
    selected = json.loads((config.CACHE_DIR / "core2_selected.json").read_text())
    pub, ext = core2.build_tables()
    grp = attach_groups(pub)
    pub = pub.assign(group=grp["group"].to_numpy(), community_area=grp["community_area"])
    train, test_all = pub[~pub["Test"]], pub[pub["Test"]]
    out["placement"] = {
        "test_inspections": int(len(test_all)),
        "no_coordinates": int(test_all["Latitude"].isna().sum()),
        "unplaced": int(test_all["community_area"].isna().sum()),
    }
    test = test_all[test_all["group"].isin(GROUPS)]
    out["placement"]["analyzed"] = int(len(test))
    midpoint = core2.first_half_end(test_all)

    city = core2.Spec("city_glm", "city", True)
    best = core2.Spec(**selected["with_inspector"])

    def scores_for(spec, drop=()):
        Xtr = core2.design(train, ext, spec.feature_set, spec.inspector).drop(columns=list(drop))
        Xte = core2.design(test, ext, spec.feature_set, spec.inspector).drop(columns=list(drop))
        return core2.fit_predict(spec, Xtr, train["criticalFound"], Xte)[0]

    scores = {
        "City model": scores_for(city),
        "City model, no density features": scores_for(city, DENSITY),
        "My model": scores_for(best),
        "My model, no density features": scores_for(best, DENSITY),
    }
    # fix 2: recalibrate my model within hardship groups, fitted on validation months only
    _, oof = core2.validate(best, pub, ext, return_oof=True)
    recal = group_recalibration(oof, pub.set_index("Inspection_ID")["group"])
    scores["My model, recalibrated within groups"] = recal(scores["My model"], test["group"])

    point = {name: group_metrics(test, s, midpoint) for name, s in scores.items()}
    boot = bootstrap_groups(test, scores, midpoint)
    summ = summarize(point, boot)
    summ.to_csv(config.TABLES_DIR / "ext_b_group_metrics.csv", index=False)
    out["group_metrics"] = {
        m: {r["group"]: {k: v for k, v in r.items() if k not in ("model", "group")}
            for r in summ[summ["model"] == m].to_dict(orient="records")}
        for m in summ["model"].unique()}

    # overall accuracy of each variant (whole test window, same rows)
    acc = {}
    dboot = evaluate.bootstrap_days_earlier(test["Inspection_Date"], test["criticalFound"],
                                            test["Inspection_ID"], scores,
                                            n_boot=config.N_BOOT, seed=config.SEED)
    for name, s in scores.items():
        r = evaluate.evaluate_ranking(test, s)
        lo, hi = evaluate.interval(dboot[name])
        acc[name] = {"days": r["days_earlier"], "lo": lo, "hi": hi, "auc": r["auc"]}
    out["accuracy"] = acc

    # ---- gap table (city vs mine) --------------------------------------------------------------
    def cell(model, group, m, fmt):
        r = summ[(summ["model"] == model) & (summ["group"] == group)].iloc[0]
        return f"{format(r[m], fmt)} {report.ci(r[m + '_lo'], r[m + '_hi'], fmt)}"

    rows = []
    for g in GROUPS:
        base = point["City model"].set_index("group").loc[g]
        rows.append({
            "Hardship group": g,
            "Test inspections (violations)": f"{int(base['n']):,} ({int(base['positives'])})",
            "Actual rate": f"{base['actual_rate']:.1%}",
            "Predicted minus actual, city": cell("City model", g, "calibration_gap", "+.3f"),
            "Predicted minus actual, mine": cell("My model", g, "calibration_gap", "+.3f"),
            "First-half share minus violation share, city":
                cell("City model", g, "attention_gap", "+.3f"),
            "First-half share minus violation share, mine":
                cell("My model", g, "attention_gap", "+.3f"),
            "Days earlier, city": cell("City model", g, "days_earlier", ".1f"),
            "Days earlier, mine": cell("My model", g, "days_earlier", ".1f"),
        })
    out["table_gaps"] = report.save_table(pd.DataFrame(rows), "ext_b_gaps")

    rows = []
    for g in GROUPS:
        rows.append({
            "Hardship group": g,
            "Clean restaurants in first half, city": cell("City model", g, "clean_first_half",
                                                          ".1%"),
            "Clean restaurants in first half, mine": cell("My model", g, "clean_first_half",
                                                          ".1%"),
            "Violations left for the second half, city":
                cell("City model", g, "second_half_violations", ".0f"),
            "Violations left for the second half, mine":
                cell("My model", g, "second_half_violations", ".0f"),
        })
    out["table_burden"] = report.save_table(pd.DataFrame(rows), "ext_b_burden")

    # ---- fixes: accuracy vs the largest calibration gap -------------------------------------
    trade = []
    for name in scores:
        s = summ[(summ["model"] == name) & (summ["group"] == "max - min")].iloc[0]
        trade.append({"label": name, "days": acc[name]["days"], "days_lo": acc[name]["lo"],
                      "days_hi": acc[name]["hi"], "gap": 100 * s["calibration_gap"],
                      "gap_lo": 100 * s["calibration_gap_lo"],
                      "gap_hi": 100 * s["calibration_gap_hi"]})
    trade = pd.DataFrame(trade)
    out["tradeoff"] = trade.to_dict(orient="records")
    tab = pd.DataFrame({
        "Model": trade["label"],
        "Days earlier [95% CI]": [f"{r.days:.2f} {report.ci(r.days_lo, r.days_hi)}"
                                  for r in trade.itertuples()],
        "AUC": [f"{acc[n]['auc']:.3f}" for n in trade["label"]],
        "Largest calibration gap, pp [95% CI]": [
            f"{r.gap:.1f} {report.ci(r.gap_lo, r.gap_hi, '.1f')}" for r in trade.itertuples()],
    })
    out["table_tradeoff"] = report.save_table(tab, "ext_b_tradeoff")
    plots.render("ext_b_tradeoff", draw_tradeoff, size=(7.6, 4.8), df=trade)

    # ---- cause check: inspector mix by group (Core 3 link) ------------------------------------
    st = pd.read_csv(config.TABLES_DIR / "core3_stringency.csv")
    strict = set(st.loc[st["Observed / expected [95% CI]"].str.split().str[0].astype(float) > 1,
                        "Cluster"])
    mix = (pub[pub["group"].isin(GROUPS)]
           .assign(strict=lambda d: d["sanitarian"].isin(strict))
           .groupby("group")["strict"].mean().reindex(GROUPS))
    out["strict_share_by_group"] = mix.to_dict()
    heat = pub[pub["group"].isin(GROUPS)].groupby("group")[DENSITY].mean().reindex(GROUPS)
    out["density_by_group"] = heat.to_dict(orient="index")

    # ---- map by community area ----------------------------------------------------------------
    by_area = test.assign(pred=scores["My model"]).groupby("community_area").agg(
        pred=("pred", "mean"), actual=("criticalFound", "mean"), n=("criticalFound", "size"))
    ca = load_community_areas().merge(by_area, left_on="community_area", right_index=True,
                                      how="left")
    ca["n"] = ca["n"].fillna(0)
    vmax = float(np.nanpercentile(ca.loc[ca["n"] >= 10, ["pred", "actual"]].to_numpy(), 97))
    plots.render("ext_b_area_maps", draw_area_maps, size=(9.5, 6.2), areas=ca,
                 pred_col="pred", act_col="actual", n_col="n", vmax=vmax)
    report.save_results("ext_b", out)
    return out


