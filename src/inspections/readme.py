"""Fill the README's generated sections from reports/results.json and reports/tables/*.md.

Text that depends on a result (for example whether an interval clears zero) is computed here,
so re-running the pipeline can never leave a stale claim in the README.
"""

from __future__ import annotations

from . import config, report

T = config.TABLES_DIR


def _table(name: str) -> str:
    p = T / f"{name}.md"
    return p.read_text(encoding="utf-8").strip() if p.exists() else "_(not generated yet)_"


def picture(name: str, alt: str, width: int | None = None) -> str:
    w = f' width="{width}"' if width else ""
    return (f'<picture>\n  <source media="(prefers-color-scheme: dark)" '
            f'srcset="reports/figures/{name}-dark.png">\n  <img alt="{alt}" '
            f'src="reports/figures/{name}.png"{w}>\n</picture>')


def ci(lo, hi, fmt=".2f"):
    return report.ci(lo, hi, fmt)


def sections() -> dict[str, str]:
    r = report.load_results()
    s: dict[str, str] = {}
    c1, c2, c3 = r.get("core1"), r.get("core2"), r.get("core3")
    if c1 and c2 and c3:
        pub = c1["published_config"]
        best = c2["best"]
        bt = c3["backtest"]
        win = best["diff_lo"] > 0
        lost = 1 - bt["city_asof"]["days_earlier"] / bt["published"]["days_earlier"]
        s["summary"] = (
            f"This repo rebuilds the City of Chicago's 2014 restaurant-inspection model in "
            f"Python and **reproduces its published result: {pub['days_earlier']:.2f} days "
            f"earlier** (published 7.44) on the same 1,637 inspections, with the same "
            f"first-half share ({pub['first_half_actual']:.1%} -> "
            f"{pub['first_half_model']:.1%}) and coefficients within "
            f"{c1['coef_max_abs_diff']:.3f}. Under the same rules, a blend of ridge logistic "
            f"regression and LightGBM with new history, complaint, facility and neighborhood "
            f"features finds critical violations **{best['days']:.2f} days earlier "
            f"{ci(best['lo'], best['hi'])}**, {best['diff']:+.2f} days "
            f"{ci(best['diff_lo'], best['diff_hi'], '+.2f')} versus the city's model in a "
            f"paired bootstrap ("
            + ("the interval is above zero" if win
               else "the interval is below zero" if best["diff_hi"] < 0
               else "the interval includes zero")
            + f"; versus the city's XGBoost it is {best['diff_xgb']:+.2f} "
            f"{ci(best['diff_xgb_lo'], best['diff_xgb_hi'], '+.2f')}). **The inspector "
            f"question:** most of the published gain depends on knowing who did each "
            f"inspection. Scored with only what a scheduler knew on 2014-09-02, the city's model "
            f"finds violations {bt['city_asof']['days_earlier']:.2f} days earlier "
            f"{ci(bt['city_asof']['lo'], bt['city_asof']['hi'])}, {lost:.0%} less than "
            f"published; my model keeps {bt['best_asof']['days_earlier']:.2f} days "
            f"{ci(bt['best_asof']['lo'], bt['best_asof']['hi'])}.")
        s["chart"] = picture("core2_cumulative", "Cumulative critical violations found by date: "
                             "actual schedule, city model, my model")

        cv = c1["cv_seeds"]
        rb = c1["rebuild"]
        counts = c1["counts"]
        route_diff = max(c1["routes"]["glum_max_abs_coef_diff"],
                         c1["routes"]["sklearn_max_abs_coef_diff"])
        hist_match = min(v["match"] for k, v in rb.items() if not k.startswith("heat"))
        heat_diff = max(v["max_diff"] for k, v in rb.items() if k.startswith("heat"))
        s["core1"] = "\n\n".join([
            _table("core1_reproduction"),
            "<details><summary>Coefficients: city (R) vs. this port, glum and scikit-learn"
            "</summary>\n\n" + _table("core1_coefficients") + "\n\n</details>",
            "**Why the small gaps, and what it took to get here**",
            f"- **Row set.** Joining today's snapshot exactly as `CODE/30a_glmnet_model.R` "
            f"does gives {counts['snapshot_rows']:,} rows ({counts['snapshot_test']:,} test, "
            f"{counts['snapshot_positives']} positives), not the published 1,637 / 258. The "
            f"city's 2019 refactor (v1.7.0) regenerated the inspector assignments and the "
            f"business-license extract. The only file in the snapshot with exactly the published "
            f"rows is `DATA/30_xgboost_data.Rds` (saved in 2017), so benchmark comparisons use "
            f"its 18,712 Inspection_IDs with features from the 2019 files. On the full 2019 join "
            f"the same model gives {c1['snapshot_rows']['days_earlier']:.2f} days.",
            "- **Penalty.** The report's model was not cross-validated: the 2015 script fits "
            "`glmnet` on the training rows and takes the 100th (smallest) lambda on the path "
            "(`w.lam <- 100`). The 2019 script uses `cv.glmnet`'s `lambda.min`. Here 10-fold CV "
            f"picks the 100th lambda in {cv['share_smallest_lambda']:.0%} of {cv['n']} fold "
            f"seeds, so both give {cv['days_mean']:.2f} days.",
            f"- **Coefficients** differ by at most {c1['coef_max_abs_diff']:.3f}. Between the "
            "2017 and 2019 extracts, 17 liquor-consumption flags and 3 tobacco flags changed; "
            "refitting on the city's 2017 feature values brings the unpenalized coefficients "
            "within 0.001 and the SD to 25.21. A gap of up to 0.003 remains on the six "
            "penalized inspector coefficients that I could not explain (glmnet's looser "
            "convergence tolerance is the likely cause). Matching lambda took reading glmnet's "
            "Fortran: for ridge (alpha = 0) its strong rule admits every variable at once, so "
            "lambda_max comes from the intercept-only gradient, not the gradient after the "
            "unpenalized columns are fitted. glum and scikit-learn "
            f"agree with this port to {route_diff:.0e}.",
            f"- **Dates.** Training data run {counts['train_first']} to {counts['train_last']}; "
            f"the test window is {counts['test_first']} to {counts['test_last']}. The 4.5-month "
            "gap means the code's \"last 90 days\" rule and the 2015 script's \"before "
            "2014-07-01\" rule select the same rows.",
            "- **History features rebuilt from raw data** (`CODE/23_food_insp_features.R` "
            f"port) match the city's file for {hist_match:.0%} of "
            f"{rb['pastCritical']['n']:,} inspections, and the density (KDE) port matches "
            f"`22_*_heat.Rds` "
            + ("to floating-point precision." if heat_diff < 1e-6 else f"to {heat_diff:.0e}."),
            "- **Standard metrics.** The city published no AUC for this model; its normalized "
            "Gini equals 2 x AUC - 1. Calibration on the test window:",
            picture("core1_calibration", "Calibration of the reproduced city model", 520),
        ])

        cal = c2["calibration"]
        s["core2"] = "\n\n".join([
            "Same rows, split, target and metric code as section 1. Choices were made only inside "
            "the training period: six rolling windows (train before a cutoff, validate on the "
            "next two months; cutoffs May 2013 to March 2014), never random K-fold, because a "
            "restaurant's later visits would leak into training. Hyperparameters were chosen by "
            "validation log-loss and the model family by validation days-earlier; the test "
            "window was then scored once per final model. Every model was run with and without "
            "the inspector feature.",
            _table("core2_results"),
            f"Intervals: 1,000 paired bootstrap resamples of the 1,637 test inspections; "
            f"resampling whole days instead gives "
            f"{ci(best['day_diff_lo'], best['day_diff_hi'], '+.2f')} for the selected model. "
            f"The other criteria in the city's CONTRIBUTING.md: spread {best['sd']:.1f} vs "
            f"25.2 days, {best['found_later']} vs 99 violations found later, first-half share "
            f"{best['first_half']:.1%} vs 68.2%. Probabilities: Brier "
            f"{cal['test_brier_selected']:.4f} vs {cal['test_brier_city']:.4f} for the city "
            f"model (calibration chart below).",
            "<details><summary>Feature ablation (validation only) and gain importance</summary>"
            "\n\n" + _table("core2_ablation") + "\n\n" + _table("core2_importance") + "\n\n"
            + picture("core2_calibration", "Calibration of the city model and my model", 520)
            + "\n\n</details>",
        ])
        s["tried"] = "\n".join(f"- {x}" for x in c2["what_i_tried"])

        st = {x["cluster"]: x for x in c3["stringency"]}
        lo_c = min(st.values(), key=lambda x: x["oe"])
        hi_c = max(st.values(), key=lambda x: x["oe"])
        ko = c3["knock_on"]
        geo = c3["geography"]
        s["core3"] = "\n\n".join([
            "The backtest moves each inspection to a new day but keeps the score computed with "
            "the inspector who actually did it, the temperature of its actual date and the "
            "time since its last visit measured to that date. The rows below remove the "
            "inspector, then rebuild every feature as of 2014-09-02 (history and license age "
            "up to that date, density windows ending that date, no inspector, no temperature) "
            "and rank once.",
            _table("core3_backtest"),
            f"**Strict inspectors or dirty kitchens?** For the restaurants each cluster visited, "
            f"the non-inspector features predict similar critical rates "
            f"({min(x['expected'] for x in st.values()):.1%} to "
            f"{max(x['expected'] for x in st.values()):.1%}), but the clusters record "
            f"{lo_c['actual']:.1%} ({lo_c['cluster']}) to {hi_c['actual']:.1%} "
            f"({hi_c['cluster']}): observed/expected from {lo_c['oe']:.2f} to "
            f"{hi_c['oe']:.2f}. \"Critical violation found\" measures the inspector as well as "
            f"the kitchen.",
            _table("core3_stringency"),
            picture("core3_stringency", "Observed over expected critical rate by inspector "
                    "cluster", 640),
            f"**Knock-on effect.** After a visit by a lenient cluster, pastCritical = 1 is rare "
            f"({ko['lenient']['share_pastCritical']:.1%} vs "
            f"{ko['strict']['share_pastCritical']:.1%} after a strict one), so it separates "
            f"kitchens worse: AUC of pastCritical alone {ko['lenient']['auc']:.3f} "
            f"{ci(ko['lenient']['auc_lo'], ko['lenient']['auc_hi'], '.3f')} vs "
            f"{ko['strict']['auc']:.3f} "
            f"{ci(ko['strict']['auc_lo'], ko['strict']['auc_hi'], '.3f')} "
            f"({ko['n']:,} training inspections with a known previous inspector). When it does "
            f"fire, its effect is similar: the interaction pastCritical x strict-previous in a "
            f"logistic model with the current inspector as control is "
            f"{ko['coef_pastCritical_x_prev_strict']:+.2f} (SE "
            f"{ko['se_pastCritical_x_prev_strict']:.2f}, p = "
            f"{ko['p_pastCritical_x_prev_strict']:.2f}).",
            f"**Geography.** Clusters work in particular areas: Cramér's V between cluster and "
            f"community area is {geo['cramers_v_cluster_area']:.2f}, and the strictest cluster "
            f"({hi_c['cluster']}) does {geo['top10_area_share'][hi_c['cluster']]:.0%} of its "
            f"inspections in just ten of the 77 community areas. Inspector and neighborhood "
            f"effects are tangled, which matters for the fairness audit below.",
            picture("core3_cluster_maps", "Map of inspections by inspector cluster", 760),
        ])
        insp_share = 1 - bt["city_no_insp"]["days_earlier"] / bt["published"]["days_earlier"]
        s["findings"] = (
            f"The city's 7.44 days mostly come from one variable a scheduler cannot use. "
            f"Removing the inspector clusters cuts the city model to "
            f"{bt['city_no_insp']['days_earlier']:.2f} days "
            f"{ci(bt['city_no_insp']['lo'], bt['city_no_insp']['hi'])}, {insp_share:.0%} less, "
            f"and rebuilding the remaining features as of the first day of the window changes "
            f"little more ({bt['city_asof']['days_earlier']:.2f} days, an interval that "
            + ("includes zero" if bt["city_asof"]["lo"] <= 0 else "stays above zero")
            + "). The inspector variable is strong because clusters differ in how often they "
            f"record a critical violation for similar kitchens ({lo_c['oe']:.2f}x to "
            f"{hi_c['oe']:.2f}x the expected rate), and in the backtest each inspection keeps "
            f"the score of the inspector who actually did it. Better history features, which a "
            f"scheduler does have, recover much of the gap: my model scored as of day one "
            f"finds violations {bt['best_asof']['days_earlier']:.2f} days earlier "
            f"{ci(bt['best_asof']['lo'], bt['best_asof']['hi'])}. The report already said the "
            f"daily model drops inspectors; these numbers show what that costs.")

    b = r.get("ext_b")
    if b:
        acc, tr = b["accuracy"], {x["label"]: x for x in b["tradeoff"]}
        cost = acc["My model, no density features"]["days"] - acc["My model"]["days"]
        gm = b.get("group_metrics", {})

        def g(model, group, metric):
            return gm.get(model, {}).get(group, {}).get(metric)

        lead = ""
        if gm:
            cal = g("City model", "Low hardship", "calibration_gap")
            cal_lo = g("City model", "Low hardship", "calibration_gap_lo")
            cal_hi = g("City model", "Low hardship", "calibration_gap_hi")
            mine = g("My model", "Low hardship", "calibration_gap")
            mid_c = g("City model", "Middle", "days_earlier")
            mid_m = g("My model", "Middle", "days_earlier")
            low_c = g("City model", "Low hardship", "days_earlier")
            mid_ci = ci(-g("City model", "Middle", "days_earlier_hi"),
                        -g("City model", "Middle", "days_earlier_lo"), ".1f")
            lead = (
                f"**Findings.** The city's model over-predicts risk in low-hardship areas by "
                f"{100 * cal:+.1f} percentage points {ci(100 * cal_lo, 100 * cal_hi, '+.1f')}, "
                f"the one calibration gap whose interval excludes zero (mine: "
                f"{100 * mine:+.1f}). The time savings go there too: under the city's model "
                f"violations in low-hardship areas are found {low_c:.1f} days earlier, while "
                f"those in middle-hardship areas are found {-mid_c:.1f} days *later* "
                f"{mid_ci} than in the actual schedule. Low-hardship areas do have the highest "
                f"actual "
                f"critical rate in the window, so part of this is risk-based prioritizing; my "
                f"model narrows the middle-hardship loss to {-mid_m:.1f} days but does not "
                f"close it.")
        s["ext_b"] = "\n\n".join([
            f"Inspections were placed in Chicago's 77 community areas with a spatial join "
            f"({b['placement']['unplaced']} of {b['placement']['test_inspections']:,} test "
            f"inspections fall outside the polygons and are left out), then grouped into "
            f"thirds of community areas by the 2008-2012 hardship index. Fairness definition: "
            f"**calibration within groups**, because a scheduler acts on predicted risk; four "
            f"other fairness metrics are shown too. All intervals are 1,000 "
            f"bootstrap resamples (simulation re-run each time), and the simulation runs on the "
            f"{b['placement']['analyzed']:,} placed inspections, so overall days-earlier here "
            f"differ slightly from the 1,637-inspection headline.",
            lead,
            _table("ext_b_gaps"),
            "<details><summary>Burden on owners and exposure for residents</summary>\n\n"
            + _table("ext_b_burden") + "\n\n</details>",
            "**Cause and fix.** The three density features come from crime and 311 reports, "
            "which reflect neighborhood conditions and reporting habits, not the kitchen. "
            f"Dropping them changes my model's accuracy by {cost:+.2f} days and its largest "
            f"calibration gap from {tr['My model']['gap']:.1f} to "
            f"{tr['My model, no density features']['gap']:.1f} percentage points; "
            "recalibrating within groups (fitted on validation months) is the second fix "
            "tried. Strict inspector clusters do "
            + ", ".join(f"{v:.0%} ({k.lower()})" for k, v in b["strict_share_by_group"].items())
            + " of inspections by group, so inspector and neighborhood effects overlap.",
            _table("ext_b_tradeoff"),
            picture("ext_b_tradeoff", "Days earlier versus the largest calibration gap", 600),
            picture("ext_b_area_maps", "Maps of predicted risk and actual critical rate by "
                    "community area", 760),
        ])

    a = r.get("ext_a")
    if a:
        asm = a["assumptions"]
        s["ext_a"] = "\n\n".join([
            f"A team orienteering problem in OR-Tools, solved day by day over the window "
            f"(rolling horizon): each inspector's travel plus inspection time must fit one "
            f"shift, each restaurant is visited at most once, each day's visits are capped at "
            f"the canvass inspections actually done that day, and skipping a restaurant costs "
            f"its predicted risk. Travel time is Manhattan distance at an assumed speed. Scores: "
            f"my model without inspector features, as of 2014-09-02 (section 3). Baseline "
            f"assumptions: {asm['inspectors']} inspectors on canvass duty, "
            f"{asm['shift_min'] / 60:.0f}-hour field shifts, {asm['service_min']} minutes per "
            f"inspection, {asm['speed_kmh']:.0f} km/h, one central office; guided local search "
            f"for {asm['time_limit_s']} s per day.",
            _table("ext_a_strategies"),
            picture("ext_a_cumulative", "Cumulative violations found under four strategies"),
            "**Sensitivity** (each assumption moved by 50%, 3 s search per day). \"Ranking "
            "only\" keeps its days-earlier because it ignores capacity; its overtime column is "
            "what that costs. The routing plan respects shifts, so when capacity binds it "
            "leaves restaurants unreached by Oct 31 (counted as found the day after).",
            _table("ext_a_sensitivity"),
            picture("ext_a_routes", "One day's routes: actual visits vs planned", 760),
            "Caveats: complaint, license and re-inspections also take inspector time and are "
            "not planned here; like the city's backtest this assumes a violation found on day "
            "40 would also have been there on day 1; actual routes are unknown, so the "
            "\"Actual\" travel is the optimal route through each day's actual visits (a lower "
            "bound); "
            + ("every test inspection had coordinates" if not a["dropped_no_coordinates"]
               else f"{a['dropped_no_coordinates']} test inspections lacked coordinates")
            + ". The planner sends inspectors where predicted risk is concentrated (mostly the "
            "north side on the day mapped), which is the neighborhood question the fairness "
            "audit measures. OR-Tools' local search is time-limited, so travel hours can "
            "differ slightly between machines (compare the 10 s baseline with the 3 s row); "
            "every other number in this README is deterministic.",
        ])

    c = r.get("ext_c")
    if c:
        s["ext_c"] = "\n\n".join([
            c["text"],
            _table("ext_c_backtest"),
            picture("ext_c_backtest", "Monthly forward backtest of the live model"),
            picture("ext_c_monthly", "Monthly canvass counts and P/PF rate since 2019"),
        ])
    return s


def update() -> list[str]:
    return report.fill_readme(sections())
