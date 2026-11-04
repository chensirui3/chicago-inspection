"""Extension A: turn risk scores into daily inspector routes (team orienteering, OR-Tools).

Each day of the Sept-Oct 2014 window the planner sees the restaurants still waiting, their
predicted risk and locations. It chooses which restaurants each inspector visits and in what
order to maximize the total predicted risk visited (the expected number of critical violations
found) subject to each inspector's shift and to the number of canvass inspections actually done
that day. Then it removes the visited restaurants and plans the next day (rolling horizon).

Scores: my best model without inspector features, computed as of 2014-09-02 (Core 3), because a
scheduler planning the window could not know who would inspect or the actual dates.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from . import config, evaluate, plots, report

OFFICE = (41.8723, -87.6816)  # CDPH Food Protection, 2133 W Lexington St (approximate)
KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON = 111.32 * np.cos(np.deg2rad(41.85))


@dataclass(frozen=True)
class Assumptions:
    inspectors: int = 10  # inspectors on canvass duty per day
    shift_min: int = 420  # field time per inspector (minutes)
    service_min: int = 60  # minutes per inspection
    speed_kmh: float = 20.0  # average door-to-door speed on Manhattan distance
    start: str = "office"  # "office" (one central office) or "zones" (one start per inspector)
    time_limit_s: int = 10  # guided local search per day
    pool: int = 4  # candidate pool = pool x that day's inspection count


# ----------------------------------------------------------------------------------------------
# One day's routing problem
# ----------------------------------------------------------------------------------------------
def travel_matrix(lat, lon, speed_kmh: float) -> np.ndarray:
    """Manhattan travel time in deci-minutes (Chicago's street grid)."""
    dy = np.abs(lat[:, None] - lat[None, :]) * KM_PER_DEG_LAT
    dx = np.abs(lon[:, None] - lon[None, :]) * KM_PER_DEG_LON
    return np.rint((dx + dy) / speed_kmh * 600).astype(np.int64)


def solve_day(lat, lon, prize, starts_latlon, a: Assumptions, max_visits: int,
              mandatory: bool = False) -> dict:
    """Route ``len(starts_latlon)`` inspectors over the candidate restaurants.

    ``prize`` is each restaurant's penalty for being skipped (risk scaled to an integer). With
    ``mandatory`` every restaurant must be visited and shift overruns are allowed but penalized
    (used to measure the travel and overtime of a set chosen without routing).
    """
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    k = len(starts_latlon)
    dep_lat = np.array([s[0] for s in starts_latlon])
    dep_lon = np.array([s[1] for s in starts_latlon])
    all_lat = np.concatenate([dep_lat, lat])
    all_lon = np.concatenate([dep_lon, lon])
    T = travel_matrix(all_lat, all_lon, a.speed_kmh)
    service = np.concatenate([np.zeros(k, np.int64), np.full(len(lat), a.service_min * 10)])
    manager = pywrapcp.RoutingIndexManager(len(all_lat), k, list(range(k)), list(range(k)))
    routing = pywrapcp.RoutingModel(manager)

    def travel_cb(i, j):
        return int(T[manager.IndexToNode(i), manager.IndexToNode(j)])

    def time_cb(i, j):
        a_, b_ = manager.IndexToNode(i), manager.IndexToNode(j)
        return int(T[a_, b_] + service[b_])

    travel_idx = routing.RegisterTransitCallback(travel_cb)
    time_idx = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(travel_idx)
    shift = a.shift_min * 10
    routing.AddDimension(time_idx, 0, shift * 3 if mandatory else shift, True, "time")
    time_dim = routing.GetDimensionOrDie("time")
    if mandatory:
        for v in range(k):
            time_dim.SetCumulVarSoftUpperBound(routing.End(v), shift, 100)
    else:
        # Visit-count dimension plus one constraint on its total over all inspectors, so each
        # day's visits never exceed the canvass inspections actually done that day.
        count_idx = routing.RegisterUnaryTransitCallback(
            lambda i: 0 if manager.IndexToNode(i) < k else 1)
        routing.AddDimension(count_idx, 0, max_visits, True, "count")
        count_dim = routing.GetDimensionOrDie("count")
        solver = routing.solver()
        solver.Add(solver.Sum([count_dim.CumulVar(routing.End(v)) for v in range(k)])
                   <= max_visits)
        for n in range(len(lat)):
            routing.AddDisjunction([manager.NodeToIndex(k + n)], int(prize[n]))

    params = pywrapcp.DefaultRoutingSearchParameters()
    # PATH_CHEAPEST_ARC (the guide's suggestion) cannot build a first solution under the global
    # visit cap and returns an empty plan, so optional-visit problems start from insertion.
    fs = routing_enums_pb2.FirstSolutionStrategy
    params.first_solution_strategy = (fs.PATH_CHEAPEST_ARC if mandatory
                                      else fs.PARALLEL_CHEAPEST_INSERTION)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.FromSeconds(a.time_limit_s)
    sol = routing.SolveWithParameters(params)
    if sol is None:
        raise RuntimeError("routing found no solution")
    routes, visited, travel, overtime = [], [], 0, 0
    for v in range(k):
        idx = routing.Start(v)
        path = []
        while not routing.IsEnd(idx):
            nxt = sol.Value(routing.NextVar(idx))
            travel += travel_cb(idx, nxt)
            node = manager.IndexToNode(nxt)
            if node >= k:
                path.append(node - k)
            idx = nxt
        end_time = sol.Value(time_dim.CumulVar(routing.End(v)))
        overtime += max(0, end_time - shift)
        routes.append(path)
        visited += path
    return {"visited": visited, "routes": routes, "travel_min": travel / 10,
            "overtime_min": overtime / 10, "used": sum(1 for r in routes if r)}


def zone_starts(lat, lon, k: int, seed: int = config.SEED) -> list[tuple[float, float]]:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(np.column_stack([lat, lon]))
    return [tuple(c) for c in km.cluster_centers_]


# ----------------------------------------------------------------------------------------------
# Rolling horizon over the window
# ----------------------------------------------------------------------------------------------
def simulate(strategy: str, df: pd.DataFrame, a: Assumptions) -> dict:
    """Plan the whole window with one strategy; returns visit dates and daily route stats.

    ``df``: one row per test inspection with Inspection_ID, Inspection_Date, Latitude,
    Longitude, risk (predicted probability) and criticalFound.
    """
    days = sorted(df["Inspection_Date"].unique())
    quota = df.groupby("Inspection_Date").size()
    starts = ([OFFICE] * a.inspectors if a.start == "office"
              else zone_starts(df["Latitude"].to_numpy(), df["Longitude"].to_numpy(),
                               a.inspectors))
    remaining = df.set_index("Inspection_ID")
    # the city's ranking order: risk descending, ties by Inspection_ID ascending
    rank = remaining.index.to_numpy()[evaluate.ranking_order(remaining["risk"].to_numpy(),
                                                             remaining.index.to_numpy())]
    order_pos = pd.Series(np.arange(len(rank)), index=rank)
    visit_date, daily, routes_by_day = {}, [], {}
    for d in days:
        n_today = int(min(quota.loc[d], len(remaining)))
        if n_today == 0:
            continue
        if strategy == "actual":
            ids = df.loc[df["Inspection_Date"] == d, "Inspection_ID"].to_numpy()
            cand = remaining.loc[ids]
            res = solve_day(cand["Latitude"].to_numpy(), cand["Longitude"].to_numpy(), None,
                            starts, replace(a, time_limit_s=min(a.time_limit_s, 3)), n_today,
                            mandatory=True)
        elif strategy == "ranking":
            ids = order_pos.loc[remaining.index].sort_values().index[:n_today]
            cand = remaining.loc[ids]
            res = solve_day(cand["Latitude"].to_numpy(), cand["Longitude"].to_numpy(), None,
                            starts, replace(a, time_limit_s=min(a.time_limit_s, 3)), n_today,
                            mandatory=True)
        elif strategy in ("routing", "geography"):
            m = min(len(remaining), max(a.pool * n_today, 120))
            if strategy == "routing":
                ids = order_pos.loc[remaining.index].sort_values().index[:m]
                cand = remaining.loc[ids]
                prize = np.rint(cand["risk"].to_numpy() * 1e5).astype(np.int64)
            else:
                lat0 = np.array([s[0] for s in starts])
                lon0 = np.array([s[1] for s in starts])
                dist = (np.abs(remaining["Latitude"].to_numpy()[:, None] - lat0) * KM_PER_DEG_LAT
                        + np.abs(remaining["Longitude"].to_numpy()[:, None] - lon0)
                        * KM_PER_DEG_LON).min(axis=1)
                cand = remaining.iloc[np.argsort(dist, kind="stable")[:m]]
                prize = np.full(len(cand), 100_000, dtype=np.int64)
            res = solve_day(cand["Latitude"].to_numpy(), cand["Longitude"].to_numpy(), prize,
                            starts, a, n_today)
        else:
            raise ValueError(strategy)
        chosen = cand.index.to_numpy()[res["visited"]]
        for i in chosen:
            visit_date[i] = d
        routes_by_day[str(pd.Timestamp(d).date())] = [
            [int(x) for x in cand.index.to_numpy()[r]] for r in res["routes"]]
        remaining = remaining.drop(index=chosen)
        daily.append({"date": d, "visits": len(chosen), "quota": n_today,
                      "travel_min": res["travel_min"], "overtime_min": res["overtime_min"],
                      "inspectors_used": res["used"]})
    return {"strategy": strategy, "assumptions": asdict(a), "visit_date": visit_date,
            "daily": daily, "routes": routes_by_day}


def score_plan(df: pd.DataFrame, plan: dict) -> dict:
    """Days earlier for the plan's visit dates; restaurants not reached count as found the day
    after the window closes (and are reported)."""
    after = df["Inspection_Date"].max() + pd.Timedelta(days=1)
    vd = df["Inspection_ID"].map(plan["visit_date"]).fillna(after)
    pos = df["criticalFound"].to_numpy() == 1
    diff = (df["Inspection_Date"] - vd).dt.days.to_numpy()[pos]
    daily = pd.DataFrame(plan["daily"])
    insp_days = daily["inspectors_used"].sum()
    mid = df.groupby("Inspection_Date").size().sort_index().cumsum()
    midpoint = mid[mid < mid.iloc[-1] / 2].index[-1]
    return {
        "days_earlier": float(diff.mean()),
        "not_reached": int((vd == after).sum()),
        "violations_not_reached": int(((vd == after).to_numpy() & pos).sum()),
        "first_half_share": float((vd[pos] <= midpoint).mean()),
        "travel_hours": float(daily["travel_min"].sum() / 60),
        "overtime_hours": float(daily["overtime_min"].sum() / 60),
        "inspections_per_inspector_day": float(daily["visits"].sum() / max(insp_days, 1)),
        "visits": int(daily["visits"].sum()),
    }


def _run_job(job):
    strategy, df_records, a = job
    df = pd.DataFrame.from_records(df_records)
    df["Inspection_Date"] = pd.to_datetime(df["Inspection_Date"])
    plan = simulate(strategy, df, a)
    return strategy, asdict(a), plan, score_plan(df, plan)


def run_jobs(jobs, workers: int):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_run_job, jobs))


# ----------------------------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------------------------
def draw_routes(fig, t, df, plans: dict[str, dict], day: str, ca_shapes):
    axes = fig.subplots(1, len(plans), gridspec_kw={"wspace": 0.03})
    risk = df.set_index("Inspection_ID")["risk"]
    lat = df.set_index("Inspection_ID")["Latitude"]
    lon = df.set_index("Inspection_ID")["Longitude"]
    for ax, (name, plan) in zip(axes, plans.items(), strict=True):
        ax.set_facecolor(t["surface"])
        ca_shapes.boundary.plot(ax=ax, color=t["grid"], linewidth=0.4)
        for r in plan["routes"].get(day, []):
            if not r:
                continue
            ys = [OFFICE[0], *lat.loc[r].to_numpy(), OFFICE[0]]
            xs = [OFFICE[1], *lon.loc[r].to_numpy(), OFFICE[1]]
            ax.plot(xs, ys, color=t["series"][0], lw=0.9, alpha=0.75)
        ids = [i for r in plan["routes"].get(day, []) for i in r]
        ax.scatter(lon.loc[ids], lat.loc[ids], s=10 + 120 * risk.loc[ids], color=t["series"][0],
                   edgecolor=t["surface"], linewidth=0.8, zorder=3)
        ax.plot(OFFICE[1], OFFICE[0], "s", color=t["ink"], ms=6, zorder=4)
        total_h = sum(d["travel_min"] for d in plan["daily"]
                      if str(pd.Timestamp(d["date"]).date()) == day) / 60
        ax.set_title(f"{name}: {len(ids)} visits, {total_h:.1f} h driving", loc="left",
                     color=t["ink"], fontsize=10)
        ax.set_axis_off()
        ax.set_aspect(1 / np.cos(np.deg2rad(41.85)))
    fig.suptitle(f"One day's routes, {day}", x=0.02, ha="left", color=t["ink"], fontsize=11.5,
                 fontweight="bold")
    fig.text(0.02, 0.92, "Square: central office. Dot size: predicted risk. Actual routes are "
             "unknown; the left map routes the day's actual visits optimally.",
             color=t["ink2"], fontsize=8.5)


# ----------------------------------------------------------------------------------------------
def run(workers: int | None = None) -> dict:
    out: dict = {}
    scores = pd.read_parquet(config.CACHE_DIR / "core3_scores.parquet")
    base = pd.read_parquet(config.CACHE_DIR / "core1_scores.parquet")
    df = base[base["Test"]].merge(scores[["Inspection_ID", "best_asof"]], on="Inspection_ID")
    df = df.rename(columns={"best_asof": "risk"})
    missing = int(df[["Latitude", "Longitude"]].isna().any(axis=1).sum())
    df = df.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
    out["dropped_no_coordinates"] = missing
    records = df[["Inspection_ID", "Inspection_Date", "Latitude", "Longitude", "risk",
                  "criticalFound"]].to_dict("records")
    workers = workers or max(1, min(10, (os.cpu_count() or 2) - 2))

    base_a = Assumptions()
    jobs = [(s, records, base_a) for s in ("actual", "ranking", "routing", "geography")]
    sens_a = []
    for field, lo, hi in [("inspectors", 5, 15), ("shift_min", 210, 630),
                          ("service_min", 30, 90), ("speed_kmh", 10.0, 30.0)]:
        for v in (lo, hi):
            sens_a.append((field, v, replace(base_a, **{field: v}, time_limit_s=3)))
    sens_a.append(("start", "zones", replace(base_a, start="zones", time_limit_s=3)))
    sens_a.append(("(baseline, 3 s limit)", "", replace(base_a, time_limit_s=3)))
    for _, _, a in sens_a:
        jobs += [("routing", records, a), ("ranking", records, a)]
    results = run_jobs(jobs, workers)

    names = {"actual": "Actual", "ranking": "Ranking only (city's method)",
             "routing": "Ranking plus routing (OR-Tools)", "geography": "Geography only"}
    main = {r[0]: (r[2], r[3]) for r in results[:4]}
    rows = []
    for s in ("actual", "ranking", "routing", "geography"):
        m = main[s][1]
        rows.append({"Strategy": names[s], "Avg. days earlier": f"{m['days_earlier']:.2f}",
                     "Critical violations found in first half": f"{m['first_half_share']:.1%}",
                     "Travel hours (window total)": f"{m['travel_hours']:,.0f}",
                     "Overtime hours": f"{m['overtime_hours']:,.0f}",
                     "Inspections per inspector-day": f"{m['inspections_per_inspector_day']:.1f}",
                     "Not reached by Oct 31": m["not_reached"]})
        out[s] = m
    out["table_strategies"] = report.save_table(pd.DataFrame(rows), "ext_a_strategies")

    # sensitivity: ranking+routing and ranking-only under each assumption change
    sens = {}
    for r in results[4:]:
        sens.setdefault(json.dumps(r[1], sort_keys=True), {})[r[0]] = r[3]
    rows = []
    for field, v, a in sens_a:
        res = sens[json.dumps(asdict(a), sort_keys=True)]
        label = field if field.startswith("(") else f"{field} = {v}"
        rows.append({"Assumption": label,
                     "Routing: days earlier": f"{res['routing']['days_earlier']:.2f}",
                     "Routing: travel h": f"{res['routing']['travel_hours']:,.0f}",
                     "Routing: not reached": res["routing"]["not_reached"],
                     "Ranking only: days earlier": f"{res['ranking']['days_earlier']:.2f}",
                     "Ranking only: travel h": f"{res['ranking']['travel_hours']:,.0f}",
                     "Ranking only: overtime h": f"{res['ranking']['overtime_hours']:,.0f}"})
    out["table_sensitivity"] = report.save_table(pd.DataFrame(rows), "ext_a_sensitivity")
    out["sensitivity"] = {k: v for k, v in sens.items()}
    out["assumptions"] = asdict(base_a)

    # cumulative chart for the four strategies
    pos = df["criticalFound"].to_numpy() == 1
    days = sorted(df["Inspection_Date"].unique())
    curves = {}
    for s in ("actual", "ranking", "routing", "geography"):
        vd = df["Inspection_ID"].map(main[s][0]["visit_date"])
        found = [int(((vd <= d) & pos).sum()) for d in days]
        label = "Actual schedule" if s == "actual" else (
            f"{names[s]} ({main[s][1]['days_earlier']:.2f} days)")
        curves[label] = pd.DataFrame({"date": days, "found": found})
    from .core2 import first_half_end

    plots.render("ext_a_cumulative", plots.draw_cumulative, curves=curves,
                 total=int(pos.sum()), midpoint=first_half_end(df),
                 heading="Critical violations found by date under four scheduling strategies",
                 subtitle=f"{base_a.inspectors} inspectors, {base_a.shift_min // 60} h field "
                          f"shifts, {base_a.service_min} min per inspection, "
                          f"{base_a.speed_kmh:.0f} km/h")
    from .load import load_community_areas

    day = "2014-09-03"
    plots.render("ext_a_routes", draw_routes, size=(10, 6.4), df=df,
                 plans={"Actual visits": main["actual"][0],
                        "Ranking plus routing": main["routing"][0]},
                 day=day, ca_shapes=load_community_areas())
    report.save_results("ext_a", out)
    return out
