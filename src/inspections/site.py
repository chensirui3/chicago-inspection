"""Static GitHub Pages dashboard for Extension C (aggregate data only).

Writes site/index.html and site/data/{summary.json,community_areas.csv,backtest.csv,
monthly.csv,score_histogram.csv}. Nothing here identifies a restaurant: the page shows counts
and mean scores by community area and model diagnostics.
"""

from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd

from . import config

SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _color(v, lo, hi):
    if v is None or np.isnan(v):
        return "var(--na)"
    k = int(np.clip((v - lo) / max(hi - lo, 1e-9) * (len(SEQ) - 1), 0, len(SEQ) - 1))
    return SEQ[k]


def map_svg(areas, values: pd.DataFrame, width=360, height=470) -> str:
    """Community-area choropleth of mean predicted P/PF probability among due restaurants."""
    b = areas.total_bounds  # minx, miny, maxx, maxy
    kx = np.cos(np.deg2rad(41.85))
    sx = width / ((b[2] - b[0]) * kx)
    sy = height / (b[3] - b[1])
    s = min(sx, sy)
    vals = values.set_index("community_area")
    shown = vals.loc[vals["due"] >= 5, "mean_score"]
    lo, hi = (float(shown.min()), float(shown.max())) if len(shown) else (0.0, 1.0)
    parts = []
    for _, r in areas.iterrows():
        geoms = getattr(r.geometry, "geoms", [r.geometry])
        d = ""
        for g in geoms:
            xs, ys = g.exterior.coords.xy
            pts = " ".join(f"{(x - b[0]) * kx * s:.1f},{(b[3] - y) * s:.1f}"
                           for x, y in zip(xs, ys, strict=True))
            d += f"M{pts}Z"
        ca = int(r["community_area"])
        row = vals.loc[ca] if ca in vals.index else None
        due = int(row["due"]) if row is not None else 0
        ms = float(row["mean_score"]) if row is not None and due >= 5 else np.nan
        tip = f"{r['community_name']}: {due} due" + (
            f", mean predicted P/PF probability {ms:.0%}" if due >= 5 else " (too few to show)")
        parts.append(f'<path d="{d}" fill="{_color(ms, lo, hi)}" data-tip="{html.escape(tip)}">'
                     f"<title>{html.escape(tip)}</title></path>")
    legend = "".join(
        f'<rect x="{10 + i * 22}" y="{height + 8}" width="20" height="8" rx="2" fill="{c}"/>'
        for i, c in enumerate(SEQ))
    return (f'<svg viewBox="0 0 {width} {height + 40}" role="img" aria-label="Map of mean '
            f'predicted risk by community area" class="map">{"".join(parts)}{legend}'
            f'<text x="10" y="{height + 32}" class="axis">{lo:.0%}</text>'
            f'<text x="{10 + len(SEQ) * 22}" y="{height + 32}" class="axis" '
            f'text-anchor="end">{hi:.0%}</text></svg>')


def bars_svg(counts, edges, width=520, height=200) -> str:
    n = len(counts)
    top = max(counts) or 1
    bw = (width - 40) / n
    out = []
    for i, c in enumerate(counts):
        h = (height - 30) * c / top
        x = 30 + i * bw
        tip = f"{edges[i]:.0%}-{edges[i + 1]:.0%}: {c} restaurants"
        out.append(f'<rect x="{x + 1:.1f}" y="{height - 20 - h:.1f}" width="{bw - 2:.1f}" '
                   f'height="{h:.1f}" rx="3" class="bar" data-tip="{tip}"><title>{tip}</title>'
                   f"</rect>")
    ticks = "".join(f'<text x="{30 + i * bw:.1f}" y="{height - 4}" class="axis">'
                    f"{edges[i]:.0%}</text>" for i in range(0, n + 1, max(1, n // 5)))
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Histogram of '
            f'predicted probabilities" class="chart"><line x1="30" x2="{width - 10}" '
            f'y1="{height - 20}" y2="{height - 20}" class="baseline"/>{"".join(out)}{ticks}</svg>')


def line_svg(df: pd.DataFrame, x: str, y: str, fmt: str, width=520, height=200,
             shade=None) -> str:
    if df.empty:
        return "<p>No data yet.</p>"
    vals = df[y].to_numpy(float)
    lo, hi = min(0.0, np.nanmin(vals)), np.nanmax(vals) * 1.1
    n = len(df)

    def px(i):
        return 40 + i * (width - 50) / max(n - 1, 1)

    def py(v):
        return height - 22 - (v - lo) / max(hi - lo, 1e-9) * (height - 40)

    pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals) if not np.isnan(v))
    dots = "".join(
        f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="6" class="hit" '
        f'data-tip="{df[x].iloc[i]}: {format(v, fmt)}"><title>{df[x].iloc[i]}: '
        f"{format(v, fmt)}</title></circle>" for i, v in enumerate(vals) if not np.isnan(v))
    band = ""
    if shade is not None and shade.any():
        idx = np.flatnonzero(shade)
        band = (f'<rect x="{px(idx[0]):.1f}" y="10" width="{px(idx[-1]) - px(idx[0]):.1f}" '
                f'height="{height - 32}" class="band"/><text x="{px(idx[0]) + 4:.1f}" y="22" '
                f'class="axis">pandemic (excluded from training)</text>')
    zero = f'<line x1="40" x2="{width - 10}" y1="{py(0):.1f}" y2="{py(0):.1f}" class="baseline"/>'
    labels = (f'<text x="4" y="{py(hi / 1.1):.1f}" class="axis">{format(hi / 1.1, fmt)}</text>'
              f'<text x="4" y="{py(lo):.1f}" class="axis">{format(lo, fmt)}</text>'
              f'<text x="40" y="{height - 4}" class="axis">{df[x].iloc[0]}</text>'
              f'<text x="{width - 10}" y="{height - 4}" class="axis" text-anchor="end">'
              f"{df[x].iloc[-1]}</text>")
    return (f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">{band}{zero}'
            f'<polyline points="{pts}" class="line"/>{dots}{labels}</svg>')


CSS = """
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--s1:#2a78d6;--na:#e9e8e3;--band:rgba(137,135,129,.15)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;
--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--s1:#3987e5;--na:#2c2c2a;--band:rgba(137,135,129,.2)}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:16px;margin:0 0 8px}
.sub{color:var(--ink2);margin:0 0 20px}.updated{font-weight:600}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;
margin-bottom:20px}.tile,.card{background:var(--surface);border:1px solid var(--grid);
border-radius:10px;padding:14px 16px}.tile .v{font-size:26px;font-weight:600}
.tile .l{color:var(--ink2);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:16px}
svg{width:100%;height:auto;display:block}.map path{stroke:var(--surface);stroke-width:.8}
.map path:hover,.bar:hover{opacity:.8}.bar{fill:var(--s1)}
.line{fill:none;stroke:var(--s1);stroke-width:2;stroke-linejoin:round}
.hit{fill:var(--s1);stroke:var(--surface);stroke-width:2;opacity:0}.hit:hover{opacity:1}
.baseline{stroke:var(--axis);stroke-width:1}.band{fill:var(--band)}
.axis{fill:var(--muted);font-size:11px}table{border-collapse:collapse;width:100%;
font-size:13px;font-variant-numeric:tabular-nums}td,th{padding:4px 8px;text-align:right;
border-bottom:1px solid var(--grid)}td:first-child,th:first-child{text-align:left}
details{margin-top:10px}summary{cursor:pointer;color:var(--ink2)}
.note{color:var(--ink2);font-size:13px}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--surface);
padding:4px 8px;border-radius:6px;font-size:12px;display:none;max-width:260px}
"""

JS = """
const tip=document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el=>{
 el.addEventListener('mousemove',e=>{tip.textContent=el.dataset.tip;tip.style.display='block';
  tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';});
 el.addEventListener('mouseleave',()=>{tip.style.display='none';});});
"""


def publish(due: pd.DataFrame, backtest: pd.DataFrame, monthly: pd.DataFrame,
            mon: dict | None, summary: dict, out_dir=None) -> None:
    from .load import load_community_areas

    out_dir = out_dir or config.SITE_DIR
    data = out_dir / "data"
    data.mkdir(parents=True, exist_ok=True)
    areas = load_community_areas()
    agg = (due.groupby("community_area")
              .agg(due=("score", "size"), mean_score=("score", "mean"),
                   expected_ppf=("score", "sum")).reset_index())
    agg = areas[["community_area", "community_name"]].merge(agg, on="community_area",
                                                            how="left")
    agg["due"] = agg["due"].fillna(0).astype(int)
    agg.loc[agg["due"] < 5, ["mean_score", "expected_ppf"]] = np.nan  # suppress tiny groups
    agg.to_csv(data / "community_areas.csv", index=False)
    edges = np.linspace(0, max(0.8, float(due["score"].max()) + 0.05), 17)
    counts, _ = np.histogram(due["score"], bins=edges)
    pd.DataFrame({"from": edges[:-1], "to": edges[1:], "restaurants": counts}).to_csv(
        data / "score_histogram.csv", index=False)
    backtest.to_csv(data / "backtest.csv", index=False)
    monthly.to_csv(data / "monthly.csv", index=False)
    (data / "summary.json").write_text(json.dumps({**summary, "monitor": mon}, indent=2,
                                                  default=str))

    bt = backtest[~backtest.get("partial", False)] if len(backtest) else backtest
    rows = "".join(
        f"<tr><td>{html.escape(str(r.community_name))}</td><td>{r.due}</td>"
        f"<td>{'' if np.isnan(r.mean_score) else f'{r.mean_score:.1%}'}</td>"
        f"<td>{'' if np.isnan(r.expected_ppf) else f'{r.expected_ppf:.1f}'}</td></tr>"
        for r in agg.sort_values("community_name").itertuples())
    checks = summary["checks"]
    recent = bt[~bt["pandemic"]].tail(12) if len(bt) else bt
    health = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{html.escape(value)}</td></tr>"
        for label, value in [
            ("Newest inspection on the portal", str(checks.get("newest_inspection"))[:10]),
            ("Canvass inspections matched to a license",
             f"{checks.get('license_match_rate', float('nan')):.1%}"),
            ("Inspections / license records in cache",
             f"{checks.get('inspections', 0):,} / {checks.get('licenses', 0):,}"),
            ("Burglaries / sanitation / garbage-cart requests",
             f"{checks.get('burglary_rows', 0):,} / {checks.get('sanitation_rows', 0):,} / "
             f"{checks.get('garbage_rows', 0):,}"),
            ("Weather (NOAA, O'Hare)", f"{checks.get('weather', 'ok')}, newest "
                                       f"{str(checks.get('weather_newest'))[:10]}"),
            ("Training rows / job runtime",
             f"{summary.get('training_rows', 0):,} / {summary.get('runtime_s', 0)} s"),
        ])
    mon_html = "<p class='note'>Collecting predictions; the first check runs after a week of " \
               "nightly scores.</p>"
    if mon and "brier" in mon:
        mon_html = (f"<p>{mon['matched']:,} canvass inspections since scoring: mean predicted "
                    f"{mon['mean_pred']:.1%}, observed {mon['observed']:.1%}, Brier "
                    f"{mon['brier']:.3f}.</p>")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chicago Inspection Outlook</title><style>{CSS}</style></head><body><main>
<h1>Chicago food inspections: nightly outlook</h1>
<p class="sub"><span class="updated">Last updated {summary['updated']}</span> &middot; scored as
of {summary['scored_on']} &middot; newest inspection on the portal
{str(checks['newest_inspection'])[:10]}</p>
<div class="tiles">
<div class="tile"><div class="v">{summary['n_due']:,}</div><div class="l">retail food
establishments due for a canvass inspection</div></div>
<div class="tile"><div class="v">{summary['expected_ppf']:,.0f}</div><div class="l">expected to
show a Priority or Priority Foundation violation</div></div>
<div class="tile"><div class="v">{summary['mean_score']:.0%}</div><div class="l">mean predicted
probability</div></div>
<div class="tile"><div class="v">{checks['license_match_rate']:.0%}</div><div class="l">canvass
inspections matched to a business license</div></div></div>
<div class="grid">
<div class="card"><h2>Mean predicted risk by community area</h2>{map_svg(areas, agg)}
<p class="note">Among establishments due for inspection; gray = fewer than 5 due.</p>
<details><summary>Table view</summary><table><tr><th>Community area</th><th>Due</th>
<th>Mean risk</th><th>Expected P/PF</th></tr>{rows}</table></details></div>
<div class="card"><h2>Distribution of predicted probabilities</h2>{bars_svg(counts, edges)}
<h2 style="margin-top:18px">Monthly backtest: days earlier (new label)</h2>
{line_svg(bt, 'month', 'days_earlier', '.1f', shade=bt['pandemic'].to_numpy() if len(bt)
          else None)}
<p class="note">Each month is scored by a model trained only on earlier months.
Last 12 months (excluding the pandemic): mean {recent['days_earlier'].mean():.1f} days earlier,
AUC {recent['auc'].mean():.3f}.</p></div>
<div class="card"><h2>Canvass inspections per month</h2>{line_svg(monthly, 'month', 'n', ',.0f')}
<h2 style="margin-top:18px">Share finding a P/PF violation</h2>
{line_svg(monthly, 'month', 'ppf_rate', '.0%')}</div>
<div class="card"><h2>Calibration monitor</h2>{mon_html}
<h2>Data checks (latest run)</h2><table>{health}</table>
<h2 style="margin-top:18px">About this page</h2><p class="note">A score is a prediction, not an
inspection result. This page publishes only neighborhood aggregates and diagnostics, never a
ranked list of named
restaurants; actual past results are on the
<a href="https://data.cityofchicago.org/d/4ijn-s7e5">city's data portal</a>. Label: an
inspection result of Fail or Pass w/ Conditions (at least one Priority or Priority Foundation
violation under the 2018 food code); not comparable with the 2014 model's critical-only target.
No inspector information is used. Rebuilt nightly by GitHub Actions; the date at the top shows
the latest run.
</p></div></div><div id="tip" role="tooltip"></div></main><script>{JS}</script></body></html>"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")
