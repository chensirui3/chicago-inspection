"""Charts for the README, rendered in a light and a dark variant.

Palette, ink and mark specs follow one validated reference palette (categorical slots in a
fixed order; hairline solid grid; 2px lines; direct labels plus a legend). Every figure is saved
as ``<name>.png`` (light) and ``<name>-dark.png`` so the README can switch with ``<picture>``.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .report import figure_path  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                   "#6250d6", "#e34948"],
        "neutral": "#c3c2b7",
        "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
        "div_lo": "#2a78d6",
        "div_mid": "#f0efec",
        "div_hi": "#e34948",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300",
                   "#9085e9", "#e66767"],
        "neutral": "#5b5a56",
        "seq": ["#104281", "#184f95", "#1c5cab", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"],
        "div_lo": "#3987e5",
        "div_mid": "#383835",
        "div_hi": "#e66767",
    },
}

LW = 1.6  # ~2px lines at the saved resolution


def style_axes(ax, t, grid: str = "y") -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], labelsize=9, length=0, pad=6)
    if grid:
        ax.grid(axis=grid, color=t["grid"], linewidth=0.6, linestyle="-")
        ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])


def title(ax, t, text: str, sub: str | None = None) -> None:
    ax.set_title(text, loc="left", color=t["ink"], fontsize=11.5, fontweight="bold",
                 pad=22 if sub else 10)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=t["ink2"], fontsize=9, va="bottom")


def legend(ax, t, **kw) -> None:
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=t["ink2"], **kw)
    for h in getattr(leg, "legend_handles", []):
        h.set_alpha(1)


def render(name: str, draw: Callable, size=(8, 4.5), **kwargs) -> list[str]:
    """Draw ``draw(fig, theme, **kwargs)`` in both themes; return the two file paths."""
    paths = []
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    for mode, t in THEMES.items():
        fig = plt.figure(figsize=size, facecolor=t["surface"])
        draw(fig, t, **kwargs)
        path = figure_path(name if mode == "light" else f"{name}-dark")
        fig.savefig(path, dpi=150, facecolor=t["surface"], bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
        paths.append(str(path))
    return paths


def date_axis(ax) -> None:
    """Month-day ticks on the 1st and 15th ("Sep 1", "Sep 15")."""
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15]))
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda x, _: (lambda d: f"{d:%b} {d.day}")(mdates.num2date(x))))


# ----------------------------------------------------------------------------------------------
# Chart forms used in the README
# ----------------------------------------------------------------------------------------------
def draw_cumulative(fig, t, curves: dict[str, pd.DataFrame], total: int, subtitle: str = "",
                    actual_label: str = "Actual schedule", midpoint=None,
                    heading: str = "Critical violations found by date, Sept 2 - Oct 31, 2014"):
    """Cumulative critical violations found by date; ``curves`` maps label -> (date, found).

    ``midpoint`` (the last day of the window's first half) gets a hairline and the count each
    curve has reached by then, which is the first-half share the city reports.
    """
    ax = fig.add_subplot(111)
    style_axes(ax, t)
    k = 0
    for label, df in curves.items():
        if label == actual_label:
            color = t["muted"]
        else:
            color = t["series"][k]
            k += 1
        ax.step(df["date"], df["found"], where="post", color=color, lw=LW, label=label,
                solid_joinstyle="round")
    start = curves[actual_label]["date"].iloc[0]
    end = curves[actual_label]["date"].iloc[-1]
    ax.axhline(total, color=t["axis"], lw=0.8)
    label_x = (midpoint + pd.Timedelta(days=1.5)) if midpoint is not None else (
        start + (end - start) * 0.42)
    ax.text(label_x, total * 1.015, f"all {total} critical violations",
            color=t["muted"], fontsize=8.5, va="bottom", ha="left")
    if midpoint is not None:
        ax.axvline(midpoint, color=t["axis"], lw=0.8)
        lines = ["Found in the first half:"]
        for label, df in curves.items():
            n = df.loc[df["date"] <= midpoint, "found"].iloc[-1]
            lines.append(f"{label.split(' (')[0]}  {n:.0f} ({n / total:.0%})")
        ax.text(midpoint + pd.Timedelta(days=1.5), total * 0.04, "\n".join(lines),
                color=t["ink2"], fontsize=8.5, va="bottom", linespacing=1.5)
    ax.set_ylabel("Critical violations found (cumulative)")
    ax.set_ylim(0, total * 1.1)
    ax.set_xlim(start - pd.Timedelta(days=1), curves[actual_label]["date"].iloc[-1]
                + pd.Timedelta(days=1))
    date_axis(ax)
    title(ax, t, heading, subtitle)
    # Legend below the plot so it never sits on the curves or the reference line.
    legend(ax, t, loc="upper left", bbox_to_anchor=(0, -0.1),
           ncol=3 if len(curves) <= 3 else 2)


def draw_calibration(fig, t, curves: dict[str, pd.DataFrame], subtitle: str = ""):
    """Reliability diagram: mean predicted vs observed rate per bin."""
    ax = fig.add_subplot(111)
    style_axes(ax, t, grid="both")
    hi = max(max(df["pred"].max(), df["obs"].max(), df.get("hi", df["obs"]).max())
             for df in curves.values()) * 1.05
    ax.plot([0, hi], [0, hi], color=t["axis"], lw=0.8)
    ax.text(hi * 0.70, hi * 0.62, "perfect calibration", ha="left", va="top", color=t["muted"],
            fontsize=8.5)
    for k, (label, df) in enumerate(curves.items()):
        c = t["series"][k]
        ax.plot(df["pred"], df["obs"], color=c, lw=LW, label=label, marker="o", ms=5,
                mec=t["surface"], mew=1.5)
        if "lo" in df:
            ax.vlines(df["pred"], df["lo"], df["hi"], color=c, lw=0.8, alpha=0.7)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("Mean predicted probability (decile of test inspections)")
    ax.set_ylabel("Observed critical-violation rate")
    if len(curves) == 1:  # one series: the title names it, no legend box
        subtitle = f"{next(iter(curves))}. {subtitle}"
    title(ax, t, "Calibration on the test window", subtitle)
    if len(curves) > 1:
        legend(ax, t, loc="upper left")


def draw_intervals(fig, t, df: pd.DataFrame, label_col: str, est: str, lo: str, hi: str,
                   ref: float | None = None, ref_label: str = "", xlabel: str = "",
                   heading: str = "", subtitle: str = "", highlight: str | None = None,
                   fmt: str = ".2f"):
    """Dot-and-whisker estimates with 95% intervals (one row per item)."""
    ax = fig.add_subplot(111)
    style_axes(ax, t, grid="x")
    y = np.arange(len(df))[::-1]
    for yi, (_, row) in zip(y, df.iterrows(), strict=True):
        c = t["series"][1] if highlight and row[label_col] == highlight else t["series"][0]
        ax.hlines(yi, row[lo], row[hi], color=c, lw=LW, capstyle="round")
        ax.plot(row[est], yi, "o", color=c, ms=7, mec=t["surface"], mew=1.5, zorder=3)
        ax.text(row[hi], yi, f"  {format(row[est], fmt)}", va="center", fontsize=8.5,
                color=t["ink2"])
    if ref is not None:
        ax.axvline(ref, color=t["axis"], lw=0.9)
        if ref_label:
            ax.text(ref, -0.62, f" {ref_label}", color=t["muted"], fontsize=8.5,
                    va="bottom")
    ax.set_yticks(y, df[label_col])
    ax.set_ylim(-0.7, len(df) - 0.2)
    ax.set_xlabel(xlabel)
    title(ax, t, heading, subtitle)


def draw_lines(fig, t, df: pd.DataFrame, x: str, series: list[str], labels: list[str],
               ylabel: str, heading: str, subtitle: str = "", ref: float | None = None,
               shade: tuple | None = None):
    ax = fig.add_subplot(111)
    style_axes(ax, t)
    for k, (s, lab) in enumerate(zip(series, labels, strict=True)):
        ax.plot(df[x], df[s], color=t["series"][k], lw=LW, label=lab, solid_capstyle="round")
    if ref is not None:
        ax.axhline(ref, color=t["axis"], lw=0.9)
    if shade is not None:
        ax.axvspan(shade[0], shade[1], color=t["neutral"], alpha=0.25, lw=0)
        ax.text(shade[0], ax.get_ylim()[1], f" {shade[2]}", color=t["muted"], fontsize=8.5,
                va="top")
    ax.set_ylabel(ylabel)
    title(ax, t, heading, subtitle)
    if len(series) > 1:
        legend(ax, t, loc="best")


def seq_cmap(t) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("seq", t["seq"])


def div_cmap(t) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("div", [t["div_lo"], t["div_mid"], t["div_hi"]])
