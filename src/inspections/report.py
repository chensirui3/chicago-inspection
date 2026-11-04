"""Small helpers that write tables, figures and headline numbers, and fill the README.

Every number in the README comes from ``reports/results.json`` or ``reports/tables/*.md``,
which ``scripts/reproduce.py`` writes; ``--update-readme`` replaces the text between
``<!-- BEGIN:name -->`` and ``<!-- END:name -->`` markers with the generated content.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

RESULTS = config.REPORTS_DIR / "results.json"


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return None if np.isnan(x) else round(float(x), 6)
    if isinstance(x, float):
        return None if np.isnan(x) else round(x, 6)
    if isinstance(x, pd.Timestamp):
        return x.strftime("%Y-%m-%d")
    return x


def load_results() -> dict:
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {}


def save_results(section: str, values: dict) -> None:
    """Merge ``values`` under ``section`` in reports/results.json."""
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    allres = load_results()
    allres[section] = _jsonable(values)
    RESULTS.write_text(json.dumps(allres, indent=2, sort_keys=False) + "\n")


def to_markdown(df: pd.DataFrame, floatfmt: dict | None = None) -> str:
    """GitHub-flavoured markdown table without extra dependencies."""
    floatfmt = floatfmt or {}
    cols = list(df.columns)

    def cell(col, v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if col in floatfmt and isinstance(v, int | float | np.number):
            return format(v, floatfmt[col])
        if isinstance(v, float | np.floating):
            return f"{v:.3f}"
        return str(v)

    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell(c, row[c]) for c in cols) + " |")
    return "\n".join(lines)


def save_table(df: pd.DataFrame, name: str, floatfmt: dict | None = None) -> str:
    """Write reports/tables/<name>.csv and .md; return the markdown."""
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.TABLES_DIR / f"{name}.csv", index=False)
    md = to_markdown(df, floatfmt)
    (config.TABLES_DIR / f"{name}.md").write_text(md + "\n", encoding="utf-8")
    return md


def figure_path(name: str) -> Path:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return config.FIGURES_DIR / f"{name}.png"


def ci(lo: float, hi: float, fmt: str = ".2f") -> str:
    return f"[{format(lo, fmt)}, {format(hi, fmt)}]"


def fill_readme(sections: dict[str, str], readme: Path | None = None) -> list[str]:
    """Replace generated sections of the README; returns the names that were updated."""
    readme = readme or config.ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    done = []
    for name, body in sections.items():
        pat = re.compile(rf"(<!-- BEGIN:{name} -->).*?(<!-- END:{name} -->)", re.S)
        if pat.search(text):
            text = pat.sub(lambda m, b=body: f"{m.group(1)}\n{b.strip()}\n{m.group(2)}", text)
            done.append(name)
    readme.write_text(text, encoding="utf-8")
    return done
