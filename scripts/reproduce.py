"""Reproduce every number, table and chart in the README.

    python scripts/reproduce.py                    # all parts, in order
    python scripts/reproduce.py --only core1 core2 # selected parts
    python scripts/reproduce.py --update-readme    # also rewrite the README's generated sections

Outputs: reports/results.json (headline numbers), reports/tables/*.{csv,md},
reports/figures/*.png (light and -dark variants). Seeds are fixed in inspections.config.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inspections import config, load  # noqa: E402

PARTS = ["core1", "core2", "core3", "ext_b", "ext_a"]


def run_part(name: str) -> None:
    if name == "core1":
        from inspections import core1

        core1.run()
    elif name == "core2":
        from inspections import core2

        core2.run()
    elif name == "core3":
        from inspections import core3

        core3.run()
    elif name == "ext_b":
        from inspections import fairness

        fairness.run()
    elif name == "ext_a":
        from inspections import schedule

        schedule.run()
    else:
        raise SystemExit(f"unknown part {name}; choose from {PARTS}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=PARTS, help="run only these parts")
    ap.add_argument("--update-readme", action="store_true", help="refresh generated README text")
    args = ap.parse_args()
    if not load.data_available():
        raise SystemExit("data/ is empty: run `python scripts/get_data.py --portal` first")
    for d in (config.TABLES_DIR, config.FIGURES_DIR, config.CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for part in args.only or PARTS:
        t0 = time.time()
        print(f"== {part}", flush=True)
        run_part(part)
        print(f"   done in {time.time() - t0:.0f}s", flush=True)
    if args.update_readme:
        from inspections import readme

        print("README sections updated:", ", ".join(readme.update()))


if __name__ == "__main__":
    main()
