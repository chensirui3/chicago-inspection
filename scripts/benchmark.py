"""Section 2 on its own: model search inside the training period, then the benchmark table.

    python scripts/benchmark.py

Runs section 1 first if its cached scores are missing, then rolling-origin validation, tuning,
the selection rule and one scoring of the test window, and prints the results table
(reports/tables/core2_results.md). Same code path as `scripts/reproduce.py --only core2`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inspections import config, core1, core2  # noqa: E402


def main() -> None:
    if not (config.CACHE_DIR / "core1_scores.parquet").exists():
        core1.run()
    out = core2.run()
    print(out["table_results"])
    b = out["best"]
    print(f"\nselected: {b['label']} -> {b['days']:.2f} days, "
          f"{b['diff']:+.2f} [{b['diff_lo']:+.2f}, {b['diff_hi']:+.2f}] vs the city model")


if __name__ == "__main__":
    main()
